import polars as pl
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from utilsforecast.preprocessing import fill_gaps
import time
import pandas as pd
import gc
from datetime import timedelta

def generar_pronostico(parquet_path: str, h: int = 7, lead_time: int = 0) -> dict:
    print("Cargando datos preparados...")
    lf = pl.scan_parquet(parquet_path)
    df_polars = lf.collect()
    df = df_polars.to_pandas()
    
    del lf
    del df_polars
    gc.collect()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.dropna(subset=['ds'])
    max_date = df['ds'].max()
    if pd.notnull(max_date):
        fecha_corte = max_date - pd.Timedelta(days=365)
        df = df[df['ds'] >= fecha_corte].copy()
        print(f"Filtrando historico desde {fecha_corte.date()} hasta {max_date.date()}")
        
    if df.empty:
        raise ValueError("El dataset quedo vacio despues de procesar las fechas. Revisa el formato de fecha del CSV.")
    df = df.drop_duplicates(subset=['unique_id', 'ds'])
    df = df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    
    id_ejemplo = df['unique_id'].value_counts().idxmax()
    fechas_ejemplo = df[df['unique_id'] == id_ejemplo]['ds'].sort_values()
    freq_inferida = pd.infer_freq(fechas_ejemplo[-30:]) 
    
    if freq_inferida is None:
        diffs = fechas_ejemplo.diff().value_counts()
        if not diffs.empty:
            modo_diff = diffs.idxmax()
            if modo_diff >= pd.Timedelta(days=28): freq_inferida = 'MS'
            elif modo_diff >= pd.Timedelta(days=7): freq_inferida = 'W'
            elif modo_diff >= pd.Timedelta(days=1): freq_inferida = 'D'
            else: freq_inferida = 'H'
        else:
            freq_inferida = 'D'
            
    freq_inferida = str(freq_inferida)
    if freq_inferida.startswith('H'):
        lags_dinamicos = [24, 48, 168]
    elif freq_inferida.startswith('W'):
        lags_dinamicos = [1, 4, 12]
    elif freq_inferida.startswith('M'):
        lags_dinamicos = [1, 3, 12]
    else:
        freq_inferida = 'D'
        lags_dinamicos = [7, 14, 28]
        
    print(f"Frecuencia temporal detectada dinamicamente: {freq_inferida}")
    
    df = fill_gaps(df, id_col='unique_id', time_col='ds', freq=freq_inferida)
    
    df['y'] = df['y'].fillna(0)
    
    for col in df.columns:
        if col not in ['unique_id', 'ds', 'y']:
            df[col] = df[col].ffill().bfill()
            
    todas_covariables = [c for c in df.columns if c not in ['unique_id', 'ds', 'y']]
    catalogo = []
    text_cols = [c for c in todas_covariables if df[c].dtype.name in ['category', 'object', 'string']]
    if text_cols:
        df_cat = df[['unique_id'] + text_cols].groupby('unique_id').first().reset_index()
        catalogo = df_cat.to_dict(orient='records')
    
    covariables_estaticas = []
    covariables_dinamicas = []
    for c in todas_covariables:
        if df.groupby('unique_id')[c].nunique().max() == 1:
            covariables_estaticas.append(c)
        else:
            covariables_dinamicas.append(c)
            
    if covariables_dinamicas:
        df = df.drop(columns=covariables_dinamicas)
        
    for col in covariables_estaticas:
        if df[col].dtype == 'object' or df[col].dtype.name == 'string':
            df[col] = df[col].astype('category')
    if freq_inferida.startswith('H'):
        date_features = ['hour', 'dayofweek', 'day']
    else:
        date_features = ['dayofweek', 'day', 'month']
    min_points_req = max(lags_dinamicos) + 2
    series_counts = df['unique_id'].value_counts()
    series_maduras = series_counts[series_counts >= min_points_req].index
    series_cortas = series_counts[series_counts < min_points_req].index
    
    df_fit = df[df['unique_id'].isin(series_maduras)].copy() if len(series_cortas) > 0 and len(series_maduras) > 0 else df
    n_filas = len(df_fit)
    n_series = df_fit['unique_id'].nunique()
    
    if n_filas < 10000 or n_series < 50:
        n_est = 80
        lr = 0.05
        min_child = 5
        num_leaves = 15
    elif n_filas < 200000:
        n_est = 120
        lr = 0.08
        min_child = 15
        num_leaves = 31
    else:
        n_est = 150
        lr = 0.10
        min_child = 30
        num_leaves = 63

    model = MLForecast(
        models=[LGBMRegressor(
            n_estimators=n_est,
            learning_rate=lr,
            min_child_samples=min_child,
            num_leaves=num_leaves,
            random_state=42,
            verbosity=-1,
            n_jobs=6
        )],
        freq=freq_inferida,
        lags=lags_dinamicos,
        date_features=date_features
    )

    model.fit(
        df_fit,
        id_col='unique_id',
        time_col='ds',
        target_col='y',
        static_features=covariables_estaticas if covariables_estaticas else []
    )
    
    pronostico = model.predict(h)
    
    if 'LGBMRegressor' in pronostico.columns:
        pronostico['LGBMRegressor'] = pronostico['LGBMRegressor'].clip(lower=0)
    ceros_por_serie = df.groupby('unique_id')['y'].apply(lambda s: (s == 0).mean()).to_dict()
    if len(series_cortas) > 0 and len(series_maduras) > 0:
        df_cortas = df[df['unique_id'].isin(series_cortas)]
        fechas_futuras = sorted(pronostico['ds'].unique())
        filas_cortas = []
        for uid in series_cortas:
            sub = df_cortas[df_cortas['unique_id'] == uid]
            media_reciente = max(0.0, float(sub['y'].tail(7).mean()))
            for f in fechas_futuras:
                filas_cortas.append({
                    'unique_id': uid,
                    'ds': f,
                    'LGBMRegressor': round(media_reciente, 2)
                })
        if filas_cortas:
            df_pronostico_cortas = pd.DataFrame(filas_cortas)
            pronostico = pd.concat([pronostico, df_pronostico_cortas], ignore_index=True)
    df['dow'] = df['ds'].dt.dayofweek
    dow_means = df.groupby(['unique_id', 'dow'])['y'].transform('mean')
    residuos = df['y'] - dow_means
    cuantil_bajo_global = float(residuos.quantile(0.10))
    cuantil_alto_global = float(residuos.quantile(0.90))
    
    q10_por_serie = residuos.groupby(df['unique_id'].astype(str)).quantile(0.10).fillna(cuantil_bajo_global).to_dict()
    q90_por_serie = residuos.groupby(df['unique_id'].astype(str)).quantile(0.90).fillna(cuantil_alto_global).to_dict()
    std_por_serie = residuos.groupby(df['unique_id'].astype(str)).std().fillna(1.0).to_dict()
    std_por_fila = df['unique_id'].astype(str).map(std_por_serie).fillna(1.0)
    q10_futuro = pronostico['unique_id'].astype(str).map(q10_por_serie).fillna(cuantil_bajo_global)
    q90_futuro = pronostico['unique_id'].astype(str).map(q90_por_serie).fillna(cuantil_alto_global)
    
    pronostico['p10'] = (pronostico['LGBMRegressor'] + q10_futuro).clip(lower=0).round(2)
    pronostico['p90'] = (pronostico['LGBMRegressor'] + q90_futuro).clip(lower=0).round(2)
    pronostico['p10'] = pronostico[['p10', 'LGBMRegressor']].min(axis=1)
    pronostico['p90'] = pronostico[['p90', 'LGBMRegressor']].max(axis=1)
    es_intermitente = pronostico['unique_id'].map(ceros_por_serie).fillna(0.0) > 0.65
    pronostico.loc[es_intermitente, 'p10'] = 0.0
    metricas_inventario = {}
    if lead_time > 0:
        factor_servicio_z = 1.645
        promedios_pred = pronostico.groupby('unique_id')['LGBMRegressor'].mean().to_dict()
        for uid, media_d in promedios_pred.items():
            sigma_diaria = std_por_serie.get(str(uid), 1.0)
            ss = round(factor_servicio_z * sigma_diaria * (lead_time ** 0.5), 2)
            rop = round((media_d * lead_time) + ss, 2)
            metricas_inventario[str(uid)] = {
                "lead_time_dias": lead_time,
                "consumo_diario_estimado": round(float(media_d), 2),
                "stock_seguridad_sugerido": max(0.0, float(ss)),
                "punto_reorden_sugerido": max(0.0, float(rop))
            }
    anomalias_mask = (residuos.abs() > 3.5 * std_por_fila) & (residuos.abs() > 3.0)
    df_anomalias = df[anomalias_mask][['unique_id', 'ds', 'y']].copy()
    df_anomalias['y_esperado'] = dow_means[anomalias_mask].round(2)
    df_anomalias['desviacion'] = (df_anomalias['y'] - df_anomalias['y_esperado']).round(2)
    df_anomalias['ds'] = df_anomalias['ds'].astype(str)
    anomalias_lista = df_anomalias.sort_values(by='desviacion', ascending=False).head(150).to_dict(orient='records')

    del df
    del model
    gc.collect()
    
    pronostico['ds'] = pronostico['ds'].astype(str)
    
    salida = {
        "pronostico": pronostico.to_dict(orient='list'),
        "catalogo": catalogo,
        "anomalias": anomalias_lista
    }
    if metricas_inventario:
        salida["metricas_inventario"] = metricas_inventario
        
    return salida

