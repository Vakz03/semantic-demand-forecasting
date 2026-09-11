import polars as pl
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from utilsforecast.preprocessing import fill_gaps
import time
import pandas as pd
import gc
from datetime import timedelta

def generar_pronostico(parquet_path: str, h: int = 7) -> dict:
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
        
    model = MLForecast(
        models=[LGBMRegressor(random_state=42, verbosity=-1, n_jobs=6)],
        freq=freq_inferida,
        lags=lags_dinamicos,
        date_features=date_features
    )
    min_points_req = max(lags_dinamicos) + 2
    series_counts = df['unique_id'].value_counts()
    series_maduras = series_counts[series_counts >= min_points_req].index
    series_cortas = series_counts[series_counts < min_points_req].index
    
    df_fit = df[df['unique_id'].isin(series_maduras)].copy() if len(series_cortas) > 0 and len(series_maduras) > 0 else df
    
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
    std_por_serie = residuos.groupby(df['unique_id'].astype(str)).std().fillna(1.0).to_dict()
    std_por_fila = df['unique_id'].astype(str).map(std_por_serie).fillna(1.0)
    dispersion_futura = pronostico['unique_id'].astype(str).map(std_por_serie).fillna(1.0)
    pronostico['p10'] = (pronostico['LGBMRegressor'] - 1.28 * dispersion_futura).clip(lower=0).round(2)
    pronostico['p90'] = (pronostico['LGBMRegressor'] + 1.28 * dispersion_futura).clip(lower=0).round(2)
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
    
    return {
        "pronostico": pronostico.to_dict(orient='list'),
        "catalogo": catalogo,
        "anomalias": anomalias_lista
    }
