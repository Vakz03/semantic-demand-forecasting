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
    max_date_str = lf.select(pl.col("ds").max()).collect().item()
    if max_date_str:
        max_date = pd.to_datetime(max_date_str)
        fecha_corte = max_date - pd.Timedelta(days=365)
        fecha_corte_str = fecha_corte.strftime('%Y-%m-%d') 
        lf = lf.filter(pl.col("ds") >= fecha_corte_str)
        print(f"Filtrando historico desde {fecha_corte_str} hasta {max_date_str}")
    df_polars = lf.collect()
    df = df_polars.to_pandas()
    del lf
    del df_polars
    gc.collect()
    df['ds'] = pd.to_datetime(df['ds'])
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
    
    model = MLForecast(
        models=[LGBMRegressor(random_state=42, verbosity=-1, n_jobs=6)],
        freq=freq_inferida,
        lags=lags_dinamicos,
    )
    
    model.fit(
        df,
        id_col='unique_id',
        time_col='ds',
        target_col='y',
        static_features=covariables_estaticas if covariables_estaticas else []
    )
    
    pronostico = model.predict(h)
    
    if 'LGBMRegressor' in pronostico.columns:
        pronostico['LGBMRegressor'] = pronostico['LGBMRegressor'].clip(lower=0)
    
    del df
    del model
    gc.collect()
    
    pronostico['ds'] = pronostico['ds'].astype(str)
    
    return {"pronostico": pronostico.to_dict(orient='list')}
