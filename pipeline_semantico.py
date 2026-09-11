import polars as pl
from pydantic import BaseModel, Field
from typing import List
import instructor
from openai import OpenAI
import os
import gc

pl.Config.set_tbl_rows(8)
os.environ["POLARS_MAX_THREADS"] = "6"

class ColumnaMapeada(BaseModel):
    nombre_original: str
    tipo_inferido: str = Field(description="""
        DEBE ser uno de los siguientes:
        - TIMESTAMP: Fechas o tiempos que indican cuando ocurrio la transaccion.
        - ITEM_ID: Identificador unico del producto o SKU.
        - LOCATION_ID: Identificador de la tienda, sucursal, almacen o ubicacion fisica.
        - TARGET_METRIC: La cantidad numerica a predecir (ventas, volumen, demanda). NUNCA puede ser texto.
        - COVARIATE: Variables estaticas o dinamicas que ayudan a predecir (promociones, dias festivos, categorias, descripciones, precios).
        - IGNORED: Columnas que no aportan valor predictivo (IDs secuenciales, hashes vacios).
    """)
    razonamiento: str
    confianza: float

class ResultadoMapeo(BaseModel):
    columnas: List[ColumnaMapeada]

def sanitizar_texto(texto: str, max_len: int = 50) -> str:
    if not isinstance(texto, str):
        texto = str(texto)
    limpio = "".join(ch for ch in texto if ch.isprintable() and ch not in "\r\n\t`'\"")
    return limpio[:max_len]

def extraer_perfil(df: pl.DataFrame) -> str:
    perfil = []
    for col in df.columns:
        col_sanitizada = sanitizar_texto(col, max_len=60)
        tipo = str(df.schema[col])
        valores_raw = df[col].drop_nulls().head(5).to_list()
        ejemplos = [sanitizar_texto(v, max_len=35) for v in valores_raw]
        perfil.append(f"Columna: '{col_sanitizada}' | Tipo nativo: {tipo} | Ejemplos: {ejemplos}")
    return "\n".join(perfil)

def transformar_datos(df_raw: pl.LazyFrame, mapeo: ResultadoMapeo) -> pl.LazyFrame:
    cols_ignored = [c.nombre_original for c in mapeo.columnas if c.tipo_inferido == "IGNORED"]
    cols_timestamp = [c.nombre_original for c in mapeo.columnas if c.tipo_inferido == "TIMESTAMP"]
    cols_target = [c.nombre_original for c in mapeo.columnas if c.tipo_inferido == "TARGET_METRIC"]
    cols_item = [c.nombre_original for c in mapeo.columnas if c.tipo_inferido == "ITEM_ID"]
    cols_location = [c.nombre_original for c in mapeo.columnas if c.tipo_inferido == "LOCATION_ID"]
    cols_covariate = [c.nombre_original for c in mapeo.columnas if c.tipo_inferido == "COVARIATE"]

    if not cols_timestamp or not cols_target:
        raise ValueError("El esquema debe contener al menos un TIMESTAMP y un TARGET_METRIC")

    df_clean = df_raw.drop(cols_ignored)

    rename_map = {}
    if cols_timestamp:
        rename_map[cols_timestamp[0]] = "ds"
    if cols_target:
        rename_map[cols_target[0]] = "y"
    df_clean = df_clean.rename(rename_map)

    covariates_cat = [c for c in cols_covariate if df_clean.schema[c] == pl.String]
    covariates_num = [c for c in cols_covariate if df_clean.schema[c] != pl.String]

    ts_keys = []
    if cols_location: ts_keys.append(cols_location[0])
    if cols_item: ts_keys.append(cols_item[0])

    if len(ts_keys) > 1:
        df_clean = df_clean.with_columns(
            pl.concat_str([pl.col(k).cast(pl.String) for k in ts_keys], separator="_").alias("unique_id")
        )
        df_clean = df_clean.drop(cols_item + cols_location)
    elif len(ts_keys) == 1:
        if ts_keys[0] in cols_item or ts_keys[0] in cols_location:
            df_clean = df_clean.rename({ts_keys[0]: "unique_id"})
        else:
            df_clean = df_clean.with_columns(pl.col(ts_keys[0]).cast(pl.String).alias("unique_id"))
    else:
        df_clean = df_clean.with_columns(pl.lit("1").alias("unique_id"))

    df_clean = df_clean.with_columns(pl.col("unique_id").cast(pl.String))

    df_clean = df_clean.with_columns([
        pl.col("y").cast(pl.Float64)
    ])
    group_cols = ["unique_id", "ds"]
    
    agg_exprs = [pl.col("y").sum()]
    for c in covariates_num:
        if c in df_clean.columns:
            agg_exprs.append(pl.col(c).max())
    for c in covariates_cat:
        if c in df_clean.columns:
            agg_exprs.append(pl.col(c).first())

    df_agg = df_clean.group_by(group_cols).agg(agg_exprs)

    downcast_exprs = [
        pl.col("y").cast(pl.Float32),
        pl.col("unique_id").cast(pl.Categorical)
    ]
    for c in covariates_cat:
        if c in df_clean.columns:
            downcast_exprs.append(pl.col(c).cast(pl.Categorical))
            
    for c in covariates_num:
        if c in df_clean.columns:
            downcast_exprs.append(pl.col(c).cast(pl.Float32))
            
    df_agg = df_agg.with_columns(downcast_exprs)

    df_final = df_agg.sort(["unique_id", "ds"])
    return df_final

def procesar_csv_a_parquet(file_path: str) -> str:
    print("Cargando datos con Polars (Lazy API)...")
    try:
        lf = pl.scan_csv(file_path, infer_schema_length=0, ignore_errors=True)
    except FileNotFoundError:
        print(f"Error: No se encontro el archivo '{file_path}'.")
        raise

    df_muestra = lf.head(10000).collect()
    perfil_texto = extraer_perfil(df_muestra)
    print("\n--- Perfil Extraido ---")
    print(perfil_texto)
    print("-----------------------\n")
    
    del df_muestra
    gc.collect()

    modelo_local = "qwen2.5" 
    
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    
    cliente = instructor.from_openai(
        OpenAI(
            base_url=ollama_base_url,
            api_key="ollama-local"
        ),
        mode=instructor.Mode.JSON
    )

    prompt = f"""
    Actua como un ingeniero de datos analizando una nueva base de datos de retail/negocios.
    Tu objetivo es clasificar cada columna para preparar los datos para un algoritmo de prevision de demanda.
    
    REGLAS CRITICAS:
    1. TARGET_METRIC debe ser siempre una columna numerica (Int o Float) que represente volumen de VENTAS o DEMANDA. 
    2. NUNCA clasifiques precios, descuentos, costos o margenes como TARGET_METRIC. Si la columna es monetaria o de precio, clasificala como COVARIATE.
    3. Si la columna contiene texto (String) como nombres de categorias, clasificala como COVARIATE.
    4. DEBES asignar exactamente UNA columna como TIMESTAMP y exactamente UNA columna como TARGET_METRIC.
    
    Analiza la siguiente metadata de la tabla:
    {perfil_texto}
    
    Clasifica estrictamente cada columna segun la ontologia definida.
    """

    print(f"Consultando al modelo {modelo_local}...")
    
    respuesta = cliente.chat.completions.create(
        model=modelo_local,
        response_model=ResultadoMapeo,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    print("\n--- Resultados del Schema Matching ---")
    for col in respuesta.columnas:
        print(f"Columna: {col.nombre_original}")
        print(f"Clasificacion: {col.tipo_inferido} (Certeza: {col.confianza})")
        print(f"Razon: {col.razonamiento}")
        print("-" * 40)

    print("\n--- Transformacion a formato Estandar ---")
    lf_transformado = transformar_datos(lf, respuesta)
    
    print("Ejecutando pipeline y guardando a disco (esto puede tomar un momento)...")
    out_path = file_path.replace(".csv", ".parquet")
    lf_transformado.collect(streaming=True).write_parquet(out_path)
    print(f"Datos transformados y guardados en '{out_path}' exitosamente.")
    
    return out_path
