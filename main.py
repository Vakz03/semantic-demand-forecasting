from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from tempfile import NamedTemporaryFile
from pipeline_semantico import procesar_csv_a_parquet
from entrenar_modelo import generar_pronostico

app = FastAPI(
    title="Demand Forecasting API",
    description="API RESTful para prevision de demanda con Ollama y MLForecast.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción cambiar esto por la URL de React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict-demand/")
def predict_demand(file: UploadFile = File(...), h: int = 7):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV.")

    temp_csv_path = None
    temp_parquet_path = None

    try:
        with NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
            shutil.copyfileobj(file.file, temp_csv)
            temp_csv_path = temp_csv.name
        temp_parquet_path = procesar_csv_a_parquet(temp_csv_path)
        resultados = generar_pronostico(temp_parquet_path, h=h)

        return resultados

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if temp_csv_path and os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        if temp_parquet_path and os.path.exists(temp_parquet_path):
            os.remove(temp_parquet_path)
        import gc
        if 'resultados' in locals():
            del resultados
        gc.collect()
