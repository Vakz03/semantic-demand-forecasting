import os
import shutil
import logging
import threading
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from pipeline_semantico import procesar_csv_a_parquet
from entrenar_modelo import generar_pronostico
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MB
MAX_CONCURRENT_TRAININGS = 2
CONCURRENCY_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_TRAININGS)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
EXPECTED_API_KEY = os.getenv("API_KEY", "").strip()
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").lower() == "production"

limiter = Limiter(key_func=get_remote_address)

def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Demasiadas solicitudes. Por favor intente más tarde."}
    )

app = FastAPI(
    title="Demand Forecasting API",
    description="API RESTful para previsión de demanda.",
    version="1.1.0",
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if EXPECTED_API_KEY:
        if not api_key or api_key != EXPECTED_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Clave de API inválida o ausente en el encabezado 'X-API-Key'."
            )
    return api_key

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "demand-forecasting-api"}

@app.post("/predict-demand/")
@limiter.limit("10/minute")
def predict_demand(
    request: Request,
    file: UploadFile = File(...),
    h: int = Query(7, ge=1, le=90, description="Horizonte de pronóstico"),
    api_key: str = Security(verify_api_key)
):
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tipo de archivo no permitido. El archivo debe tener extensión .csv."
        )
    acquired = CONCURRENCY_SEMAPHORE.acquire(blocking=False)
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El servidor ha alcanzado su capacidad máxima de cómputo en paralelo. Intente de nuevo en un momento."
        )

    temp_csv_path = None
    temp_parquet_path = None
    total_bytes_read = 0

    try:
        with NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
            temp_csv_path = temp_csv.name
            while True:
                chunk = file.file.read(CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                total_bytes_read += len(chunk)
                if total_bytes_read > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="El archivo excede el tamaño máximo permitido."
                    )
                temp_csv.write(chunk)

        logging.info("Archivo recibido con exito (%d bytes). Iniciando pipeline semantico.", total_bytes_read)
        temp_parquet_path = procesar_csv_a_parquet(temp_csv_path)
        resultados = generar_pronostico(temp_parquet_path, h=h)

        return resultados

    except HTTPException:
        raise
    except Exception as e:
        logging.error("Excepcion no controlada durante el procesamiento: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No fue posible procesar el conjunto de datos. Verifique que el archivo sea un CSV estructurado con registros temporales y métricas de venta."
        )

    finally:
        CONCURRENCY_SEMAPHORE.release()
        if temp_csv_path and os.path.exists(temp_csv_path):
            try:
                os.remove(temp_csv_path)
            except OSError as err:
                logging.warning("No se pudo eliminar archivo temporal CSV: %s", err)

        if temp_parquet_path and os.path.exists(temp_parquet_path):
            try:
                os.remove(temp_parquet_path)
            except OSError as err:
                logging.warning("No se pudo eliminar archivo temporal Parquet: %s", err)
        import gc
        if 'resultados' in locals():
            del resultados
        gc.collect()
