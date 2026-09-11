# Demand Forecasting API

This repository contains a robust, containerized REST API that performs automated Demand Forecasting. It is designed to be easily deployable in B2B environments where businesses can upload their historical sales data and receive highly accurate, seasonal forecasts.

The API implements a fully automated end-to-end data pipeline powered by Semantic AI and advanced Machine Learning.

## Architecture & Algorithms

### 1. Semantic Data Profiling (LLM & Schema Cache)
Instead of forcing users to strictly format their data, this API is intelligent enough to analyze a raw CSV or Parquet file and logically deduce the meaning of each column. 
- The system reads a sample of the uploaded dataset and extracts its metadata profile.
- An in-memory cryptographic cache (SHA-256 schema signature) recognizes previously uploaded structures instantly, bypassing the LLM on recurrent pipelines.
- When new structures are detected, a local **Large Language Model (LLM)** via `Ollama` acts as a Data Engineer, inferring the correct ontology (e.g., identifying which column represents `TARGET_METRIC`, `TIMESTAMP`, `ITEM_ID`, or `COVARIATE`).
- We use **Pydantic** and **Instructor** to constrain the LLM's output to strict, deterministic JSON formatting.

### 2. High-Performance ETL (Polars)
Handling millions of rows requires extreme optimization.
- The pipeline uses **Polars (LazyFrames)** to perform out-of-core streaming computations. It processes datasets larger than RAM by evaluating execution graphs efficiently.
- Native ingestion of both `.csv` and `.parquet` files.
- Automatic **Downcasting** is applied (e.g., converting strings to Categoricals, and `Float64` to `Float32`) to halve the memory footprint.

### 3. Machine Learning Forecasting (LightGBM & MLForecast)
- The pipeline dynamically infers the chronological frequency of the data (Hourly, Daily, Weekly, or Monthly).
- Adaptive model hyperparameters automatically adjust based on dataset row volume and series cardinality to avoid overfitting and maximize training throughput.
- Based on the inferred frequency, it calculates appropriate lagged features (e.g., 7-day, 14-day, and 28-day lags for daily data) to capture deep seasonality.
- The core algorithm is **LightGBM** (Gradient Boosting), heavily optimized for execution speed and parallelization across CPU cores.
- Empirical conformal prediction intervals (`p10` and `p90`) provide well-calibrated distribution-free uncertainty bounds with guaranteed non-negative boundaries.
- When supply chain lead time is specified, the API automatically calculates suggested Safety Stock and Reorder Points (ROP) per SKU.

## Prerequisites

- **Docker** and **Docker Compose**
- **Ollama**: Ensure you have an instance of Ollama running on your host machine with the `qwen2.5` model installed (`ollama run qwen2.5`).

## Deployment (Docker)

To deploy the API in a background container:

```bash
docker-compose up --build -d
```

This will expose the API locally on port `8000`.

## Configuration (Environment Variables)

The service can be configured via environment variables in `docker-compose.yml` or a `.env` file:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `development` | Operating environment. When set to `production`, interactive documentation (`/docs`, `/redoc`, `/openapi.json`) is automatically disabled. |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Base URL pointing to the Ollama server hosting the local LLM. |
| `API_KEY` | *(empty)* | Optional secret key for endpoint protection. When specified, requests must supply this key via the `X-API-Key` header. |
| `ALLOWED_ORIGINS` | *(local addresses)* | Comma-separated list of allowed CORS origins for web integrations (e.g., `http://localhost:5173,http://localhost:3000`). |

## API Usage & Documentation

### Interactive Documentation

The API exposes interactive API documentation interfaces in development mode:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Alternative**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

In `production` mode (`ENVIRONMENT=production`), these documentation endpoints are completely disabled to prevent unintended schema exposure.

### Endpoints

#### `GET /health`
Verifies service availability and container health status.
- **Response**: `{"status": "healthy", "service": "demand-forecasting-api"}`

#### `POST /predict-demand/`
Executes end-to-end dataset profiling, ETL processing, and machine learning demand forecasting.

- **Payload**: A structured `.csv` or `.parquet` file (`multipart/form-data`).
- **Query Parameters**:
  - `h` (int, optional, default: `7`, range: `1` to `90`): Number of time steps to forecast into the future.
  - `lead_time` (int, optional, default: `0`, range: `0` to `180`): Supplier lead time in days for inventory safety stock and reorder point calculation.
- **Headers**:
  - `X-API-Key` *(optional, required only if `API_KEY` is configured on the server)*.
- **Response Format**:
  - `pronostico`: Forecasted demand per `unique_id` and timestamp `ds` with conformal uncertainty bands (`p10`, `p90`).
  - `catalogo`: Extracted static product and location metadata (categories, descriptions).
  - `anomalias`: Historical out-of-distribution sales records flagged by residual deviation analysis.
  - `metricas_inventario` *(optional, calculated when `lead_time > 0`)*: Suggested safety stock, daily consumption average, and reorder point (ROP) per series.

## Enterprise Reliability & Security

The service is built following modern enterprise standards, featuring:
- Non-root container execution (`appuser` UID 1001).
- Request rate limiting via SlowAPI.
- Streaming file upload size validation (50 MB limit) with protection against memory exhaustion.
- Bounded concurrency semaphore limiting simultaneous model training tasks.
- Sanitized input handling and generic client-side error responses to prevent stack trace disclosure.

## License & Legal Policies

- **License**: Released under the terms of the **MIT License**. See [LICENSE](LICENSE) for details.
- **Disclaimer**: Use of algorithmic forecasts is subject to the conditions detailed in [DISCLAIMER.md](DISCLAIMER.md).



