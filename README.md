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

## API Usage

You can access the interactive Swagger UI at:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

### `POST /predict-demand/`
- **Payload**: A structured `.csv` or `.parquet` file.
- **Parameters**:
  - `h` (int, optional, default: 7): Number of horizon steps to forecast into the future.
  - `lead_time` (int, optional, default: 0): Supplier lead time in days to calculate safety stock and reorder point.
- **Headers**: `X-API-Key` *(optional, if authentication is configured)*.
- **Response**: A JSON dictionary containing:
  - `pronostico`: Forecasted demand values per `unique_id` and `ds` with uncertainty bounds (`p10`, `p90`).
  - `catalogo`: Extracted static product metadata (descriptions, categories).
  - `anomalias`: Historical out-of-distribution sales events flagged using weekday residual deviation.
  - `metricas_inventario` *(optional, when lead_time > 0)*: Suggested safety stock and reorder point per series.

## Enterprise Reliability & Security

The service is built following modern enterprise standards, featuring input validation, rate limiting, access control mechanisms, and containerized deployment designed to protect system resources and ensure consistent operation in production environments.



