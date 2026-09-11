# Demand Forecasting System & Supply Chain Console

A containerized, enterprise-ready B2B application designed for automated demand planning and inventory optimization. The system connects an intelligent backend (FastAPI + Ollama Semantic AI + LightGBM / MLForecast) with a modern, high-density Operations & Logistics web dashboard (React + Vite + Tailwind CSS).

---

## Architecture Overview

<<<<<<< HEAD
The system uses a decoupled architecture with two primary layers:

```
[ Raw CSV ] ──> [ Frontend SPA (React + Vite) ] ──> [ FastAPI Backend (Docker) ]
                       │                                      │
                       │ Pre-inspection (5 rows)             ├─> Semantic Profiling (Ollama Qwen2.5)
                       │ Inventory KPIs & Charts             ├─> Out-of-core ETL (Polars)
                       │ CSV Report Export                   └─> Parallel Multi-Series ML (LightGBM)
```

### 1. Semantic Data Profiling (LLM & Schema Cache)
Instead of forcing users to strictly format their data, this API is intelligent enough to analyze a raw CSV or Parquet file and logically deduce the meaning of each column. 
- The system reads a sample of the uploaded dataset and extracts its metadata profile.
- An in-memory cryptographic cache (SHA-256 schema signature) recognizes previously uploaded structures instantly, bypassing the LLM on recurrent pipelines.
- When new structures are detected, a local **Large Language Model (LLM)** via `Ollama` acts as a Data Engineer, inferring the correct ontology (e.g., identifying which column represents `TARGET_METRIC`, `TIMESTAMP`, `ITEM_ID`, or `COVARIATE`).
- We use **Pydantic** and **Instructor** to constrain the LLM's output to strict, deterministic JSON formatting.

### 2. High-Performance ETL (Polars Streaming)
- Evaluates execution graphs using **Polars LazyFrames** with streaming computation to prevent out-of-memory errors on large datasets.
- Native ingestion of both `.csv` and `.parquet` files.
- Applies strict aggregation per `[unique_id, ds]` to prevent non-unique multi-indices in intraday data.
- Downcasts data types (`Float32`, `Categorical`) to reduce memory consumption by >50%.

### 3. Machine Learning Forecasting (LightGBM & MLForecast)
- The pipeline dynamically infers the chronological frequency of the data (Hourly, Daily, Weekly, Monthly).
- Adaptive model hyperparameters automatically adjust based on dataset row volume and series cardinality to avoid overfitting and maximize training throughput.
- Based on the inferred frequency, it calculates appropriate lagged features (e.g., 7-day, 14-day, and 28-day lags for daily data) to capture deep seasonality.
- The core algorithm is **LightGBM** (Gradient Boosting), heavily optimized for execution speed and parallelization across CPU cores.
- Empirical conformal prediction intervals (`p10` and `p90`) provide well-calibrated distribution-free uncertainty bounds with guaranteed non-negative boundaries.
- When supply chain lead time is specified, the API automatically calculates suggested Safety Stock and Reorder Points (ROP) per SKU.
- Extracts a complete static catalog (product descriptions, categories) to link raw IDs with human-readable product names in the UI.

### 4. Operations & Logistics Console (Frontend SPA)
Built with React, Vite, Tailwind CSS v4, Lucide Icons, and Recharts:
- **Client-side Data Inspector:** Previews the first 5 rows, detects column headers, and validates file size before upload.
- **Execution Telemetry:** Calibrated progress monitor tracking inference and training time.
- **SKU Navigator:** Sidebar with real-time text search and rotation filters (*All, High, Intermittent*).
- **Supply Chain KPI Cards:**
  - *Projected Total Demand (units)* for the forecast horizon.
  - *Daily Estimated Consumption (units/day)*.
  - *Suggested Safety Stock (SS)* calculated using standard deviation buffer for ~95% service level.
  - *Reorder Point (ROP)* incorporating estimated lead times.
- **Technical Forecasting Chart:** Composed step/line visualization featuring shaded confidence/dispersion bands (+25%).
- **Structured Planning Table & CSV Export:** Monospace right-aligned data, logistical action badges (*Stock suficiente*, *Reabastecimiento preventivo*, *Monitoreo pasivo*), and one-click CSV report export.

## Prerequisites

- **Docker** and **Docker Compose**
- **Node.js** (v18+) & **npm**
- **Ollama**: Running locally with the `qwen2.5` model installed:
  ```bash
  ollama run qwen2.5
  ```

---

## Quick Start

### 1. Start the Backend API (Docker)

From the project root:

```bash
docker-compose up --build -d
```

The REST API will be available at `http://127.0.0.1:8000`. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### 2. Start the Frontend Dashboard

In a new terminal window:

```bash
cd frontend
npm install
npm run dev
```

Open your browser at **`http://localhost:5173`**.

---

## API Reference

### `POST /predict-demand/`
- **Method:** `POST` (multipart/form-data)
- **Parameters:**
  - `file`: CSV or Parquet file containing historical sales records.
  - `h` *(optional, default: 7)*: Number of forecast horizons.
  - `lead_time` *(optional, default: 0)*: Supplier lead time in days to calculate safety stock and reorder point.
- **Headers:** `X-API-Key` *(optional, if configured)*.
- **Response Format:**
  ```json
  {
    "pronostico": {
      "unique_id": ["001_000001", "..."],
      "ds": ["2023-12-29", "..."],
      "LGBMRegressor": [14.2, "..."],
      "p10": [10.5, "..."],
      "p90": [18.1, "..."]
    },
    "catalogo": [
      {
        "unique_id": "001_000001",
        "DES_PROD": "ACEITE 1-2-3 LITRO"
      }
    ],
    "anomalias": [
      {
        "unique_id": "001_000001",
        "ds": "2023-11-15",
        "venta_real": 95.0,
        "media_esperada": 12.4,
        "std": 4.1,
        "z_score": 20.14
      }
    ],
    "metricas_inventario": {
      "001_000001": {
        "lead_time_dias": 7,
        "consumo_diario_estimado": 14.2,
        "stock_seguridad_sugerido": 17.85,
        "punto_reorden_sugerido": 117.25
      }
    }
  }
  ```

---

## Enterprise Reliability & Security

The service is built following modern enterprise standards, featuring input validation, rate limiting, access control mechanisms, and containerized deployment designed to protect system resources and ensure consistent operation in production environments.

