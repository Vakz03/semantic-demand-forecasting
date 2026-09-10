# Demand Forecasting System & Supply Chain Console

A containerized, enterprise-ready B2B application designed for automated demand planning and inventory optimization. The system connects an intelligent backend (FastAPI + Ollama Semantic AI + LightGBM / MLForecast) with a modern, high-density Operations & Logistics web dashboard (React + Vite + Tailwind CSS).

---

## Architecture Overview

The system uses a decoupled architecture with two primary layers:

```
[ Raw CSV ] ──> [ Frontend SPA (React + Vite) ] ──> [ FastAPI Backend (Docker) ]
                       │                                      │
                       │ Pre-inspection (5 rows)             ├─> Semantic Profiling (Ollama Qwen2.5)
                       │ Inventory KPIs & Charts             ├─> Out-of-core ETL (Polars)
                       │ CSV Report Export                   └─> Parallel Multi-Series ML (LightGBM)
```

### 1. Semantic Data Profiling (LLM Layer)
- Users don't need to conform to fixed column schemas.
- A local **Large Language Model (Qwen 2.5 via Ollama)** inspects the dataset sample and dynamically maps each column into the pipeline ontology (`TARGET_METRIC`, `TIMESTAMP`, `ITEM_ID`, `LOCATION_ID`, `COVARIATE`, or `IGNORED`).
- Pydantic and **Instructor** enforce strict, deterministic JSON generation.

### 2. High-Performance ETL (Polars Streaming)
- Evaluates execution graphs using **Polars LazyFrames** with streaming computation to prevent out-of-memory errors on large datasets.
- Applies strict aggregation per `[unique_id, ds]` to prevent non-unique multi-indices in intraday data.
- Downcasts data types (`Float32`, `Categorical`) to reduce memory consumption by >50%.

### 3. Machine Learning Forecasting (LightGBM & MLForecast)
- Autonomously detects dataset frequency (Hourly, Daily, Weekly, Monthly) via dynamic frequency inference.
- Applies dynamic lags (e.g., 7, 14, 28 for daily series) to model seasonality.
- Employs **LightGBM** multi-threading across CPU cores.
- Truncates non-physical negative demand predictions at zero (`clip(lower=0)`).
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

---

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
  - `file`: CSV file containing historical sales records.
  - `h` *(optional, default: 7)*: Number of forecast horizons.
- **Response Format:**
  ```json
  {
    "pronostico": {
      "unique_id": ["001_000001", ...],
      "ds": ["2023-12-29", ...],
      "LGBMRegressor": [14.2, ...]
    },
    "catalogo": [
      {
        "unique_id": "001_000001",
        "DES_PROD": "ACEITE 1-2-3 LITRO"
      }
    ]
  }
  ```
