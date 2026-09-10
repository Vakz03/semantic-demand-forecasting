# Demand Forecasting API

This repository contains a robust, containerized REST API that performs automated Demand Forecasting. It is designed to be easily deployable in B2B environments where businesses can upload their historical sales data and receive highly accurate, seasonal forecasts.

The API implements a fully automated end-to-end data pipeline powered by Semantic AI and advanced Machine Learning.

## Architecture & Algorithms

### 1. Semantic Data Profiling (LLM)
Instead of forcing users to strictly format their data, this API is intelligent enough to analyze a raw CSV file and logically deduce the meaning of each column. 
- The system reads a sample of the uploaded dataset and extracts its metadata profile.
- A local **Large Language Model (LLM)** via `Ollama` acts as a Data Engineer, inferring the correct ontology (e.g., identifying which column represents the `TARGET_METRIC`, `TIMESTAMP`, `ITEM_ID`, or `COVARIATE`).
- We use **Pydantic** and **Instructor** to constrain the LLM's output to strict, deterministic JSON formatting.

### 2. High-Performance ETL (Polars)
Handling millions of rows requires extreme optimization.
- The pipeline uses **Polars (LazyFrames)** to perform out-of-core streaming computations. It processes datasets larger than RAM by evaluating execution graphs efficiently.
- Automatic **Downcasting** is applied (e.g., converting strings to Categoricals, and `Float64` to `Float32`) to halve the memory footprint.

### 3. Machine Learning Forecasting (LightGBM & MLForecast)
- The pipeline dynamically infers the chronological frequency of the data (Hourly, Daily, Weekly, or Monthly).
- Based on the inferred frequency, it calculates appropriate lagged features (e.g., 7-day, 14-day, and 28-day lags for daily data) to capture deep seasonality.
- The core algorithm is **LightGBM** (Gradient Boosting), heavily optimized for execution speed and parallelization.
- We utilize `utilsforecast` to ensure contiguous time series (filling gaps autonomously), separating static and dynamic covariates before projecting values into the future.

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
- **Payload**: A raw `.csv` file.
- **Parameters**: `h` (int) - The number of steps/days to forecast into the future.
- **Response**: A JSON dictionary containing the forecasted values for every identified item/location optimally formatted in columnar arrays.
