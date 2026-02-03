# Spatial Analytics & Data Pipeline

A modular, cloud-native platform designed for **high-performance spatial data processing**, **AI-driven analytics**, and **modern data warehousing workflows**. This project integrates **Apache Airflow**, **DuckDB**, **MinIO**, and **PyTorch** to create a versatile engine for geospatial intelligence.

## Overview

This repository demonstrates a production-grade architecture combining orchestration, object storage, and vectorized compute. It is designed to handle complex use cases ranging from traditional ETL to cutting-edge AI integrations.

### Key Use Cases

*   **AI-Enhanced Spatial Reporting (DuckDB + LLM)**:
    *   Leverage **DuckDB** to query massive Parquet datasets directly from **MinIO (S3)** without moving data.
    *   Enable **LLM-driven workflows** (e.g., text-to-SQL) by providing a high-speed SQL interface over raw file storage, similar to modern lakehouse patterns.
    *   Generate automated reports and insights by coupling SQL analysis with Generative AI agents.

*   **Computer Vision on Rasters**:
    *   Compute vector embeddings for satellite/aerial imagery using foundational models like **DINOv3**.
    *   Perform semantic search and similarity analysis on raster datasets.

*   **Scalable Data Engineering**:
    *   Orchestrate complex DAGs with **Apache Airflow**.
    *   Manage data lifecycle with **Apache Iceberg** and **MinIO**.
    *   Isolate environments using **Docker** and **Micromamba**.

## Architecture

The system is composed of specialized microservices orchestrated via Docker Compose:

*   **Orchestration**: `airflow-scheduler` & `airflow-webserver` manage workflows.
*   **Storage**: `minio` provides S3-compatible object storage for raw rasters, Parquet files, and embeddings.
*   **Compute & Analytics**:
    *   `analytics-embeddings`: A lightweight, **DuckDB** and **PyTorch** enabled environment for vectorized query processing and AI inference.
    *   `processing`: Dedicated workers for heavy raster transformations.
*   **Observability**: Integrated Prometheus and Grafana for monitoring pipeline health.

## Getting Started

### Prerequisites
*   Docker & Docker Compose

### Setup
1.  **Start the Platform**:
    ```bash
    docker-compose up --build -d
    ```
2.  **Access the Interfaces**:
    *   **Airflow UI**: http://localhost:8080 (Trigger DAGs like `raster_ingest` or `raster_embeddings`)
    *   **MinIO Console**: http://localhost:9001 (User: `minioadmin`, Pass: `minioadmin`)

### Run Without Docker (Local Dev)
If you prefer running components locally, install dependencies using the constraint files:
```bash
pip install -r requirements-airflow.txt -c https://raw.githubusercontent.com/apache/airflow/constraints-3.1.3/constraints-3.10.txt
pip install -r requirements.txt
```

## Testing

The project includes a comprehensive test suite covering both logic and integrations.

**Run all tests (Powershell):**
```powershell
./run_all_tests.ps1
```

**Run manually via Pytest:**
```bash
pytest tests/
```

## Project Structure
*   `airflow/dags/`: Workflow definitions (Python).
*   `processing/` & `docker/`: Application logic and container definitions.
*   `scripts/`: Standalone utilities for embedding generation and tiling.
*   `storage/`: Abstraction layer for S3/Local I/O.
*   `tests/`: Unit and integration validation.

## Roadmap
For upcoming features and architectural evolution, check the [Project Roadmap](roadmap.md).
