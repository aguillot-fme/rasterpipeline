# Raster Data Pipeline

A modular, cloud-ready data pipeline for processing raster data. This project leverages **Apache Airflow** for orchestration, **MinIO** for object storage, and **Apache Iceberg** for structured data management.

## Architecture

The pipeline consists of several key services orchestrated via Docker Compose:

- **Airflow**: Manages the workflow orchestration.
  - `airflow-scheduler`: Schedules and triggers tasks.
  - `airflow-webserver`: Runs the Airflow API server (`api-server`) and UI.
- **MinIO**: S3-compatible object storage for raw and processed raster files.
- **Postgres**: Metadata database for Airflow.

You can trigger the Airflow DAG (`raster_pipeline`) via the Airflow REST API or UI. A simple Python example is provided in `client_example.py`.

## Setup & Running

### Prerequisites
- Docker & Docker Compose

### Start the Pipeline
```bash
docker-compose up --build
```

## Usage

Use the Airflow UI (http://localhost:8080) or REST API to trigger runs of `raster_pipeline`, passing a `source_path` in the run configuration.

### Access Interfaces
- **MinIO Console**: http://localhost:9001 (User: `minioadmin`, Pass: `minioadmin`)

## Testing

Run the test suite using the provided script (Powershell):
```powershell
./run_all_tests.ps1
```
Or manually with pytest:
```bash
pytest tests/
```

## Project Structure
- `airflow/dags/`: Airflow DAGs.
- `processing/`: Core processing logic (ingest, transform, metrics).
- `storage/`: Storage abstraction layer.
- `tests/`: Unit and integration tests.

