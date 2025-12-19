# Implementation Plan - Raster Embeddings DAG

## Goal
Implement a new Airflow DAG `raster_prep_and_embeddings` to ingest, tile, and generate embeddings for raster datasets using DINOv3. This enables semantic search and analytics on raster data.

## User Review Required
> [!IMPORTANT]
> **New Docker Image**: A new `analytics-embeddings` Docker image will be created. This image will be large (PyTorch + Transformers).
> **Model Weights**: DINOv3 model weights will need to be downloaded.
> **Database**: We will use DuckDB with `vss` extension for vector storage and analytics, keeping the main Postgres instance for Airflow metadata only.

## Proposed Changes

### Configuration
#### [MODIFY] [docker-compose.yml](file:///d:/rasterpipeline/docker-compose.yml)
- Add `analytics-embeddings` service.
- Revert any changes to `postgres` service (keep it as standard `postgres:13`).

### Docker Images
#### [NEW] [docker/Dockerfile.analytics](file:///d:/rasterpipeline/docker/Dockerfile.analytics)
- Base: `mambaorg/micromamba`
- Dependencies: `duckdb`, `pytorch`, etc.

### Data Storage Strategy
- **Metadata**: Simple JSON manifests in MinIO or keep `datasets` table in Postgres (optional).
- **Embeddings**: Stored as Parquet files in MinIO (`embeddings/{dataset_id}/*.parquet`).
- **Analytics/Search**: Use Dockerized DuckDB processing to load Parquet, build local indexes (vss), and run queries.


### Docker Images
#### [NEW] [docker/Dockerfile.analytics](file:///d:/rasterpipeline/docker/Dockerfile.analytics)
- Base: `mambaorg/micromamba:1.5-bullseye-slim` (or similar)
- Package Manager: `micromamba`
- Dependencies: Create an `environment.yml` with `pytorch`, `torchvision`, `transformers`, `rasterio`, `duckdb`, `psycopg2`, etc.
- This allows for faster and more reliable environment solving compared to pip/conda.

### Database
#### [SKIP] [scripts/init_db.sql](file:///d:/rasterpipeline/scripts/init_db.sql)
- Skip new Postgres tables for now. Use DuckDB for analytics.

### Airflow DAGs
#### [NEW] [dags/raster_embeddings_dag.py](file:///d:/rasterpipeline/dags/raster_embeddings_dag.py)
- **Tasks**:
    1.  `register_dataset`: Logic to check/create path in MinIO.
    2.  `ingest_raster`: DockerOperator (`raster-processing`).
    3.  `tile_raster`: DockerOperator (`raster-processing`).
    4.  `compute_embeddings`: DockerOperator (`analytics-embeddings`). Inference -> Parquet.
    5.  `qa_metrics`: DockerOperator (`analytics-embeddings`). DuckDB query on Parquet.
    6.  `mark_dataset_ready`: Optional.


### Scripts (Task Implementations)
#### [NEW] [scripts/tile_raster.py](file:///d:/rasterpipeline/scripts/tile_raster.py)
- Uses `rasterio` or `gdal` to tile the input raster.
- Generates `index.parquet` (DuckDB or Pandas) with geometries.

#### [NEW] [scripts/compute_embeddings.py](file:///d:/rasterpipeline/scripts/compute_embeddings.py)
- Loads DINOv3 model.
- Iterates over tiles.
- Computes embeddings.
- Saves to MinIO/Postgres.

#### [NEW] [scripts/qa_metrics.py](file:///d:/rasterpipeline/scripts/qa_metrics.py)
- Computes coverage and stats.

## Verification Plan

### Automated Tests
- Create a test DAG run using the `run_manual_test.ps1` or a new test script.
- Verify data in MinIO (files exist).
- Verify data in Postgres (rows exist).

### Manual Verification
- Check Airflow UI for DAG success.
- Inspect generated tiles and embeddings (sample check).
