# Implementation Plan: DuckDB + LLM Automated Reporting

## Goal
Implement a new Airflow DAG (`duckdb_llm_reporting`) that replicates the workflow described in [SpatialWorld's FME Blog](https://spatialworld.fi/fi/fme-blog-duckdb-llm-reporting/). This workflow automates data analysis by using DuckDB for efficient querying and LLMs for reasoning, SQL generation, and result interpretation.

## User Review Required
> [!IMPORTANT]
> **LLM API Key**: This implementation requires access to an LLM provider (e.g., OpenAI). The user must provide an `OPENAI_API_KEY` (or compatible) via Airflow Connections or Environment Variables.
>
> **Docker Image Update**: The `analytics-embeddings` image will be updated to include `openai` and `jinja2`. Rebuilding the image will be required.

## Proposed Changes

### 1. Docker Environment (`docker/`)
#### [MODIFY] [environment.yml](file:///d:/rasterpipeline/docker/environment.yml)
- Add `openai` (for API access).
- Add `jinja2` (for report templating).
- Add `openpyxl` (if we need to support Excel ingestion as per the blog example).

### 2. Workflow Logic (`scripts/llm_reporting/`)
Create a new package for the reporting logic tasks.

#### [NEW] `scripts/llm_reporting/profile.py`
- **Function**: `analyze_dataset(s3_path)`
- **Logic**: Use DuckDB to load the file (Parquet/CSV), extract schema (`DESCRIBE`), calculate basic stats (`SUMMARIZE`), and grab a sample row. Return as JSON string.

#### [NEW] `scripts/llm_reporting/planner.py`
- **Function**: `generate_questions(profile_json, user_goal)`
- **Logic**: Call LLM (System: Planner) to generate analytical questions and abstract SQL plans based on the schema.

#### [NEW] `scripts/llm_reporting/sql_generator.py`
- **Function**: `generate_queries(questions_plan)`
- **Logic**: Call LLM (System: SQL Expert) to convert plans into valid DuckDB SQL queries (targeting the S3 Parquet file).

#### [NEW] `scripts/llm_reporting/executor.py`
- **Function**: `run_queries(queries_dict)`
- **Logic**: Execute generated SQL against MinIO data using DuckDB. Save results to MinIO (e.g., as JSON or CSV).

#### [NEW] `scripts/llm_reporting/reporter.py`
- **Function**: `compile_report(results_dict)`
- **Logic**: Call LLM (System: Analyst) to synthesize findings into Markdown. Convert to HTML using Jinja2/Markdown lib.

### 3. Airflow DAG (`airflow/dags/`)
#### [NEW] `airflow/dags/llm_reporting_dag.py`
- **Orchestration**:
    1.  **Ingest**: Upload generic file to MinIO (Sensor or manual trigger param).
    2.  **Profile**: `DockerOperator` running `profile.py`.
    3.  **Plan & Generate**: `DockerOperator` running `planner.py` & `sql_generator.py`.
    4.  **Execute**: `DockerOperator` running `executor.py`.
    5.  **Report**: `DockerOperator` running `reporter.py`.

## Verification Plan

### Automated Tests
- Unit tests for the Python scripts in `scripts/llm_reporting/`.
- Mock LLM responses to test the pipeline flow without spending API credits.

### Manual Verification
1.  **Security**: Configure `OPENAI_API_KEY` as an Airflow Variable (or Connection) in the UI. **Do not** bake this into `docker-compose.yml`.
2.  **Data**: Place a sample CSV (e.g., sales data or the Helsinki cycling data) in `data/`.
3.  **Execution**: Trigger the DAG via Airflow UI.
4.  **Result**: Verify the generated HTML report exists in MinIO.
