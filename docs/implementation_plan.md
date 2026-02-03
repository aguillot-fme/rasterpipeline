# Implementation Plan: DuckDB + LLM Automated Reporting

## Goal
Implement a new Airflow DAG (`duckdb_llm_reporting`) that replicates the workflow described in SpatialWorld's FME blog. The workflow automates data analysis by using DuckDB for efficient querying and LLMs for reasoning, SQL generation, and result interpretation.

## User Review Required
> [!IMPORTANT]
> **LLM API Key**: This implementation requires access to an LLM provider (e.g., OpenAI). The user must provide an `OPENAI_API_KEY` (or compatible) via Airflow Connections or Environment Variables.
>
> **Docker Image Update**: The `analytics-embeddings` image will be updated to include `openai` and `jinja2`. Rebuilding the image will be required.

## Design Principles
- **Contract-first**: each step produces validated JSON outputs with a strict schema.
- **Run isolation**: every run writes to a run-scoped path to avoid overwrites.
- **Safe SQL**: only `SELECT` queries, with enforced `LIMIT` and SQL validation.
- **Low coupling**: keep LLM prompts, SQL generation, and execution loosely coupled via artifacts.
- **Observable**: log timing, tokens, and query runtimes; emit a run manifest.

## Proposed Changes

### 1. Docker Environment (`docker/`)
#### [MODIFY] [environment.yml](file:///d:/rasterpipeline/docker/environment.yml)
- Add `openai` (for API access).
- Add `jinja2` (for report templating).
- Add `openpyxl` (if we need to support Excel ingestion).
- Add `sqlglot` (SQL validation and safety gating).
- Add `markdown-it-py` (or similar) for Markdown to HTML conversion.

### 2. Workflow Logic (`scripts/llm_reporting/`)
Create a new package for the reporting logic tasks.

#### [NEW] `scripts/llm_reporting/profile.py`
- **Function**: `analyze_dataset(s3_path)`
- **Logic**: Use DuckDB to load the file (Parquet/CSV), extract schema (`DESCRIBE` + exported schema SQL), calculate basic stats (`SUMMARIZE`), and grab sample rows. Return JSON.
- **Output**: `profile.json` with `schema_sql`, `summary`, `sample_rows`, and `dataset_signature` (hash of input).

#### [NEW] `scripts/llm_reporting/planner.py`
- **Function**: `generate_questions(profile_json, user_goal)`
- **Logic**: Call LLM (System: Planner) to generate analytical questions and abstract SQL plans based on the schema.
- **Output**: `plan.json` with a strict schema: `questions[]`, `sql_intent[]`, `assumptions[]`.

#### [NEW] `scripts/llm_reporting/sql_generator.py`
- **Function**: `generate_queries(questions_plan)`
- **Logic**: Call LLM (System: SQL Expert) to convert plans into valid DuckDB SQL queries.
- **Validation**: Parse with `sqlglot`, enforce `SELECT` only, disallow DDL/DML, and append `LIMIT 30` if missing.
- **Output**: `queries.json` (validated SQL + metadata).

#### [NEW] `scripts/llm_reporting/executor.py`
- **Function**: `run_queries(queries_dict)`
- **Logic**: Execute validated SQL against MinIO data using DuckDB. Wrap as `COPY (SELECT ...) TO` for consistent output. Save results to MinIO (JSON/CSV/Parquet).
- **Output**: `results/` folder + `results_index.json` with row counts, durations, and failures.

#### [NEW] `scripts/llm_reporting/reporter.py`
- **Function**: `compile_report(results_dict)`
- **Logic**: Call LLM (System: Analyst) to synthesize findings into Markdown. Convert to HTML using a versioned template.
- **Output**: `report.md`, `report.html`, and `report_meta.json` (model version, template version).

#### [NEW] `scripts/llm_reporting/contracts.py`
- **Function**: JSON schema definitions for `profile.json`, `plan.json`, `queries.json`, `results_index.json`, and `report_meta.json`.
- **Logic**: Centralized validation and error formatting.

### 3. Airflow DAG (`airflow/dags/`)
#### [NEW] `airflow/dags/llm_reporting_dag.py`
- **Orchestration**:
    1.  **Ingest**: Upload generic file to MinIO (Sensor or manual trigger param).
    2.  **Profile**: `DockerOperator` running `profile.py`.
    3.  **Plan & Generate**: `DockerOperator` running `planner.py` & `sql_generator.py`.
    4.  **Execute**: `DockerOperator` running `executor.py`.
    5.  **Report**: `DockerOperator` running `reporter.py`.
    6.  **Manifest**: `PythonOperator` writing `run_manifest.json` with pointers to all artifacts.

#### Run Isolation
- Use `run_id` to scope outputs: `s3://.../llm_reports/<dataset_id>/<run_id>/...`.
- Prevent collisions by default; allow overriding `output_path` only if explicitly provided.

#### Observability
- Log per-step latency, query runtime, and LLM token usage.
- Emit a summary row into `reports_index.parquet` for cross-run auditability.

## Verification Plan

### Automated Tests
- Unit tests for the Python scripts in `scripts/llm_reporting/`.
- Mock LLM responses to test the pipeline flow without spending API credits.
- Contract validation tests to ensure schema changes fail fast.
- SQL safety tests to confirm non-SELECT queries are rejected.

### Manual Verification
1.  **Security**: Configure `OPENAI_API_KEY` as an Airflow Variable (or Connection) in the UI. **Do not** bake this into `docker-compose.yml`.
2.  **Data**: Place a sample CSV (e.g., sales data or the Helsinki cycling data) in `data/`.
3.  **Execution**: Trigger the DAG via Airflow UI.
4.  **Result**: Verify the generated HTML report exists in MinIO.
5.  **Artifacts**: Verify `profile.json`, `queries.json`, `results_index.json`, and `run_manifest.json` exist for the run.
