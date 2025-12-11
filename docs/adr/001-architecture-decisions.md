# ADR 001: Architecture Decisions

## Status
Accepted

## Context
We need a modular, cloud-ready data pipeline for raster processing.

## Decision
1. **Orchestrator**: Use Apache Airflow for code-driven workflows.
2. **Storage**: Implement a Storage Abstraction Layer to support both local filesystem and S3/MinIO.
3. **Lakehouse**: Use Apache Iceberg for structured output to support time-travel and schema evolution.
4. **API**: Use FastAPI for the REST interface.

## Consequences
- **Positive**: Decoupled storage allows easy cloud migration. Iceberg provides robust data management.
- **Negative**: Added complexity with Iceberg and Airflow setup compared to simple scripts.
