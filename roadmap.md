# Project Roadmap

## 1. Core Architecture (Completed)
- [x] **Containerization**: Fully Dockerized environment using Docker Compose.
- [x] **Orchestration**: Transitioned to Airflow-centric architecture (Pattern A).
  - [x] `raster_ingest` DAG implementation.
  - [x] `DockerOperator` for isolated task execution.
  - [x] Branching logic for validation handling.
- [x] **Storage**: MinIO integration for S3-compatible object storage.
- [x] **API**: Direct usage of Airflow REST API (removal of custom FastAPI wrapper).
  - [x] `client_example.py` for API interaction demonstration.
- [x] **Security**:
  - [x] Secure Docker socket access using `tecnativa/docker-socket-proxy`.
- [x] **Testing**:
  - [x] Comprehensive test suite (`pytest`).
  - [x] Automated test runner (`run_all_tests.ps1`).
- [x] **Cleanup**:
  - [x] Migration of legacy code to `legacy/`.

## 2. Data Management & Integrations (In Progress)
- [ ] **Apache Iceberg Integration**:
  - [x] Basic `IcebergWriter` and Schema.
  - [x] Initial write task in `raster_ingest` DAG.
  - [ ] Full Catalog implementation (REST/Hive).
  - [ ] Schema evolution support.
- [ ] **Observability**:
  - [x] Basic metrics calculation task.
  - [ ] Integration with Prometheus/Grafana for visualization.
  - [ ] Airflow Monitoring/Alerting.

## 3. Production Readiness (Backlog)
- [ ] **CI/CD Pipeline**:
  - [ ] GitHub Actions / GitLab CI workflows.
  - [ ] Automated image building and pushing.
- [ ] **Deployment**:
  - [ ] Kubernetes (Helm Charts).
  - [ ] Cloud Deployment (AWS/GCP/Azure).
- [ ] **Advanced Security**:
  - [ ] Secret Management (Vault/AWS Secrets Manager).
  - [ ] Non-root user enforcement in all containers.
  - [ ] Network policy hardening.
- [ ] **Performance**:
  - [ ] Parallel processing optimization.
  - [ ] Resource limit fine-tuning.
