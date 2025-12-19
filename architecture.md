# Project Architecture

This document illustrates the high-level architecture of the `rasterpipeline` and `pointcloudlib` integration.

## System Overview

The system is designed as a **Docker-in-Docker** (or Docker-beside-Docker) pipeline orchestrated by **Apache Airflow**.

- **Orchestration**: Airflow (Scheduler & Webserver) runs in Docker containers.
- **Execution**: Heavy processing tasks are spawned as *ephemeral Docker containers* using `DockerOperator`.
- **Storage**: **MinIO** serves as the S3-compatible object storage for input/output data. **PostgreSQL** handles Airflow metadata.
- **Monitoring**: A full observability stack with **Prometheus**, **Grafana**, and **StatsD** is integrated.

## detailed Diagram

```mermaid
graph TD
    subgraph Host["Host Machine (Windows)"]
        Workspaces["User Workspaces<br/>(d:\rasterpipeline, d:\...\pointcloudlib)"]
        DockerSock["Docker Socket<br/>(//./pipe/docker_engine)"]
    end

    subgraph DockerNetwork["Docker Network: rasterpipeline_default"]
        
        subgraph Orchestration["Orchestration & Management"]
            UI["Airflow Webserver<br/>(Port 8080)"]
            Scheduler["Airflow Scheduler"]
            Proxy["Docker Socket Proxy<br/>(Secure Gateway)"]
        end

        subgraph StorageLayer["Storage Layer"]
            Postgres[("Postgres DB<br/>(Metadata)")]
            MinIO[("MinIO<br/>(S3 Compatible Storage)")]
        end

        subgraph Observability["Observability Stack"]
            StatsD["StatsD Exporter"]
            Prom["Prometheus"]
            Graf["Grafana<br/>(Port 3000)"]
        end

        subgraph Processing["Ephemeral Processing"]
            %% This node represents the container spawned by Airflow
            JobContainer["Pointcloud Processing Container<br/>(Image: pointcloud-processing)"]
        end
    end

    %% Relationships and Flows
    
    %% User Interation
    Host --> UI
    
    %% Storage Connections
    UI & Scheduler --> Postgres
    UI & Scheduler --> MinIO
    JobContainer -->|"Read/Write Data (S3)"| MinIO

    %% Orchestration Flow (Docker Operator)
    Scheduler -->|"1. Trigger Task via TCP"| Proxy
    Proxy -->|"2. API Call"| DockerSock
    DockerSock -.->|"3. Spawn Container"| JobContainer

    %% Code Injection (Bind Mounts in Dev)
    Workspaces -.->|"Bind Mount<br/>/opt/airflow"| Scheduler
    Workspaces -.->|"Bind Mount<br/>/opt/airflow"| UI
    Workspaces -.->|"Bind Mount<br/>/opt/airflow"| JobContainer

    %% Monitoring Flow
    Scheduler -->|"Emit Metrics"| StatsD
    JobContainer -->|"Emit Metrics"| StatsD
    StatsD -->|"Scrape"| Prom
    Prom -->|"Query"| Graf

    %% Styling
    classDef storage fill:#f9f,stroke:#333,stroke-width:2px;
    classDef orchest fill:#bbf,stroke:#333,stroke-width:2px;
    classDef process fill:#bfb,stroke:#333,stroke-width:2px;
    classDef monitor fill:#ffd,stroke:#333,stroke-width:2px;

    class Postgres,MinIO storage;
    class UI,Scheduler,Proxy orchest;
    class JobContainer process;
    class StatsD,Prom,Graf monitor;
```

## Component Details

| Component | Service Name | Description |
|-----------|--------------|-------------|
| **Airflow Scheduler** | `airflow-scheduler` | Triggers DAGs and tasks. Uses `DockerOperator` to offload work. |
| **Airflow Webserver** | `airflow-webserver` | UI for monitoring DAGs. |
| **Socket Proxy** | `docker-proxy` | Securely exposes the host Docker socket to Airflow containers. |
| **PostgreSQL** | `postgres` | Stores Airflow state, users, and connections. |
| **MinIO** | `minio` | Stores raw point cloud data (`.las`, `.laz`) and processed outputs (`.tif`, etc.). |
| **Processing Image** | `processing-image` / `pointcloud-processing-image` | Custom Docker images containing `pointcloudlib`, `pdal`, and other dependencies. |