## Iceberg JDBC Catalog Setup

This branch assumes you want Apache Iceberg metadata stored directly in the existing Postgres container and table data in MinIO. No additional container is required beyond Postgres and MinIO from `docker-compose.yml`.

### 1. Prepare the metadata schema

Run once inside the Postgres container:

```powershell
docker-compose exec postgres psql -U airflow -d airflow -c "CREATE SCHEMA IF NOT EXISTS iceberg;"
```

Adjust schema/database names if you prefer a dedicated DB.

### 2. Create the warehouse bucket

Open the MinIO console (`http://localhost:9001`, user/pass `minioadmin/minioadmin`) and create a bucket named `iceberg-warehouse`. Iceberg table files will be stored there via S3 API calls.

### 3. Use the catalog from `pyiceberg`

Install dependencies (`pip install -r requirements.txt`). Then point `pyiceberg` at the JDBC catalog:

```python
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, LongType

catalog = load_catalog(
    "jdbc",
    uri="jdbc:postgresql://postgres:5432/airflow",
    warehouse="s3://iceberg-warehouse",
    jdbc_user="airflow",
    jdbc_password="airflow",
    catalog_name="iceberg",
    s3={
        "endpoint": "http://minio:9000",
        "region": "us-east-1",
        "access_key_id": "minioadmin",
        "secret_access_key": "minioadmin",
    },
)

table = catalog.create_table(
    identifier=("default", "demo_table"),
    schema=Schema(NestedField.required(1, "id", LongType())),
)
table.append({"id": 1})
```

This stores metadata inside Postgres (schema `iceberg`) and data files in the MinIO bucket.
