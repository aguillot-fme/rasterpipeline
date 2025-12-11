from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, LongType, StringType

catalog = SqlCatalog(
    name="iceberg",
    uri="jdbc:postgresql://postgres:5432/airflow",
    warehouse="s3://iceberg-warehouse",
    s3={
        "endpoint": "http://minio:9000",
        "region": "us-east-1",
        "access_key_id": "minioadmin",
        "secret_access_key": "minioadmin",
        "path_style_access": True,
    },
)

schema = Schema(
    NestedField.required(1, "id", LongType()),
    NestedField.optional(2, "note", StringType()),
)

table = catalog.create_table(
    identifier=("default", "sample_data"),
    schema=schema,
    or_replace=True,
)

table.append({"id": 1, "note": "hello"})
