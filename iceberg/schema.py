"""
Lightweight schema placeholders for Iceberg tables without pulling in an Iceberg
client library. These can be expanded later when a catalog is available.
"""

# Schema for pc_rasters table (per-image metadata and metrics)
PC_RASTERS_SCHEMA = {
    "file_id": "string",
    "filename": "string",
    "ingested_at": "timestamp",
    "width": "int",
    "height": "int",
    "crs": "string",
    "min_val": "double?",
    "max_val": "double?",
    "mean_val": "double?",
    "std_val": "double?",
    "s3_path": "string",
}

# Schema for pc_metadata table (ingestion info)
PC_METADATA_SCHEMA = {
    "file_id": "string",
    "source_path": "string",
    "ingest_status": "string",
    "error_message": "string?",
}
