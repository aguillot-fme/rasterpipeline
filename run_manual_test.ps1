$json = '{"pipeline": [{"type": "readers.las", "filename": "__INPUT_FILE__"}, {"type": "filters.normal", "knn": 8}, {"type": "filters.dbscan", "min_points": 10, "eps": 2.0, "dimensions": "X,Y,Z"}, {"type": "writers.las", "filename": "__OUTPUT_FILE__", "extra_dims": "all", "minor_version": "4"}]}'
# Escape double quotes for the command line argument if passing to python
# Or better, Python script can interpret the single quoted string if passed correctly.
# But docker run needs careful handling.

# Let's simplify and pass the json structure carefully.
# We replace inner quotes \" with \\" to pass through shell? 
# Actually, powershell passing arguments to exe is tricky.

# Best approach: Use a temporary file for the pipeline if the script supported it.
# But the script supports string.

# Let's try to just run it with simple string if possible, or construct it carefully.
$pipeline = $json.Replace('"', '\"')

docker run --rm -v "d:\rasterpipeline:/opt/airflow" -w /opt/airflow pointcloud-worker:latest conda run -n pdal python scripts/pdal_processor.py --input /opt/airflow/data/raw/sample.las --output /opt/airflow/data/processed/sample_processed.las --pipeline "$pipeline"
