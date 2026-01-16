# Raster Patch Similarity UI

This is a lightweight Leaflet app to load a small GeoTIFF, click a point, and trigger the
`raster_patch_similarity` DAG via the Airflow REST API (v2).

## Usage

1) Serve this folder (for example: `python -m http.server` from `docs/leaflet`).
2) Open the page in a browser.
3) Load a local GeoTIFF or provide a URL (CORS must allow it).
4) Click the map to select a point; X/Y populate automatically.
5) Fill Airflow URL/credentials and click **Trigger DAG**.
6) Set the MinIO HTTP base and click **Load Results** once the DAG finishes.

## Notes

- Coordinate conversion assumes the raster is in EPSG:4326, EPSG:3857, or a proj4 definition you provide.
- Leaflet + GeoTIFF dependencies are vendored locally under `docs/leaflet/vendor` (no CDN required).
- Airflow requires a valid token from `/auth/token` and uses `/api/v2` for DAG runs.
- The DAG will write results to Parquet + CSV under the configured embeddings directory.
- For results loading from MinIO, ensure the bucket/object is reachable over HTTP (public or presigned).
- CORS is enabled for the Airflow API in `docker-compose.yml` (`AIRFLOW__API__ENABLE_CORS=true`). Consider tightening or disabling in production.
