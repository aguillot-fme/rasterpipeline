import argparse
import os
import tempfile
from datetime import datetime
from urllib.parse import urlparse

from storage import StorageBackend, get_storage_backend

from pointcloudlib.enums import FileType
from pointcloudlib.features import GeometricFeatures
from pointcloudlib.io import PointCloudIO

from .helpers.storage_paths import find_pointcloud_by_id


def _filetype_for_path(path: str) -> FileType:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".las":
        return FileType.LAS
    if ext == ".csv":
        return FileType.CSV
    raise ValueError(f"Unsupported extension for pointcloud: {ext}")


def _download_to_temp(storage: StorageBackend, uri: str) -> tuple[str, dict | None]:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        local_path = uri if parsed.scheme else os.path.abspath(uri)
        return local_path, None

    suffix = os.path.splitext(parsed.path)[1] or ".las"
    data = storage.read_file(uri)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name, None


def enrich_pointcloud(
    storage: StorageBackend,
    file_id: str,
    raw_base_dir: str = "pointcloud/raw",
    enriched_base_dir: str = "pointcloud/enriched",
    k_neighbors: int = 20,
) -> str:
    raw_uri = find_pointcloud_by_id(storage, file_id=file_id, base_dir=raw_base_dir, pattern="*.las")
    raw_path, _ = _download_to_temp(storage, raw_uri)

    filetype = _filetype_for_path(raw_path)
    pc, header = PointCloudIO.read(raw_path, filetype=filetype, return_header=True)

    GeometricFeatures.compute_eigenvalues_and_normals(pc, k_neighbors=k_neighbors)
    GeometricFeatures.compute_eigenvalue_derived_features(pc)
    GeometricFeatures.compute_additional_features(pc)

    out_suffix = ".las" if filetype == FileType.LAS else ".csv"
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=out_suffix)
    out_tmp.close()

    out_name = os.path.basename(raw_path)
    stem, ext = os.path.splitext(out_name)
    enriched_filename = f"{stem}_enriched{ext}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_path = f"{enriched_base_dir}/{ts}_{file_id}/{enriched_filename}"

    PointCloudIO.write(out_tmp.name, pc, filetype=filetype, header_info=header)
    with open(out_tmp.name, "rb") as f:
        storage.write_file(dest_path, f.read())

    return dest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich a point cloud stored in the configured backend.")
    parser.add_argument("--file-id", required=True, help="Ingestion file id.")
    parser.add_argument("--raw-base-dir", default="pointcloud/raw", help="Base prefix for raw pointclouds.")
    parser.add_argument("--enriched-base-dir", default="pointcloud/enriched", help="Base prefix for enriched outputs.")
    parser.add_argument("--k-neighbors", type=int, default=20, help="k-NN size for feature computation.")
    args = parser.parse_args()

    storage = get_storage_backend()
    dest_path = enrich_pointcloud(
        storage,
        file_id=args.file_id,
        raw_base_dir=args.raw_base_dir,
        enriched_base_dir=args.enriched_base_dir,
        k_neighbors=args.k_neighbors,
    )
    print(dest_path)


if __name__ == "__main__":
    main()

