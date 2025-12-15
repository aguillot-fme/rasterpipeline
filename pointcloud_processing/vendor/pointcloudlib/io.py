# pointcloudlib/io.py

import csv
import numpy as np
from typing import List, Optional, Tuple, Union
import logging

try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False

from .pointcloud import PointCloud
from .enums import FileType

logger = logging.getLogger(__name__)

class PointCloudIO:
    """
    A class containing static methods for reading and writing PointCloud objects to various file formats.
    """

    @staticmethod
    def read(
        filepath: str,
        filetype: FileType = FileType.CSV,
        return_header: bool = False,
        additional_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[PointCloud, Tuple[PointCloud, dict]]:
        if filetype == FileType.CSV:
            data, fields = PointCloudIO._read_csv(filepath, additional_fields=additional_fields, **kwargs)
            if data.shape[1] < 3:
                raise ValueError("CSV file must contain at least three columns for 'x', 'y', and 'z'.")
            point_cloud = PointCloud(data[:, :3])
            logger.info(f"PointCloud created from CSV with {data.shape[0]} points.")
            if additional_fields is None:
                additional_fields = fields
            if len(additional_fields) > 3:
                extra_data = data[:, 3:]
                extra_fields = additional_fields[3:]
                data_types = [str(data.dtype)] * len(extra_fields)
                PointCloudIO._assign_additional_attributes(point_cloud, extra_data, extra_fields, data_types)
            if return_header:
                header_info = {"scale": [1.0, 1.0, 1.0], "offset": [0.0, 0.0, 0.0]}
                return point_cloud, header_info
            else:
                return point_cloud
        elif filetype == FileType.LAS:
            return PointCloudIO._read_las(filepath, return_header=return_header, **kwargs)
        else:
            raise ValueError("Unsupported filetype. Use 'csv' or 'las'.")

    @staticmethod
    def _assign_additional_attributes(point_cloud: PointCloud, extra_data: np.ndarray, fields: List[str], data_fields: List[str]):
        field_mapping = {
            "nx": ("nx", "float32"),
            "ny": ("ny", "float32"),
            "nz": ("nz", "float32"),
            "eig1": ("eig1", "float32"),
            "eig2": ("eig2", "float32"),
            "eig3": ("eig3", "float32"),
            "height": ("height", "float32"),
            "intensity": ("intensity", "uint16"),
            "classification": ("classification", "uint8"),
            "cluster_id": ("cluster_id", "int32"),
            "red": ("red", "uint16"),
            "green": ("green", "uint16"),
            "blue": ("blue", "uint16"),
            "gps_time": ("gps_time", "float64"),
        }
        for idx, field in enumerate(fields):
            data = extra_data[:, idx]
            dtype = data_fields[idx]
            attr_name, enforced_dtype = field_mapping.get(field, (field, dtype))
            try:
                data_casted = data.astype(enforced_dtype)
            except TypeError:
                logger.warning(f"Failed to cast field '{field}' to '{enforced_dtype}'. Storing as original dtype.")
                data_casted = data.astype(dtype)
            point_cloud.set_attribute(attr_name, data_casted)
            logger.info(f"Additional attribute '{attr_name}' assigned with dtype '{data_casted.dtype}'.")

    @staticmethod
    def write(filepath: str, point_cloud: PointCloud, filetype: FileType = FileType.CSV, header_info: Optional[dict] = None, **kwargs):
        if filetype == FileType.CSV:
            PointCloudIO._write_csv(filepath, point_cloud, **kwargs)
        elif filetype == FileType.LAS:
            PointCloudIO._write_las(filepath, point_cloud, header_info=header_info, **kwargs)
        else:
            raise ValueError("Unsupported filetype. Use 'csv' or 'las'.")

    @staticmethod
    def _write_las(filepath: str, point_cloud: PointCloud, header_info: Optional[dict] = None):
        if not LASPY_AVAILABLE:
            raise ImportError("The 'laspy' library is not installed. Please install laspy to write LAS files.")
        header_info = header_info or point_cloud.header
        if not header_info:
            raise ValueError("No header information provided or available in the PointCloud object.")
        try:
            point_format_id = header_info.get("point_format_id", 3)
            version = header_info.get("version", "1.2")
            scales = np.asarray(header_info.get("scale", [0.001, 0.001, 0.001]), dtype=np.float64)
            offsets = np.asarray(header_info.get("offset", [0.0, 0.0, 0.0]), dtype=np.float64)
            las = laspy.create(point_format=point_format_id, file_version=version)
            las.header.scales = scales
            las.header.offsets = offsets
            las.x = point_cloud.points[:, 0]
            las.y = point_cloud.points[:, 1]
            las.z = point_cloud.points[:, 2]
            for extra_dim in header_info.get("extra_dimensions", []):
                name = extra_dim["name"]
                dtype = extra_dim["dtype"]
                if name in {"x", "y", "z"}:
                    continue
                data = point_cloud.get_attribute(name)
                if data is None:
                    logging.warning(f"Field '{name}' specified in header is missing from the PointCloud.")
                    continue
                if name not in las.point_format.dimension_names:
                    try:
                        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype, description=f"Custom field {name}"))
                        logger.info(f"Extra dimension '{name}' added to LAS format.")
                    except Exception as e:
                        logger.error(f"Failed to add extra dimension '{name}' to LAS format: {e}")
                        continue
                try:
                    las[name] = data
                    logger.info(f"Data for extra dimension '{name}' written to LAS file.")
                except Exception as e:
                    logger.error(f"Failed to write data for dimension '{name}' to LAS file: {e}")
            las.write(filepath)
            logger.info(f"Successfully wrote LAS file to '{filepath}'.")
        except Exception as e:
            raise OSError(f"Failed to write LAS file '{filepath}': {e}") from e

    @staticmethod
    def _read_csv(filepath: str, additional_fields: Optional[List[str]] = None, header: bool = True) -> Tuple[np.ndarray, List[str]]:
        try:
            with open(filepath, "r", newline="") as csvfile:
                reader = csv.reader(csvfile)
                if header:
                    header_row = next(reader)
                    if additional_fields is None:
                        fields = header_row
                    else:
                        fields = additional_fields
                    field_indices = []
                    for field in fields:
                        if field in header_row:
                            field_indices.append(header_row.index(field))
                        else:
                            raise ValueError(f"Field '{field}' not found in CSV header.")
                else:
                    if additional_fields is None:
                        raise ValueError("additional_fields must be provided if CSV has no header.")
                    fields = additional_fields
                    field_indices = list(range(len(fields)))
                data_list = []
                for row in reader:
                    if not row:
                        continue
                    try:
                        data_row = [float(row[idx]) for idx in field_indices]
                        data_list.append(data_row)
                    except ValueError as ve:
                        logger.warning(f"Value conversion error in row {row}: {ve}")
                        continue
                data_array = np.array(data_list)
                logger.info(f"CSV file '{filepath}' read with shape: {data_array.shape}")
                return data_array, fields
        except Exception as e:
            raise OSError(f"Failed to read CSV file '{filepath}': {e}") from e

    @staticmethod
    def _write_csv(filepath: str, point_cloud: PointCloud, header: bool = True, **kwargs):
        data_dict = {
            "x": point_cloud.points[:, 0],
            "y": point_cloud.points[:, 1],
            "z": point_cloud.points[:, 2],
        }
        for field, data in point_cloud._attributes.items():
            data_dict[field] = data
        field_names = ["x", "y", "z"] + list(point_cloud._attributes.keys())
        data_rows = np.column_stack([data_dict[field] for field in field_names])
        try:
            fmt = []
            for field in field_names:
                dtype = data_dict[field].dtype
                if np.issubdtype(dtype, np.floating):
                    fmt.append("%.6f")
                elif np.issubdtype(dtype, np.integer):
                    fmt.append("%d")
                else:
                    fmt.append("%s")
            with open(filepath, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if header:
                    writer.writerow(field_names)
                for row in data_rows:
                    writer.writerow(row)
            logger.info(f"Successfully wrote CSV file to '{filepath}'.")
        except Exception as e:
            raise OSError(f"Failed to write CSV file '{filepath}': {e}") from e

    @staticmethod
    def _read_las(filepath: str, return_header: bool = False) -> Union[PointCloud, Tuple[PointCloud, dict]]:
        if not LASPY_AVAILABLE:
            raise ImportError("The 'laspy' library is not installed. Please install laspy to read LAS files.")
        try:
            las = laspy.read(filepath)
            xyz = np.vstack((las.x, las.y, las.z)).T
            header_info = {
                "scale": las.header.scales,
                "offset": las.header.offsets,
                "point_format_id": las.header.point_format.id,
                "version": las.header.version,
                "extra_dimensions": [],
            }
            dimensions = {}
            for dim in las.point_format.dimensions:
                if dim is not None:
                    dim_name = dim.name
                    dim_dtype = dim.dtype.name if dim.dtype else "unknown"
                    if dim_name not in {"x", "y", "z"}:
                        try:
                            dim_data = np.array(getattr(las, dim_name))
                            dimensions[dim_name] = dim_data
                            header_info["extra_dimensions"].append({"name": dim_name, "dtype": dim_dtype})
                            logger.debug(f"Loaded dimension: {dim_name}, dtype: {dim_dtype}, shape: {dim_data.shape}")
                        except AttributeError as attr_err:
                            logger.warning(f"Attribute '{dim_name}' not found in LAS file '{filepath}': {attr_err}")
                else:
                    logger.warning(f"Encountered a 'None' dimension in LAS file '{filepath}'. Skipping.")
            point_cloud = PointCloud(points=xyz, header=header_info)
            logger.info(f"PointCloud created from LAS file '{filepath}' with {xyz.shape[0]} points.")
            for name, data in dimensions.items():
                point_cloud.set_attribute(name, data)
                logger.info(f"Attribute '{name}' assigned from LAS file with shape {data.shape}.")
            if return_header:
                return point_cloud, header_info
            return point_cloud
        except Exception as e:
            logger.exception(f"Failed to read LAS file '{filepath}': {e}")
            raise OSError(f"Failed to read LAS file '{filepath}': {e}") from e
