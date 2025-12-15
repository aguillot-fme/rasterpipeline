"""
Profiling script for the pointcloudlib toolset.
This script runs several processing steps and writes profiling statistics both
to a SQLite database and to text files.
"""

import os
import sys
import copy
import cProfile
import pstats
import sqlite3
import numpy as np
import logging
import argparse
from collections import OrderedDict
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Import necessary classes from pointcloudlib
from pointcloudlib import (
    PointCloudIO,
    PointCloud,
    PointCloudPreprocessing,
    Classification,
    Clustering,
    PointCloudRegistration,
    GeometryUtils,
    ClassificationRule,
    FileType,
    GeometricFeatures,
)

def create_run_directory(base_dir="profiling_runs"):
    """
    Creates a directory for the current profiling run.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        logger.info(f"Created base directory: {base_dir}")
    existing_runs = [
        int(folder.split("#")[-1])
        for folder in os.listdir(base_dir)
        if folder.startswith("run#") and folder.split("#")[-1].isdigit()
    ]
    next_run_number = max(existing_runs, default=0) + 1
    run_dir = os.path.join(base_dir, f"run#{next_run_number}")
    os.makedirs(run_dir)
    logger.info(f"Created run directory: {run_dir}")
    return run_dir

def resolve_profiling_base_dir(input_file: str, output_dir: Optional[str] = None) -> str:
    """Resolve where profiling outputs should be written.

    Precedence: CLI --output_dir > $POINTCLOUDLIB_PROFILING_DIR > <input_file_dir>/profiling_runs
    """
    if output_dir:
        return output_dir
    env_dir = os.getenv('POINTCLOUDLIB_PROFILING_DIR')
    if env_dir:
        return env_dir
    input_dir = os.path.dirname(os.path.abspath(input_file))
    return os.path.join(input_dir, 'profiling_runs')


def configure_run_logging(run_dir: str) -> None:
    """Add/update a file handler so logs go into the current run directory."""
    root = logging.getLogger()
    desired_path = os.path.join(run_dir, 'profiling_log.txt')
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            if os.path.abspath(getattr(handler, 'baseFilename', '')) == os.path.abspath(desired_path):
                return
    file_handler = logging.FileHandler(desired_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    root.addHandler(file_handler)
def init_database(db_path: str) -> sqlite3.Connection:
    """
    Initializes a SQLite database for profiling statistics.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS profiling_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_name TEXT,
            func_name TEXT,
            file_name TEXT,
            line_number INTEGER,
            ncalls INTEGER,
            nprimitive_calls INTEGER,
            tottime REAL,
            cumtime REAL,
            callers TEXT
        )
        '''
    )
    conn.commit()
    logger.info(f"Initialized profiling database at {db_path}")
    return conn

def profile_function(func, *args, profile_name="profile", conn=None, output_dir=None, **kwargs):
    """
    Profiles a function and stores its statistics in a database and text file.
    """
    if conn is None or output_dir is None:
        raise ValueError("Both 'conn' and 'output_dir' must be provided.")
    pr = cProfile.Profile()
    pr.enable()
    try:
        result = func(*args, **kwargs)
    finally:
        pr.disable()
        ps = pstats.Stats(pr)
        stats = ps.stats
        cursor = conn.cursor()
        for func_key, func_stats in stats.items():
            file_name, line_number, func_name = func_key
            cc, nc, tt, ct, callers = func_stats
            cursor.execute(
                '''
                INSERT INTO profiling_stats (
                    profile_name, func_name, file_name, line_number,
                    ncalls, nprimitive_calls, tottime, cumtime, callers
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (profile_name, func_name, file_name, line_number, cc, nc, tt, ct, str(callers))
            )
        conn.commit()
        output_file = os.path.join(output_dir, f"{profile_name}_profiling.txt")
        with open(output_file, "w") as f:
            ps.stream = f
            ps.sort_stats("cumulative").print_stats()
    return result

def main():
    parser = argparse.ArgumentParser(description="Profile steps of pointcloudlib toolset.")
    parser.add_argument("--steps", nargs="+", default=["all"], help="Steps to profile. Use 'all' for all steps.")
    parser.add_argument("--input_file", type=str, default="sample.las", help="Path to the input LAS file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to write profiling outputs (default: <input_file_dir>/profiling_runs).")
    args = parser.parse_args()

    # Load the input LAS file
    las_file_path = args.input_file

    base_dir = resolve_profiling_base_dir(las_file_path, output_dir=args.output_dir)
    run_dir = create_run_directory(base_dir=base_dir)
    configure_run_logging(run_dir)
    db_path = os.path.join(run_dir, "profiling_results.sqlite")
    conn = init_database(db_path)
    try:
        point_cloud, header_info = PointCloudIO.read(las_file_path, filetype=FileType.LAS, return_header=True)
        logger.info(f"Loaded {len(point_cloud.points)} points from {las_file_path}")
    except Exception as e:
        logger.exception(f"Failed to read LAS file '{las_file_path}': {e}")
        sys.exit(1)

    # Define profiling steps (adapted to the current structure)
    steps = OrderedDict([
        ("compute_normals", {
            "func": GeometricFeatures.compute_eigenvalues_and_normals,
            "args": [point_cloud, 20],
            "kwargs": {}
        }),
        ("write_las_normals", {
            "func": PointCloudIO.write,
            "args": [os.path.join(run_dir, "output_normals.las"), point_cloud],
            "kwargs": {"filetype": FileType.LAS, "header_info": header_info}
        }),
        ("compute_additional_features", {
            "func": GeometricFeatures.compute_additional_features,
            "args": [point_cloud],
            "kwargs": {}
        }),
        ("write_las_features", {
            "func": PointCloudIO.write,
            "args": [os.path.join(run_dir, "output_features.las"), point_cloud],
            "kwargs": {"filetype": FileType.LAS, "header_info": header_info}
        }),
        ("classify_ground", {
            "func": Classification.classify_ground_axelsson,
            "args": [point_cloud],
            "kwargs": {"max_slope": 0.1, "elevation_threshold": 0.1, "grid_size": 3.5}
        }),
        ("write_las_ground", {
            "func": PointCloudIO.write,
            "args": [os.path.join(run_dir, "output_ground.las"), point_cloud],
            "kwargs": {"filetype": FileType.LAS, "header_info": header_info}
        }),
        ("classify_ruggedness", {
            "func": Classification.classify_ruggedness,
            "args": [point_cloud],
            "kwargs": {"planar_threshold": 0.4, "rugged_threshold": 0.1, "normal_threshold": 10.0}
        }),
        ("write_las_ruggedness", {
            "func": PointCloudIO.write,
            "args": [os.path.join(run_dir, "output_ruggedness.las"), point_cloud],
            "kwargs": {"filetype": FileType.LAS, "header_info": header_info}
        }),
        ("dbscan_clustering", {
            "func": Clustering.dbscan_clustering,
            "args": [point_cloud],
            "kwargs": {"eps": 0.5, "min_samples": 70}
        }),
        ("write_las_dbscan", {
            "func": PointCloudIO.write,
            "args": [os.path.join(run_dir, "output_dbscan.las"), point_cloud],
            "kwargs": {"filetype": FileType.LAS, "header_info": header_info}
        }),
        ("ransac_clustering", {
            "func": Clustering.ransac_clustering,
            "args": [point_cloud],
            "kwargs": {"iterations": 100, "min_inliers": 500, "normal_threshold": 5.0}
        }),
        ("write_las_ransac", {
            "func": PointCloudIO.write,
            "args": [os.path.join(run_dir, "output_ransac.las"), point_cloud],
            "kwargs": {"filetype": FileType.LAS, "header_info": header_info}
        }),
        ("classify_noise", {
            "func": Classification.classify_noise,
            "args": [point_cloud],
            "kwargs": {"radius": 0.5, "min_neighbors": 5}
        }),
        ("write_las_noise", {
            "func": PointCloudIO.write,
            "args": [os.path.join(run_dir, "output_noise.las"), point_cloud],
            "kwargs": {"filetype": FileType.LAS, "header_info": header_info}
        }),
    ])

    selected_steps = list(steps.keys()) if "all" in args.steps else args.steps

    for step in selected_steps:
        if step not in steps:
            logger.warning(f"Step '{step}' is not recognized and will be skipped.")
            continue
        try:
            step_info = steps[step]
            func = step_info["func"]
            args_step = step_info.get("args", [])
            kwargs_step = step_info.get("kwargs", {})
            profile_name = step
            profile_function(func, *args_step, profile_name=profile_name, conn=conn, output_dir=run_dir, **kwargs_step)
            logger.info(f"Step '{step}' completed successfully.")
        except Exception as e:
            logger.error(f"Step '{step}' failed: {e}")

    # Registration step: create a transformed source cloud and perform registration
    angle = np.deg2rad(10)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    transformation_matrix[:3, 3] = [1.0, 0.5, 0.0]
    source_cloud = copy.deepcopy(point_cloud)
    source_cloud.points = PointCloudRegistration.apply_transformation(source_cloud.points, transformation_matrix)
    registration = PointCloudRegistration(source_cloud, point_cloud)
    profile_function(registration.register_with_feature_matching, profile_name="register_with_feature_matching", conn=conn, output_dir=run_dir, k_neighbors=10, curvature_threshold=0.1, feature_max_distance=0.1, max_iterations=15, tolerance=1e-6)
    profile_function(registration.register_icp, profile_name="register_icp", conn=conn, output_dir=run_dir, max_iterations=15, tolerance=1e-6)
    profile_function(PointCloudIO.write, os.path.join(run_dir, "output_registration.las"), point_cloud, filetype=FileType.LAS, header_info=header_info, profile_name="write_las_registration", conn=conn, output_dir=run_dir)

    conn.close()
    logger.info("Closed the profiling database connection.")

if __name__ == "__main__":
    main()
