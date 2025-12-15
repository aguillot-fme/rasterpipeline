# PointCloudLib

## Overview
PointCloudLib is a Python library designed for processing and analyzing 3D point cloud data. It provides efficient methods for reading, writing, preprocessing, clustering, classification, and feature extraction of point cloud datasets. The library is built with NumPy, Scikit-Learn, and other numerical computing libraries to ensure high-performance processing.

## Features
- **Point Cloud IO**: Read and write point cloud data in CSV and LAS formats.
- **Preprocessing**: Includes voxel down-sampling, neighborhood computation, and height filtering.
- **Clustering**: Supports K-Means, DBSCAN, Mean-Shift, Spectral Clustering, and Agglomerative Clustering.
- **Classification**: Noise detection, outlier classification using Local Outlier Factor (LOF) and Isolation Forest.
- **Feature Extraction**: Computes eigenvalues, normals, curvature, and other geometric attributes.
- **Registration**: Point cloud alignment using feature matching and Iterative Closest Point (ICP).
- **Profiling**: Provides tools to profile performance using SQLite-based logging.

## Installation
To install the required dependencies, run:
```bash
pip install numpy scipy scikit-learn laspy tqdm
```

## Usage
### Reading a Point Cloud
```python
from pointcloudlib.io import PointCloudIO
from pointcloudlib.enums import FileType

point_cloud = PointCloudIO.read("data/sample.las", filetype=FileType.LAS)
print(point_cloud.points.shape)  # Output: (N, 3)
```

### Preprocessing
```python
from pointcloudlib.preprocessing import PointCloudPreprocessing

filtered_indices = PointCloudPreprocessing.voxel_down_sample(point_cloud, voxel_size=0.1)
point_cloud = point_cloud.points[filtered_indices]
```

### Clustering
```python
from pointcloudlib.clustering import Clustering

labels, cluster_info = Clustering.kmeans_clustering(point_cloud, k=5)
```

### Classification
```python
from pointcloudlib.classification import Classification

noise_mask = Classification.classify_noise(point_cloud, radius=0.1, min_neighbors=5)
```

### Feature Extraction
```python
from pointcloudlib.features import GeometricFeatures

GeometricFeatures.compute_eigenvalues_and_normals(point_cloud, k_neighbors=10)
```

### Registration
```python
from pointcloudlib.registration import PointCloudRegistration

registrar = PointCloudRegistration(source_cloud, target_cloud)
transformation_matrix, aligned_cloud = registrar.register_with_feature_matching()
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue on the repository.

## License
This project is licensed under the MIT License.

## Docker Development & Tests

This repo includes a CPU-only Docker/Compose setup for a reproducible dev shell and test runs.

- Dev shell: 
  - docker compose run --rm dev`r
- Run tests: 
  - docker compose run --rm test`r

On Windows PowerShell you can also use:
- scripts\\run_docker.ps1 -Service dev`r
- scripts\\run_docker.ps1 -Service test`r
