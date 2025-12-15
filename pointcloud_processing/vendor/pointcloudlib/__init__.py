# pointcloudlib/__init__.py

from .enums import FileType, ShapeType, ClassificationRule
from .pointcloud import PointCloud
from .io import PointCloudIO
from .preprocessing import PointCloudPreprocessing
from .geometry import GeometryUtils, MathUtils
from .features import GeometricFeatures, StatisticalFeatures, FeatureAggregation

try:
    from .classification import Classification
except ImportError:  # optional dependency (scikit-learn)
    Classification = None

try:
    from .clustering import Clustering
except ImportError:  # optional dependency (scikit-learn / hdbscan)
    Clustering = None

try:
    from .registration import PointCloudRegistration
except ImportError:  # optional dependency (scipy)
    PointCloudRegistration = None

__all__ = [
    "FileType",
    "ShapeType",
    "ClassificationRule",
    "PointCloud",
    "PointCloudIO",
    "PointCloudPreprocessing",
    "GeometryUtils",
    "MathUtils",
    "GeometricFeatures",
    "StatisticalFeatures",
    "FeatureAggregation",
    "Classification",
    "Clustering",
    "PointCloudRegistration",
]