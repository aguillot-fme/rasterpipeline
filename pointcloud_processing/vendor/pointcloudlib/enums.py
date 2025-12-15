# pointcloudlib/enums.py

from enum import Enum

class FileType(Enum):
    CSV = "csv"
    LAS = "las"

class ShapeType(Enum):
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"
    TRAPEZOID = "trapezoid"

class ClassificationRule(Enum):
    MAJORITY = "majority"
    MIN = "min"
    MAX = "max"
    CLOSEST = "closest"
