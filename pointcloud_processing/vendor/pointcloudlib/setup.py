from setuptools import setup, find_packages

setup(
    name="pointcloudlib",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "laspy",
        "scikit-learn"
    ],
    author="Antoine Guillot",
    author_email="aguillot35@gmail.com",
    description="A library for point cloud processing",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)