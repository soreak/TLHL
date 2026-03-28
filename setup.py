from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="two-layer-hnsw-like",
    version="0.1.0",
    description="Two-layer HNSW-like ANN index",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Your Name",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=False,
    package_data={},
    exclude_package_data={
        "two_layer_hnsw_like": ["data/*", "data/**/*"],
    },
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scikit-learn",
    ],
)