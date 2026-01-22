"""
Setup script for CTGV System
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ctgv-system",
    version="1.0.0",
    author="BEGNOMAR DOS SANTOS PORTO",
    author_email="begnomar@gmail.com",
    description="Versatile Geometric Topology Computing System - A revolutionary cognitive engine for distributed topological processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bear-urso/CTGV-System---Versatile-Geometric-Topology-Computing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "gpu": ["cupy>=10.0.0", "numba>=0.56.0"],
        "ml": ["scikit-learn>=1.0.0"],
        "dev": ["pytest>=6.0", "black", "flake8", "sphinx", "coverage"],
        "all": ["cupy>=10.0.0", "numba>=0.56.0", "scikit-learn>=1.0.0"]
    },
)