from setuptools import setup, find_packages

setup(
    name="distributed_gnn",
    version="0.1.0",
    description="Distributed Graph Neural Network Training Framework",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "networkx>=3.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "grpcio>=1.60.0",
        "grpcio-tools>=1.60.0",
        "protobuf>=4.25.0",
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
    ],
)
