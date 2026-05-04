"""Setup script for qneural package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qneural",
    version="0.6.0-beta",
    author="Madhav Mohan, Julius de Hond",
    author_email="madhav.mohan@protonmail.com",
    description="Machine Learning for Quantum Control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantockhills/qneural",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "numpy",
        "torchdiffeq",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
        "analysis": [
            "qutip",
        ],
        "tensorflow": [
            "tensorflow>=2.12",
            "tfdiffeq>=0.1.0",
        ],
        "jax": [
            "jax>=0.4.20,<0.10",
            "jaxlib>=0.4.20,<0.10",
            "diffrax>=0.5.0",
            "equinox>=0.11.0",
            "optax>=0.1.7",
        ],
        "all-backends": [
            "tensorflow>=2.12",
            "tfdiffeq>=0.1.0",
            "jax>=0.4.20,<0.10",
            "jaxlib>=0.4.20,<0.10",
            "diffrax>=0.5.0",
            "equinox>=0.11.0",
            "optax>=0.1.7",
        ],
    },
)
