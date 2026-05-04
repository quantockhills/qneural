"""
qneural: Machine Learning for Quantum Control

A flexible framework for optimizing quantum control protocols using machine learning.
Supports multiple quantum hardware platforms, ML methods, and optimization objectives.

The framework is designed to be modular and extensible:
    - hardware: Platform-specific physics (Rydberg atoms, superconducting qubits, ions, etc.)
    - core: Platform-agnostic quantum operations (states, gates, metrics)
    - ml: Machine learning methods (neural networks, RL, gradient-free optimization, etc.)
    - control: Control pulse generation and optimization
    - backend: Computational backends (PyTorch, TensorFlow, JAX)

Initially developed for neural network-based pulse optimization on Rydberg atom systems,
demonstrating state-of-the-art results for parametrized multi-qubit gates. The architecture
allows straightforward extension to other hardware platforms and ML approaches.

Example use cases:
    - Parametrized gate pulse optimization (proven on neutral atoms)
    - Robust control against noise and decoherence
    - Time-optimal gate synthesis
    - Multi-objective optimization (speed, fidelity, resource efficiency)
    - Transfer learning across hardware platforms
"""

__version__ = "0.5.0-beta"
__authors__ = ["Madhav Mohan", "Julius de Hond"]

# Backend selection
from .config import BACKEND, set_backend, get_backend

BACKEND_AVAILABLE = True

if BACKEND == "pytorch":
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyTorch backend selected but PyTorch not installed. "
            "Install with: pip install torch, or switch backend with "
            "set_backend('tensorflow') / set_backend('jax')"
        )
elif BACKEND == "tensorflow":
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        raise ImportError(
            "TensorFlow backend selected but TensorFlow not installed. "
            "Install with: pip install qneural[tensorflow], "
            "or switch backend with set_backend('pytorch') / set_backend('jax')"
        )
elif BACKEND == "jax":
    try:
        import jax  # noqa: F401
    except ImportError:
        raise ImportError(
            "JAX backend selected but JAX not installed. "
            "Install with: pip install qneural[jax], "
            "or switch backend with set_backend('pytorch') / set_backend('tensorflow')"
        )

import sys
from . import config
from . import backend as _backend_pkg

_self = sys.modules[__name__]
if hasattr(_self, 'backend'):
    del _self.backend

def __getattr__(name):
    if name == 'backend':
        return _backend_pkg.backend
    raise AttributeError(f"module 'qneural' has no attribute '{name}'")

# Main imports will be added as we build the package
__all__ = [
    "__version__",
    "__authors__",
    "config",
    "backend",
    "set_backend",
    "get_backend",
]
