"""
qneural: Machine Learning for Quantum Control

A flexible framework for optimizing quantum control protocols using machine learning.
Supports multiple quantum hardware platforms, ML methods, and optimization objectives.

The framework is designed to be modular and extensible:
    - hardware: Platform-specific physics (Rydberg atoms, superconducting qubits, ions, etc.)
    - core: Platform-agnostic quantum operations (states, gates, metrics)
    - ml: Machine learning methods (neural networks, RL, gradient-free optimization, etc.)
    - control: Control pulse generation and optimization
    - backend: Computational backends (currently PyTorch, JAX support planned)

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

__version__ = "0.1.0"
__authors__ = ["Madhav Mohan", "Julius de Hond"]

# PyTorch backend (primary backend for now)
try:
    import torch

    BACKEND_AVAILABLE = True
except ImportError:
    raise ImportError("qneural requires PyTorch. Install with: pip install torch")

# Configuration
from . import config

# Main imports will be added as we build the package
__all__ = [
    "__version__",
    "__authors__",
    "config",
]
