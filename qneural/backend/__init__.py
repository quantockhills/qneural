"""
Backend abstraction layer for qneural.

Provides a unified interface for numerical operations across different backends
(PyTorch, JAX, NumPy). Currently implements PyTorch backend, with JAX support planned.

The backend system allows users to write hardware- and ML-agnostic code that can
run on different computational frameworks.
"""

from .torch_backend import TorchBackend

# Default backend
backend = TorchBackend()

__all__ = ['backend', 'TorchBackend']
