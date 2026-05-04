"""
Backend abstraction layer for qneural.

Provides a unified interface for numerical operations across different backends
(PyTorch, TensorFlow, JAX).

The backend system allows users to write hardware- and ML-agnostic code that can
run on different computational frameworks. Backend selection is controlled via:

    - Environment variable: QNEURAL_BACKEND=pytorch|tensorflow|jax
    - Runtime:  qneural.config.set_backend("tensorflow")
"""

from ..config import BACKEND

backend = None

if BACKEND == "pytorch":
    from .torch_backend import TorchBackend
    backend = TorchBackend()
elif BACKEND == "tensorflow":
    from .tf_backend import TfBackend
    backend = TfBackend()
elif BACKEND == "jax":
    from .jax_backend import JaxBackend
    backend = JaxBackend()

def _reinit_backend():
    global backend
    from ..config import BACKEND
    if BACKEND == "pytorch":
        from .torch_backend import TorchBackend
        backend = TorchBackend()
    elif BACKEND == "tensorflow":
        from .tf_backend import TfBackend
        backend = TfBackend()
    elif BACKEND == "jax":
        from .jax_backend import JaxBackend
        backend = JaxBackend()

__all__ = ["backend", "_reinit_backend"]
