"""JAX backend implementation."""

from typing import Optional
import numpy as np
from ..config import DEVICE, DTYPE_REAL, DTYPE_COMPLEX


class JaxBackend:
    """
    JAX backend for qneural.

    Provides a consistent interface for tensor operations, linear algebra,
    and automatic differentiation using JAX. Uses functional random key
    management for PRNG.
    """

    def __init__(self, device: Optional[str] = None):
        import jax
        import jax.numpy as jnp
        import jax.scipy.linalg

        self.device = device if device is not None else DEVICE
        self.lib = jnp
        self._jax = jax
        self._scipy_linalg = jax.scipy.linalg
        self._rng_key = jax.random.PRNGKey(42)

    def _next_key(self):
        self._rng_key, subkey = self._jax.random.split(self._rng_key)
        return subkey

    # =========================================================================
    # Tensor Creation
    # =========================================================================

    def zeros(self, shape, dtype=None, device=None):
        dtype = dtype or DTYPE_REAL
        return self.lib.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None, device=None):
        dtype = dtype or DTYPE_REAL
        return self.lib.ones(shape, dtype=dtype)

    def eye(self, n, dtype=None, device=None):
        dtype = dtype or DTYPE_COMPLEX
        return self.lib.eye(n, dtype=dtype)

    def tensor(self, data, dtype=None, device=None, requires_grad=False):
        return self.lib.array(data, dtype=dtype)

    def arange(self, *args, dtype=None, device=None):
        return self.lib.arange(*args, dtype=dtype)

    def linspace(self, start, end, steps, dtype=None, device=None):
        return self.lib.linspace(start, end, steps, dtype=dtype)

    # =========================================================================
    # Linear Algebra
    # =========================================================================

    def kron(self, a, b):
        return self.lib.kron(a, b)

    def matmul(self, a, b):
        return self.lib.matmul(a, b)

    def bmm(self, a, b):
        return self.lib.matmul(a, b)

    def matrix_exp(self, mat):
        return self._scipy_linalg.expm(mat)

    def einsum(self, equation, *tensors):
        return self.lib.einsum(equation, *tensors)

    def trace(self, mat):
        return self.lib.trace(mat)

    def conj(self, tensor):
        return self.lib.conj(tensor)

    def transpose(self, tensor, dim0, dim1):
        return self.lib.swapaxes(tensor, dim0, dim1)

    # =========================================================================
    # Element-wise Operations
    # =========================================================================

    def exp(self, tensor):
        return self.lib.exp(tensor)

    def sin(self, tensor):
        return self.lib.sin(tensor)

    def cos(self, tensor):
        return self.lib.cos(tensor)

    def sqrt(self, tensor):
        return self.lib.sqrt(tensor)

    def abs(self, tensor):
        return self.lib.abs(tensor)

    def angle(self, tensor):
        return self.lib.angle(tensor)

    def square(self, tensor):
        return self.lib.square(tensor)

    # =========================================================================
    # Reductions
    # =========================================================================

    def sum(self, tensor, dim=None, keepdim=False):
        if dim is None:
            return self.lib.sum(tensor)
        return self.lib.sum(tensor, axis=dim, keepdims=keepdim)

    def mean(self, tensor, dim=None, keepdim=False):
        if dim is None:
            return self.lib.mean(tensor)
        return self.lib.mean(tensor, axis=dim, keepdims=keepdim)

    def max(self, tensor, dim=None, keepdim=False):
        if dim is None:
            return self.lib.max(tensor)
        return self.lib.max(tensor, axis=dim, keepdims=keepdim)

    def min(self, tensor, dim=None, keepdim=False):
        if dim is None:
            return self.lib.min(tensor)
        return self.lib.min(tensor, axis=dim, keepdims=keepdim)

    # =========================================================================
    # Shape Manipulation
    # =========================================================================

    def reshape(self, tensor, shape):
        return self.lib.reshape(tensor, shape)

    def view(self, tensor, *shape):
        return self.lib.reshape(tensor, shape)

    def unsqueeze(self, tensor, dim):
        return self.lib.expand_dims(tensor, axis=dim)

    def expand(self, tensor, shape):
        return self.lib.broadcast_to(tensor, shape)

    def squeeze(self, tensor, dim=None):
        if dim is None:
            return self.lib.squeeze(tensor)
        return self.lib.squeeze(tensor, axis=dim)

    def cat(self, tensors, dim=0):
        return self.lib.concatenate(tensors, axis=dim)

    def stack(self, tensors, dim=0):
        return self.lib.stack(tensors, axis=dim)

    # =========================================================================
    # Indexing
    # =========================================================================

    def select(self, tensor, dim, index):
        return self.lib.take(tensor, index, axis=dim)

    def gather(self, tensor, dim, index):
        return self.lib.take_along_axis(tensor, index[..., None], axis=dim)

    # =========================================================================
    # Random
    # =========================================================================

    def rand(self, *shape, dtype=None, device=None):
        dtype = dtype or DTYPE_REAL
        return self._jax.random.uniform(self._next_key(), shape, dtype=dtype)

    def randn(self, *shape, dtype=None, device=None):
        dtype = dtype or DTYPE_REAL
        return self._jax.random.normal(self._next_key(), shape, dtype=dtype)

    # =========================================================================
    # Conditional and Clamping
    # =========================================================================

    def where(self, condition, x, y):
        return self.lib.where(condition, x, y)

    def floor(self, tensor):
        return self.lib.floor(tensor)

    def clamp(self, tensor, min_val, max_val):
        return self.lib.clip(tensor, min_val, max_val)

    def zeros_like(self, tensor):
        return self.lib.zeros_like(tensor)

    # =========================================================================
    # Integration
    # =========================================================================

    def trapz(self, y, x):
        dx = x[1] - x[0] if x.shape[0] > 1 else 0.0
        return self.lib.trapz(y, x=dx, axis=0)

    # =========================================================================
    # Type Conversion
    # =========================================================================

    def long(self, tensor):
        return self.lib.array(tensor).astype(self.lib.int64)

    def diag(self, tensor, diagonal=0):
        return self.lib.diag(tensor, k=diagonal)

    @property
    def pi(self):
        return self.lib.array(3.141592653589793, dtype=DTYPE_REAL)

    # =========================================================================
    # Utilities
    # =========================================================================

    def norm(self, tensor, p="fro"):
        return self.lib.linalg.norm(tensor, ord=p)

    def is_complex(self, tensor):
        return tensor.dtype.kind == 'c'

    def is_tensor(self, obj):
        return isinstance(obj, self.lib.ndarray)

    def no_grad_context(self):
        import contextlib
        return contextlib.nullcontext()

    def to_numpy(self, tensor):
        return np.asarray(tensor)

    def from_numpy(self, array, dtype=None, device=None):
        return self.lib.array(array, dtype=dtype)

    def detach(self, tensor):
        return self._jax.lax.stop_gradient(tensor)

    def clone(self, tensor):
        return self.lib.array(tensor)

    def item(self, tensor):
        return tensor.item()

    def index_set(self, tensor, indices, value):
        return tensor.at[indices].set(value)
