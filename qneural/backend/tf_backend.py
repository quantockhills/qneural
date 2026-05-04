"""
TensorFlow backend implementation.

Provides TF-based implementations of all backend operations.
"""

from typing import Optional
import numpy as np
from ..config import DEVICE, DTYPE_REAL, DTYPE_COMPLEX


class TfBackend:
    """
    TensorFlow backend for qneural.

    Provides a consistent interface for tensor operations, linear algebra,
    and automatic differentiation using TensorFlow.
    """

    def __init__(self, device: Optional[str] = None):
        import tensorflow as tf

        self.device = device if device is not None else DEVICE
        self.lib = tf

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
        return self.lib.constant(data, dtype=dtype)

    def arange(self, *args, dtype=None, device=None):
        return self.lib.range(*args, dtype=dtype)

    def linspace(self, start, end, steps, dtype=None, device=None):
        return self.lib.linspace(start, end, steps)

    # =========================================================================
    # Linear Algebra
    # =========================================================================

    def kron(self, a, b):
        a_t = self.lib.convert_to_tensor(a)
        b_t = self.lib.convert_to_tensor(b)
        a_r = self.lib.reshape(a_t, [self.lib.reduce_prod(a_t.shape)])
        b_r = self.lib.reshape(b_t, [self.lib.reduce_prod(b_t.shape)])
        result = self.lib.tensordot(a_r, b_r, axes=0)
        result = self.lib.reshape(
            result,
            [s * b_t.shape[1] if i % 2 == 1 else s for i, s in enumerate(a_t.shape * 2)],
        )
        shape_out = tuple(a_t.shape[i] * b_t.shape[j] for i in range(a_t.shape.rank)
                          for j in range(b_t.shape.rank))
        return self.lib.reshape(result, [int(d) for d in shape_out])

    def matmul(self, a, b):
        return self.lib.linalg.matmul(a, b)

    def bmm(self, a, b):
        return self.lib.linalg.matmul(a, b)

    def matrix_exp(self, mat):
        return self.lib.linalg.expm(mat)

    def einsum(self, equation, *tensors):
        return self.lib.einsum(equation, *tensors)

    def trace(self, mat):
        return self.lib.linalg.trace(mat)

    def conj(self, tensor):
        return self.lib.math.conj(tensor)

    def transpose(self, tensor, dim0, dim1):
        rank = len(tensor.shape)
        perm = list(range(rank))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return self.lib.transpose(tensor, perm)

    # =========================================================================
    # Element-wise Operations
    # =========================================================================

    def exp(self, tensor):
        return self.lib.math.exp(tensor)

    def sin(self, tensor):
        return self.lib.math.sin(tensor)

    def cos(self, tensor):
        return self.lib.math.cos(tensor)

    def sqrt(self, tensor):
        return self.lib.math.sqrt(tensor)

    def abs(self, tensor):
        return self.lib.math.abs(tensor)

    def angle(self, tensor):
        return self.lib.math.angle(tensor)

    def square(self, tensor):
        return self.lib.math.square(tensor)

    # =========================================================================
    # Reductions
    # =========================================================================

    def sum(self, tensor, dim=None, keepdim=False):
        if dim is None:
            return self.lib.math.reduce_sum(tensor)
        return self.lib.math.reduce_sum(tensor, axis=dim, keepdims=keepdim)

    def mean(self, tensor, dim=None, keepdim=False):
        if dim is None:
            return self.lib.math.reduce_mean(tensor)
        return self.lib.math.reduce_mean(tensor, axis=dim, keepdims=keepdim)

    def max(self, tensor, dim=None, keepdim=False):
        if dim is None:
            return self.lib.math.reduce_max(tensor)
        return self.lib.math.reduce_max(tensor, axis=dim, keepdims=keepdim)

    def min(self, tensor, dim=None, keepdim=False):
        if dim is None:
            return self.lib.math.reduce_min(tensor)
        return self.lib.math.reduce_min(tensor, axis=dim, keepdims=keepdim)

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
        return self.lib.concat(tensors, axis=dim)

    def stack(self, tensors, dim=0):
        return self.lib.stack(tensors, axis=dim)

    # =========================================================================
    # Indexing
    # =========================================================================

    def select(self, tensor, dim, index):
        begin = [0] * len(tensor.shape)
        size = list(tensor.shape)
        begin[dim] = index
        size[dim] = 1
        result = self.lib.slice(tensor, begin, size)
        return self.lib.squeeze(result, axis=dim)

    def gather(self, tensor, dim, index):
        return self.lib.gather(tensor, index, axis=dim, batch_dims=dim)

    # =========================================================================
    # Random
    # =========================================================================

    def rand(self, *shape, dtype=None, device=None):
        dtype = dtype or DTYPE_REAL
        return self.lib.random.uniform(shape, dtype=dtype)

    def randn(self, *shape, dtype=None, device=None):
        dtype = dtype or DTYPE_REAL
        return self.lib.random.normal(shape, dtype=dtype)

    # =========================================================================
    # Conditional and Clamping
    # =========================================================================

    def where(self, condition, x, y):
        return self.lib.where(condition, x, y)

    def floor(self, tensor):
        return self.lib.math.floor(tensor)

    def clamp(self, tensor, min_val, max_val):
        return self.lib.clip_by_value(tensor, min_val, max_val)

    def zeros_like(self, tensor):
        return self.lib.zeros_like(tensor)

    # =========================================================================
    # Integration
    # =========================================================================

    def trapz(self, y, x):
        return self.lib.math.trapezoid(y, x)

    # =========================================================================
    # Type Conversion
    # =========================================================================

    def long(self, tensor):
        return self.lib.cast(tensor, self.lib.int64)

    def diag(self, tensor, diagonal=0):
        return self.lib.linalg.diag(tensor)

    @property
    def pi(self):
        return self.lib.constant(3.141592653589793, dtype=DTYPE_REAL)

    # =========================================================================
    # Utilities
    # =========================================================================

    def norm(self, tensor, p="fro"):
        ord_map = {"fro": "fro", "frobenius": "fro", 2: 2, 1: 1, "inf": float("inf")}
        return self.lib.norm(tensor, ord=ord_map.get(p, p))

    def is_complex(self, tensor):
        return tensor.dtype.is_complex

    def is_tensor(self, obj):
        return self.lib.is_tensor(obj)

    def no_grad_context(self):
        import contextlib
        return contextlib.nullcontext()

    def to_numpy(self, tensor):
        return tensor.numpy()

    def from_numpy(self, array, dtype=None, device=None):
        return self.lib.convert_to_tensor(array, dtype=dtype)

    def detach(self, tensor):
        return self.lib.stop_gradient(tensor)

    def clone(self, tensor):
        return self.lib.identity(tensor)

    def item(self, tensor):
        return tensor.numpy().item()

    def index_set(self, tensor, indices, value):
        return self.lib.tensor_scatter_nd_update(
            tensor, [list(indices)], [self.lib.cast(value, tensor.dtype)]
        )
