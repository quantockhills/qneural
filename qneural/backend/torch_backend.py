"""
PyTorch backend implementation.

Provides PyTorch-based implementations of all backend operations.
"""

import torch
from typing import Optional
from ..config import DEVICE, DTYPE_REAL, DTYPE_COMPLEX


class TorchBackend:
    """
    PyTorch backend for qneural.

    Provides a consistent interface for tensor operations, linear algebra,
    and automatic differentiation using PyTorch.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the PyTorch backend.

        Parameters
        ----------
        device : str, optional
            Device to use ('cpu', 'cuda', 'mps'). If None, uses global config.
        """
        self.device = device if device is not None else DEVICE
        self.lib = torch  # Underlying library

    # =========================================================================
    # Tensor Creation
    # =========================================================================

    def zeros(self, shape, dtype=None, device=None):
        """Create a tensor of zeros."""
        dtype = dtype or DTYPE_REAL
        device = device or self.device
        return torch.zeros(shape, dtype=dtype, device=device)

    def ones(self, shape, dtype=None, device=None):
        """Create a tensor of ones."""
        dtype = dtype or DTYPE_REAL
        device = device or self.device
        return torch.ones(shape, dtype=dtype, device=device)

    def eye(self, n, dtype=None, device=None):
        """Create an identity matrix."""
        dtype = dtype or DTYPE_COMPLEX
        device = device or self.device
        return torch.eye(n, dtype=dtype, device=device)

    def tensor(self, data, dtype=None, device=None, requires_grad=False):
        """Create a tensor from data."""
        device = device or self.device
        return torch.tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )

    def arange(self, *args, dtype=None, device=None):
        """Create a range tensor."""
        device = device or self.device
        return torch.arange(*args, dtype=dtype, device=device)

    def linspace(self, start, end, steps, dtype=None, device=None):
        """Create a linearly spaced tensor."""
        device = device or self.device
        return torch.linspace(start, end, steps, dtype=dtype, device=device)

    # =========================================================================
    # Linear Algebra
    # =========================================================================

    def kron(self, a, b):
        """Kronecker product (tensor product)."""
        return torch.kron(a, b)

    def matmul(self, a, b):
        """Matrix multiplication."""
        return torch.matmul(a, b)

    def bmm(self, a, b):
        """Batch matrix multiplication."""
        return torch.bmm(a, b)

    def matrix_exp(self, mat):
        """Matrix exponential."""
        return torch.matrix_exp(mat)

    def einsum(self, equation, *tensors):
        """Einstein summation."""
        return torch.einsum(equation, *tensors)

    def trace(self, mat):
        """Matrix trace."""
        return torch.trace(mat)

    def conj(self, tensor):
        """Complex conjugate."""
        return torch.conj(tensor)

    def transpose(self, tensor, dim0, dim1):
        """Transpose dimensions."""
        return torch.transpose(tensor, dim0, dim1)

    # =========================================================================
    # Element-wise Operations
    # =========================================================================

    def exp(self, tensor):
        """Element-wise exponential."""
        return torch.exp(tensor)

    def sin(self, tensor):
        """Element-wise sine."""
        return torch.sin(tensor)

    def cos(self, tensor):
        """Element-wise cosine."""
        return torch.cos(tensor)

    def sqrt(self, tensor):
        """Element-wise square root."""
        return torch.sqrt(tensor)

    def abs(self, tensor):
        """Element-wise absolute value / modulus."""
        return torch.abs(tensor)

    def angle(self, tensor):
        """Complex argument (phase)."""
        return torch.angle(tensor)

    def square(self, tensor):
        """Element-wise square."""
        return torch.square(tensor)

    # =========================================================================
    # Reductions
    # =========================================================================

    def sum(self, tensor, dim=None, keepdim=False):
        """Sum over dimensions."""
        if dim is None:
            return torch.sum(tensor)
        return torch.sum(tensor, dim=dim, keepdim=keepdim)

    def mean(self, tensor, dim=None, keepdim=False):
        """Mean over dimensions."""
        if dim is None:
            return torch.mean(tensor)
        return torch.mean(tensor, dim=dim, keepdim=keepdim)

    def max(self, tensor, dim=None, keepdim=False):
        """Maximum value."""
        if dim is None:
            return torch.max(tensor)
        return torch.max(tensor, dim=dim, keepdim=keepdim)

    def min(self, tensor, dim=None, keepdim=False):
        """Minimum value."""
        if dim is None:
            return torch.min(tensor)
        return torch.min(tensor, dim=dim, keepdim=keepdim)

    # =========================================================================
    # Shape Manipulation
    # =========================================================================

    def reshape(self, tensor, shape):
        """Reshape tensor."""
        return torch.reshape(tensor, shape)

    def view(self, tensor, *shape):
        """View tensor with new shape."""
        return tensor.view(*shape)

    def unsqueeze(self, tensor, dim):
        """Add dimension."""
        return torch.unsqueeze(tensor, dim)

    def expand(self, tensor, shape):
        return tensor.expand(*shape)

    def squeeze(self, tensor, dim=None):
        """Remove dimension."""
        if dim is None:
            return torch.squeeze(tensor)
        return torch.squeeze(tensor, dim=dim)

    def cat(self, tensors, dim=0):
        """Concatenate tensors."""
        return torch.cat(tensors, dim=dim)

    def stack(self, tensors, dim=0):
        """Stack tensors."""
        return torch.stack(tensors, dim=dim)

    # =========================================================================
    # Indexing
    # =========================================================================

    def select(self, tensor, dim, index):
        """Select along dimension."""
        return tensor.select(dim, index)

    def gather(self, tensor, dim, index):
        """Gather values along dimension."""
        return torch.gather(tensor, dim, index)

    # =========================================================================
    # Random
    # =========================================================================

    def rand(self, *shape, dtype=None, device=None):
        """Uniform random in [0, 1)."""
        dtype = dtype or DTYPE_REAL
        device = device or self.device
        return torch.rand(*shape, dtype=dtype, device=device)

    def randn(self, *shape, dtype=None, device=None):
        """Standard normal random."""
        dtype = dtype or DTYPE_REAL
        device = device or self.device
        return torch.randn(*shape, dtype=dtype, device=device)

    # =========================================================================
    # Conditional and Clamping
    # =========================================================================

    def where(self, condition, x, y):
        return torch.where(condition, x, y)

    def floor(self, tensor):
        return torch.floor(tensor)

    def clamp(self, tensor, min_val, max_val):
        return torch.clamp(tensor, min_val, max_val)

    def zeros_like(self, tensor):
        return torch.zeros_like(tensor)

    # =========================================================================
    # Integration
    # =========================================================================

    def trapz(self, y, x):
        return torch.trapz(y, x)

    # =========================================================================
    # Type Conversion
    # =========================================================================

    def long(self, tensor):
        return tensor.long()

    def diag(self, tensor, diagonal=0):
        return torch.diag(tensor, diagonal)

    @property
    def pi(self):
        return torch.tensor(torch.pi, dtype=DTYPE_REAL)

    # =========================================================================
    # Utilities
    # =========================================================================

    def norm(self, tensor, p="fro"):
        return torch.norm(tensor, p=p)

    def is_complex(self, tensor):
        return tensor.dtype.is_complex

    def is_tensor(self, obj):
        return torch.is_tensor(obj)

    def no_grad_context(self):
        return torch.no_grad()

    def to_numpy(self, tensor):
        """Convert to numpy array."""
        return tensor.detach().cpu().numpy()

    def from_numpy(self, array, dtype=None, device=None):
        """Create tensor from numpy array."""
        device = device or self.device
        return torch.from_numpy(array).to(dtype=dtype, device=device)

    def detach(self, tensor):
        """Detach from computation graph."""
        return tensor.detach()

    def clone(self, tensor):
        """Clone tensor."""
        return tensor.clone()

    def item(self, tensor):
        """Extract scalar value."""
        return tensor.item()

    def index_set(self, tensor, indices, value):
        """Set tensor at indices to value. Returns modified tensor."""
        tensor[indices] = value
        return tensor
