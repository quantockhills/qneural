"""
General quantum operators (hardware-agnostic).

Standard operators used across quantum computing platforms:
    - Pauli matrices (X, Y, Z)
    - Rotation operators
    - Projection operators
"""

from ..backend import backend
from ..config import DTYPE_COMPLEX, DEVICE


# =============================================================================
# Pauli Matrices
# =============================================================================


def pauli_matrices(device=None):
    """
    Get the standard Pauli matrices.

    Returns
    -------
    dict
        Dictionary with keys 'X', 'Y', 'Z', 'I' containing the Pauli matrices

    Examples
    --------
    >>> paulis = pauli_matrices()
    >>> sigma_x = paulis['X']
    >>> sigma_z = paulis['Z']
    """
    device = device or DEVICE

    sigma_x = backend.tensor([[0, 1], [1, 0]], dtype=DTYPE_COMPLEX, device=device)
    sigma_y = backend.tensor([[0, -1j], [1j, 0]], dtype=DTYPE_COMPLEX, device=device)
    sigma_z = backend.tensor([[1, 0], [0, -1]], dtype=DTYPE_COMPLEX, device=device)
    identity = backend.eye(2, dtype=DTYPE_COMPLEX, device=device)

    return {"X": sigma_x, "Y": sigma_y, "Z": sigma_z, "I": identity}


# Pre-defined Pauli matrices for convenience
_paulis = pauli_matrices()
SIGMA_X = _paulis["X"]
SIGMA_Y = _paulis["Y"]
SIGMA_Z = _paulis["Z"]
IDENTITY_2 = _paulis["I"]


# =============================================================================
# Single-Qubit Rotations
# =============================================================================


def rotation_x(theta, device=None):
    """
    Rotation around X-axis: R_x(θ) = exp(-iθ σ_x / 2).

    Parameters
    ----------
    theta : float or torch.Tensor
        Rotation angle
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        2×2 rotation matrix
    """
    device = device or DEVICE
    sigma_x = SIGMA_X.to(device)
    return backend.matrix_exp(-0.5j * theta * sigma_x)


def rotation_y(theta, device=None):
    """
    Rotation around Y-axis: R_y(θ) = exp(-iθ σ_y / 2).

    Parameters
    ----------
    theta : float or torch.Tensor
        Rotation angle
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        2×2 rotation matrix
    """
    device = device or DEVICE
    sigma_y = SIGMA_Y.to(device)
    return backend.matrix_exp(-0.5j * theta * sigma_y)


def rotation_z(theta, device=None):
    """
    Rotation around Z-axis: R_z(θ) = exp(-iθ σ_z / 2).

    Parameters
    ----------
    theta : float or torch.Tensor
        Rotation angle
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        2×2 rotation matrix
    """
    device = device or DEVICE
    sigma_z = SIGMA_Z.to(device)
    return backend.matrix_exp(-0.5j * theta * sigma_z)


def arbitrary_rotation(alpha_x, alpha_y, alpha_z, device=None):
    """
    Arbitrary single-qubit rotation via Euler angles.

    Implements: R_z(α_z) R_y(α_y) R_z(α_x)

    Parameters
    ----------
    alpha_x, alpha_y, alpha_z : float or torch.Tensor
        Euler angles
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        2×2 unitary matrix

    Notes
    -----
    Any single-qubit unitary can be decomposed this way (up to global phase).
    """
    device = device or DEVICE

    R_z1 = rotation_z(alpha_x, device)
    R_y = rotation_y(alpha_y, device)
    R_z2 = rotation_z(alpha_z, device)

    return backend.matmul(backend.matmul(R_z2, R_y), R_z1)


# =============================================================================
# Projection Operators
# =============================================================================


def projector(state_index, dim=2, device=None):
    """
    Create a projection operator |i⟩⟨i|.

    Parameters
    ----------
    state_index : int
        Index of state to project onto
    dim : int, optional
        Hilbert space dimension (default: 2)
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        Projection matrix
    """
    device = device or DEVICE
    proj = backend.zeros((dim, dim), dtype=DTYPE_COMPLEX, device=device)
    proj = backend.index_set(proj, (state_index, state_index), 1.0)
    return proj
