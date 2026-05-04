"""
Quantum gate construction and manipulation.

Functions for creating and analyzing quantum gates, including:
    - Standard gates (CZ, CNOT, etc.)
    - Parametrized gate families (CZ_φ, CCZ_φ)
    - Gate fidelity corrections
    - Basis reductions
"""

from ..backend import backend
from ..config import DTYPE_COMPLEX, DEVICE


# =============================================================================
# Two-Qubit Gates
# =============================================================================


def cz_gate(device=None):
    """
    Standard controlled-Z gate.

    Returns
    -------
    torch.Tensor
        4×4 CZ gate matrix in computational basis {|00⟩, |01⟩, |10⟩, |11⟩}

    Notes
    -----
    CZ = diag(1, 1, 1, -1)
    Equivalent to CZ_φ with φ = π
    """
    device = device or DEVICE
    cz = backend.eye(4, dtype=DTYPE_COMPLEX, device=device)
    cz = backend.index_set(cz, (3, 3), -1.0)
    return cz


def czphi_gate(phi, device=None):
    """
    Parametrized controlled-Z gate: CZ_φ = diag(1, 1, 1, e^{iφ}).

    Parameters
    ----------
    phi : float or torch.Tensor
        Phase angle (radians)
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        4×4 gate matrix

    Notes
    -----
    This is the controlled phase gate, also called CPHASE(φ).
    Acts as: |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |10⟩, |11⟩ → e^{iφ}|11⟩
    """
    device = device or DEVICE
    gate = backend.eye(4, dtype=DTYPE_COMPLEX, device=device)
    if not hasattr(phi, 'shape'):
        phi = backend.tensor(float(phi), device=device)
    gate = backend.index_set(
        gate, (3, 3),
        backend.exp(backend.tensor(1.0j, device=device) * phi)
    )
    return gate


def czp_gate_stack(phi_tensor, device=None):
    """
    Create a batch of CZ_φ gates for different phase angles.

    Useful for training neural networks over multiple target angles.

    Parameters
    ----------
    phi_tensor : torch.Tensor
        Tensor of phase angles, shape [n] or [n, 1]
    device : str, optional
        Device to place tensors on

    Returns
    -------
    torch.Tensor
        Stacked gates, shape [n, 4, 4]

    Examples
    --------
    >>> angles = torch.linspace(0, torch.pi, 10)
    >>> gates = czp_gate_stack(angles)
    >>> gates.shape
    torch.Size([10, 4, 4])
    """
    device = device or DEVICE

    # Ensure phi_tensor is 1D
    if len(phi_tensor.shape) > 1:
        phi_tensor = phi_tensor.squeeze()

    batch_size = len(phi_tensor)
    gates = []

    for phi in phi_tensor:
        gate = czphi_gate(phi, device)
        gates.append(gate)

    return backend.stack(gates)


# =============================================================================
# Three-Qubit Gates
# =============================================================================


def cczphi_gate(phi, device=None):
    """
    Three-qubit controlled-controlled-Z_φ gate.

    Applies phase only when all three qubits are in |1⟩.

    Parameters
    ----------
    phi : float or torch.Tensor
        Phase angle (radians)
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        8×8 gate matrix

    Notes
    -----
    CCZ_φ = diag(1, 1, 1, 1, 1, 1, 1, e^{iφ})
    Acts on basis {|000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩}
    """
    device = device or DEVICE
    gate = backend.eye(8, dtype=DTYPE_COMPLEX, device=device)
    if not hasattr(phi, 'shape'):
        phi = backend.tensor(float(phi), device=device)
    gate = backend.index_set(
        gate, (-1, -1),
        backend.exp(backend.tensor(1.0j, device=device) * phi)
    )
    return gate


def cczphi_gate_zzz(phi, device=None):
    """
    Three-qubit CCZ_φ gate with phase on |000⟩ state.

    Alternative implementation where phase is on |000⟩ instead of |111⟩,
    up to a global phase factor.

    Parameters
    ----------
    phi : float or torch.Tensor
        Phase angle (radians)
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        8×8 gate matrix

    Notes
    -----
    CCZ_φ^{zzz} = diag(e^{iφ}, -1, -1, -1, -1, -1, -1, -1)
    Related to standard CCZ_φ by basis change.
    """
    device = device or DEVICE
    gate = -1.0 * backend.eye(8, dtype=DTYPE_COMPLEX, device=device)
    if not hasattr(phi, 'shape'):
        phi = backend.tensor(float(phi), device=device)
    gate = backend.index_set(
        gate, (0, 0),
        backend.exp(backend.tensor(1.0j, device=device) * phi)
    )
    return gate


def cczp_gate_stack(phi_tensor, device=None):
    """
    Create a batch of CCZ_φ gates for different phase angles.

    Parameters
    ----------
    phi_tensor : torch.Tensor
        Tensor of phase angles, shape [n] or [n, 1]
    device : str, optional
        Device to place tensors on

    Returns
    -------
    torch.Tensor
        Stacked gates, shape [n, 8, 8]
    """
    device = device or DEVICE

    if len(phi_tensor.shape) > 1:
        phi_tensor = phi_tensor.squeeze()

    batch_size = len(phi_tensor)
    gates = []

    for phi in phi_tensor:
        gate = cczphi_gate(phi, device)
        gates.append(gate)

    return backend.stack(gates)


def cczp_gate_stack_zzz(phi_tensor, device=None):
    """
    Create a batch of CCZ_φ^{zzz} gates (phase on |000⟩).

    Parameters
    ----------
    phi_tensor : torch.Tensor
        Tensor of phase angles, shape [n] or [n, 1]
    device : str, optional
        Device to place tensors on

    Returns
    -------
    torch.Tensor
        Stacked gates, shape [n, 8, 8]
    """
    device = device or DEVICE

    if len(phi_tensor.shape) > 1:
        phi_tensor = phi_tensor.squeeze()

    batch_size = len(phi_tensor)
    gates = []

    for phi in phi_tensor:
        gate = cczphi_gate_zzz(phi, device)
        gates.append(gate)

    return backend.stack(gates)


# =============================================================================
# Single-Qubit Corrections
# =============================================================================


def single_qubit_phase_correction(phi, nqubits=2, device=None):
    """
    Single-qubit Z-rotation correction for multi-qubit gates.

    Applies Z-rotations to all qubits to correct unwanted single-qubit phases
    that accumulate during gate implementation.

    Parameters
    ----------
    phi : float or torch.Tensor
        Phase correction angle
    nqubits : int, optional
        Number of qubits (default: 2)
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        Correction unitary, shape [2^nqubits, 2^nqubits]

    Notes
    -----
    For a 2-qubit gate, this creates:
    diag(1, e^{-iφ}, e^{-iφ}, e^{-2iφ})

    This corrects phases like those from AC Stark shifts or global rotations.
    """
    device = device or DEVICE
    dim = 2**nqubits

    correction = backend.eye(dim, dtype=DTYPE_COMPLEX, device=device)
    if not hasattr(phi, 'shape'):
        phi = backend.tensor(phi, device=device)
    phase_factor = backend.exp(backend.tensor(-1.0j, device=device) * phi)

    # Apply phase correction based on number of |1⟩ states
    for idx in range(dim):
        # Count number of 1's in binary representation
        n_ones = bin(idx).count("1")
        correction[idx, idx] = phase_factor**n_ones

    return correction


def single_qubit_phase_correction_batch(phi_batch, nqubits=2, device=None):
    """
    Batched version of single_qubit_phase_correction.

    Parameters
    ----------
    phi_batch : torch.Tensor
        Batch of phase angles, shape [batch_size]
    nqubits : int, optional
        Number of qubits
    device : str, optional
        Device to place tensors on

    Returns
    -------
    torch.Tensor
        Batch of correction unitaries, shape [batch_size, 2^nqubits, 2^nqubits]
    """
    device = device or DEVICE
    batch_size = len(phi_batch)
    dim = 2**nqubits

    corrections = backend.zeros(
        (batch_size, dim, dim), dtype=DTYPE_COMPLEX, device=device
    )

    for i, phi in enumerate(phi_batch):
        corrections[i] = single_qubit_phase_correction(phi, nqubits, device)

    return corrections
