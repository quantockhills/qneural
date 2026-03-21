"""
Quantum gate fidelity and performance metrics.

Functions for evaluating the quality of quantum gates:
    - Process fidelity
    - Average gate fidelity
    - Infidelity (1 - fidelity)
    - Distance measures
"""

import torch
import numpy as np
from ..backend import backend
from ..config import DEVICE


def unitary_fidelity(U1, U2, dim=2, nqubits=1):
    """
    Compute average gate fidelity between two unitary operations.

    Uses the definition from Phys. Rev. Lett. 129, 050507 (2022):
    F = |⟨U1|U2⟩|² / d² where ⟨U1|U2⟩ = Tr(U1† U2)

    Parameters
    ----------
    U1 : torch.Tensor
        First unitary, shape [d, d] or [batch, d, d]
    U2 : torch.Tensor
        Second unitary (target), shape [d, d] or [batch, d, d]
    dim : int, optional
        Local Hilbert space dimension (default: 2)
    nqubits : int, optional
        Number of qubits (default: 1)

    Returns
    -------
    torch.Tensor (scalar)
        Gate fidelity between 0 and 1

    References
    ----------
    Pedersen et al., Phys. Rev. Lett. 129, 050507 (2022)
    """
    if U1.dim() == 2:
        # Single pair
        hilbert_dim = dim ** nqubits
        overlap = torch.trace(torch.matmul(U1.conj().T, U2))
        fidelity = torch.abs(overlap) ** 2 / (hilbert_dim ** 2)
    else:
        # Batch - use original implementation
        hilbert_dim = dim ** nqubits
        c = torch.einsum('mij, nji -> mn', U1.mH, U2)
        g = torch.einsum('mm ->', c ** 2)
        fidelity = torch.abs(g) / (U1.shape[0] * hilbert_dim ** 2)

    return fidelity


def unitary_fidelity_batch(U1_batch, U2_batch, nqubits=1):
    """
    Compute fidelity for a batch of unitary pairs.

    Uses the same formula as unitary_fidelity:
    F = |Tr(U1† U2)|² / d²

    Parameters
    ----------
    U1_batch : torch.Tensor
        Batch of unitaries, shape [batch_size, d, d]
    U2_batch : torch.Tensor
        Batch of target unitaries, shape [batch_size, d, d]
    nqubits : int, optional
        Number of qubits

    Returns
    -------
    torch.Tensor
        Fidelities for each pair, shape [batch_size]
    """
    d = 2 ** nqubits

    # Batch matrix multiplication: U1†  U2
    # U1.conj().transpose(1, 2) gives batch of U1†
    overlap_matrices = torch.bmm(U1_batch.conj().transpose(1, 2), U2_batch)

    # Trace of each matrix: einsum for batch trace
    overlaps = torch.einsum('bii->b', overlap_matrices)

    # Fidelity for each: |Tr(U1† U2)|² / d²
    fidelities = torch.abs(overlaps) ** 2 / (d ** 2)

    return fidelities


def unitary_infidelity(U1, U2, dim=2, nqubits=1):
    """
    Compute infidelity: 1 - F.

    This is the primary metric used in optimization, as it should be minimized.

    Parameters
    ----------
    U1 : torch.Tensor
        First unitary, shape [d, d] or [batch, d, d]
    U2 : torch.Tensor
        Second unitary (target), shape [d, d] or [batch, d, d]
    dim : int, optional
        Local Hilbert space dimension (default: 2)
    nqubits : int, optional
        Number of qubits

    Returns
    -------
    torch.Tensor (scalar)
        Infidelity between 0 and 1
    """
    return 1.0 - unitary_fidelity(U1, U2, dim, nqubits)


def unitary_infidelity_batch(U1_batch, U2_batch, nqubits=1):
    """
    Compute infidelity for a batch of unitary pairs.

    This is the vectorized version used in neural network training.

    Parameters
    ----------
    U1_batch : torch.Tensor
        Batch of unitaries, shape [batch_size, d, d]
    U2_batch : torch.Tensor
        Batch of target unitaries, shape [batch_size, d, d]
    nqubits : int, optional
        Number of qubits

    Returns
    -------
    torch.Tensor
        Infidelities for each pair, shape [batch_size]

    Examples
    --------
    >>> # Training loop usage
    >>> target_gates = czp_gate_stack(angles)  # [batch, 4, 4]
    >>> achieved_gates = simulate_pulse(pulse)  # [batch, 4, 4]
    >>> infidelities = unitary_infidelity_batch(achieved_gates, target_gates, nqubits=2)
    >>> loss = infidelities.mean()
    """
    return 1.0 - unitary_fidelity_batch(U1_batch, U2_batch, nqubits)


def unitary_infidelity_array(U1, U2, dim=2, nqbits=1):
    """
    Legacy interface for infidelity computation.

    Compatible with original cst_n_fn.py naming.

    Parameters
    ----------
    U1 : torch.Tensor
        First unitary or batch, shape [n, n] or [batch, n, n]
    U2 : torch.Tensor
        Second unitary or batch, shape [n, n] or [batch, n, n]
    dim : int, optional
        Local Hilbert space dimension (not used, kept for compatibility)
    nqbits : int, optional
        Number of qubits

    Returns
    -------
    torch.Tensor
        Infidelity (scalar if single pair, vector if batch)
    """
    if U1.dim() == 2:
        # Single unitary pair
        return unitary_infidelity(U1, U2, nqbits)
    else:
        # Batch
        return unitary_infidelity_batch(U1, U2, nqbits)


# =============================================================================
# Process Fidelity (Alternative Definition)
# =============================================================================

def process_fidelity(U1, U2):
    """
    Compute process fidelity: |⟨U1|U2⟩|² / d².

    This is an alternative fidelity measure that treats unitaries as vectors
    in Hilbert-Schmidt space.

    Parameters
    ----------
    U1 : torch.Tensor
        First unitary
    U2 : torch.Tensor
        Second unitary

    Returns
    -------
    torch.Tensor (scalar)
        Process fidelity

    Notes
    -----
    Related to average gate fidelity but simpler. Not used as frequently in
    the literature but useful for some applications.
    """
    d = U1.shape[0]
    overlap = torch.trace(torch.matmul(U1.conj().T, U2))
    return torch.abs(overlap) ** 2 / d ** 2


# =============================================================================
# Diamond Distance (Advanced)
# =============================================================================

def diamond_distance_estimate(U1, U2):
    """
    Estimate diamond norm distance between two unitaries.

    The diamond norm is the most operationally relevant distance measure
    for quantum channels, but it's expensive to compute exactly.

    This provides a rough estimate based on the Frobenius norm.

    Parameters
    ----------
    U1, U2 : torch.Tensor
        Unitary matrices

    Returns
    -------
    torch.Tensor (scalar)
        Estimated diamond distance

    Notes
    -----
    For true diamond norm, use specialized libraries like cvxpy or qiskit.
    This is a quick approximation for monitoring during optimization.
    """
    difference = U1 - U2
    frobenius_norm = torch.norm(difference, p='fro')
    return frobenius_norm / np.sqrt(2)  # Rough scaling


# =============================================================================
# Utilities
# =============================================================================

def gate_error_rate(fidelity):
    """
    Convert fidelity to gate error rate.

    Parameters
    ----------
    fidelity : torch.Tensor or float
        Gate fidelity (0 to 1)

    Returns
    -------
    torch.Tensor or float
        Error rate (0 to 1)

    Notes
    -----
    Error rate = 1 - F, the complement of fidelity.
    For small errors, this approximates the depolarizing error rate.
    """
    return 1.0 - fidelity


def fidelity_to_gate_error(fidelity, nqubits):
    """
    Convert average gate fidelity to gate error.

    Uses the relation for depolarizing noise:
    ε = (d - 1) / d * (1 - F)
    where d = 2^nqubits.

    Parameters
    ----------
    fidelity : torch.Tensor or float
        Average gate fidelity
    nqubits : int
        Number of qubits

    Returns
    -------
    torch.Tensor or float
        Gate error rate

    Notes
    -----
    This is the standard conversion used in quantum computing benchmarks.
    """
    d = 2 ** nqubits
    return (d - 1) / d * (1.0 - fidelity)
