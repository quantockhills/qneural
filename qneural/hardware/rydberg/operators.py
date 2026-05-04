"""
Operators for Rydberg atom systems.

Defines the fundamental operators for Rydberg quantum computing:
    - Transition operators (|0⟩⟨1|, |1⟩⟨r|, etc.)
    - Projection operators (|r⟩⟨r|, etc.)
    - Pauli-like operators for Rydberg transitions
"""

from ...backend import backend
from ...config import DTYPE_COMPLEX, DEVICE
from .constants import HILBERT_DIM_GG


def create_basis_ket(state_index, dim=HILBERT_DIM_GG, device=None):
    """
    Create a basis state ket |i⟩.

    Parameters
    ----------
    state_index : int
        Index of the basis state (0, 1, or 2 for GG-qubits)
    dim : int, optional
        Local Hilbert space dimension (default: 3 for GG-qubits)
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        Column vector representing |i⟩ (shape: [dim, 1])
    """
    device = device or DEVICE
    ket = backend.zeros((dim, 1), dtype=DTYPE_COMPLEX, device=device)
    ket = backend.index_set(ket, (state_index, 0), 1.0)
    return ket


def create_basis_bra(state_index, dim=HILBERT_DIM_GG, device=None):
    """
    Create a basis state bra ⟨i|.

    Parameters
    ----------
    state_index : int
        Index of the basis state
    dim : int, optional
        Local Hilbert space dimension
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        Row vector representing ⟨i| (shape: [1, dim])
    """
    return create_basis_ket(state_index, dim, device).conj().T


def create_transition_operator(from_state, to_state, dim=HILBERT_DIM_GG, device=None):
    """
    Create a transition operator |to⟩⟨from|.

    Parameters
    ----------
    from_state : int
        Initial state index
    to_state : int
        Final state index
    dim : int, optional
        Local Hilbert space dimension
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        Operator matrix (shape: [dim, dim])
    """
    device = device or DEVICE
    operator = backend.zeros((dim, dim), dtype=DTYPE_COMPLEX, device=device)
    operator = backend.index_set(operator, (to_state, from_state), 1.0)
    return operator


def create_projection_operator(state_index, dim=HILBERT_DIM_GG, device=None):
    """
    Create a projection operator |i⟩⟨i|.

    Parameters
    ----------
    state_index : int
        State to project onto
    dim : int, optional
        Local Hilbert space dimension
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        Projection matrix (shape: [dim, dim])
    """
    return create_transition_operator(state_index, state_index, dim, device)


# =============================================================================
# Standard Rydberg Operators (GG-qubit encoding)
# =============================================================================


def create_rydberg_operators(dim=HILBERT_DIM_GG, device=None):
    """
    Create standard operators for Rydberg systems.

    For GG-qubits (dim=3):
        - States: |0⟩, |1⟩, |r⟩  (indices 0, 1, 2)
        - Rabi coupling: σ_{1r} = |1⟩⟨r| + |r⟩⟨1|
        - Detuning: n_r = |r⟩⟨r|
        - Phase: n_1 = |1⟩⟨1|

    Parameters
    ----------
    dim : int, optional
        Local Hilbert space dimension
    device : str, optional
        Device to place tensors on

    Returns
    -------
    dict
        Dictionary containing standard operators:
        - 'rabi': Rabi coupling operator (1-r transition)
        - 'detuning': Rydberg state projection
        - 'n_1': |1⟩ state projection
        - 'n_0': |0⟩ state projection
        - 'interaction': Rydberg-Rydberg interaction operator (for 2+ qubits)
    """
    device = device or DEVICE
    operators = {}

    if dim == 3:  # GG-qubit
        # Transition operators
        ket_r_bra_1 = create_transition_operator(1, 2, dim, device)  # |r⟩⟨1|
        ket_1_bra_r = create_transition_operator(2, 1, dim, device)  # |1⟩⟨r|

        # Rabi coupling: (|1⟩⟨r| + |r⟩⟨1|)
        operators["rabi"] = ket_r_bra_1 + ket_1_bra_r

        # Detuning: |r⟩⟨r|
        operators["detuning"] = create_projection_operator(2, dim, device)

        # Auxiliary projectors
        operators["n_0"] = create_projection_operator(0, dim, device)
        operators["n_1"] = create_projection_operator(1, dim, device)
        operators["n_r"] = operators["detuning"]  # Alias

    elif dim == 2:  # GR-qubit
        # For ground-Rydberg qubits: |0⟩ (ground), |1⟩ (Rydberg)
        ket_1_bra_0 = create_transition_operator(0, 1, dim, device)
        ket_0_bra_1 = create_transition_operator(1, 0, dim, device)

        operators["rabi"] = ket_1_bra_0 + ket_0_bra_1
        operators["detuning"] = create_projection_operator(1, dim, device)
        operators["n_0"] = create_projection_operator(0, dim, device)
        operators["n_1"] = create_projection_operator(1, dim, device)

    else:
        raise ValueError(f"Unsupported Hilbert dimension: {dim}")

    return operators


# =============================================================================
# Hyperfine Transition Operators (for GG-qubits with additional lasers)
# =============================================================================


def create_hyperfine_operators(phase=0.0, dim=HILBERT_DIM_GG, device=None):
    """
    Create operators for hyperfine transitions (|0⟩ ↔ |1⟩).

    Used when additional control is applied between ground states.

    Parameters
    ----------
    phase : float
        Laser phase (in radians)
    dim : int, optional
        Local Hilbert space dimension
    device : str, optional
        Device to place tensors on

    Returns
    -------
    dict
        Dictionary with 'rabi_hf' and 'detuning_hf' operators
    """
    device = device or DEVICE

    if dim != 3:
        raise ValueError("Hyperfine operators only defined for GG-qubits (dim=3)")

    operators = {}

    # Phase-dependent Rabi coupling: e^{iφ}|0⟩⟨1| + e^{-iφ}|1⟩⟨0|
    ket_0_bra_1 = create_transition_operator(1, 0, dim, device)
    ket_1_bra_0 = create_transition_operator(0, 1, dim, device)

    operators["rabi_hf"] = (
        backend.exp(backend.tensor(-1j * phase)) * ket_0_bra_1
        + backend.exp(backend.tensor(1j * phase)) * ket_1_bra_0
    )

    # Detuning on |1⟩ state
    operators["detuning_hf"] = create_projection_operator(1, dim, device)

    return operators
