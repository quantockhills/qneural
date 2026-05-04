"""
Quantum state manipulation (hardware-agnostic).

Functions for creating and manipulating quantum states in arbitrary bases.
"""

import numpy as np
import functools as ft
from ..backend import backend
from ..config import DTYPE_COMPLEX, DEVICE


def number_to_base(n, base):
    """
    Convert a number to a given base representation.

    For quantum states, converts computational basis index to qudit representation.
    For example, with base=3 (qutrits): 5 → '12', meaning |1⟩⊗|2⟩.

    Parameters
    ----------
    n : int
        Number to convert
    base : int
        Target base (e.g., 2 for qubits, 3 for qutrits)

    Returns
    -------
    str
        String representation in the given base, with '2' replaced by 'r' for Rydberg states
    """
    if n == 0:
        return "0"

    digits = []
    while n:
        digits.append(int(n % base))
        n //= base

    # Build string, replacing 2 with 'r' for Rydberg notation
    str_rep = ""
    for i in digits[::-1]:
        if i == 2:
            str_rep += "r"
        else:
            str_rep += str(i)

    return str_rep


def basis_tensor(state_str, dim=3, device=None):
    """
    Create a basis state tensor from a string representation.

    Parameters
    ----------
    state_str : str
        State string, e.g., '001', '01r', 'rr0', etc.
        Convention: states ordered from left to right
        Special character 'r' represents Rydberg state (index 2)
    dim : int, optional
        Local Hilbert space dimension (default: 3 for qutrits)
        Use 2 for qubits, 3 for qutrits/GG-qubits
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        Basis state as column vector, shape [dim^n, 1] for n qudits

    Examples
    --------
    >>> basis_tensor('0', dim=3)  # |0⟩
    >>> basis_tensor('01', dim=3)  # |0⟩⊗|1⟩
    >>> basis_tensor('r1', dim=3)  # |r⟩⊗|1⟩ (equivalent to |2⟩⊗|1⟩)
    """
    device = device or DEVICE
    n_qudits = len(state_str)
    hilbert_dim = dim**n_qudits

    # Create zero state
    state = backend.zeros((hilbert_dim, 1), dtype=DTYPE_COMPLEX, device=device)

    state_numeric = state_str.replace("r", "2")
    index = int(state_numeric, dim)
    state = backend.index_set(state, (index, 0), 1.0)
    return state


def tensor_product(tensor_list):
    """
    Compute the tensor product of a list of tensors.

    Equivalent to qt.tensor for QuTiP, but using PyTorch.

    Parameters
    ----------
    tensor_list : list of torch.Tensor
        List of tensors to be tensored together

    Returns
    -------
    torch.Tensor
        Tensor product of all input tensors

    Examples
    --------
    >>> ket_0 = basis_tensor('0', dim=3)
    >>> ket_1 = basis_tensor('1', dim=3)
    >>> ket_01 = tensor_product([ket_0, ket_1])  # |0⟩⊗|1⟩
    """
    return ft.reduce(backend.kron, tensor_list)


def basis_states_output(wavefunction):
    """
    Convert a wavefunction to a readable output format.

    Useful for inspecting quantum states and understanding which basis
    states have significant amplitude.

    Parameters
    ----------
    wavefunction : torch.Tensor
        Quantum state vector (shape: [dim, 1] or [dim])

    Returns
    -------
    dict
        Dictionary with keys:
        - 'main': List of [amplitude, basis_string] for amplitudes > 0.01
        - 'minor': List of [amplitude, basis_string] for smaller amplitudes

    Examples
    --------
    >>> state = (basis_tensor('00') + basis_tensor('11')) / np.sqrt(2)
    >>> output = basis_states_output(state)
    >>> print(output['main'])
    [[0.707..., '00'], [0.707..., '11']]
    """
    # Ensure wavefunction is 1D
    if wavefunction.dim() > 1:
        wavefunction = wavefunction.squeeze()

    # Determine number of qudits
    total_dim = len(wavefunction)
    n_qudits = int(np.emath.logn(3, total_dim))  # Assumes dim=3

    main_list = {"main": [], "minor": []}

    for idx, amplitude in enumerate(wavefunction):
        amplitude_val = amplitude.item()

        # Convert index to basis string
        basis_str = number_to_base(idx, 3)

        # Pad with zeros on the right if necessary
        while len(basis_str) < int(n_qudits):
            basis_str += "0"

        entry = [amplitude_val, basis_str]

        # Categorize by amplitude magnitude
        if abs(amplitude_val) > 0.01:
            main_list["main"].append(entry)
        else:
            main_list["minor"].append(entry)

    return main_list


def reduce_to_computational_basis(unitary, excluded_state="2", dim=3):
    """
    Reduce a unitary matrix by excluding states containing a specific level.

    Useful for extracting the computational subspace (|0⟩, |1⟩) from a larger
    Hilbert space that includes auxiliary states (e.g., Rydberg state |r⟩ = |2⟩).

    Parameters
    ----------
    unitary : torch.Tensor
        Unitary matrix in the full Hilbert space
    excluded_state : str, optional
        State to exclude (default: '2' for Rydberg)
    dim : int, optional
        Local Hilbert space dimension (default: 3)

    Returns
    -------
    torch.Tensor
        Reduced unitary matrix (generally non-unitary due to leakage)

    Notes
    -----
    The resulting matrix is typically *not* unitary because population may leak
    to excluded states. Use this to analyze gate fidelity in the computational basis.
    """
    n_qudits = int(np.log(unitary.shape[0]) / np.log(dim))

    # Find indices to keep (those not containing excluded_state)
    indices_to_keep = []
    for idx in range(unitary.shape[0]):
        basis_str = np.base_repr(idx, dim)
        if excluded_state not in basis_str:
            indices_to_keep.append(idx)

    # Extract submatrix
    reduced = unitary[indices_to_keep, :][:, indices_to_keep]

    return reduced
