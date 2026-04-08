"""
Rydberg atom quantum computing platform.

This module implements the physics of neutral atom quantum computing using Rydberg states.

Key components:
    - RydbergHamiltonian: Time-dependent Hamiltonian for Rydberg atom arrays
    - Operators: Single-qubit and two-qubit operators for Rydberg systems
    - Constants: Physical parameters (Rabi frequencies, interaction strengths, etc.)

Physics background:
    Rydberg atoms can be used to implement quantum gates through blockade mechanisms.
    Atoms are trapped in optical tweezers and excited to high-lying Rydberg states,
    where they experience strong van der Waals interactions. This enables entangling
    gates between qubits.

Two common qubit encodings:
    1. Ground-Rydberg (gr): |0⟩ = ground state, |1⟩ = Rydberg state
    2. Ground-Ground (gg): |0⟩ and |1⟩ are hyperfine ground states, with Rydberg as auxiliary

The Hamiltonian includes:
    - Rabi coupling between qubit states and Rydberg state
    - Detuning from resonance
    - Van der Waals interactions between Rydberg atoms
    - Optional decay from Rydberg state
"""

from .constants import *
from .operators import *
from .hamiltonian import RydbergHamiltonian, create_constant_hamiltonian
from .pulses import (
    zero_pulse,
    constant_pulse,
    piecewise_constant,
    piecewise_constant_nn_output,
    create_simple_detuning_pulse,
    gaussian_pulse,
    blackman_pulse,
    cutoff_pulse,
    add_pulses,
    pulse_area,
    normalize_pulse,
)

__all__ = [
    # Constants
    "RABI_DEFAULT",
    "RABI_GG_DEFAULT",
    "VDD_COUPLING",
    "DECAY_WIDTH_DEFAULT",
    # Operators
    "create_rydberg_operators",
    # Hamiltonian
    "RydbergHamiltonian",
    "create_constant_hamiltonian",
    # Pulses
    "zero_pulse",
    "constant_pulse",
    "piecewise_constant",
    "piecewise_constant_nn_output",
    "create_simple_detuning_pulse",
    "gaussian_pulse",
    "blackman_pulse",
    "cutoff_pulse",
    "add_pulses",
    "pulse_area",
    "normalize_pulse",
]
