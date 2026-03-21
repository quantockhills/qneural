"""
Hardware-specific quantum system implementations.

This module contains platform-specific physics:
    - Hamiltonians for different quantum hardware (Rydberg atoms, ions, superconducting qubits, etc.)
    - Physical constants and parameters
    - Control operators and coupling terms
    - Noise models and decoherence channels

Each hardware platform is implemented as a submodule with its own:
    - Hamiltonian class
    - Operators and states
    - Physical constants
    - Default control schemes
"""

from . import rydberg

__all__ = ['rydberg']
