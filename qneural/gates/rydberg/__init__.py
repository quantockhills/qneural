"""
Rydberg atom gate implementations.

Provides optimized pulse sequences for quantum gates on neutral atom systems:
    - Controlled-phase gates (CZ_φ, CCZ_φ, etc.)
    - General N-controlled phase gates
    - Phase correction utilities

All gates are implemented using the generalized framework where specific gates
(CZ_φ, CCZ_φ) are special cases of the N-controlled phase gate.
"""

# General controlled-phase framework
from .controlled_phase import (
    ControlledPhaseGate,
    CZPhiGate,
    CCZPhiGate,
    ControlledPhaseOptimizer,
    create_czphi_optimizer,
    create_cczphi_optimizer,
)

__all__ = [
    # General framework
    "ControlledPhaseGate",
    "ControlledPhaseOptimizer",
    # Specific gates
    "CZPhiGate",
    "CCZPhiGate",
    # Factory functions
    "create_czphi_optimizer",
    "create_cczphi_optimizer",
]
