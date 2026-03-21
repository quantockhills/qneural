"""
Core quantum operations (hardware-agnostic).

This module provides platform-independent quantum operations:
    - Quantum states: basis states, tensor products, state manipulation
    - Quantum gates: gate construction, composition, decomposition
    - Operators: Pauli matrices, general operators
    - Metrics: fidelity, infidelity, distance measures

These operations work with any quantum hardware platform and computational backend.
"""

from .states import (
    number_to_base,
    basis_tensor,
    tensor_product,
    basis_states_output,
    reduce_to_computational_basis,
)

from .gates import (
    cz_gate,
    czphi_gate,
    czp_gate_stack,
    cczphi_gate,
    cczphi_gate_zzz,
    cczp_gate_stack,
    cczp_gate_stack_zzz,
    single_qubit_phase_correction,
    single_qubit_phase_correction_batch,
)

from .operators import (
    pauli_matrices,
    SIGMA_X,
    SIGMA_Y,
    SIGMA_Z,
    IDENTITY_2,
    rotation_x,
    rotation_y,
    rotation_z,
    arbitrary_rotation,
    projector,
)

from .metrics import (
    unitary_fidelity,
    unitary_fidelity_batch,
    unitary_infidelity,
    unitary_infidelity_batch,
    unitary_infidelity_array,
    process_fidelity,
    diamond_distance_estimate,
    gate_error_rate,
    fidelity_to_gate_error,
)

from .evolution import (
    schrodinger_evolution,
    evolve_unitary,
    time_evolution_operator,
    evolve_state,
    mesolve,
)

__all__ = [
    # States
    'number_to_base',
    'basis_tensor',
    'tensor_product',
    'basis_states_output',
    'reduce_to_computational_basis',
    # Gates
    'cz_gate',
    'czphi_gate',
    'czp_gate_stack',
    'cczphi_gate',
    'cczphi_gate_zzz',
    'cczp_gate_stack',
    'cczp_gate_stack_zzz',
    'single_qubit_phase_correction',
    'single_qubit_phase_correction_batch',
    # Operators
    'pauli_matrices',
    'SIGMA_X',
    'SIGMA_Y',
    'SIGMA_Z',
    'IDENTITY_2',
    'rotation_x',
    'rotation_y',
    'rotation_z',
    'arbitrary_rotation',
    'projector',
    # Metrics
    'unitary_fidelity',
    'unitary_fidelity_batch',
    'unitary_infidelity',
    'unitary_infidelity_batch',
    'unitary_infidelity_array',
    'process_fidelity',
    'diamond_distance_estimate',
    'gate_error_rate',
    'fidelity_to_gate_error',
    # Evolution
    'schrodinger_evolution',
    'evolve_unitary',
    'time_evolution_operator',
    'evolve_state',
    'mesolve',
]
