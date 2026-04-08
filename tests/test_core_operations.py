"""
Unit tests for core quantum operations.

Tests the hardware-agnostic quantum operations in qneural.core:
    - State creation and manipulation
    - Gate construction
    - Operators (Pauli matrices, rotations)
    - Fidelity metrics

Testing methodology: pytest with AAA pattern (Arrange-Act-Assert)
Coverage goal: >90% of core functionality
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qneural.core import (
    # States
    basis_tensor,
    tensor_product,
    number_to_base,
    czphi_gate,
    czp_gate_stack,
    cczphi_gate,
    single_qubit_phase_correction,
    # Operators
    SIGMA_X,
    SIGMA_Y,
    SIGMA_Z,
    rotation_x,
    rotation_z,
    # Metrics
    unitary_fidelity,
    unitary_infidelity,
    unitary_fidelity_batch,
    unitary_infidelity_batch,
)


# =============================================================================
# Fixtures (Reusable Test Data)
# =============================================================================


@pytest.fixture
def single_qubit_states():
    """Standard single-qubit basis states."""
    return {
        "0": basis_tensor("0", dim=3),
        "1": basis_tensor("1", dim=3),
        "r": basis_tensor("r", dim=3),
    }


@pytest.fixture
def test_angles():
    """Common test angles for gates."""
    return torch.tensor([0.0, torch.pi / 4, torch.pi / 2, torch.pi])


# =============================================================================
# Test Basis States
# =============================================================================


class TestBasisStates:
    """Tests for quantum state creation and manipulation."""

    def test_single_qubit_state_shape(self):
        """Test that single-qubit states have correct shape."""
        # Arrange & Act
        ket_0 = basis_tensor("0", dim=3)

        # Assert
        assert ket_0.shape == (3, 1), "Single qutrit should be 3×1 vector"

    def test_single_qubit_state_values(self, single_qubit_states):
        """Test that basis states have amplitude 1 in correct position."""
        # Assert: |0⟩ has amplitude 1 at index 0
        assert single_qubit_states["0"][0, 0] == 1.0
        assert torch.allclose(
            single_qubit_states["0"][1:], torch.zeros(2, 1, dtype=torch.cfloat)
        )

        # Assert: |1⟩ has amplitude 1 at index 1
        assert single_qubit_states["1"][1, 0] == 1.0

        # Assert: |r⟩ has amplitude 1 at index 2
        assert single_qubit_states["r"][2, 0] == 1.0

    def test_two_qubit_state_shape(self):
        """Test two-qubit state dimensions."""
        # Arrange & Act
        ket_00 = basis_tensor("00", dim=3)

        # Assert
        assert ket_00.shape == (9, 1), "Two qutrits should be 9×1 vector"

    @pytest.mark.parametrize(
        "state_str,expected_index",
        [
            ("00", 0),  # |00⟩
            ("01", 1),  # |01⟩
            ("10", 3),  # |10⟩ (base-3: 1*3 + 0 = 3)
            ("11", 4),  # |11⟩ (base-3: 1*3 + 1 = 4)
            ("rr", 8),  # |rr⟩ (base-3: 2*3 + 2 = 8)
        ],
    )
    def test_two_qubit_state_indices(self, state_str, expected_index):
        """Test that two-qubit states have correct computational basis index."""
        # Arrange & Act
        state = basis_tensor(state_str, dim=3)

        # Assert
        assert state[expected_index, 0] == 1.0

    def test_tensor_product_matches_direct_construction(self):
        """Test that tensor product gives same result as direct construction."""
        # Arrange
        ket_0 = basis_tensor("0", dim=3)
        ket_1 = basis_tensor("1", dim=3)

        # Act
        ket_01_from_product = tensor_product([ket_0, ket_1])
        ket_01_direct = basis_tensor("01", dim=3)

        # Assert
        assert torch.allclose(ket_01_from_product, ket_01_direct)

    def test_number_to_base_conversion(self):
        """Test base conversion utility."""
        # Arrange & Act & Assert
        assert number_to_base(0, 3) == "0"
        assert number_to_base(1, 3) == "1"
        assert number_to_base(2, 3) == "r"  # 2 → 'r' for Rydberg
        assert number_to_base(4, 3) == "11"  # 4 in base-3
        assert number_to_base(8, 3) == "rr"  # 8 in base-3


# =============================================================================
# Test Gates
# =============================================================================


class TestGateConstruction:
    """Tests for quantum gate construction."""

    def test_czphi_gate_at_zero_is_identity(self):
        """CZ_φ(0) should be identity."""
        # Arrange
        phi = 0.0

        # Act
        gate = czphi_gate(phi)

        # Assert
        expected = torch.eye(4, dtype=torch.cfloat)
        assert torch.allclose(gate, expected), "CZ_φ(0) should be identity"

    def test_czphi_gate_at_pi_is_cz(self):
        """CZ_φ(π) should be standard CZ gate."""
        # Arrange
        phi = torch.pi

        # Act
        gate = czphi_gate(phi)

        # Assert
        expected = torch.eye(4, dtype=torch.cfloat)
        expected[3, 3] = -1.0  # CZ: diag(1, 1, 1, -1)
        assert torch.allclose(gate, expected), "CZ_φ(π) should be CZ gate"

    def test_czphi_gate_is_diagonal(self):
        """CZ_φ gates should be diagonal."""
        # Arrange
        phi = 0.5

        # Act
        gate = czphi_gate(phi)

        # Assert: Check off-diagonal elements are zero
        off_diagonal = gate - torch.diag(torch.diag(gate))
        assert torch.allclose(off_diagonal, torch.zeros_like(off_diagonal)), (
            "CZ_φ should be diagonal"
        )

    def test_czphi_gate_is_unitary(self):
        """CZ_φ gates should be unitary: U U† = I."""
        # Arrange
        phi = 0.7

        # Act
        gate = czphi_gate(phi)
        product = torch.matmul(gate, gate.conj().T)

        # Assert
        identity = torch.eye(4, dtype=torch.cfloat)
        assert torch.allclose(product, identity), "Gate should be unitary"

    def test_czp_gate_stack_shape(self, test_angles):
        """Test batch gate creation has correct shape."""
        # Act
        gates = czp_gate_stack(test_angles)

        # Assert
        expected_shape = (len(test_angles), 4, 4)
        assert gates.shape == expected_shape

    def test_czp_gate_stack_values(self, test_angles):
        """Test that stacked gates match individual construction."""
        # Arrange & Act
        gates = czp_gate_stack(test_angles)

        # Assert: Check each gate individually
        for i, angle in enumerate(test_angles):
            individual_gate = czphi_gate(angle)
            assert torch.allclose(gates[i], individual_gate)

    def test_cczphi_gate_shape(self):
        """Three-qubit gate should be 8×8."""
        # Act
        gate = cczphi_gate(torch.pi / 3)

        # Assert
        assert gate.shape == (8, 8)

    def test_cczphi_gate_at_zero_is_identity(self):
        """CCZ_φ(0) should be identity."""
        # Act
        gate = cczphi_gate(0.0)

        # Assert
        assert torch.allclose(gate, torch.eye(8, dtype=torch.cfloat))

    def test_single_qubit_phase_correction_shape(self):
        """Phase correction should have correct dimension."""
        # Act
        correction = single_qubit_phase_correction(0.5, nqubits=2)

        # Assert
        assert correction.shape == (4, 4)


# =============================================================================
# Test Operators
# =============================================================================


class TestOperators:
    """Tests for quantum operators."""

    def test_pauli_matrices_are_hermitian(self):
        """Pauli matrices should be Hermitian."""
        # Assert: σ = σ†
        assert torch.allclose(SIGMA_X, SIGMA_X.conj().T)
        assert torch.allclose(SIGMA_Y, SIGMA_Y.conj().T)
        assert torch.allclose(SIGMA_Z, SIGMA_Z.conj().T)

    def test_pauli_matrices_square_to_identity(self):
        """σ² = I for all Pauli matrices."""
        # Arrange
        identity = torch.eye(2, dtype=torch.cfloat)

        # Assert
        assert torch.allclose(torch.matmul(SIGMA_X, SIGMA_X), identity)
        assert torch.allclose(torch.matmul(SIGMA_Y, SIGMA_Y), identity)
        assert torch.allclose(torch.matmul(SIGMA_Z, SIGMA_Z), identity)

    def test_rotation_x_at_zero_is_identity(self):
        """R_x(0) = I."""
        # Act
        rotation = rotation_x(0.0)

        # Assert
        assert torch.allclose(rotation, torch.eye(2, dtype=torch.cfloat))

    def test_rotation_x_at_2pi_is_minus_identity(self):
        """R_x(2π) = -I (accounts for spin-1/2)."""
        # Act
        rotation = rotation_x(2 * torch.pi)

        # Assert
        expected = -torch.eye(2, dtype=torch.cfloat)
        assert torch.allclose(rotation, expected, atol=1e-6)

    def test_rotation_z_is_diagonal(self):
        """R_z should be diagonal."""
        # Act
        rotation = rotation_z(0.7)

        # Assert: Check off-diagonal is zero
        off_diag = rotation - torch.diag(torch.diag(rotation))
        assert torch.allclose(off_diag, torch.zeros_like(off_diag))


# =============================================================================
# Test Metrics
# =============================================================================


class TestFidelityMetrics:
    """Tests for fidelity calculations."""

    def test_fidelity_of_identical_unitaries_is_one(self):
        """F(U, U) = 1."""
        # Arrange
        U = czphi_gate(0.5)

        # Act
        fidelity = unitary_fidelity(U, U, nqubits=2)

        # Assert
        assert torch.allclose(fidelity, torch.tensor(1.0))

    def test_fidelity_is_bounded_zero_to_one(self):
        """Fidelity should always be in [0, 1]."""
        # Arrange
        U1 = czphi_gate(0.1)
        U2 = czphi_gate(2.5)

        # Act
        fidelity = unitary_fidelity(U1, U2, nqubits=2)

        # Assert
        assert 0 <= fidelity <= 1

    def test_infidelity_is_complement_of_fidelity(self):
        """F + (1-F) = 1."""
        # Arrange
        U1 = czphi_gate(0.3)
        U2 = czphi_gate(0.8)

        # Act
        fidelity = unitary_fidelity(U1, U2, nqubits=2)
        infidelity = unitary_infidelity(U1, U2, nqubits=2)

        # Assert
        assert torch.allclose(fidelity + infidelity, torch.tensor(1.0))

    def test_fidelity_is_symmetric(self):
        """F(U1, U2) = F(U2, U1)."""
        # Arrange
        U1 = czphi_gate(0.4)
        U2 = czphi_gate(1.2)

        # Act
        f_12 = unitary_fidelity(U1, U2, nqubits=2)
        f_21 = unitary_fidelity(U2, U1, nqubits=2)

        # Assert
        assert torch.allclose(f_12, f_21)

    def test_fidelity_batch_matches_individual(self, test_angles):
        """Batched fidelity should match individual calculations."""
        # Arrange
        U1_batch = czp_gate_stack(test_angles)
        U2_batch = czp_gate_stack(test_angles + 0.1)

        # Act
        fidelities_batch = unitary_fidelity_batch(U1_batch, U2_batch, nqubits=2)

        # Assert: Check each fidelity individually
        for i in range(len(test_angles)):
            fidelity_individual = unitary_fidelity(U1_batch[i], U2_batch[i], nqubits=2)
            assert torch.allclose(fidelities_batch[i], fidelity_individual)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for combined operations."""

    @pytest.mark.integration
    def test_gate_construction_and_fidelity_pipeline(self):
        """Test complete workflow: construct gates and compute fidelity."""
        # Arrange
        target_angles = torch.linspace(0, torch.pi, 10)
        achieved_angles = target_angles + torch.randn(10) * 0.01  # Small noise

        # Act
        target_gates = czp_gate_stack(target_angles)
        achieved_gates = czp_gate_stack(achieved_angles)
        infidelities = unitary_infidelity_batch(achieved_gates, target_gates, nqubits=2)

        # Assert
        assert infidelities.shape == (10,)
        assert torch.all(infidelities >= 0)
        assert torch.all(infidelities <= 1)
        # With small noise, infidelities should be small
        assert torch.all(infidelities < 0.1)


# =============================================================================
# Property-Based Tests (Mathematical Properties)
# =============================================================================


class TestMathematicalProperties:
    """Tests for mathematical properties that should always hold."""

    @pytest.mark.parametrize("phi", [0.0, 0.5, 1.0, torch.pi / 2, torch.pi])
    def test_czphi_gates_are_unitary(self, phi):
        """All CZ_φ gates should be unitary."""
        # Act
        U = czphi_gate(phi)
        product = torch.matmul(U, U.conj().T)

        # Assert: U U† = I
        assert torch.allclose(product, torch.eye(4, dtype=torch.cfloat), atol=1e-6)

    @pytest.mark.parametrize("nqubits", [1, 2, 3])
    def test_basis_states_are_normalized(self, nqubits):
        """All basis states should have norm 1."""
        # Arrange: Create random basis state
        state_index = np.random.randint(0, 3**nqubits)
        state_str = number_to_base(state_index, 3).zfill(nqubits)

        # Act
        state = basis_tensor(state_str, dim=3)
        norm = torch.norm(state)

        # Assert
        assert torch.allclose(norm, torch.tensor(1.0))


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
