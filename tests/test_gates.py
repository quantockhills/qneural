"""
Unit tests for generalized gate implementations.

Tests the controlled-phase gate framework:
    - General ControlledPhaseGate base class
    - CZ_φ (2-qubit) specialization
    - CCZ_φ (3-qubit) specialization
    - Phase corrections
    - Subspace reduction
    - Optimizer integration
"""

import pytest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qneural.gates import (
    CZPhiGate,
    CCZPhiGate,
    ControlledPhaseOptimizer,
    create_czphi_optimizer,
    create_cczphi_optimizer,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def czphi_gate():
    """CZ_φ gate instance."""
    return CZPhiGate()


@pytest.fixture
def cczphi_gate():
    """CCZ_φ gate instance."""
    return CCZPhiGate()


# =============================================================================
# Test CZ_φ Gate (2-qubit)
# =============================================================================


class TestCZPhiGate:
    """Tests for CZ_φ gate (2-qubit controlled-phase)."""

    def test_initialization(self, czphi_gate):
        """Gate should initialize with correct properties."""
        # Assert
        assert czphi_gate.n_controls == 1
        assert czphi_gate.n_targets == 1
        assert czphi_gate.total_qubits == 2
        assert czphi_gate.full_dim == 9  # 3^2
        assert czphi_gate.comp_dim == 4  # 2^2

    def test_target_unitary_shape(self, czphi_gate):
        """Target unitary should have correct shape."""
        # Act
        target = czphi_gate.get_target_unitary(torch.pi / 2)

        # Assert
        assert target.shape == (4, 4)

    def test_target_unitary_structure(self, czphi_gate):
        """Target should be diagonal with phase on |11⟩."""
        # Arrange
        phi = torch.pi / 3

        # Act
        target = czphi_gate.get_target_unitary(phi)

        # Assert
        # Should be diagonal
        off_diag = target - torch.diag(torch.diag(target))
        assert torch.allclose(off_diag, torch.zeros_like(off_diag))

        # Diagonal should be [1, 1, 1, e^{iφ}]
        diag = torch.diag(target)
        assert torch.allclose(diag[0], torch.tensor(1.0 + 0.0j))
        assert torch.allclose(diag[1], torch.tensor(1.0 + 0.0j))
        assert torch.allclose(diag[2], torch.tensor(1.0 + 0.0j))
        expected_phase = torch.exp(torch.tensor(1.0j) * phi)
        assert torch.allclose(diag[3], expected_phase)

    def test_cz_at_pi(self, czphi_gate):
        """CZ_φ at φ=π should be standard CZ gate."""
        # Act
        target = czphi_gate.get_target_unitary(torch.pi)

        # Assert
        expected = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
            dtype=torch.cfloat,
        )
        assert torch.allclose(target, expected)

    def test_identity_at_zero(self, czphi_gate):
        """CZ_φ at φ=0 should be identity."""
        # Act
        target = czphi_gate.get_target_unitary(0.0)

        # Assert
        expected = torch.eye(4, dtype=torch.cfloat)
        assert torch.allclose(target, expected)


# =============================================================================
# Test CCZ_φ Gate (3-qubit)
# =============================================================================


class TestCCZPhiGate:
    """Tests for CCZ_φ gate (3-qubit controlled-phase)."""

    def test_initialization(self, cczphi_gate):
        """Gate should initialize with correct properties."""
        # Assert
        assert cczphi_gate.n_controls == 2
        assert cczphi_gate.n_targets == 1
        assert cczphi_gate.total_qubits == 3
        assert cczphi_gate.full_dim == 27  # 3^3
        assert cczphi_gate.comp_dim == 8  # 2^3

    def test_target_unitary_shape(self, cczphi_gate):
        """Target unitary should have correct shape."""
        # Act
        target = cczphi_gate.get_target_unitary(torch.pi / 2)

        # Assert
        assert target.shape == (8, 8)

    def test_target_unitary_structure(self, cczphi_gate):
        """Target should apply phase only to |111⟩ state."""
        # Arrange
        phi = torch.pi / 4

        # Act
        target = cczphi_gate.get_target_unitary(phi)

        # Assert
        # Should be diagonal
        off_diag = target - torch.diag(torch.diag(target))
        assert torch.allclose(off_diag, torch.zeros_like(off_diag))

        # All diagonal elements should be 1 except last (|111⟩)
        diag = torch.diag(target)
        for i in range(7):
            assert torch.allclose(diag[i], torch.tensor(1.0 + 0.0j))

        # Last element should be e^{iφ}
        expected_phase = torch.exp(torch.tensor(1.0j) * phi)
        assert torch.allclose(diag[7], expected_phase)

    def test_ccz_at_pi(self, cczphi_gate):
        """CCZ_φ at φ=π should apply -1 to |111⟩."""
        # Act
        target = cczphi_gate.get_target_unitary(torch.pi)

        # Assert
        diag = torch.diag(target)
        assert torch.allclose(diag[7], torch.tensor(-1.0 + 0.0j))


# =============================================================================
# Test Subspace Reduction
# =============================================================================


class TestSubspaceReduction:
    """Tests for reducing 3^N to 2^N Hilbert space."""

    def test_computational_indices_2q(self):
        """Should identify correct computational basis indices for 2 qubits."""
        # Arrange
        gate = CZPhiGate()

        # Act
        indices = gate._get_computational_indices()

        # Assert
        # For 2 GG-qubits: states are |00⟩, |01⟩, |0r⟩, |10⟩, |11⟩, |1r⟩, |r0⟩, |r1⟩, |rr⟩
        # We want indices for |00⟩=0, |01⟩=1, |10⟩=3, |11⟩=4
        expected = [0, 1, 3, 4]
        assert indices == expected

    def test_computational_indices_3q(self):
        """Should identify correct computational basis indices for 3 qubits."""
        # Arrange
        gate = CCZPhiGate()

        # Act
        indices = gate._get_computational_indices()

        # Assert
        # Should be 8 indices (no 2s in base-3 representation)
        assert len(indices) == 8
        # |000⟩ = 0, |001⟩ = 1, |010⟩ = 3, etc.
        assert 0 in indices  # |000⟩
        assert 1 in indices  # |001⟩
        assert 13 in indices  # |111⟩ (1*9 + 1*3 + 1 = 13)
        assert 2 not in indices  # |00r⟩ should not be included

    def test_reduce_unitary_2q(self):
        """Should correctly reduce 9x9 to 4x4 for 2 qubits."""
        # Arrange
        gate = CZPhiGate()
        full_unitary = torch.eye(9, dtype=torch.cfloat)

        # Act
        reduced = gate.reduce_to_computational_basis(full_unitary)

        # Assert
        assert reduced.shape == (4, 4)
        assert torch.allclose(reduced, torch.eye(4, dtype=torch.cfloat))


# =============================================================================
# Test Phase Corrections
# =============================================================================


class TestPhaseCorrections:
    """Tests for single-qubit phase corrections."""

    def test_correction_removes_diagonal_phases(self):
        """Should apply symmetric phase corrections."""
        # Arrange
        gate = CZPhiGate()
        # Create a diagonal unitary (like a CZ_phi gate would be)
        phi = torch.pi / 3
        target = gate.get_target_unitary(phi)

        # Add spurious single-qubit phases
        local_phases = torch.tensor(
            [0.2, 0.3, 0.3, 0.6]
        )  # Note: |01⟩ and |10⟩ same, |11⟩ is 2×
        unitary = torch.diag(torch.exp(1.0j * local_phases)) @ target

        # Act
        corrected = gate.apply_phase_corrections(unitary)

        # Assert - should match target up to global phase
        # (symmetric correction removes single-qubit phases)
        fidelity = gate.compute_fidelity(corrected, target)
        assert fidelity > 0.9999  # Very high fidelity after correction

    def test_correction_preserves_structure(self):
        """Correction should preserve unitarity."""
        # Arrange
        gate = CZPhiGate()
        unitary = gate.get_target_unitary(torch.pi / 3)

        # Act
        corrected = gate.apply_phase_corrections(unitary)

        # Assert - should still be unitary
        identity = corrected @ corrected.conj().T
        assert torch.allclose(identity, torch.eye(4, dtype=torch.cfloat), atol=1e-5)

    def test_correction_matrix_computation(self):
        """Should compute correct correction matrix."""
        # Arrange
        gate = CZPhiGate()
        unitary = gate.get_target_unitary(torch.pi / 4)
        # Add some spurious phases
        phases = torch.tensor([0.2, 0.3, 0.4, 0.5])
        unitary = torch.diag(torch.exp(1.0j * phases)) @ unitary

        # Act
        correction = gate.compute_phase_correction_matrix(unitary)

        # Assert
        # Should be diagonal
        off_diag = correction - torch.diag(torch.diag(correction))
        assert torch.allclose(off_diag, torch.zeros_like(off_diag))

        # First element should be 1
        assert torch.allclose(correction[0, 0], torch.tensor(1.0 + 0.0j))


# =============================================================================
# Test Infidelity Computation
# =============================================================================


class TestInfidelityComputation:
    """Tests for gate infidelity calculations."""

    def test_identical_unitaries_zero_infidelity(self):
        """Identical unitaries should have zero infidelity."""
        # Arrange
        gate = CZPhiGate()
        target = gate.get_target_unitary(torch.pi / 2)

        # Act
        infidelity = gate.compute_infidelity(target, target)

        # Assert
        assert torch.allclose(infidelity, torch.tensor(0.0), atol=1e-6)

    def test_different_unitaries_nonzero_infidelity(self):
        """Different unitaries should have nonzero infidelity."""
        # Arrange
        gate = CZPhiGate()
        target1 = gate.get_target_unitary(torch.pi / 2)
        target2 = gate.get_target_unitary(torch.pi / 3)

        # Act
        infidelity = gate.compute_infidelity(target1, target2)

        # Assert
        assert infidelity > 0


# =============================================================================
# Test Optimizer Factory Functions
# =============================================================================


class TestOptimizerFactories:
    """Tests for optimizer factory functions."""

    def test_create_czphi_optimizer(self):
        """Should create CZ_φ optimizer."""
        # Act
        optimizer = create_czphi_optimizer()

        # Assert
        assert isinstance(optimizer, ControlledPhaseOptimizer)
        assert optimizer.gate.total_qubits == 2
        assert not optimizer.time_optimal

    def test_create_czphi_optimizer_time_optimal(self):
        """Should create time-optimal CZ_φ optimizer."""
        # Act
        optimizer = create_czphi_optimizer(time_optimal=True, time_bounds=(3.0, 8.0))

        # Assert
        assert isinstance(optimizer, ControlledPhaseOptimizer)
        assert optimizer.time_optimal

    def test_create_cczphi_optimizer(self):
        """Should create CCZ_φ optimizer."""
        # Act
        optimizer = create_cczphi_optimizer()

        # Assert
        assert isinstance(optimizer, ControlledPhaseOptimizer)
        assert optimizer.gate.total_qubits == 3


# =============================================================================
# Integration Tests
# =============================================================================


class TestGateIntegration:
    """Integration tests for full gate pipeline."""

    def test_czphi_evaluate(self):
        """Should evaluate CZ_φ gate fidelity."""
        # Arrange - create optimizer but don't train
        optimizer = create_czphi_optimizer()

        # Act - evaluate with random initialization
        result = optimizer.evaluate(torch.pi / 2)

        # Assert
        assert "angle" in result
        assert "gate_time" in result
        assert "infidelity" in result
        assert "achieved_unitary" in result
        assert "target_unitary" in result
        assert result["angle"] == pytest.approx(torch.pi / 2, abs=0.01)
        assert result["infidelity"] >= 0  # Should be valid infidelity

    def test_full_pipeline_shapes(self):
        """Full pipeline should maintain correct tensor shapes."""
        # Arrange
        optimizer = create_czphi_optimizer()
        angle = torch.pi / 4

        # Act
        pulses, gate_time = optimizer.generate_pulse(angle)
        final_U = optimizer.evolver.evolve(pulses, gate_time)
        corrected = optimizer.gate.apply_phase_corrections(final_U)
        target = optimizer.gate.get_target_unitary(angle)

        # Assert
        assert len(pulses) == 2  # rabi and detuning
        assert isinstance(gate_time, float)
        # Evolver already returns computational subspace (4x4 for 2 qubits)
        assert final_U.shape == (4, 4)  # Computational 2^2 space
        assert corrected.shape == (4, 4)
        assert target.shape == (4, 4)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
