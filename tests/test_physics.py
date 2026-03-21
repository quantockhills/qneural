"""
Unit tests for physics layer: pulses, Hamiltonians, and time evolution.

Tests the Rydberg-specific physics in qneural.hardware.rydberg and
time evolution in qneural.core.evolution.

Testing methodology: pytest with AAA pattern (Arrange-Act-Assert)
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qneural.hardware.rydberg import (
    # Pulses
    zero_pulse,
    constant_pulse,
    piecewise_constant,
    gaussian_pulse,
    blackman_pulse,
    cutoff_pulse,
    add_pulses,
    pulse_area,
    # Hamiltonian
    RydbergHamiltonian,
    create_constant_hamiltonian,
)
from qneural.hardware.rydberg.constants import HILBERT_DIM_GG, RABI_DEFAULT
from qneural.core import (
    # Evolution
    schrodinger_evolution,
    evolve_unitary,
    time_evolution_operator,
    evolve_state,
    basis_tensor,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def single_qubit_state():
    """Single qubit in ground state |0⟩."""
    return basis_tensor('0', dim=3)


@pytest.fixture
def two_qubit_state():
    """Two qubits in ground state |00⟩."""
    return basis_tensor('00', dim=3)


@pytest.fixture
def constant_hamiltonian_1q():
    """Constant Hamiltonian for 1 qubit."""
    return create_constant_hamiltonian(
        nqubits=1,
        rabi_amplitude=0.0,  # No drive
        detuning_amplitude=0.0  # No detuning
    )


@pytest.fixture
def constant_hamiltonian_2q():
    """Constant Hamiltonian for 2 qubits."""
    return create_constant_hamiltonian(
        nqubits=2,
        rabi_amplitude=0.0,
        detuning_amplitude=0.0
    )


# =============================================================================
# Test Pulses
# =============================================================================

class TestZeroPulse:
    """Tests for zero_pulse function."""

    def test_returns_zero_at_any_time(self):
        """zero_pulse should return 0 for any input."""
        # Arrange & Act
        result = zero_pulse(0.0)

        # Assert
        assert result == 0.0

    def test_returns_zero_for_tensor_input(self):
        """zero_pulse should handle tensor inputs."""
        # Arrange & Act
        result = zero_pulse(torch.tensor(5.0))

        # Assert
        assert result == 0.0


class TestConstantPulse:
    """Tests for constant_pulse function."""

    def test_returns_constant_value(self):
        """constant_pulse should return the same value always."""
        # Arrange
        amplitude = 2.0 * torch.pi
        pulse_fn = constant_pulse(amplitude)

        # Act
        result1 = pulse_fn(0.0)
        result2 = pulse_fn(1.0)
        result3 = pulse_fn(100.0)

        # Assert
        assert torch.allclose(result1, torch.tensor(amplitude))
        assert torch.allclose(result2, torch.tensor(amplitude))
        assert torch.allclose(result3, torch.tensor(amplitude))

    def test_different_amplitudes(self):
        """Different amplitudes should give different pulses."""
        # Arrange
        pulse1 = constant_pulse(1.0)
        pulse2 = constant_pulse(2.0)

        # Act
        r1 = pulse1(0.0)
        r2 = pulse2(0.0)

        # Assert
        assert r1 == 1.0
        assert r2 == 2.0
        assert r2 == 2 * r1


class TestPiecewiseConstant:
    """Tests for piecewise_constant function."""

    def test_single_segment(self):
        """Single value should return constant."""
        # Arrange
        values = torch.tensor([5.0])
        pulse_fn = piecewise_constant(values, total_time=1.0)

        # Act
        result = pulse_fn(0.5)

        # Assert
        assert result == 5.0

    def test_multiple_segments(self):
        """Should return correct segment value."""
        # Arrange
        values = torch.tensor([1.0, 2.0, 3.0])
        pulse_fn = piecewise_constant(values, total_time=3.0)

        # Act & Assert
        assert pulse_fn(0.5) == 1.0  # First second
        assert pulse_fn(1.5) == 2.0  # Second second
        assert pulse_fn(2.5) == 3.0  # Third second

    def test_batched_values(self):
        """Should handle batched values."""
        # Arrange
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # [batch=2, time_steps=2]
        pulse_fn = piecewise_constant(values, total_time=2.0)

        # Act
        result = pulse_fn(0.5)  # First half

        # Assert
        assert result.shape == (2,)
        assert result[0] == 1.0
        assert result[1] == 3.0


class TestGaussianPulse:
    """Tests for gaussian_pulse function."""

    def test_peak_at_center(self):
        """Peak should be at center time."""
        # Arrange
        pulse_fn = gaussian_pulse(amplitude=1.0, center=0.5, width=0.1)

        # Act
        result = pulse_fn(0.5)

        # Assert
        assert torch.allclose(result, torch.tensor(1.0), atol=1e-6)

    def test_decreases_away_from_center(self):
        """Should decrease away from center."""
        # Arrange
        pulse_fn = gaussian_pulse(amplitude=1.0, center=0.5, width=0.1)

        # Act
        peak = pulse_fn(0.5)
        off_center = pulse_fn(0.6)

        # Assert
        assert off_center < peak


class TestBlackmanPulse:
    """Tests for blackman_pulse function."""

    def test_zero_at_boundaries(self):
        """Should be near zero at start and end."""
        # Arrange
        pulse_fn = blackman_pulse(amplitude=1.0, duration=1.0)

        # Act
        start = pulse_fn(0.0)
        end = pulse_fn(1.0)

        # Assert
        assert torch.abs(start) < 0.01
        assert torch.abs(end) < 0.01

    def test_peak_in_middle(self):
        """Should have maximum in the middle."""
        # Arrange
        pulse_fn = blackman_pulse(amplitude=1.0, duration=1.0)

        # Act
        peak = pulse_fn(0.5)

        # Assert
        assert peak > pulse_fn(0.1)
        assert peak > pulse_fn(0.9)


class TestCutoffPulse:
    """Tests for cutoff_pulse function."""

    def test_active_before_cutoff(self):
        """Should return original pulse value before cutoff."""
        # Arrange
        original = constant_pulse(5.0)
        pulse_fn = cutoff_pulse(original, cutoff_time=1.0)

        # Act
        result = pulse_fn(0.5)

        # Assert
        assert result == 5.0

    def test_zero_after_cutoff(self):
        """Should return zero after cutoff."""
        # Arrange
        original = constant_pulse(5.0)
        pulse_fn = cutoff_pulse(original, cutoff_time=1.0)

        # Act
        result = pulse_fn(2.0)

        # Assert
        assert result == 0.0


class TestAddPulses:
    """Tests for add_pulses function."""

    def test_sums_pulses(self):
        """Should sum multiple pulses."""
        # Arrange
        p1 = constant_pulse(1.0)
        p2 = constant_pulse(2.0)
        combined = add_pulses(p1, p2)

        # Act
        result = combined(0.0)

        # Assert
        assert result == 3.0


# =============================================================================
# Test Hamiltonian
# =============================================================================

class TestRydbergHamiltonian:
    """Tests for RydbergHamiltonian class."""

    def test_single_qubit_hamiltonian_shape(self):
        """1-qubit Hamiltonian should be 3x3."""
        # Arrange
        ham = create_constant_hamiltonian(1, 0.0, 0.0)

        # Act
        H = ham(0.0)

        # Assert
        assert H.shape == (3, 3)

    def test_two_qubit_hamiltonian_shape(self):
        """2-qubit Hamiltonian should be 9x9."""
        # Arrange
        ham = create_constant_hamiltonian(2, 0.0, 0.0)

        # Act
        H = ham(0.0)

        # Assert
        assert H.shape == (9, 9)

    def test_hamiltonian_is_hermitian_no_decay(self):
        """Hamiltonian should be Hermitian (no decay)."""
        # Arrange
        ham = create_constant_hamiltonian(2, RABI_DEFAULT, 0.0)

        # Act
        H = ham(0.0)

        # Assert
        assert torch.allclose(H, H.conj().T, atol=1e-10)

    def test_global_addressing_same_pulses(self):
        """Global addressing should use same pulse for all qubits."""
        # Arrange
        pulse = constant_pulse(1.0)
        ham = RydbergHamiltonian(
            nqubits=2,
            rabi_pulse=pulse,
            detuning_pulse=pulse,
            addressing='global'
        )

        # Act
        H = ham(0.0)

        # Assert - just checking it runs without error
        assert H.shape == (9, 9)

    def test_local_addressing_requires_list(self):
        """Local addressing should require list of pulses."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError):
            RydbergHamiltonian(
                nqubits=2,
                rabi_pulse=constant_pulse(1.0),  # Single pulse, not list
                detuning_pulse=constant_pulse(0.0),
                addressing='local'
            )

    def test_interaction_present_for_2qubits(self):
        """Van der Waals interaction should be present for 2+ qubits."""
        # Arrange
        ham = create_constant_hamiltonian(2, 0.0, 0.0)

        # Act
        H = ham(0.0)

        # Assert - interaction should make H non-zero
        assert not torch.allclose(H, torch.zeros_like(H))

    def test_no_interaction_for_1qubit(self):
        """No interaction term for single qubit."""
        # Arrange
        ham = create_constant_hamiltonian(1, 0.0, 0.0)

        # Act
        H = ham(0.0)

        # Assert - should be zero when no drive
        assert torch.allclose(H, torch.zeros_like(H))

    def test_hilbert_dim_property(self):
        """get_hilbert_dim should return correct dimension."""
        # Arrange
        ham1 = create_constant_hamiltonian(1, 0.0, 0.0)
        ham2 = create_constant_hamiltonian(2, 0.0, 0.0)

        # Assert
        assert ham1.get_hilbert_dim() == 3
        assert ham2.get_hilbert_dim() == 9

    def test_batch_mode(self):
        """batch_size parameter should expand Hamiltonian."""
        # Arrange
        ham = create_constant_hamiltonian(1, 1.0, 0.0)

        # Act
        H = ham(0.0, batch_size=5)

        # Assert
        assert H.shape == (5, 3, 3)


# =============================================================================
# Test Time Evolution
# =============================================================================

class TestSchrodingerEvolution:
    """Tests for schrodinger_evolution function."""

    def test_trivial_evolution_no_hamiltonian(self, single_qubit_state):
        """No Hamiltonian = no evolution."""
        # Arrange
        ham = create_constant_hamiltonian(1, 0.0, 0.0)

        # Act
        result = schrodinger_evolution(
            single_qubit_state,
            ham,
            t_span=(0.0, 1.0),
            method='euler'  # Simple method for testing
        )

        # Assert - state should be unchanged
        final_state = result[-1]
        assert torch.allclose(final_state, single_qubit_state, atol=1e-5)

    def test_evolution_preserves_norm(self, single_qubit_state):
        """Evolution should preserve state norm."""
        # Arrange
        # Create a simple rotation Hamiltonian
        rabi = constant_pulse(2.0 * torch.pi)  # 1 Hz Rabi frequency
        ham = RydbergHamiltonian(
            nqubits=1,
            rabi_pulse=rabi,
            detuning_pulse=zero_pulse,
            addressing='global'
        )

        # Act
        result = schrodinger_evolution(
            single_qubit_state,
            ham,
            t_span=(0.0, 0.25),  # Quarter period
            method='dopri5'
        )

        # Assert - norm should be preserved
        for state in result:
            norm = torch.norm(state)
            assert torch.allclose(norm, torch.tensor(1.0), atol=1e-4)

    def test_full_rabi_cycle(self, single_qubit_state):
        """Full Rabi cycle should return to initial state."""
        # Arrange
        omega = 2.0 * torch.pi  # 1 Hz
        rabi = constant_pulse(omega)
        ham = RydbergHamiltonian(
            nqubits=1,
            rabi_pulse=rabi,
            detuning_pulse=zero_pulse,
            addressing='global'
        )

        # Act - one full period
        result = schrodinger_evolution(
            single_qubit_state,
            ham,
            t_span=(0.0, 1.0),
            method='dopri5'
        )

        # Assert - should return close to initial state
        final_state = result[-1]
        overlap = torch.abs(torch.matmul(single_qubit_state.conj().T, final_state))
        assert overlap > 0.99  # Should be very close to 1

    def test_two_qubit_evolution(self, two_qubit_state):
        """Can evolve two-qubit states."""
        # Arrange
        ham = create_constant_hamiltonian(2, 0.0, 0.0)

        # Act
        result = schrodinger_evolution(
            two_qubit_state,
            ham,
            t_span=(0.0, 0.1),
            method='euler'
        )

        # Assert
        assert result.shape[1:] == (9, 1)  # [n_times, 9, 1]


class TestEvolveUnitary:
    """Tests for evolve_unitary function."""

    def test_identity_evolution(self):
        """Evolution of identity gives time evolution operator."""
        # Arrange
        ham = create_constant_hamiltonian(1, 0.0, 0.0)
        U0 = torch.eye(3, dtype=torch.cfloat)

        # Act
        result = evolve_unitary(U0, ham, t_span=(0.0, 1.0), method='euler')

        # Assert - should still be identity (no evolution)
        assert torch.allclose(result[-1], U0, atol=1e-4)

    def test_unitary_preserved(self):
        """Evolution should preserve unitarity."""
        # Arrange
        rabi = constant_pulse(2.0 * torch.pi)
        ham = RydbergHamiltonian(
            nqubits=1,
            rabi_pulse=rabi,
            detuning_pulse=zero_pulse,
            addressing='global'
        )
        U0 = torch.eye(3, dtype=torch.cfloat)

        # Act
        result = evolve_unitary(U0, ham, t_span=(0.0, 0.5), method='dopri5')

        # Assert - check unitarity at each time
        for U in result:
            identity = torch.matmul(U, U.conj().T)
            assert torch.allclose(identity, torch.eye(3, dtype=torch.cfloat), atol=1e-4)


class TestTimeEvolutionOperator:
    """Tests for time_evolution_operator function."""

    def test_returns_unitary(self):
        """Should return a unitary matrix."""
        # Arrange
        ham = create_constant_hamiltonian(1, 0.0, 0.0)

        # Act
        U = time_evolution_operator(ham, t_span=(0.0, 1.0), method='euler')

        # Assert
        assert U.shape == (3, 3)
        identity = torch.matmul(U, U.conj().T)
        assert torch.allclose(identity, torch.eye(3, dtype=torch.cfloat), atol=1e-4)


class TestEvolveState:
    """Tests for evolve_state function."""

    def test_identity_does_nothing(self, single_qubit_state):
        """Identity operator should leave state unchanged."""
        # Arrange
        U = torch.eye(3, dtype=torch.cfloat)

        # Act
        result = evolve_state(single_qubit_state, U)

        # Assert
        assert torch.allclose(result, single_qubit_state)

    def test_pauli_x_flips_qubit(self):
        """Pauli X should flip |0⟩ to |1⟩ (for 2-level)."""
        # Arrange - use 2D Hilbert space
        psi0 = torch.tensor([[1.0], [0.0]], dtype=torch.cfloat)
        sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)

        # Act
        result = evolve_state(psi0, sigma_x)

        # Assert
        expected = torch.tensor([[0.0], [1.0]], dtype=torch.cfloat)
        assert torch.allclose(result, expected)

    def test_batch_evolution(self):
        """Should handle batched unitaries."""
        # Arrange
        psi = torch.tensor([[1.0], [0.0]], dtype=torch.cfloat)
        # Batch of two unitaries
        U_batch = torch.eye(2, dtype=torch.cfloat).unsqueeze(0).expand(2, -1, -1)

        # Act
        result = evolve_state(psi, U_batch)

        # Assert
        assert result.shape == (2, 2, 1)


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhysicsIntegration:
    """Integration tests combining pulses, Hamiltonian, and evolution."""

    def test_full_pipeline_single_qubit(self):
        """Complete pipeline: pulse → Hamiltonian → evolution."""
        # Arrange
        psi0 = basis_tensor('0', dim=3)

        # Create pulse
        rabi = constant_pulse(2.0 * torch.pi * 1e-3)  # 1 kHz
        detuning = constant_pulse(0.0)

        # Create Hamiltonian
        ham = RydbergHamiltonian(
            nqubits=1,
            rabi_pulse=rabi,
            detuning_pulse=detuning,
            addressing='global'
        )

        # Act
        result = schrodinger_evolution(psi0, ham, t_span=(0.0, 1.0), method='dopri5')

        # Assert - state should have evolved and still be normalized
        final_state = result[-1]
        norm = torch.norm(final_state)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)

    def test_full_pipeline_two_qubit_interaction(self):
        """Two qubits with interaction."""
        # Arrange
        psi0 = basis_tensor('00', dim=3)

        # Create Hamiltonian with interaction
        ham = create_constant_hamiltonian(
            nqubits=2,
            rabi_amplitude=0.0,  # No drive
            detuning_amplitude=0.0  # No detuning
            # But interaction is still present!
        )

        # Act - evolve with just interaction
        result = schrodinger_evolution(psi0, ham, t_span=(0.0, 0.1), method='dopri5')

        # Assert - |00⟩ is eigenstate of interaction, so shouldn't change much
        final_state = result[-1]
        overlap = torch.abs(torch.matmul(psi0.conj().T, final_state))
        # Should still be mostly |00⟩ since there's no population in |r⟩ initially
        assert overlap > 0.99

    @pytest.mark.slow
    def test_rabi_oscillations(self):
        """Test Rabi oscillations match expected frequency."""
        # Arrange
        omega_rabi = 2.0 * torch.pi * 1.0  # 1 Hz Rabi frequency
        rabi = constant_pulse(omega_rabi)
        ham = RydbergHamiltonian(
            nqubits=1,
            rabi_pulse=rabi,
            detuning_pulse=zero_pulse,
            addressing='global'
        )

        # Start from |1⟩ (state 1) to see Rabi oscillations to |r⟩
        # Rabi coupling in Rydberg systems is between |1⟩ and |r⟩
        psi0 = basis_tensor('1', dim=3)

        # Evolve for half a Rabi period (should flip population from |1⟩ to |r⟩)
        # Full Rabi period = 2π/Ω, so half period = π/Ω = π/(2π*1) = 0.5 seconds
        t_half_period = 0.5  # seconds

        # Act
        result = schrodinger_evolution(
            psi0,
            ham,
            t_span=(0.0, t_half_period),
            method='dopri5'
        )

        # Assert - check that |1⟩ population has decreased (transferred to |r⟩)
        final_state = result[-1]
        p1 = torch.abs(final_state[1, 0]) ** 2  # Population in |1⟩
        # After half period of Rabi oscillation, should be near 0 (in |r⟩)
        assert p1 < 0.1  # Should have flipped to |r⟩


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])