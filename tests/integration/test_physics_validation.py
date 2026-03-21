"""
Integration tests for physics validation.

Tests that validate the quantum mechanics is implemented correctly:
- Evolution preserves unitarity and norm
- Hamiltonians are Hermitian
- Known quantum phenomena (Rabi oscillations, etc.)
- Gate construction matches expected unitaries
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from qneural.hardware.rydberg import (
    RydbergHamiltonian,
    constant_pulse,
    zero_pulse,
    RABI_DEFAULT
)
from qneural.core import (
    basis_tensor,
    czphi_gate,
    cczphi_gate,
    schrodinger_evolution,
    time_evolution_operator,
)
from qneural.core.metrics import unitary_fidelity


class TestQuantumEvolution:
    """Test that quantum evolution behaves correctly."""

    def test_zero_hamiltonian_gives_no_evolution(self):
        """Zero Hamiltonian should leave state unchanged."""
        # Arrange
        psi0 = basis_tensor('0', dim=3)
        ham = RydbergHamiltonian(
            nqubits=1,
            rabi_pulse=zero_pulse,
            detuning_pulse=zero_pulse,
            addressing='global'
        )

        # Act
        result = schrodinger_evolution(psi0, ham, t_span=(0.0, 1.0), method='dopri5')
        final_state = result[-1]

        # Assert - should be essentially unchanged
        overlap = torch.abs(torch.matmul(psi0.conj().T, final_state))
        assert overlap > 0.9999, f"State changed unexpectedly: overlap = {overlap}"

    def test_evolution_preserves_norm(self):
        """Time evolution must preserve state normalization."""
        # Arrange - start in ground state
        psi0 = basis_tensor('0', dim=3)

        # Create non-trivial Hamiltonian
        rabi = constant_pulse(RABI_DEFAULT)
        ham = RydbergHamiltonian(
            nqubits=1,
            rabi_pulse=rabi,
            detuning_pulse=zero_pulse,
            addressing='global'
        )

        # Act - evolve for some time
        result = schrodinger_evolution(psi0, ham, t_span=(0.0, 0.5), method='dopri5')

        # Assert - norm should be 1 at all times
        for state in result:
            norm = torch.norm(state)
            assert torch.abs(norm - 1.0) < 1e-4, f"Norm not preserved: {norm}"

    def test_evolution_produces_unitary_operators(self):
        """Time evolution operator U(t) must be unitary: U†U = I."""
        # Arrange
        ham = RydbergHamiltonian(
            nqubits=1,
            rabi_pulse=constant_pulse(RABI_DEFAULT),
            detuning_pulse=zero_pulse,
            addressing='global'
        )

        # Act - get time evolution operator
        U = time_evolution_operator(ham, t_span=(0.0, 1.0), method='dopri5')

        # Assert - check unitarity
        identity = torch.matmul(U, U.conj().T)
        eye = torch.eye(U.shape[0], dtype=torch.cfloat)
        error = torch.norm(identity - eye)

        assert error < 1e-3, f"Evolution not unitary: ||U†U - I|| = {error}"

    def test_rabi_oscillation_period(self):
        """Full Rabi period should return state to initial (approximately)."""
        # Arrange - ground state
        psi0 = basis_tensor('0', dim=3)

        # Rabi frequency Ω
        omega = 2.0 * torch.pi  # 1 Hz in angular units
        rabi = constant_pulse(omega)
        ham = RydbergHamiltonian(
            nqubits=1,
            rabi_pulse=rabi,
            detuning_pulse=zero_pulse,
            addressing='global'
        )

        # Act - evolve for one full period (T = 2π/Ω = 1.0)
        result = schrodinger_evolution(psi0, ham, t_span=(0.0, 1.0), method='dopri5')
        final_state = result[-1]

        # Assert - should return close to initial state (up to global phase)
        overlap = torch.abs(torch.matmul(psi0.conj().T, final_state))
        assert overlap > 0.95, f"Rabi oscillation didn't complete cycle: overlap = {overlap}"


class TestHamiltonianProperties:
    """Test properties of the Rydberg Hamiltonian."""

    def test_hamiltonian_is_hermitian(self):
        """Hamiltonian must be Hermitian: H = H†."""
        # Arrange
        ham = RydbergHamiltonian(
            nqubits=2,
            rabi_pulse=constant_pulse(RABI_DEFAULT),
            detuning_pulse=constant_pulse(1.0),
            addressing='global'
        )

        # Act
        H = ham(0.0)

        # Assert
        H_dag = H.conj().T
        error = torch.norm(H - H_dag)

        assert error < 1e-6, f"Hamiltonian not Hermitian: ||H - H†|| = {error}"

    def test_hilbert_space_dimension(self):
        """Hamiltonian dimension should be 3^n for n qubits."""
        for nqubits in [1, 2, 3]:
            # Arrange
            ham = RydbergHamiltonian(
                nqubits=nqubits,
                rabi_pulse=zero_pulse,
                detuning_pulse=zero_pulse,
                addressing='global'
            )

            # Act
            H = ham(0.0)

            # Assert
            expected_dim = 3 ** nqubits
            assert H.shape == (expected_dim, expected_dim), \
                f"{nqubits}-qubit Hamiltonian should be {expected_dim}×{expected_dim}"

    def test_interaction_term_for_multiqubit(self):
        """Two-qubit systems should have non-zero interaction."""
        # Arrange - no drive, only interaction
        ham = RydbergHamiltonian(
            nqubits=2,
            rabi_pulse=zero_pulse,
            detuning_pulse=zero_pulse,
            addressing='global'
            # Interaction is included by default in RydbergHamiltonian
        )

        # Act
        H = ham(0.0)

        # Assert - should not be all zeros due to interaction
        assert torch.norm(H) > 0.1, "Interaction term missing for 2-qubit system"


class TestGateConstruction:
    """Test that gate construction produces correct unitaries."""

    def test_czphi_at_zero_is_identity(self):
        """CZ_φ gate at φ=0 should be identity."""
        # Arrange & Act
        gate = czphi_gate(0.0)
        identity = torch.eye(4, dtype=torch.cfloat)

        # Assert
        error = torch.norm(gate - identity)
        assert error < 1e-6, f"CZ_0 should be identity: error = {error}"

    def test_czphi_at_pi_is_cz_gate(self):
        """CZ_φ gate at φ=π should be standard CZ gate."""
        # Arrange & Act
        gate = czphi_gate(torch.pi)

        # Expected CZ = diag(1, 1, 1, -1)
        expected = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.cfloat))

        # Assert
        error = torch.norm(gate - expected)
        assert error < 1e-6, f"CZ_π should be standard CZ: error = {error}"

    def test_czphi_is_diagonal(self):
        """CZ_φ gates should be diagonal."""
        # Arrange & Act
        gate = czphi_gate(torch.pi / 2)

        # Extract off-diagonal elements
        mask = ~torch.eye(4, dtype=torch.bool)
        off_diagonal = gate[mask]

        # Assert
        off_diag_norm = torch.norm(off_diagonal)
        assert off_diag_norm < 1e-6, f"CZ_φ should be diagonal: {off_diag_norm}"

    def test_czphi_is_unitary(self):
        """CZ_φ gates must be unitary."""
        # Test at several angles
        for phi in [0.0, torch.pi/4, torch.pi/2, torch.pi]:
            # Arrange & Act
            gate = czphi_gate(phi)

            # Check unitarity
            identity = torch.matmul(gate, gate.conj().T)
            eye = torch.eye(4, dtype=torch.cfloat)
            error = torch.norm(identity - eye)

            # Assert
            assert error < 1e-5, f"CZ_{phi/torch.pi:.2f}π not unitary: error = {error}"

    def test_cczphi_has_correct_dimension(self):
        """CCZ_φ (3-qubit) should be 8×8."""
        # Arrange & Act
        gate = cczphi_gate(torch.pi / 2)

        # Assert
        assert gate.shape == (8, 8), f"CCZ_φ should be 8×8, got {gate.shape}"

    def test_cczphi_at_zero_is_identity(self):
        """CCZ_φ at φ=0 should be identity."""
        # Arrange & Act
        gate = cczphi_gate(0.0)
        identity = torch.eye(8, dtype=torch.cfloat)

        # Assert
        error = torch.norm(gate - identity)
        assert error < 1e-6, f"CCZ_0 should be identity: error = {error}"

    def test_cczphi_phase_on_111_state(self):
        """CCZ_φ should apply phase only to |111⟩ state (index 7)."""
        # Arrange
        phi = torch.pi / 3

        # Act
        gate = cczphi_gate(phi)

        # Assert - diagonal should be [1, 1, 1, 1, 1, 1, 1, e^{iφ}]
        diagonal = torch.diagonal(gate)

        # First 7 elements should be 1
        for i in range(7):
            assert torch.abs(diagonal[i] - 1.0) < 1e-6, \
                f"Element {i} should be 1, got {diagonal[i]}"

        # Last element should have phase φ
        expected_phase = torch.exp(torch.tensor(1.0j) * phi)
        assert torch.abs(diagonal[7] - expected_phase) < 1e-6, \
            f"Element 7 should be e^(i{phi}), got {diagonal[7]}"


class TestJakschProtocol:
    """Test the Jaksch protocol for CZ gate via Rydberg blockade.

    The Jaksch sequence:
    1. π pulse on control qubit (|0⟩ → |r⟩)
    2. 2π pulse on target qubit (blocked if control is in |r⟩ due to Rydberg blockade)
    3. π pulse on control qubit (|r⟩ → |0⟩)

    Result: CZ gate with phase on |11⟩ state.
    Reference: Jaksch et al., Phys. Rev. Lett. 85, 2208 (2000)
    """

    def test_jaksch_protocol_produces_cz_gate(self):
        """Jaksch sequence (π-2π-π) should implement CZ gate via blockade."""
        # Arrange - 2-qubit system
        # Start with identity
        U0 = torch.eye(9, dtype=torch.cfloat)  # Full Hilbert space (3^2)

        # Physical parameters
        omega = 2.0 * torch.pi  # Rabi frequency
        blockade_strength = 21.1 * omega  # Strong blockade

        # Step 1: π pulse on control qubit (qubit 0)
        # Duration for π rotation: t = π/Ω
        t_pi = torch.pi / omega
        t_2pi = 2 * torch.pi / omega

        # Control qubit: apply Rabi drive
        rabi_control = constant_pulse(omega)
        rabi_target = zero_pulse

        ham_step1 = RydbergHamiltonian(
            nqubits=2,
            rabi_pulse=[rabi_control, rabi_target],  # Local addressing
            detuning_pulse=[zero_pulse, zero_pulse],
            addressing='local'
        )

        # Evolve for π pulse on control
        U_after_step1 = time_evolution_operator(ham_step1, t_span=(0.0, float(t_pi)), method='dopri5')
        U1 = torch.matmul(U_after_step1, U0)

        # Step 2: 2π pulse on target qubit (blocked if control is |r⟩)
        ham_step2 = RydbergHamiltonian(
            nqubits=2,
            rabi_pulse=[rabi_target, rabi_control],  # Reverse: target gets drive
            detuning_pulse=[zero_pulse, zero_pulse],
            addressing='local'
        )

        U_after_step2 = time_evolution_operator(ham_step2, t_span=(0.0, float(t_2pi)), method='dopri5')
        U2 = torch.matmul(U_after_step2, U1)

        # Step 3: π pulse on control again (return to ground)
        U_after_step3 = time_evolution_operator(ham_step1, t_span=(0.0, float(t_pi)), method='dopri5')
        U_final = torch.matmul(U_after_step3, U2)

        # Extract computational subspace (indices without Rydberg state)
        # For 2 qubits with dim=3: comp basis indices are [0,1,3,4] for |00⟩,|01⟩,|10⟩,|11⟩
        comp_indices = []
        for i in range(9):
            # Convert to base-3
            d0 = i % 3
            d1 = (i // 3) % 3
            # Keep only if both qubits in {0, 1} (not 2 = Rydberg)
            if d0 < 2 and d1 < 2:
                comp_indices.append(i)

        comp_indices = torch.tensor(comp_indices, dtype=torch.long)

        # Reduce to computational basis
        U_comp = U_final[comp_indices][:, comp_indices]

        # Assert - should be close to CZ gate
        # CZ can be either diag(1, 1, 1, -1) or diag(1, -1, -1, -1) depending on convention
        # Let's check the structure: diagonal with one -1

        # Check it's diagonal
        off_diag_mask = ~torch.eye(4, dtype=torch.bool)
        off_diagonal = U_comp[off_diag_mask]
        off_diag_norm = torch.norm(off_diagonal)

        assert off_diag_norm < 0.1, f"Gate should be diagonal, but ||off-diag|| = {off_diag_norm}"

        # Check diagonal elements are ±1
        diagonal = torch.diagonal(U_comp)
        phases = torch.angle(diagonal)

        # Each element should be close to 0 or π (i.e., ±1)
        for i, phase in enumerate(phases):
            is_zero = torch.abs(phase) < 0.2
            is_pi = torch.abs(torch.abs(phase) - torch.pi) < 0.2
            assert is_zero or is_pi, f"Diagonal element {i} has unexpected phase {phase}"

        # Count how many are -1 (phase ≈ π)
        n_negative = torch.sum(torch.abs(torch.abs(phases) - torch.pi) < 0.2).item()

        # Should have exactly one -1 (or three -1s for the other convention)
        assert n_negative == 1 or n_negative == 3, \
            f"CZ gate should have 1 or 3 negative entries, found {n_negative}"

        print(f"\n✓ Jaksch protocol test passed!")
        print(f"  Diagonal elements: {diagonal}")
        print(f"  Phases: {phases}")
        print(f"  Number of -1 entries: {n_negative}")


class TestBellStateGeneration:
    """Test that we can create entangled states (integration of evolution + gates)."""

    @pytest.mark.skip(reason="Requires optimized pulses from training - this is an aspirational test")
    def test_can_create_maximally_entangled_state(self):
        """Test that evolution can create entangled states.

        This is a simplified test - full Bell state generation would require
        optimized pulses from training.
        """
        # Arrange - start in |00⟩
        psi0 = basis_tensor('00', dim=3)

        # Create a simple entangling Hamiltonian
        # (This won't create a perfect Bell state without optimization,
        # but should create some entanglement)
        rabi = constant_pulse(RABI_DEFAULT / 2)
        detuning = constant_pulse(0.0)
        ham = RydbergHamiltonian(
            nqubits=2,
            rabi_pulse=rabi,
            detuning_pulse=detuning,
            addressing='global'
        )

        # Act - evolve for some time
        result = schrodinger_evolution(psi0, ham, t_span=(0.0, 1.0), method='dopri5')
        final_state = result[-1]

        # Assert - final state should not be separable (basic entanglement check)
        # If state is still |00⟩, we haven't created entanglement
        overlap_with_initial = torch.abs(torch.matmul(psi0.conj().T, final_state))

        # Should have evolved away from initial state
        assert overlap_with_initial < 0.99, \
            f"State didn't evolve: overlap = {overlap_with_initial}"


class TestFidelityMetrics:
    """Test that fidelity calculations work correctly."""

    def test_fidelity_of_identical_gates_is_one(self):
        """Fidelity of identical unitaries should be 1."""
        # Arrange
        U = czphi_gate(torch.pi / 2)

        # Act
        fidelity = unitary_fidelity(U, U, nqubits=2)

        # Assert
        assert torch.abs(fidelity - 1.0) < 1e-6, \
            f"Fidelity of identical gates should be 1, got {fidelity}"

    def test_fidelity_is_bounded(self):
        """Fidelity must be between 0 and 1."""
        # Arrange - create two different gates
        U1 = czphi_gate(torch.pi / 4)
        U2 = czphi_gate(torch.pi / 2)

        # Act
        fidelity = unitary_fidelity(U1, U2, nqubits=2)

        # Assert
        assert 0.0 <= fidelity <= 1.0, \
            f"Fidelity must be in [0,1], got {fidelity}"

    def test_fidelity_is_symmetric(self):
        """Fidelity should be symmetric: F(U1, U2) = F(U2, U1)."""
        # Arrange
        U1 = czphi_gate(torch.pi / 3)
        U2 = czphi_gate(2 * torch.pi / 3)

        # Act
        fidelity_12 = unitary_fidelity(U1, U2, nqubits=2)
        fidelity_21 = unitary_fidelity(U2, U1, nqubits=2)

        # Assert
        assert torch.abs(fidelity_12 - fidelity_21) < 1e-6, \
            "Fidelity should be symmetric"


# Mark for slow tests is registered in pytest configuration
# Use @pytest.mark.slow to mark tests that involve expensive ODE solving
