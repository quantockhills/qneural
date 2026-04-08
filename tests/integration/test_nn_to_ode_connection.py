"""
Test that Neural Network → ODE connection works.

Validates the full pipeline from NN output to quantum evolution via ODE solving.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from qneural.neural import (
    FeedForwardNN,
    create_default_physical_pulse_generator,
    create_evolver,
)
from qneural.hardware.rydberg import RABI_DEFAULT


@pytest.mark.slow
@pytest.mark.integration
class TestNNtoODEConnection:
    """Test that NN output correctly flows through to ODE evolution."""

    def test_nn_generates_valid_output(self):
        """Test that NN produces valid output shape."""
        # Arrange
        network = FeedForwardNN(
            input_dim=2, output_dim=2, hidden_layers=2, hidden_units=20
        )

        # Create input: (angle, time) pairs for 201 time steps
        angle = torch.tensor([torch.pi / 2])
        time_grid = torch.linspace(0, 1, 201)
        inputs = torch.stack([angle.repeat(201), time_grid], dim=1)  # [201, 2]

        # Act
        nn_output = network(inputs)

        # Assert
        assert nn_output.shape == (201, 2), f"Expected [201, 2], got {nn_output.shape}"
        assert torch.isfinite(nn_output).all(), "NN output contains NaN or Inf"

    def test_pulse_generator_converts_nn_output(self):
        """Test that pulse generator converts NN output to callable functions."""
        # Arrange
        pulse_gen = create_default_physical_pulse_generator(rabi_max=RABI_DEFAULT)

        # Simulate NN output (random values in [0, 1] from sigmoid)
        nn_output = torch.rand(201, 2)

        # Act
        pulses = pulse_gen.generate(nn_output, gate_time=5.0)

        # Assert
        assert len(pulses) == 2, "Should have 2 pulse functions (Rabi, detuning)"
        assert callable(pulses[0]), "Rabi pulse should be callable"
        assert callable(pulses[1]), "Detuning pulse should be callable"

        # Test that pulses can be evaluated
        t = 2.5  # halfway through gate time
        rabi_val = pulses[0](t)
        detuning_val = pulses[1](t)

        assert isinstance(rabi_val, (float, int, torch.Tensor))
        assert isinstance(detuning_val, (float, int, torch.Tensor))

    def test_evolver_uses_pulses_for_ode(self):
        """Test that evolver correctly uses pulse functions in ODE evolution."""
        # Arrange
        evolver = create_evolver(nqubits=2)
        pulse_gen = create_default_physical_pulse_generator(rabi_max=RABI_DEFAULT)

        # Create simple constant pulses (deterministic test)
        nn_output = torch.ones(201, 2) * 0.5  # Constant values
        pulses = pulse_gen.generate(nn_output, gate_time=1.0)

        # Act - evolve quantum system
        final_unitary = evolver.evolve(pulses, gate_time=1.0)

        # Assert
        assert final_unitary.shape == (4, 4), "2-qubit gate should be 4×4"
        assert torch.is_complex(final_unitary), "Unitary should be complex"

        # Check unitarity: U†U = I
        identity_test = torch.matmul(final_unitary, final_unitary.conj().T)
        eye = torch.eye(4, dtype=torch.cfloat)
        error = torch.norm(identity_test - eye)

        assert error < 0.01, (
            f"Evolution produced non-unitary result: ||U†U - I|| = {error}"
        )

    def test_full_nn_to_ode_pipeline(self):
        """Test complete pipeline: NN → Pulse → Hamiltonian → ODE → Unitary."""
        # Arrange
        network = FeedForwardNN(
            input_dim=2, output_dim=2, hidden_layers=2, hidden_units=20
        )
        pulse_gen = create_default_physical_pulse_generator(rabi_max=RABI_DEFAULT)
        evolver = create_evolver(nqubits=2)

        # Create input
        angle = torch.tensor([torch.pi / 2])
        time_grid = torch.linspace(0, 1, 201)
        inputs = torch.stack([angle.repeat(201), time_grid], dim=1)

        # Act - Full pipeline
        # Step 1: NN generates pulse parameters
        nn_output = network(inputs)

        # Step 2: Convert to pulse functions
        gate_time = 5.0
        pulses = pulse_gen.generate(nn_output, gate_time)

        # Step 3: Evolve via ODE (this uses torchdiffeq internally!)
        final_unitary = evolver.evolve(pulses, gate_time)

        # Assert
        assert final_unitary.shape == (4, 4)
        assert torch.is_complex(final_unitary)

        # Verify unitarity
        U_dag_U = torch.matmul(final_unitary, final_unitary.conj().T)
        eye = torch.eye(4, dtype=torch.cfloat)
        unitarity_error = torch.norm(U_dag_U - eye)

        assert unitarity_error < 0.01, (
            f"Full pipeline produced non-unitary: error = {unitarity_error}"
        )

        print("\n✓ Full NN→ODE pipeline test passed!")
        print(f"  Final unitary shape: {final_unitary.shape}")
        print(f"  Unitarity error: {unitarity_error:.6e}")

    def test_gradients_flow_through_ode(self):
        """Test that gradients can flow back through the ODE solver (autodiff works)."""
        # Arrange
        network = FeedForwardNN(
            input_dim=2, output_dim=2, hidden_layers=2, hidden_units=10
        )
        pulse_gen = create_default_physical_pulse_generator(rabi_max=RABI_DEFAULT)
        evolver = create_evolver(nqubits=2)

        # Input with gradients enabled
        angle = torch.tensor([torch.pi / 2], requires_grad=False)
        time_grid = torch.linspace(0, 1, 201)
        inputs = torch.stack([angle.repeat(201), time_grid], dim=1)

        # Act - Forward pass
        nn_output = network(inputs)
        pulses = pulse_gen.generate(nn_output, gate_time=5.0)
        final_unitary = evolver.evolve(pulses, gate_time=5.0)

        # Compute some loss (e.g., trace of unitary)
        loss = torch.abs(torch.trace(final_unitary))

        # Try to backpropagate
        loss.backward()

        # Assert - gradients should exist
        has_gradients = any(p.grad is not None for p in network.parameters())

        assert has_gradients, "Gradients didn't flow through ODE solver!"

        print("\n✓ Autodiff through ODE works!")
        print(f"  Loss value: {loss.item():.6f}")
        print(
            f"  Gradients computed: {sum(1 for p in network.parameters() if p.grad is not None)} parameters"
        )
