"""
Test that autodiff works through the ODE solver.

This is CRITICAL - if gradients don't flow through the ODE,
training won't work at all!
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from qneural.neural import (
    FeedForwardNN,
    create_default_physical_pulse_generator,
    create_evolver
)
from qneural.hardware.rydberg import RABI_DEFAULT


@pytest.mark.slow
@pytest.mark.integration
class TestAutodiffThroughODE:
    """Test that gradients flow through ODE solver."""

    def test_gradients_exist_after_ode_evolution(self):
        """Test that we can compute gradients through ODE evolution."""
        print("\n" + "="*60)
        print("Testing Autodiff Through ODE")
        print("="*60)

        # Arrange
        network = FeedForwardNN(
            input_dim=2,
            output_dim=2,
            hidden_layers=2,
            hidden_units=10
        )
        pulse_gen = create_default_physical_pulse_generator(rabi_max=RABI_DEFAULT)
        evolver = create_evolver(nqubits=2)

        # Create input
        angle = torch.tensor([torch.pi / 2])
        time_grid = torch.linspace(0, 1, 201)
        inputs = torch.stack([angle.repeat(201), time_grid], dim=1)

        # Act - Forward pass through full pipeline
        print("\nForward pass: NN → Pulses → ODE → Unitary")
        nn_output = network(inputs)
        pulses = pulse_gen.generate(nn_output, gate_time=5.0)
        final_unitary = evolver.evolve(pulses, gate_time=5.0)

        # Compute some loss
        loss = torch.abs(torch.trace(final_unitary))

        print(f"  Loss value: {loss.item():.6f}")

        # Backward pass
        print("\nBackward pass: computing gradients...")
        loss.backward()

        # Assert - check gradients exist
        params_with_grad = [p for p in network.parameters() if p.grad is not None]
        total_params = list(network.parameters())

        print(f"  Parameters with gradients: {len(params_with_grad)}/{len(total_params)}")

        assert len(params_with_grad) > 0, \
            "No gradients computed! Autodiff through ODE failed."

        # Check gradients are not all zero
        grad_norms = [torch.norm(p.grad).item() for p in params_with_grad]
        max_grad = max(grad_norms)
        print(f"  Max gradient norm: {max_grad:.6e}")

        assert max_grad > 1e-10, \
            f"All gradients are zero! Autodiff not working. Max: {max_grad}"

        print("\n✓ Autodiff through ODE works!")
        print("  Gradients flow: NN → Pulses → Hamiltonian → ODE → Loss")
        print("="*60)

    def test_gradients_change_with_different_inputs(self):
        """Test that gradients are actually meaningful (not constant)."""
        print("\n" + "="*60)
        print("Testing Gradient Variation")
        print("="*60)

        # Arrange
        network = FeedForwardNN(input_dim=2, output_dim=2, hidden_layers=2, hidden_units=10)
        pulse_gen = create_default_physical_pulse_generator(rabi_max=RABI_DEFAULT)
        evolver = create_evolver(nqubits=2)

        def compute_gradient_for_angle(angle_val):
            """Helper to compute gradient for a given angle."""
            # Zero gradients
            network.zero_grad()

            # Forward
            angle = torch.tensor([angle_val])
            time_grid = torch.linspace(0, 1, 201)
            inputs = torch.stack([angle.repeat(201), time_grid], dim=1)

            nn_output = network(inputs)
            pulses = pulse_gen.generate(nn_output, gate_time=5.0)
            final_unitary = evolver.evolve(pulses, gate_time=5.0)
            loss = torch.abs(torch.trace(final_unitary))

            # Backward
            loss.backward()

            # Get first parameter's gradient
            first_param_grad = list(network.parameters())[0].grad.clone()
            return first_param_grad, loss.item()

        # Act - compute gradients for two different angles
        grad1, loss1 = compute_gradient_for_angle(torch.pi / 4)
        grad2, loss2 = compute_gradient_for_angle(torch.pi / 2)

        # Assert - gradients should be different
        grad_diff = torch.norm(grad1 - grad2).item()

        print(f"\n  Angle π/4: loss = {loss1:.6f}")
        print(f"  Angle π/2: loss = {loss2:.6f}")
        print(f"  Gradient difference: {grad_diff:.6e}")

        assert grad_diff > 1e-6, \
            f"Gradients don't change with input! Diff = {grad_diff}"

        print("\n✓ Gradients vary with input (autodiff is meaningful)")
        print("="*60)

    def test_backward_pass_doesnt_crash(self):
        """Simplest test: backward() doesn't crash."""
        # Arrange
        network = FeedForwardNN(input_dim=2, output_dim=2, hidden_layers=1, hidden_units=5)
        pulse_gen = create_default_physical_pulse_generator(rabi_max=RABI_DEFAULT)
        evolver = create_evolver(nqubits=2)

        # Forward
        angle = torch.tensor([torch.pi])
        time_grid = torch.linspace(0, 1, 201)
        inputs = torch.stack([angle.repeat(201), time_grid], dim=1)

        nn_output = network(inputs)
        pulses = pulse_gen.generate(nn_output, gate_time=5.0)
        final_unitary = evolver.evolve(pulses, gate_time=5.0)
        loss = torch.norm(final_unitary)

        # Act & Assert - this should not crash
        try:
            loss.backward()
            print("\n✓ backward() completed without crashing")
        except Exception as e:
            pytest.fail(f"backward() crashed with: {e}")
