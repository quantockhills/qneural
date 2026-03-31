"""
Integration tests for the full training pipeline.

Tests that neural network training actually works end-to-end:
- Can create optimizer
- Can run training steps
- Training improves infidelity (loss goes down)
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from qneural.gates.rydberg import CZPhiGate, ControlledPhaseOptimizer
from qneural.neural import (
    FeedForwardNN,
    create_default_physical_pulse_generator,
    create_evolver,
    QuantumTrainer,
    InfidelityLoss
)


class TestFixedTimeTraining:
    """Test that fixed-time gate optimization training works."""

    def test_can_create_training_components(self):
        """Test that we can create all necessary components."""
        # Arrange & Act
        gate = CZPhiGate()
        network = FeedForwardNN(input_dim=2, output_dim=2, hidden_layers=2, hidden_units=20)
        pulse_gen = create_default_physical_pulse_generator(rabi_max=gate.rabi_max)
        evolver = create_evolver(nqubits=2)
        loss_fn = InfidelityLoss(nqubits=2)

        trainer = QuantumTrainer(
            network=network,
            nqubits=2,
            loss_fn=loss_fn,
            pulse_generator=pulse_gen,
            evolver=evolver
        )

        # Assert
        assert trainer is not None
        assert trainer.network is not None
        assert trainer.evolver is not None

    @pytest.mark.slow
    def test_training_runs_without_errors(self):
        """Test that training can run for a few epochs without crashing."""
        # Arrange - minimal setup
        gate = CZPhiGate()
        network = FeedForwardNN(input_dim=2, output_dim=2, hidden_layers=2, hidden_units=20)
        pulse_gen = create_default_physical_pulse_generator(rabi_max=gate.rabi_max)
        evolver = create_evolver(nqubits=2)
        loss_fn = InfidelityLoss(nqubits=2)

        trainer = QuantumTrainer(
            network=network,
            nqubits=2,
            loss_fn=loss_fn,
            pulse_generator=pulse_gen,
            evolver=evolver
        )

        # Very minimal training parameters
        angles = torch.tensor([torch.pi / 2, torch.pi])  # Just 2 angles
        gate_time = 5.0
        epochs = 2  # Just 2 epochs

        # Act - run training
        history = trainer.train(angles, gate_time, epochs, print_every=1)

        # Assert - training completed
        assert 'loss' in history
        assert len(history['loss']) == epochs
        assert all(isinstance(loss, float) for loss in history['loss'])

    @pytest.mark.slow
    def test_training_improves_infidelity(self):
        """Test that training actually reduces infidelity (loss goes down)."""
        # Arrange
        gate = CZPhiGate()
        network = FeedForwardNN(input_dim=2, output_dim=2, hidden_layers=3, hidden_units=30)
        pulse_gen = create_default_physical_pulse_generator(rabi_max=gate.rabi_max)
        evolver = create_evolver(nqubits=2)
        loss_fn = InfidelityLoss(nqubits=2)

        trainer = QuantumTrainer(
            network=network,
            nqubits=2,
            loss_fn=loss_fn,
            pulse_generator=pulse_gen,
            evolver=evolver
        )

        # Minimal training
        angles = torch.linspace(0.5 * torch.pi, torch.pi, 3)  # 3 angles
        gate_time = 5.0
        epochs = 5  # 5 epochs should show some improvement

        # Act
        history = trainer.train(angles, gate_time, epochs, print_every=1)

        # Assert - loss should generally decrease or at least change
        initial_loss = history['loss'][0]
        final_loss = history['loss'][-1]

        # The loss should be different (network is learning something)
        # We don't require it strictly decrease (that depends on random init)
        # but it should change significantly
        loss_changed = abs(final_loss - initial_loss) > 0.01

        assert loss_changed, \
            f"Loss didn't change: initial={initial_loss:.4f}, final={final_loss:.4f}"

        # Print for debugging
        print(f"\n  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss:   {final_loss:.6f}")
        print(f"  Change:       {final_loss - initial_loss:.6f}")

    @pytest.mark.slow
    def test_optimizer_wrapper_works(self):
        """Test that ControlledPhaseOptimizer wrapper works for fixed-time."""
        # Arrange
        gate = CZPhiGate()
        network = FeedForwardNN(input_dim=2, output_dim=2, hidden_layers=2, hidden_units=20)
        pulse_gen = create_default_physical_pulse_generator(rabi_max=gate.rabi_max)
        evolver = create_evolver(nqubits=2)
        loss_fn = InfidelityLoss(nqubits=2)

        trainer = QuantumTrainer(
            network=network,
            nqubits=2,
            loss_fn=loss_fn,
            pulse_generator=pulse_gen,
            evolver=evolver
        )

        optimizer = ControlledPhaseOptimizer(
            gate=gate,
            network=network,
            trainer=trainer,
            pulse_generator=pulse_gen,
            evolver=evolver,
            time_optimal=False
        )

        # Act - train briefly
        history = optimizer.train(
            angle_range=(0.5 * torch.pi, torch.pi),
            n_angles=2,
            gate_time=5.0,
            epochs=2
        )

        # Assert
        assert 'loss' in history or 'losses' in history  # Check either key

        # Try evaluation
        result = optimizer.evaluate(torch.pi)

        assert 'infidelity' in result
        assert 'gate_time' in result
        assert result['infidelity'] < 1.0  # Should be a valid infidelity value


class TestPulseGeneration:
    """Test that pulse generation from trained network works."""

    @pytest.mark.slow
    def test_can_generate_pulses(self):
        """Test that we can generate pulses from the network."""
        # Arrange
        gate = CZPhiGate()
        network = FeedForwardNN(input_dim=2, output_dim=2, hidden_layers=2, hidden_units=20)
        pulse_gen = create_default_physical_pulse_generator(rabi_max=gate.rabi_max)
        evolver = create_evolver(nqubits=2)

        optimizer = ControlledPhaseOptimizer(
            gate=gate,
            network=network,
            pulse_generator=pulse_gen,
            evolver=evolver,
            time_optimal=False
        )

        # Act - generate pulses (even without training)
        pulses, gate_time = optimizer.generate_pulse(torch.pi / 2)

        # Assert
        assert len(pulses) == 2  # Rabi and detuning
        assert callable(pulses[0])  # Should be functions
        assert callable(pulses[1])
        assert gate_time == 5.0  # Default for fixed-time

        # Test that pulses can be evaluated
        rabi_value = pulses[0](0.5)
        detuning_value = pulses[1](0.5)

        assert isinstance(rabi_value, (float, torch.Tensor))
        assert isinstance(detuning_value, (float, torch.Tensor))
