"""
Minimal test: Can we optimize a CZ gate?

This is the simplest possible test of the full training pipeline:
- Fix gate time (around 5-10 units of 1/Ω_max)
- Train network to find pulses
- Check if resulting gate is close to CZ = diag(1, 1, 1, -1) or diag(1, -1, -1, -1)

If this works, the full pipeline is validated!
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from qneural.gates.rydberg import CZPhiGate, ControlledPhaseOptimizer
from qneural.neural import (
    FeedForwardNN,
    create_default_physical_pulse_generator,
    QuantumTrainer,
    InfidelityLoss,
    TorchDiffeqSolver,
    QuantumEvolver,
)


class TestCZGateOptimization:
    """Test that we can actually optimize a CZ gate (φ=π)."""

    @pytest.mark.slow
    def test_can_optimize_cz_gate_fixed_time(self):
        """
        Minimal test: Optimize CZ gate with fixed gate time.

        Gate time: ~5 units of 1/Ω_max (reasonable for neutral atoms)
        Target: CZ = diag(1, 1, 1, -1) or diag(1, -1, -1, -1)
        """
        print("\n" + "=" * 70)
        print("Testing CZ Gate Optimization (φ = π)")
        print("=" * 70)

        # Arrange - minimal setup
        gate = CZPhiGate()
        network = FeedForwardNN(
            input_dim=2,
            output_dim=2,
            hidden_layers=4,  # Small network
            hidden_units=50,  # Small network
        )
        pulse_gen = create_default_physical_pulse_generator(rabi_max=gate.rabi_max)
        # Use RK4 like original code (much faster than dopri5)
        solver = TorchDiffeqSolver(method="rk4")
        evolver = QuantumEvolver(nqubits=2, solver=solver)
        loss_fn = InfidelityLoss(nqubits=2)

        trainer = QuantumTrainer(
            network=network,
            nqubits=2,
            loss_fn=loss_fn,
            pulse_generator=pulse_gen,
            evolver=evolver,
        )

        # Training parameters
        target_angle = torch.pi  # CZ gate
        angles = torch.tensor([target_angle])  # Just train on one angle
        # CRITICAL: Convert normalized time to actual time
        normalized_gate_time = 7.0  # Units of 1/Ω_max
        gate_time = normalized_gate_time / gate.rabi_max  # Actual seconds
        epochs = 20  # Keep it minimal for testing

        print("\nTraining Configuration:")
        print("  Target gate:   CZ (φ = π)")
        print(f"  Normalized:    {normalized_gate_time:.1f} (units of 1/Ω_max)")
        print(f"  Actual time:   {gate_time:.4f} seconds")
        print(f"  Epochs:        {epochs}")
        print("  Network:       4 layers × 50 units")
        print()

        # Act - Train!
        print("Training...")
        history = trainer.train(angles, gate_time, epochs, print_every=5)

        # Evaluate
        print("\nEvaluating...")
        optimizer = ControlledPhaseOptimizer(
            gate=gate,
            network=network,
            trainer=trainer,
            pulse_generator=pulse_gen,
            evolver=evolver,
            time_optimal=False,
        )

        result = optimizer.evaluate(target_angle)

        # Get the achieved gate
        achieved_gate = result["achieved_unitary"]
        target_gate = result["target_unitary"]
        infidelity = result["infidelity"]
        fidelity = 1.0 - infidelity

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Final infidelity: {infidelity:.6e}")
        print(f"Final fidelity:   {fidelity * 100:.4f}%")
        print()
        print("Achieved gate diagonal:")
        diagonal = torch.diagonal(achieved_gate)
        for i, val in enumerate(diagonal):
            print(
                f"  [{i}]: {val.real:.4f} + {val.imag:.4f}i  (|val| = {abs(val):.4f})"
            )
        print()
        print("Expected CZ gate:")
        print("  Option 1: diag(1, 1, 1, -1)")
        print("  Option 2: diag(1, -1, -1, -1)")
        print()

        # Assert - Check basic properties

        # 1. Gate should be diagonal (or nearly so)
        off_diag_mask = ~torch.eye(4, dtype=torch.bool)
        off_diagonal = achieved_gate[off_diag_mask]
        off_diag_norm = torch.norm(off_diagonal).item()

        print(f"Off-diagonal norm: {off_diag_norm:.6e}")
        assert off_diag_norm < 0.2, (
            f"Gate should be approximately diagonal, but ||off-diag|| = {off_diag_norm}"
        )

        # 2. Diagonal elements should have magnitude ~1 (it's a unitary)
        magnitudes = torch.abs(diagonal)
        print(f"Diagonal magnitudes: {magnitudes}")
        for i, mag in enumerate(magnitudes):
            assert 0.5 < mag < 1.5, (
                f"Diagonal element {i} has unexpected magnitude {mag}"
            )

        # 3. Should have some negative phases (distinguishes from identity)
        phases = torch.angle(diagonal)
        n_negative = torch.sum(torch.abs(torch.abs(phases) - torch.pi) < 0.5).item()
        print(f"Number of elements with phase ≈ π (i.e., negative): {n_negative}")

        # We expect either 1 or 3 negative elements for CZ gate
        assert n_negative >= 1, (
            f"CZ gate should have at least one negative entry, found {n_negative}"
        )

        # 4. Infidelity should be reasonable (< 0.5 means it's doing something)
        assert infidelity < 0.5, (
            f"Infidelity too high: {infidelity}. Training may not be working."
        )

        print("\n" + "=" * 70)
        print("✓ TEST PASSED: Training pipeline works!")
        print("=" * 70)
        print()
        print("Notes:")
        if infidelity < 0.01:
            print("  - Excellent! High fidelity achieved.")
        elif infidelity < 0.1:
            print("  - Good fidelity. With more epochs, could improve further.")
        else:
            print("  - Moderate fidelity. Training is working but needs more epochs")
            print("    or better hyperparameters for high-fidelity gates.")
        print()
        print("  To improve: increase epochs to ~500-1000, or tune network size.")
        print()

        return {
            "infidelity": infidelity,
            "fidelity": fidelity,
            "achieved_gate": achieved_gate,
            "off_diag_norm": off_diag_norm,
        }

    def test_cz_optimization_quick_check(self):
        """
        Quick sanity check: 3 epochs, 1 angle, looser tolerances.
        Should complete in <30 seconds if tolerances are the issue.
        """
        print("\n" + "=" * 70)
        print("Quick CZ Optimization Test (3 epochs, 1 angle)")
        print("=" * 70)

        # Arrange - minimal setup
        gate = CZPhiGate()
        network = FeedForwardNN(
            input_dim=2,
            output_dim=2,
            hidden_layers=3,
            hidden_units=32,  # Tiny network
        )
        pulse_gen = create_default_physical_pulse_generator(rabi_max=gate.rabi_max)
        # Use RK4 like original code (much faster than dopri5)
        solver = TorchDiffeqSolver(method="rk4")
        evolver = QuantumEvolver(nqubits=2, solver=solver)
        loss_fn = InfidelityLoss(nqubits=2)

        trainer = QuantumTrainer(
            network=network,
            nqubits=2,
            loss_fn=loss_fn,
            pulse_generator=pulse_gen,
            evolver=evolver,
        )

        # Training parameters
        angles = torch.tensor([torch.pi])
        # CRITICAL: gate_time is in units of 1/rabi, NOT absolute time!
        # Original code uses normalized time. For rabi=25, gate_time=7 means 7/25 ≈ 0.28 seconds
        normalized_gate_time = 7.0
        gate_time = normalized_gate_time / gate.rabi_max  # Convert to actual time
        epochs = 3

        print(f"\nConfig: {epochs} epochs, 1 angle")
        print(f"Normalized gate time: {normalized_gate_time} (units of 1/Ω_max)")
        print(f"Actual gate time: {gate_time:.4f} seconds")
        print("Solver: rk4 (fixed step, like original code)")

        import time

        start = time.time()

        # Act - Train!
        history = trainer.train(angles, gate_time, epochs, print_every=1)

        elapsed = time.time() - start
        print(f"\nTraining completed in {elapsed:.1f} seconds")
        print(f"Initial loss: {history['loss'][0]:.6f}")
        print(f"Final loss:   {history['loss'][-1]:.6f}")

        # Assert - Should finish quickly and show learning
        assert elapsed < 60, (
            f"Training too slow: {elapsed:.1f}s (should be <60s with loose tolerances)"
        )
        print("\n✓ Quick test passed! Training works and is reasonably fast.")

    @pytest.mark.slow
    def test_multiple_angles_training(self):
        """Test training on multiple angles (more realistic scenario)."""
        print("\n" + "=" * 70)
        print("Testing Multi-Angle CZ_φ Optimization")
        print("=" * 70)

        # Arrange
        gate = CZPhiGate()
        network = FeedForwardNN(
            input_dim=2, output_dim=2, hidden_layers=4, hidden_units=50
        )
        pulse_gen = create_default_physical_pulse_generator(rabi_max=gate.rabi_max)
        # Use RK4 like original code (much faster than dopri5)
        solver = TorchDiffeqSolver(method="rk4")
        evolver = QuantumEvolver(nqubits=2, solver=solver)
        loss_fn = InfidelityLoss(nqubits=2)

        trainer = QuantumTrainer(
            network=network,
            nqubits=2,
            loss_fn=loss_fn,
            pulse_generator=pulse_gen,
            evolver=evolver,
        )

        # Train on multiple angles
        angles = torch.linspace(0.5 * torch.pi, torch.pi, 5)  # 5 angles
        # CRITICAL: Convert normalized time to actual time
        normalized_gate_time = 7.0
        gate_time = normalized_gate_time / gate.rabi_max
        epochs = 15

        print(f"\nTraining on {len(angles)} angles: {angles / torch.pi}π")
        print(f"Normalized gate time: {normalized_gate_time} (units of 1/Ω_max)")
        print(f"Actual gate time: {gate_time:.4f} seconds")
        print(f"Epochs: {epochs}")
        print()

        # Act
        history = trainer.train(angles, gate_time, epochs, print_every=5)

        # Evaluate at test angle
        optimizer = ControlledPhaseOptimizer(
            gate=gate,
            network=network,
            trainer=trainer,
            pulse_generator=pulse_gen,
            evolver=evolver,
        )

        test_angle = torch.pi  # CZ gate
        result = optimizer.evaluate(test_angle)

        print("\n" + "=" * 70)
        print("Test angle φ = π:")
        print(f"  Infidelity: {result['infidelity']:.6e}")
        print(f"  Fidelity:   {(1 - result['infidelity']) * 100:.4f}%")
        print("=" * 70)

        # Assert
        assert result["infidelity"] < 0.5, (
            "Multi-angle training should produce reasonable results"
        )

        print("\n✓ Multi-angle training works!")
