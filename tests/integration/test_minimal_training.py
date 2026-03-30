"""
Ultra-minimal training test: Can training run at all?

Just verify that:
1. We can create all components
2. Training runs for 3 epochs without crashing
3. We get some output (even if fidelity is poor)
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from qneural.gates.rydberg import CZPhiGate
from qneural.neural import (
    FeedForwardNN,
    create_default_physical_pulse_generator,
    create_evolver,
    QuantumTrainer,
    InfidelityLoss
)


@pytest.mark.slow
@pytest.mark.integration
def test_training_runs_for_three_epochs():
    """Ultra-minimal: Just verify training can run for 3 epochs."""
    print("\n" + "="*60)
    print("MINIMAL TRAINING TEST")
    print("="*60)

    # Arrange - simplest possible setup
    gate = CZPhiGate()
    network = FeedForwardNN(
        input_dim=2,
        output_dim=2,
        hidden_layers=2,    # Tiny network
        hidden_units=10     # Very small
    )
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
    angles = torch.tensor([torch.pi])  # Just ONE angle
    gate_time = 5.0
    epochs = 3  # Just THREE epochs

    print(f"\nConfiguration:")
    print(f"  Angles:    1 (φ = π)")
    print(f"  Gate time: {gate_time}")
    print(f"  Epochs:    {epochs}")
    print(f"  Network:   2 layers × 10 units (tiny)")
    print()

    # Act
    print("Training...")
    history = trainer.train(angles, gate_time, epochs, print_every=1)

    # Assert - just check it completed
    assert 'loss' in history
    assert len(history['loss']) == epochs

    print()
    print("="*60)
    print("✓ TRAINING COMPLETED!")
    print("="*60)
    print(f"Epoch losses: {[f'{l:.4f}' for l in history['loss']]}")
    print()
    print("This confirms the full pipeline works:")
    print("  NN → Pulses → Hamiltonian → ODE → Loss → Backprop ✓")
    print()
