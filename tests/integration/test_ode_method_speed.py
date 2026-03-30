"""
Test to verify that ODE solver method affects training speed.

Original code uses rk4 (fast, fixed-step).
New code uses dopri5 (slow, adaptive with tight tolerances).
"""

import pytest
import torch
import time
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
def test_rk4_vs_dopri5_speed():
    """Compare training speed with rk4 vs dopri5."""
    print("\n" + "="*60)
    print("ODE METHOD SPEED COMPARISON")
    print("="*60)

    # Setup
    gate = CZPhiGate()
    network = FeedForwardNN(
        input_dim=2,
        output_dim=2,
        hidden_layers=2,
        hidden_units=10
    )
    pulse_gen = create_default_physical_pulse_generator(rabi_max=gate.rabi_max)
    loss_fn = InfidelityLoss(nqubits=2)

    angles = torch.tensor([torch.pi])
    gate_time = 5.0
    epochs = 2  # Just 2 epochs to compare speed

    # Test with dopri5 (default)
    print("\n1. Testing with dopri5 (default)...")
    evolver_dopri5 = create_evolver(nqubits=2)  # Uses dopri5 by default
    trainer_dopri5 = QuantumTrainer(
        network=network,
        nqubits=2,
        loss_fn=loss_fn,
        pulse_generator=pulse_gen,
        evolver=evolver_dopri5
    )

    start = time.time()
    history_dopri5 = trainer_dopri5.train(angles, gate_time, epochs, print_every=1)
    time_dopri5 = time.time() - start

    print(f"   Time with dopri5: {time_dopri5:.2f}s")

    # Test with rk4 (original method)
    print("\n2. Testing with rk4 (original method)...")
    from qneural.core.solvers import TorchDiffeqSolver

    evolver_rk4 = create_evolver(nqubits=2)
    evolver_rk4.solver = TorchDiffeqSolver(method='rk4')

    # Reset network weights to be fair
    network_rk4 = FeedForwardNN(
        input_dim=2,
        output_dim=2,
        hidden_layers=2,
        hidden_units=10
    )

    trainer_rk4 = QuantumTrainer(
        network=network_rk4,
        nqubits=2,
        loss_fn=loss_fn,
        pulse_generator=pulse_gen,
        evolver=evolver_rk4
    )

    start = time.time()
    history_rk4 = trainer_rk4.train(angles, gate_time, epochs, print_every=1)
    time_rk4 = time.time() - start

    print(f"   Time with rk4:    {time_rk4:.2f}s")

    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"dopri5 time: {time_dopri5:.2f}s")
    print(f"rk4 time:    {time_rk4:.2f}s")
    print(f"Speedup:     {time_dopri5 / time_rk4:.1f}x")
    print()

    if time_rk4 < time_dopri5 * 0.5:
        print("✓ rk4 is significantly faster!")
    else:
        print("⚠ No significant speed difference")

    print("="*60)
