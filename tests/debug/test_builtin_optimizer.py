"""
Test using built-in ControlledPhaseOptimizer
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/madhav22m/gitrepos/qneural')

from qneural.gates.rydberg import CZPhiGate, ControlledPhaseOptimizer
from qneural.neural import (
    FeedForwardNN, create_default_physical_pulse_generator,
    QuantumEvolver, TorchDiffeqSolver, QuantumTrainer, InfidelityLoss
)

print("Testing with built-in ControlledPhaseOptimizer")
print("="*60)

# Setup
gate = CZPhiGate()
rabi_max = gate.rabi_max

# Create network with correct settings
network = FeedForwardNN(
    input_dim=2, output_dim=1,  # Only detuning
    hidden_layers=6, hidden_units=150,
    activation='relu', output_activation='sigmoid',
    use_batch_norm=True, weight_scale=1.8
)

# Create optimizer
optimizer = ControlledPhaseOptimizer(
    gate=gate,
    network=network,
    pulse_generator=create_default_physical_pulse_generator(rabi_max=rabi_max),
    evolver=QuantumEvolver(nqubits=2, solver=TorchDiffeqSolver(method='rk4')),
    loss_fn=InfidelityLoss(nqubits=2),
    time_optimal=False
)

print("Training 500 epochs using built-in optimizer...")

# Train
angle = torch.tensor([np.pi])
gate_time = 7.62 / rabi_max  # Normalized

history = optimizer.train(
    angles=angle,
    gate_time=gate_time,
    epochs=500,
    learning_rate=1e-4,
    print_every=50
)

print(f"\nFinal loss: {history['loss'][-1]:.6f}")

# Evaluate
result = optimizer.evaluate(angle)
print(f"Infidelity: {result['infidelity']:.6f}")
print(f"Fidelity: {(1-result['infidelity'])*100:.2f}%")
