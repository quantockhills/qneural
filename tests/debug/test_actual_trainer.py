"""
Test: Actual FixedRabiTrainer.train() method
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/madhav22m/gitrepos/qneural')

from qneural.gates.rydberg import CZPhiGate
from qneural.neural import FeedForwardNN, TorchDiffeqSolver, QuantumEvolver, InfidelityLoss, FixedRabiTrainer
from qneural.neural.pulse_generator import PhysicalPulseGenerator

gate = CZPhiGate()
rabi_max = gate.rabi_max
gate_time = 7.62 / rabi_max
angle = torch.tensor([np.pi])

network = FeedForwardNN(
    input_dim=2, output_dim=1,
    hidden_layers=6, hidden_units=150,
    activation='relu', output_activation='sigmoid',
    use_batch_norm=True, weight_scale=1.8
)

solver = TorchDiffeqSolver(method='rk4')
evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=101)

optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

trainer = FixedRabiTrainer(
    network=network,
    nqubits=2,
    rabi_max=rabi_max,
    evolver=evolver,
    optimizer=optimizer
)

print("Testing FixedRabiTrainer.train() method...")
print("="*60)

history = trainer.train(
    angles=angle,
    gate_time=gate_time,
    epochs=100,
    print_every=20
)

final_fidelity = (1 - history['loss'][-1]) * 100
print(f"\nFinal: {final_fidelity:.2f}% fidelity")

print("\nFull loss history (last 10):")
for i in range(max(0, len(history['loss'])-10), len(history['loss'])):
    print(f"  Epoch {i}: loss={history['loss'][i]:.6f}, infidelity={history['infidelity'][i]:.6f}")
