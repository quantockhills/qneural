"""
Test: Verify time grid is being reused
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/madhav22m/gitrepos/qneural')

from qneural.gates.rydberg import CZPhiGate
from qneural.neural import FeedForwardNN, TorchDiffeqSolver, QuantumEvolver, FixedRabiTrainer

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

print(f"Initial time_grid: {trainer._time_grid}")

# First call
trainer.train(angles=angle, gate_time=gate_time, epochs=1, print_every=1)
grid_after_first = trainer._time_grid
print(f"\nAfter first epoch: {trainer._time_grid}")
print(f"Same object? {grid_after_first is trainer._time_grid}")

# Second call
trainer.train(angles=angle, gate_time=gate_time, epochs=1, print_every=1)
print(f"\nAfter second epoch: {trainer._time_grid}")
print(f"Same object? {grid_after_first is trainer._time_grid}")

# Now test if the actual training converges
print("\n" + "="*60)
print("Testing convergence with 100 epochs...")
print("="*60)

# Reset network for fresh test
network2 = FeedForwardNN(
    input_dim=2, output_dim=1,
    hidden_layers=6, hidden_units=150,
    activation='relu', output_activation='sigmoid',
    use_batch_norm=True, weight_scale=1.8
)

optimizer2 = torch.optim.Adam(network2.parameters(), lr=1e-4)

trainer2 = FixedRabiTrainer(
    network=network2,
    nqubits=2,
    rabi_max=rabi_max,
    evolver=evolver,
    optimizer=optimizer2
)

history = trainer2.train(angles=angle, gate_time=gate_time, epochs=100, print_every=20)

final_fidelity = (1 - history['loss'][-1]) * 100
print(f"\nFinal: {final_fidelity:.2f}% fidelity")
