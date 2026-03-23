"""
Test with more time steps for ODE solver
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/madhav22m/gitrepos/qneural')

from qneural.gates.rydberg import CZPhiGate
from qneural.neural import FeedForwardNN, TorchDiffeqSolver, QuantumEvolver, InfidelityLoss
from qneural.neural.pulse_generator import PhysicalPulseGenerator
from qneural.core.metrics import unitary_fidelity

gate = CZPhiGate()
rabi_max = gate.rabi_max
gate_time = 7.62 / rabi_max
angle = torch.tensor([np.pi])
target_U = gate.get_target_unitary(angle)

network = FeedForwardNN(input_dim=2, output_dim=1,
    hidden_layers=6, hidden_units=150,
    activation='relu', output_activation='sigmoid',
    use_batch_norm=True, weight_scale=1.8)

# More time steps
n_steps = 401  # Was 101
pulse_gen = PhysicalPulseGenerator(n_controls=1, n_time_steps=n_steps,
    control_ranges=[(-2*rabi_max, 2*rabi_max)])

def rabi_pulse(t): return torch.tensor(rabi_max)

def make_detuning_fn(values, gate_time):
    def fn(t):
        idx = int(t / gate_time * (len(values) - 1))
        return values[min(idx, len(values) - 1)]
    return fn

solver = TorchDiffeqSolver(method='rk4')
evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=n_steps)
loss_fn = InfidelityLoss(nqubits=2)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

time_grid = torch.linspace(0, 1, n_steps)
inputs = torch.stack([angle.repeat(n_steps), time_grid], dim=1)

print(f"Training with {n_steps} time steps...")

for epoch in range(500):
    optimizer.zero_grad()
    
    detuning_out = network(inputs).reshape(n_steps)
    detuning_vals = pulse_gen.scale_output(detuning_out, 0)
    detuning_fn = make_detuning_fn(detuning_vals, gate_time)
    final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time)
    loss = loss_fn(final_U, target_U)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}: Loss={loss.item():.6f}")

with torch.no_grad():
    final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time)
    fidelity = unitary_fidelity(final_U, target_U, dim=2, nqubits=2)

print(f"\nFinal: Infidelity={1-fidelity:.6f}, Fidelity={fidelity*100:.2f}%")
