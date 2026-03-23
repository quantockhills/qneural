"""
Minimal CZ Test - Match Original Parameters

Key settings from original code:
- LR: 1e-4
- Steps: 101
- Detuning range: [-2*Ω_max, 2*Ω_max]
- Gate time: 7.62/Ω_max
- Epochs: 500
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

network = FeedForwardNN(
    input_dim=2, output_dim=1,
    hidden_layers=6, hidden_units=150,
    activation='relu', output_activation='sigmoid',
    use_batch_norm=True, weight_scale=1.8
)

pulse_gen = PhysicalPulseGenerator(
    n_controls=1, n_time_steps=101,
    control_ranges=[(-2*rabi_max, 2*rabi_max)]
)

def rabi_pulse(t): return torch.tensor(rabi_max)

def make_detuning_fn(values, gate_time):
    def fn(t):
        idx = int(t / gate_time * (len(values) - 1))
        return values[min(idx, len(values) - 1)]
    return fn

solver = TorchDiffeqSolver(method='rk4')
evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=101)
loss_fn = InfidelityLoss(nqubits=2)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

angle = torch.tensor([np.pi])
target_U = gate.get_target_unitary(angle)

print("Training 500 epochs...")
losses = []

for epoch in range(500):
    optimizer.zero_grad()
    
    time_grid = torch.linspace(0, 1, 101)
    inputs = torch.stack([angle.repeat(101), time_grid], dim=1)
    
    detuning_out = network(inputs).reshape(101)
    detuning_vals = pulse_gen.scale_output(detuning_out, 0)
    
    detuning_fn = make_detuning_fn(detuning_vals, gate_time)
    final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time)
    
    loss = loss_fn(final_U, target_U)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}: {loss.item():.6f}")

# Final eval
with torch.no_grad():
    detuning_out = network(inputs).reshape(101)
    detuning_vals = pulse_gen.scale_output(detuning_out, 0)
    detuning_fn = make_detuning_fn(detuning_vals, gate_time)
    final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time)
    fidelity = unitary_fidelity(final_U, target_U, dim=2, nqubits=2)

print(f"\nFinal: Loss={losses[-1]:.6f}, Infidelity={1-fidelity:.6f}, Fidelity={fidelity*100:.2f}%")

if 1-fidelity < 0.1:
    print("✓ SUCCESS!")
else:
    print("Need more training...")
