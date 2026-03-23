"""
Test: Compare corrected unitary to target
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
def make_detuning_fn(values, gt):
    def fn(t):
        idx = int(t / gt * (len(values) - 1))
        return values[min(idx, len(values) - 1)]
    return fn

solver = TorchDiffeqSolver(method='rk4')
evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=101)

time_grid = torch.linspace(0, 1, 101)
inputs = torch.stack([angle.repeat(101), time_grid], dim=1)

print("Training - comparing CORRECTED achieved to target...")
print("="*60)

optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

for epoch in range(1000):
    optimizer.zero_grad()
    
    detuning_out = network(inputs).reshape(101)
    detuning_vals = pulse_gen.scale_output(detuning_out, 0)
    detuning_fn = make_detuning_fn(detuning_vals, gate_time)
    
    # Get corrected unitary
    final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time, apply_corrections=True)
    
    # Compare CORRECTED to target
    loss = 1 - unitary_fidelity(final_U, target_U, dim=2, nqubits=2)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        with torch.no_grad():
            infidelity = loss.item()
            fidelity = (1 - infidelity) * 100
        print(f"Epoch {epoch:4d}: Infidelity={infidelity:.6f}, Fidelity={fidelity:.2f}%")

print(f"\nFinal: {100*(1-loss.item()):.2f}% fidelity")
