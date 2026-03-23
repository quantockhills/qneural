"""
Debug training - see why it's stuck
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

pulse_gen = PhysicalPulseGenerator(n_controls=1, n_time_steps=101,
    control_ranges=[(-2*rabi_max, 2*rabi_max)])

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

time_grid = torch.linspace(0, 1, 101)
inputs = torch.stack([angle.repeat(101), time_grid], dim=1)

print("Training first 10 epochs with debug:\n")

for epoch in range(10):
    optimizer.zero_grad()
    
    detuning_out = network(inputs).reshape(101)
    detuning_vals = pulse_gen.scale_output(detuning_out, 0)
    detuning_fn = make_detuning_fn(detuning_vals, gate_time)
    final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time)
    loss = loss_fn(final_U, target_U)
    
    loss.backward()
    
    # Debug info
    grad_norm = sum(p.grad.norm().item() for p in network.parameters() if p.grad is not None)
    detuning_range = (detuning_vals.max() - detuning_vals.min()).item()
    
    optimizer.step()
    
    print(f"Epoch {epoch}: Loss={loss.item():.6f}, "
          f"GradNorm={grad_norm:.2f}, "
          f"DetuningRange={detuning_range:.1f}MHz")

# Check output range evolution
with torch.no_grad():
    final_out = network(inputs).reshape(101)
    print(f"\nFinal network output: [{final_out.min():.3f}, {final_out.max():.3f}]")
    print(f"Initial was: [0.017, 0.871] (from debug_cz.py)")
    print(f"Range change: {(final_out.max() - final_out.min() - 0.854):.3f}")
