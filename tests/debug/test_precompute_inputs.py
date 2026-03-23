"""
Test: Precompute ALL inputs, not just time_grid
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/madhav22m/gitrepos/qneural')

from qneural.gates.rydberg import CZPhiGate
from qneural.neural import FeedForwardNN, TorchDiffeqSolver, QuantumEvolver, InfidelityLoss
from qneural.neural.pulse_generator import PhysicalPulseGenerator
from qneural.core.metrics import unitary_infidelity
from qneural.core.gates import czphi_gate

gate = CZPhiGate()
rabi_max = gate.rabi_max
gate_time = 7.62 / rabi_max
angle = torch.tensor([np.pi])
target_U = gate.get_target_unitary(angle)

pulse_gen = PhysicalPulseGenerator(
    n_controls=1, n_time_steps=101,
    control_ranges=[(-2*rabi_max, 2*rabi_max)]
)

def rabi_pulse(t): return torch.tensor(rabi_max)
def make_detuning_fn(values, gt):
    def fn(t):
        idx = int(t / gt * (len(values) - 1))
        idx = min(idx, len(values) - 1)
        return values[idx]
    return fn

solver = TorchDiffeqSolver(method='rk4')
evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=101)

loss_fn = InfidelityLoss(nqubits=2)

print("="*60)
print("TEST 1: Precompute ALL inputs once (like working test)")
print("="*60)

network1 = FeedForwardNN(
    input_dim=2, output_dim=1,
    hidden_layers=6, hidden_units=150,
    activation='relu', output_activation='sigmoid',
    use_batch_norm=True, weight_scale=1.8
)

optimizer1 = torch.optim.Adam(network1.parameters(), lr=1e-4)

# Precompute inputs ONCE
n_steps = 101
time_grid = torch.linspace(0, 1, n_steps)
angles_repeated = angle.repeat_interleave(n_steps)
time_repeated = time_grid.repeat(1)
inputs_precomputed = torch.stack([angles_repeated, time_repeated], dim=1)

for epoch in range(100):
    optimizer1.zero_grad()
    network1.train()
    
    # Use precomputed inputs
    nn_outputs = network1(inputs_precomputed)
    nn_outputs = nn_outputs.reshape(1, n_steps, -1)
    
    detuning_values = pulse_gen.scale_output(nn_outputs[0, :, 0], 0)
    detuning_fn = make_detuning_fn(detuning_values, gate_time)
    
    final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time, apply_corrections=True)
    
    from qneural.core.metrics import unitary_fidelity
    loss = 1 - unitary_fidelity(final_U, target_U, dim=2, nqubits=2)
    
    loss.backward()
    optimizer1.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:4d}: loss={loss.item():.6f}")

print(f"\nFinal: {(1-loss.item())*100:.2f}% fidelity")

print("\n" + "="*60)
print("TEST 2: Recreate inputs every epoch (like _train_step)")
print("="*60)

network2 = FeedForwardNN(
    input_dim=2, output_dim=1,
    hidden_layers=6, hidden_units=150,
    activation='relu', output_activation='sigmoid',
    use_batch_norm=True, weight_scale=1.8
)

optimizer2 = torch.optim.Adam(network2.parameters(), lr=1e-4)

for epoch in range(100):
    optimizer2.zero_grad()
    network2.train()
    
    # RECREATE inputs every epoch
    time_grid = torch.linspace(0, 1, n_steps)
    angles_repeated = angle.repeat_interleave(n_steps)
    time_repeated = time_grid.repeat(1)
    inputs = torch.stack([angles_repeated, time_repeated], dim=1)
    
    nn_outputs = network2(inputs)
    nn_outputs = nn_outputs.reshape(1, n_steps, -1)
    
    detuning_values = pulse_gen.scale_output(nn_outputs[0, :, 0], 0)
    detuning_fn = make_detuning_fn(detuning_values, gate_time)
    
    final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time, apply_corrections=True)
    
    loss = 1 - unitary_fidelity(final_U, target_U, dim=2, nqubits=2)
    
    loss.backward()
    optimizer2.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:4d}: loss={loss.item():.6f}")

print(f"\nFinal: {(1-loss.item())*100:.2f}% fidelity")
