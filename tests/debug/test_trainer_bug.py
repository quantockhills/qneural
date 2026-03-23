"""
Test: Compare trainer vs direct loop to find the bug
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/madhav22m/gitrepos/qneural')

from qneural.gates.rydberg import CZPhiGate
from qneural.neural import FeedForwardNN, TorchDiffeqSolver, QuantumEvolver, InfidelityLoss
from qneural.neural.pulse_generator import PhysicalPulseGenerator
from qneural.core.metrics import unitary_fidelity, unitary_infidelity
from qneural.core.gates import czphi_gate

# Setup
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

print("="*60)
print("TEST 1: Direct loop (like test_corrected_compare.py)")
print("="*60)

# Reset network
network1 = FeedForwardNN(
    input_dim=2, output_dim=1,
    hidden_layers=6, hidden_units=150,
    activation='relu', output_activation='sigmoid',
    use_batch_norm=True, weight_scale=1.8
)

time_grid = torch.linspace(0, 1, 101)
inputs = torch.stack([angle.repeat(101), time_grid], dim=1)

optimizer1 = torch.optim.Adam(network1.parameters(), lr=1e-4)

for epoch in range(100):
    optimizer1.zero_grad()
    
    detuning_out = network1(inputs).reshape(101)
    detuning_vals = pulse_gen.scale_output(detuning_out, 0)
    detuning_fn = make_detuning_fn(detuning_vals, gate_time)
    
    final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time, apply_corrections=True)
    
    # Direct loss computation (like test script)
    loss = 1 - unitary_fidelity(final_U, target_U, dim=2, nqubits=2)
    loss.backward()
    optimizer1.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:4d}: Infidelity={loss.item():.6f}")

print(f"\nFinal: {(1-loss.item())*100:.2f}% fidelity")

print("\n" + "="*60)
print("TEST 2: FixedRabiTrainer approach")
print("="*60)

# Reset network
network2 = FeedForwardNN(
    input_dim=2, output_dim=1,
    hidden_layers=6, hidden_units=150,
    activation='relu', output_activation='sigmoid',
    use_batch_norm=True, weight_scale=1.8
)

loss_fn = InfidelityLoss(nqubits=2)
optimizer2 = torch.optim.Adam(network2.parameters(), lr=1e-4)

for epoch in range(100):
    optimizer2.zero_grad()
    network2.train()
    
    # Like FixedRabiTrainer - create inputs fresh each epoch
    n_steps = 101
    time_grid = torch.linspace(0, 1, n_steps)
    angles_repeated = angle.repeat_interleave(n_steps)
    time_repeated = time_grid.repeat(1)
    inputs2 = torch.stack([angles_repeated, time_repeated], dim=1)
    
    # Forward pass
    nn_outputs = network2(inputs2)
    nn_outputs = nn_outputs.reshape(1, n_steps, -1)
    
    # Get detuning values
    detuning_values = pulse_gen.scale_output(nn_outputs[0, :, 0], 0)
    detuning_fn = make_detuning_fn(detuning_values, gate_time)
    
    # Evolve
    final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time, apply_corrections=True)
    
    # Compute loss using loss_fn (like trainer)
    loss = loss_fn(final_U, target_U)
    
    # ALSO compute infidelity for comparison (like trainer does)
    infidelity = unitary_infidelity(final_U, target_U, nqubits=2)
    
    loss.backward()
    optimizer2.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:4d}: loss_fn={loss.item():.6f}, infidelity={infidelity.item():.6f}")

print(f"\nFinal: {(1-loss.item())*100:.2f}% fidelity")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"Direct loop:      {(1-loss.item())*100:.2f}% fidelity")
print(f"Trainer approach: {(1-loss.item())*100:.2f}% fidelity")
