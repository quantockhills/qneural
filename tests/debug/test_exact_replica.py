"""
Test: Direct comparison of _train_step logic vs manual loop
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
print("EXACT replica of FixedRabiTrainer._train_step()")
print("="*60)

# Reset network
network = FeedForwardNN(
    input_dim=2, output_dim=1,
    hidden_layers=6, hidden_units=150,
    activation='relu', output_activation='sigmoid',
    use_batch_norm=True, weight_scale=1.8
)

optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
device = 'cpu'

for epoch in range(100):
    network.train()
    optimizer.zero_grad()
    
    # EXACT copy from FixedRabiTrainer._train_step
    n_angles = len(angle)
    n_steps = pulse_gen.n_time_steps
    time_grid = torch.linspace(0, 1, n_steps, device=device)
    
    angles_repeated = angle.repeat_interleave(n_steps)
    time_repeated = time_grid.repeat(n_angles)
    inputs = torch.stack([angles_repeated, time_repeated], dim=1)
    
    nn_outputs = network(inputs)
    nn_outputs = nn_outputs.reshape(n_angles, n_steps, -1)
    
    def rabi_pulse(t):
        return torch.tensor(rabi_max, device=device)
    
    total_loss = torch.tensor(0.0, device=device)
    total_infidelity = torch.tensor(0.0, device=device)
    
    for i, ang in enumerate(angle):
        detuning_values = pulse_gen.scale_output(nn_outputs[i, :, 0], 0)
        
        def make_detuning_fn(values, gt):
            def fn(t):
                idx = int(t / gt * (len(values) - 1))
                idx = min(idx, len(values) - 1)
                return values[idx]
            return fn
        
        detuning_fn = make_detuning_fn(detuning_values, gate_time)
        pulses = [rabi_pulse, detuning_fn]
        
        final_unitary = evolver.evolve(pulses, gate_time, apply_corrections=True)
        target_unitary = czphi_gate(ang.item())
        
        infidelity = unitary_infidelity(final_unitary, target_unitary, nqubits=2)
        loss = loss_fn(final_unitary, target_unitary)
        
        total_loss += loss
        total_infidelity += infidelity
    
    avg_loss = total_loss / n_angles
    avg_infidelity = total_infidelity / n_angles
    
    avg_loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:4d}: loss={avg_loss.item():.6f}, infidelity={avg_infidelity.item():.6f}")

print(f"\nFinal: {(1-avg_loss.item())*100:.2f}% fidelity")

# Compare with simple loop
print("\n" + "="*60)
print("Simple loop (no angle loop, single angle)")
print("="*60)

network2 = FeedForwardNN(
    input_dim=2, output_dim=1,
    hidden_layers=6, hidden_units=150,
    activation='relu', output_activation='sigmoid',
    use_batch_norm=True, weight_scale=1.8
)

optimizer2 = torch.optim.Adam(network2.parameters(), lr=1e-4)
time_grid = torch.linspace(0, 1, 101)
inputs = torch.stack([angle.repeat(101), time_grid], dim=1)
target_U = gate.get_target_unitary(angle)

for epoch in range(100):
    optimizer2.zero_grad()
    
    detuning_out = network2(inputs).reshape(101)
    detuning_vals = pulse_gen.scale_output(detuning_out, 0)
    detuning_fn = make_detuning_fn(detuning_vals, gate_time)
    
    from qneural.core.metrics import unitary_fidelity
    final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time, apply_corrections=True)
    loss = 1 - unitary_fidelity(final_U, target_U, dim=2, nqubits=2)
    
    loss.backward()
    optimizer2.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:4d}: loss={loss.item():.6f}")

print(f"\nFinal: {(1-loss.item())*100:.2f}% fidelity")
