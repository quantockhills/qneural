"""
Test the bugfixes - should achieve <0.1 infidelity now
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/madhav22m/gitrepos/qneural')

from qneural.gates.rydberg import CZPhiGate
from qneural.neural import FeedForwardNN, TorchDiffeqSolver, QuantumEvolver, InfidelityLoss
from qneural.neural.pulse_generator import PhysicalPulseGenerator
from qneural.core.metrics import unitary_fidelity

print("Testing bugfixes...")
print("="*60)

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
loss_fn = InfidelityLoss(nqubits=2)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

time_grid = torch.linspace(0, 1, 101)
inputs = torch.stack([angle.repeat(101), time_grid], dim=1)

print("Training 1000 epochs...")
best_infidelity = 1.0

for epoch in range(1000):
    optimizer.zero_grad()
    
    detuning_out = network(inputs).reshape(101)
    detuning_vals = pulse_gen.scale_output(detuning_out, 0)
    detuning_fn = make_detuning_fn(detuning_vals, gate_time)
    
    # Test both with and without corrections
    final_U_raw = evolver.evolve([rabi_pulse, detuning_fn], gate_time, apply_corrections=False)
    final_U_corrected = evolver.evolve([rabi_pulse, detuning_fn], gate_time, apply_corrections=True)
    
    # Use raw for training (compare raw to raw target)
    loss = loss_fn(final_U_raw, target_U)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        with torch.no_grad():
            # Evaluate with corrections
            fidelity = unitary_fidelity(final_U_corrected, target_U, dim=2, nqubits=2)
            infidelity = 1 - fidelity
            if infidelity < best_infidelity:
                best_infidelity = infidelity
        print(f"Epoch {epoch:4d}: Loss={loss.item():.6f}, Infidelity={infidelity:.6f}")

print(f"\n{'='*60}")
print(f"Best infidelity: {best_infidelity:.6f}")
print(f"Best fidelity: {(1-best_infidelity)*100:.2f}%")

if best_infidelity < 0.1:
    print("✅ SUCCESS! Bugfixes work!")
else:
    print("⚠️  Still high - need to investigate further")
