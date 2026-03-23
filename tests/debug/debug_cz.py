"""
Debug CZ Training
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

print(f"Rabi max: {rabi_max}")
print(f"Gate time: {gate_time}")

network = FeedForwardNN(
    input_dim=2, output_dim=1,
    hidden_layers=6, hidden_units=150,
    activation='relu', output_activation='sigmoid',
    use_batch_norm=True, weight_scale=1.8
)

# Check network output range
angle = torch.tensor([np.pi])
time_grid = torch.linspace(0, 1, 101)
inputs = torch.stack([angle.repeat(101), time_grid], dim=1)

with torch.no_grad():
    test_out = network(inputs).reshape(101)
    print(f"\nNetwork output range: [{test_out.min():.3f}, {test_out.max():.3f}]")
    print(f"Has grad: {test_out.requires_grad}")

pulse_gen = PhysicalPulseGenerator(
    n_controls=1, n_time_steps=101,
    control_ranges=[(-2*rabi_max, 2*rabi_max)]
)

# Check scaled detuning
detuning_vals = pulse_gen.scale_output(test_out, 0)
print(f"Detuning range: [{detuning_vals.min():.1f}, {detuning_vals.max():.1f}] MHz")

def rabi_pulse(t): return torch.tensor(rabi_max)

def make_detuning_fn(values, gate_time):
    def fn(t):
        idx = int(t / gate_time * (len(values) - 1))
        return values[min(idx, len(values) - 1)]
    return fn

solver = TorchDiffeqSolver(method='rk4')
evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=101)

# Test evolution
detuning_fn = make_detuning_fn(detuning_vals, gate_time)
final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time)

print(f"\nFinal unitary shape: {final_U.shape}")
print(f"Is unitary: {torch.allclose(final_U @ final_U.conj().T, torch.eye(4, dtype=torch.cfloat), atol=1e-2)}")

target_U = gate.get_target_unitary(angle)
print(f"Target unitary diagonal: {torch.diag(target_U)}")

# Check loss
loss_fn = InfidelityLoss(nqubits=2)
loss = loss_fn(final_U, target_U)
print(f"\nInitial loss: {loss.item():.6f}")

# Test gradient flow
loss.backward()
has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters())
print(f"Gradients computed: {has_grad}")

if has_grad:
    grad_norm = sum(p.grad.norm().item() for p in network.parameters() if p.grad is not None)
    print(f"Gradient norm: {grad_norm:.6f}")
