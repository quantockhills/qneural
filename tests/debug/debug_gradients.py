"""
Debug gradient flow
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/madhav22m/gitrepos/qneural')

from qneural.gates.rydberg import CZPhiGate
from qneural.neural import FeedForwardNN, TorchDiffeqSolver, QuantumEvolver, InfidelityLoss
from qneural.neural.pulse_generator import PhysicalPulseGenerator

gate = CZPhiGate()
rabi_max = gate.rabi_max
gate_time = 7.62 / rabi_max
angle = torch.tensor([np.pi])

time_grid = torch.linspace(0, 1, 101)
inputs = torch.stack([angle.repeat(101), time_grid], dim=1)

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
target_U = gate.get_target_unitary(angle)

# Check 1: Network output has grad?
detuning_out = network(inputs).reshape(101)
print(f"1. Network output requires_grad: {detuning_out.requires_grad}")

# Check 2: After scaling
detuning_vals = pulse_gen.scale_output(detuning_out, 0)
print(f"2. Scaled values requires_grad: {detuning_vals.requires_grad}")

# Check 3: After making function
detuning_fn = make_detuning_fn(detuning_vals, gate_time)
test_val = detuning_fn(gate_time / 2)  # Test at middle
print(f"3. Function output type: {type(test_val)}")
if torch.is_tensor(test_val):
    print(f"   Function output requires_grad: {test_val.requires_grad}")

# Check 4: After evolution
final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time)
print(f"4. Final unitary requires_grad: {final_U.requires_grad}")

# Check 5: Loss
loss = loss_fn(final_U, target_U)
print(f"5. Loss requires_grad: {loss.requires_grad}")
print(f"   Loss value: {loss.item():.6f}")

# Try backward
if loss.requires_grad:
    loss.backward()
    print("6. Backward pass succeeded!")
    
    # Check gradients
    grad_norm = sum(p.grad.norm().item() for p in network.parameters() if p.grad is not None)
    print(f"   Gradient norm: {grad_norm:.6f}")
else:
    print("6. ERROR: Loss doesn't require grad!")
