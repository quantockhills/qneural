"""
Try different gate times
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
angle = torch.tensor([np.pi])
target_U = gate.get_target_unitary(angle)

# Test different gate times
gate_times = [5.0, 7.0, 7.62, 10.0, 12.0]

for gt_norm in gate_times:
    gate_time = gt_norm / rabi_max
    
    network = FeedForwardNN(input_dim=2, output_dim=1,
        hidden_layers=6, hidden_units=150,
        activation='relu', output_activation='sigmoid',
        use_batch_norm=True, weight_scale=1.8)
    
    pulse_gen = PhysicalPulseGenerator(n_controls=1, n_time_steps=101,
        control_ranges=[(-2*rabi_max, 2*rabi_max)])
    
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
    
    # Quick train (200 epochs)
    for epoch in range(200):
        optimizer.zero_grad()
        detuning_out = network(inputs).reshape(101)
        detuning_vals = pulse_gen.scale_output(detuning_out, 0)
        detuning_fn = make_detuning_fn(detuning_vals, gate_time)
        final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time)
        loss = loss_fn(final_U, target_U)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        fidelity = unitary_fidelity(final_U, target_U, dim=2, nqubits=2)
    
    print(f"Gate time {gt_norm:4.1f}/Ω_max: Infidelity={1-fidelity:.4f}, Fidelity={fidelity*100:.1f}%")
