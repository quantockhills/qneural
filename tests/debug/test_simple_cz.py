"""
Simplified High-Fidelity CZ Test - Single Angle, Fixed Time

Goal: Achieve infidelity < 0.1 (fidelity > 90%)
Approach: Detuning-only optimization with simplified training
"""

import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qneural.gates.rydberg import CZPhiGate
from qneural.neural import FeedForwardNN, TorchDiffeqSolver, QuantumEvolver, InfidelityLoss
from qneural.neural.pulse_generator import PhysicalPulseGenerator
from qneural.core.metrics import unitary_fidelity, unitary_infidelity


def train_simple_cz(epochs=2000, lr=5e-4, print_every=100):
    """Train CZ gate with minimal setup."""
    
    print("="*60)
    print("Simple CZ Training - Detuning Only")
    print("="*60)
    
    # Setup
    gate = CZPhiGate()
    rabi_max = gate.rabi_max
    gate_time = 10.0 / rabi_max
    angle = torch.tensor([np.pi])
    
    print(f"\nConfig:")
    print(f"  Gate time: 10/Ω_max = {gate_time:.4f}s")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    
    # Network with weight scaling
    network = FeedForwardNN(
        input_dim=2,
        output_dim=1,
        hidden_layers=6,
        hidden_units=150,
        activation='relu',
        output_activation='sigmoid',
        use_batch_norm=True,
        weight_scale=1.8  # Critical!
    )
    
    n_params = sum(p.numel() for p in network.parameters())
    print(f"\nNetwork: 6×150 ({n_params:,} params), weight_scale=1.8")
    
    # Components
    pulse_gen = PhysicalPulseGenerator(
        n_controls=1,
        n_time_steps=201,
        control_ranges=[(-50.0, 50.0)]
    )
    
    def rabi_pulse(t):
        return torch.tensor(rabi_max)
    
    def make_detuning_fn(values, gate_time):
        def fn(t):
            idx = int(t / gate_time * (len(values) - 1))
            idx = min(idx, len(values) - 1)
            return values[idx]
        return fn
    
    solver = TorchDiffeqSolver(method='rk4')
    evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=201)
    loss_fn = InfidelityLoss(nqubits=2)
    
    # Optimizer - simple Adam, no scheduler
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    
    # Training
    print(f"\nTraining...")
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward
        n_steps = 201
        time_grid = torch.linspace(0, 1, n_steps)
        inputs = torch.stack([angle.repeat(n_steps), time_grid], dim=1)
        
        detuning_out = network(inputs).reshape(n_steps)
        detuning_vals = pulse_gen.scale_output(detuning_out, 0)
        
        detuning_fn = make_detuning_fn(detuning_vals, gate_time)
        pulses = [rabi_pulse, detuning_fn]
        
        # Evolve
        final_U = evolver.evolve(pulses, gate_time)
        target_U = gate.get_target_unitary(angle)
        
        # Loss
        loss = loss_fn(final_U, target_U)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % print_every == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss.item():.6f}")
    
    # Final evaluation
    with torch.no_grad():
        detuning_out = network(inputs).reshape(n_steps)
        detuning_vals = pulse_gen.scale_output(detuning_out, 0)
        detuning_fn = make_detuning_fn(detuning_vals, gate_time)
        pulses = [rabi_pulse, detuning_fn]
        final_U = evolver.evolve(pulses, gate_time)
        fidelity = unitary_fidelity(final_U, target_U, dim=2, nqubits=2)
        infidelity = 1 - fidelity
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Final infidelity: {infidelity:.6f}")
    print(f"  Final fidelity: {fidelity*100:.2f}%")
    print(f"{'='*60}")
    
    if infidelity < 0.1:
        print(f"\n🎉 SUCCESS! Infidelity < 0.1")
        return True
    elif infidelity < 0.5:
        print(f"\n✓ Progress: Infidelity = {infidelity:.3f}")
        print(f"  Try more epochs or different LR")
        return False
    else:
        print(f"\n⚠ Poor result: Infidelity = {infidelity:.3f}")
        print(f"  Need to debug architecture/training")
        return False


if __name__ == "__main__":
    import time
    start = time.time()
    
    # Try different learning rates
    learning_rates = [1e-3, 5e-4, 1e-4]
    
    for lr in learning_rates:
        print(f"\n\n{'='*60}")
        print(f"Testing with LR = {lr}")
        print(f"{'='*60}")
        success = train_simple_cz(epochs=500, lr=lr, print_every=50)
        if success:
            break
    
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
