"""
Quick Architecture Test - Large Network (6×150)

This uses the same architecture as the published paper:
6 hidden layers × 150 units
"""

import torch
import numpy as np
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qneural.gates.rydberg import CZPhiGate, ControlledPhaseOptimizer
from qneural.neural import (
    FeedForwardNN,
    create_default_physical_pulse_generator,
    QuantumEvolver,
    TorchDiffeqSolver,
    QuantumTrainer,
    InfidelityLoss
)
from qneural.analysis import plot_loss_convergence, plot_pulses_vs_time


def train_large_network():
    """Train with 6×150 architecture (paper configuration)."""
    
    print("="*70)
    print("Large Network Training: 6 layers × 150 units")
    print("(Matches published paper architecture)")
    print("="*70)
    
    # Configuration
    gate = CZPhiGate()
    rabi_max = gate.rabi_max
    normalized_gate_time = 10.0
    actual_gate_time = normalized_gate_time / rabi_max
    target_angle = torch.pi
    epochs = 1000  # More epochs for large network
    
    print(f"\nSetup:")
    print(f"  Architecture: 6 layers × 150 units")
    print(f"  Epochs: {epochs}")
    print(f"  Gate time: {normalized_gate_time}/Ω_max")
    
    # Create large network
    network = FeedForwardNN(
        input_dim=2,
        output_dim=2,
        hidden_layers=6,
        hidden_units=150,
        activation='relu',
        use_batchnorm=True
    )
    
    n_params = sum(p.numel() for p in network.parameters())
    print(f"  Total parameters: {n_params:,}")
    
    # Training setup
    pulse_gen = create_default_physical_pulse_generator(rabi_max=rabi_max)
    solver = TorchDiffeqSolver(method='rk4')
    evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=201)
    loss_fn = InfidelityLoss(nqubits=2)
    
    # Optimizer with good settings for large network
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    
    trainer = QuantumTrainer(
        network=network,
        nqubits=2,
        loss_fn=loss_fn,
        pulse_generator=pulse_gen,
        evolver=evolver,
        optimizer=optimizer
    )
    
    # Train
    angles = torch.tensor([target_angle])
    history = {'loss': []}
    
    print(f"\nTraining... (this will take ~5-6 minutes)")
    start = time.time()
    
    for epoch in range(epochs):
        epoch_loss = trainer._train_step(angles, actual_gate_time)
        history['loss'].append(epoch_loss)
        scheduler.step()
        
        if epoch % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d}: Loss = {epoch_loss:.6f}  (LR: {current_lr:.2e})")
    
    elapsed = time.time() - start
    
    print(f"\n{'='*70}")
    print(f"Training Complete! ({elapsed/60:.1f} minutes)")
    print(f"{'='*70}")
    print(f"Initial loss: {history['loss'][0]:.6f}")
    print(f"Final loss: {history['loss'][-1]:.6f}")
    print(f"Loss reduction: {(1 - history['loss'][-1]/history['loss'][0])*100:.1f}%")
    
    # Evaluate
    optimizer_obj = ControlledPhaseOptimizer(
        gate=gate, network=network, trainer=trainer,
        pulse_generator=pulse_gen, evolver=evolver,
        time_optimal=False
    )
    
    result = optimizer_obj.evaluate(target_angle)
    infidelity = result['infidelity']
    fidelity = 1 - infidelity
    
    print(f"\nResults:")
    print(f"  Fidelity: {fidelity*100:.4f}%")
    print(f"  Infidelity: {infidelity:.2e}")
    
    # Visualizations
    print(f"\nSaving results...")
    plot_loss_convergence(
        history, 
        save_path='cz_large_network_loss.png', 
        show=False,
        title=f'6×150 Network - Fidelity: {fidelity*100:.2f}%'
    )
    
    pulses, _ = optimizer_obj.generate_pulse(target_angle)
    plot_pulses_vs_time(
        pulses, actual_gate_time,
        labels=['Ω/Ω_max', 'Δ/Ω_max'],
        save_path='cz_large_network_pulses.png',
        show=False,
        title='Optimized Pulses (6×150 network)'
    )
    
    # Save model
    torch.save({
        'network_state_dict': network.state_dict(),
        'fidelity': fidelity,
        'architecture': '6x150',
        'epochs': epochs
    }, 'cz_model_6x150.pt')
    
    print(f"  ✓ cz_large_network_loss.png")
    print(f"  ✓ cz_large_network_pulses.png")
    print(f"  ✓ cz_model_6x150.pt (model checkpoint)")
    
    return fidelity


if __name__ == "__main__":
    fidelity = train_large_network()
    
    print(f"\n{'='*70}")
    if fidelity >= 0.99:
        print(f"🎉 EXCELLENT! {fidelity*100:.2f}% fidelity achieved!")
        print(f"   Ready for publication-quality results")
    elif fidelity >= 0.95:
        print(f"✅ GOOD! {fidelity*100:.2f}% fidelity")
        print(f"   Try 2000 epochs for >99%")
    else:
        print(f"⚠ {fidelity*100:.2f}% fidelity")
        print(f"  Try more epochs or different hyperparameters")
    print(f"{'='*70}
")
