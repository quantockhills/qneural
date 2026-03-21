"""
Example: High-Fidelity CZ Gate Training - Extended Training Version

This version trains for 2000 epochs to achieve high fidelity (>99%).
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


def train_high_fidelity_cz_extended():
    """Train CZ gate with extended training for high fidelity."""
    
    print("="*70)
    print("High-Fidelity CZ Gate Training - Extended (2000 epochs)")
    print("="*70)
    
    # Configuration
    target_angle = torch.pi
    normalized_gate_time = 10.0
    epochs = 2000
    
    gate = CZPhiGate()
    rabi_max = gate.rabi_max
    actual_gate_time = normalized_gate_time / rabi_max
    
    print(f"\nConfiguration:")
    print(f"  Target: CZ gate (π phase)")
    print(f"  Gate time: {normalized_gate_time}/Ω_max = {actual_gate_time:.4f}s")
    print(f"  Epochs: {epochs} (be patient!)")
    print(f"  Expected time: ~7-8 minutes")
    
    # Create network
    network = FeedForwardNN(
        input_dim=2,
        output_dim=2,
        hidden_layers=4,
        hidden_units=64
    )
    
    # Setup with learning rate scheduler
    pulse_gen = create_default_physical_pulse_generator(rabi_max=rabi_max)
    solver = TorchDiffeqSolver(method='rk4')
    evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=201)
    loss_fn = InfidelityLoss(nqubits=2)
    
    # Optimizer with learning rate decay
    optimizer = torch.optim.Adam(network.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    trainer = QuantumTrainer(
        network=network,
        nqubits=2,
        loss_fn=loss_fn,
        pulse_generator=pulse_gen,
        evolver=evolver,
        optimizer=optimizer
    )
    
    # Training
    angles = torch.tensor([target_angle])
    print(f"\nTraining started at {time.strftime('%H:%M:%S')}")
    print(f"This will take several minutes...\n")
    
    start = time.time()
    
    # Manual training loop with scheduler
    history = {'loss': []}
    for epoch in range(epochs):
        epoch_loss = trainer._train_step(angles, actual_gate_time)
        history['loss'].append(epoch_loss)
        scheduler.step()
        
        if epoch % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d}: Loss = {epoch_loss:.6f}, LR = {current_lr:.2e}")
    
    elapsed = time.time() - start
    
    print(f"\n{'='*70}")
    print(f"Training Complete! (took {elapsed/60:.1f} minutes)")
    print(f"{'='*70}")
    print(f"Initial loss: {history['loss'][0]:.6f}")
    print(f"Final loss: {history['loss'][-1]:.6f}")
    
    # Evaluate
    optimizer_obj = ControlledPhaseOptimizer(
        gate=gate, network=network, trainer=trainer,
        pulse_generator=pulse_gen, evolver=evolver,
        time_optimal=False
    )
    
    result = optimizer_obj.evaluate(target_angle)
    infidelity = result['infidelity']
    fidelity = 1 - infidelity
    
    print(f"\nFinal Results:")
    print(f"  Infidelity: {infidelity:.6e}")
    print(f"  Fidelity: {fidelity*100:.4f}%")
    
    # Visualizations
    print(f"\nSaving visualizations...")
    plot_loss_convergence(history, save_path='cz_extended_loss.png', show=False,
                         title=f'CZ Training - Fidelity: {fidelity*100:.2f}%')
    
    pulses, _ = optimizer_obj.generate_pulse(target_angle)
    plot_pulses_vs_time(pulses, actual_gate_time, 
                       labels=['Rabi (Ω/Ω_max)', 'Detuning (Δ/Ω_max)'],
                       save_path='cz_extended_pulses.png', show=False)
    
    print(f"  ✓ cz_extended_loss.png")
    print(f"  ✓ cz_extended_pulses.png")
    
    # Save model
    torch.save(network.state_dict(), 'cz_high_fidelity_model.pt')
    print(f"  ✓ cz_high_fidelity_model.pt")
    
    return fidelity


if __name__ == "__main__":
    fidelity = train_high_fidelity_cz_extended()
    
    print(f"\n{'='*70}")
    if fidelity > 0.99:
        print(f"🎉 SUCCESS! Achieved {fidelity*100:.2f}% fidelity!")
    elif fidelity > 0.95:
        print(f"✓ Good result: {fidelity*100:.2f}% fidelity")
    else:
        print(f"⚠ Result: {fidelity*100:.2f}% fidelity")
        print(f"  Try even longer training or different hyperparameters")
    print(f"{'='*70}")
