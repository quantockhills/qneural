"""
Example: High-Fidelity CZ Gate Training

This example demonstrates training a neural network to generate 
a high-fidelity CZ gate (π-phase) with fixed gate time.

Target: Infidelity < 0.01 (fidelity > 99%)
Gate time: 10.0 / rabi_max (normalized units)
"""

import torch
import numpy as np
import sys
import os
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


def train_high_fidelity_cz(
    target_angle=torch.pi,
    normalized_gate_time=10.0,
    epochs=500,
    learning_rate=1e-3,
    print_every=50
):
    """
    Train a high-fidelity CZ gate.
    
    Parameters
    ----------
    target_angle : float
        Target phase angle (default: π for CZ gate)
    normalized_gate_time : float
        Gate time in units of 1/Ω_max (default: 10.0)
    epochs : int
        Number of training epochs (default: 500)
    learning_rate : float
        Adam optimizer learning rate (default: 1e-3)
    print_every : int
        Print progress every N epochs
        
    Returns
    -------
    dict
        Training results including final infidelity, network, pulses, etc.
    """
    print("="*70)
    print("High-Fidelity CZ Gate Training")
    print("="*70)
    
    # Setup
    gate = CZPhiGate()
    rabi_max = gate.rabi_max
    actual_gate_time = normalized_gate_time / rabi_max
    
    print(f"\nConfiguration:")
    print(f"  Target angle: {target_angle/np.pi:.2f}π")
    print(f"  Normalized gate time: {normalized_gate_time} (units of 1/Ω_max)")
    print(f"  Actual gate time: {actual_gate_time:.4f} seconds")
    print(f"  Rabi max: {rabi_max:.2f} MHz")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    
    # Create network
    print("\nCreating neural network...")
    network = FeedForwardNN(
        input_dim=2,
        output_dim=2,
        hidden_layers=4,
        hidden_units=64
    )
    print(f"  Network: 4 layers × 64 units")
    
    # Setup training components
    pulse_gen = create_default_physical_pulse_generator(rabi_max=rabi_max)
    solver = TorchDiffeqSolver(method='rk4')
    evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=201)
    loss_fn = InfidelityLoss(nqubits=2)
    
    # Create trainer
    trainer = QuantumTrainer(
        network=network,
        nqubits=2,
        loss_fn=loss_fn,
        pulse_generator=pulse_gen,
        evolver=evolver,
        optimizer=torch.optim.Adam(network.parameters(), lr=learning_rate)
    )
    
    # Training data
    angles = torch.tensor([target_angle])
    
    print(f"\nStarting training...")
    print(f"  Training on {len(angles)} angle(s): {angles/np.pi}π")
    
    # Train!
    import time
    start_time = time.time()
    
    history = trainer.train(
        angles=angles,
        gate_time=actual_gate_time,
        epochs=epochs,
        print_every=print_every
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Initial loss: {history['loss'][0]:.6f}")
    print(f"Final loss: {history['loss'][-1]:.6f}")
    print(f"Improvement: {(history['loss'][0] - history['loss'][-1]):.6f}")
    
    # Evaluate final performance
    print(f"\nEvaluating trained gate...")
    optimizer_obj = ControlledPhaseOptimizer(
        gate=gate,
        network=network,
        trainer=trainer,
        pulse_generator=pulse_gen,
        evolver=evolver,
        time_optimal=False
    )
    
    result = optimizer_obj.evaluate(target_angle)
    infidelity = result['infidelity']
    fidelity = 1 - infidelity
    
    print(f"  Infidelity: {infidelity:.6e}")
    print(f"  Fidelity: {fidelity*100:.4f}%")
    
    # Check if we achieved high fidelity
    if fidelity > 0.99:
        print(f"\n✓ SUCCESS! Fidelity > 99%")
    elif fidelity > 0.95:
        print(f"\n✓ Good! Fidelity > 95% (may need more training for >99%)")
    else:
        print(f"\n⚠ Fidelity < 95%. Consider:")
        print(f"    - More epochs (try 1000+)")
        print(f"    - Different learning rate")
        print(f"    - Longer gate time")
    
    # Generate and save visualizations
    print(f"\nGenerating visualizations...")
    
    # Plot 1: Loss convergence
    fig1 = plot_loss_convergence(
        history,
        save_path='cz_training_loss.png',
        show=False,
        title=f'CZ Gate Training (Fidelity: {fidelity*100:.2f}%)'
    )
    print(f"  Saved: cz_training_loss.png")
    
    # Plot 2: Pulses
    pulses, _ = optimizer_obj.generate_pulse(target_angle)
    fig2 = plot_pulses_vs_time(
        pulses,
        gate_time=actual_gate_time,
        labels=['Rabi Frequency (Ω/Ω_max)', 'Detuning (Δ/Ω_max)'],
        save_path='cz_pulses.png',
        show=False,
        title=f'Optimized Pulses (Gate time: {normalized_gate_time}/Ω_max)'
    )
    print(f"  Saved: cz_pulses.png")
    
    # Return results
    return {
        'network': network,
        'gate': gate,
        'trainer': trainer,
        'optimizer': optimizer_obj,
        'infidelity': infidelity,
        'fidelity': fidelity,
        'history': history,
        'pulses': pulses,
        'gate_time': actual_gate_time,
        'normalized_gate_time': normalized_gate_time,
        'training_time': elapsed
    }


if __name__ == "__main__":
    # Run training
    results = train_high_fidelity_cz(
        target_angle=torch.pi,           # CZ gate
        normalized_gate_time=10.0,       # 10 / rabi_max
        epochs=500,                      # Training epochs
        learning_rate=1e-3,              # Adam LR
        print_every=50                   # Print every 50 epochs
    )
    
    print(f"\n{'='*70}")
    print("Example complete! Check the generated PNG files for visualizations.")
    print(f"{'='*70}")
