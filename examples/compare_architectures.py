"""
Architecture Comparison for High-Fidelity CZ Gates

Compare different neural network architectures to find the best
for achieving high-fidelity CZ gates.

Architectures tested:
- Small:  4 layers × 64 units (original example)
- Medium: 6 layers × 100 units  
- Large:  6 layers × 150 units (matches published paper)
- XL:     8 layers × 200 units (overkill test)
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
from qneural.analysis import plot_loss_convergence


def train_with_architecture(
    name,
    hidden_layers,
    hidden_units,
    epochs=500,
    learning_rate=5e-3
):
    """
    Train CZ gate with specific architecture.
    
    Returns final fidelity and training time.
    """
    print(f"\n{'='*60}")
    print(f"Architecture: {name}")
    print(f"  Layers: {hidden_layers}, Units: {hidden_units}")
    print(f"  Parameters: ~{hidden_layers * hidden_units * 2}k")
    print(f"{'='*60}")
    
    # Setup
    gate = CZPhiGate()
    rabi_max = gate.rabi_max
    normalized_gate_time = 10.0
    actual_gate_time = normalized_gate_time / rabi_max
    target_angle = torch.pi
    
    # Create network with specified architecture
    network = FeedForwardNN(
        input_dim=2,
        output_dim=2,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        activation='relu',
        use_batchnorm=True
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in network.parameters())
    print(f"  Total parameters: {n_params:,}")
    
    # Setup training
    pulse_gen = create_default_physical_pulse_generator(rabi_max=rabi_max)
    solver = TorchDiffeqSolver(method='rk4')
    evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=201)
    loss_fn = InfidelityLoss(nqubits=2)
    
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
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
    
    start = time.time()
    for epoch in range(epochs):
        epoch_loss = trainer._train_step(angles, actual_gate_time)
        history['loss'].append(epoch_loss)
        scheduler.step()
        
        if epoch % 100 == 0:
            print(f"    Epoch {epoch:3d}: Loss = {epoch_loss:.6f}")
    
    elapsed = time.time() - start
    
    # Evaluate
    optimizer_obj = ControlledPhaseOptimizer(
        gate=gate, network=network, trainer=trainer,
        pulse_generator=pulse_gen, evolver=evolver,
        time_optimal=False
    )
    
    result = optimizer_obj.evaluate(target_angle)
    infidelity = result['infidelity']
    fidelity = 1 - infidelity
    
    print(f"\n  Results:")
    print(f"    Training time: {elapsed:.1f}s")
    print(f"    Final loss: {history['loss'][-1]:.6f}")
    print(f"    Fidelity: {fidelity*100:.2f}%")
    print(f"    Infidelity: {infidelity:.2e}")
    
    return {
        'name': name,
        'layers': hidden_layers,
        'units': hidden_units,
        'params': n_params,
        'fidelity': fidelity,
        'infidelity': infidelity,
        'loss_final': history['loss'][-1],
        'time': elapsed,
        'history': history
    }


def compare_architectures():
    """Compare multiple architectures."""
    
    print("="*70)
    print("NEURAL NETWORK ARCHITECTURE COMPARISON")
    print("="*70)
    print("\nGoal: Achieve >99% fidelity on CZ gate")
    print("Training: 500 epochs each")
    print("Gate time: 10/Ω_max")
    
    # Define architectures to test
    architectures = [
        ("Small", 4, 64),      # Original example
        ("Medium", 6, 100),    # Balanced
        ("Large", 6, 150),     # Matches paper
        # ("XL", 8, 200),      # Uncomment if you want to test huge network
    ]
    
    results = []
    
    for name, layers, units in architectures:
        result = train_with_architecture(name, layers, units, epochs=500)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Architecture':<12} {'Layers':<8} {'Units':<8} {'Params':<10} {'Fidelity':<12} {'Time':<8}")
    print("-"*70)
    
    for r in results:
        print(f"{r['name']:<12} {r['layers']:<8} {r['units']:<8} "
              f"{r['params']:<10,} {r['fidelity']*100:<11.2f}% {r['time']:<7.1f}s")
    
    # Find best
    best = max(results, key=lambda x: x['fidelity'])
    print(f"\n🏆 Best Architecture: {best['name']}")
    print(f"   Fidelity: {best['fidelity']*100:.2f}%")
    print(f"   Config: {best['layers']} layers × {best['units']} units")
    
    # Save comparison plot
    print(f"\nSaving comparison plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss curves
    for r in results:
        ax1.plot(r['history']['loss'], label=f"{r['name']} ({r['layers']}×{r['units']})")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Convergence Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Fidelity bar chart
    names = [r['name'] for r in results]
    fidelities = [r['fidelity']*100 for r in results]
    colors = ['red' if f < 95 else 'orange' if f < 99 else 'green' for f in fidelities]
    ax2.bar(names, fidelities, color=colors, alpha=0.7)
    ax2.axhline(y=99, color='green', linestyle='--', label='99% target')
    ax2.set_ylabel('Fidelity (%)')
    ax2.set_title('Final Fidelity by Architecture')
    ax2.legend()
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('architecture_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: architecture_comparison.png")
    
    return results, best


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    results, best = compare_architectures()
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    if best['fidelity'] > 0.99:
        print(f"✅ Use {best['name']} architecture: {best['layers']}×{best['units']}")
        print(f"   Achieved {best['fidelity']*100:.2f}% fidelity in {best['time']:.1f}s")
    elif best['fidelity'] > 0.95:
        print(f"✓ {best['name']} architecture is good ({best['fidelity']*100:.1f}%)")
        print(f"  For >99% fidelity, try:")
        print(f"    - More epochs (1000+)")
        print(f"    - Larger architecture (8×200)")
        print(f"    - Different learning rate")
    else:
        print(f"⚠ All architectures need more training time")
        print(f"  Best was {best['name']} with {best['fidelity']*100:.1f}%")
    print(f"{'='*70}")
