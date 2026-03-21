"""
Basic CZ_φ Gate Optimization Tutorial

Simplified example that demonstrates CZ_φ gate optimization with fixed gate time.
This is a minimal working example to validate the pipeline.

Author: Madhav Mohan, Julius de Hond
"""

import torch
import numpy as np
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qneural.gates.rydberg import CZPhiGate, ControlledPhaseOptimizer
from qneural.neural import (
    FeedForwardNN,
    create_default_physical_pulse_generator,
    create_evolver,
    QuantumTrainer,
    InfidelityLoss
)
from qneural.config import print_config

print("=" * 70)
print("CZ_φ Gate Optimization - Basic Example")
print("=" * 70)
print()

# Configuration
print_config()
print()

# Parameters
ANGLE_MIN = 0.5 * torch.pi
ANGLE_MAX = torch.pi
N_ANGLES = 40
EPOCHS = 50  # Keep small for quick demo
GATE_TIME = 5.0

print("Training Parameters:")
print(f"  Angle range: [{ANGLE_MIN/torch.pi:.2f}π, {ANGLE_MAX/torch.pi:.2f}π]")
print(f"  N angles: {N_ANGLES}")
print(f"  Epochs: {EPOCHS}")
print(f"  Gate time: {GATE_TIME}")
print()

# Step 1: Create gate specification
print("Creating CZ_φ gate...")
gate = CZPhiGate()
print(f"✓ Gate: {gate.total_qubits}-qubit CZ_φ")
print()

# Step 2: Create neural network
print("Creating neural network...")
network = FeedForwardNN(
    input_dim=2,  # (angle, time)
    output_dim=2,  # (Rabi, detuning)
    hidden_layers=6,
    hidden_units=150
)
print(f"✓ Network created with {sum(p.numel() for p in network.parameters())} parameters")
print()

# Step 3: Create components
print("Creating training components...")
pulse_generator = create_default_physical_pulse_generator(rabi_max=gate.rabi_max)
evolver = create_evolver(nqubits=gate.total_qubits)
loss_fn = InfidelityLoss(nqubits=gate.total_qubits)
print("✓ Components created")
print()

# Step 4: Create trainer
print("Creating trainer...")
trainer = QuantumTrainer(
    network=network,
    nqubits=gate.total_qubits,
    loss_fn=loss_fn,
    pulse_generator=pulse_generator,
    evolver=evolver
)
print("✓ Trainer ready")
print()

# Step 5: Train
print("Training...")
print("-" * 70)

angles = torch.linspace(ANGLE_MIN, ANGLE_MAX, N_ANGLES)
history = trainer.train(
    angles=angles,
    gate_time=GATE_TIME,
    epochs=EPOCHS,
    print_every=10
)

print()
print(f"✓ Training complete! Final loss: {history['loss'][-1]:.6e}")
print()

# Step 6: Evaluate
print("Evaluating at φ = π...")
print("-" * 70)

# Create optimizer wrapper for evaluation
optimizer = ControlledPhaseOptimizer(
    gate=gate,
    network=network,
    trainer=trainer,
    pulse_generator=pulse_generator,
    evolver=evolver,
    time_optimal=False
)

result = optimizer.evaluate(torch.pi)
fidelity = 1.0 - result['infidelity']

print(f"Angle:      {result['angle']/torch.pi:.2f}π")
print(f"Gate time:  {result['gate_time']:.4f}")
print(f"Infidelity: {result['infidelity']:.6e}")
print(f"Fidelity:   {fidelity*100:.4f}%")
print()

# Check if it's working
if fidelity > 0.9:
    print("✓ SUCCESS! The pipeline is working.")
    print("  Try increasing EPOCHS for better fidelity.")
elif fidelity > 0.5:
    print("⚠ Partial success - increase training epochs.")
else:
    print("✗ Low fidelity - may need more training or debugging.")

print()
print("=" * 70)
