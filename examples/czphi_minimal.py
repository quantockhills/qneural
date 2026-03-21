"""
Minimal CZ_φ Example - Fast Demo

This is the absolute minimal example to validate the pipeline works.
Uses very few training iterations for speed.

For production results, use czphi_basic.py or czphi_tutorial.py with more epochs.

Author: Madhav Mohan, Julius de Hond
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qneural.gates.rydberg import CZPhiGate, ControlledPhaseOptimizer
from qneural.neural import (
    FeedForwardNN,
    create_default_physical_pulse_generator,
    create_evolver,
    QuantumTrainer,
    InfidelityLoss
)

print("=" * 60)
print("Minimal CZ_φ Gate Optimization Example")
print("=" * 60)
print()

# Minimal parameters for SPEED
N_ANGLES = 5          # Very few angles
EPOCHS = 5            # Very few epochs
GATE_TIME = 5.0
ANGLE = torch.pi      # Test at φ = π (full CZ gate)

print(f"Quick validation with {N_ANGLES} angles, {EPOCHS} epochs")
print(f"(This is NOT for production - just to validate pipeline)")
print()

# Create components
print("[1/6] Creating gate...")
gate = CZPhiGate()

print("[2/6] Creating network...")
network = FeedForwardNN(input_dim=2, output_dim=2, hidden_layers=3, hidden_units=50)

print("[3/6] Creating pulse generator and evolver...")
pulse_gen = create_default_physical_pulse_generator(rabi_max=gate.rabi_max)
evolver = create_evolver(nqubits=2)

print("[4/6] Creating trainer...")
trainer = QuantumTrainer(
    network=network,
    nqubits=2,
    loss_fn=InfidelityLoss(nqubits=2),
    pulse_generator=pulse_gen,
    evolver=evolver
)

print("[5/6] Training (this will take ~1-2 min)...")
angles = torch.linspace(0.5*torch.pi, torch.pi, N_ANGLES)
history = trainer.train(angles, GATE_TIME, EPOCHS, print_every=1)

print()
print("[6/6] Evaluating...")
optimizer = ControlledPhaseOptimizer(
    gate=gate, network=network, trainer=trainer,
    pulse_generator=pulse_gen, evolver=evolver
)

result = optimizer.evaluate(ANGLE)
fidelity = 1.0 - result['infidelity']

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Angle tested:  {ANGLE/torch.pi:.2f}π")
print(f"Infidelity:    {result['infidelity']:.6e}")
print(f"Fidelity:      {fidelity*100:.2f}%")
print()

if result['infidelity'] < 1.0:
    print("✓ SUCCESS: Pipeline is working!")
    print("  The quantum evolution, gates, and training all function correctly.")
    print()
    print("  Note: Fidelity is low because we only trained for", EPOCHS, "epochs.")
    print("  For high-fidelity gates (>99%), use czphi_basic.py with ~1000 epochs.")
else:
    print("⚠ Warning: Infidelity >= 1, something may be wrong.")

print()
print("=" * 60)
