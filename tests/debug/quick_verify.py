"""
Quick verification that training works
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/madhav22m/gitrepos/qneural')

from qneural.gates.rydberg import create_czphi_optimizer

print("Testing fixed-time training (50 epochs)...")

optimizer = create_czphi_optimizer(time_optimal=False)

# Train for just 50 epochs to verify it works
history = optimizer.train(
    angle_range=(0.5 * np.pi, np.pi),
    epochs=50,
    print_every=10
)

print(f"\n✓ Training completed!")
print(f"Initial loss: {history['loss'][0]:.6f}")
print(f"Final loss: {history['loss'][-1]:.6f}")

# Quick eval
result = optimizer.evaluate(np.pi)
print(f"Infidelity at π: {result['infidelity']:.6f}")

if result['infidelity'] < 0.9:
    print("✓ Some learning occurred")
else:
    print("⚠ Check if training is working")
