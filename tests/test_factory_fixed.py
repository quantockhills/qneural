"""
Test using factory - fixed API
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/madhav22m/gitrepos/qneural')

from qneural.gates.rydberg import create_czphi_optimizer

print("Creating optimizer with default settings (2 outputs: rabi+detuning)...")

optimizer = create_czphi_optimizer(time_optimal=False)

print(f"Network outputs: {optimizer.network.output_dim}")
print(f"\nTraining 500 epochs on angle range [0.5π, π]...")

# Train on angle range
history = optimizer.train(
    angle_range=(0.5 * np.pi, np.pi),
    epochs=500,
    print_every=50
)

print(f"\nFinal loss: {history['loss'][-1]:.6f}")

# Evaluate at π
result = optimizer.evaluate(np.pi)
print(f"At angle=π: Infidelity={result['infidelity']:.6f}, Fidelity={(1-result['infidelity'])*100:.2f}%")
