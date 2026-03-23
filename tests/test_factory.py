"""
Minimal working example using factory function
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/madhav22m/gitrepos/qneural')

from qneural.gates.rydberg import create_czphi_optimizer, CZPhiGate

print("Creating optimizer via factory function...")

# Create optimizer with default settings
gate = CZPhiGate()
optimizer = create_czphi_optimizer(time_optimal=False)

print(f"Network: {optimizer.network}")

# Train
angle = torch.tensor([np.pi])
print(f"\nTraining on angle={float(angle/np.pi):.2f}π, gate_time=7.62/rabi...")

history = optimizer.train(
    angles=angle,
    gate_time=7.62 / gate.rabi_max,
    epochs=500,
    print_every=50
)

print(f"\nFinal loss: {history['loss'][-1]:.6f}")

# Evaluate
result = optimizer.evaluate(angle)
print(f"Infidelity: {result['infidelity']:.6f}")
print(f"Fidelity: {(1-result['infidelity'])*100:.2f}%")
