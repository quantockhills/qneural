"""
Debug: What do the unitaries look like?
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/madhav22m/gitrepos/qneural')

from qneural.gates.rydberg import CZPhiGate
from qneural.neural import FeedForwardNN, TorchDiffeqSolver, QuantumEvolver
from qneural.neural.pulse_generator import PhysicalPulseGenerator
from qneural.core.metrics import unitary_fidelity

gate = CZPhiGate()
angle = torch.tensor([np.pi])

# Get target
target_U = gate.get_target_unitary(angle)
print("Target unitary:")
print(target_U)
print(f"\nDiagonal: {torch.diag(target_U)}")

# What if we apply correction to target?
print("\n" + "="*60)
print("Applying phase correction to target...")

# Correction formula
phases = torch.angle(torch.diag(target_U))
print(f"Phases: {phases}")

correction = torch.diag(torch.exp(-1.0j * phases))
correction[0, 0] = 1.0
target_corrected = correction @ target_U

print(f"\nCorrected target diagonal: {torch.diag(target_corrected)}")
print(f"Target after correction:\n{target_corrected}")

# What about the symmetric correction?
print("\n" + "="*60)
print("Applying symmetric correction...")
phi_01 = torch.angle(target_U[1, 1])
j1 = torch.exp(-1.0j * phi_01)
sym_correction = torch.eye(4, dtype=torch.cfloat)
sym_correction[1, 1] = j1
sym_correction[2, 2] = j1
sym_correction[3, 3] = j1 ** 2

target_sym = sym_correction @ target_U
print(f"Symmetric corrected diagonal: {torch.diag(target_sym)}")
print(f"\nTarget U is already ideal CZ: {torch.allclose(target_U, torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.cfloat)))}")
