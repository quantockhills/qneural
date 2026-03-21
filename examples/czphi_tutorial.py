"""
Tutorial: CZ_φ Gate Optimization with Neural Networks

This example demonstrates the complete pipeline for optimizing parametrized
two-qubit CZ_φ gates on Rydberg atom systems using neural networks.

The CZ_φ gate applies a phase φ to the |11⟩ state:
    CZ_φ = diag(1, 1, 1, e^{iφ})

We train a neural network to generate time-optimal pulse sequences that
implement this gate for any angle φ ∈ [0.1π, π].

Author: Madhav Mohan, Julius de Hond
"""

import torch
import numpy as np
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qneural.gates import create_czphi_optimizer
from qneural.config import print_config

# =============================================================================
# Configuration
# =============================================================================

print("=" * 70)
print("CZ_φ Gate Optimization Tutorial")
print("=" * 70)
print()

# Print qneural configuration
print_config()
print()

# Training parameters
ANGLE_MIN = 0.1 * torch.pi  # Minimum angle to train
ANGLE_MAX = torch.pi         # Maximum angle to train
N_ANGLES = 80                # Number of angles to sample
EPOCHS = 100                 # Training epochs (use 1000+ for production)
TIME_OPTIMAL = False         # Optimize gate time as well as pulses (set to True for time-optimal)
TIME_BOUNDS = (3.0, 8.0)     # Expected gate time range (units of 1/Ω_max)
GATE_TIME = 5.0              # Fixed gate time for non-time-optimal training

print(f"Training Configuration:")
print(f"  Angle range:     [{ANGLE_MIN/torch.pi:.2f}π, {ANGLE_MAX/torch.pi:.2f}π]")
print(f"  Number of angles: {N_ANGLES}")
print(f"  Training epochs:  {EPOCHS}")
print(f"  Time-optimal:     {TIME_OPTIMAL}")
if TIME_OPTIMAL:
    print(f"  Time bounds:      {TIME_BOUNDS}")
else:
    print(f"  Fixed gate time:  {GATE_TIME}")
print()

# =============================================================================
# Step 1: Create the Optimizer
# =============================================================================

print("Step 1: Creating CZ_φ optimizer...")
print("-" * 70)

optimizer = create_czphi_optimizer(
    time_optimal=TIME_OPTIMAL,
    time_bounds=TIME_BOUNDS
)

print(f"✓ Optimizer created for {optimizer.gate.total_qubits}-qubit CZ_φ gate")
print(f"  Rabi frequency:    {optimizer.gate.rabi_max:.2f} rad/s")
print(f"  Detuning range:    {optimizer.gate.detuning_range}")
print(f"  Hilbert dimension: {optimizer.gate.full_dim} (with Rydberg state)")
print(f"  Computational dim: {optimizer.gate.comp_dim} (without Rydberg state)")
print()

# =============================================================================
# Step 2: Train the Neural Network
# =============================================================================

print("Step 2: Training neural network to generate optimal pulses...")
print("-" * 70)

history = optimizer.train(
    angle_range=(ANGLE_MIN, ANGLE_MAX),
    n_angles=N_ANGLES,
    gate_time=GATE_TIME if not TIME_OPTIMAL else None,
    epochs=EPOCHS
)

print(f"✓ Training complete!")
print(f"  Final loss: {history['losses'][-1]:.6f}")
print()

# =============================================================================
# Step 3: Evaluate at Specific Angles
# =============================================================================

print("Step 3: Evaluating gate fidelity at specific angles...")
print("-" * 70)

# Test at canonical angles
test_angles = [
    0.25 * torch.pi,  # π/4
    0.5 * torch.pi,   # π/2
    0.75 * torch.pi,  # 3π/4
    torch.pi          # π (full CZ gate)
]

print(f"{'Angle':<12} {'Gate Time':<15} {'Infidelity':<15} {'Fidelity (%)':<15}")
print("-" * 70)

results = []
for angle in test_angles:
    result = optimizer.evaluate(angle.item())
    fidelity = 1.0 - result['infidelity']

    results.append(result)

    print(f"{angle/torch.pi:.2f}π      "
          f"{result['gate_time']:<15.4f} "
          f"{result['infidelity']:<15.6e} "
          f"{fidelity*100:<15.4f}")

print()

# =============================================================================
# Step 4: Generate and Inspect Pulses
# =============================================================================

print("Step 4: Generating pulse sequence for φ = π/2...")
print("-" * 70)

angle = torch.pi / 2
pulses, gate_time = optimizer.generate_pulse(angle.item())

print(f"✓ Pulse sequence generated")
print(f"  Number of controls: {len(pulses)}")
print(f"  Optimal gate time:  {gate_time:.4f} (units of 1/Ω_max)")
print()

# Sample pulse values at a few time points
n_samples = 5
time_points = np.linspace(0, gate_time, n_samples)

print(f"Pulse values at {n_samples} time points:")
print(f"{'Time':<12} {'Rabi (Ω)':<15} {'Detuning (Δ)':<15}")
print("-" * 42)

for t in time_points:
    rabi = pulses[0](t)
    detuning = pulses[1](t)
    print(f"{t:<12.4f} {rabi:<15.4f} {detuning:<15.4f}")

print()

# =============================================================================
# Step 5: Verify Physics Properties
# =============================================================================

print("Step 5: Verifying physics properties...")
print("-" * 70)

# Check that achieved unitary is close to target
result_pi_half = optimizer.evaluate(torch.pi / 2)
achieved = result_pi_half['achieved_unitary']
target = result_pi_half['target_unitary']

# Test 1: Unitarity
identity = torch.matmul(achieved, achieved.conj().T)
eye = torch.eye(achieved.shape[0], dtype=torch.cfloat)
unitarity_error = torch.norm(identity - eye).item()

print(f"✓ Unitarity check:")
print(f"  ||U†U - I||: {unitarity_error:.6e}")
print(f"  {'PASS' if unitarity_error < 1e-4 else 'FAIL'}")
print()

# Test 2: Diagonal structure (CZ gates should be diagonal)
off_diag = achieved - torch.diag(torch.diagonal(achieved))
off_diag_norm = torch.norm(off_diag).item()

print(f"✓ Diagonal structure check:")
print(f"  ||off-diagonal||: {off_diag_norm:.6e}")
print(f"  {'PASS' if off_diag_norm < 1e-4 else 'FAIL'}")
print()

# Test 3: Correct phase on |11⟩
phase_achieved = torch.angle(achieved[3, 3])
phase_expected = torch.pi / 2

print(f"✓ Phase check (φ = π/2):")
print(f"  Expected phase: {phase_expected:.4f}")
print(f"  Achieved phase: {phase_achieved:.4f}")
print(f"  Error:          {abs(phase_achieved - phase_expected):.6e}")
print(f"  {'PASS' if abs(phase_achieved - phase_expected) < 0.1 else 'FAIL'}")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 70)
print("Summary")
print("=" * 70)

# Calculate average fidelity across test angles
avg_fidelity = 1.0 - np.mean([r['infidelity'] for r in results])
avg_gate_time = np.mean([r['gate_time'] for r in results])

print(f"Training performance:")
print(f"  Average fidelity:    {avg_fidelity*100:.4f}%")
print(f"  Average gate time:   {avg_gate_time:.4f}")
print(f"  Time in bounds:      {'Yes' if TIME_BOUNDS[0] <= avg_gate_time <= TIME_BOUNDS[1] else 'No'}")
print()

if avg_fidelity > 0.99:
    print("✓ SUCCESS: High-fidelity gates achieved!")
    print("  The neural network successfully learned to generate optimal pulses.")
elif avg_fidelity > 0.95:
    print("⚠ PARTIAL SUCCESS: Decent fidelity, but could be improved.")
    print("  Try increasing training epochs or adjusting network architecture.")
else:
    print("✗ NEEDS WORK: Low fidelity.")
    print("  Consider longer training, different hyperparameters, or checking setup.")

print()
print("Next steps:")
print("  - Visualize pulses (see examples/visualize_pulses.py)")
print("  - Train for more epochs for production-quality results")
print("  - Try CCZ_φ gates (3-qubit) with create_cczphi_optimizer()")
print("  - Save and load trained models for reuse")
print()
print("=" * 70)
