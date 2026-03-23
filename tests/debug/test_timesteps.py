"""
Test different time step configurations to fix >100% fidelity issue
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/madhav22m/gitrepos/qneural')

from qneural.gates.rydberg import CZPhiGate
from qneural.neural import FeedForwardNN, FixedRabiTrainer, TorchDiffeqSolver, QuantumEvolver
from qneural.neural.pulse_generator import PhysicalPulseGenerator

def test_timesteps(n_steps):
    """Test with specific number of time steps"""
    print(f"\n{'='*60}")
    print(f"Testing with {n_steps} time steps")
    print(f"{'='*60}")

    # Setup
    gate = CZPhiGate()
    rabi_max = gate.rabi_max
    gate_time = 7.62 / rabi_max
    angle = torch.tensor([np.pi])

    # Network
    network = FeedForwardNN(
        input_dim=2, output_dim=1,
        hidden_layers=6, hidden_units=150,
        activation='relu', output_activation='sigmoid',
        use_batch_norm=True, weight_scale=1.8
    )

    # Pulse generator with n_steps
    pulse_gen = PhysicalPulseGenerator(
        n_controls=1,
        n_time_steps=n_steps,
        control_ranges=[(-2*rabi_max, 2*rabi_max)]
    )

    # Solver and evolver with n_steps
    solver = TorchDiffeqSolver(method='rk4')
    evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=n_steps)

    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

    # Trainer
    trainer = FixedRabiTrainer(
        network=network,
        nqubits=2,
        rabi_max=rabi_max,
        pulse_generator=pulse_gen,
        evolver=evolver,
        optimizer=optimizer
    )

    print(f"  Pulse gen time steps: {trainer.pulse_generator.n_time_steps}")
    print(f"  Evolver time steps: {trainer.evolver.n_time_steps}")

    # Train
    print(f"\nTraining for 200 epochs...")
    history = trainer.train(
        angles=angle,
        gate_time=gate_time,
        epochs=200,
        print_every=50
    )

    final_infidelity = history['infidelity'][-1]
    final_fidelity = (1 - final_infidelity) * 100

    print(f"\nResults:")
    print(f"  Final infidelity: {final_infidelity:.6f}")
    print(f"  Final fidelity: {final_fidelity:.4f}%")

    # Check for >100%
    if final_fidelity > 100:
        print(f"  ⚠ WARNING: Fidelity >100% (numerical issue)")
    else:
        print(f"  ✓ Fidelity ≤100% (good!)")

    return final_fidelity

# Test different time step counts
timesteps_to_test = [101, 201, 301, 501]

results = {}
for n in timesteps_to_test:
    fid = test_timesteps(n)
    results[n] = fid

print(f"\n\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for n, fid in results.items():
    status = "✓" if fid <= 100 else "⚠"
    print(f"{status} {n:3d} steps: {fid:.4f}% fidelity")
