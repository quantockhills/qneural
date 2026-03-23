"""
Machine learning methods for quantum control.

This module provides neural network architectures and training infrastructure
for optimizing quantum control pulses.

Key components:
    - Neural network models (FeedForwardNN, PulseGenerator, TimeOptimalController)
    - Loss functions (InfidelityLoss, TimePenaltyLoss, CompositeLoss)
    - ODE solvers (TorchDiffeqSolver with abstract interface for future backends)
    - Pulse generation (converting NN outputs to callable pulse functions)
    - Quantum evolution (integrating pulses with Hamiltonian evolution)
    - Training infrastructure (QuantumTrainer, TimeOptimalTrainer)

Examples
--------
>>> from qneural.neural import FeedForwardNN, QuantumTrainer
>>> from qneural.neural.losses import create_infidelity_loss
>>>
>>> # Create network and trainer
>>> network = FeedForwardNN(input_dim=2, output_dim=2, hidden_layers=6, hidden_units=150)
>>> trainer = QuantumTrainer(
...     network=network,
...     nqubits=2,
...     loss_fn=create_infidelity_loss(nqubits=2)
... )
>>>
>>> # Train
>>> angles = torch.linspace(0.1 * torch.pi, torch.pi, 80)
>>> history = trainer.train(angles, gate_time=5.0, epochs=1000)
"""

# Neural network models
from .models import (
    FeedForwardNN,
    PulseGenerator,
)

# Time-optimal control
from .time_optimal import (
    TimeOptimalController,
    TimeOptimalTrainer,
)

# Loss functions
from .losses import (
    QuantumLoss,
    InfidelityLoss,
    TimePenaltyLoss,
    RobustnessLoss,
    ResourceLoss,
    CompositeLoss,
    create_infidelity_loss,
    create_time_optimal_loss,
)

# ODE solvers
from .solvers import (
    ODESolver,
    TorchDiffeqSolver,
    FixedStepSolver,
    DiffraxSolver,
    create_solver,
    solve_ivp,
)

# Pulse generation
from .pulse_generator import (
    PhysicalPulseGenerator,
    TimeOptimalPulseGenerator,
    BatchedPulseGenerator,
    create_default_physical_pulse_generator,
    pulses_to_hamiltonian,
)

# Quantum evolution
from .evolution import (
    QuantumEvolver,
    BatchedQuantumEvolver,
    create_evolver,
    quick_evolve,
)

# Training
from .trainer import (
    QuantumTrainer,
    FixedRabiTrainer,
    create_trainer,
)

__all__ = [
    # Models
    'FeedForwardNN',
    'PulseGenerator',
    
    # Time-optimal control
    'TimeOptimalController',
    'TimeOptimalTrainer',
    
    # Losses
    'QuantumLoss',
    'InfidelityLoss',
    'TimePenaltyLoss',
    'RobustnessLoss',
    'ResourceLoss',
    'CompositeLoss',
    'create_infidelity_loss',
    'create_time_optimal_loss',
    
    # Solvers
    'ODESolver',
    'TorchDiffeqSolver',
    'FixedStepSolver',
    'DiffraxSolver',
    'create_solver',
    'solve_ivp',
    
    # Pulse generation
    'PhysicalPulseGenerator',
    'TimeOptimalPulseGenerator',
    'BatchedPulseGenerator',
    'create_default_physical_pulse_generator',
    'pulses_to_hamiltonian',
    
    # Evolution
    'QuantumEvolver',
    'BatchedQuantumEvolver',
    'create_evolver',
    'quick_evolve',
    
    # Training
    'QuantumTrainer',
    'FixedRabiTrainer',
    'create_trainer',
]