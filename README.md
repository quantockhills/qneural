# qneural: Machine Learning for Quantum Control

A flexible, modular framework for optimizing quantum control protocols using machine learning.

**Authors:** Madhav Mohan, Julius de Hond

## Overview

`qneural` provides a platform-agnostic toolkit for quantum control optimization. This library was developed during PhD research at Eindhoven University of Technology and is now being open-sourced and extended. While initially developed for neural network-based pulse optimization on Rydberg atom systems (demonstrating state-of-the-art results for parametrized multi-qubit gates), the framework is designed to support:

- **Multiple quantum hardware platforms**: Rydberg atoms, superconducting qubits, trapped ions, etc.
- **Various ML approaches**: Neural networks (current), reinforcement learning (planned), gradient-free optimization (planned)
- **Flexible objectives**: Gate fidelity, time-optimality, robustness, resource efficiency
- **Computational backends**: PyTorch (current), JAX (planned)

## Key Features

### Current Capabilities ✅

#### Core Features
- ✅ **High-fidelity gate optimization**: >99% fidelity on CZ gates (validated)
- ✅ **FixedRabiTrainer**: Specialized class for detuning-only optimization
- ✅ **Rydberg atom Hamiltonians**: Full support for ground-Rydberg and ground-ground qubits
- ✅ **2-qubit gates**: CZ_φ with optimal pulse sequences
- ✅ **Differentiable ODE solvers**: torchdiffeq integration with automatic differentiation
- ✅ **Phase corrections**: Automatic single-qubit phase correction during training
- ✅ **Comprehensive metrics**: Fidelity, infidelity, unitary analysis

#### Recent Improvements (March 2025)
- ✅ **Fixed phase correction bugs**: Now achieves >99% fidelity (was stuck at ~40%)
- ✅ **Symmetric phase correction**: Matches published paper methodology
- ✅ **Simplified API**: `FixedRabiTrainer` for common use cases
- ✅ **Working examples**: Jupyter notebook with validated training approach

### In Progress 🔄
- 🔄 **Time-optimal training**: Variable gate time optimization (infrastructure present, NN chaining needs completion)
- 🔄 **Visualization tools**: Centralized plotting and analysis module
- 🔄 **Checkpoint system**: Save/resume training functionality

### Planned Extensions
- 🔄 **Visualization & analysis**: Comprehensive plotting tools for training analysis
- 🔄 **Checkpoint system**: Auto-save and resume training
- 🔄 **JAX backend**: For improved performance and XLA compilation
- 🔄 **Additional hardware platforms**: Superconducting qubits, trapped ions
- 🔄 **Reinforcement learning**: Model-free optimization approaches
- 🔄 **Gradient-free methods**: CMA-ES, evolutionary algorithms
- 🔄 **Robustness optimization**: Control against noise and systematic errors

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 1.10
- NumPy
- torchdiffeq (for ODE integration)
- matplotlib (for visualization)
- qutip (optional, for verification)

### Install from source
```bash
git clone https://github.com/yourusername/qneural.git
cd qneural
pip install -e .
```

## Quick Start

### High-Fidelity CZ Gate (Working Example)

```python
import torch
import numpy as np
from qneural.neural import FeedForwardNN, FixedRabiTrainer
from qneural.gates.rydberg import CZPhiGate

# Setup
gate = CZPhiGate()
rabi_max = gate.rabi_max

# Create network (detuning only, fixed rabi at max)
network = FeedForwardNN(
    input_dim=2,      # [angle, normalized_time]
    output_dim=1,     # Detuning only
    hidden_layers=6,
    hidden_units=150,
    weight_scale=1.8  # Critical for good initialization
)

# Create specialized trainer for fixed-rabi optimization
trainer = FixedRabiTrainer(
    network=network,
    nqubits=2,
    rabi_max=rabi_max
)

# Train on CZ gate (π phase)
history = trainer.train(
    angles=torch.tensor([np.pi]),
    gate_time=7.62 / rabi_max,  # Optimal time
    epochs=500,
    print_every=50
)

print(f"Final fidelity: {(1 - history['loss'][-1])*100:.2f}%")
# Output: Final fidelity: 99.87%
```

### Key Components

- **`FeedForwardNN`**: Neural network architecture with configurable layers
- **`FixedRabiTrainer`**: Specialized trainer for detuning-only optimization  
- **Phase corrections**: Applied automatically during training for accurate fidelity computation
- **Optimal gate time**: 7.62/Ω_max for CZ gates on neutral atoms

See `examples/01_high_fidelity_cz_gate.ipynb` for a complete working example.

## Project Structure

```
qneural/
├── qneural/               # Main package
│   ├── backend/          # Computational backend (PyTorch/JAX)
│   ├── hardware/         # Hardware-specific implementations
│   │   └── rydberg/     # Rydberg atom systems
│   ├── core/            # Hardware-agnostic quantum operations
│   ├── ml/              # Machine learning methods
│   │   ├── neural/      # Neural network architectures
│   │   └── solvers/     # ODE solvers for time evolution
│   ├── gates/           # Gate-specific implementations
│   ├── control/         # Pulse generation and optimization
│   └── utils/           # Utilities (plotting, I/O, etc.)
│
├── research/             # Research notebooks (working files)
│   ├── czphi/           # 2-qubit gate research
│   ├── cczphi/          # 3-qubit gate research
│   └── analysis/        # Visualization and verification
│
├── examples/             # Clean, documented tutorials
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Documentation

### Active Documentation (`docs/`)
- **[AGENTS.md](docs/AGENTS.md)** - Coding guidelines and best practices
- **[STRUCTURE.md](docs/STRUCTURE.md)** - Package architecture and design principles
- **[VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md)** - Current status and test results (142 tests passing ✓)

### Development Planning (`planning/`)
See [planning/README.md](planning/README.md) for:
- **Active Development** - Features being worked on now
- **Backlog** - Future ideas and enhancements
- **Completed** - Archive of finished milestones

Current active plans:
- [Checkpoint & Resume System](planning/active/checkpoint_resume_plan.md) - Save/resume training
- [Visualization Tools](planning/active/visualization_analysis.md) - Analysis and plotting

### Project History
- **[CHANGELOG.md](CHANGELOG.md)** - Detailed history of changes and fixes
- **Original Migration** - See `planning/completed/` for initial package development notes

## Examples

See `examples/` for Jupyter notebooks demonstrating:
1. Single qubit rotations on Rydberg atoms
2. Two-qubit CZ_φ gate optimization
3. Multi-angle pulse families
4. Three-qubit CCZ_φ gates
5. Pulse visualization and analysis

## Research Notebooks

The `research/` directory contains the original research notebooks used to develop and validate the methods. These demonstrate real-world usage and published results.

## Citation

If you use qneural in your research, please cite our publication:

**Reference Publication:**
```bibtex
@article{PhysRevApplied.23.054074,
  title = {Parametrized multiqubit gates for neutral-atom quantum platforms},
  author = {Mohan, Madhav and de Hond, Julius and Kokkelmans, Servaas},
  journal = {Phys. Rev. Appl.},
  volume = {23},
  issue = {5},
  pages = {054074},
  numpages = {12},
  year = {2025},
  month = {May},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevApplied.23.054074},
  url = {https://link.aps.org/doi/10.1103/PhysRevApplied.23.054074}
}
```

This paper demonstrates state-of-the-art results for parametrized multi-qubit gates achieved using the methods implemented in this library.

**Software:**
```bibtex
@software{qneural2024,
  title = {qneural: Machine Learning for Quantum Control},
  author = {Mohan, Madhav and de Hond, Julius},
  year = {2024},
  url = {https://github.com/quantockhills/qneural}
}
```

## License

[To be determined]

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## Contact

For questions or collaborations:
- Madhav Mohan: [email]
- Julius de Hond: [email]
