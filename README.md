# qneural: Machine Learning for Quantum Control

![Tests](https://github.com/quantockhills/qneural/workflows/Tests/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Status](https://img.shields.io/badge/status-beta-yellow)

> ⚠️ **BETA SOFTWARE**: This package is currently in beta (v0.5.0). While the core functionality is validated against published results, APIs may evolve before the stable 1.0 release. We welcome feedback and bug reports via [GitHub Issues](https://github.com/quantockhills/qneural/issues).



**Authors:** Madhav Mohan (Eindhoven University of Technology), Julius de Hond (Pasqal)
**License:** Apache-2.0

## Overview

`qneural` provides a modular toolkit for quantum control optimization using machine learning. Developed initially as part of a research project at Pasqal, this library implements the methods described in our Physical Review Applied publication, which demonstrates state-of-the-art results for parametrized multi-qubit gates on Rydberg atom platforms.

While initially focused on neural network-based pulse optimization for neutral atoms, the framework architecture supports extensibility to:

- **Hardware platforms**: Rydberg atoms (current), superconducting qubits (planned), trapped ions (planned)
- **Optimization methods**: Neural networks (current), reinforcement learning (planned), gradient-free optimization (planned)
- **Objectives**: Gate fidelity, time-optimality, robustness to noise, resource efficiency
- **Computational backends**: PyTorch (current), JAX (planned)

## Key Features

### Current Capabilities ✅

#### Quantum Control
- **Parametrized gates**: Optimize gate families CZ_φ for arbitrary phase angles φ or angles ranges.
- **High-fidelity gates**: Achieves ~99.9% fidelity on two-qubit CZ_φ and three-qubit CCZ_φ gates for $\phi\in\[0,\pi]$.
- **Time-optimal synthesis**: Neural networks predict optimal gate times and pulse sequences
- **Transfer learning**: Warm-start optimization from pre-trained models for faster convergence

#### Physics Simulation
- **Rydberg Hamiltonians**: Full quantum dynamics for ground-Rydberg coupling with van der Waals interactions
- **Differentiable evolution**: Automatic differentiation through ODE solvers (torchdiffeq integration)
- **Phase corrections**: Automatic single-qubit phase tracking for accurate two-qubit fidelity calculation

#### Machine Learning
- **Neural architectures**: Dual-network design (time predictor + pulse generator) with configurable depth
- **Specialized trainers**: `FixedRabiTrainer` for detuning-only optimization, `TimeOptimalTrainer` for variable gate times
- **Batch optimization**: Multi-angle training with automatic angle resampling
- **Checkpoint management**: Save/load trained models with metadata and configuration

#### Analysis & Visualization
- **Training diagnostics**: Loss curves, fidelity tracking, convergence analysis
- **Pulse visualization**: Plot optimized control sequences (Rabi, detuning) vs. time
- **Gate analysis**: Unitary fidelity, gate time vs. angle relationships

### Known Limitations (Beta)

- **Platform support**: Currently Rydberg atoms only; superconducting qubit and ion trap support planned
- **Backend**: PyTorch only; JAX backend for improved performance planned for v1.0
- **Incomplete features**: Some loss functions (robustness, resource optimization) not yet fully implemented
- **Scalability**: Optimized for 2-3 qubit gates; larger systems under development

### Roadmap to v1.0

- **v0.6**: JAX backend, additional loss functions, improved visualization tools
- **v0.8**: Release candidate with comprehensive documentation and >85% test coverage
- **v1.0**: Production release with PyPI distribution, CI/CD, and multi-platform support

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 1.10
- NumPy
- torchdiffeq (for ODE integration)
- matplotlib (for visualization)
- qutip (optional, for verification)

### Install from Source (Beta)

```bash
git clone https://github.com/quantockhills/qneural.git
cd qneural
pip install -e .
```

For development installation with testing dependencies:
```bash
pip install -e ".[dev]"
```

**Note**: This is a beta release. Install from source is currently the only distribution method. PyPI distribution planned for v1.0.

### Quick Start

See the `examples/` directory for complete Jupyter notebook tutorials:

**🎓 For Beginners**:
- [`getting_started_2qubit.ipynb`](examples/getting_started_2qubit.ipynb) - **START HERE!** Comprehensive beginner-friendly introduction to qneural
  - Explains quantum gates, neural networks, and control pulses from scratch
  - No quantum computing background required
  - Step-by-step walkthrough with visualizations
  - Includes transfer learning tutorial

**📚 Technical Tutorials**:
- [`cphase_transfer_learning.ipynb`](examples/cphase_transfer_learning.ipynb) - Transfer learning for 2-qubit CZ_φ gates (more concise)
- [`ccphase_transfer_learning.ipynb`](examples/ccphase_transfer_learning.ipynb) - 3-qubit CCZ_φ gates with pre-trained models
- [`cz_gate_optimization.ipynb`](examples/cz_gate_optimization.ipynb) - Training from scratch for single angles
- [`cphase_gate_optimization.ipynb`](examples/cphase_gate_optimization.ipynb) - Full CZ_φ family optimization

## Project Structure

```
qneural/
├── qneural/                  # Main package
│   ├── backend/             # Computational backend abstraction (PyTorch)
│   ├── hardware/rydberg/    # Rydberg atom physics (Hamiltonians, operators, pulses)
│   ├── core/                # Hardware-agnostic quantum operations (states, gates, metrics)
│   ├── neural/              # Neural network architectures and trainers
│   ├── gates/rydberg/       # Gate-specific implementations (CZPhi, etc.)
│   ├── analysis/            # Visualization and plotting utilities
│   └── utils/               # Helper functions (conversion, loading, etc.)
│
├── examples/                 # Jupyter notebook tutorials
├── tests/                    # Unit and integration tests (142+ tests)
├── docs/                     # Documentation and design notes
├── archival/                 # Original research code (preserved for validation)
└── planning/                 # Development roadmap and planning documents
```

## Documentation

The best way to learn qneural is through the Jupyter notebooks in the `examples/` directory:
- **[getting_started_2qubit.ipynb](examples/getting_started_2qubit.ipynb)** - Beginner-friendly tutorial for 2-qubit CZ_φ gates
- **[ccphase_transfer_learning.ipynb](examples/ccphase_transfer_learning.ipynb)** - Advanced 3-qubit CCZ_φ transfer learning

API documentation will be added in v1.0.

## Validation

This implementation has been validated against the original research code used in our Physical Review Applied publication. Key validation points:

- ✅ **Physics equivalence**: Identical Hamiltonian, operators, and time evolution
- ✅ **Numerical equivalence**: Same infidelity metrics and loss functions
- ✅ **Result reproduction**: Achieves >99.9% fidelity matching published results
- ✅ **Test coverage**: 142+ unit and integration tests passing

The `archival/` directory preserves the original research code for reference and reproducibility.

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

## Contributing

We welcome contributions, bug reports, and feedback! This is a beta release and we're actively seeking input from the quantum computing community.

**How to contribute:**
- **Report bugs**: Open an issue on [GitHub Issues](https://github.com/quantockhills/qneural/issues)
- **Request features**: Describe your use case in an issue
- **Submit code**: Fork the repository and submit a pull request
- **Share results**: We'd love to hear about your applications!

**Development setup:**
```bash
git clone https://github.com/quantockhills/qneural.git
cd qneural
pip install -e ".[dev]"
pytest tests/  # Run test suite
```

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## Contact

**Madhav Mohan** - Eindhoven University of Technology
**Julius de Hond** - Pasqal

For questions, collaborations, or support:
- Open an issue: [GitHub Issues](https://github.com/quantockhills/qneural/issues)
- Email: madhav.mohan@protonmail.com

## Acknowledgments

This project was conducted at Pasqal, and received funding from the European Union’s Horizon 2020 research and innovation programme via the project 101070144 (EuRyQa) and under the Marie Sklodowska-Curie grant agreement number 955479. We thank the quantum computing community for their continued support and feedback.
