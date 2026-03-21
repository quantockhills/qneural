# Project Handoff Document: qneural

## Project Overview

**qneural** is a modular Python framework for Machine Learning in Quantum Control, specifically for optimizing quantum gate pulse sequences using neural networks. Originally developed for Rydberg atom systems (neutral atoms), designed to be hardware-agnostic and backend-agnostic.

**Current Status**: ~70% complete core infrastructure

---

## ✅ What's Been Completed

### 1. Core Infrastructure (`qneural/core/`)
- **states.py**: Basis states, tensor products, state manipulation
- **gates.py**: Gate constructions (CZ_φ, CCZ_φ, batch operations)
- **operators.py**: Pauli matrices, rotations
- **metrics.py**: Fidelity, infidelity calculations
- **evolution.py**: Schrödinger equation solver with ODE integration

**Tests**: 38 tests passing (`tests/test_core_operations.py`)

### 2. Physics Layer (`qneural/hardware/rydberg/`)
- **constants.py**: Physical parameters (RABI_DEFAULT, VDD_COUPLING, etc.)
- **operators.py**: Rydberg-specific operators (basis kets, transitions)
- **pulses.py**: Time-dependent pulse functions (constant, piecewise, Gaussian, Blackman)
- **hamiltonian.py**: `RydbergHamiltonian` class with global/local addressing

**Tests**: 36 tests passing (`tests/test_physics.py`)

### 3. ML Methods (`qneural/neural/`)
- **models.py**: 
  - `FeedForwardNN` - Configurable feedforward networks
  - `PulseGenerator` - NN → pulse sequences
  - `TimeOptimalController` - Two-network system (time + pulses)
- **losses.py**: 
  - `InfidelityLoss`, `TimePenaltyLoss`
  - `CompositeLoss` - Weighted combination
- **solvers.py**: 
  - Abstract `ODESolver` interface
  - `TorchDiffeqSolver` implementation
  - Placeholder for `DiffraxSolver` (JAX backend)
- **pulse_generator.py**: `PhysicalPulseGenerator` - NN outputs → callable pulses
- **evolution.py**: `QuantumEvolver` - Pulses → Hamiltonian → evolution → corrections
- **trainer.py**: `QuantumTrainer`, `TimeOptimalTrainer` - Training loops

**Tests**: 30+ tests (`tests/test_neural.py`)

### 4. Gate Implementations (`qneural/gates/rydberg/`)
- **controlled_phase.py**: 
  - **GENERALIZED FRAMEWORK** - N-controlled phase gates
  - `ControlledPhaseGate` - Abstract base (arbitrary N controls)
  - `CZPhiGate` - 2-qubit specialization
  - `CCZPhiGate` - 3-qubit specialization
  - `ControlledPhaseOptimizer` - Unified optimizer
  - Factory functions: `create_czphi_optimizer()`, `create_cczphi_optimizer()`

**Key Achievement**: Instead of separate 2-qubit and 3-qubit code (like original), everything is generalized!

**Tests**: 20+ tests (`tests/test_gates.py`)

---

## 📁 Project Structure

```
qneural/
├── qneural/                      # Main package
│   ├── __init__.py
│   ├── config.py                 # Global configuration (DEVICE, dtypes)
│   ├── backend/
│   │   ├── __init__.py
│   │   └── torch_backend.py      # PyTorch abstraction (~60 functions)
│   ├── core/                     # Hardware-agnostic quantum operations
│   │   ├── __init__.py
│   │   ├── states.py
│   │   ├── gates.py
│   │   ├── operators.py
│   │   ├── metrics.py
│   │   └── evolution.py
│   ├── hardware/
│   │   └── rydberg/              # Rydberg atom specifics
│   │       ├── __init__.py
│   │       ├── constants.py
│   │       ├── operators.py
│   │       ├── pulses.py
│   │       └── hamiltonian.py
│   ├── neural/                   # ML methods
│   │   ├── __init__.py
│   │   ├── models.py             # FeedForwardNN, TimeOptimalController
│   │   ├── losses.py             # InfidelityLoss, CompositeLoss
│   │   ├── solvers.py            # ODESolver interface
│   │   ├── pulse_generator.py    # PhysicalPulseGenerator
│   │   ├── evolution.py          # QuantumEvolver
│   │   └── trainer.py            # QuantumTrainer
│   └── gates/
│       ├── __init__.py
│       └── rydberg/
│           ├── __init__.py
│           └── controlled_phase.py  # Generalized gate framework
│
├── tests/
│   ├── test_core_operations.py   # 38 tests
│   ├── test_physics.py           # 36 tests
│   ├── test_neural.py            # 30+ tests
│   └── test_gates.py             # 20+ tests
│
├── main_version/                 # ORIGINAL CODE - DO NOT MODIFY
│   ├── cst_n_fn.py
│   ├── schsolve.py
│   ├── const_czphi.py
│   └── const_cczphi.py
│
├── AGENTS.md                     # Coding guidelines (CRITICAL)
├── README.md
└── STRUCTURE.md
```

---

## 🚨 CRITICAL: READ AGENTS.md

**File**: `/home/madhav22m/gitrepos/qneural/AGENTS.md`

This contains essential coding guidelines:
- Build/test commands
- Import ordering
- Docstring style (Google-style)
- Type hints requirement
- Naming conventions
- Testing patterns (AAA)
- Architecture principles

**Run tests with**: `python -m pytest tests/ -v`

---

## 🎯 Next Steps / TODO

### High Priority

1. **Examples & Tutorials** (`examples/` directory)
   - Basic single-qubit rotation
   - CZ_φ optimization walkthrough
   - Time-optimal control example
   - Visualization utilities (pulse plotting)

2. **Integration Tests**
   - Reproduce original CZ_φ results from `main_version/`
   - Verify against published paper results
   - End-to-end validation

3. **Documentation**
   - API reference (can generate from docstrings)
   - Physics background document
   - Migration guide from old code

4. **Additional Features**
   - Checkpointing/resuming training
   - Learning rate schedulers
   - Early stopping
   - TensorBoard logging

### Medium Priority

5. **More Gate Types**
   - Single-qubit rotations (X, Y, Z)
   - SWAP gates
   - iSWAP gates
   - General SU(4) gates

6. **Hardware Extensions**
   - Superconducting qubits module (`hardware/superconducting/`)
   - Trapped ions module (`hardware/ions/`)

7. **Advanced ML**
   - Reinforcement learning approach
   - Gradient-free methods (CMA-ES)
   - Multi-objective optimization

### Low Priority / Future

8. **JAX Backend**
   - Implement `DiffraxSolver`
   - JAX versions of all functions

9. **Performance**
   - GPU optimization
   - Parallel angle evaluation
   - JIT compilation

---

## 🔑 Key Design Patterns

### 1. Backend Abstraction
```python
from ..backend import backend
# Use backend.matmul(), backend.eye() instead of torch directly
```

### 2. Hardware-Agnostic Core
- Physics in `core/` - pure quantum mechanics
- Hardware specifics in `hardware/<platform>/`

### 3. Generalized Gates
Instead of separate 2-qubit/3-qubit code:
```python
class ControlledPhaseGate:
    def __init__(self, n_controls, n_targets): ...
    # Works for any N!

CZPhiGate = ControlledPhaseGate(n_controls=1, n_targets=1)
CCZPhiGate = ControlledPhaseGate(n_controls=2, n_targets=1)
```

### 4. Configurable Loss
```python
loss_fn = CompositeLoss([
    (InfidelityLoss(nqubits=2), 1.0),
    (TimePenaltyLoss(weight=0.1), 0.1)
])
```

### 5. Abstract ODE Solvers
```python
class ODESolver(ABC): ...
class TorchDiffeqSolver(ODESolver): ...
# Easy to add DiffraxSolver later!
```

---

## 💻 Usage Examples

### Basic CZ_φ Optimization
```python
from qneural.gates import create_czphi_optimizer
import torch

# Create optimizer
optimizer = create_czphi_optimizer(
    time_optimal=True,
    time_bounds=(3.0, 8.0)  # in units of 1/Ω_max
)

# Train
history = optimizer.train(
    angle_range=(0.1 * torch.pi, torch.pi),
    n_angles=80,
    epochs=1000
)

# Evaluate
result = optimizer.evaluate(torch.pi / 2)
print(f"Infidelity: {result['infidelity']}")
print(f"Gate time: {result['gate_time']}")
```

### Manual Pulse Generation
```python
from qneural.neural import FeedForwardNN, PhysicalPulseGenerator
from qneural.neural import create_evolver

# Create network
network = FeedForwardNN(input_dim=2, output_dim=2)
pulse_gen = PhysicalPulseGenerator(
    n_controls=2,
    n_time_steps=201,
    control_ranges=[(0, 25.0), (-50.0, 50.0)]
)

# Generate pulses
angle = torch.tensor([0.5 * torch.pi])
time_points = torch.linspace(0, 1, 201)
inputs = torch.stack([angle.repeat(201), time_points], dim=1)
nn_output = network(inputs).reshape(201, 2)

pulses = pulse_gen.generate(nn_output, gate_time=5.0)

# Evolve
evol = create_evolver(nqubits=2)
final_U = evol.evolve(pulses, gate_time=5.0)
```

---

## ⚠️ Important Notes

1. **NEVER modify `main_version/`** - It's the original reference code
2. **Always use type hints** on public functions
3. **Follow AAA pattern** for tests (Arrange, Act, Assert)
4. **Use absolute imports** with proper nesting (`from ..core import ...`)
5. **Google-style docstrings** with Parameters, Returns, Examples
6. **Complex numbers**: Use `torch.cfloat` and `1.0j` for imaginary unit
7. **Device handling**: Use `device` parameter or `DEVICE` from config

---

## 🧪 Testing Strategy

**All tests should pass before committing:**
```bash
python -m pytest tests/ -v
```

**Run specific test file:**
```bash
python -m pytest tests/test_gates.py -v
```

**Run specific test:**
```bash
python -m pytest tests/test_gates.py::TestCZPhiGate::test_cz_at_pi -v
```

**Current test counts:**
- Core: 38 tests
- Physics: 36 tests
- Neural: 30+ tests
- Gates: 20+ tests
- **Total: 124+ tests passing**

---

## 📚 Key Files to Understand

1. **`qneural/core/gates.py`** - Gate constructions
2. **`qneural/core/metrics.py`** - Fidelity calculations
3. **`qneural/hardware/rydberg/hamiltonian.py`** - Time-dependent Hamiltonian
4. **`qneural/neural/models.py`** - Neural network architectures
5. **`qneural/neural/trainer.py`** - Training loop
6. **`qneural/gates/rydberg/controlled_phase.py`** - Generalized gate framework

---

## 🐛 Known Issues / TODOs

1. **LSP errors** in some files (non-blocking, mostly typing issues)
2. **Batching** in neural module needs more thorough testing
3. **Time-optimal training** needs integration with the new gate framework
4. **Visualization** utilities not yet implemented
5. **Checkpoint saving/loading** basic but could be enhanced

---

## 🎯 Immediate Next Task Recommendation

**Create a working example** in `examples/czphi_tutorial.py` that:
1. Creates a CZ_φ optimizer
2. Trains for a small number of epochs (e.g., 100)
3. Generates and plots pulses
4. Evaluates gate fidelity
5. Compares to target unitary

This will validate the full pipeline works end-to-end and provide a template for users.

---

## 📞 Questions?

The codebase is well-documented with docstrings. Key patterns:
- Abstract base classes for extensibility
- Factory functions for easy object creation
- Configurable components (losses, solvers, etc.)
- Comprehensive test coverage

Good luck! The foundation is solid - time to build on top of it! 🚀