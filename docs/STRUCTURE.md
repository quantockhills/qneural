# Package Structure Overview

This document explains the organization of the qneural package and the rationale behind the design.

## Current Progress (Side-by-Side Build)

### ✅ Completed Modules

```
qneural/
├── __init__.py                 # Package initialization with backend detection
├── config.py                   # Centralized configuration and constants
│
├── backend/                    # Backend abstraction layer
│   ├── __init__.py
│   └── torch_backend.py       # PyTorch implementation (complete)
│
├── hardware/                   # Hardware-specific implementations
│   ├── __init__.py
│   └── rydberg/               # Rydberg atom platform
│       ├── __init__.py
│       ├── constants.py       # Physical constants (Rabi, V_dd, decay, etc.)
│       └── operators.py       # Rydberg-specific operators
│
└── core/                      # Hardware-agnostic quantum operations
    ├── __init__.py
    └── states.py              # Basis states, tensor products, state manipulation
```

### 🔄 In Progress

- `hardware/rydberg/hamiltonian.py` - Migrate `Rydberg_Hamiltonian` class
- `core/gates.py` - Gate construction functions
- `core/operators.py` - Pauli matrices, general operators
- `core/metrics.py` - Fidelity and infidelity calculations

### 📋 Planned

- `ml/neural/` - Neural network architectures
- `ml/solvers/` - ODE solvers
- `ml/optimizers.py` - AdaBound and other optimizers
- `gates/rydberg/` - CZphi and CCZphi implementations
- `control/pulses.py` - Pulse generation utilities
- `utils/` - Plotting, I/O, conversions

## Design Principles

### 1. Backend Abstraction

**Goal**: Support multiple computational frameworks (PyTorch, JAX, NumPy)

**Implementation**:
- `backend/` provides unified interface for tensor operations
- All core modules use `backend` instead of directly calling PyTorch
- Easy to add new backends by implementing the same interface

**Example**:
```python
from qneural.backend import backend

# This code works regardless of backend
tensor = backend.zeros((3, 3))
result = backend.matmul(A, B)
```

### 2. Hardware Modularity

**Goal**: Support multiple quantum hardware platforms

**Implementation**:
- Each platform is a submodule of `hardware/`
- Platform-specific constants and physics in separate files
- Common interface for Hamiltonians across platforms

**Current**: `hardware/rydberg/`
**Planned**: `hardware/superconducting/`, `hardware/ions/`

**Structure within each hardware module**:
```
hardware/platform_name/
├── __init__.py
├── constants.py      # Physical parameters
├── operators.py      # Platform-specific operators
├── hamiltonian.py    # Hamiltonian class
└── noise.py          # Noise and decoherence models (future)
```

### 3. ML Method Flexibility

**Goal**: Support neural networks, RL, gradient-free optimization

**Implementation**:
- `ml/` contains all machine learning methods
- `ml/neural/` for neural network approaches (current focus)
- `ml/rl/` for reinforcement learning (future)
- `ml/gradient_free/` for CMA-ES, etc. (future)

### 4. Separation of Research and Library Code

**Goal**: Keep working research notebooks separate from production code

**Implementation**:
```
main_version/          # Original code (untouched!)
research/              # Organized research notebooks
qneural/               # Clean library code
examples/              # Polished tutorials
```

**Workflow**:
1. Research happens in `research/` notebooks
2. Once methods are validated, extract to `qneural/` library
3. Create clean examples for `examples/`
4. Original `main_version/` preserved for reference

## Migration Strategy

### Phase 1: Core Infrastructure (Current)
- ✅ Package skeleton
- ✅ Configuration system
- ✅ Backend abstraction (PyTorch)
- ✅ Rydberg constants and operators
- 🔄 Core quantum operations

### Phase 2: Physics Engine
- Rydberg Hamiltonian class
- Time evolution (ODE solvers)
- Gate construction and metrics
- Pulse generation utilities

### Phase 3: ML Components
- Neural network architectures
- Training loops and optimizers
- Loss functions and metrics
- Checkpoint/resume functionality

### Phase 4: Gate Implementations
- CZphi gate optimizer (2-qubit)
- CCZphi gate optimizer (3-qubit)
- Multi-angle families
- Time-optimal protocols

### Phase 5: Polish
- Complete documentation
- Comprehensive tests
- Examples and tutorials
- Verification against original code

## File Naming Conventions

### Original → New Package

| Original File | New Location | Notes |
|--------------|--------------|-------|
| `cst_n_fn.py` | Multiple modules | Split into `core/`, `hardware/rydberg/`, `control/pulses.py` |
| `schsolve.py` | `ml/neural/`, `ml/solvers/` | Architectures and ODE solvers |
| `const_czphi.py` | `gates/rydberg/czphi.py` | 2-qubit gate implementation |
| `const_cczphi.py` | `gates/rydberg/cczphi.py` | 3-qubit gate implementation |
| `adabound.py` | `ml/optimizers.py` | Custom optimizers |

### Rationale for Splits

**`cst_n_fn.py` (589 lines)** was doing too much:
- Physical constants → `hardware/rydberg/constants.py`
- Rydberg operators → `hardware/rydberg/operators.py`
- Hamiltonian class → `hardware/rydberg/hamiltonian.py`
- Basis states → `core/states.py`
- Gate construction → `core/gates.py`
- Metrics → `core/metrics.py`
- Pulse functions → `control/pulses.py`
- Utilities → `utils/conversions.py`, `utils/plotting.py`

## Compatibility Layer

To ensure your research notebooks continue working, we'll create:

```python
# qneural/compat/cst_n_fn.py
"""
Compatibility layer for original cst_n_fn imports.
Allows existing code to work unchanged.
"""
from ..hardware.rydberg.constants import *
from ..hardware.rydberg.operators import *
from ..core.states import *
from ..core.gates import *
# ... etc
```

Usage in notebooks:
```python
# Old way (still works)
import cst_n_fn as cfn

# New way (recommended)
from qneural.hardware.rydberg import RydbergHamiltonian
from qneural.core import basis_tensor, unitary_infidelity
```

## Testing Strategy

Each module will have corresponding tests:

```
tests/
├── test_backend.py          # Backend operations
├── test_rydberg_ops.py      # Rydberg operators
├── test_hamiltonian.py      # Hamiltonian time evolution
├── test_states.py           # State manipulation
├── test_gates.py            # Gate construction
├── test_metrics.py          # Fidelity calculations
└── integration/             # End-to-end tests
    ├── test_czphi.py        # Reproduce CZphi results
    └── test_cczphi.py       # Reproduce CCZphi results
```

## Next Steps

1. **Complete core modules**: gates.py, operators.py, metrics.py
2. **Migrate Hamiltonian**: Create `hardware/rydberg/hamiltonian.py`
3. **Set up ODE solvers**: `ml/solvers/torch_solver.py`
4. **Migrate neural networks**: `ml/neural/architectures.py`
5. **Create gate implementations**: `gates/rydberg/czphi.py`
6. **Test against original code**: Verify results match
7. **Update research notebooks**: Import from new package
8. **Write documentation**: API docs, tutorials, theory

---

**Questions or suggestions?** This structure is designed for long-term maintainability and extensibility. Feedback welcome!
