# AGENTS.md - Coding Guidelines for qneural

## Build & Test Commands

```bash
# Install dependencies
pip install -e .

# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_core_operations.py -v

# Run single test (most common)
python -m pytest tests/test_core_operations.py::TestGateConstruction::test_czphi_gate_at_pi -v

# Run tests matching pattern
python -m pytest tests/ -k "test_czphi" -v

# Run with coverage
python -m pytest tests/ --cov=qneural --cov-report=html

# Run only fast tests (exclude slow/integration)
python -m pytest tests/ -m "not slow and not integration" -v
```

## Code Style Guidelines

### 1. Imports

**Order:** Standard library → Third-party → Local (qneural)

```python
# Standard library
import torch
import numpy as np
from typing import Union, Callable, Optional

# Third-party
import torchdiffeq as tde

# Local - use absolute imports with proper nesting
from ..backend import backend
from ..config import DEVICE, DTYPE_COMPLEX
from ..core.states import basis_tensor, tensor_product
```

**Rules:**
- Use explicit imports (no `import *` except in `__init__.py`)
- Group related imports
- Use `functools as ft`, `numpy as np`, `torchdiffeq as tde`

### 2. Docstrings

Use Google-style docstrings with sections: Description, Parameters, Returns, Examples

```python
def basis_tensor(state_str, dim=3, device=None):
    """
    Create a basis state tensor from a string representation.

    Parameters
    ----------
    state_str : str
        State string, e.g., '001', '01r', 'rr0'
    dim : int, optional
        Local Hilbert space dimension (default: 3)
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        Basis state as column vector, shape [dim^n, 1]

    Examples
    --------
    >>> basis_tensor('0', dim=3)  # |0⟩
    >>> basis_tensor('01', dim=3)  # |0⟩⊗|1⟩
    """
```

### 3. Type Hints

Always use type hints for function signatures:

```python
from typing import Union, Callable, Optional, Tuple

def schrodinger_evolution(
    initial_state: torch.Tensor,
    hamiltonian_fn: Callable[[float], torch.Tensor],
    t_span: Tuple[float, float],
    method: str = 'dopri5'
) -> torch.Tensor:
```

### 4. Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Functions | snake_case | `basis_tensor`, `czphi_gate` |
| Classes | PascalCase | `RydbergHamiltonian`, `NeuralTrainer` |
| Constants | UPPER_SNAKE_CASE | `RABI_DEFAULT`, `HILBERT_DIM_GG` |
| Private functions | _leading_underscore | `_build_interaction_operator` |
| Type variables | PascalCase | `T`, `StateType` |

### 5. Error Handling

Raise specific exceptions with descriptive messages:

```python
if addressing not in ['global', 'local']:
    raise ValueError(f"Unknown addressing mode: {addressing}. Must be 'global' or 'local'.")

if dim != 3:
    raise ValueError(f"Hyperfine operators only defined for GG-qubits (dim=3), got dim={dim}")
```

### 6. Testing - AAA Pattern

Every test follows **Arrange → Act → Assert**:

```python
def test_czphi_gate_at_pi():
    # ARRANGE
    phi = torch.pi
    
    # ACT
    gate = czphi_gate(phi)
    
    # ASSERT
    expected = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.cfloat))
    assert torch.allclose(gate, expected)
```

**Test Organization:**
- Group tests in classes: `class TestGateConstruction:`
- Use descriptive test names: `test_czphi_gate_is_diagonal`
- Use parametrization for multiple cases
- Mark slow tests: `@pytest.mark.slow`

### 7. File Organization

```python
"""
Module docstring explaining purpose.
"""

# Imports
import torch
from typing import ...

# =============================================================================
# Section Header
# =============================================================================

# Public functions/classes

def public_function():
    """Docstring."""
    pass

# =============================================================================
# Private/Helper Functions
# =============================================================================

def _helper_function():
    """Private helper."""
    pass
```

### 8. Code Patterns

**Backend abstraction:**
```python
from ..backend import backend
# Use backend.zeros(), backend.matmul() instead of torch directly
```

**Device/dtype handling:**
```python
from ..config import DEVICE, DTYPE_COMPLEX, DTYPE_REAL
# Always specify device and dtype for new tensors
tensor = torch.zeros((3, 3), dtype=DTYPE_COMPLEX, device=device or DEVICE)
```

**Complex number handling:**
```python
# Use 1.0j for imaginary unit
phase = torch.exp(1.0j * theta)

# Convert real to complex
if not torch.is_complex(tensor):
    tensor = tensor.to(dtype=DTYPE_COMPLEX)
```

### 9. Project Structure Rules

- **NEVER modify `main_version/`** - it's the original reference code
- Build new code in `qneural/` side-by-side
- Keep `qneural/` hardware-agnostic in `core/`, hardware-specific in `hardware/`
- All ML methods go in `ml/` (to be created)
- Each module has corresponding tests in `tests/`

### 10. Pre-commit Checklist

Before submitting code:
- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] Type hints present on all public functions
- [ ] Docstrings follow Google style
- [ ] No `import *` (except `__init__.py`)
- [ ] Error messages are descriptive
- [ ] Constants use UPPER_SNAPE_CASE
- [ ] Functions use snake_case, classes use PascalCase

## Architecture Principles

1. **Backend-agnostic:** Use `backend/` abstraction, not direct torch calls
2. **Hardware-agnostic core:** Physics in `core/`, hardware-specific in `hardware/`
3. **Differentiable everything:** All code must work with PyTorch autograd
4. **Batch-friendly:** Support batch operations for neural network training
5. **No side effects:** Functions should be pure (no global state modification)