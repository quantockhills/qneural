# Testing Guide for qneural

## Why Test?

Testing ensures your code:
1. **Works correctly** - catches bugs before users do
2. **Keeps working** - prevents regressions when you add features
3. **Is well-designed** - hard-to-test code is usually poorly designed
4. **Documents behavior** - tests show how to use your code

## Testing Pyramid

```
         /\
        /  \  E2E Tests (few)
       /____\
      /      \
     / Integration \ (some)
    /_______________\
   /                 \
  /   Unit Tests      \ (many)
 /____________________\
```

### 1. Unit Tests (Foundation)
- Test **individual functions** in isolation
- Fast to run (milliseconds)
- Easy to debug - pinpoints exact problem
- **Most of your tests should be unit tests**

### 2. Integration Tests (Middle)
- Test **multiple components** working together
- Slower (seconds to minutes)
- Test realistic workflows

### 3. End-to-End (E2E) Tests (Top)
- Test **entire system** from start to finish
- Slowest (minutes to hours)
- Fewest in number, highest confidence

## Testing Frameworks

### pytest (Recommended for Python)
- **Simple**: Just write functions starting with `test_`
- **Powerful**: Fixtures, parametrization, plugins
- **Standard**: Industry standard for Python

### Alternatives
- **unittest**: Built into Python, more verbose
- **nose2**: Similar to pytest, less popular
- **doctest**: Tests in docstrings, good for examples

## Test Structure: AAA Pattern

Every test follows **Arrange → Act → Assert**:

```python
def test_czphi_gate_at_pi():
    # ARRANGE: Set up test data
    angle = torch.pi

    # ACT: Perform the operation
    result = czphi_gate(angle)

    # ASSERT: Verify the result
    expected = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.cfloat))
    assert torch.allclose(result, expected)
```

## Types of Tests

### 1. Smoke Tests
"Does it even run?"
```python
def test_imports():
    """Test that package imports without errors."""
    import qneural
    assert qneural.__version__
```

### 2. Happy Path Tests
"Does it work correctly with valid inputs?"
```python
def test_basis_tensor_creates_valid_state():
    ket_0 = basis_tensor('0')
    assert ket_0.shape == (3, 1)
    assert ket_0[0, 0] == 1.0
```

### 3. Edge Case Tests
"What about boundary conditions?"
```python
def test_czphi_gate_at_zero():
    """Test CZ_φ at φ=0 (should be identity)."""
    gate = czphi_gate(0.0)
    assert torch.allclose(gate, torch.eye(4, dtype=torch.cfloat))

def test_czphi_gate_at_pi():
    """Test CZ_φ at φ=π (should be CZ)."""
    gate = czphi_gate(torch.pi)
    # ... check it's the CZ gate
```

### 4. Error Handling Tests
"Does it fail gracefully with bad inputs?"
```python
def test_basis_tensor_invalid_dimension():
    with pytest.raises(ValueError):
        basis_tensor('0', dim=5)  # Unsupported dimension
```

### 5. Property-Based Tests
"Does it satisfy mathematical properties?"
```python
def test_unitary_is_actually_unitary():
    """U U† = I for any gate."""
    U = czphi_gate(0.5)
    identity = torch.matmul(U, U.conj().T)
    assert torch.allclose(identity, torch.eye(4, dtype=torch.cfloat))
```

### 6. Regression Tests
"Does this specific bug stay fixed?"
```python
def test_issue_42_fidelity_calculation():
    """Regression test for issue #42: fidelity was > 1."""
    # Specific case that caused bug
    U1 = czphi_gate(1.234)
    U2 = czphi_gate(5.678)
    fidelity = unitary_fidelity(U1, U2, nqubits=2)
    assert fidelity <= 1.0
```

## Test Organization

### File Structure
```
tests/
├── __init__.py
├── conftest.py              # pytest fixtures (shared test helpers)
├── test_core_operations.py  # Tests for core module
├── test_rydberg_hamiltonian.py
├── test_neural_networks.py
└── integration/
    ├── test_czphi_optimization.py
    └── test_pulse_generation.py
```

### Test Class Organization
Group related tests into classes:
```python
class TestBasisStates:
    """Tests for quantum state creation."""

    def test_single_qubit(self):
        ...

    def test_two_qubits(self):
        ...

class TestGates:
    """Tests for gate construction."""

    def test_czphi_gate(self):
        ...
```

## pytest Features

### 1. Fixtures (Reusable Setup)
```python
@pytest.fixture
def sample_hamiltonian():
    """Create a sample Hamiltonian for tests."""
    return RydbergHamiltonian(nqubits=2)

def test_time_evolution(sample_hamiltonian):
    # Use the fixture
    result = sample_hamiltonian.evolve(...)
    assert ...
```

### 2. Parametrization (Multiple Test Cases)
```python
@pytest.mark.parametrize("angle,expected_phase", [
    (0.0, 1.0),
    (torch.pi/2, 1j),
    (torch.pi, -1.0),
])
def test_czphi_phases(angle, expected_phase):
    gate = czphi_gate(angle)
    assert gate[3, 3] == expected_phase
```

### 3. Markers (Categorize Tests)
```python
@pytest.mark.slow
def test_large_system_evolution():
    """This test takes 10 minutes."""
    ...

@pytest.mark.integration
def test_full_optimization_pipeline():
    ...

# Run only fast tests: pytest -m "not slow"
# Run only integration tests: pytest -m integration
```

## Test Coverage

### What to Aim For
- **Core functionality**: 90%+ coverage
- **Edge cases**: All important branches
- **Don't chase 100%**: Some code (like error messages) doesn't need tests

### Measuring Coverage
```bash
pip install pytest-cov
pytest --cov=qneural --cov-report=html
# Open htmlcov/index.html to see what's not tested
```

## Best Practices

### ✅ DO:
1. **Test behavior, not implementation**
   ```python
   # GOOD: Tests what it does
   def test_czphi_is_diagonal():
       gate = czphi_gate(0.5)
       off_diag = gate - torch.diag(torch.diag(gate))
       assert torch.allclose(off_diag, torch.zeros_like(off_diag))

   # BAD: Tests how it does it
   def test_czphi_uses_torch_eye():
       # Don't test implementation details
       ...
   ```

2. **One concept per test**
   ```python
   # GOOD: Focused test
   def test_fidelity_of_identical_gates_is_one():
       U = czphi_gate(0.5)
       assert unitary_fidelity(U, U) == 1.0

   # BAD: Testing multiple things
   def test_fidelity():
       # Tests identity, different gates, and edge cases
       ...  # Too much in one test
   ```

3. **Use descriptive names**
   ```python
   # GOOD
   def test_basis_tensor_raises_error_for_invalid_dimension():
       ...

   # BAD
   def test_bt():
       ...
   ```

4. **Make tests fast**
   - Mock expensive operations (network, disk I/O)
   - Use small test cases
   - Run slow tests separately

5. **Make tests deterministic**
   - Set random seeds
   - Avoid time-dependent behavior
   - Tests should pass/fail consistently

### ❌ DON'T:
1. **Don't test library code**
   - Don't test that PyTorch works
   - Test *your* code using PyTorch

2. **Don't have tests depend on each other**
   ```python
   # BAD: Order-dependent tests
   def test_a():
       global state
       state = initialize()

   def test_b():
       # Assumes test_a ran first
       assert state.valid
   ```

3. **Don't use production data in tests**
   - Use small, synthetic test data
   - Tests should be reproducible

## Continuous Integration (CI)

Run tests automatically on every commit:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -e .[dev]
      - run: pytest --cov=qneural
```

## Interview Questions & Answers

### Q: "How do you test your code?"
**A**: "I use pytest for unit, integration, and E2E tests. I follow the testing pyramid - mostly unit tests for fast feedback, some integration tests for workflows, and a few E2E tests for critical paths. I aim for 80-90% coverage of core functionality and use CI to run tests on every commit."

### Q: "What makes a good test?"
**A**: "A good test is:
1. **Fast** - runs in milliseconds
2. **Isolated** - doesn't depend on other tests
3. **Repeatable** - same result every time
4. **Readable** - clear what's being tested
5. **Maintainable** - easy to update when code changes"

### Q: "What's the difference between unit and integration tests?"
**A**: "Unit tests test individual functions in isolation - like testing that `czphi_gate(π)` returns a CZ matrix. Integration tests test multiple components together - like testing that a neural network + ODE solver + Hamiltonian can optimize a full gate. Unit tests are faster and easier to debug; integration tests give more confidence the system works end-to-end."

### Q: "How do you test code that uses randomness?"
**A**: "I set a fixed random seed for reproducibility:
```python
def test_random_optimization():
    torch.manual_seed(42)  # Fixed seed
    result = optimize_with_random_init()
    assert result.loss < 0.01
```
Or I test properties that should hold regardless:
```python
def test_random_state_is_normalized():
    state = generate_random_state()
    assert torch.allclose(state.norm(), torch.tensor(1.0))
```"

### Q: "What's test-driven development (TDD)?"
**A**: "Write tests *before* code:
1. **Red**: Write a failing test
2. **Green**: Write minimal code to pass
3. **Refactor**: Clean up the code

I don't always do strict TDD, but I find writing tests early helps me think through the API design."

## Example Test File Structure

```python
"""
Tests for core quantum operations.
"""
import pytest
import torch

# Organize into classes
class TestBasisStates:
    """Tests for state creation."""

    def test_single_qubit(self): ...
    def test_two_qubits(self): ...

    @pytest.mark.parametrize("state_str,expected_index", [
        ('0', 0),
        ('1', 1),
        ('r', 2),
    ])
    def test_basis_indices(self, state_str, expected_index): ...

class TestGates:
    """Tests for gate construction."""

    def test_czphi_at_zero(self): ...
    def test_czphi_at_pi(self): ...

    @pytest.mark.slow
    def test_large_gate_stack(self): ...

class TestMetrics:
    """Tests for fidelity calculations."""

    def test_perfect_fidelity(self): ...
    def test_fidelity_bounds(self): ...

# Integration tests
class TestIntegration:
    """Integration tests for full workflows."""

    @pytest.mark.integration
    def test_full_optimization_pipeline(self): ...
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://testdriven.io/blog/testing-best-practices/)
- [Effective Python Testing with pytest](https://realpython.com/pytest-python-testing/)

## Summary

1. **Use pytest** - industry standard, simple yet powerful
2. **Follow AAA pattern** - Arrange, Act, Assert
3. **Test pyramid** - many unit tests, some integration, few E2E
4. **Be systematic** - test happy paths, edge cases, errors
5. **Make tests FIRST** - Fast, Isolated, Repeatable, Self-validating, Timely
6. **Integrate with CI** - run tests automatically

**For interviews**: Know the testing pyramid, AAA pattern, and be able to write a simple unit test on a whiteboard!
