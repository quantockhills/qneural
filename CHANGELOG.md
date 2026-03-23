# Changelog

All notable changes to the qneural project.

## [Unreleased]

### Fixed
- **CRITICAL**: Double phase correction bug in `ControlledPhaseOptimizer.evaluate()`
  - Was applying phase corrections twice (in both compute_loss() and evaluate())
  - Result: Training stuck at ~40-60% fidelity
  - Fix: Remove redundant correction in evaluate()
  
- **CRITICAL**: Phase correction formula
  - Original: Independent diagonal phase corrections (incorrect)
  - Fixed: Symmetric correction using |01⟩ phase (matching original paper)
  - Result: Training now achieves **>99% fidelity**

- Training performance: Fixed gate time units (using normalized time instead of absolute seconds)
- Training now 100-200x faster (0.6s vs 90s for quick test)
- Fixed unitary evolution NaN issue by using complex ODEs directly (no real/imag conversion)

### Added
- **FixedRabiTrainer** class for detuning-only optimization
  - Clean API for constant rabi + learned detuning
  - Achieves >99% fidelity on CZ gates reliably
  - Working example in `01_high_fidelity_cz_gate.ipynb`
  
- Comprehensive integration tests (18 physics validation tests)
- Jaksch protocol test validates Rydberg blockade mechanism
- All 142 tests passing (124 unit + 18 integration)

### Changed
- Updated documentation (README, VALIDATION_REPORT) to reflect working state
- FixedRabiTrainer now recommended for production use
- Phase correction formula now matches original research paper

## [2026-03-21]

### Fixed
- **CRITICAL**: Training speed issue resolved
  - Root cause: Gate time was in absolute seconds instead of normalized units (1/Ω_max)
  - With rabi=25, gate_time=7 meant 7 seconds instead of 7/25 ≈ 0.28 seconds
  - This caused ODE solver instability (step size too large for V_dd=530 interaction)
  - Fix: Convert normalized time to actual time: `gate_time = normalized_time / rabi_max`
  - Result: Training 100-200x faster (0.6s vs 90s for test)
  
- Complex ODE evolution
  - Changed from real/imag conversion to native complex tensors
  - Simplifies code and improves stability
  - Located in `qneural/core/evolution.py`

### Validated
- Jaksch protocol produces correct CZ gate
  - π-2π-π pulse sequence on 2-qubit system
  - Validates Rydberg blockade mechanism
  - Result: diag(1, -1, -1, -1) as expected

### Architecture
- Organized markdown documentation
  - Moved active docs to `docs/`
  - Created `planning/` folder structure
  - Added comprehensive plans for checkpoint/resume and visualization tools

## [2026-03-20]

### Added
- Complete physics validation suite
  - 18 integration tests covering:
    - Quantum evolution (unitarity, norm preservation)
    - Hamiltonian properties (Hermitian, dimensions)
    - Gate construction (CZ_φ, CCZ_φ)
    - Fidelity metrics
  - All tests passing

- Jaksch protocol implementation
  - Local addressing for Rydberg atoms
  - Sequential pulse application
  - Blockade mechanism validation

### Validated
- Core physics engine correctness
  - Schrödinger evolution preserves unitarity
  - Hamiltonian is Hermitian
  - Gates are unitary and have correct structure
  - Fidelity calculations match published formulas

## Project Foundation

### Original Migration (by OpenCode)
- Migrated from research code to professional package
- Created modular architecture:
  - Backend abstraction (PyTorch → JAX ready)
  - Hardware abstraction (Rydberg → extensible)
  - ML-method abstraction (NN → RL/gradient-free ready)
- Implemented core modules:
  - `qneural/core/` - Hardware-agnostic quantum ops
  - `qneural/hardware/rydberg/` - Rydberg specifics
  - `qneural/neural/` - ML methods
  - `qneural/gates/` - Generalized N-controlled phase gates
- Added comprehensive testing (124 unit tests)
- Created documentation structure

### Key Design Decisions
1. Side-by-side migration (original code preserved in `main_version/`)
2. Generalized N-controlled phase gates (not separate 2-qubit/3-qubit code)
3. Backend abstraction layer for future JAX support
4. Hardware-agnostic core operations
5. Comprehensive test coverage from day one

---

## Legend

- **Fixed**: Bug fixes
- **Added**: New features
- **Changed**: Modifications to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Deleted features
- **Security**: Security-related changes
- **Validated**: Physics/software correctness verification

---

**Last Updated:** March 23, 2026
