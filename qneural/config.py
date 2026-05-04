"""
Configuration and default constants for qneural.

This module provides centralized configuration for:
    - Backend settings (device, dtype)
    - Physical constants (hardware-specific defaults in hardware modules)
    - Numerical parameters (ODE solver tolerances, discretization)
    - Optimization defaults (batch sizes, time steps)

Users can override these settings at runtime or via environment variables.
"""

import os

# =============================================================================
# Backend Configuration
# =============================================================================

BACKEND = os.getenv("QNEURAL_BACKEND", "pytorch")

_device = os.getenv("QNEURAL_DEVICE", "cpu")
DTYPE_REAL = None
DTYPE_COMPLEX = None

if BACKEND == "pytorch":
    import torch
    DEVICE = _device
    DTYPE_REAL = torch.float32
    DTYPE_COMPLEX = torch.cfloat
elif BACKEND == "tensorflow":
    import tensorflow as tf
    DEVICE = "cpu" if _device == "cpu" else f"/{_device}:0"
    DTYPE_REAL = tf.float32
    DTYPE_COMPLEX = tf.complex64
elif BACKEND == "jax":
    import jax.numpy as jnp
    DEVICE = _device
    DTYPE_REAL = jnp.float32
    DTYPE_COMPLEX = jnp.complex64

# =============================================================================
# Numerical Defaults
# =============================================================================

# ODE solver defaults
ODE_SOLVER_DEFAULT = "dopri5"  # Dormand-Prince adaptive step size solver
ODE_RTOL_DEFAULT = 1e-6  # Relative tolerance
ODE_ATOL_DEFAULT = 1e-6  # Absolute tolerance

# Discretization defaults
TIME_STEPS_DEFAULT = 201  # Number of time discretization points
ANGLE_BATCH_DEFAULT = 80  # Default batch size for angle sampling

# =============================================================================
# Machine Learning Defaults
# =============================================================================

# Neural network architecture defaults
NN_HIDDEN_LAYERS_DEFAULT = 6
NN_HIDDEN_UNITS_DEFAULT = 150
NN_ACTIVATION_DEFAULT = "relu"
NN_OUTPUT_ACTIVATION_DEFAULT = "sigmoid"

# Training defaults
LEARNING_RATE_DEFAULT = 1e-4
OPTIMIZER_DEFAULT = "adam"
BATCH_SIZE_DEFAULT = 1

# =============================================================================
# Physical Constants (General)
# =============================================================================
# Note: Hardware-specific constants (e.g., Rydberg Rabi frequencies, interaction
# strengths) are defined in their respective hardware modules (e.g., hardware/rydberg/constants.py)

# Planck constant (for reference, not used in dimensionless calculations)
HBAR = 1.054571817e-34  # J·s

# =============================================================================
# Utility Functions
# =============================================================================


def set_backend(backend: str):
    """
    Set the computational backend.

    Parameters
    ----------
    backend : str
        Backend identifier: 'pytorch', 'tensorflow', or 'jax'
    """
    global BACKEND, DEVICE, DTYPE_REAL, DTYPE_COMPLEX
    if backend not in ["pytorch", "tensorflow", "jax"]:
        raise ValueError(
            f"Invalid backend: {backend}. Must be 'pytorch', 'tensorflow', or 'jax'."
        )
    BACKEND = backend
    if backend == "pytorch":
        import torch
        DEVICE = "cpu"
        DTYPE_REAL = torch.float32
        DTYPE_COMPLEX = torch.cfloat
    elif backend == "tensorflow":
        import tensorflow as tf
        DEVICE = "cpu"
        DTYPE_REAL = tf.float32
        DTYPE_COMPLEX = tf.complex64
    elif backend == "jax":
        import jax.numpy as jnp
        DEVICE = "cpu"
        DTYPE_REAL = jnp.float32
        DTYPE_COMPLEX = jnp.complex64

    from .backend import _reinit_backend
    _reinit_backend()


def set_device(device: str):
    """Set the global device for computations."""
    global _device, DEVICE
    _device = device
    if BACKEND == "tensorflow":
        DEVICE = "cpu" if device == "cpu" else f"/{device}:0"
    else:
        DEVICE = device


def get_device():
    """Get the current global device."""
    return DEVICE


def get_backend():
    """Get the current computational backend."""
    return BACKEND


def set_precision(precision: str):
    """Set numerical precision for the framework."""
    global DTYPE_REAL, DTYPE_COMPLEX
    if BACKEND == "pytorch":
        import torch
        if precision == "single":
            DTYPE_REAL = torch.float32
            DTYPE_COMPLEX = torch.cfloat
        elif precision == "double":
            DTYPE_REAL = torch.float64
            DTYPE_COMPLEX = torch.cdouble
    elif BACKEND == "tensorflow":
        import tensorflow as tf
        if precision == "single":
            DTYPE_REAL = tf.float32
            DTYPE_COMPLEX = tf.complex64
        elif precision == "double":
            DTYPE_REAL = tf.float64
            DTYPE_COMPLEX = tf.complex128
    elif BACKEND == "jax":
        import jax.numpy as jnp
        if precision == "single":
            DTYPE_REAL = jnp.float32
            DTYPE_COMPLEX = jnp.complex64
        elif precision == "double":
            DTYPE_REAL = jnp.float64
            DTYPE_COMPLEX = jnp.complex128
    else:
        raise ValueError(
            f"Invalid precision: {precision}. Must be 'single' or 'double'."
        )


# =============================================================================
# Display Configuration
# =============================================================================


def print_config():
    """Print current configuration settings."""
    print("=" * 60)
    print("qneural Configuration")
    print("=" * 60)
    print(f"Backend:             {BACKEND}")
    print(f"Device:              {DEVICE}")
    print(f"Real dtype:          {DTYPE_REAL}")
    print(f"Complex dtype:       {DTYPE_COMPLEX}")
    print(f"ODE solver:          {ODE_SOLVER_DEFAULT}")
    print(f"ODE tolerances:      rtol={ODE_RTOL_DEFAULT}, atol={ODE_ATOL_DEFAULT}")
    print(f"Default time steps:  {TIME_STEPS_DEFAULT}")
    print(f"Default angle batch: {ANGLE_BATCH_DEFAULT}")
    print("=" * 60)
