"""
Configuration and default constants for qneural.

This module provides centralized configuration for:
    - Backend settings (device, dtype)
    - Physical constants (hardware-specific defaults in hardware modules)
    - Numerical parameters (ODE solver tolerances, discretization)
    - Optimization defaults (batch sizes, time steps)

Users can override these settings at runtime or via environment variables.
"""

import torch
import os

# =============================================================================
# Backend Configuration
# =============================================================================

# Device selection: 'cpu', 'cuda', or 'mps'
DEVICE = os.getenv("QNEURAL_DEVICE", "cpu")

# Default dtype for real-valued tensors
DTYPE_REAL = torch.float32

# Default dtype for complex-valued tensors
DTYPE_COMPLEX = torch.cfloat

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


def set_device(device: str):
    """
    Set the global device for computations.

    Parameters
    ----------
    device : str
        Device identifier: 'cpu', 'cuda', or 'mps'
    """
    global DEVICE
    if device not in ["cpu", "cuda", "mps"]:
        raise ValueError(f"Invalid device: {device}. Must be 'cpu', 'cuda', or 'mps'.")
    DEVICE = device


def get_device():
    """
    Get the current global device.

    Returns
    -------
    str
        Current device identifier
    """
    return DEVICE


def set_precision(precision: str):
    """
    Set numerical precision for the framework.

    Parameters
    ----------
    precision : str
        Either 'single' (float32) or 'double' (float64)
    """
    global DTYPE_REAL, DTYPE_COMPLEX
    if precision == "single":
        DTYPE_REAL = torch.float32
        DTYPE_COMPLEX = torch.cfloat
    elif precision == "double":
        DTYPE_REAL = torch.float64
        DTYPE_COMPLEX = torch.cdouble
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
    print(f"Device:              {DEVICE}")
    print(f"Real dtype:          {DTYPE_REAL}")
    print(f"Complex dtype:       {DTYPE_COMPLEX}")
    print(f"ODE solver:          {ODE_SOLVER_DEFAULT}")
    print(f"ODE tolerances:      rtol={ODE_RTOL_DEFAULT}, atol={ODE_ATOL_DEFAULT}")
    print(f"Default time steps:  {TIME_STEPS_DEFAULT}")
    print(f"Default angle batch: {ANGLE_BATCH_DEFAULT}")
    print("=" * 60)
