"""
Physical constants for Rydberg atom systems.

All frequencies are in angular units (rad/s or equivalently 2π × Hz).
"""

import numpy as np
import math

# =============================================================================
# Rabi Frequencies
# =============================================================================

RABI_DEFAULT = 2 * math.pi * 4.0

# Default Rabi frequency for ground-ground (hyperfine) transitions
# Units: 2π × 4 kHz (much weaker than Rydberg coupling)
RABI_GG_DEFAULT = 2 * np.pi * 4.0 * 1e-3

# =============================================================================
# Interaction Strengths
# =============================================================================

# Van der Waals interaction coefficient
# Typical value: V_dd ≈ 21.1 × Ω_max
# This is the interaction strength when two atoms are both in Rydberg state,
# normalized to the maximum Rabi frequency
VDD_COUPLING = 21.1

# =============================================================================
# Decoherence
# =============================================================================

# Rydberg state decay width (inverse lifetime)
# Units: same as Rabi frequency
# Typical value: Γ ≈ 2π × 10 kHz for Rydberg states
# Often expressed as lifetime τ: Γ = 1/τ
# For τ ≈ 96.5 μs at Ω_max = 2π × 4 MHz:
#   Γ = 1/(96.5 × 10 / 4) in units of Ω_max
DECAY_WIDTH_DEFAULT = 1.0 / (96.5 * 10 / 4)  # Default: ~241.25 μs effective

# =============================================================================
# State Basis
# =============================================================================

# Dimension of local Hilbert space
# For ground-Rydberg qubits: 2 (|0⟩, |r⟩)
# For ground-ground qubits: 3 (|0⟩, |1⟩, |r⟩)
HILBERT_DIM_GR = 2
HILBERT_DIM_GG = 3

# =============================================================================
# Utility Functions
# =============================================================================


def rabi_to_mhz(rabi_angular):
    """
    Convert Rabi frequency from angular units (rad/s) to MHz.

    Parameters
    ----------
    rabi_angular : float
        Rabi frequency in rad/s

    Returns
    -------
    float
        Rabi frequency in MHz
    """
    return rabi_angular / (2 * np.pi * 1e6)


def mhz_to_rabi(rabi_mhz):
    """
    Convert Rabi frequency from MHz to angular units (rad/s).

    Parameters
    ----------
    rabi_mhz : float
        Rabi frequency in MHz

    Returns
    -------
    float
        Rabi frequency in rad/s
    """
    return rabi_mhz * (2 * np.pi * 1e6)


def gatetime_to_us(gatetime_normalized, rabi_max=None):
    """
    Convert normalized gate time to microseconds.

    Parameters
    ----------
    gatetime_normalized : float
        Gate time in units of 1/Ω_max
    rabi_max : float, optional
        Maximum Rabi frequency (rad/s). If None, uses RABI_DEFAULT.

    Returns
    -------
    float
        Gate time in microseconds
    """
    if rabi_max is None:
        rabi_max = RABI_DEFAULT
    return gatetime_normalized * 1e6 / rabi_max


def us_to_gatetime(time_us, rabi_max=None):
    """
    Convert time in microseconds to normalized gate time.

    Parameters
    ----------
    time_us : float
        Time in microseconds
    rabi_max : float, optional
        Maximum Rabi frequency (rad/s). If None, uses RABI_DEFAULT.

    Returns
    -------
    float
        Normalized gate time (units of 1/Ω_max)
    """
    if rabi_max is None:
        rabi_max = RABI_DEFAULT
    return time_us * rabi_max / 1e6
