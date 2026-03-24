"""
Time-dependent pulse and control functions for quantum control.

This module provides time-dependent control functions used in quantum optimal control:
    - Constant pulses
    - Piecewise-constant pulses
    - Differentiable pulse shapes

All functions support both single values and batch processing for neural network training.
"""

import torch
from typing import Union, Callable, Optional
from ...config import DEVICE, DTYPE_REAL, DTYPE_COMPLEX


# =============================================================================
# Base Pulse Functions
# =============================================================================

def zero_pulse(t: Union[float, torch.Tensor], device: Optional[str] = None) -> torch.Tensor:
    """
    Zero pulse (no drive).

    Parameters
    ----------
    t : float or torch.Tensor
        Time (ignored, returns 0)
    device : str, optional
        Device to place tensor on

    Returns
    -------
    torch.Tensor
        Scalar 0.0
    """
    return torch.tensor(0.0, dtype=DTYPE_REAL, device=device or DEVICE)


def constant_pulse(amplitude: float, device: Optional[str] = None) -> Callable:
    """
    Create a constant pulse function.

    Parameters
    ----------
    amplitude : float
        Constant amplitude value
    device : str, optional
        Device to place tensor on

    Returns
    -------
    Callable
        Function f(t) = amplitude (ignores t)

    Examples
    --------
    >>> pulse = constant_pulse(2.0 * torch.pi * 4)  # 4 MHz Rabi
    >>> pulse(0.0)  # Returns constant amplitude
    tensor(25.1327)
    """
    val = torch.tensor(amplitude, dtype=DTYPE_REAL, device=device or DEVICE)

    def pulse_fn(t):
        return val

    return pulse_fn


def piecewise_constant(
    values: torch.Tensor,
    total_time: float,
    device: Optional[str] = None
) -> Callable:
    """
    Create a piecewise-constant pulse from a list of values.

    Parameters
    ----------
    values : torch.Tensor
        Pulse values at each time step, shape [time_steps] or [batch, time_steps]
    total_time : float
        Total gate time
    device : str, optional
        Device to place tensor on

    Returns
    -------
    Callable
        Function f(t) that returns the pulse value at time t

    Examples
    --------
    >>> values = torch.tensor([1.0, 2.0, 3.0])
    >>> pulse = piecewise_constant(values, total_time=3.0)
    >>> pulse(0.5)  # Returns 1.0 (first segment)
    >>> pulse(1.5)  # Returns 2.0 (second segment)
    """
    device = device or DEVICE
    step_size = total_time / values.shape[-1]

    def pulse_fn(t):
        # Determine step index
        if isinstance(t, torch.Tensor):
            step_idx = torch.floor(t / step_size).long()
            step_idx = torch.clamp(step_idx, 0, values.shape[-1] - 1)
        else:
            step_idx = min(int(t // step_size), values.shape[-1] - 1)

        if values.dim() == 1:
            # Single values array
            return values[step_idx]
        else:
            # Batched values [batch, time_steps]
            batch_size = values.shape[0]
            if isinstance(step_idx, torch.Tensor):
                # Batched time - gather for each batch element
                batch_indices = torch.arange(batch_size, device=device)
                return values[batch_indices, step_idx]
            else:
                # Scalar time - return value for this step from all batches
                return values[:, step_idx]

    return pulse_fn


def piecewise_constant_nn_output(
    nn_output: torch.Tensor,
    gate_time: float,
    time_steps: int,
    device: Optional[str] = None
) -> Callable:
    """
    Create a piecewise-constant pulse from neural network output.

    This is the standard interface for neural network-based pulse generation.
    The NN outputs a flat tensor that is reshaped into [batch, time_steps].

    Parameters
    ----------
    nn_output : torch.Tensor
        Neural network output, shape [batch * time_steps] or [batch, time_steps]
    gate_time : float
        Total gate time
    time_steps : int
        Number of time steps
    device : str, optional
        Device to place tensor on

    Returns
    -------
    Callable
        Function f(t) that returns pulse amplitude at time t for each batch element

    Examples
    --------
    >>> # nn_output shape: [40] for batch=10, time_steps=4
    >>> pulse_fn = piecewise_constant_nn_output(nn_output, gate_time=1.0, time_steps=4)
    >>> pulse_fn(0.3)  # Returns tensor shape [10, 1]
    """
    device = device or DEVICE
    step_size = gate_time / time_steps

    # Ensure shape [batch, time_steps]
    if nn_output.dim() == 1:
        batch_size = nn_output.shape[0] // time_steps
        values = nn_output.reshape(batch_size, time_steps)
    else:
        values = nn_output

    def pulse_fn(t):
        # Determine time step
        step_idx = torch.floor(torch.tensor(t / step_size)).long()
        step_idx = torch.clamp(step_idx, 0, time_steps - 1)

        # Gather values for each batch element at this time step
        batch_size = values.shape[0]
        return values[:, step_idx].reshape(batch_size, 1)

    return pulse_fn


def create_simple_detuning_pulse(
    values: torch.Tensor,
    gate_time: float
) -> Callable:
    """
    Create a simple detuning pulse function for single-angle evolution.

    This is the canonical formula used in FixedRabiTrainer for consistency.
    Uses the formula: idx = int(t / gate_time * (len(values) - 1))

    Parameters
    ----------
    values : torch.Tensor
        Detuning values at each time step, shape [n_time_steps]
    gate_time : float
        Total gate time in seconds

    Returns
    -------
    Callable
        Function f(t) that returns detuning value at time t

    Examples
    --------
    >>> detuning_vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> pulse = create_simple_detuning_pulse(detuning_vals, gate_time=1.0)
    >>> pulse(0.0)   # Returns 1.0 (first value)
    >>> pulse(0.5)   # Returns 2.0 (middle value)
    >>> pulse(1.0)   # Returns 4.0 (last value)
    """
    def pulse_fn(t):
        idx = int(t / gate_time * (len(values) - 1))
        idx = min(idx, len(values) - 1)
        return values[idx]
    return pulse_fn


# =============================================================================
# Smooth Pulse Shapes
# =============================================================================

def gaussian_pulse(
    amplitude: float,
    center: float,
    width: float,
    device: Optional[str] = None
) -> Callable:
    """
    Create a Gaussian-shaped pulse.

    Parameters
    ----------
    amplitude : float
        Peak amplitude
    center : float
        Center time of pulse
    width : float
        Width (standard deviation) of Gaussian
    device : str, optional
        Device to place tensor on

    Returns
    -------
    Callable
        Function f(t) = amplitude * exp(-(t-center)^2 / (2*width^2))

    Examples
    --------
    >>> pulse = gaussian_pulse(amplitude=1.0, center=0.5, width=0.1)
    >>> pulse(0.5)  # Peak value
    tensor(1.0)
    """
    device = device or DEVICE
    amp = torch.tensor(amplitude, dtype=DTYPE_REAL, device=device)
    c = torch.tensor(center, dtype=DTYPE_REAL, device=device)
    w = torch.tensor(width, dtype=DTYPE_REAL, device=device)

    def pulse_fn(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=DTYPE_REAL, device=device)
        return amp * torch.exp(-((t - c) ** 2) / (2 * w ** 2))

    return pulse_fn


def blackman_pulse(
    amplitude: float,
    duration: float,
    device: Optional[str] = None
) -> Callable:
    """
    Create a Blackman window pulse (smooth start/end to reduce leakage).

    The Blackman window has excellent spectral properties with minimal sidelobes.

    Parameters
    ----------
    amplitude : float
        Peak amplitude
    duration : float
        Total pulse duration
    device : str, optional
        Device to place tensor on

    Returns
    -------
    Callable
        Blackman window function

    Examples
    --------
    >>> pulse = blackman_pulse(amplitude=1.0, duration=1.0)
    >>> pulse(0.0)  # ~0 (smooth start)
    >>> pulse(0.5)  # Peak
    """
    device = device or DEVICE
    amp = torch.tensor(amplitude, dtype=DTYPE_REAL, device=device)
    T = torch.tensor(duration, dtype=DTYPE_REAL, device=device)

    # Blackman coefficients
    a0 = 7938/18608
    a1 = 9240/18608
    a2 = 1430/18608

    def pulse_fn(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=DTYPE_REAL, device=device)

        # Normalized time [0, 1]
        x = t / T

        # Blackman window
        window = a0 - a1 * torch.cos(2 * torch.pi * x) + a2 * torch.cos(4 * torch.pi * x)

        # Zero outside [0, T]
        window = torch.where((t >= 0) & (t <= T), window, torch.zeros_like(window))

        return amp * window

    return pulse_fn


# =============================================================================
# Composite Pulses
# =============================================================================

def cutoff_pulse(
    pulse_fn: Callable,
    cutoff_time: float,
    device: Optional[str] = None
) -> Callable:
    """
    Create a pulse that is active only up to a cutoff time.

    Parameters
    ----------
    pulse_fn : Callable
        Original pulse function
    cutoff_time : float
        Time after which pulse is zero
    device : str, optional
        Device to place tensor on

    Returns
    -------
    Callable
        Modified pulse function that returns 0 for t > cutoff_time
    """
    cutoff = torch.tensor(cutoff_time, dtype=DTYPE_REAL, device=device or DEVICE)

    def modified_fn(t):
        if isinstance(t, torch.Tensor):
            result = pulse_fn(t)
            result = torch.where(t <= cutoff, result, torch.zeros_like(result))
            return result
        else:
            if t <= cutoff_time:
                return pulse_fn(t)
            else:
                return torch.tensor(0.0, dtype=DTYPE_REAL, device=device or DEVICE)

    return modified_fn


def add_pulses(*pulse_fns: Callable) -> Callable:
    """
    Sum multiple pulse functions.

    Parameters
    ----------
    *pulse_fns : Callable
        Pulse functions to sum

    Returns
    -------
    Callable
        Function that returns the sum of all pulses

    Examples
    --------
    >>> pulse1 = constant_pulse(1.0)
    >>> pulse2 = gaussian_pulse(0.5, 0.5, 0.1)
    >>> combined = add_pulses(pulse1, pulse2)
    """
    def summed_fn(t):
        result = None
        for fn in pulse_fns:
            val = fn(t)
            if result is None:
                result = val
            else:
                result = result + val
        return result

    return summed_fn


# =============================================================================
# Utility Functions
# =============================================================================

def pulse_area(pulse_fn: Callable, t_start: float, t_end: float, n_points: int = 1000) -> torch.Tensor:
    """
    Compute the area (integral) of a pulse over a time interval.

    Parameters
    ----------
    pulse_fn : Callable
        Pulse function
    t_start : float
        Start time
    t_end : float
        End time
    n_points : int
        Number of points for numerical integration

    Returns
    -------
    torch.Tensor
        Pulse area (integral of pulse_fn from t_start to t_end)
    """
    t = torch.linspace(t_start, t_end, n_points)
    dt = (t_end - t_start) / (n_points - 1)
    values = torch.stack([pulse_fn(ti) for ti in t])
    return torch.trapz(values, t)


def normalize_pulse(pulse_fn: Callable, target_area: float, t_start: float, t_end: float) -> Callable:
    """
    Normalize a pulse to have a specific area.

    Parameters
    ----------
    pulse_fn : Callable
        Original pulse function
    target_area : float
        Desired pulse area
    t_start, t_end : float
        Time interval for normalization

    Returns
    -------
    Callable
        Normalized pulse function
    """
    current_area = pulse_area(pulse_fn, t_start, t_end)
    scale = target_area / current_area

    def normalized_fn(t):
        return scale * pulse_fn(t)

    return normalized_fn