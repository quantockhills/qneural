"""
Pulse generation from neural network outputs.

Converts raw NN outputs into callable pulse functions that can be used
with Hamiltonians for quantum evolution.
"""

import torch
from typing import Callable, Optional, Tuple, Union, List

from ..hardware.rydberg.pulses import piecewise_constant, constant_pulse


class PhysicalPulseGenerator:
    """
    Generates quantum control pulses from neural network outputs.
    
    This class bridges the gap between NN output (normalized [0, 1] values)
    and physical pulse functions (with proper units and ranges).
    
    Parameters
    ----------
    n_controls : int
        Number of control outputs (e.g., 2 for rabi + detuning)
    n_time_steps : int
        Number of discretized time steps
    control_ranges : List[Tuple[float, float]]
        Physical ranges for each control, e.g., [(0, rabi_max), (-delta_max, delta_max)]
    
    Examples
    --------
    >>> from qneural.neural.models import FeedForwardNN
    >>> 
    >>> # Create NN and pulse generator
    >>> nn = FeedForwardNN(input_dim=2, output_dim=2)
    >>> pulse_gen = PhysicalPulseGenerator(
    ...     n_controls=2,
    ...     n_time_steps=201,
    ...     control_ranges=[(0, 25.0), (-50.0, 50.0)]  # rabi, detuning ranges
    ... )
    >>> 
    >>> # Generate pulses for angle=0.5π
    >>> angle = torch.tensor([0.5 * torch.pi])
    >>> time_points = torch.linspace(0, 1, 201)
    >>> nn_output = nn(torch.stack([angle.repeat(201), time_points], dim=1))
    >>> rabi_fn, detuning_fn = pulse_gen.generate(nn_output, gate_time=5.0)
    """
    
    def __init__(
        self,
        n_controls: int,
        n_time_steps: int,
        control_ranges: List[Tuple[float, float]]
    ):
        self.n_controls = n_controls
        self.n_time_steps = n_time_steps
        self.control_ranges = control_ranges
        
        if len(control_ranges) != n_controls:
            raise ValueError(
                f"control_ranges must have {n_controls} elements, "
                f"got {len(control_ranges)}"
            )
    
    def scale_output(
        self,
        nn_output: torch.Tensor,
        control_idx: int
    ) -> torch.Tensor:
        """
        Scale NN output from [0, 1] to physical range.
        
        Parameters
        ----------
        nn_output : torch.Tensor
            NN output values in [0, 1], shape [..., 1] or [...]
        control_idx : int
            Index of control to scale
        
        Returns
        -------
        torch.Tensor
            Scaled values in physical units
        """
        range_min, range_max = self.control_ranges[control_idx]
        return nn_output * (range_max - range_min) + range_min
    
    def generate(
        self,
        nn_output: torch.Tensor,
        gate_time: Union[float, torch.Tensor]
    ) -> List[Callable]:
        """
        Generate callable pulse functions from NN output.
        
        Parameters
        ----------
        nn_output : torch.Tensor
            NN output values, shape [n_time_steps, n_controls] or [batch, n_time_steps, n_controls]
        gate_time : float or torch.Tensor
            Total gate time (physical units)
        
        Returns
        -------
        List[Callable]
            List of pulse functions, one per control
        
        Examples
        --------
        >>> # Single angle, 201 time steps, 2 controls
        >>> nn_output = torch.rand(201, 2)  # NN predictions
        >>> pulses = pulse_gen.generate(nn_output, gate_time=5.0)
        >>> 
        >>> # Evaluate at specific time
        >>> t = 2.5
        >>> rabi_val = pulses[0](t)
        >>> detuning_val = pulses[1](t)
        """
        # Handle different input shapes
        if nn_output.dim() == 2:
            # Single batch: [n_time_steps, n_controls]
            nn_output = nn_output.unsqueeze(0)  # [1, n_time_steps, n_controls]
            single_batch = True
        else:
            single_batch = False
        
        batch_size, n_times, n_controls = nn_output.shape
        
        if n_times != self.n_time_steps:
            raise ValueError(
                f"Expected {self.n_time_steps} time steps, got {n_times}"
            )
        
        if n_controls != self.n_controls:
            raise ValueError(
                f"Expected {self.n_controls} controls, got {n_controls}"
            )
        
        # Generate pulse functions for each control
        pulse_functions = []
        
        for i in range(self.n_controls):
            # Extract control values and scale to physical range
            control_values = nn_output[:, :, i]  # [batch_size, n_time_steps]
            
            # Scale from [0, 1] to physical range
            control_values_scaled = self.scale_output(control_values, i)
            
            # Create piecewise-constant pulse function
            if single_batch:
                # Remove batch dimension for single batch
                control_values_scaled = control_values_scaled.squeeze(0)
            
            pulse_fn = self._create_pulse_fn(
                control_values_scaled,
                gate_time
            )
            
            pulse_functions.append(pulse_fn)
        
        return pulse_functions
    
    def _create_pulse_fn(
        self,
        values: torch.Tensor,
        gate_time: Union[float, torch.Tensor]
    ) -> Callable:
        """Create a callable pulse function from values."""
        # Convert gate_time to tensor if needed
        if isinstance(gate_time, (int, float)):
            gate_time = torch.tensor(gate_time, dtype=values.dtype)
        
        # For batched values, gate_time should also be batched
        if values.dim() == 2 and gate_time.dim() == 0:
            # Expand gate_time for batch
            gate_time = gate_time.unsqueeze(0).expand(values.shape[0])
        
        return piecewise_constant(values, gate_time)


class TimeOptimalPulseGenerator(PhysicalPulseGenerator):
    """
    Pulse generator for time-optimal control.
    
    Extends base PulseGenerator with support for variable gate times
    predicted by a neural network.
    
    Parameters
    ----------
    n_controls : int
        Number of control outputs
    n_time_steps : int
        Number of discretized time steps
    control_ranges : List[Tuple[float, float]]
        Physical ranges for each control
    time_range : Tuple[float, float]
        Min and max gate times (physical units)
    
    Examples
    --------
    >>> pulse_gen = TimeOptimalPulseGenerator(
    ...     n_controls=2,
    ...     n_time_steps=201,
    ...     control_ranges=[(0, 25.0), (-50.0, 50.0)],
    ...     time_range=(3.0, 8.0)
    ... )
    >>> 
    >>> # Generate with variable gate time
    >>> nn_output = torch.rand(201, 2)
    >>> predicted_time = torch.tensor([5.5])
    >>> pulses = pulse_gen.generate(nn_output, predicted_time)
    """
    
    def __init__(
        self,
        n_controls: int,
        n_time_steps: int,
        control_ranges: List[Tuple[float, float]],
        time_range: Tuple[float, float]
    ):
        super().__init__(n_controls, n_time_steps, control_ranges)
        self.time_range = time_range
    
    def scale_time(self, normalized_time: torch.Tensor) -> torch.Tensor:
        """
        Scale normalized time prediction to physical time.
        
        Parameters
        ----------
        normalized_time : torch.Tensor
            Time in [0, 1] (sigmoid output) or [-1, 1] (tanh output)
        
        Returns
        -------
        torch.Tensor
            Physical gate time
        """
        t_min, t_max = self.time_range
        
        # Handle both [0, 1] and [-1, 1] normalized inputs
        if normalized_time.min() < 0:
            # Assume [-1, 1] range (tanh output)
            return 0.5 * (normalized_time + 1) * (t_max - t_min) + t_min
        else:
            # Assume [0, 1] range (sigmoid output)
            return normalized_time * (t_max - t_min) + t_min


class BatchedPulseGenerator:
    """
    Generates pulses for multiple angles simultaneously (batch processing).

    This is useful for training on angle families, where you optimize
    a single NN to work across a range of angles.

    Parameters
    ----------
    pulse_generator : PhysicalPulseGenerator
        Base pulse generator to use for each angle

    Examples
    --------
    >>> base_gen = PhysicalPulseGenerator(n_controls=2, n_time_steps=201, ...)
    >>> batched_gen = BatchedPulseGenerator(base_gen)
    >>>
    >>> # Generate for multiple angles
    >>> angles = torch.linspace(0.1, torch.pi, 80)
    >>> nn_outputs = torch.rand(80, 201, 2)  # [batch, time, controls]
    >>> all_pulses = batched_gen.generate_batch(nn_outputs, gate_times)
    """

    def __init__(self, pulse_generator: "PhysicalPulseGenerator"):
        self.pulse_generator = pulse_generator
    
    def generate_batch(
        self,
        nn_outputs: torch.Tensor,
        gate_times: torch.Tensor
    ) -> List[List[Callable]]:
        """
        Generate pulses for a batch of angles.
        
        Parameters
        ----------
        nn_outputs : torch.Tensor
            NN outputs, shape [batch_size, n_time_steps, n_controls]
        gate_times : torch.Tensor
            Gate times for each angle, shape [batch_size]
        
        Returns
        -------
        List[List[Callable]]
            List of pulse function lists, one per angle in batch
        """
        batch_size = nn_outputs.shape[0]
        
        all_pulses = []
        for i in range(batch_size):
            pulses = self.pulse_generator.generate(
                nn_outputs[i],  # [n_time_steps, n_controls]
                gate_times[i]   # scalar
            )
            all_pulses.append(pulses)
        
        return all_pulses


# Convenience functions

def create_default_physical_pulse_generator(
    rabi_max: float,
    detuning_range: Tuple[float, float] = None,
    n_time_steps: int = 201
) -> PulseGenerator:
    """
    Create a standard pulse generator for Rydberg systems.
    
    Parameters
    ----------
    rabi_max : float
        Maximum Rabi frequency
    detuning_range : Tuple[float, float], optional
        Detuning range (default: [-2*rabi_max, 2*rabi_max])
    n_time_steps : int
        Number of time steps
    
    Returns
    -------
    PulseGenerator
        Configured pulse generator
    
    Examples
    --------
    >>> pulse_gen = create_default_pulse_generator(rabi_max=25.0)
    >>> 
    >>> # Or with custom detuning range
    >>> pulse_gen = create_default_pulse_generator(
    ...     rabi_max=25.0,
    ...     detuning_range=(-50.0, 50.0)
    ... )
    """
    if detuning_range is None:
        detuning_range = (-2 * rabi_max, 2 * rabi_max)
    
    return PhysicalPulseGenerator(
        n_controls=2,  # rabi and detuning
        n_time_steps=n_time_steps,
        control_ranges=[(0, rabi_max), detuning_range]
    )


def pulses_to_hamiltonian(
    hamiltonian_class,
    pulse_functions: List[Callable],
    addressing: str = 'global',
    **hamiltonian_kwargs
):
    """
    Create a Hamiltonian with the given pulse functions.
    
    Parameters
    ----------
    hamiltonian_class : type
        Hamiltonian class (e.g., RydbergHamiltonian)
    pulse_functions : List[Callable]
        List of pulse functions [rabi_fn, detuning_fn]
    addressing : str
        'global' or 'local' addressing
    **hamiltonian_kwargs
        Additional arguments for Hamiltonian constructor
    
    Returns
    -------
    Hamiltonian instance
        Configured with the pulse functions
    
    Examples
    --------
    >>> from qneural.hardware.rydberg import RydbergHamiltonian
    >>> 
    >>> pulses = pulse_gen.generate(nn_output, gate_time=5.0)
    >>> ham = pulses_to_hamiltonian(
    ...     RydbergHamiltonian,
    ...     pulses,
    ...     nqubits=2,
    ...     addressing='global'
    ... )
    """
    if len(pulse_functions) != 2:
        raise ValueError("Expected exactly 2 pulse functions (rabi, detuning)")
    
    rabi_fn, detuning_fn = pulse_functions
    
    return hamiltonian_class(
        rabi_pulse=rabi_fn,
        detuning_pulse=detuning_fn,
        addressing=addressing,
        **hamiltonian_kwargs
    )