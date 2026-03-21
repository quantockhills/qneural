"""
Neural network architectures for quantum control.

Provides feedforward networks for generating quantum control pulses.
Supports configurable depth, width, activations, and output constraints.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union


class FeedForwardNN(nn.Module):
    """
    Configurable feedforward neural network for quantum control.
    
    Architecture: Input → [Linear → Activation] × n → Output
    
    Parameters
    ----------
    input_dim : int
        Input dimension (e.g., 2 for angle + time)
    output_dim : int
        Output dimension (e.g., 1 for single pulse, 2 for rabi + detuning)
    hidden_layers : int
        Number of hidden layers (default: 2)
    hidden_units : int
        Number of units per hidden layer (default: 64)
    activation : str
        Hidden layer activation: 'relu', 'tanh', 'sigmoid' (default: 'relu')
    output_activation : str
        Output activation: 'sigmoid', 'tanh', 'none' (default: 'sigmoid')
    use_batch_norm : bool
        Whether to use batch normalization (default: False)
    weight_scale : float
        Scale factor for weight initialization (default: 1.0)
    
    Examples
    --------
    >>> # Simple network for single pulse output
    >>> net = FeedForwardNN(input_dim=2, output_dim=1)
    >>> 
    >>> # Deeper network with tanh output for time bounds
    >>> net = FeedForwardNN(
    ...     input_dim=1, output_dim=1,
    ...     hidden_layers=6, hidden_units=150,
    ...     output_activation='tanh'
    ... )
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 64,
        activation: str = 'relu',
        output_activation: str = 'sigmoid',
        use_batch_norm: bool = False,
        weight_scale: float = 1.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_units))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_units))
        layers.append(self._get_activation(activation))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(self._get_activation(activation))
        
        # Output layer
        layers.append(nn.Linear(hidden_units, output_dim))
        
        # Output activation (if specified)
        if output_activation != 'none':
            layers.append(self._get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights(weight_scale)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'none': nn.Identity()
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name]
    
    def _initialize_weights(self, scale: float):
        """Initialize weights with optional scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if scale != 1.0:
                    module.weight.data *= scale
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape [batch_size, input_dim] or [input_dim]
        
        Returns
        -------
        torch.Tensor
            Output tensor, shape [batch_size, output_dim] or [output_dim]
        """
        # Handle single input
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        output = self.network(x)
        
        # Return to original shape if single input
        if single_input:
            output = output.squeeze(0)
        
        return output
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PulseGenerator(nn.Module):
    """
    Generates quantum control pulses from input parameters.
    
    Maps (angle, time) → (control_values) where control_values
    are normalized to [0, 1] and can be scaled to physical ranges.
    
    Parameters
    ----------
    n_controls : int
        Number of control outputs (e.g., 2 for rabi + detuning)
    n_time_steps : int
        Number of discretized time steps
    hidden_layers : int
        Number of hidden layers
    hidden_units : int
        Number of units per hidden layer
    
    Examples
    --------
    >>> generator = PulseGenerator(n_controls=2, n_time_steps=201)
    >>> 
    >>> # Generate pulses for angle=0.5π at 201 time points
    >>> angle = torch.tensor([0.5 * torch.pi])
    >>> time_points = torch.linspace(0, 1, 201)
    >>> pulses = generator(angle, time_points)
    """
    
    def __init__(
        self,
        n_controls: int = 2,
        n_time_steps: int = 201,
        hidden_layers: int = 6,
        hidden_units: int = 150
    ):
        super().__init__()
        
        self.n_controls = n_controls
        self.n_time_steps = n_time_steps
        
        # Input: (angle, normalized_time), Output: n_controls
        self.network = FeedForwardNN(
            input_dim=2,
            output_dim=n_controls,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            activation='relu',
            output_activation='sigmoid'
        )
    
    def forward(
        self,
        angle: torch.Tensor,
        normalized_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate pulses for given angle and time points.
        
        Parameters
        ----------
        angle : torch.Tensor
            Gate angle(s), shape [] or [batch_size]
        normalized_time : torch.Tensor
            Normalized time points [0, 1], shape [n_time_steps]
        
        Returns
        -------
        torch.Tensor
            Control pulses, shape [batch_size, n_time_steps, n_controls]
        """
        # Ensure angle is batched
        if angle.dim() == 0:
            angle = angle.unsqueeze(0)
        
        batch_size = angle.shape[0]
        n_times = normalized_time.shape[0]
        
        # Create input tensor: [angle, time] for each time point
        # Shape: [batch_size * n_time_steps, 2]
        angle_repeated = angle.repeat_interleave(n_times)
        time_repeated = normalized_time.repeat(batch_size)
        
        inputs = torch.stack([angle_repeated, time_repeated], dim=1)
        
        # Generate controls
        outputs = self.network(inputs)  # [batch_size * n_time_steps, n_controls]
        
        # Reshape to [batch_size, n_time_steps, n_controls]
        outputs = outputs.reshape(batch_size, n_times, self.n_controls)
        
        return outputs


class TimeOptimalController(nn.Module):
    """
    Two-network system for time-optimal quantum control.
    
    Combines:
    1. Time predictor: angle → gate_time
    2. Pulse generator: (angle, normalized_time) → controls
    
    This allows optimization of both pulse shape AND gate duration.
    
    Parameters
    ----------
    time_bounds : tuple[float, float]
        Min and max gate time (physical units)
    n_controls : int
        Number of control outputs
    n_time_steps : int
        Number of discretized time steps
    time_hidden_layers : int
        Hidden layers for time predictor
    time_hidden_units : int
        Hidden units for time predictor
    pulse_hidden_layers : int
        Hidden layers for pulse generator
    pulse_hidden_units : int
        Hidden units for pulse generator
    time_output_activation : str
        Activation for time output ('sigmoid', 'tanh', etc.)
    
    Examples
    --------
    >>> controller = TimeOptimalController(
    ...     time_bounds=(3.0, 8.0),
    ...     n_controls=2,
    ...     n_time_steps=201
    ... )
    >>> 
    >>> # Generate time-optimal pulses
    >>> angle = torch.tensor([0.5 * torch.pi])
    >>> gate_time, pulses = controller(angle)
    """
    
    def __init__(
        self,
        time_bounds: tuple,
        n_controls: int = 2,
        n_time_steps: int = 201,
        time_hidden_layers: int = 3,
        time_hidden_units: int = 40,
        pulse_hidden_layers: int = 6,
        pulse_hidden_units: int = 150,
        time_output_activation: str = 'sigmoid'
    ):
        super().__init__()
        
        self.time_bounds = time_bounds
        self.n_controls = n_controls
        self.n_time_steps = n_time_steps
        
        # Time predictor: angle → normalized_time
        self.time_predictor = FeedForwardNN(
            input_dim=1,
            output_dim=1,
            hidden_layers=time_hidden_layers,
            hidden_units=time_hidden_units,
            activation='relu',
            output_activation=time_output_activation
        )
        
        # Pulse generator: (angle, normalized_time) → controls
        self.pulse_generator = PulseGenerator(
            n_controls=n_controls,
            n_time_steps=n_time_steps,
            hidden_layers=pulse_hidden_layers,
            hidden_units=pulse_hidden_units
        )
        
        # Time grid for discretization [0, 1]
        self.register_buffer(
            'time_grid',
            torch.linspace(0, 1, n_time_steps)
        )
    
    def forward(self, angle: torch.Tensor) -> tuple:
        """
        Generate time-optimal pulses.
        
        Parameters
        ----------
        angle : torch.Tensor
            Gate angle, shape [] or [batch_size]
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - gate_time: Predicted gate time [batch_size]
            - pulses: Control pulses [batch_size, n_time_steps, n_controls]
        """
        # Predict normalized time [0, 1] or [-1, 1]
        normalized_time = self.time_predictor(angle)
        
        # Scale to physical time bounds
        t_min, t_max = self.time_bounds
        if self.time_predictor.network[-1].__class__.__name__ == 'Tanh':
            # Tanh outputs [-1, 1], scale to [t_min, t_max]
            gate_time = 0.5 * (normalized_time + 1) * (t_max - t_min) + t_min
        else:
            # Sigmoid outputs [0, 1], scale to [t_min, t_max]
            gate_time = normalized_time * (t_max - t_min) + t_min
        
        # Generate pulses using the predicted time
        pulses = self.pulse_generator(angle, self.time_grid)
        
        return gate_time, pulses