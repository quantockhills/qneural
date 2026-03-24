"""
Time-optimal quantum control with fixed Rabi frequency.

This module provides time-optimal control where:
- Rabi frequency is constant at maximum (by definition of time-optimal)
- Only detuning is learned
- Gate time is predicted and optimized jointly

Following the archival implementation patterns from cphase_optim.ipynb.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Callable, Dict
from pathlib import Path
import time

from ..core.gates import czphi_gate
from ..core.metrics import unitary_infidelity
from .solvers import ODESolver, TorchDiffeqSolver
from .evolution import QuantumEvolver, create_evolver


class TimeOptimalController(nn.Module):
    """
    Two-network system for time-optimal quantum control.
    
    Architecture:
    1. Time Predictor: angle → normalized_time [0,1]
    2. Control Generator: (angle, time) → detuning values
    
    The time network predicts optimal gate duration, which is fed into
    the control network along with the angle to generate detuning pulses.
    Rabi frequency is held constant at maximum (time-optimal by definition).
    
    Parameters
    ----------
    time_bounds : tuple[float, float]
        Min and max gate time in units of 1/rabi_max (default: (3.0, 20.0))
    rabi_max : float
        Maximum Rabi frequency (held constant)
    detuning_range : tuple[float, float], optional
        Min and max detuning in same units as rabi_max. 
        Defaults to (-2*rabi_max, 2*rabi_max)
    n_time_steps : int
        Number of discretized time steps (default: 301)
    time_hidden_layers : int
        Hidden layers in time network (default: 3)
    time_hidden_units : int
        Units per layer in time network (default: 45)
    control_hidden_layers : int
        Hidden layers in control network (default: 10)
    control_hidden_units : int
        Units per layer in control network (default: 300)
    time_output_activation : str
        'sigmoid' ([0,1]) or 'tanh' ([-1,1]) (default: 'sigmoid')
    weight_scale_time : float
        Weight init scale for time network (default: 1.8)
    weight_scale_control : float
        Weight init scale for control network (default: 1.55)
    
    Attributes
    ----------
    time_predictor : nn.Module
        Network predicting normalized gate time from angle
    control_generator : nn.Module
        Network generating detuning from (angle, time) pairs
    time_grid : torch.Tensor
        Normalized time grid [0, 1] with n_time_steps points
    
    Examples
    --------
    >>> controller = TimeOptimalController(
    ...     time_bounds=(3.0, 8.5),
    ...     rabi_max=25.13,
    ...     n_time_steps=301
    ... )
    >>> angle = torch.tensor([3.14159])  # CZ gate
    >>> gate_time, detuning = controller(angle)
    """
    
    def __init__(
        self,
        time_bounds: Tuple[float, float] = (3.0, 20.0),  # In units of 1/rabi_max
        rabi_max: float = 25.13,
        detuning_range: Optional[Tuple[float, float]] = None,
        n_time_steps: int = 301,
        time_hidden_layers: int = 3,
        time_hidden_units: int = 45,
        control_hidden_layers: int = 10,
        control_hidden_units: int = 300,
        time_output_activation: str = 'sigmoid',
        weight_scale_time: float = 1.8,
        weight_scale_control: float = 1.55
    ):
        super().__init__()
        
        self.time_bounds = time_bounds
        self.rabi_max = rabi_max
        self.n_time_steps = n_time_steps
        self.time_output_activation = time_output_activation
        
        # Default detuning range: ±2*rabi_max
        if detuning_range is None:
            detuning_range = (-2.0 * rabi_max, 2.0 * rabi_max)
        self.detuning_range = detuning_range
        
        # Build networks
        self.time_predictor = self._build_time_predictor(
            time_hidden_layers,
            time_hidden_units,
            time_output_activation,
            weight_scale_time
        )
        
        self.control_generator = self._build_control_generator(
            control_hidden_layers,
            control_hidden_units,
            weight_scale_control
        )
        
        # Fixed normalized time grid [0, 1]
        self.register_buffer('time_grid', torch.linspace(0, 1, n_time_steps))
        
        # Store last prediction
        self._last_predicted_time = None
    
    def _build_time_predictor(
        self,
        n_layers: int,
        n_units: int,
        output_activation: str,
        weight_scale: float
    ) -> nn.Module:
        """Build time predictor network: angle → normalized_time."""
        layers = []
        
        # Input: angle [batch, 1] → n_units
        layers.append(nn.Linear(1, n_units))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_units, n_units))
            layers.append(nn.ReLU())
        
        # Output: n_units → 1
        layers.append(nn.Linear(n_units, 1))
        
        # Output activation
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())
        else:
            raise ValueError(f"Unknown activation: {output_activation}")
        
        network = nn.Sequential(*layers)
        self._initialize_weights(network, weight_scale)
        return network
    
    def _build_control_generator(
        self,
        n_layers: int,
        n_units: int,
        weight_scale: float
    ) -> nn.Module:
        """Build control generator: (angle, time) → detuning."""
        layers = []
        
        # Input: [angle, time] → n_units
        layers.append(nn.Linear(2, n_units))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_units, n_units))
            layers.append(nn.ReLU())
        
        # Output: detuning only (Rabi is fixed!)
        layers.append(nn.Linear(n_units, 1))
        layers.append(nn.Sigmoid())  # [0, 1], scaled later
        
        network = nn.Sequential(*layers)
        self._initialize_weights(network, weight_scale)
        return network
    
    def _initialize_weights(self, network: nn.Module, scale: float):
        """Initialize with Xavier uniform and optional scaling."""
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if scale != 1.0:
                    module.weight.data *= scale
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, angle: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate time-optimal detuning pulses.
        
        Parameters
        ----------
        angle : torch.Tensor
            Gate angle(s), shape [batch] or [batch, 1]
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - gate_time: Predicted time [batch, 1] (in seconds)
            - detuning_normalized: Detuning values [batch, n_time_steps, 1] in [0, 1]
        """
        # Ensure proper shape [batch, 1]
        if angle.dim() == 0:
            angle = angle.unsqueeze(0)
        if angle.dim() == 1:
            angle = angle.unsqueeze(-1)
        
        batch_size = angle.shape[0]
        
        # Step 1: Predict normalized time from angle
        normalized_time = self.time_predictor(angle)  # [batch, 1]

        # Step 2: Scale to physical time bounds
        # time_bounds is expected in actual seconds
        t_min, t_max = self.time_bounds
        if self.time_output_activation == 'tanh':
            # [-1, 1] → [t_min, t_max]
            gate_time = 0.5 * (normalized_time + 1) * (t_max - t_min) + t_min
        else:
            # [0, 1] → [t_min, t_max]
            gate_time = normalized_time * (t_max - t_min) + t_min
        
        self._last_predicted_time = gate_time
        
        # Step 3: Create scaled time grid
        # time_grid: [n_time_steps] → [batch, n_time_steps]
        time_scaled = self.time_grid.unsqueeze(0) * gate_time  # [batch, n_time_steps]
        
        # Step 4: Prepare (angle, time) pairs for control network
        # Repeat angle for each time step
        angle_repeated = angle.repeat_interleave(self.n_time_steps, dim=0)  # [batch*n_time_steps, 1]
        time_flat = time_scaled.reshape(-1, 1)  # [batch*n_time_steps, 1]
        control_inputs = torch.cat([angle_repeated, time_flat], dim=-1)  # [batch*n_time_steps, 2]
        
        # Step 5: Generate detuning
        detuning_normalized = self.control_generator(control_inputs)  # [batch*n_time_steps, 1]
        detuning_normalized = detuning_normalized.reshape(batch_size, self.n_time_steps, 1)
        
        return gate_time, detuning_normalized
    
    def get_rabi_pulse_fn(self, gate_time: torch.Tensor) -> Callable:
        """
        Get constant-then-zero Rabi pulse function.
        
        Following archival pattern: constant at rabi_max until gate_time, then zero.
        
        Parameters
        ----------
        gate_time : torch.Tensor
            Gate time for each batch element [batch, 1] or scalar
        
        Returns
        -------
        Callable
            Function rabi(t) returning rabi_max for t <= gate_time, else 0
        """
        # Ensure tensor
        if not isinstance(gate_time, torch.Tensor):
            gate_time = torch.tensor(gate_time, dtype=torch.float32)
        
        # Handle batched gate_time
        batch_size = gate_time.numel()
        
        def rabi_pulse(t):
            """Rabi pulse: constant until cutoff, then zero."""
            # Handle both scalar and batched inputs
            if isinstance(t, torch.Tensor):
                device = t.device
            else:
                device = gate_time.device
                t = torch.tensor(t, device=device)
            
            if batch_size == 1:
                # Scalar case
                return torch.where(
                    t <= gate_time.item(),
                    torch.tensor(self.rabi_max, device=device),
                    torch.tensor(0.0, device=device)
                )
            else:
                # Batched case
                # gate_time shape: [batch, 1]
                # t is scalar time
                result = torch.full((batch_size, 1), self.rabi_max, device=device)
                mask = t > gate_time  # [batch, 1]
                result[mask] = 0.0
                return result
        
        return rabi_pulse
    
    def get_detuning_pulse_fn(
        self,
        detuning_normalized: torch.Tensor,
        gate_time: torch.Tensor
    ) -> Callable:
        """
        Get piecewise-constant detuning pulse function.
        
        Following archival list_to_fn_tensor_var_gatetime pattern.
        
        Parameters
        ----------
        detuning_normalized : torch.Tensor
            Normalized detuning [batch, n_time_steps, 1] in [0, 1]
        gate_time : torch.Tensor
            Gate time [batch, 1] or scalar
        
        Returns
        -------
        Callable
            Function detuning(t) returning detuning at time t
        """
        # Scale to physical range
        d_min, d_max = self.detuning_range
        detuning_values = detuning_normalized * (d_max - d_min) + d_min  # [batch, n_time_steps, 1]
        
        batch_size = detuning_values.shape[0]
        step_size = gate_time / self.n_time_steps  # [batch, 1] or scalar
        
        # Precompute the off-resonant value as a buffer
        off_resonant_val = torch.tensor(20.0 * self.rabi_max, device=detuning_values.device)
        
        def detuning_pulse(t):
            """
            Piecewise-constant detuning pulse.

            For batched case: Each batch element has its own pulse sequence and gate_time.
            We use the SAME time t but different step_size for each batch element.
            """
            if isinstance(t, torch.Tensor):
                device = t.device
            else:
                device = detuning_values.device
                t = torch.tensor(t, device=device)

            if batch_size == 1:
                # Scalar case
                if t >= gate_time.item():
                    return off_resonant_val
                idx = int(torch.floor(t / step_size).item())
                idx = min(idx, self.n_time_steps - 1)
                return detuning_values[0, idx, 0]
            else:
                # Batched case
                # Each batch element has step_size: [batch, 1]
                # t is scalar, so t / step_size gives [batch, 1]
                result = torch.full((batch_size, 1), 20.0 * self.rabi_max, device=device)

                # Compute time indices for each batch element
                indices = torch.floor(t / step_size).long()  # [batch, 1]
                indices = torch.clamp(indices, 0, self.n_time_steps - 1).squeeze(-1)  # [batch]

                # Check which are within gate time
                within_gate = (t < gate_time).squeeze(-1)  # [batch]

                # For each batch element within gate time, get its detuning value
                if within_gate.any():
                    batch_indices = torch.arange(batch_size, device=device)
                    within_indices = batch_indices[within_gate]  # Indices of valid batch elements
                    time_indices = indices[within_gate]  # Time indices for those elements
                    values = detuning_values[within_indices, time_indices, 0]  # [n_within]
                    result[within_gate] = values.unsqueeze(-1)  # [n_within, 1]

                return result

        return detuning_pulse
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each network."""
        time_params = sum(p.numel() for p in self.time_predictor.parameters())
        control_params = sum(p.numel() for p in self.control_generator.parameters())
        return {
            'time_predictor': time_params,
            'control_generator': control_params,
            'total': time_params + control_params
        }

    def scale_detuning(self, detuning_normalized: torch.Tensor) -> torch.Tensor:
        """
        Scale normalized detuning [0, 1] to physical range.
        
        Parameters
        ----------
        detuning_normalized : torch.Tensor
            Detuning values in [0, 1], shape [..., 1] or [...]
        
        Returns
        -------
        torch.Tensor
            Scaled detuning in physical units
        """
        d_min, d_max = self.detuning_range
        return detuning_normalized * (d_max - d_min) + d_min


class TimeOptimalTrainer:
    """
    Trainer for time-optimal quantum control with dual optimizers.
    
    Uses separate optimizers for time and control networks with different
    learning rates, following the archival training pattern.
    
    Parameters
    ----------
    controller : TimeOptimalController
        The controller with time and control networks
    nqubits : int
        Number of qubits (2 for CPHASE)
    time_weight : float
        Weight for time penalty in loss (default: 1e-4)
    time_lr : float
        Learning rate for time network (default: 1e-5)
    control_lr : float
        Learning rate for control network (default: 1e-4)
    solver : ODESolver, optional
        ODE solver (default: TorchDiffeqSolver with RK4)
    device : str
        Device to run on ('cpu' or 'cuda')
    
    Attributes
    ----------
    time_optimizer : torch.optim.Optimizer
        Optimizer for time network parameters
    control_optimizer : torch.optim.Optimizer
        Optimizer for control network parameters
    history : dict
        Training history with loss, infidelity, mean_gate_time
    
    Examples
    --------
    >>> controller = TimeOptimalController(time_bounds=(3.0, 8.5), rabi_max=25.13)
    >>> trainer = TimeOptimalTrainer(
    ...     controller=controller,
    ...     nqubits=2,
    ...     time_weight=1e-4,
    ...     time_lr=1e-5,
    ...     control_lr=1e-4
    ... )
    >>> angles = torch.linspace(0.1 * torch.pi, torch.pi, 80)
    >>> history = trainer.train(angles, epochs=1000)
    """
    
    def __init__(
        self,
        controller: TimeOptimalController,
        nqubits: int,
        time_weight: float = 1e-4,
        time_lr: float = 1e-5,
        control_lr: float = 1e-4,
        solver: Optional[ODESolver] = None,
        device: str = 'cpu'
    ):
        self.controller = controller.to(device)
        self.nqubits = nqubits
        self.time_weight = time_weight
        self.device = device
        
        # Separate optimizers with archival settings
        self.time_optimizer = torch.optim.Adam(
            controller.time_predictor.parameters(),
            lr=time_lr,
            eps=1e-5,
            amsgrad=True
        )
        
        self.control_optimizer = torch.optim.Adam(
            controller.control_generator.parameters(),
            lr=control_lr,
            eps=1e-5,
            amsgrad=True
        )
        
        # Quantum evolver - use RK4 (not dopri!) following archival
        if solver is None:
            self.evolver = create_evolver(
                nqubits=nqubits,
                backend='torchdiffeq',
                n_time_steps=controller.n_time_steps,
                method='rk4'
            )
        else:
            # Wrap solver in QuantumEvolver
            self.evolver = QuantumEvolver(
                nqubits=nqubits,
                solver=solver,
                n_time_steps=controller.n_time_steps
            )
        
        # Training history
        self.history = {
            'epoch': [],
            'loss': [],
            'infidelity': [],
            'mean_gate_time': []
        }
        self.current_epoch = 0
    
    def train(
        self,
        angles: torch.Tensor,
        epochs: int = 1000,
        print_every: int = 50,
        save_path: Optional[str] = None,
        angle_range: Optional[Tuple[float, float]] = None,
        resample_every: int = 25
    ) -> Dict:
        """
        Train the time-optimal controller.

        Parameters
        ----------
        angles : torch.Tensor
            Initial target angles [n_angles] (multi-angle optimization)
        epochs : int
            Number of training epochs
        print_every : int
            Print progress every N epochs
        save_path : str, optional
            Path to save best model
        angle_range : tuple, optional
            (min, max) angle range for resampling. If None, uses fixed angles.
        resample_every : int
            Resample angles from angle_range every N epochs (archival: 25)

        Returns
        -------
        dict
            Training history
        """
        angles = angles.to(self.device)
        best_loss = float('inf')
        batch_size = len(angles)

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Resample angles (archival pattern)
            if angle_range is not None and epoch % resample_every == 0:
                angles = torch.rand(batch_size, 1, device=self.device) * \
                        (angle_range[1] - angle_range[0]) + angle_range[0]

            # Training step
            loss, metrics = self._train_step(angles)

            # Update history
            self.history['epoch'].append(epoch)
            self.history['loss'].append(loss)
            self.history['infidelity'].append(metrics['infidelity'])
            self.history['mean_gate_time'].append(metrics['mean_gate_time'])

            # Print progress
            if epoch % print_every == 0:
                # Convert gate time to normalized Rabi units for display (archival pattern)
                mean_time_normalized = metrics['mean_gate_time'] * self.controller.rabi_max
                print(f"Epoch {epoch}: Loss = {loss:.6f}, "
                      f"Infidelity = {metrics['infidelity']:.6f}, "
                      f"Mean Time = {mean_time_normalized:.4f}")

            # Save best model
            if save_path and loss < best_loss:
                best_loss = loss
                self.save_checkpoint(save_path)

        return self.history
    
    def _train_step(self, angles: torch.Tensor) -> Tuple[float, Dict]:
        """
        Single training step with dual optimizers.

        BATCHED implementation following archival pattern:
        1. Zero both optimizers
        2. Forward pass entire batch (time → control)
        3. Batched evolution using gate_time.max()
        4. Batched corrections and loss computation
        5. Backward pass and step both optimizers

        This is ~80x faster than looping through angles!
        """
        self.controller.train()
        self.time_optimizer.zero_grad()
        self.control_optimizer.zero_grad()

        # Ensure angles has shape [batch, 1]
        if angles.dim() == 1:
            angles = angles.unsqueeze(-1)

        batch_size = angles.shape[0]

        # Forward pass: get times and detuning for entire batch
        gate_times, detuning_normalized = self.controller(angles)  # [batch, 1], [batch, n_steps, 1]

        # Use max gate time for evolution (archival pattern)
        max_gate_time = gate_times.max()

        # Create batched pulse functions
        # Note: get_detuning_pulse_fn handles scaling internally
        rabi_fn = self.controller.get_rabi_pulse_fn(max_gate_time)
        detuning_fn = self.controller.get_detuning_pulse_fn(detuning_normalized, max_gate_time)

        # Batched evolution: evolve all angles together
        final_unitaries = self._evolve_batch(
            rabi_fn, detuning_fn,
            max_gate_time.item(),
            batch_size,
            apply_corrections=True
        )

        # Create target gates for entire batch
        target_unitaries = torch.stack([czphi_gate(angle.item()) for angle in angles])

        # Batched infidelity computation
        infidelities = self._compute_batch_infidelity(final_unitaries, target_unitaries)

        # Loss: mean infidelity + time_weight * mean gate_time (archival pattern)
        mean_infidelity = infidelities.mean()
        mean_gate_time = gate_times.mean()
        time_penalty = self.time_weight * mean_gate_time

        total_loss = mean_infidelity + time_penalty

        # Backward and optimize
        total_loss.backward()
        self.time_optimizer.step()
        self.control_optimizer.step()

        metrics = {
            'loss': total_loss.item(),
            'infidelity': mean_infidelity.item(),
            'mean_gate_time': mean_gate_time.item()
        }

        return total_loss.item(), metrics
    
    def _evolve_batch(
        self,
        rabi_fn: Callable,
        detuning_fn: Callable,
        gate_time: float,
        batch_size: int,
        apply_corrections: bool = True
    ) -> torch.Tensor:
        """
        Batched quantum evolution for multiple angles.

        Following archival pattern: all angles evolved with same max gate_time
        in a SINGLE ODE call using batched Hamiltonian.

        Parameters
        ----------
        rabi_fn : Callable
            Batched Rabi pulse function returning [batch, 1]
        detuning_fn : Callable
            Batched detuning pulse function returning [batch, 1]
        gate_time : float
            Max gate time for all angles
        batch_size : int
            Number of angles in batch
        apply_corrections : bool
            Apply single-qubit phase corrections

        Returns
        -------
        torch.Tensor
            Final unitaries [batch, 4, 4]
        """
        from ..hardware.rydberg import RydbergHamiltonian
        import torchdiffeq

        # Create base Hamiltonian (will be used to get operators)
        # Use dummy constant pulses just to initialize
        dummy_rabi = lambda t: torch.tensor(1.0, device=self.device)
        dummy_detuning = lambda t: torch.tensor(0.0, device=self.device)

        base_hamiltonian = RydbergHamiltonian(
            nqubits=self.nqubits,
            rabi_pulse=dummy_rabi,
            detuning_pulse=dummy_detuning,
            addressing='global',
            device=self.device
        )

        # Batched initial state: identity for each angle
        hilbert_dim = 3 ** self.nqubits  # 9 for 2 qubits with 0,1,r
        init_matrix = torch.eye(hilbert_dim, dtype=torch.cfloat, device=self.device)
        init_batch = init_matrix.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, 9, 9]

        # Time array
        t_eval = torch.linspace(0.0, gate_time, self.controller.n_time_steps, device=self.device)

        # Create batched Hamiltonian function following archival pattern
        def hamiltonian_fn(t, y):
            """
            Batched Hamiltonian for ODE solver.

            This is the key to matching archival speed: we compute H for ALL batch elements
            at once using batched pulse values.
            """
            # Get batched pulse values [batch] or [batch, 1]
            rabi_batch = rabi_fn(t)  # [batch, 1] or [batch]
            detuning_batch = detuning_fn(t)  # [batch, 1] or [batch]

            # Ensure shape [batch, 1]
            if rabi_batch.dim() == 1:
                rabi_batch = rabi_batch.unsqueeze(-1)
            if detuning_batch.dim() == 1:
                detuning_batch = detuning_batch.unsqueeze(-1)

            # Build batched Hamiltonian: [batch, 9, 9]
            # Start with zeros for each batch element
            H_batch = torch.zeros(batch_size, hilbert_dim, hilbert_dim,
                                 dtype=torch.cfloat, device=self.device)

            # Add terms for each qubit (global addressing: all qubits same pulse)
            for i in range(self.nqubits):
                # Rabi term: (Ω(t)/2) * σ_x, batched
                # rabi_batch: [batch, 1], rabi_ops[i]: [9, 9]
                # Result: [batch, 9, 9]
                H_batch += 0.5 * rabi_batch.unsqueeze(-1) * base_hamiltonian.rabi_ops[i]

                # Detuning term: Δ(t) * n_r, batched
                H_batch += detuning_batch.unsqueeze(-1) * base_hamiltonian.detuning_ops[i]

            # Add interaction term (same for all batch elements)
            if base_hamiltonian.interaction_op is not None:
                H_batch += base_hamiltonian.vdd * base_hamiltonian.interaction_op

            # Compute dy/dt = -iH*y for entire batch
            # H_batch: [batch, 9, 9], y: [batch, 9, 9]
            return -1j * torch.bmm(H_batch, y)

        # Solve ODE for entire batch in ONE call (archival pattern!)
        solution = torchdiffeq.odeint(
            hamiltonian_fn,
            init_batch,
            t_eval,
            method='rk4'
        )  # [n_steps, batch, 9, 9]

        final_unitaries_full = solution[-1]  # [batch, 9, 9]

        # Reduce to computational subspace [batch, 4, 4]
        final_unitaries = self._reduce_to_computational(final_unitaries_full, batch_size)

        # Apply batched corrections
        if apply_corrections:
            final_unitaries = self._apply_batch_corrections(final_unitaries)

        return final_unitaries

    def _reduce_to_computational(self, unitaries: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Reduce from full 9x9 to computational 4x4 subspace.

        Following archival reduce_r_dim_2q_vector pattern.
        """
        # Indices for computational subspace (00, 01, 10, 11)
        # In full space: 0,1,3,4 (skipping r states)
        a_to_keep = [0, 1, 3, 4] * 4
        b_to_keep = [0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4]

        # Extract computational subspace
        reduced = unitaries[:, a_to_keep, b_to_keep].view(batch_size, 4, 4).transpose(1, 2)

        return reduced

    def _apply_batch_corrections(self, unitaries: torch.Tensor) -> torch.Tensor:
        """
        Apply single-qubit phase corrections using batched operations.

        Following archival correction_1q pattern with torch.bmm.
        """
        batch_size = unitaries.shape[0]

        # Extract phase from |01⟩ element (following archival)
        phi_01 = torch.angle(unitaries[:, 1, 1])  # [batch]

        # Create correction matrix for each angle
        identity = torch.eye(4, dtype=torch.cfloat, device=self.device)
        correction_batch = identity.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, 4, 4]

        # Apply phase corrections
        phase_factor = torch.exp(-1j * phi_01)
        correction_batch[:, 1, 1] = phase_factor
        correction_batch[:, 2, 2] = phase_factor
        correction_batch[:, 3, 3] = phase_factor ** 2

        # Batched matrix multiplication (archival uses torch.bmm)
        corrected = torch.bmm(correction_batch, unitaries)

        return corrected

    def _compute_batch_infidelity(
        self,
        achieved: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute infidelity for batch of unitaries.

        Parameters
        ----------
        achieved : torch.Tensor
            Achieved unitaries [batch, 4, 4]
        target : torch.Tensor
            Target unitaries [batch, 4, 4]

        Returns
        -------
        torch.Tensor
            Infidelities [batch]
        """
        from ..core.metrics import unitary_fidelity_batch

        # Use batched fidelity computation
        fidelities = unitary_fidelity_batch(achieved, target, nqubits=self.nqubits)
        infidelities = 1.0 - fidelities

        return infidelities

    def _evolve(
        self,
        rabi_fn: Callable,
        detuning_fn: Callable,
        gate_time: float,
        apply_corrections: bool = True
    ) -> torch.Tensor:
        """
        Evolve quantum system with given pulses using QuantumEvolver.

        NOTE: This is the single-angle version. _evolve_batch is used for training.

        Parameters
        ----------
        rabi_fn : Callable
            Rabi pulse function
        detuning_fn : Callable
            Detuning pulse function
        gate_time : float
            Gate time
        apply_corrections : bool
            Apply single-qubit phase corrections

        Returns
        -------
        torch.Tensor
            Final unitary in computational subspace
        """
        # Use QuantumEvolver which handles all the details
        pulses = [rabi_fn, detuning_fn]
        final_unitary = self.evolver.evolve(
            pulses,
            gate_time,
            apply_corrections=apply_corrections
        )

        return final_unitary
    
    def save_checkpoint(self, path: str, metadata: Optional[dict] = None):
        """Save both networks and optimizers."""
        checkpoint = {
            'time_network_state_dict': self.controller.time_predictor.state_dict(),
            'control_network_state_dict': self.controller.control_generator.state_dict(),
            'time_optimizer_state_dict': self.time_optimizer.state_dict(),
            'control_optimizer_state_dict': self.control_optimizer.state_dict(),
            'history': self.history,
            'epoch': self.current_epoch,
            'time_weight': self.time_weight,
            'controller_config': {
                'time_bounds': self.controller.time_bounds,
                'rabi_max': self.controller.rabi_max,
                'detuning_range': self.controller.detuning_range,
                'n_time_steps': self.controller.n_time_steps,
                'time_hidden_layers': len([l for l in self.controller.time_predictor if isinstance(l, nn.Linear)]),
                'time_hidden_units': self.controller.time_predictor[0].out_features,
                'control_hidden_layers': len([l for l in self.controller.control_generator if isinstance(l, nn.Linear)]),
                'control_hidden_units': self.controller.control_generator[0].out_features,
                'time_output_activation': self.controller.time_output_activation,
            },
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load both networks and optimizers."""
        checkpoint = torch.load(path, map_location=self.device)
        self.controller.time_predictor.load_state_dict(
            checkpoint['time_network_state_dict']
        )
        self.controller.control_generator.load_state_dict(
            checkpoint['control_network_state_dict']
        )
        self.time_optimizer.load_state_dict(
            checkpoint['time_optimizer_state_dict']
        )
        self.control_optimizer.load_state_dict(
            checkpoint['control_optimizer_state_dict']
        )
        self.history = checkpoint['history']
        self.current_epoch = checkpoint['epoch']
        if 'time_weight' in checkpoint:
            self.time_weight = checkpoint['time_weight']
    
    def evaluate(self, angles: torch.Tensor) -> Dict:
        """
        Evaluate controller on given angles.
        
        Returns dict with predicted times, infidelities, etc.
        """
        self.controller.eval()
        angles = angles.to(self.device)
        
        results = {
            'angles': [],
            'predicted_times': [],
            'infidelities': []
        }
        
        with torch.no_grad():
            for angle in angles:
                gate_time, detuning = self.controller(angle.unsqueeze(0))
                
                rabi_fn = self.controller.get_rabi_pulse_fn(gate_time)
                detuning_fn = self.controller.get_detuning_pulse_fn(detuning, gate_time)
                
                final_U = self._evolve(rabi_fn, detuning_fn, gate_time.item())
                target_U = czphi_gate(angle.item())
                
                infidelity = unitary_infidelity(final_U, target_U, nqubits=self.nqubits)
                
                results['angles'].append(angle.item())
                results['predicted_times'].append(gate_time.item())
                results['infidelities'].append(infidelity.item())
        
        results['mean_infidelity'] = torch.tensor(results['infidelities']).mean().item()
        results['mean_time'] = torch.tensor(results['predicted_times']).mean().item()
        
        return results
