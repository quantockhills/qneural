"""
Training infrastructure for quantum control with neural networks.

Provides the main training loop that orchestrates:
- Neural network forward pass
- Pulse generation
- Quantum evolution
- Loss computation
- Backpropagation and optimization
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from ..core.gates import czphi_gate
from ..core.metrics import unitary_infidelity
from .models import FeedForwardNN
from .losses import QuantumLoss, InfidelityLoss
from .pulse_generator import (
    PhysicalPulseGenerator,
    create_default_physical_pulse_generator,
)
from .evolution import QuantumEvolver, create_evolver


class QuantumTrainer:
    """
    Main trainer for quantum control optimization.

    Orchestrates the training loop:
    1. Generate pulses from NN
    2. Evolve quantum system
    3. Compute loss
    4. Backpropagate
    5. Update weights

    Parameters
    ----------
    network : nn.Module
        Neural network (FeedForwardNN or TimeOptimalController)
    nqubits : int
        Number of qubits
    loss_fn : QuantumLoss
        Loss function (e.g., InfidelityLoss or CompositeLoss)
        pulse_generator : Optional[PhysicalPulseGenerator]
        Pulse generator for converting NN outputs
    evolver : QuantumEvolver
        Quantum evolver for time evolution
    optimizer : torch.optim.Optimizer, optional
        Optimizer (created automatically if not provided)
    device : str
        Device to run on ('cpu' or 'cuda')

    Examples
    --------
    >>> from qneural.neural import FeedForwardNN
    >>> from qneural.neural.losses import create_infidelity_loss
    >>>
    >>> # Setup
    >>> network = FeedForwardNN(input_dim=2, output_dim=2, hidden_layers=6, hidden_units=150)
    >>> trainer = QuantumTrainer(
    ...     network=network,
    ...     nqubits=2,
    ...     loss_fn=create_infidelity_loss(nqubits=2),
    ...     rabi_max=25.0
    ... )
    >>>
    >>> # Train
    >>> history = trainer.train(
    ...     angles=torch.linspace(0.1*torch.pi, torch.pi, 80),
    ...     gate_time=5.0,
    ...     epochs=1000
    ... )
    """

    def __init__(
        self,
        network: nn.Module,
        nqubits: int,
        loss_fn: QuantumLoss,
        pulse_generator: Optional[PhysicalPulseGenerator] = None,
        evolver: Optional[QuantumEvolver] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ):
        self.network = network.to(device)
        self.nqubits = nqubits
        self.loss_fn = loss_fn
        self.device = device

        # Create default pulse generator if not provided
        if pulse_generator is None:
            # Default for Rydberg with 4MHz Rabi
            rabi_max = 2 * torch.pi * 4.0  # ~25.13
            self.pulse_generator = create_default_physical_pulse_generator(rabi_max)
        else:
            self.pulse_generator = pulse_generator

        # Create default evolver if not provided
        if evolver is None:
            self.evolver = create_evolver(
                nqubits, n_time_steps=self.pulse_generator.n_time_steps
            )
        else:
            self.evolver = evolver

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer

        # Training history
        self.history = {"epoch": [], "loss": [], "infidelity": [], "time": []}

        self.current_epoch = 0

    def train(
        self,
        angles: torch.Tensor,
        gate_time: float,
        epochs: int = 1000,
        print_every: int = 50,
        save_path: Optional[str] = None,
    ) -> Dict:
        """
        Train the neural network.

        Parameters
        ----------
        angles : torch.Tensor
            Target angles [n_angles]
        gate_time : float
            Gate time (constant for now)
        epochs : int
            Number of training epochs
        print_every : int
            Print progress every N epochs
        save_path : str, optional
            Path to save best model

        Returns
        -------
        Dict
            Training history
        """
        angles = angles.to(self.device)
        best_loss = float("inf")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Training step
            loss, metrics = self._train_step(angles, gate_time)

            # Update history
            self.history["epoch"].append(epoch)
            self.history["loss"].append(loss)
            self.history["infidelity"].append(metrics["infidelity"])

            # Print progress
            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch}: Loss = {loss:.6f}, "
                    f"Infidelity = {metrics['infidelity']:.6f}"
                )

            # Save best model
            if save_path and loss < best_loss:
                best_loss = loss
                self.save_checkpoint(save_path)

        return self.history

    def _train_step(self, angles: torch.Tensor, gate_time: float) -> Tuple[float, Dict]:
        """
        Single training step.

        Returns
        -------
        Tuple[float, Dict]
            (loss value, metrics dict)
        """
        self.network.train()
        self.optimizer.zero_grad()

        # Generate inputs for NN: [angle, normalized_time] pairs
        n_angles = len(angles)
        n_steps = self.pulse_generator.n_time_steps
        time_grid = torch.linspace(0, 1, n_steps, device=self.device)

        # Create input tensor: repeat angles for each time step
        angles_repeated = angles.repeat_interleave(n_steps)
        time_repeated = time_grid.repeat(n_angles)
        inputs = torch.stack([angles_repeated, time_repeated], dim=1)

        # Forward pass through NN
        nn_outputs = self.network(inputs)  # [n_angles * n_steps, n_controls]

        # Reshape to [n_angles, n_steps, n_controls]
        nn_outputs = nn_outputs.reshape(n_angles, n_steps, -1)

        # Process each angle
        total_loss = torch.tensor(0.0, device=self.device)
        total_infidelity = torch.tensor(0.0, device=self.device)

        for i, angle in enumerate(angles):
            # Generate pulses for this angle
            pulses = self.pulse_generator.generate(nn_outputs[i], gate_time)

            # Evolve quantum system
            final_unitary = self.evolver.evolve(pulses, gate_time)

            # Compute target unitary
            target_unitary = czphi_gate(angle.item())

            # Compute infidelity
            infidelity = unitary_infidelity(
                final_unitary, target_unitary, nqubits=self.nqubits
            )

            # Compute loss (may include time penalty, etc.)
            loss = self.loss_fn(final_unitary, target_unitary)

            total_loss += loss
            total_infidelity += infidelity

        # Average over angles
        avg_loss = total_loss / n_angles
        avg_infidelity = total_infidelity / n_angles

        # Backward pass
        avg_loss.backward()
        self.optimizer.step()

        # Return metrics
        metrics = {"loss": avg_loss.item(), "infidelity": avg_infidelity.item()}

        return avg_loss.item(), metrics


class FixedRabiTrainer(QuantumTrainer):
    """
    Specialized trainer for fixed-rabi, variable-detuning optimization.

    This trainer keeps the Rabi frequency constant at its maximum value
    and only optimizes the detuning pulse. This is the standard approach
    for achieving high-fidelity CZ gates with neural networks.

    Parameters
    ----------
    network : nn.Module
        Neural network with output_dim=1 (detuning only)
    nqubits : int
        Number of qubits
    rabi_max : float
        Maximum Rabi frequency (constant value to use)
    loss_fn : QuantumLoss, optional
        Loss function (defaults to InfidelityLoss)
    pulse_generator : PhysicalPulseGenerator, optional
        Pulse generator for detuning
    evolver : QuantumEvolver, optional
        Quantum evolver
    optimizer : torch.optim.Optimizer, optional
        Optimizer
    device : str
        Device to run on

    Examples
    --------
    >>> from qneural.neural import FeedForwardNN
    >>> from qneural.gates.rydberg import CZPhiGate
    >>>
    >>> gate = CZPhiGate()
    >>> network = FeedForwardNN(
    ...     input_dim=2, output_dim=1,  # Detuning only!
    ...     hidden_layers=6, hidden_units=150,
    ...     weight_scale=1.8
    ... )
    >>> trainer = FixedRabiTrainer(
    ...     network=network,
    ...     nqubits=2,
    ...     rabi_max=gate.rabi_max
    ... )
    >>> history = trainer.train(
    ...     angles=torch.tensor([np.pi]),  # CZ gate
    ...     gate_time=7.62 / gate.rabi_max,
    ...     epochs=500
    ... )
    """

    def __init__(
        self,
        network: nn.Module,
        nqubits: int,
        rabi_max: float,
        loss_fn: Optional[QuantumLoss] = None,
        pulse_generator: Optional[PhysicalPulseGenerator] = None,
        evolver: Optional[QuantumEvolver] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ):
        # Use InfidelityLoss by default
        if loss_fn is None:
            loss_fn = InfidelityLoss(nqubits=nqubits)

        # Create pulse generator for detuning only if not provided
        if pulse_generator is None:
            pulse_generator = PhysicalPulseGenerator(
                n_controls=1,  # Detuning only!
                n_time_steps=301,  # Higher resolution for better numerical accuracy
                control_ranges=[(-2 * rabi_max, 2 * rabi_max)],
            )

        # Create evolver with matching time steps if not provided
        if evolver is None:
            evolver = create_evolver(nqubits, n_time_steps=pulse_generator.n_time_steps)

        super().__init__(
            network=network,
            nqubits=nqubits,
            loss_fn=loss_fn,
            pulse_generator=pulse_generator,
            evolver=evolver,
            optimizer=optimizer,
            device=device,
        )

        self.rabi_max = rabi_max
        # Precompute time grid for efficiency and convergence
        self._time_grid = None

    def _train_step(self, angles: torch.Tensor, gate_time: float) -> Tuple[float, Dict]:
        """
        Training step with fixed Rabi, learned detuning.

        Key difference from base class: Rabi is constant, only
        detuning comes from the neural network.
        """
        self.network.train()
        self.optimizer.zero_grad()

        # Generate inputs for NN: [angle, normalized_time] pairs
        n_angles = len(angles)
        n_steps = self.pulse_generator.n_time_steps

        # CRITICAL: Precompute time grid once for convergence
        # Creating it fresh every epoch breaks training!
        if self._time_grid is None or len(self._time_grid) != n_steps:
            self._time_grid = torch.linspace(0, 1, n_steps, device=self.device)
        time_grid = self._time_grid

        # Create input tensor
        angles_repeated = angles.repeat_interleave(n_steps)
        time_repeated = time_grid.repeat(n_angles)
        inputs = torch.stack([angles_repeated, time_repeated], dim=1)

        # Forward pass through NN (outputs detuning only)
        nn_outputs = self.network(inputs)  # [n_angles * n_steps, 1]
        nn_outputs = nn_outputs.reshape(n_angles, n_steps, -1)

        # Constant Rabi pulse function
        def rabi_pulse(t):
            return torch.tensor(self.rabi_max, device=self.device)

        # Process each angle
        total_loss = torch.tensor(0.0, device=self.device)
        total_infidelity = torch.tensor(0.0, device=self.device)

        for i, angle in enumerate(angles):
            # Get detuning values from NN
            detuning_values = self.pulse_generator.scale_output(nn_outputs[i, :, 0], 0)

            # Create piecewise detuning function using shared utility
            from ..hardware.rydberg.pulses import create_simple_detuning_pulse

            detuning_fn = create_simple_detuning_pulse(detuning_values, gate_time)

            # Pulses: constant Rabi + learned detuning
            pulses = [rabi_pulse, detuning_fn]

            # Evolve with phase corrections applied
            final_unitary = self.evolver.evolve(
                pulses, gate_time, apply_corrections=True
            )

            # Compute target
            target_unitary = czphi_gate(angle.item())

            # Compute loss (compare corrected to target)
            infidelity = unitary_infidelity(
                final_unitary, target_unitary, nqubits=self.nqubits
            )
            loss = self.loss_fn(final_unitary, target_unitary)

            total_loss += loss
            total_infidelity += infidelity

        # Average over angles
        avg_loss = total_loss / n_angles
        avg_infidelity = total_infidelity / n_angles

        # Backward pass
        avg_loss.backward()
        self.optimizer.step()

        metrics = {"loss": avg_loss.item(), "infidelity": avg_infidelity.item()}

        return avg_loss.item(), metrics

    def evaluate(self, angles: torch.Tensor, gate_time: float) -> Dict:
        """
        Evaluate the trained network.

        Parameters
        ----------
        angles : torch.Tensor
            Test angles
        gate_time : float
            Gate time

        Returns
        -------
        Dict
            Evaluation metrics
        """
        self.network.eval()
        angles = angles.to(self.device)

        with torch.no_grad():
            # Similar to train step but without backprop
            n_angles = len(angles)
            n_steps = self.pulse_generator.n_time_steps
            time_grid = torch.linspace(0, 1, n_steps, device=self.device)

            angles_repeated = angles.repeat_interleave(n_steps)
            time_repeated = time_grid.repeat(n_angles)
            inputs = torch.stack([angles_repeated, time_repeated], dim=1)

            nn_outputs = self.network(inputs)
            nn_outputs = nn_outputs.reshape(n_angles, n_steps, -1)

            results = {"angles": [], "infidelities": [], "gate_times": []}

            for i, angle in enumerate(angles):
                pulses = self.pulse_generator.generate(nn_outputs[i], gate_time)
                final_unitary = self.evolver.evolve(pulses, gate_time)
                target_unitary = czphi_gate(angle.item())

                infidelity = unitary_infidelity(
                    final_unitary, target_unitary, nqubits=self.nqubits
                )

                results["angles"].append(angle.item())
                results["infidelities"].append(infidelity.item())
                results["gate_times"].append(gate_time)

            results["mean_infidelity"] = (
                torch.tensor(results["infidelities"]).mean().item()
            )

        return results

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "epoch": self.current_epoch,
            "nqubits": self.nqubits,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        self.current_epoch = checkpoint["epoch"]


# Convenience functions


def create_trainer(
    nqubits: int = 2,
    hidden_layers: int = 6,
    hidden_units: int = 150,
    learning_rate: float = 1e-3,
    **kwargs,
) -> QuantumTrainer:
    """
    Factory function to create a trainer with default settings.

    Parameters
    ----------
    nqubits : int
        Number of qubits
    hidden_layers : int
        Number of hidden layers in NN
    hidden_units : int
        Number of units per hidden layer
    learning_rate : float
        Learning rate for optimizer
    **kwargs
        Additional arguments

    Returns
    -------
    QuantumTrainer
        Configured trainer

    Examples
    --------
    >>> # Standard trainer
    >>> trainer = create_trainer(nqubits=2, hidden_layers=6, hidden_units=150)
    """
    from .losses import create_infidelity_loss

    network = FeedForwardNN(
        input_dim=2,
        output_dim=2,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
    )

    loss_fn = create_infidelity_loss(nqubits=nqubits)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    return QuantumTrainer(
        network=network, nqubits=nqubits, loss_fn=loss_fn, optimizer=optimizer
    )
