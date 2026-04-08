"""
Loss functions for quantum control optimization.

Provides modular, composable loss functions for training neural networks
to generate optimal quantum control pulses.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

from ..core.metrics import unitary_infidelity, unitary_infidelity_batch


class QuantumLoss(ABC, nn.Module):
    """
    Abstract base class for quantum control loss functions.

    All loss functions should inherit from this class and implement
    the forward method.
    """

    @abstractmethod
    def forward(
        self, achieved_unitary: torch.Tensor, target_unitary: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Compute loss between achieved and target unitaries.

        Parameters
        ----------
        achieved_unitary : torch.Tensor
            Unitary achieved by the pulse, shape [d, d] or [batch, d, d]
        target_unitary : torch.Tensor
            Target unitary to compare against, shape [d, d] or [batch, d, d]
        **kwargs
            Additional loss-specific arguments

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        pass


class InfidelityLoss(QuantumLoss):
    """
    Gate infidelity loss.

    Primary objective: minimize 1 - F where F is gate fidelity.

    Parameters
    ----------
    nqubits : int
        Number of qubits (for computing fidelity)

    Examples
    --------
    >>> loss_fn = InfidelityLoss(nqubits=2)
    >>> loss = loss_fn(achieved_U, target_U)
    """

    def __init__(self, nqubits: int = 2):
        super().__init__()
        self.nqubits = nqubits

    def forward(
        self, achieved_unitary: torch.Tensor, target_unitary: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute infidelity loss."""
        if achieved_unitary.dim() == 2:
            # Single unitary
            return unitary_infidelity(
                achieved_unitary, target_unitary, nqubits=self.nqubits
            )
        else:
            # Batch of unitaries
            return unitary_infidelity_batch(
                achieved_unitary, target_unitary, nqubits=self.nqubits
            )


class TimePenaltyLoss(QuantumLoss):
    """
    Time regularization loss for time-optimal control.

    Penalizes long gate times to encourage time-optimal solutions.

    Parameters
    ----------
    weight : float
        Weight factor for time penalty (default: 0.1)
    reference_time : float, optional
        Reference time for normalization. If None, uses mean.

    Examples
    --------
    >>> loss_fn = TimePenaltyLoss(weight=0.1)
    >>> loss = loss_fn(achieved_U, target_U, gate_time=5.0)
    """

    def __init__(self, weight: float = 0.1, reference_time: Optional[float] = None):
        super().__init__()
        self.weight = weight
        self.reference_time = reference_time

    def forward(
        self,
        achieved_unitary: torch.Tensor,
        target_unitary: torch.Tensor,
        gate_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute time penalty loss.

        Parameters
        ----------
        gate_time : torch.Tensor
            Gate time(s), shape [] or [batch_size]
        """
        if gate_time is None:
            raise ValueError("TimePenaltyLoss requires 'gate_time' argument")

        # Normalize by reference time if provided
        if self.reference_time is not None:
            normalized_time = gate_time / self.reference_time
        else:
            normalized_time = gate_time

        # Mean over batch if batched
        if normalized_time.dim() > 0:
            normalized_time = normalized_time.mean()

        return self.weight * normalized_time


class RobustnessLoss(QuantumLoss):
    """
    Robustness loss against noise (NOT YET IMPLEMENTED - Beta Feature).

    This class is a placeholder for future implementation. It will be used to
    penalize sensitivity to various noise sources:
    - Rabi frequency fluctuations
    - Detuning errors
    - Timing jitter

    **Status**: Planned for v1.0 release

    Raises
    ------
    NotImplementedError
        This feature is not yet available in the current beta release.
    """

    def __init__(self, noise_strength: float = 0.01):
        super().__init__()
        self.noise_strength = noise_strength
        import warnings

        warnings.warn(
            "RobustnessLoss is not yet implemented in this beta release. "
            "This feature is planned for v1.0. The loss will return zero.",
            FutureWarning,
            stacklevel=2,
        )

    def forward(
        self, achieved_unitary: torch.Tensor, target_unitary: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Compute robustness loss (not yet implemented).

        Returns
        -------
        torch.Tensor
            Zero tensor (placeholder implementation)
        """
        # Placeholder: returns zero until proper implementation
        return torch.tensor(0.0, device=achieved_unitary.device)


class ResourceLoss(QuantumLoss):
    """
    Resource utilization loss (MINIMAL IMPLEMENTATION - Beta Feature).

    Currently implements basic pulse amplitude penalization.
    Future versions will include:
    - Rapid pulse variation penalties
    - Total pulse energy constraints
    - Hardware-specific resource limits

    **Status**: Basic implementation in beta, full features planned for v1.0

    Parameters
    ----------
    weight : float
        Weight for resource penalty (default: 0.01)
    """

    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
        import warnings

        warnings.warn(
            "ResourceLoss has minimal implementation in this beta release. "
            "Only basic pulse amplitude penalization is available. "
            "Full resource optimization planned for v1.0.",
            FutureWarning,
            stacklevel=2,
        )

    def forward(
        self,
        achieved_unitary: torch.Tensor,
        target_unitary: torch.Tensor,
        pulses: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute resource loss (minimal implementation).

        Currently penalizes high pulse amplitudes only.

        Parameters
        ----------
        pulses : torch.Tensor, optional
            Control pulse amplitudes

        Returns
        -------
        torch.Tensor
            Resource penalty (zero if no pulses provided)
        """
        if pulses is None:
            return torch.tensor(0.0, device=achieved_unitary.device)

        # Basic implementation: penalize high pulse amplitudes
        return self.weight * torch.mean(pulses**2)


class CompositeLoss(QuantumLoss):
    """
    Combines multiple loss functions with configurable weights.

    This is the main loss function for quantum control training,
    allowing flexible combination of objectives.

    Parameters
    ----------
    losses : List[Tuple[QuantumLoss, float]]
        List of (loss_function, weight) tuples

    Examples
    --------
    >>> # Standard time-optimal control
    >>> loss_fn = CompositeLoss([
    ...     (InfidelityLoss(nqubits=2), 1.0),
    ...     (TimePenaltyLoss(weight=0.1), 0.1)
    ... ])
    >>>
    >>> # Infidelity-only training
    >>> loss_fn = CompositeLoss([
    ...     (InfidelityLoss(nqubits=2), 1.0)
    ... ])
    """

    def __init__(self, losses: List[Tuple[QuantumLoss, float]]):
        super().__init__()
        self.losses = nn.ModuleList([loss for loss, _ in losses])
        self.weights = [weight for _, weight in losses]

    def forward(
        self, achieved_unitary: torch.Tensor, target_unitary: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Compute weighted sum of all loss components.

        Parameters
        ----------
        achieved_unitary : torch.Tensor
            Achieved unitary
        target_unitary : torch.Tensor
            Target unitary
        **kwargs
            Additional arguments passed to all loss functions
            (e.g., gate_time for TimePenaltyLoss)

        Returns
        -------
        torch.Tensor
            Total weighted loss
        """
        total_loss = torch.tensor(0.0, device=achieved_unitary.device)

        for loss_fn, weight in zip(self.losses, self.weights):
            component = loss_fn(achieved_unitary, target_unitary, **kwargs)
            total_loss += weight * component

        return total_loss

    def get_component_losses(
        self, achieved_unitary: torch.Tensor, target_unitary: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Get individual loss components for logging/analysis.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary mapping loss names to values
        """
        component_losses = {}

        for i, (loss_fn, weight) in enumerate(zip(self.losses, self.weights)):
            component = loss_fn(achieved_unitary, target_unitary, **kwargs)
            loss_name = f"loss_{i}_{loss_fn.__class__.__name__}"
            component_losses[loss_name] = component
            component_losses[loss_name + "_weighted"] = weight * component

        return component_losses


# Convenience factory functions


def create_infidelity_loss(nqubits: int = 2) -> InfidelityLoss:
    """Create infidelity-only loss."""
    return InfidelityLoss(nqubits=nqubits)


def create_time_optimal_loss(
    nqubits: int = 2,
    infidelity_weight: float = 1.0,
    time_weight: float = 0.1,
    reference_time: Optional[float] = None,
) -> CompositeLoss:
    """
    Create standard time-optimal control loss.

    Parameters
    ----------
    nqubits : int
        Number of qubits
    infidelity_weight : float
        Weight for infidelity term
    time_weight : float
        Weight for time penalty term
    reference_time : float, optional
        Reference time for normalization

    Returns
    -------
    CompositeLoss
        Combined infidelity + time penalty loss
    """
    return CompositeLoss(
        [
            (InfidelityLoss(nqubits=nqubits), infidelity_weight),
            (
                TimePenaltyLoss(weight=time_weight, reference_time=reference_time),
                time_weight,
            ),
        ]
    )
