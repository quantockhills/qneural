"""
JAX loss functions for quantum control optimization.
"""

import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

from ...core.metrics import unitary_infidelity, unitary_infidelity_batch


class QuantumLoss(ABC):
    """Abstract base class for quantum control loss functions."""

    @abstractmethod
    def __call__(
        self, achieved_unitary, target_unitary, **kwargs
    ):
        pass


class InfidelityLoss(QuantumLoss):
    def __init__(self, nqubits: int = 2):
        self.nqubits = nqubits

    def __call__(
        self, achieved_unitary, target_unitary, **kwargs
    ):
        if achieved_unitary.ndim == 2:
            return unitary_infidelity(
                achieved_unitary, target_unitary, nqubits=self.nqubits
            )
        else:
            return unitary_infidelity_batch(
                achieved_unitary, target_unitary, nqubits=self.nqubits
            )


class TimePenaltyLoss(QuantumLoss):
    def __init__(self, weight: float = 0.1, reference_time: Optional[float] = None):
        self.weight = weight
        self.reference_time = reference_time

    def __call__(
        self,
        achieved_unitary,
        target_unitary,
        gate_time=None,
        **kwargs,
    ):
        if gate_time is None:
            return jnp.array(0.0)
        ref = self.reference_time if self.reference_time else jnp.mean(gate_time)
        penalty = (gate_time / ref) ** 2
        return self.weight * jnp.mean(penalty)


class RobustnessLoss(QuantumLoss):
    """Loss promoting robustness to parameter variation (placeholder)."""

    def __init__(self, weight: float = 0.01, n_samples: int = 5):
        self.weight = weight
        self.n_samples = n_samples

    def __call__(self, achieved_unitary, target_unitary, **kwargs):
        return jnp.array(0.0)


class ResourceLoss(QuantumLoss):
    """Loss penalizing resource usage (placeholder)."""

    def __init__(self, weight: float = 0.001):
        self.weight = weight

    def __call__(self, achieved_unitary, target_unitary, **kwargs):
        return jnp.array(0.0)


class CompositeLoss:
    """Weighted combination of multiple loss functions."""

    def __init__(self, losses: List[Tuple[float, QuantumLoss]]):
        self.losses = losses

    def __call__(
        self, achieved_unitary, target_unitary, **kwargs
    ):
        total = jnp.array(0.0)
        for weight, loss_fn in self.losses:
            total = total + weight * loss_fn(achieved_unitary, target_unitary, **kwargs)
        return total


def create_infidelity_loss(nqubits: int = 2) -> InfidelityLoss:
    return InfidelityLoss(nqubits=nqubits)


def create_time_optimal_loss(
    nqubits: int = 2,
    time_weight: float = 0.1,
) -> CompositeLoss:
    return CompositeLoss([
        (1.0, InfidelityLoss(nqubits=nqubits)),
        (time_weight, TimePenaltyLoss(weight=1.0)),
    ])
