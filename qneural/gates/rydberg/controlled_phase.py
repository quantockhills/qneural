"""
Generalized controlled-phase gate optimization for Rydberg atoms.

Provides a unified framework for optimizing parametrized multi-controlled
phase gates (CZ_φ, CCZ_φ, CCCZ_φ, etc.) using neural networks.

Key insight: An N-controlled phase gate applies phase φ to |11...1⟩ state.
- CZ_φ: 1 control + 1 target = 2 qubits
- CCZ_φ: 2 controls + 1 target = 3 qubits
- CCCZ_φ: 3 controls + 1 target = 4 qubits

This module generalizes the approach so any N can be handled with the same code.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Callable, Dict
from abc import ABC, abstractmethod

from ...core.gates import czphi_gate, cczphi_gate
from ...core.metrics import unitary_infidelity
from ...neural import (
    FeedForwardNN,
    TimeOptimalController,
    PhysicalPulseGenerator,
    create_default_physical_pulse_generator,
    QuantumEvolver,
    create_evolver,
    QuantumTrainer,
    create_time_optimal_loss,
    InfidelityLoss,
)
from ...hardware.rydberg import RABI_DEFAULT


class ControlledPhaseGate(ABC):
    """
    Abstract base class for N-controlled phase gate optimization.

    This generalizes the optimization of multi-controlled phase gates
    (CZ_φ, CCZ_φ, etc.) to arbitrary number of control qubits.

    Parameters
    ----------
    n_controls : int
        Number of control qubits (N-1 for N-qubit gate)
    n_targets : int
        Number of target qubits (usually 1)
    total_qubits : int
        Total number of qubits (n_controls + n_targets)

    Examples
    --------
    >>> # CZ_φ gate (1 control, 1 target = 2 qubits)
    >>> gate = ControlledPhaseGate(n_controls=1, n_targets=1)
    >>>
    >>> # CCZ_φ gate (2 controls, 1 target = 3 qubits)
    >>> gate = ControlledPhaseGate(n_controls=2, n_targets=1)
    """

    def __init__(
        self,
        n_controls: int,
        n_targets: int = 1,
        rabi_max: float = None,
        detuning_range: Tuple[float, float] = None,
    ):
        self.n_controls = n_controls
        self.n_targets = n_targets
        self.total_qubits = n_controls + n_targets

        # Default physical parameters
        if rabi_max is None:
            rabi_max = RABI_DEFAULT
        self.rabi_max = rabi_max

        if detuning_range is None:
            detuning_range = (-2 * rabi_max, 2 * rabi_max)
        self.detuning_range = detuning_range

        # Hilbert space dimensions
        self.full_dim = 3**self.total_qubits  # With Rydberg state
        self.comp_dim = 2**self.total_qubits  # Computational subspace

        # Precompute computational basis indices
        self._comp_indices = self._get_computational_indices()

    def _get_computational_indices(self) -> List[int]:
        """
        Get indices of computational basis states (no Rydberg).

        For GG-qubits (dim=3), states are |0⟩, |1⟩, |r⟩.
        We want only states with 0s and 1s (no 2s which is |r⟩).
        """
        indices = []
        for i in range(self.full_dim):
            # Convert index to base-3 representation
            digits = []
            n = i
            for _ in range(self.total_qubits):
                digits.append(n % 3)
                n //= 3

            # Check if any digit is 2 (Rydberg state)
            if 2 not in digits:
                indices.append(i)

        return indices

    def reduce_to_computational_basis(self, unitary: torch.Tensor) -> torch.Tensor:
        """
        Reduce full unitary to computational subspace.

        Parameters
        ----------
        unitary : torch.Tensor
            Full unitary in 3^N Hilbert space

        Returns
        -------
        torch.Tensor
            Unitary in 2^N computational subspace
        """
        # Check if already reduced (4x4 for 2 qubits)
        expected_comp_dim = 2**self.total_qubits
        if unitary.shape[-1] == expected_comp_dim:
            # Already in computational basis
            return unitary

        if unitary.dim() == 2:
            # Single unitary
            return unitary[self._comp_indices][:, self._comp_indices]
        else:
            # Batched unitaries
            return unitary[:, self._comp_indices][:, :, self._comp_indices]

    def apply_phase_corrections(self, unitary: torch.Tensor) -> torch.Tensor:
        """
        Apply single-qubit phase corrections.

        Removes local phases by making |00...0⟩ the reference.

        Parameters
        ----------
        unitary : torch.Tensor
            Unitary in computational basis

        Returns
        -------
        torch.Tensor
            Corrected unitary
        """
        if unitary.dim() == 3:
            # Batched - apply to each
            corrected = []
            for u in unitary:
                corrected.append(self._apply_correction_single(u))
            return torch.stack(corrected)
        else:
            return self._apply_correction_single(unitary)

    def _apply_correction_single(self, unitary: torch.Tensor) -> torch.Tensor:
        """Apply symmetric phase correction to single unitary."""
        # Extract phase from |01⟩ state only (symmetric formula)
        phi_01 = torch.angle(unitary[1, 1])

        # Construct symmetric correction
        j1 = torch.exp(-1.0j * phi_01)
        correction = torch.eye(4, dtype=torch.cfloat, device=unitary.device)
        correction[1, 1] = j1  # |01⟩
        correction[2, 2] = j1  # |10⟩ (same phase)
        correction[3, 3] = j1**2  # |11⟩ (squared)

        return correction @ unitary

    def compute_phase_correction_matrix(self, unitary: torch.Tensor) -> torch.Tensor:
        """
        Compute the phase correction matrix.

        For N qubits, this is diag(1, e^{-iφ₁}, e^{-iφ₂}, ..., e^{-iφ_{2^N-1}})
        where φ_k is the phase of the k-th diagonal element.

        Parameters
        ----------
        unitary : torch.Tensor
            Unitary before correction

        Returns
        -------
        torch.Tensor
            Correction matrix
        """
        phases = torch.angle(torch.diag(unitary))
        correction = torch.diag(torch.exp(-1.0j * phases))
        correction[0, 0] = 1.0
        return correction

    @abstractmethod
    def get_target_unitary(self, phi: float) -> torch.Tensor:
        """
        Get target unitary for phase φ.

        Parameters
        ----------
        phi : float
            Phase angle

        Returns
        -------
        torch.Tensor
            Target unitary in computational basis
        """
        pass

    def compute_infidelity(
        self, achieved: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute infidelity between achieved and target."""
        return unitary_infidelity(achieved, target, nqubits=self.total_qubits)


class CZPhiGate(ControlledPhaseGate):
    """
    CZ_φ gate (1-control, 1-target = 2 qubits).

    Target unitary: diag(1, 1, 1, e^{iφ}) in computational basis.

    Parameters
    ----------
    rabi_max : float
        Maximum Rabi frequency
    detuning_range : Tuple[float, float]
        Detuning range

    Examples
    --------
    >>> gate = CZPhiGate()
    >>> target = gate.get_target_unitary(torch.pi / 2)
    >>> print(target.shape)  # (4, 4)
    """

    def __init__(
        self, rabi_max: float = None, detuning_range: Tuple[float, float] = None
    ):
        super().__init__(
            n_controls=1, n_targets=1, rabi_max=rabi_max, detuning_range=detuning_range
        )

    def get_target_unitary(self, phi: float) -> torch.Tensor:
        """Get CZ_φ target unitary."""
        return czphi_gate(phi)


class CCZPhiGate(ControlledPhaseGate):
    """
    CCZ_φ gate (2-controls, 1-target = 3 qubits).

    Target unitary: diag(1, 1, 1, 1, 1, 1, 1, e^{iφ}) in computational basis.
    Applies phase only to |111⟩ state.

    Parameters
    ----------
    rabi_max : float
        Maximum Rabi frequency
    detuning_range : Tuple[float, float]
        Detuning range

    Examples
    --------
    >>> gate = CCZPhiGate()
    >>> target = gate.get_target_unitary(torch.pi / 2)
    >>> print(target.shape)  # (8, 8)
    """

    def __init__(
        self, rabi_max: float = None, detuning_range: Tuple[float, float] = None
    ):
        super().__init__(
            n_controls=2, n_targets=1, rabi_max=rabi_max, detuning_range=detuning_range
        )

    def get_target_unitary(self, phi: float) -> torch.Tensor:
        """Get CCZ_φ target unitary."""
        return cczphi_gate(phi)


class ControlledPhaseOptimizer:
    """
    Neural network optimizer for controlled-phase gates.

    Generalizes the training of pulse sequences for CZ_φ, CCZ_φ, etc.

    Parameters
    ----------
    gate : ControlledPhaseGate
        Gate specification (CZPhiGate, CCZPhiGate, etc.)
    network : nn.Module
        Neural network for pulse generation
    trainer : QuantumTrainer
        Trainer instance
    pulse_generator : PhysicalPulseGenerator
        Pulse generator
    evolver : QuantumEvolver
        Quantum evolver

    Examples
    --------
    >>> # Optimize CZ_φ for angles in [0.1π, π]
    >>> gate = CZPhiGate()
    >>> optimizer = ControlledPhaseOptimizer(gate)
    >>> optimizer.train(angle_range=(0.1*torch.pi, torch.pi), epochs=1000)
    >>>
    >>> # Generate pulse for specific angle
    >>> pulse = optimizer.generate_pulse(torch.pi / 2)
    """

    def __init__(
        self,
        gate: ControlledPhaseGate,
        network: Optional[nn.Module] = None,
        trainer: Optional[QuantumTrainer] = None,
        pulse_generator: Optional[PhysicalPulseGenerator] = None,
        evolver: Optional[QuantumEvolver] = None,
        time_optimal: bool = False,
        time_bounds: Optional[Tuple[float, float]] = None,
    ):
        self.gate = gate
        self.time_optimal = time_optimal

        # Create or use provided network
        if network is None:
            if time_optimal:
                if time_bounds is None:
                    time_bounds = (3.0, 8.0)
                self.network = TimeOptimalController(
                    time_bounds=time_bounds, rabi_max=gate.rabi_max, n_time_steps=201
                )
            else:
                self.network = FeedForwardNN(
                    input_dim=2, output_dim=2, hidden_layers=6, hidden_units=150
                )
        else:
            self.network = network

        # Create or use provided components
        if pulse_generator is None:
            self.pulse_generator = create_default_physical_pulse_generator(
                rabi_max=gate.rabi_max
            )
        else:
            self.pulse_generator = pulse_generator

        if evolver is None:
            self.evolver = create_evolver(nqubits=gate.total_qubits)
        else:
            self.evolver = evolver

        if trainer is None:
            if time_optimal:
                loss_fn = create_time_optimal_loss(
                    nqubits=gate.total_qubits, infidelity_weight=1.0, time_weight=0.1
                )
            else:
                loss_fn = InfidelityLoss(nqubits=gate.total_qubits)

            self.trainer = QuantumTrainer(
                network=self.network,
                nqubits=gate.total_qubits,
                loss_fn=loss_fn,
                pulse_generator=self.pulse_generator,
                evolver=self.evolver,
            )
        else:
            self.trainer = trainer

    def train(
        self,
        angle_range: Tuple[float, float],
        n_angles: int = 80,
        gate_time: float = 5.0,
        epochs: int = 1000,
        **kwargs,
    ) -> Dict:
        """
        Train the neural network.

        Parameters
        ----------
        angle_range : Tuple[float, float]
            Min and max angles (e.g., (0.1*π, π))
        n_angles : int
            Number of angles to sample
        gate_time : float
            Gate time (ignored if time_optimal=True)
        epochs : int
            Training epochs
        **kwargs
            Additional training arguments

        Returns
        -------
        Dict
            Training history
        """
        angles = torch.linspace(angle_range[0], angle_range[1], n_angles)

        if self.time_optimal:
            # Time-optimal training
            return self.trainer.train(angles, None, epochs=epochs, **kwargs)
        else:
            # Fixed-time training
            return self.trainer.train(angles, gate_time, epochs=epochs, **kwargs)

    def generate_pulse(
        self, angle: float, gate_time: float = None
    ) -> Tuple[List[Callable], float]:
        """
        Generate pulse for specific angle.

        Parameters
        ----------
        angle : float
            Target phase angle
        gate_time : float, optional
            Gate time. If None and time_optimal=False, uses the gate's rabi_max
            to compute appropriate time (7.0 / rabi_max).

        Returns
        -------
        Tuple[List[Callable], float]
            (pulse_functions, gate_time)
        """
        angle_tensor = torch.tensor([angle])

        if self.time_optimal:
            gate_time, nn_output = self.network(angle_tensor)
            gate_time = gate_time.item()
        else:
            if gate_time is None:
                # Default: use normalized time 7.0 converted to actual time
                gate_time = 7.0 / self.gate.rabi_max
            time_grid = torch.linspace(0, 1, 201)
            nn_input = torch.stack([angle_tensor.repeat(201), time_grid], dim=1)
            nn_output = self.network(nn_input).reshape(201, 2)

        pulses = self.pulse_generator.generate(nn_output, gate_time)

        return pulses, gate_time

    def evaluate(self, angle: float) -> Dict:
        """
        Evaluate gate fidelity for specific angle.

        Parameters
        ----------
        angle : float
            Target angle

        Returns
        -------
        Dict
            Evaluation results including infidelity
        """
        pulses, gate_time = self.generate_pulse(angle)

        # Evolve (NO automatic corrections - we'll apply manually)
        final_unitary = self.evolver.evolve(pulses, gate_time, apply_corrections=False)

        # Reduce and correct
        reduced = self.gate.reduce_to_computational_basis(final_unitary)
        corrected = self.gate.apply_phase_corrections(reduced)

        # Compare to target
        target = self.gate.get_target_unitary(angle)
        infidelity = self.gate.compute_infidelity(corrected, target)

        return {
            "angle": angle,
            "gate_time": gate_time,
            "infidelity": infidelity.item(),
            "achieved_unitary": corrected,
            "target_unitary": target,
        }


# Convenience factory functions


def create_czphi_optimizer(
    time_optimal: bool = False, time_bounds: Tuple[float, float] = None, **kwargs
) -> ControlledPhaseOptimizer:
    """
    Create optimizer for CZ_φ gate.

    Parameters
    ----------
    time_optimal : bool
        Whether to optimize gate time
    time_bounds : Tuple[float, float]
        Time bounds for time-optimal optimization
    **kwargs
        Additional arguments

    Returns
    -------
    ControlledPhaseOptimizer
        Configured optimizer

    Examples
    --------
    >>> # Standard CZ_φ optimizer
    >>> optimizer = create_czphi_optimizer()
    >>>
    >>> # Time-optimal CZ_φ optimizer
    >>> optimizer = create_czphi_optimizer(time_optimal=True, time_bounds=(3.0, 8.0))
    """
    gate = CZPhiGate()
    return ControlledPhaseOptimizer(
        gate=gate, time_optimal=time_optimal, time_bounds=time_bounds, **kwargs
    )


def create_cczphi_optimizer(
    time_optimal: bool = False, time_bounds: Tuple[float, float] = None, **kwargs
) -> ControlledPhaseOptimizer:
    """
    Create optimizer for CCZ_φ gate.

    Parameters
    ----------
    time_optimal : bool
        Whether to optimize gate time
    time_bounds : Tuple[float, float]
        Time bounds for time-optimal optimization
    **kwargs
        Additional arguments

    Returns
    -------
    ControlledPhaseOptimizer
        Configured optimizer

    Examples
    --------
    >>> # Standard CCZ_φ optimizer
    >>> optimizer = create_cczphi_optimizer()
    >>>
    >>> # Time-optimal CCZ_φ optimizer
    >>> optimizer = create_cczphi_optimizer(time_optimal=True, time_bounds=(5.0, 12.0))
    """
    gate = CCZPhiGate()
    return ControlledPhaseOptimizer(
        gate=gate, time_optimal=time_optimal, time_bounds=time_bounds, **kwargs
    )
