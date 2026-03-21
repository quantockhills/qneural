"""
Rydberg Hamiltonian for neutral atom quantum computing.

Implements the Hamiltonian for Rydberg atom systems with:
    - Rabi coupling (|1⟩ ↔ |r⟩ transition)
    - Detuning (energy of |r⟩ state)
    - Van der Waals interactions (between Rydberg states)

Supports both global and local addressing schemes.
"""

import torch
from typing import Callable, Dict, List, Optional, Union
from itertools import combinations

from ...backend import backend
from ...config import DEVICE, DTYPE_COMPLEX, DTYPE_REAL
from ...core.states import tensor_product
from .constants import HILBERT_DIM_GG, RABI_DEFAULT, VDD_COUPLING
from .operators import create_rydberg_operators


class RydbergHamiltonian:
    """
    Hamiltonian for Rydberg atom quantum computing.

    Models the dynamics of neutral atoms in their ground and Rydberg states
    under laser driving and interatomic interactions.

    Hamiltonian structure:
        H(t) = Σᵢ [Ωᵢ(t)/2 * σ_x⁽ⁱ⁾ + Δᵢ(t) * n_r⁽ⁱ⁾] + Σᵢ<ⱼ V_dd * n_r⁽ⁱ⁾ n_r⁽ʲ⁾

    where:
        - Ωᵢ(t): Rabi frequency (drive strength)
        - Δᵢ(t): Detuning (energy offset of |r⟩)
        - σ_x⁽ⁱ⁾: Rabi coupling operator for qubit i
        - n_r⁽ⁱ⁾: Rydberg state projection for qubit i
        - V_dd: Van der Waals interaction strength

    Attributes
    ----------
    nqubits : int
        Number of qubits in the system
    addressing : str
        'global' or 'local' addressing scheme
    rabi_pulse : Callable or List[Callable]
        Rabi frequency pulse function(s)
    detuning_pulse : Callable or List[Callable]
        Detuning pulse function(s)
    vdd : torch.Tensor
        Van der Waals interaction strength

    Examples
    --------
    >>> # Global addressing with constant pulses
    >>> from .pulses import constant_pulse
    >>> rabi = constant_pulse(2 * torch.pi * 4)  # 4 MHz
    >>> detuning = constant_pulse(0.0)
    >>> ham = RydbergHamiltonian(
    ...     nqubits=2,
    ...     rabi_pulse=rabi,
    ...     detuning_pulse=detuning,
    ...     addressing='global'
    ... )
    """

    def __init__(
        self,
        nqubits: int,
        rabi_pulse: Union[Callable, List[Callable]],
        detuning_pulse: Union[Callable, List[Callable]],
        addressing: str = 'global',
        vdd: Optional[float] = None,
        decay_rate: float = 0.0,
        device: Optional[str] = None
    ):
        """
        Initialize Rydberg Hamiltonian.

        Parameters
        ----------
        nqubits : int
            Number of qubits
        rabi_pulse : Callable or List[Callable]
            Rabi frequency pulse function(s). For 'global' addressing, a single
            function applied to all qubits. For 'local' addressing, a list of
            functions, one per qubit.
        detuning_pulse : Callable or List[Callable]
            Detuning pulse function(s), same structure as rabi_pulse
        addressing : str
            'global' or 'local' addressing scheme
        vdd : float, optional
            Van der Waals interaction strength (default: from constants.VDD_COUPLING)
        decay_rate : float
            Rydberg state decay rate (for non-Hermitian dynamics)
        device : str, optional
            Device to place tensors on
        """
        self.nqubits = nqubits
        self.addressing = addressing
        self.device = device or DEVICE
        self.decay_rate = decay_rate

        # Set default V_dd if not provided
        if vdd is None:
            # V_dd = coupling_constant * Rabi_max (see constants.VDD_COUPLING)
            self.vdd = VDD_COUPLING * RABI_DEFAULT
        else:
            self.vdd = torch.tensor(vdd, dtype=DTYPE_REAL, device=self.device)

        # Validate and store pulse functions
        self._setup_pulses(rabi_pulse, detuning_pulse)

        # Create operators for each qubit
        self._setup_operators()

    def _setup_pulses(
        self,
        rabi_pulse: Union[Callable, List[Callable]],
        detuning_pulse: Union[Callable, List[Callable]]
    ):
        """Setup pulse functions for each qubit."""
        if self.addressing == 'global':
            # Single pulse function for all qubits
            self.rabi_pulses = [rabi_pulse] * self.nqubits
            self.detuning_pulses = [detuning_pulse] * self.nqubits
        elif self.addressing == 'local':
            # Separate pulse for each qubit
            if not isinstance(rabi_pulse, (list, tuple)):
                raise ValueError("Local addressing requires list of rabi_pulse functions")
            if not isinstance(detuning_pulse, (list, tuple)):
                raise ValueError("Local addressing requires list of detuning_pulse functions")
            if len(rabi_pulse) != self.nqubits:
                raise ValueError(f"Expected {self.nqubits} rabi pulses, got {len(rabi_pulse)}")
            if len(detuning_pulse) != self.nqubits:
                raise ValueError(f"Expected {self.nqubits} detuning pulses, got {len(detuning_pulse)}")
            self.rabi_pulses = rabi_pulse
            self.detuning_pulses = detuning_pulse
        else:
            raise ValueError(f"Unknown addressing mode: {self.addressing}")

    def _setup_operators(self):
        """Setup the operator tensors for the Hamiltonian."""
        # Get single-qubit operators
        local_ops = create_rydberg_operators(dim=HILBERT_DIM_GG, device=self.device)

        # Identity for local Hilbert space
        identity = backend.eye(HILBERT_DIM_GG, dtype=DTYPE_COMPLEX, device=self.device)

        # Build tensored operators for each qubit
        self.rabi_ops = []
        self.detuning_ops = []

        for i in range(self.nqubits):
            # Rabi operator: acts on qubit i
            op_list = [identity] * self.nqubits
            op_list[i] = local_ops['rabi']
            self.rabi_ops.append(tensor_product(op_list))

            # Detuning operator: acts on qubit i
            op_list = [identity] * self.nqubits
            op_list[i] = local_ops['detuning']
            self.detuning_ops.append(tensor_product(op_list))

        # Build interaction operators (for nqubits >= 2)
        if self.nqubits >= 2:
            self.interaction_op = self._build_interaction_operator(
                local_ops['detuning'], identity
            )
        else:
            self.interaction_op = None

    def _build_interaction_operator(
        self,
        n_r_operator: torch.Tensor,
        identity: torch.Tensor
    ) -> torch.Tensor:
        """
        Build the total Van der Waals interaction operator.

        For n qubits: H_int = V_dd * Σᵢ<ⱼ n_r⁽ⁱ⁾ n_r⁽ʲ⁾

        Parameters
        ----------
        n_r_operator : torch.Tensor
            Single-qubit Rydberg projection operator
        identity : torch.Tensor
            Single-qubit identity

        Returns
        -------
        torch.Tensor
            Total interaction operator
        """
        total_interaction = backend.zeros(
            (HILBERT_DIM_GG ** self.nqubits, HILBERT_DIM_GG ** self.nqubits),
            dtype=DTYPE_COMPLEX,
            device=self.device
        )

        # Sum over all pairs
        for i, j in combinations(range(self.nqubits), 2):
            # Build n_r⁽ⁱ⁾ ⊗ n_r⁽ʲ⁾
            op_list = [identity] * self.nqubits
            op_list[i] = n_r_operator
            op_list[j] = n_r_operator

            total_interaction += tensor_product(op_list)

        return total_interaction

    def __call__(self, t: float, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Evaluate the Hamiltonian at time t.

        Parameters
        ----------
        t : float
            Time point
        batch_size : int, optional
            If provided, return batch of Hamiltonians with different pulse values.
            Used for batch training with neural networks.

        Returns
        -------
        torch.Tensor
            Hamiltonian matrix at time t, shape:
            - [d, d] if batch_size is None
            - [batch_size, d, d] if batch_size is provided
            where d = 3^nqubits
        """
        H = backend.zeros(
            (HILBERT_DIM_GG ** self.nqubits, HILBERT_DIM_GG ** self.nqubits),
            dtype=DTYPE_COMPLEX,
            device=self.device
        )

        # Add Rabi and detuning terms for each qubit
        for i in range(self.nqubits):
            # Rabi term: (Ω(t)/2) * σ_x
            omega_t = self.rabi_pulses[i](t)
            if isinstance(omega_t, torch.Tensor):
                omega_t = omega_t.to(dtype=DTYPE_REAL, device=self.device)
            else:
                omega_t = torch.tensor(omega_t, dtype=DTYPE_REAL, device=self.device)

            H += 0.5 * omega_t * self.rabi_ops[i]

            # Detuning term: Δ(t) * n_r
            delta_t = self.detuning_pulses[i](t)
            if isinstance(delta_t, torch.Tensor):
                delta_t = delta_t.to(dtype=DTYPE_REAL, device=self.device)
            else:
                delta_t = torch.tensor(delta_t, dtype=DTYPE_REAL, device=self.device)

            H += delta_t * self.detuning_ops[i]

            # Decay term (non-Hermitian): -i * Γ/2 * n_r
            if self.decay_rate > 0:
                decay_term = -1.0j * (self.decay_rate / 2.0) * self.detuning_ops[i]
                H += decay_term

        # Add interaction term
        if self.interaction_op is not None:
            H += self.vdd * self.interaction_op

        # Handle batching if requested
        if batch_size is not None:
            # Expand for batch processing
            H = H.unsqueeze(0).expand(batch_size, -1, -1)

        return H

    def get_hilbert_dim(self) -> int:
        """Return the dimension of the Hilbert space."""
        return HILBERT_DIM_GG ** self.nqubits

    def get_matrix_shape(self) -> tuple:
        """Return the shape of the Hamiltonian matrix."""
        d = self.get_hilbert_dim()
        return (d, d)

    @property
    def rabi_max(self) -> torch.Tensor:
        """Maximum Rabi frequency (for regularization)."""
        return torch.tensor(RABI_DEFAULT, dtype=DTYPE_REAL, device=self.device)


def create_constant_hamiltonian(
    nqubits: int,
    rabi_amplitude: float,
    detuning_amplitude: float,
    addressing: str = 'global',
    device: Optional[str] = None
) -> RydbergHamiltonian:
    """
    Convenience function to create a Hamiltonian with constant pulses.

    Parameters
    ----------
    nqubits : int
        Number of qubits
    rabi_amplitude : float
        Constant Rabi frequency
    detuning_amplitude : float
        Constant detuning
    addressing : str
        'global' or 'local'
    device : str, optional
        Device to place tensors on

    Returns
    -------
    RydbergHamiltonian
        Hamiltonian with constant pulses

    Examples
    --------
    >>> # 2-qubit Hamiltonian with 4 MHz Rabi and 1 MHz detuning
    >>> ham = create_constant_hamiltonian(
    ...     nqubits=2,
    ...     rabi_amplitude=2 * torch.pi * 4,  # 4 MHz
    ...     detuning_amplitude=2 * torch.pi * 1  # 1 MHz
    ... )
    """
    from .pulses import constant_pulse

    return RydbergHamiltonian(
        nqubits=nqubits,
        rabi_pulse=constant_pulse(rabi_amplitude, device=device),
        detuning_pulse=constant_pulse(detuning_amplitude, device=device),
        addressing=addressing,
        device=device
    )