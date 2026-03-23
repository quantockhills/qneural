"""
Quantum evolution with neural network-generated pulses.

Integrates pulse generation, Hamiltonian evolution, and automatic
corrections to compute final unitaries for quantum control.
"""

import torch
from typing import Callable, Optional, Tuple, List, Dict

from ..core.gates import single_qubit_phase_correction
from ..core.evolution import evolve_unitary
from ..core.metrics import unitary_infidelity
from ..hardware.rydberg import RydbergHamiltonian
from .solvers import ODESolver, TorchDiffeqSolver


class QuantumEvolver:
    """
    Evolves quantum system using NN-generated pulses.
    
    This class orchestrates the full pipeline:
    1. Generate pulses from NN output
    2. Create Hamiltonian with pulse functions
    3. Evolve initial state/unitary under time-dependent Hamiltonian
    4. Apply automatic single-qubit phase corrections
    5. Return final unitary
    
    Parameters
    ----------
    nqubits : int
        Number of qubits
    solver : ODESolver, optional
        ODE solver to use (default: TorchDiffeqSolver)
    hilbert_dim : int, optional
        Hilbert space dimension per qubit (default: 3 for GG-qubits)
    
    Examples
    --------
    >>> from qneural.neural import FeedForwardNN, PulseGenerator
    >>> 
    >>> # Setup
    >>> nn = FeedForwardNN(input_dim=2, output_dim=2)
    >>> pulse_gen = PulseGenerator(n_controls=2, n_time_steps=201, ...)
    >>> evolver = QuantumEvolver(nqubits=2)
    >>> 
    >>> # Generate pulses and evolve
    >>> angle = torch.tensor([0.5 * torch.pi])
    >>> nn_output = nn(generate_input(angle, 201))
    >>> pulses = pulse_gen.generate(nn_output, gate_time=5.0)
    >>> 
    >>> # Evolve
    >>> final_U = evolver.evolve(pulses, gate_time=5.0)
    """
    
    def __init__(
        self,
        nqubits: int,
        solver: Optional[ODESolver] = None,
        hilbert_dim: int = 3,
        n_time_steps: int = 201
    ):
        self.nqubits = nqubits
        self.solver = solver or TorchDiffeqSolver()
        self.hilbert_dim = hilbert_dim
        self.full_dim = hilbert_dim ** nqubits
        self.comp_dim = 2 ** nqubits  # Computational subspace
        self.n_time_steps = n_time_steps
    
    def evolve(
        self,
        pulse_functions: List[Callable],
        gate_time: float,
        initial_unitary: Optional[torch.Tensor] = None,
        apply_corrections: bool = True
    ) -> torch.Tensor:
        """
        Evolve system with given pulses and return final unitary.
        
        Parameters
        ----------
        pulse_functions : List[Callable]
            Pulse functions [rabi_fn, detuning_fn]
        gate_time : float
            Total gate time
        initial_unitary : torch.Tensor, optional
            Initial unitary (default: identity)
        apply_corrections : bool
            Whether to apply single-qubit phase corrections (default: True)
        
        Returns
        -------
        torch.Tensor
            Final unitary in computational subspace [comp_dim, comp_dim]
        """
        # Create Hamiltonian with pulse functions
        hamiltonian = RydbergHamiltonian(
            nqubits=self.nqubits,
            rabi_pulse=pulse_functions[0],
            detuning_pulse=pulse_functions[1],
            addressing='global'
        )
        
        # Set initial unitary
        if initial_unitary is None:
            initial_unitary = torch.eye(
                self.full_dim,
                dtype=torch.cfloat,
                device=hamiltonian.rabi_ops[0].device
            )
        
        # Evolve
        evolved_unitary = self._evolve_with_solver(
            hamiltonian,
            initial_unitary,
            gate_time
        )
        
        # Reduce to computational subspace
        reduced_unitary = self._reduce_to_computational_basis(evolved_unitary)
        
        # Apply single-qubit phase corrections if requested
        if apply_corrections:
            reduced_unitary = self._apply_phase_corrections(reduced_unitary)
        
        return reduced_unitary
    
    def _evolve_with_solver(
        self,
        hamiltonian: RydbergHamiltonian,
        initial_unitary: torch.Tensor,
        gate_time: float
    ) -> torch.Tensor:
        """Evolve unitary using the configured solver."""
        # Create time evaluation points matching pulse discretization
        # This is critical for RK4 to work properly (needs many steps)
        t_eval = torch.linspace(0.0, gate_time, self.n_time_steps)
        
        # Use our evolution module
        U_t = evolve_unitary(
            initial_unitary,
            hamiltonian,
            t_span=(0.0, gate_time),
            t_eval=t_eval,
            method=self.solver.method if hasattr(self.solver, 'method') else 'dopri5'
        )
        
        return U_t[-1]  # Return final unitary
    
    def _reduce_to_computational_basis(
        self,
        unitary: torch.Tensor
    ) -> torch.Tensor:
        """
        Reduce full unitary to computational subspace.
        
        For GG-qubits (dim=3), this removes the Rydberg state |r⟩
        to get the 2^n computational subspace.
        """
        if self.hilbert_dim == 2:
            # Already in computational basis (GR-qubits)
            return unitary
        
        # For GG-qubits, extract computational subspace
        # Keep only states with indices 0, 1 (not 2 which is |r⟩)
        indices_to_keep = self._get_computational_indices()
        
        # Extract submatrix
        reduced = unitary[indices_to_keep][:, indices_to_keep]
        
        return reduced
    
    def _get_computational_indices(self) -> List[int]:
        """Get indices of computational basis states (no Rydberg)."""
        indices = []
        for i in range(self.full_dim):
            # Convert index to base-hilbert_dim representation
            digits = []
            n = i
            for _ in range(self.nqubits):
                digits.append(n % self.hilbert_dim)
                n //= self.hilbert_dim
            
            # Check if any digit is 2 (Rydberg state)
            if 2 not in digits:
                indices.append(i)
        
        return indices
    
    def _apply_phase_corrections(
        self,
        unitary: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply single-qubit phase corrections using symmetric formula.
        
        Uses only |01⟩ phase and applies symmetrically:
        - |01⟩ and |10⟩ get e^{-iφ}
        - |11⟩ gets e^{-2iφ} = (e^{-iφ})²
        
        This matches the original paper's approach for symmetric pulses.
        """
        # Extract phase from |01⟩ state only
        phi_01 = torch.angle(unitary[1, 1])
        
        # Construct symmetric correction
        j1 = torch.exp(-1.0j * phi_01)
        correction = torch.eye(4, dtype=torch.cfloat, device=unitary.device)
        correction[1, 1] = j1      # |01⟩
        correction[2, 2] = j1      # |10⟩ (same phase)
        correction[3, 3] = j1 ** 2  # |11⟩ (squared)
        
        # Apply correction
        corrected = correction @ unitary
        
        return corrected
    
    def compute_infidelity(
        self,
        achieved_unitary: torch.Tensor,
        target_unitary: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute infidelity between achieved and target unitaries.
        
        Parameters
        ----------
        achieved_unitary : torch.Tensor
            Achieved unitary [comp_dim, comp_dim]
        target_unitary : torch.Tensor
            Target unitary [comp_dim, comp_dim]
        
        Returns
        -------
        torch.Tensor
            Infidelity (scalar)
        """
        return unitary_infidelity(
            achieved_unitary,
            target_unitary,
            nqubits=self.nqubits
        )


class BatchedQuantumEvolver(QuantumEvolver):
    """
    Evolves multiple angles simultaneously (batch processing).
    
    This is useful for training on angle families.
    
    Parameters
    ----------
    nqubits : int
        Number of qubits
    batch_size : int
        Number of angles to process simultaneously
    solver : ODESolver, optional
        ODE solver
    
    Examples
    --------
    >>> evolver = BatchedQuantumEvolver(nqubits=2, batch_size=80)
    >>> 
    >>> # Evolve batch of angles
    >>> all_pulses = [pulse_gen.generate(out, t) for out, t in zip(nn_outputs, gate_times)]
    >>> final_unitaries = evolver.evolve_batch(all_pulses, gate_times)
    """
    
    def __init__(
        self,
        nqubits: int,
        batch_size: int,
        solver: Optional[ODESolver] = None,
        hilbert_dim: int = 3
    ):
        super().__init__(nqubits, solver, hilbert_dim)
        self.batch_size = batch_size
    
    def evolve_batch(
        self,
        pulse_batches: List[List[Callable]],
        gate_times: torch.Tensor,
        apply_corrections: bool = True
    ) -> torch.Tensor:
        """
        Evolve batch of angles and return final unitaries.
        
        Parameters
        ----------
        pulse_batches : List[List[Callable]]
            List of pulse function lists, one per angle
        gate_times : torch.Tensor
            Gate times for each angle [batch_size]
        apply_corrections : bool
            Whether to apply phase corrections
        
        Returns
        -------
        torch.Tensor
            Final unitaries [batch_size, comp_dim, comp_dim]
        """
        unitaries = []
        
        for pulses, t in zip(pulse_batches, gate_times):
            U = self.evolve(pulses, t.item(), apply_corrections=apply_corrections)
            unitaries.append(U)
        
        return torch.stack(unitaries)


# Convenience functions

def create_evolver(
    nqubits: int,
    backend: str = 'torchdiffeq',
    **solver_kwargs
) -> QuantumEvolver:
    """
    Factory function to create QuantumEvolver.
    
    Parameters
    ----------
    nqubits : int
        Number of qubits
    backend : str
        ODE solver backend
    **solver_kwargs
        Solver-specific options
    
    Returns
    -------
    QuantumEvolver
        Configured evolver
    
    Examples
    --------
    >>> # Default evolver
    >>> evolver = create_evolver(nqubits=2)
    >>> 
    >>> # With custom solver
    >>> evolver = create_evolver(nqubits=2, backend='fixedstep', n_steps=200)
    """
    from .solvers import create_solver
    
    solver = create_solver(backend, **solver_kwargs)
    return QuantumEvolver(nqubits=nqubits, solver=solver)


def quick_evolve(
    pulse_functions: List[Callable],
    gate_time: float,
    nqubits: int = 2,
    apply_corrections: bool = True
) -> torch.Tensor:
    """
    Quick evolution without creating evolver instance.
    
    Parameters
    ----------
    pulse_functions : List[Callable]
        [rabi_fn, detuning_fn]
    gate_time : float
        Gate time
    nqubits : int
        Number of qubits
    apply_corrections : bool
        Apply phase corrections
    
    Returns
    -------
    torch.Tensor
        Final unitary
    
    Examples
    --------
    >>> pulses = pulse_gen.generate(nn_output, gate_time=5.0)
    >>> U = quick_evolve(pulses, gate_time=5.0, nqubits=2)
    """
    evolver = create_evolver(nqubits)
    return evolver.evolve(pulse_functions, gate_time, apply_corrections=apply_corrections)