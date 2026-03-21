"""
Time evolution solver for quantum systems.

Provides methods for simulating quantum dynamics:
    - Schrödinger equation evolution
    - Unitary evolution
    - Time-dependent Hamiltonian evolution

Uses torchdiffeq for efficient differentiable ODE solving.
"""

import torch
from typing import Callable, Optional, Tuple, Union
import torchdiffeq as tde

from ..backend import backend
from ..config import DEVICE, DTYPE_COMPLEX


# =============================================================================
# Schrödinger Equation Solver
# =============================================================================

def _complex_to_real(z: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor to real representation [..., 2] with [real, imag]."""
    return torch.stack([z.real, z.imag], dim=-1)


def _real_to_complex(y: torch.Tensor) -> torch.Tensor:
    """Convert real representation [..., 2] back to complex tensor."""
    return torch.complex(y[..., 0], y[..., 1])


def schrodinger_evolution(
    initial_state: torch.Tensor,
    hamiltonian_fn: Callable[[float], torch.Tensor],
    t_span: Tuple[float, float],
    t_eval: Optional[torch.Tensor] = None,
    method: str = 'dopri5',
    rtol: float = 1e-7,
    atol: float = 1e-9,
    **kwargs
) -> torch.Tensor:
    """
    Solve the Schrödinger equation: i dψ/dt = H(t) ψ.

    Parameters
    ----------
    initial_state : torch.Tensor
        Initial quantum state, shape [d, 1] or [batch, d, 1]
    hamiltonian_fn : Callable[[float], torch.Tensor]
        Function that returns H(t) given time t. Should return tensor
        of shape [d, d] or [batch, d, d]
    t_span : Tuple[float, float]
        Time interval (t_start, t_end)
    t_eval : torch.Tensor, optional
        Specific time points to evaluate at. If None, uses t_span.
    method : str
        ODE solver method ('dopri5', 'rk4', 'euler', etc.)
    rtol, atol : float
        Relative and absolute tolerances for adaptive stepping
    **kwargs
        Additional arguments for torchdiffeq.odeint

    Returns
    -------
    torch.Tensor
        Time-evolved state(s). Shape:
        - [len(t_eval), d, 1] if no batch
        - [len(t_eval), batch, d, 1] if batched

    Examples
    --------
    >>> # Evolve a single qubit under constant Hamiltonian
    >>> from ..hardware.rydberg import create_constant_hamiltonian
    >>> psi0 = basis_tensor('0', dim=3)
    >>> ham = create_constant_hamiltonian(1, rabi_amplitude=1.0, detuning_amplitude=0.0)
    >>> result = schrodinger_evolution(psi0, ham, (0.0, 1.0))
    """
    # Ensure initial state is complex
    if not torch.is_complex(initial_state):
        initial_state = initial_state.to(dtype=DTYPE_COMPLEX)

    # Determine if we're doing batched evolution
    is_batched = initial_state.dim() == 3 and initial_state.shape[0] != 1

    if is_batched:
        # Batched evolution
        batch_size = initial_state.shape[0]
        state_dim = initial_state.shape[1]

        # Flatten batch dimensions and convert to real representation
        # Shape: [batch_size, state_dim, 1] -> [batch_size * state_dim] (complex)
        initial_flat = initial_state.reshape(batch_size * state_dim)
        # Convert to real: [batch_size * state_dim, 2]
        initial_real = _complex_to_real(initial_flat)

        def ode_func(t, y):
            # y shape: [batch_size * state_dim, 2]
            # Convert back to complex
            y_complex = _real_to_complex(y)  # [batch_size * state_dim]
            # Reshape to [batch_size, state_dim, 1]
            y_reshaped = y_complex.reshape(batch_size, state_dim, 1)

            # Get Hamiltonian
            H = hamiltonian_fn(t)
            if H.dim() == 2:
                # Same H for all batch elements
                H = H.unsqueeze(0).expand(batch_size, -1, -1)

            # Schrödinger equation: dψ/dt = -i H ψ
            dydt_complex = -1.0j * torch.bmm(H, y_reshaped)

            # Flatten and convert to real
            dydt_flat = dydt_complex.reshape(batch_size * state_dim)
            dydt_real = _complex_to_real(dydt_flat)

            return dydt_real

    else:
        # Single state evolution
        state_dim = initial_state.shape[0]
        initial_flat = initial_state.reshape(-1)
        initial_real = _complex_to_real(initial_flat)

        def ode_func(t, y):
            # y shape: [state_dim, 2]
            # Convert to complex
            y_complex = _real_to_complex(y)  # [state_dim]
            # Reshape to [state_dim, 1]
            y_reshaped = y_complex.reshape(state_dim, 1)

            # Get Hamiltonian
            H = hamiltonian_fn(t)

            # Schrödinger equation: dψ/dt = -i H ψ
            dydt_complex = -1.0j * torch.matmul(H, y_reshaped)

            # Flatten and convert to real
            dydt_flat = dydt_complex.reshape(-1)
            dydt_real = _complex_to_real(dydt_flat)

            return dydt_real

    # Solve ODE
    if t_eval is None:
        t_eval = torch.linspace(t_span[0], t_span[1], 2)

    solution_real = tde.odeint(
        ode_func,
        initial_real,
        t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        **kwargs
    )

    # Convert solution back to complex and reshape
    # solution_real shape: [n_times, state_dim, 2] or [n_times, batch_size * state_dim, 2]
    if is_batched:
        n_times = solution_real.shape[0]
        # Convert to complex: [n_times, batch_size * state_dim]
        solution_complex = _real_to_complex(solution_real.reshape(n_times, -1, 2))
        # Reshape: [n_times, batch_size, state_dim, 1]
        solution = solution_complex.reshape(n_times, batch_size, state_dim, 1)
    else:
        n_times = solution_real.shape[0]
        # Convert to complex: [n_times, state_dim]
        solution = _real_to_complex(solution_real.reshape(n_times, -1, 2))
        # Reshape: [n_times, state_dim, 1]
        solution = solution.reshape(n_times, state_dim, 1)

    return solution


def evolve_unitary(
    initial_unitary: torch.Tensor,
    hamiltonian_fn: Callable[[float], torch.Tensor],
    t_span: Tuple[float, float],
    t_eval: Optional[torch.Tensor] = None,
    method: str = 'dopri5',
    **kwargs
) -> torch.Tensor:
    """
    Evolve a unitary operator under time-dependent Hamiltonian.

    Solves: i dU/dt = H(t) U

    Parameters
    ----------
    initial_unitary : torch.Tensor
        Initial unitary matrix, shape [d, d]
    hamiltonian_fn : Callable[[float], torch.Tensor]
        Hamiltonian function H(t)
    t_span : Tuple[float, float]
        Time interval
    t_eval : torch.Tensor, optional
        Evaluation time points
    method : str
        ODE solver method
    **kwargs
        Additional arguments for odeint

    Returns
    -------
    torch.Tensor
        Evolved unitary operators, shape [n_times, d, d]

    Examples
    --------
    >>> # Evolve identity under some Hamiltonian
    >>> U0 = torch.eye(4, dtype=torch.cfloat)
    >>> U_t = evolve_unitary(U0, hamiltonian, (0.0, 1.0))
    """
    d = initial_unitary.shape[0]

    # Flatten unitary to state vector - keep complex!
    y0 = initial_unitary.reshape(-1)

    def ode_func(t, y):
        # y shape: [d*d] - complex
        # Reshape to [d, d]
        U = y.reshape(d, d)

        # Get Hamiltonian
        H = hamiltonian_fn(t)

        # dU/dt = -i H U
        dUdt = -1.0j * torch.matmul(H, U)

        # Flatten back
        return dUdt.reshape(-1)

    if t_eval is None:
        t_eval = torch.linspace(t_span[0], t_span[1], 2)

    solution = tde.odeint(
        ode_func,
        y0,
        t_eval,
        method=method,
        **kwargs
    )

    # Reshape back to unitary matrices
    # solution shape: [n_times, d*d]
    n_times = solution.shape[0]
    solution = solution.reshape(n_times, d, d)

    return solution


# =============================================================================
# Evolution Operators
# =============================================================================

def time_evolution_operator(
    hamiltonian_fn: Callable[[float], torch.Tensor],
    t_span: Tuple[float, float],
    method: str = 'dopri5',
    **kwargs
) -> torch.Tensor:
    """
    Compute the time evolution operator U(t) over a time interval.

    This is equivalent to evolving the identity matrix.

    Parameters
    ----------
    hamiltonian_fn : Callable[[float], torch.Tensor]
        Hamiltonian function H(t)
    t_span : Tuple[float, float]
        Time interval (t_start, t_end)
    method : str
        ODE solver method
    **kwargs
        Additional arguments

    Returns
    -------
    torch.Tensor
        Time evolution operator U(t_end), shape [d, d]
    """
    # Get Hilbert space dimension from a sample Hamiltonian
    H_sample = hamiltonian_fn(t_span[0])
    d = H_sample.shape[-1]

    # Evolve identity
    U0 = backend.eye(d, dtype=DTYPE_COMPLEX, device=H_sample.device)
    U_t = evolve_unitary(U0, hamiltonian_fn, t_span, method=method, **kwargs)

    # Return final unitary
    return U_t[-1]


def evolve_state(
    initial_state: torch.Tensor,
    unitary: torch.Tensor
) -> torch.Tensor:
    """
    Evolve a quantum state by applying a unitary operator.

    Parameters
    ----------
    initial_state : torch.Tensor
        Initial state |ψ⟩, shape [d, 1] or [batch, d, 1]
    unitary : torch.Tensor
        Unitary operator U, shape [d, d] or [batch, d, d]

    Returns
    -------
    torch.Tensor
        Evolved state |ψ'⟩ = U|ψ⟩, same shape as initial_state

    Examples
    --------
    >>> psi = basis_tensor('00', dim=3)
    >>> U = czphi_gate(torch.pi)
    >>> psi_final = evolve_state(psi, U)
    """
    if initial_state.dim() == 2 and unitary.dim() == 2:
        # Single state, single unitary
        return torch.matmul(unitary, initial_state)
    elif initial_state.dim() == 3 and unitary.dim() == 3:
        # Batched
        return torch.bmm(unitary, initial_state)
    elif initial_state.dim() == 2 and unitary.dim() == 3:
        # Single state, batched unitaries
        return torch.bmm(unitary, initial_state.unsqueeze(0).expand(unitary.shape[0], -1, -1))
    else:
        raise ValueError(f"Shape mismatch: state {initial_state.shape}, unitary {unitary.shape}")


# =============================================================================
# Helper Functions
# =============================================================================

def mesolve(
    hamiltonian_fn: Callable[[float], torch.Tensor],
    initial_state: torch.Tensor,
    t_list: torch.Tensor,
    method: str = 'dopri5',
    **kwargs
) -> torch.Tensor:
    """
    Master equation solver (wrapper for schrodinger_evolution).

    Mimics the QuTiP mesolve interface for compatibility.

    Parameters
    ----------
    hamiltonian_fn : Callable
        Hamiltonian function H(t)
    initial_state : torch.Tensor
        Initial state
    t_list : torch.Tensor
        List of time points
    method : str
        ODE solver method
    **kwargs
        Additional arguments

    Returns
    -------
    torch.Tensor
        States at each time point, shape [len(t_list), d, 1]
    """
    t_span = (t_list[0].item(), t_list[-1].item())

    return schrodinger_evolution(
        initial_state,
        hamiltonian_fn,
        t_span,
        t_eval=t_list,
        method=method,
        **kwargs
    )