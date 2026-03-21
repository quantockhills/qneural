"""
ODE solvers for quantum evolution.

Provides an abstract interface for ODE solvers, allowing easy swapping
between different backends (torchdiffeq, diffrax, custom implementations).
"""

import torch
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Dict, Any
import torchdiffeq as tde


class ODESolver(ABC):
    """
    Abstract base class for ODE solvers.
    
    This interface allows swapping between different ODE solving backends
    (torchdiffeq, diffrax, custom implementations) without changing
the rest of the codebase.
    
    Examples
    --------
    >>> # Using torchdiffeq backend
    >>> solver = TorchDiffeqSolver(method='dopri5')
    >>> 
    >>> # Future: Using diffrax
    >>> # solver = DiffraxSolver()
    >>> 
    >>> # Solve ODE
    >>> solution = solver.solve(dy_dt, y0, t_span=(0, 1))
    """
    
    @abstractmethod
    def solve(
        self,
        ode_func: Callable[[float, torch.Tensor], torch.Tensor],
        y0: torch.Tensor,
        t_span: Tuple[float, float],
        t_eval: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Solve ODE: dy/dt = f(t, y)
        
        Parameters
        ----------
        ode_func : Callable
            Function f(t, y) returning dy/dt
        y0 : torch.Tensor
            Initial condition
        t_span : Tuple[float, float]
            Time interval (t_start, t_end)
        t_eval : torch.Tensor, optional
            Specific time points to evaluate at
        **kwargs
            Solver-specific options
        
        Returns
        -------
        torch.Tensor
            Solution at evaluation points
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return solver name for logging/debugging."""
        pass


class TorchDiffeqSolver(ODESolver):
    """
    ODE solver using torchdiffeq backend.
    
    Wraps torchdiffeq.odeint with our standard interface.
    
    Parameters
    ----------
    method : str
        Integration method: 'dopri5', 'rk4', 'euler', etc.
    rtol : float
        Relative tolerance (for adaptive methods)
    atol : float
        Absolute tolerance (for adaptive methods)
    options : dict
        Additional solver-specific options
    
    Examples
    --------
    >>> solver = TorchDiffeqSolver(method='dopri5', rtol=1e-7, atol=1e-9)
    >>> 
    >>> # Define ODE
    >>> def f(t, y):
    ...     return -y  # dy/dt = -y
    >>> 
    >>> # Solve
    >>> y0 = torch.tensor([1.0])
    >>> solution = solver.solve(f, y0, t_span=(0, 1))
    """
    
    def __init__(
        self,
        method: str = 'dopri5',
        rtol: float = 1e-7,
        atol: float = 1e-9,
        options: Optional[Dict[str, Any]] = None
    ):
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.options = options or {}
    
    def solve(
        self,
        ode_func: Callable[[float, torch.Tensor], torch.Tensor],
        y0: torch.Tensor,
        t_span: Tuple[float, float],
        t_eval: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Solve ODE using torchdiffeq."""
        if t_eval is None:
            t_eval = torch.linspace(t_span[0], t_span[1], 2)
        
        # Merge default options with any overrides
        solve_kwargs = {
            'method': self.method,
            'rtol': self.rtol,
            'atol': self.atol,
            **self.options,
            **kwargs
        }
        
        solution = tde.odeint(ode_func, y0, t_eval, **solve_kwargs)
        
        return solution
    
    def get_name(self) -> str:
        return f"TorchDiffeq_{self.method}"


class FixedStepSolver(ODESolver):
    """
    Simple fixed-step ODE solver (e.g., RK4).
    
    Useful for debugging or when adaptive stepping is not needed.
    
    Parameters
    ----------
    method : str
        'rk4' or 'euler'
    n_steps : int
        Number of fixed steps
    
    Examples
    --------
    >>> solver = FixedStepSolver(method='rk4', n_steps=100)
    """
    
    def __init__(self, method: str = 'rk4', n_steps: int = 100):
        self.method = method
        self.n_steps = n_steps
    
    def solve(
        self,
        ode_func: Callable[[float, torch.Tensor], torch.Tensor],
        y0: torch.Tensor,
        t_span: Tuple[float, float],
        t_eval: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Solve ODE with fixed steps."""
        t_start, t_end = t_span
        dt = (t_end - t_start) / self.n_steps
        
        y = y0
        t = t_start
        
        if t_eval is not None:
            # Store solution at specific points
            solution = [y]
            eval_idx = 1
            
            for _ in range(self.n_steps):
                y = self._step(ode_func, t, y, dt)
                t = t + dt
                
                # Check if we need to save this point
                while eval_idx < len(t_eval) and t >= t_eval[eval_idx].item():
                    solution.append(y)
                    eval_idx += 1
            
            # Ensure we have all evaluation points
            while len(solution) < len(t_eval):
                solution.append(y)
            
            return torch.stack(solution)
        else:
            # Just return final value
            for _ in range(self.n_steps):
                y = self._step(ode_func, t, y, dt)
                t = t + dt
            
            return y
    
    def _step(
        self,
        f: Callable,
        t: float,
        y: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """Take one integration step."""
        if self.method == 'euler':
            # Euler method
            return y + dt * f(t, y)
        
        elif self.method == 'rk4':
            # Runge-Kutta 4
            k1 = f(t, y)
            k2 = f(t + dt/2, y + dt*k1/2)
            k3 = f(t + dt/2, y + dt*k2/2)
            k4 = f(t + dt, y + dt*k3)
            return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_name(self) -> str:
        return f"FixedStep_{self.method}_{self.n_steps}"


# Future placeholder for Diffrax backend
class DiffraxSolver(ODESolver):
    """
    ODE solver using Diffrax (JAX-based).
    
    This is a placeholder for future JAX integration.
    Diffrax provides high-performance ODE solving with XLA compilation.
    
    Note: This requires JAX to be installed.
    """
    
    def __init__(self, **kwargs):
        raise NotImplementedError(
            "DiffraxSolver is a placeholder for future JAX integration. "
            "Please use TorchDiffeqSolver for now."
        )
    
    def solve(self, *args, **kwargs):
        raise NotImplementedError()
    
    def get_name(self):
        return "Diffrax"


# Solver factory
def create_solver(
    backend: str = 'torchdiffeq',
    **kwargs
) -> ODESolver:
    """
    Factory function to create ODE solvers.
    
    Parameters
    ----------
    backend : str
        Solver backend: 'torchdiffeq', 'fixedstep'
    **kwargs
        Backend-specific options
    
    Returns
    -------
    ODESolver
        Configured ODE solver
    
    Examples
    --------
    >>> # Default adaptive solver
    >>> solver = create_solver('torchdiffeq', method='dopri5')
    >>> 
    >>> # Fixed-step solver
    >>> solver = create_solver('fixedstep', method='rk4', n_steps=200)
    """
    if backend == 'torchdiffeq':
        return TorchDiffeqSolver(**kwargs)
    elif backend == 'fixedstep':
        return FixedStepSolver(**kwargs)
    elif backend == 'diffrax':
        return DiffraxSolver(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Convenience functions

def solve_ivp(
    ode_func: Callable,
    y0: torch.Tensor,
    t_span: Tuple[float, float],
    solver: Optional[ODESolver] = None,
    **kwargs
) -> torch.Tensor:
    """
    Convenience function to solve initial value problem.
    
    Parameters
    ----------
    ode_func : Callable
        ODE function f(t, y)
    y0 : torch.Tensor
        Initial condition
    t_span : Tuple[float, float]
        Time span
    solver : ODESolver, optional
        Solver instance. If None, uses default TorchDiffeqSolver
    **kwargs
        Additional solver arguments
    
    Returns
    -------
    torch.Tensor
        Solution
    
    Examples
    --------
    >>> solution = solve_ivp(f, y0, (0, 1), method='rk4')
    """
    if solver is None:
        method = kwargs.pop('method', 'dopri5')
        solver = TorchDiffeqSolver(method=method)
    
    return solver.solve(ode_func, y0, t_span, **kwargs)