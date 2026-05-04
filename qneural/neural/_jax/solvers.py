"""
JAX ODE solvers for quantum evolution using diffrax.
"""

import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, Tsit5, Euler, PIDController
from diffrax import SaveAt
from typing import Callable, Optional, Tuple, Dict, Any


class JaxOdeSolver:
    def __init__(
        self,
        method: str = "dopri5",
        rtol: float = 1e-7,
        atol: float = 1e-9,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.options = options or {}

    def solve(
        self,
        ode_func: Callable,
        y0,
        t_span: Tuple[float, float],
        t_eval=None,
        **kwargs,
    ):
        if t_eval is None:
            t_eval = jnp.linspace(t_span[0], t_span[1], 2)

        solver = _get_diffrax_solver(kwargs.get("method", self.method))
        r_tol = kwargs.get("rtol", self.rtol)
        a_tol = kwargs.get("atol", self.atol)

        wrapped = lambda t, y, args: ode_func(t, y)

        term = ODETerm(wrapped)
        saveat = SaveAt(ts=t_eval)
        stepsize_controller = None  # default adaptive

        sol = diffeqsolve(
            term,
            solver,
            t0=t_span[0],
            t1=t_span[-1],
            dt0=0.01,
            y0=y0,
            saveat=saveat,
            stepsize_controller=PIDController(rtol=r_tol, atol=a_tol),
            max_steps=kwargs.get("max_steps", 4096),
        )
        return sol.ys

    def get_name(self) -> str:
        return f"JaxOde_{self.method}"


def _get_diffrax_solver(method: str):
    solvers = {
        "dopri5": Dopri5(),
        "tsit5": Tsit5(),
        "euler": Euler(),
    }
    if method in solvers:
        return solvers[method]
    if method in ("rk4", "midpoint"):
        return Dopri5()
    raise ValueError(f"Unknown method: {method}")
