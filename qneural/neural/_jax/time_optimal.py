"""
JAX/equinox implementation of time-optimal quantum control with fixed Rabi frequency.

This module provides time-optimal control where:
- Rabi frequency is constant at maximum (by definition of time-optimal)
- Only detuning is learned
- Gate time is predicted and optimized jointly

Uses equinox.Module for networks, optax for optimization, jax.grad for training,
and diffrax for ODE solving.
"""

import math
import re
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Optional, Tuple, Callable, Dict, List
from itertools import combinations

from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController
from .models import _Dense
from .solvers import JaxOdeSolver

# =============================================================================
# Physical constants
# =============================================================================

RABI_DEFAULT = 2 * math.pi * 4.0  # ~25.13 rad/s
VDD_COUPLING = 21.1
VDD = VDD_COUPLING * RABI_DEFAULT


# =============================================================================
# JAX Rydberg operator helpers
# =============================================================================

def _tensor_product(ops: List[jax.Array]) -> jax.Array:
    """Kronecker product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = jnp.kron(result, op)
    return result


def _create_local_rydberg_operators() -> Dict[str, jax.Array]:
    """Create single-qubit Rydberg operators (GG-qubit, dim=3) as JAX arrays."""
    # |r><1|
    ket_r_bra_1 = jnp.zeros((3, 3), dtype=jnp.complex64)
    ket_r_bra_1 = ket_r_bra_1.at[2, 1].set(1.0 + 0j)
    # |1><r|
    ket_1_bra_r = jnp.zeros((3, 3), dtype=jnp.complex64)
    ket_1_bra_r = ket_1_bra_r.at[1, 2].set(1.0 + 0j)

    rabi = ket_r_bra_1 + ket_1_bra_r  # σ_x^{1,r}

    # |r><r|
    n_r = jnp.zeros((3, 3), dtype=jnp.complex64)
    n_r = n_r.at[2, 2].set(1.0 + 0j)

    return {"rabi": rabi, "detuning": n_r}


def _build_full_operators(nqubits: int):
    """Build rabi_ops, detuning_ops, interaction_op for n qubits as JAX arrays.

    Returns
    -------
    rabi_ops : list of jax.Array
        Length nqubits, each [3^n, 3^n]
    detuning_ops : list of jax.Array
        Length nqubits, each [3^n, 3^n]
    interaction_op : jax.Array or None
        [3^n, 3^n] or None if nqubits < 2
    """
    local = _create_local_rydberg_operators()
    identity = jnp.eye(3, dtype=jnp.complex64)

    rabi_ops = []
    detuning_ops = []
    for i in range(nqubits):
        ops = [identity] * nqubits
        ops[i] = local["rabi"]
        rabi_ops.append(_tensor_product(ops))

        ops = [identity] * nqubits
        ops[i] = local["detuning"]
        detuning_ops.append(_tensor_product(ops))

    if nqubits >= 2:
        interaction_op = jnp.zeros(
            (3**nqubits, 3**nqubits), dtype=jnp.complex64
        )
        for i, j in combinations(range(nqubits), 2):
            ops = [identity] * nqubits
            ops[i] = local["detuning"]
            ops[j] = local["detuning"]
            interaction_op += _tensor_product(ops)
    else:
        interaction_op = None

    return rabi_ops, detuning_ops, interaction_op


# =============================================================================
# JAX target gate functions
# =============================================================================

def _czphi_gate_jax(phi: float) -> jax.Array:
    """CZ_φ gate: diag(1, 1, 1, e^{iφ}) as JAX array [4, 4]."""
    gate = jnp.eye(4, dtype=jnp.complex64)
    gate = gate.at[3, 3].set(jnp.exp(1j * phi))
    return gate


def _cczphi_gate_jax(phi: float) -> jax.Array:
    """CCZ_φ gate: diag(1,...,1, e^{iφ}) as JAX array [8, 8]."""
    gate = jnp.eye(8, dtype=jnp.complex64)
    gate = gate.at[7, 7].set(jnp.exp(1j * phi))
    return gate


# =============================================================================
# Batched fidelity / infidelity (JAX)
# =============================================================================

def _compute_batch_fidelity(
    achieved: jax.Array, target: jax.Array, nqubits: int
) -> jax.Array:
    """Average gate fidelity for batched unitaries: F = |Tr(U1^† U2)|^2 / d^2.

    achieved: [batch, d, d], target: [batch, d, d]  where d = 2^nqubits.
    Returns [batch] fidelities.
    """
    d = 2**nqubits
    # U1^† U2 = achieved^† @ target  [batch, d, d]
    product = jnp.einsum("bji,bjk->bik", jnp.conj(achieved), target)
    overlap = jnp.einsum("bii->b", product).real
    return overlap**2 / (d * d)


def _compute_batch_infidelity(
    achieved: jax.Array, target: jax.Array, nqubits: int
) -> jax.Array:
    """1 - fidelity per batch element."""
    return 1.0 - _compute_batch_fidelity(achieved, target, nqubits)


# =============================================================================
# Time-Optimal Controller (eqx.Module)
# =============================================================================

class TimeOptimalController(eqx.Module):
    """Two-network system for time-optimal quantum control.

    Architecture:
    1. Time Predictor: angle -> normalized_time [0,1]
    2. Control Generator: (angle, time) -> detuning values

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
    key : jax.random.PRNGKey, optional
        Random key for weight initialisation
    """

    time_predictor: eqx.nn.Sequential
    control_generator: eqx.nn.Sequential
    time_bounds: Tuple[float, float]
    rabi_max: float
    detuning_range: Tuple[float, float]
    n_time_steps: int
    time_output_activation: str
    time_grid: jax.Array

    def __init__(
        self,
        time_bounds: Tuple[float, float] = (3.0, 20.0),
        rabi_max: float = 25.13,
        detuning_range: Optional[Tuple[float, float]] = None,
        n_time_steps: int = 301,
        time_hidden_layers: int = 3,
        time_hidden_units: int = 45,
        control_hidden_layers: int = 10,
        control_hidden_units: int = 300,
        time_output_activation: str = "sigmoid",
        weight_scale_time: float = 1.8,
        weight_scale_control: float = 1.55,
        *,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)

        self.time_bounds = time_bounds
        self.rabi_max = rabi_max
        self.n_time_steps = n_time_steps
        self.time_output_activation = time_output_activation

        if detuning_range is None:
            detuning_range = (-2.0 * rabi_max, 2.0 * rabi_max)
        self.detuning_range = detuning_range

        k_time, k_ctrl = jax.random.split(key)
        self.time_predictor = _build_mlp(
            k_time,
            in_dim=1,
            out_dim=1,
            n_layers=time_hidden_layers,
            n_units=time_hidden_units,
            output_activation=time_output_activation,
            weight_scale=weight_scale_time,
        )
        self.control_generator = _build_mlp(
            k_ctrl,
            in_dim=2,
            out_dim=1,
            n_layers=control_hidden_layers,
            n_units=control_hidden_units,
            output_activation="sigmoid",
            weight_scale=weight_scale_control,
        )

        # Fixed normalized time grid [0, 1]
        self.time_grid = jnp.linspace(0.0, 1.0, n_time_steps)

    def __call__(self, angle: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Generate time-optimal detuning pulses.

        Parameters
        ----------
        angle : jax.Array
            Gate angle(s), shape [batch] or [batch, 1]

        Returns
        -------
        tuple[jax.Array, jax.Array]
            - gate_time: Predicted time [batch, 1] (in seconds)
            - detuning_normalized: Detuning values [batch, n_time_steps, 1] in [0, 1]
        """
        # Ensure proper shape [batch, 1]
        if angle.ndim == 0:
            angle = jnp.expand_dims(angle, 0)
        if angle.ndim == 1:
            angle = jnp.expand_dims(angle, -1)

        batch_size = angle.shape[0]

        # Step 1: Predict normalized time from angle
        normalized_time = self.time_predictor(angle)  # [batch, 1]

        # Step 2: Scale to physical time bounds
        t_min, t_max = self.time_bounds

        if self.time_output_activation == "tanh":
            # [-1, 1] -> [t_min, t_max]
            gate_time = 0.5 * (normalized_time + 1.0) * (t_max - t_min) + t_min
        else:
            # [0, 1] -> [t_min, t_max]
            gate_time = normalized_time * (t_max - t_min) + t_min

        # Step 3: Create scaled time grid  [batch, n_time_steps]
        time_grid = jnp.linspace(0.0, 1.0, self.n_time_steps)
        time_scaled = time_grid[jnp.newaxis, :] * gate_time

        # Step 4: Prepare (angle, time) pairs for control network
        angle_repeated = jnp.repeat(angle, self.n_time_steps, axis=0)  # [batch*n_steps, 1]
        time_flat = time_scaled.reshape(-1, 1)  # [batch*n_steps, 1]
        control_inputs = jnp.concatenate(
            [angle_repeated, time_flat], axis=-1
        )  # [batch*n_steps, 2]

        # Step 5: Generate detuning
        detuning_normalized = self.control_generator(
            control_inputs
        )  # [batch*n_steps, 1]
        detuning_normalized = detuning_normalized.reshape(
            batch_size, self.n_time_steps, 1
        )

        return gate_time, detuning_normalized

    def get_rabi_pulse_fn(self, gate_time: jax.Array) -> Callable:
        """Get constant-then-zero Rabi pulse function.

        Parameters
        ----------
        gate_time : jax.Array
            Gate time for each batch element [batch, 1] or scalar

        Returns
        -------
        Callable
            Function rabi(t) returning rabi_max for t <= gate_time, else 0
        """
        rabi_max = self.rabi_max
        # Ensure gate_time is at least 2D [batch, 1]
        if gate_time.ndim == 0:
            gate_time = jnp.expand_dims(gate_time, 0)
        if gate_time.ndim == 1:
            gate_time = jnp.expand_dims(gate_time, -1)

        def rabi_pulse(t):
            """Rabi pulse: constant until cutoff, then zero."""
            t = jnp.asarray(t)
            # gate_time shape: [batch, 1]; rabi output: [batch, 1]
            return jnp.where(t <= gate_time, rabi_max, 0.0)

        return rabi_pulse

    def get_detuning_pulse_fn(
        self, detuning_normalized: jax.Array, gate_time: jax.Array
    ) -> Callable:
        """Get piecewise-constant detuning pulse function.

        Parameters
        ----------
        detuning_normalized : jax.Array
            Normalized detuning [batch, n_time_steps, 1] in [0, 1]
        gate_time : jax.Array
            Gate time [batch, 1] or scalar

        Returns
        -------
        Callable
            Function detuning(t) returning detuning at time t
        """
        # Ensure 3D shape [batch, n_time_steps, 1]
        if detuning_normalized.ndim == 2:
            detuning_normalized = jnp.expand_dims(detuning_normalized, 0)

        # Scale to physical range
        d_min, d_max = self.detuning_range
        detuning_values = (
            detuning_normalized * (d_max - d_min) + d_min
        )  # [batch, n_time_steps, 1]

        pulse_batch_size = detuning_values.shape[0]
        step_size = gate_time / self.n_time_steps  # [batch, 1]
        n_steps = self.n_time_steps
        off_resonant_val = 20.0 * self.rabi_max

        def detuning_pulse(t):
            """Piecewise-constant detuning pulse."""
            t = jnp.asarray(t)

            if pulse_batch_size == 1:
                off_res = jnp.full((1,), off_resonant_val, dtype=jnp.float32)
                idx = jnp.clip(
                    jnp.floor(t / step_size[0, 0]).astype(jnp.int32),
                    0,
                    n_steps - 1,
                )
                det = detuning_values[0, idx, 0]
                return jnp.where(t >= gate_time[0, 0], off_res, det)
            else:
                # Batched: each element has its own step_size
                # t is scalar; t / step_size gives [batch, 1]
                indices = jnp.clip(
                    jnp.floor(t / step_size).astype(jnp.int32), 0, n_steps - 1
                ).squeeze(-1)  # [batch]

                # Start with off-resonant for all
                result = jnp.full(
                    (pulse_batch_size, 1), off_resonant_val, dtype=jnp.float32
                )

                # Within gate time: pick detuning value
                within = (t < gate_time).squeeze(-1)  # [batch]
                # Use where to select: off-resonant if outside, detuning otherwise
                batch_idx = jnp.arange(pulse_batch_size)
                selected = detuning_values[batch_idx, indices, 0]  # [batch]
                result = jnp.where(
                    within[:, None], selected[:, None], result
                )

                return result

        return detuning_pulse

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each network."""
        time_params = sum(
            jnp.size(leaf)
            for leaf in jax.tree_util.tree_leaves(
                eqx.filter(self.time_predictor, eqx.is_array)
            )
        )
        control_params = sum(
            jnp.size(leaf)
            for leaf in jax.tree_util.tree_leaves(
                eqx.filter(self.control_generator, eqx.is_array)
            )
        )
        return {
            "time_predictor": int(time_params),
            "control_generator": int(control_params),
            "total": int(time_params + control_params),
        }

    def scale_detuning(self, detuning_normalized: jax.Array) -> jax.Array:
        """Scale normalized detuning [0, 1] to physical range.

        Parameters
        ----------
        detuning_normalized : jax.Array
            Detuning values in [0, 1], shape [..., 1] or [...]

        Returns
        -------
        jax.Array
            Scaled detuning in physical units
        """
        d_min, d_max = self.detuning_range
        return detuning_normalized * (d_max - d_min) + d_min


# =============================================================================
# MLP builder helper
# =============================================================================

def _build_mlp(
    key: jax.random.PRNGKey,
    in_dim: int,
    out_dim: int,
    n_layers: int,
    n_units: int,
    output_activation: str,
    weight_scale: float,
) -> eqx.nn.Sequential:
    """Build a Sequential MLP using _Dense layers (batch-compatible)."""
    layers = []
    keys = jax.random.split(key, n_layers + 2)

    # Input layer
    layers.append(_Dense(in_dim, n_units, key=keys[0]))
    layers.append(eqx.nn.Lambda(jax.nn.relu))

    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(_Dense(n_units, n_units, key=keys[i + 1]))
        layers.append(eqx.nn.Lambda(jax.nn.relu))

    # Output layer
    layers.append(_Dense(n_units, out_dim, key=keys[n_layers]))

    # Output activation
    if output_activation == "sigmoid":
        layers.append(eqx.nn.Lambda(jax.nn.sigmoid))
    elif output_activation == "tanh":
        layers.append(eqx.nn.Lambda(jnp.tanh))
    else:
        raise ValueError(f"Unknown activation: {output_activation}")

    network = eqx.nn.Sequential(layers)

    if weight_scale != 1.0:
        network = _scale_weights(network, weight_scale)

    return network


def _scale_weights(network: eqx.nn.Sequential, scale: float):
    """Scale all floating-point weights in a PyTree by `scale`."""
    def _scale(x):
        if isinstance(x, jax.Array) and x.dtype.kind == "f":
            return x * scale
        return x

    return jax.tree_util.tree_map(_scale, network)


# =============================================================================
# Computational-subspace reduction and corrections (JAX, batched)
# =============================================================================

def _get_computational_indices(nqubits: int) -> List[int]:
    """List indices of computational subspace (no Rydberg state '2') within
    the full (3^n) Hilbert space."""
    full_dim = 3**nqubits
    indices = []
    for i in range(full_dim):
        digits = []
        n = i
        for _ in range(nqubits):
            digits.append(n % 3)
            n //= 3
        if 2 not in digits:
            indices.append(i)
    return indices


def _reduce_to_computational(
    unitaries: jax.Array, batch_size: int, nqubits: int
) -> jax.Array:
    """Reduce from full (3^n)x(3^n) to computational (2^n)x(2^n) subspace.

    unitaries: [batch, full_dim, full_dim] -> [batch, comp_dim, comp_dim]
    """
    comp_indices = _get_computational_indices(nqubits)
    comp_dim = 2**nqubits

    # Extract computational submatrix for each batch element
    # unitaries[:, comp_indices][:, :, comp_indices]
    reduced = unitaries[:, comp_indices, :][:, :, comp_indices]
    return reduced


def _apply_batch_corrections(
    unitaries: jax.Array, nqubits: int
) -> jax.Array:
    """Apply single-qubit phase corrections (batched).

    unitaries: [batch, comp_dim, comp_dim]
    Returns  [batch, comp_dim, comp_dim]
    """
    batch_size = unitaries.shape[0]
    comp_dim = 2**nqubits

    # Extract phase from |00...01> element (index 1 in computational basis)
    phi_01 = jnp.angle(unitaries[:, 1, 1])  # [batch]
    phase_factor = jnp.exp(-1j * phi_01)  # [batch]

    # Build correction matrices: diag with j1^{popcount(i)}
    # [comp_dim, comp_dim] identity, then scale diagonals per batch
    identity = jnp.eye(comp_dim, dtype=jnp.complex64)
    correction_batch = jnp.tile(identity[jnp.newaxis], (batch_size, 1, 1))

    # Count number of 1's in binary representation of each index
    # and apply (phase_factor)^{num_ones}
    for i in range(1, comp_dim):
        num_ones = bin(i).count("1")
        # phase_factor: [batch], broadcasted to diagonal element
        correction_batch = correction_batch.at[:, i, i].set(
            phase_factor**num_ones
        )

    # Batched matmul
    corrected = jnp.einsum("bij,bjk->bik", correction_batch, unitaries)
    return corrected


# =============================================================================
# Time-Optimal Trainer (JAX / functional style)
# =============================================================================

class TimeOptimalTrainer:
    """Trainer for time-optimal quantum control with dual optimizers.

    Uses separate optimizers for time and control networks with different
    learning rates, following the archival training pattern.

    Parameters
    ----------
    controller : TimeOptimalController
        The controller with time and control networks
    nqubits : int
        Number of qubits (2 for CZ_φ, 3 for CCZ_φ, etc.)
    time_weight : float
        Weight for time penalty in loss (default: 1e-4)
    time_lr : float
        Learning rate for time network (default: 1e-5)
    control_lr : float
        Learning rate for control network (default: 1e-4)
    solver : JaxOdeSolver, optional
        ODE solver (default: JaxOdeSolver with Dopri5)
    device : str
        Ignored in JAX; kept for API compatibility.
    target_gate_fn : Callable[[float], jax.Array], optional
        Function that takes an angle and returns target unitary.
        If None, defaults to _czphi_gate_jax for 2 qubits,
        _cczphi_gate_jax for 3 qubits.
    key : jax.random.PRNGKey, optional
        Random key for angle resampling
    """

    def __init__(
        self,
        controller: TimeOptimalController,
        nqubits: int,
        time_weight: float = 1e-4,
        time_lr: float = 1e-5,
        control_lr: float = 1e-4,
        solver: Optional[JaxOdeSolver] = None,
        device: str = "cpu",
        target_gate_fn: Optional[Callable] = None,
        *,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        self.controller = controller
        self.nqubits = nqubits
        self.time_weight = time_weight

        # Target gate function
        if target_gate_fn is None:
            if nqubits == 2:
                self.target_gate_fn = _czphi_gate_jax
            elif nqubits == 3:
                self.target_gate_fn = _cczphi_gate_jax
            else:
                raise ValueError(
                    f"No default target gate for {nqubits} qubits. "
                    "Please provide target_gate_fn parameter."
                )
        else:
            self.target_gate_fn = target_gate_fn

        # Solver
        if solver is None:
            self.solver = JaxOdeSolver(method="dopri5", rtol=1e-5, atol=1e-5)
        else:
            self.solver = solver

        # Dual optimizers (archival pattern)
        self.time_optimizer = optax.adamw(time_lr, eps=1e-5, b1=0.9, b2=0.999)
        self.control_optimizer = optax.adamw(control_lr, eps=1e-5, b1=0.9, b2=0.999)

        # Initialise optimizer states (will be set on first train step)
        self._time_opt_state = None
        self._control_opt_state = None
        self._initialised = False

        # Precompute full-basis operators (depend only on nqubits)
        self._rabi_ops, self._detuning_ops, self._interaction_op = (
            _build_full_operators(nqubits)
        )
        self._full_dim = 3**nqubits
        self._comp_indices = _get_computational_indices(nqubits)
        self._comp_dim = 2**nqubits

        # Training history
        self.history: Dict[str, List] = {
            "epoch": [], "loss": [], "infidelity": [], "mean_gate_time": [],
        }
        self.current_epoch = 0

        # PRNG key
        if key is None:
            key = jax.random.PRNGKey(42)
        self._key = key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        angles: jax.Array,
        epochs: int = 1000,
        print_every: int = 50,
        save_path: Optional[str] = None,
        angle_range: Optional[Tuple[float, float]] = None,
        resample_every: int = 25,
    ) -> Dict:
        """Train the time-optimal controller.

        Parameters
        ----------
        angles : jax.Array
            Initial target angles [n_angles] (multi-angle optimization)
        epochs : int
            Number of training epochs
        print_every : int
            Print progress every N epochs
        save_path : str, optional
            Path to save best model (uses equinox serialisation)
        angle_range : tuple, optional
            (min, max) angle range for resampling. If None, uses fixed angles.
        resample_every : int
            Resample angles from angle_range every N epochs (archival: 25)

        Returns
        -------
        dict
            Training history
        """
        if angles.ndim == 1:
            angles = jnp.expand_dims(angles, -1)

        batch_size = angles.shape[0]
        best_loss = float("inf")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Resample angles (archival pattern)
            if angle_range is not None and epoch % resample_every == 0:
                self._key, subkey = jax.random.split(self._key)
                angles = (
                    jax.random.uniform(subkey, (batch_size, 1))
                    * (angle_range[1] - angle_range[0])
                    + angle_range[0]
                )

            # Training step
            loss, metrics = self._train_step(angles)

            # Update history
            self.history["epoch"].append(epoch)
            self.history["loss"].append(loss)
            self.history["infidelity"].append(metrics["infidelity"])
            self.history["mean_gate_time"].append(metrics["mean_gate_time"])

            # Print progress
            if epoch % print_every == 0:
                mean_time_norm = (
                    metrics["mean_gate_time"] * self.controller.rabi_max
                )
                print(
                    f"Epoch {epoch}: Loss = {loss:.6f}, "
                    f"Infidelity = {metrics['infidelity']:.6f}, "
                    f"Mean Time = {mean_time_norm:.4f}"
                )

            # Save best model
            if save_path and loss < best_loss:
                best_loss = loss
                self.save_checkpoint(save_path)

        return self.history

    def evaluate(self, angles: jax.Array) -> Dict:
        """Evaluate controller on given angles using batched evolution.

        Returns dict with predicted times, infidelities, etc.
        """
        if angles.ndim == 1:
            angles = jnp.expand_dims(angles, -1)

        batch_size = angles.shape[0]

        # Forward pass
        gate_times, detuning_normalized = self.controller(angles)
        max_gate_time = gate_times.max()

        rabi_fn = self.controller.get_rabi_pulse_fn(max_gate_time)
        detuning_fn = self.controller.get_detuning_pulse_fn(
            detuning_normalized, gate_times
        )

        # Evolve
        final_unitaries = self._evolve_batch(
            rabi_fn, detuning_fn, max_gate_time, batch_size
        )

        # Target gates
        target_unitaries = jax.vmap(self.target_gate_fn)(angles.flatten())

        # Infidelity
        infidelities = _compute_batch_infidelity(
            final_unitaries, target_unitaries, self.nqubits
        )

        results = {
            "angles": [float(a) for a in angles.squeeze(-1)],
            "predicted_times": [float(gt) for gt in gate_times.squeeze(-1)],
            "infidelities": [float(inf) for inf in infidelities],
        }
        results["mean_infidelity"] = float(jnp.mean(jnp.array(results["infidelities"])))
        results["mean_time"] = float(jnp.mean(jnp.array(results["predicted_times"])))

        return results

    def save_checkpoint(self, path: str, metadata: Optional[dict] = None):
        """Save network weights as numpy arrays (pickle-safe)."""
        import pickle

        # Extract trainable params as numpy arrays
        time_params, _ = eqx.partition(
            self.controller.time_predictor, eqx.is_array
        )
        ctrl_params, _ = eqx.partition(
            self.controller.control_generator, eqx.is_array
        )

        time_np = _tree_to_numpy(time_params)
        ctrl_np = _tree_to_numpy(ctrl_params)

        checkpoint = {
            "controller_config": {
                "time_bounds": self.controller.time_bounds,
                "rabi_max": self.controller.rabi_max,
                "detuning_range": self.controller.detuning_range,
                "n_time_steps": self.controller.n_time_steps,
                "time_hidden_layers": len(
                    [l for l in self.controller.time_predictor.layers if hasattr(l, 'weight')]
                ),
                "time_hidden_units": _get_dense_width(self.controller.time_predictor),
                "control_hidden_layers": len(
                    [l for l in self.controller.control_generator.layers if hasattr(l, 'weight')]
                ),
                "control_hidden_units": _get_dense_width(self.controller.control_generator),
                "time_output_activation": self.controller.time_output_activation,
            },
            "time_weights": time_np,
            "ctrl_weights": ctrl_np,
            "history": self.history,
            "epoch": self.current_epoch,
            "time_weight": self.time_weight,
            "metadata": metadata or {},
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path: str):
        """Load network weights from checkpoint."""
        import pickle

        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        time_weights = checkpoint["time_weights"]
        ctrl_weights = checkpoint["ctrl_weights"]

        # Load into controller using tree_at
        def _set_leaf_at(model, subtree, idx, param, arr):
            return eqx.tree_at(
                lambda m, _i=idx, _sn=subtree, _p=param: getattr(
                    getattr(m, _sn).layers[_i], _p
                ),
                model,
                jnp.array(arr),
            )

        for (idx, param), arr in time_weights.items():
            self.controller = _set_leaf_at(
                self.controller, "time_predictor", idx, param, arr
            )
        for (idx, param), arr in ctrl_weights.items():
            self.controller = _set_leaf_at(
                self.controller, "control_generator", idx, param, arr
            )

        self.history = checkpoint.get("history", self.history)
        self.current_epoch = checkpoint.get("epoch", 0)
        if "time_weight" in checkpoint:
            self.time_weight = checkpoint["time_weight"]
        self._initialised = self._time_opt_state is not None

    # ------------------------------------------------------------------
    # Internal training
    # ------------------------------------------------------------------

    def _train_step(self, angles: jax.Array) -> Tuple[float, Dict]:
        """Single training step with dual optimizers using eqx.filter_value_and_grad."""
        if angles.ndim == 1:
            angles = jnp.expand_dims(angles, -1)

        # Lazy init optimiser states
        diffable, static = eqx.partition(self.controller, eqx.is_array)
        if not self._initialised:
            self._time_opt_state = self.time_optimizer.init(
                eqx.filter(diffable.time_predictor, eqx.is_array)
            )
            self._control_opt_state = self.control_optimizer.init(
                eqx.filter(diffable.control_generator, eqx.is_array)
            )
            self._initialised = True

        # Compute gradients w.r.t. the full controller params
        grad_fn = eqx.filter_value_and_grad(self._loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(diffable, static, angles)

        # Split gradients for dual optimizers
        time_grads = eqx.filter(grads.time_predictor, eqx.is_array)
        control_grads = eqx.filter(grads.control_generator, eqx.is_array)

        time_params = eqx.filter(diffable.time_predictor, eqx.is_array)
        control_params = eqx.filter(diffable.control_generator, eqx.is_array)

        time_updates, self._time_opt_state = self.time_optimizer.update(
            time_grads, self._time_opt_state, time_params
        )
        new_time_params = optax.apply_updates(time_params, time_updates)

        control_updates, self._control_opt_state = self.control_optimizer.update(
            control_grads, self._control_opt_state, control_params
        )
        new_ctrl_params = optax.apply_updates(control_params, control_updates)

        # Recombine into controller
        diffable = eqx.tree_at(
            lambda d: d.time_predictor, diffable,
            eqx.combine(new_time_params, eqx.filter(
                static.time_predictor, eqx.is_array, inverse=True
            ))
        )
        diffable = eqx.tree_at(
            lambda d: d.control_generator, diffable,
            eqx.combine(new_ctrl_params, eqx.filter(
                static.control_generator, eqx.is_array, inverse=True
            ))
        )
        self.controller = eqx.combine(diffable, static)

        return float(loss), {k: float(v) for k, v in metrics.items()}

    @eqx.filter_jit
    def _loss_fn(self, diffable, static, angles: jax.Array):
        """Pure loss function: (diffable, static, angles) -> (loss, metrics)."""
        controller = eqx.combine(diffable, static)

        batch_size = angles.shape[0]

        gate_times, detuning_normalized = controller(angles)
        max_gate_time = gate_times.max()

        rabi_fn = controller.get_rabi_pulse_fn(max_gate_time)
        detuning_fn = controller.get_detuning_pulse_fn(
            detuning_normalized, gate_times
        )

        final_unitaries = self._evolve_batch(
            rabi_fn, detuning_fn, max_gate_time, batch_size
        )

        target_unitaries = jax.vmap(self.target_gate_fn)(angles.flatten())

        infidelities = _compute_batch_infidelity(
            final_unitaries, target_unitaries, self.nqubits
        )
        mean_infidelity = jnp.mean(infidelities)
        mean_gate_time = jnp.mean(gate_times)

        total_loss = mean_infidelity + self.time_weight * mean_gate_time

        metrics = {
            "loss": total_loss,
            "infidelity": mean_infidelity,
            "mean_gate_time": mean_gate_time,
        }
        return total_loss, metrics

    # ------------------------------------------------------------------
    # Batched evolution (JAX / diffrax)
    # ------------------------------------------------------------------

    def _evolve_batch(
        self,
        rabi_fn: Callable,
        detuning_fn: Callable,
        gate_time: float,
        batch_size: int,
    ) -> jax.Array:
        """Batched quantum evolution for multiple angles using diffrax.

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

        Returns
        -------
        jax.Array
            Final unitaries [batch, comp_dim, comp_dim]
        """
        full_dim = self._full_dim
        nqubits = self.nqubits
        rabi_ops = self._rabi_ops
        detuning_ops = self._detuning_ops
        interaction_op = self._interaction_op
        vdd = VDD

        # Initial state: identity for each batch element [batch, full_dim, full_dim]
        init_matrix = jnp.eye(full_dim, dtype=jnp.complex64)
        init_batch = jnp.tile(init_matrix[jnp.newaxis], (batch_size, 1, 1))

        # Time evaluation points
        t_eval = jnp.linspace(0.0, gate_time, self.controller.n_time_steps)

        # Batched Hamiltonian dynamics
        def hamiltonian_fn(t, y, args):
            """Compute dy/dt = -i H(t) y for all batch elements at time t.

            y: [batch, full_dim, full_dim]
            """
            # Batched pulse values [batch, 1]
            rabi_batch = rabi_fn(t)
            detuning_batch = detuning_fn(t)

            if rabi_batch.ndim == 1:
                rabi_batch = jnp.expand_dims(rabi_batch, -1)
            if detuning_batch.ndim == 1:
                detuning_batch = jnp.expand_dims(detuning_batch, -1)

            # Build batched Hamiltonian: [batch, full_dim, full_dim]
            H_batch = jnp.zeros(
                (batch_size, full_dim, full_dim), dtype=jnp.complex64
            )

            for i in range(nqubits):
                # Rabi term: (Ω(t)/2) * σ_x  ->  [batch, 1, 1] * [full_dim, full_dim]
                H_batch += (
                    0.5 * rabi_batch[:, jnp.newaxis, :]
                    * rabi_ops[i][jnp.newaxis, :, :]
                )

                # Detuning term: Δ(t) * n_r
                H_batch += (
                    detuning_batch[:, jnp.newaxis, :]
                    * detuning_ops[i][jnp.newaxis, :, :]
                )

            # Interaction term (same for all batch elements)
            if interaction_op is not None:
                H_batch += vdd * interaction_op[jnp.newaxis, :, :]

            # dy/dt = -i H y
            return -1j * jnp.einsum("bij,bjk->bik", H_batch, y)

        # Solve ODE with diffrax
        term = ODETerm(hamiltonian_fn)
        solver = Dopri5()
        saveat = SaveAt(ts=t_eval)

        solution = diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=gate_time,
            dt0=0.01,
            y0=init_batch,
            saveat=saveat,
            stepsize_controller=PIDController(rtol=1e-6, atol=1e-6),
            max_steps=10_000,
        )

        final_unitaries_full = solution.ys  # [n_steps, batch, full_dim, full_dim]
        # Take last time step
        final_unitaries_full = final_unitaries_full[-1]  # [batch, full_dim, full_dim]

        # Reduce to computational subspace
        final_unitaries = _reduce_to_computational(
            final_unitaries_full, batch_size, nqubits
        )

        # Apply phase corrections
        final_unitaries = _apply_batch_corrections(final_unitaries, nqubits)

        return final_unitaries


def _tree_to_numpy(tree):
    """Convert a PyTree of JAX arrays to numpy arrays keyed by path."""
    result = {}
    leaves = jax.tree_util.tree_leaves_with_path(tree)
    for path, leaf in leaves:
        key_str = jax.tree_util.keystr(path)
        import re
        m = re.search(r'layers\[(\d+)\]\.(\w+)', key_str)
        if m:
            idx = int(m.group(1))
            param = m.group(2)
            result[(idx, param)] = np.array(leaf)
    return result


def _get_dense_width(seq):
    """Get output width of first Dense layer in Sequential."""
    for layer in seq.layers:
        if hasattr(layer, 'weight'):
            return layer.weight.shape[0]
    return 0
