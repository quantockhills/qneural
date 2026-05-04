"""
JAX training infrastructure for quantum control optimization.
Uses optax for optimization with functional training loops.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Callable, Optional, List, Dict, Tuple
import equinox as eqx

from ..pulse_generator import PhysicalPulseGenerator
from ..evolution import QuantumEvolver
from ...hardware.rydberg.hamiltonian import RydbergHamiltonian
from ...core.gates import czphi_gate


class JaxQuantumTrainer:
    def __init__(
        self,
        network,
        nqubits: int,
        loss_fn,
        optimizer=None,
        device: str = "cpu",
    ):
        self.network = network
        self.nqubits = nqubits
        self.loss_fn = loss_fn
        self.history: Dict[str, List] = {
            "epoch": [], "loss": [], "infidelity": [], "mean_time": []
        }
        self.current_epoch = 0

        if optimizer is None:
            optimizer = optax.adam(1e-4)
        self.optimizer = optimizer
        self.opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

    def train_step(self, angles, gate_time, target_gates):
        def loss_fn(params, static, x):
            model = eqx.combine(params, static)
            loss = self._compute_loss(model, angles, x[0], x[1])
            infid = self._compute_infidelity(model, angles, x[0], x[1])
            return loss, infid

        params, static = eqx.partition(self.network, eqx.is_array)
        value_and_grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)

        (loss, infid), grads = value_and_grad(params, static, (gate_time, target_gates))

        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, params
        )
        new_params = optax.apply_updates(params, updates)

        self.network = eqx.combine(new_params, static)
        self.current_epoch += 1

        return loss, infid

    def train(self, angles, gate_time, epochs=1000):
        history = {"loss": [], "infidelity": [], "epoch": []}

        for epoch in range(epochs):
            angle_batch = jnp.atleast_1d(angles)
            target_gates = jnp.stack([
                czphi_gate(a) for a in angle_batch
            ])

            loss, infid = self.train_step(angles, gate_time, target_gates)

            history["loss"].append(float(loss))
            history["infidelity"].append(float(infid))
            history["epoch"].append(epoch)

        self.history = history
        return history

    def _compute_loss(self, model, angles, gate_time, target_gates):
        return self.loss_fn(model, target_gates, gate_time=gate_time)

    def _compute_infidelity(self, model, angles, gate_time, target_gates):
        return self.loss_fn(model, target_gates, gate_time=gate_time)


class JaxFixedRabiTrainer(JaxQuantumTrainer):
    pass


def create_trainer(
    network,
    nqubits: int,
    loss_fn,
    optimizer=None,
    device: str = "cpu",
):
    return JaxQuantumTrainer(
        network=network,
        nqubits=nqubits,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )
