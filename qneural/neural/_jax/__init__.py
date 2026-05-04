"""JAX neural backend for qneural."""

import pickle

import jax
import jax.numpy as jnp
import equinox as eqx

from .models import FeedForwardNN, PulseGenerator, _Dense
from .losses import (
    QuantumLoss,
    InfidelityLoss,
    TimePenaltyLoss,
    RobustnessLoss,
    ResourceLoss,
    CompositeLoss,
    create_infidelity_loss,
    create_time_optimal_loss,
)
from .solvers import JaxOdeSolver
from .trainer import (
    JaxQuantumTrainer,
    JaxFixedRabiTrainer,
    create_trainer,
)
from .time_optimal import TimeOptimalController, TimeOptimalTrainer


def load_jax_model(path: str):
    """Load a .jax checkpoint file and return (controller, checkpoint).

    The checkpoint file should have been created by scripts/convert_to_jax.py
    or be in the same format: a pickled dict with 'controller_config',
    'metadata', 'time_weights', 'control_weights'.
    """
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)

    config = checkpoint['controller_config']
    metadata = checkpoint['metadata']
    time_weights = checkpoint['time_weights']
    control_weights = checkpoint['control_weights']

    # Create controller with matching architecture
    key = jax.random.PRNGKey(0)
    controller = TimeOptimalController(
        time_bounds=config['time_bounds'],
        rabi_max=config['rabi_max'],
        detuning_range=config['detuning_range'],
        n_time_steps=config['n_time_steps'],
        time_hidden_layers=config['time_hidden_layers'],
        time_hidden_units=config['time_hidden_units'],
        control_hidden_layers=config['control_hidden_layers'],
        control_hidden_units=config['control_hidden_units'],
        time_output_activation=config['time_output_activation'],
        weight_scale_time=config.get('weight_scale_time', 1.8),
        weight_scale_control=config.get('weight_scale_control', 1.55),
        key=key,
    )

    # Map weights from state dict to equinox model
    controller = _load_weights(controller, 'time_predictor', time_weights)
    controller = _load_weights(controller, 'control_generator', control_weights)

    return controller, checkpoint


def _load_weights(model, subnet_name, state_dict):
    """Load numpy weights into an equinox model subtree.

    PyTorch state dict keys like "0.weight", "2.bias" map to equinox
    layers[0].weight, layers[2].bias (indices match since activations
    are interspersed).
    """
    weight_map = {}
    for key, arr in state_dict.items():
        parts = key.split('.')
        idx = int(parts[0])
        param = parts[1]
        weight_map[(idx, param)] = arr

    # Apply each weight using tree_at with stable closure via default args
    for (idx, param), arr in weight_map.items():
        jax_arr = jnp.array(arr)

        if param == 'weight':
            model = eqx.tree_at(
                lambda m, _i=idx, _sn=subnet_name: getattr(m, _sn).layers[_i].weight,
                model,
                jax_arr,
            )
        elif param == 'bias':
            model = eqx.tree_at(
                lambda m, _i=idx, _sn=subnet_name: getattr(m, _sn).layers[_i].bias,
                model,
                jax_arr,
            )

    return model


def _get_leaf(tree, path):
    """Extract a leaf from a PyTree at the given path."""
    for key in path:
        if hasattr(key, 'idx'):
            tree = tree[key.idx]
        else:
            tree = getattr(tree, key.name)
    return tree


def _leaf_attr(leaf, attr_name):
    return getattr(leaf, attr_name)

