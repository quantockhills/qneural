"""
JAX neural network models for quantum control.

Uses equinox for PyTorch-like neural network definitions.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional


class FeedForwardNN(eqx.Module):
    """Configurable feedforward neural network using equinox."""

    layers: list

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int = 2,
        hidden_units: int = 64,
        activation: str = "relu",
        output_activation: str = "sigmoid",
        use_batch_norm: bool = False,
        weight_scale: float = 1.0,
        *,
        key=None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)

        layers = []
        keys = jax.random.split(key, hidden_layers + 2)

        k1, keys = keys[0], keys[1:]
        layers.append(_Dense(input_dim, hidden_units, key=k1))
        if use_batch_norm:
            layers.append(_eqx_batchnorm(hidden_units))
        layers.append(_get_activation(activation))

        for i in range(hidden_layers - 1):
            k, keys = keys[i], keys[1:]
            layers.append(_Dense(hidden_units, hidden_units, key=k))
            if use_batch_norm:
                layers.append(_eqx_batchnorm(hidden_units))
            layers.append(_get_activation(activation))

        k_out, keys = keys[0], keys[1:]
        layers.append(_Dense(hidden_units, output_dim, key=k_out))
        if output_activation != "none":
            layers.append(_get_activation(output_activation))

        if weight_scale != 1.0:
            def _scale(leaf):
                if isinstance(leaf, jax.Array) and leaf.dtype.kind == 'f':
                    return leaf * weight_scale
                return leaf
            layers = jax.tree_util.tree_map(_scale, layers, is_leaf=lambda x: isinstance(x, jax.Array))

        self.layers = layers

    @eqx.filter_jit
    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            if isinstance(layer, _eqx_batchnorm):
                x = jax.vmap(layer, axis_name="batch")(x)
            else:
                x = layer(x)
        return x

    def count_parameters(self) -> int:
        return sum(
            jnp.size(leaf)
            for leaf in jax.tree_util.tree_leaves(
                eqx.filter(self, eqx.is_array)
            )
        )


class PulseGenerator(eqx.Module):
    network: FeedForwardNN
    n_controls: int
    n_time_steps: int

    def __init__(
        self,
        n_controls: int = 2,
        n_time_steps: int = 201,
        hidden_layers: int = 6,
        hidden_units: int = 150,
        *,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)
        self.n_controls = n_controls
        self.n_time_steps = n_time_steps
        self.network = FeedForwardNN(
            input_dim=2,
            output_dim=n_controls,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            activation="relu",
            output_activation="sigmoid",
            key=key,
        )

    def __call__(
        self, angle: jax.Array, normalized_time: jax.Array
    ) -> jax.Array:
        if angle.ndim == 0:
            angle = jnp.expand_dims(angle, 0)

        batch_size = angle.shape[0]
        n_times = normalized_time.shape[0]

        angle_repeated = jnp.repeat(angle, n_times, axis=0)
        time_repeated = jnp.tile(normalized_time, batch_size)

        inputs = jnp.stack([angle_repeated, time_repeated], axis=1)
        outputs = self.network(inputs)
        outputs = outputs.reshape(batch_size, n_times, self.n_controls)
        return outputs


def _get_activation(name: str):
    activations = {
        "relu": eqx.nn.Lambda(jax.nn.relu),
        "tanh": eqx.nn.Lambda(jnp.tanh),
        "sigmoid": eqx.nn.Lambda(jax.nn.sigmoid),
        "none": eqx.nn.Lambda(lambda x: x),
    }
    if name not in activations:
        raise ValueError(
            f"Unknown activation: {name}. Choose from {list(activations.keys())}"
        )
    return activations[name]


class _Dense(eqx.Module):
    """Torch-style dense layer: accepts batched inputs (batch, in_features)."""
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_features: int, out_features: int, *, key):
        wkey, bkey = jax.random.split(key)
        lim = 1.0 / jnp.sqrt(in_features) if in_features > 0 else 1.0
        self.weight = jax.random.uniform(wkey, (out_features, in_features),
                                         minval=-lim, maxval=lim)
        self.bias = jax.random.uniform(bkey, (out_features,),
                                       minval=-lim, maxval=lim)

    def __call__(self, x: jax.Array, *, key=None) -> jax.Array:
        return x @ self.weight.T + self.bias


class _eqx_batchnorm(eqx.Module):
    """Wrapper for equinox BatchNorm with axis_name."""
    bn: eqx.nn.BatchNorm

    def __init__(self, size: int):
        self.bn = eqx.nn.BatchNorm(size, axis_name="batch")

    def __call__(self, x: jax.Array, *, key=None) -> jax.Array:
        return self.bn(x)
