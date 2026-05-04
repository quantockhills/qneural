#!/usr/bin/env python3
"""Convert PyTorch qneural checkpoints to JAX/equinox format.

Usage:
    python scripts/convert_to_jax.py input.pt output.jax

The output is a pickle-serialised dict with:
    - controller_config: dict (same as pytorch)
    - metadata: dict
    - time_weights: dict of numpy arrays (state dict for time predictor)
    - control_weights: dict of numpy arrays (state dict for control generator)
"""

import sys
import pickle
from pathlib import Path

import torch
import numpy as np
import jax.numpy as jnp
import equinox as eqx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _pt_state_to_numpy(state_dict):
    """Convert PyTorch state dict to dict of numpy arrays."""
    return {k: v.cpu().numpy() for k, v in state_dict.items()}


def convert_model(pt_path: str, jax_path: str):
    """Convert a PyTorch .pt checkpoint to JAX .jax checkpoint."""
    # Load PyTorch checkpoint
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
    config = checkpoint['controller_config']
    metadata = checkpoint.get('metadata', {})

    # Extract weights as numpy arrays
    time_weights = _pt_state_to_numpy(checkpoint['time_network_state_dict'])
    control_weights = _pt_state_to_numpy(checkpoint['control_network_state_dict'])

    # Save as .jax file (pickle with numpy arrays + config)
    output = {
        'controller_config': config,
        'metadata': metadata,
        'time_weights': time_weights,
        'control_weights': control_weights,
    }
    with open(jax_path, 'wb') as f:
        pickle.dump(output, f)

    print(f"Converted {pt_path} → {jax_path}")
    print(f"  Time predictor: {config['time_hidden_layers']} layers x {config['time_hidden_units']} units")
    print(f"  Control generator: {config['control_hidden_layers']} layers x {config['control_hidden_units']} units")
    print(f"  Angle range: {metadata.get('angle_range', 'unknown')}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python scripts/convert_to_jax.py input.pt output.jax")
        sys.exit(1)
    convert_model(sys.argv[1], sys.argv[2])
