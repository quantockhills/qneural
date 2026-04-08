"""
Convert archival models to new qneural format.

This module provides functions to convert old archival model files
to the new qneural checkpoint format.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import sys
from pathlib import Path
import numpy as np


def detect_architecture(old_network) -> Dict:
    """
    Auto-detect architecture parameters from old model networks.

    Parameters
    ----------
    old_network : neural_trainer_time_optimal_cz
        The old model network object

    Returns
    -------
    dict
        Dictionary with architecture parameters:
        - time_hidden_layers: int
        - time_hidden_units: int
        - time_output_activation: str ('sigmoid' or 'tanh')
        - control_hidden_layers: int
        - control_hidden_units: int
    """
    # Analyze time network (ansatz_time)
    time_layers = list(old_network.ansatz_time.children())
    time_linear_layers = [l for l in time_layers if isinstance(l, nn.Linear)]

    time_hidden_layers = (
        len(time_linear_layers) - 1
    )  # Total linear layers minus 1 for proper constructor arg
    time_hidden_units = time_linear_layers[0].out_features

    # Detect output activation
    final_activation = time_layers[-1]
    if isinstance(final_activation, nn.Tanh):
        time_output_activation = "tanh"
    elif isinstance(final_activation, nn.Sigmoid):
        time_output_activation = "sigmoid"
    else:
        raise ValueError(f"Unknown activation: {type(final_activation)}")

    # Analyze control network (ansatz_control)
    control_layers = list(old_network.ansatz_control.children())
    control_linear_layers = [l for l in control_layers if isinstance(l, nn.Linear)]

    control_hidden_layers = (
        len(control_linear_layers) - 1
    )  # Total linear layers minus 1 for proper constructor arg
    control_hidden_units = control_linear_layers[0].out_features

    return {
        "time_hidden_layers": time_hidden_layers,
        "time_hidden_units": time_hidden_units,
        "time_output_activation": time_output_activation,
        "control_hidden_layers": control_hidden_layers,
        "control_hidden_units": control_hidden_units,
    }


def transfer_weights(old_network, new_controller):
    """
    Transfer weights from old network to new controller.

    Maps weights layer-by-layer from old Sequential networks
    to new controller networks.

    Parameters
    ----------
    old_network : neural_trainer_time_optimal_cz
        Old model with ansatz_time and ansatz_control
    new_controller : TimeOptimalController
        New controller to receive weights
    """
    # Transfer time network weights
    old_time_state = old_network.ansatz_time.state_dict()
    new_time_state = {}

    # Map old indices to new indices
    # Old: 0 (Linear), 1 (ReLU), 2 (Linear), 3 (ReLU), 4 (Linear), 5 (Tanh)
    # New: 0 (Linear), 1 (ReLU), 2 (Linear), 3 (ReLU), 4 (Linear), 5 (Tanh/Sigmoid)
    old_time_keys = list(old_time_state.keys())
    for key in old_time_keys:
        new_time_state[key] = old_time_state[key]

    new_controller.time_predictor.load_state_dict(new_time_state)

    # Transfer control network weights (straightforward mapping)
    old_control_state = old_network.ansatz_control.state_dict()
    new_controller.control_generator.load_state_dict(old_control_state)


def convert_archival_to_new_format(
    old_model_path: str, output_path: str, metadata: Optional[Dict] = None
) -> Dict:
    """
    Convert old archival model to new qneural format.

    Parameters
    ----------
    old_model_path : str
        Path to old model file (no extension)
    output_path : str
        Path for new .pt file
    metadata : dict, optional
        Additional metadata to include in checkpoint

    Returns
    -------
    dict
        Conversion info including:
        - success: bool
        - architecture_detected: dict
        - angle_range: [min, max]
        - time_bounds: [min, max]
    """
    # Add archival path to sys.path temporarily for loading
    archival_path = str(Path(old_model_path).parent.parent.parent)
    if archival_path not in sys.path:
        sys.path.insert(0, archival_path)

    try:
        # Load old model first to detect type
        old_data = torch.load(old_model_path, map_location="cpu", weights_only=False)
        old_network = old_data["network"]

        # Detect if 2-qubit or 3-qubit based on module name
        network_module = type(old_network).__module__
        if "cczphi" in network_module or "ccphase" in network_module:
            nqubits = 3
        else:
            nqubits = 2

        # Detect architecture
        arch = detect_architecture(old_network)

        # Extract configuration from old network
        time_bounds = old_network.time_bounds
        rabi_max = 25.13  # Standard value from cfn.rabi
        detuning_range = tuple(old_network.range_detuning)
        n_time_steps = len(old_network._aux_tensor)

        # Get angle range from old data - check both tensor and network attribute
        angles = old_data["angle"]
        angle_range_tensor = [angles.min().item(), angles.max().item()]

        # Check for angle_range attribute on network (this is the authoritative range)
        # The tensor might have gotten corrupted or not saved properly
        angle_range = angle_range_tensor  # default to tensor
        angle_range_source = "tensor"
        if hasattr(old_network, "angle_range"):
            attr_range = old_network.angle_range
            if isinstance(attr_range, (list, tuple)) and len(attr_range) == 2:
                angle_range = [float(attr_range[0]), float(attr_range[1])]
                angle_range_source = "network_attribute"

        # Import new controller
        from ..neural.time_optimal import TimeOptimalController

        # Create new controller with matching architecture
        new_controller = TimeOptimalController(
            time_bounds=time_bounds,
            rabi_max=rabi_max,
            detuning_range=detuning_range,
            n_time_steps=n_time_steps,
            time_hidden_layers=arch["time_hidden_layers"],
            time_hidden_units=arch["time_hidden_units"],
            control_hidden_layers=arch["control_hidden_layers"],
            control_hidden_units=arch["control_hidden_units"],
            time_output_activation=arch["time_output_activation"],
            weight_scale_time=1.8,  # Default from archival
            weight_scale_control=1.55,  # Default from archival
        )

        # Transfer weights
        transfer_weights(old_network, new_controller)

        # Build checkpoint
        checkpoint = {
            "time_network_state_dict": new_controller.time_predictor.state_dict(),
            "control_network_state_dict": new_controller.control_generator.state_dict(),
            "time_optimizer_state_dict": {},  # Empty - old format didn't save optimizer state dict
            "control_optimizer_state_dict": {},  # Empty
            "history": {
                "epoch": [],
                "loss": [],
                "infidelity": [],
                "mean_gate_time": [],
            },  # Empty
            "epoch": 0,
            "time_weight": 0.005,  # Default from archival
            "controller_config": {
                "time_bounds": time_bounds,
                "rabi_max": rabi_max,
                "detuning_range": detuning_range,
                "n_time_steps": n_time_steps,
                "time_hidden_layers": arch["time_hidden_layers"],
                "time_hidden_units": arch["time_hidden_units"],
                "control_hidden_layers": arch["control_hidden_layers"],
                "control_hidden_units": arch["control_hidden_units"],
                "time_output_activation": arch["time_output_activation"],
                "weight_scale_time": 1.8,
                "weight_scale_control": 1.55,
            },
            "metadata": (metadata or {})
            | {
                "source": "archival_publication",
                "original_file": Path(old_model_path).name,
                "angle_range": angle_range,
                "angle_range_tensor": angle_range_tensor,
                "angle_range_source": angle_range_source,
                "nqubits": nqubits,
                "target_gate_type": "cczphi_gate" if nqubits == 3 else "czphi_gate",
                "note": (metadata or {}).get(
                    "note",
                    f"Publication-quality results for {nqubits}-qubit phase gate converted from archival format",
                ),
                "missing_data": ["training_history", "epoch_count", "optimizer_states"],
            },
        }

        # Save
        torch.save(checkpoint, output_path)

        return {
            "success": True,
            "architecture_detected": arch,
            "angle_range": angle_range,
            "angle_range_source": angle_range_source,
            "time_bounds": time_bounds,
            "nqubits": nqubits,
            "output_path": output_path,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "architecture_detected": None,
            "angle_range": None,
            "time_bounds": None,
        }
    finally:
        # Clean up sys.path
        if archival_path in sys.path:
            sys.path.remove(archival_path)


def load_saved_model(
    model_path: str,
    print_metadata: bool = True,
    evaluate_fidelity: bool = True,
    n_eval_angles: int = 50,
    device: str = "cpu",
) -> Tuple[object, Dict]:
    """
    Load a converted publication model and create controller.

    This convenience function loads a .pt checkpoint file and creates
    a fully configured TimeOptimalController with trained weights.

    Parameters
    ----------
    model_path : str
        Path to the .pt checkpoint file
    print_metadata : bool, default True
        Print model metadata and configuration
    evaluate_fidelity : bool, default True
        Evaluate and print fidelity statistics
    n_eval_angles : int, default 50
        Number of angles to evaluate for fidelity statistics
    device : str, default 'cpu'
        Device to load model on ('cpu' or 'cuda')

    Returns
    -------
    tuple
        - controller: TimeOptimalController with loaded weights
        - checkpoint: Dict containing full checkpoint data

    Examples
    --------
    >>> controller, checkpoint = load_publication_model(
    ...     'qneural/data/publication_models/pt5pi_to_pi.pt'
    ... )
    >>> # Controller is ready to use
    >>> gate_time, detuning = controller(torch.tensor([[np.pi/2]]))
    """
    from ..neural.time_optimal import TimeOptimalController, TimeOptimalTrainer

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["controller_config"]
    metadata = checkpoint["metadata"]

    # Create controller with exact configuration
    controller = TimeOptimalController(
        time_bounds=config["time_bounds"],
        rabi_max=config["rabi_max"],
        detuning_range=config["detuning_range"],
        n_time_steps=config["n_time_steps"],
        time_hidden_layers=config["time_hidden_layers"],
        time_hidden_units=config["time_hidden_units"],
        control_hidden_layers=config["control_hidden_layers"],
        control_hidden_units=config["control_hidden_units"],
        time_output_activation=config["time_output_activation"],
        weight_scale_time=config.get("weight_scale_time", 1.8),
        weight_scale_control=config.get("weight_scale_control", 1.55),
    )

    # Load trained weights
    controller.time_predictor.load_state_dict(checkpoint["time_network_state_dict"])
    controller.control_generator.load_state_dict(
        checkpoint["control_network_state_dict"]
    )
    controller = controller.to(device)

    # Print metadata if requested
    if print_metadata:
        print("Model Metadata:")
        print("=" * 50)
        for key, value in metadata.items():
            if key == "note":
                print(f"  {key}: {value[:50]}...")
            elif (
                key == "angle_range"
                and isinstance(value, (list, tuple))
                and len(value) == 2
            ):
                print(f"  {key}: [{value[0] / np.pi:.4f}π, {value[1] / np.pi:.4f}π]")
            elif (
                key == "angle_range_tensor"
                and isinstance(value, (list, tuple))
                and len(value) == 2
            ):
                print(f"  {key}: [{value[0] / np.pi:.4f}π, {value[1] / np.pi:.4f}π]")
            else:
                print(f"  {key}: {value}")

        print("\nController Configuration:")
        print("=" * 50)
        nqubits = metadata.get("nqubits", 2)
        print(f"  Qubits: {nqubits}")
        print(
            f"  Time network: {config['time_hidden_layers']} layers x {config['time_hidden_units']} units ({config['time_output_activation']})"
        )
        print(
            f"  Control network: {config['control_hidden_layers']} layers x {config['control_hidden_units']} units"
        )
        print(
            f"  Time bounds: [{config['time_bounds'][0]:.4f}, {config['time_bounds'][1]:.4f}] s"
        )
        print(f"  Rabi max: {config['rabi_max']:.2f} MHz")
        print(f"  Time steps: {config['n_time_steps']}")
        print(
            f"  Total parameters: {sum(p.numel() for p in controller.parameters()):,}"
        )

    # Evaluate fidelity if requested
    if evaluate_fidelity and "angle_range" in metadata:
        angle_range = metadata["angle_range"]
        eval_angles = torch.linspace(
            angle_range[0], angle_range[1], n_eval_angles, device=device
        )

        # Get nqubits from metadata (default to 2 for backward compatibility)
        nqubits = metadata.get("nqubits", 2)

        # Determine target gate function based on nqubits and model type
        # Archival models used standard gate definitions (phase on |11...1>)
        target_gate_fn = (
            None  # Use default (czphi_gate for 2-qubit, cczphi_gate for 3-qubit)
        )

        # Create trainer for evaluation
        trainer = TimeOptimalTrainer(
            controller=controller,
            nqubits=nqubits,
            time_weight=config.get("time_weight", 0.005),
            target_gate_fn=target_gate_fn,
        )

        print(f"\nEvaluating fidelity on {n_eval_angles} angles...")
        results = trainer.evaluate(eval_angles)

        # Calculate statistics
        fidelities = [(1 - inf) * 100 for inf in results["infidelities"]]

        print("\nFidelity Statistics:")
        print("=" * 50)
        print(f"  Mean: {np.mean(fidelities):.4f}%")
        print(f"  Min: {np.min(fidelities):.4f}%")
        print(f"  Max: {np.max(fidelities):.4f}%")
        print(f"  Std: {np.std(fidelities):.4f}%")
        print(f"  All > 99%: {all(f > 99 for f in fidelities)}")

    return controller, checkpoint
