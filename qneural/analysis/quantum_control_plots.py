"""
Unified visualization utilities for quantum gate optimization.

Provides comprehensive plotting functions for both fixed-time (CZ) and
time-optimal (CPHASE) gate optimization, including:
- Training progress visualization
- Pulse visualization (detuning, Rabi)
- Gate time analysis (for time-optimal)
- Fidelity evaluation across angles
- Multi-angle comparisons

Designed to be flexible and work with both FixedRabiTrainer and TimeOptimalTrainer.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path


# Configure matplotlib for publication-quality plots (Physical Review style)
def _configure_publication_style():
    """Configure matplotlib for LaTeX-style, publication-ready plots using mathtext."""
    mpl.rcParams.update(
        {
            # Use mathtext (matplotlib's built-in TeX-like rendering, no LaTeX installation needed)
            "text.usetex": False,
            "mathtext.fontset": "cm",  # Computer Modern font (LaTeX default)
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            # Line widths and sizes
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "patch.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            # Figure settings
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.fancybox": False,
            "legend.edgecolor": "black",
        }
    )


# Apply publication style on module import
_configure_publication_style()


def plot_training_progress(
    history: Dict,
    rabi_max: Optional[float] = None,
    time_bounds: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (18, 4),
) -> plt.Figure:
    """
    Plot comprehensive training progress for gate optimization.

    Automatically detects whether this is fixed-time (CZ) or time-optimal (CPHASE)
    training based on available metrics in history.

    Parameters
    ----------
    history : dict
        Training history with keys: 'epoch', 'loss', 'infidelity',
        and optionally 'mean_gate_time' for time-optimal training
    rabi_max : float, optional
        Maximum Rabi frequency for converting time to normalized units.
        Required if plotting gate time.
    time_bounds : tuple of (float, float), optional
        (t_min, t_max) in normalized Rabi units for time-optimal plots
    save_path : str, optional
        Path to save figure
    show : bool, default=True
        Whether to display the plot
    figsize : tuple, default=(18, 4)
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object

    Examples
    --------
    >>> # Fixed-time CZ gate training
    >>> history = trainer.train(angles, gate_time=7.0, epochs=1000)
    >>> plot_training_progress(history)

    >>> # Time-optimal CPHASE training
    >>> history = trainer.train(angles, epochs=1000)
    >>> plot_training_progress(history, rabi_max=25.13, time_bounds=(3.0, 8.5))
    """
    # Detect training type
    is_time_optimal = "mean_gate_time" in history

    # Determine number of subplots
    n_plots = 3 if is_time_optimal else 2

    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    epochs = np.array(history["epoch"])

    # Plot 1: Loss
    axes[0].plot(epochs, history["loss"], linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.2, linestyle=":")
    axes[0].set_yscale("log")

    # Plot 2: Infidelity
    axes[1].plot(epochs, history["infidelity"], linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Infidelity")
    axes[1].set_title("Gate Infidelity")
    axes[1].grid(True, alpha=0.2, linestyle=":")
    axes[1].set_yscale("log")

    # Plot 3: Gate Time (only for time-optimal)
    if is_time_optimal:
        # Convert to normalized Rabi units if rabi_max provided
        if rabi_max is not None:
            gate_times = np.array(history["mean_gate_time"]) * rabi_max
            ylabel = r"Mean Gate Time ($\Omega_{\mathrm{max}} T$)"
        else:
            gate_times = np.array(history["mean_gate_time"])
            ylabel = "Mean Gate Time (s)"

        axes[2].plot(epochs, gate_times, linewidth=1.5)

        # Add time bounds if provided
        if time_bounds is not None and rabi_max is not None:
            axes[2].axhline(
                y=time_bounds[0],
                color="red",
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                label="Min bound",
            )
            axes[2].axhline(
                y=time_bounds[1],
                color="red",
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                label="Max bound",
            )
            axes[2].legend()

        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel(ylabel)
        axes[2].set_title("Optimized Gate Time")
        axes[2].grid(True, alpha=0.2, linestyle=":")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"✓ Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_detuning_pulses(
    controller,
    angles: Union[torch.Tensor, List[float]],
    n_time_steps: int = 201,
    gate_time: Optional[float] = None,
    rabi_max: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (18, 4),
    single_plot: bool = False,
) -> plt.Figure:
    """
    Plot detuning pulses for multiple angles.

    Works with both FixedRabiTrainer (fixed gate time) and TimeOptimalController
    (variable gate times). Automatically detects controller type and adapts.

    Parameters
    ----------
    controller : TimeOptimalController, FixedRabiTrainer, or network
        The trained controller or trainer
    angles : torch.Tensor or list of float
        Angles to visualize (typically 3-5 angles)
    n_time_steps : int, default=201
        Number of time steps for visualization
    gate_time : float, optional
        Fixed gate time in seconds (only for FixedRabiTrainer).
        Ignored for TimeOptimalController. If not provided for FixedRabi,
        plots use normalized time [0, 1].
    rabi_max : float, optional
        Maximum Rabi frequency in MHz (for unit conversion)
    save_path : str, optional
        Path to save figure
    show : bool, default=True
        Whether to display the plot
    figsize : tuple, default=(18, 4)
        Figure size (for single_plot=True, use smaller like (8, 6))
    single_plot : bool, default=False
        If True, plot all angles on same axes with legend (Physical Review style).
        If False, plot each angle in separate subplot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object

    Examples
    --------
    >>> # Time-optimal CPHASE (gate_time parameter ignored)
    >>> angles = torch.tensor([[0.4*np.pi], [0.5*np.pi], [0.6*np.pi]])
    >>> plot_detuning_pulses(controller, angles, rabi_max=25.13)

    >>> # Fixed-time CZ with explicit gate time
    >>> angle = torch.tensor([[np.pi]])
    >>> plot_detuning_pulses(trainer, [angle], gate_time=0.304,
    ...                      rabi_max=25.13, n_time_steps=301)

    >>> # Publication-style single plot with multiple angles
    >>> angles = [np.pi/8, np.pi/4, np.pi/2, np.pi]
    >>> plot_detuning_pulses(controller, angles, single_plot=True,
    ...                      figsize=(8, 6), rabi_max=25.13)
    """
    # Convert angles to tensor if needed
    if not isinstance(angles, torch.Tensor):
        angles = torch.tensor(angles)

    if angles.dim() == 1:
        angles = angles.unsqueeze(-1)

    n_angles = len(angles)

    # Create figure
    if single_plot:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax] * n_angles  # All angles share the same axes
        # Define colors for different angles (matching Physical Review style)
        colors = plt.cm.tab10(np.linspace(0, 0.9, n_angles))
    else:
        fig, axes = plt.subplots(1, n_angles, figsize=figsize)
        if n_angles == 1:
            axes = [axes]
        colors = [None] * n_angles  # matplotlib default colors

    # Detect controller type
    is_time_optimal = hasattr(controller, "time_predictor")

    with torch.no_grad():
        if hasattr(controller, "eval"):
            controller.eval()

        for i, (angle, ax, color) in enumerate(zip(angles, axes, colors)):
            angle_reshaped = angle.reshape(1, 1)

            if is_time_optimal:
                # TimeOptimalController
                gate_time, detuning_normalized = controller(angle_reshaped)
                detuning = controller.scale_detuning(detuning_normalized)
                times = np.linspace(0, gate_time.item(), n_time_steps)

                # Subplot title with gate time
                if rabi_max is not None:
                    time_label = f"$T = {gate_time.item() * rabi_max:.2f}$ ($\\Omega_{{\\mathrm{{max}}}} T$)"
                else:
                    time_label = f"$T = {gate_time.item():.4f}$ s"
            else:
                # Fixed-time case: could be network, pulse_generator, or trainer
                # Try to detect what we have and extract detuning appropriately
                if hasattr(controller, "network"):
                    # It's a FixedRabiTrainer - use its network and pulse_generator
                    network = controller.network
                    pulse_gen = controller.pulse_generator
                    n_time_steps = pulse_gen.n_time_steps

                    # Generate network inputs
                    time_grid = torch.linspace(0, 1, n_time_steps)
                    angles_rep = angle_reshaped.repeat_interleave(n_time_steps)
                    time_rep = time_grid.repeat(len(angle_reshaped))
                    inputs = torch.stack([angles_rep, time_rep], dim=1)

                    # Get detuning from network
                    detuning_out = network(inputs).reshape(n_time_steps)
                    detuning = pulse_gen.scale_output(detuning_out, 0)

                    # Use gate_time parameter if provided, else normalized time
                    if gate_time is not None:
                        times = np.linspace(0, gate_time, n_time_steps)
                        if rabi_max is not None:
                            time_label = f"$T = {gate_time * rabi_max:.2f}$ ($\\Omega_{{\\mathrm{{max}}}} T$)"
                        else:
                            time_label = f"$T = {gate_time:.4f}$ s"
                    else:
                        times = np.linspace(0, 1.0, n_time_steps)
                        time_label = ""

                elif hasattr(controller, "forward"):
                    # It's a network directly
                    output = controller(angle_reshaped)
                    detuning = output[:, :, 0] if output.dim() > 2 else output
                    times = np.linspace(0, 1.0, n_time_steps)
                    time_label = ""
                else:
                    raise ValueError(
                        "Controller must be TimeOptimalController, FixedRabiTrainer, or Network"
                    )

            # Plot
            detuning_np = detuning.squeeze().cpu().numpy()

            # Create label for legend (if single_plot mode)
            if single_plot:
                # Format angle as multiple of π (e.g., "0.50π", "0.75π")
                angle_mult = angle.item() / np.pi
                if abs(angle_mult - round(angle_mult)) < 0.01:
                    # It's close to a simple fraction like π/2, π/4, etc.
                    if round(angle_mult) == 0:
                        label = "$0$"
                    elif round(angle_mult) == 1:
                        label = "$\\pi$"
                    else:
                        label = f"${int(round(angle_mult))}\\pi$"
                else:
                    # Show as decimal multiple of π
                    label = f"${angle_mult:.2f}\\pi$"
                ax.plot(times, detuning_np, linewidth=1.5, color=color, label=label)
            else:
                ax.plot(times, detuning_np, linewidth=1.5)

            # Add horizontal line at y=0 (only once for single_plot)
            if not single_plot or i == 0:
                ax.axhline(y=0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)

            # Labels and formatting (only set once for single_plot)
            if not single_plot or i == 0:
                # Choose x-axis label based on whether gate_time was provided
                if is_time_optimal or gate_time is not None:
                    ax.set_xlabel(r"Gatetime $\Omega_{\mathrm{max}} t$")
                else:
                    ax.set_xlabel(r"Normalized Time")
                ax.set_ylabel(r"Detuning $\Delta/\Omega_{\mathrm{max}}$")
                ax.grid(True, alpha=0.2, linestyle=":")

            # Title (only for separate subplots)
            if not single_plot:
                title = f"$\\theta = {angle.item() / np.pi:.2g}\\pi$"
                if time_label:
                    title += f"\n{time_label}"
                ax.set_title(title)

    # Add legend for single_plot mode
    if single_plot:
        ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"✓ Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_gate_time_vs_angle(
    controller,
    angle_range: Tuple[float, float],
    n_angles: int = 100,
    rabi_max: Optional[float] = None,
    time_bounds: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot learned optimal gate time as a function of angle.

    Only applicable for time-optimal training (CPHASE).

    Parameters
    ----------
    controller : TimeOptimalController
        Trained time-optimal controller
    angle_range : tuple of (float, float)
        (min_angle, max_angle) to evaluate
    n_angles : int, default=100
        Number of angles to sample
    rabi_max : float, optional
        Maximum Rabi frequency for converting to normalized units
    time_bounds : tuple of (float, float), optional
        (t_min, t_max) in normalized units to show as bounds
    save_path : str, optional
        Path to save figure
    show : bool, default=True
        Whether to display the plot
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object

    Examples
    --------
    >>> plot_gate_time_vs_angle(
    ...     controller,
    ...     angle_range=(0.4*np.pi, 0.6*np.pi),
    ...     rabi_max=25.13,
    ...     time_bounds=(3.0, 8.5)
    ... )
    """
    # Generate test angles
    test_angles = torch.linspace(angle_range[0], angle_range[1], n_angles).reshape(
        n_angles, 1
    )

    with torch.no_grad():
        controller.eval()
        predicted_times, _ = controller(test_angles)

        # Convert to normalized units if rabi_max provided
        if rabi_max is not None:
            predicted_times_display = predicted_times.squeeze() * rabi_max
            ylabel = "Predicted Gate Time (Ω_max T)"
        else:
            predicted_times_display = predicted_times.squeeze()
            ylabel = "Predicted Gate Time (s)"

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        test_angles.numpy(),
        predicted_times_display.numpy(),
        s=10,
        c="blue",
        alpha=0.6,
        label="Learned time",
    )

    # Add bounds if provided
    if time_bounds is not None and rabi_max is not None:
        ax.axhline(
            y=time_bounds[0],
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            label="Time bounds",
        )
        ax.axhline(
            y=time_bounds[1], color="red", linestyle="--", linewidth=2, alpha=0.5
        )

    ax.set_xlabel("Angle (radians)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Learned Optimal Gate Time vs Angle", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add statistics
    stats_text = (
        f"Min: {predicted_times_display.min().item():.2f}\n"
        f"Max: {predicted_times_display.max().item():.2f}\n"
        f"Mean: {predicted_times_display.mean().item():.2f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_fidelity_vs_angle(
    trainer,
    angle_range: Tuple[float, float],
    n_angles: int = 50,
    target_fidelity: float = 99.0,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Evaluate and plot gate fidelity across a range of angles.

    Works with both FixedRabiTrainer and TimeOptimalTrainer.

    Parameters
    ----------
    trainer : FixedRabiTrainer or TimeOptimalTrainer
        Trained trainer with evaluate() method
    angle_range : tuple of (float, float)
        (min_angle, max_angle) to evaluate
    n_angles : int, default=50
        Number of angles to sample
    target_fidelity : float, default=99.0
        Target fidelity line to show (in percent)
    save_path : str, optional
        Path to save figure
    show : bool, default=True
        Whether to display the plot
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object

    Examples
    --------
    >>> plot_fidelity_vs_angle(
    ...     trainer,
    ...     angle_range=(0.4*np.pi, 0.6*np.pi),
    ...     target_fidelity=99.0
    ... )
    """
    # Generate evaluation angles
    eval_angles = torch.linspace(angle_range[0], angle_range[1], n_angles)

    print(f"Evaluating fidelity across {n_angles} angles...")
    eval_results = trainer.evaluate(eval_angles)

    # Convert infidelities to fidelities (percent)
    fidelities = [(1 - inf) * 100 for inf in eval_results["infidelities"]]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(eval_results["angles"], fidelities, s=30, c="blue", alpha=0.6)
    ax.axhline(
        y=target_fidelity,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"{target_fidelity}% target",
        alpha=0.7,
    )

    ax.set_xlabel("Angle (radians)", fontsize=12)
    ax.set_ylabel("Gate Fidelity (%)", fontsize=12)
    ax.set_title("Gate Fidelity vs Angle", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim([max(95, min(fidelities) - 1), 100.5])

    # Add statistics
    stats_text = (
        f"Mean: {np.mean(fidelities):.4f}%\n"
        f"Min: {np.min(fidelities):.4f}%\n"
        f"Max: {np.max(fidelities):.4f}%\n"
        f"Std: {np.std(fidelities):.4f}%"
    )
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Figure saved to {save_path}")

    if show:
        plt.show()

    print("\n✓ Fidelity evaluation complete")
    print(f"  Mean: {np.mean(fidelities):.4f}%")
    print(f"  Min: {np.min(fidelities):.4f}%")

    return fig


def create_optimization_summary(
    history: Dict,
    controller,
    trainer,
    angle_range: Tuple[float, float],
    rabi_max: Optional[float] = None,
    time_bounds: Optional[Tuple[float, float]] = None,
    n_sample_angles: int = 3,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> Dict[str, plt.Figure]:
    """
    Create a complete summary of gate optimization results.

    Generates multiple figures showing training progress, pulse visualization,
    and fidelity evaluation. Works with both fixed-time and time-optimal training.

    Parameters
    ----------
    history : dict
        Training history
    controller : TimeOptimalController or PhysicalPulseGenerator
        Trained controller
    trainer : FixedRabiTrainer or TimeOptimalTrainer
        Trained trainer
    angle_range : tuple of (float, float)
        Angle range used for training
    rabi_max : float, optional
        Maximum Rabi frequency
    time_bounds : tuple of (float, float), optional
        Time bounds for time-optimal training
    n_sample_angles : int, default=3
        Number of sample angles for pulse visualization
    save_dir : str, optional
        Directory to save all figures
    show : bool, default=True
        Whether to display plots

    Returns
    -------
    figures : dict
        Dictionary mapping figure names to Figure objects

    Examples
    --------
    >>> figures = create_optimization_summary(
    ...     history, controller, trainer,
    ...     angle_range=(0.4*np.pi, 0.6*np.pi),
    ...     rabi_max=25.13,
    ...     time_bounds=(3.0, 8.5),
    ...     save_dir='results/'
    ... )
    """
    figures = {}

    # Create save directory if needed
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 1. Training progress
    print("\n📊 Generating training progress plot...")
    save_path = f"{save_dir}/training_progress.png" if save_dir else None
    fig1 = plot_training_progress(history, rabi_max, time_bounds, save_path, show)
    figures["training_progress"] = fig1

    # 2. Detuning pulses
    print("📊 Generating detuning pulse plots...")
    sample_angles = torch.linspace(angle_range[0], angle_range[1], n_sample_angles)
    save_path = f"{save_dir}/detuning_pulses.png" if save_dir else None
    fig2 = plot_detuning_pulses(
        controller, sample_angles, rabi_max=rabi_max, save_path=save_path, show=show
    )
    figures["detuning_pulses"] = fig2

    # 3. Gate time vs angle (only for time-optimal)
    is_time_optimal = "mean_gate_time" in history
    if is_time_optimal:
        print("📊 Generating gate time vs angle plot...")
        save_path = f"{save_dir}/gate_time_vs_angle.png" if save_dir else None
        fig3 = plot_gate_time_vs_angle(
            controller,
            angle_range,
            rabi_max=rabi_max,
            time_bounds=time_bounds,
            save_path=save_path,
            show=show,
        )
        figures["gate_time_vs_angle"] = fig3

    # 4. Fidelity vs angle
    print("📊 Generating fidelity evaluation plot...")
    save_path = f"{save_dir}/fidelity_vs_angle.png" if save_dir else None
    fig4 = plot_fidelity_vs_angle(trainer, angle_range, save_path=save_path, show=show)
    figures["fidelity_vs_angle"] = fig4

    print(f"\n✓ Summary complete! Generated {len(figures)} figures")
    if save_dir:
        print(f"  Saved to: {save_dir}/")

    return figures
