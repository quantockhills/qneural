"""Pulse visualization utilities.

Functions for visualizing quantum control pulses, pulse sequences,
and pulse comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np
from ..backend import backend
from typing import Optional, List, Dict, Callable


def plot_pulses_vs_time(
    pulses: List[Callable],
    gate_time: float,
    n_points: int = 200,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Control Pulses",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot control pulses (Rabi frequency and detuning) vs time.

    Parameters
    ----------
    pulses : list of callable
        List of pulse functions. Typically [rabi_pulse, detuning_pulse]
    gate_time : float
        Total gate time (in units of 1/Ω_max)
    n_points : int, default=200
        Number of time points to evaluate
    labels : list of str, optional
        Labels for each pulse (e.g., ['Rabi Frequency', 'Detuning'])
    save_path : str, optional
        Path to save figure
    show : bool, default=True
        Whether to display the plot
    title : str, default="Control Pulses"
        Plot title
    figsize : tuple, default=(12, 5)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object

    Examples
    --------
    >>> from qneural.neural import create_default_physical_pulse_generator
    >>> pulse_gen = create_default_physical_pulse_generator()
    >>> pulses = pulse_gen.generate(nn_output, gate_time=7.0)
    >>> plot_pulses_vs_time(pulses, gate_time=7.0,
    ...                     labels=['Rabi Frequency', 'Detuning'])
    """
    # Generate time points
    times = np.linspace(0, gate_time, n_points)

    # Evaluate pulses at each time point
    pulse_values = []
    for pulse_fn in pulses:
        values = [
            pulse_fn(t).item() if backend.is_tensor(pulse_fn(t)) else pulse_fn(t)
            for t in times
        ]
        pulse_values.append(np.array(values))

    # Create subplots
    n_pulses = len(pulses)
    fig, axes = plt.subplots(1, n_pulses, figsize=figsize, sharex=True)
    if n_pulses == 1:
        axes = [axes]

    # Default labels
    if labels is None:
        labels = [f"Pulse {i + 1}" for i in range(n_pulses)]

    # Plot each pulse
    colors = ["blue", "red", "green", "orange"]
    for idx, (ax, values, label) in enumerate(zip(axes, pulse_values, labels)):
        ax.plot(
            times, values, linewidth=2, color=colors[idx % len(colors)], label=label
        )
        ax.fill_between(times, 0, values, alpha=0.2, color=colors[idx % len(colors)])
        ax.set_xlabel("Time (1/Ω_max)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add statistics
        ax.text(
            0.02,
            0.98,
            f"Max: {np.max(values):.2f}\nMin: {np.min(values):.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_pulse_comparison(
    pulse_dict: Dict[str, List[Callable]],
    gate_time: float,
    pulse_idx: int = 0,
    n_points: int = 200,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Pulse Comparison",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Compare pulses from multiple models or configurations.

    Parameters
    ----------
    pulse_dict : dict
        Dictionary mapping labels to pulse lists, e.g.,
        {'Model A': pulses_a, 'Model B': pulses_b}
    gate_time : float
        Total gate time
    pulse_idx : int, default=0
        Which pulse to compare (0 for Rabi, 1 for detuning, etc.)
    n_points : int, default=200
        Number of time points
    save_path : str, optional
        Path to save figure
    show : bool, default=True
        Whether to display the plot
    title : str, default="Pulse Comparison"
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object

    Examples
    --------
    >>> pulse_dict = {
    ...     'Initial': initial_pulses,
    ...     'Optimized': optimized_pulses
    ... }
    >>> plot_pulse_comparison(pulse_dict, gate_time=7.0, pulse_idx=0,
    ...                       save_path='comparison.png')
    """
    # Generate time points
    times = np.linspace(0, gate_time, n_points)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each model's pulse
    colors = plt.cm.tab10(np.linspace(0, 1, len(pulse_dict)))

    for idx, (label, pulses) in enumerate(pulse_dict.items()):
        pulse_fn = pulses[pulse_idx]
        values = [
            pulse_fn(t).item() if backend.is_tensor(pulse_fn(t)) else pulse_fn(t)
            for t in times
        ]
        ax.plot(times, values, linewidth=2, label=label, color=colors[idx])

    # Formatting
    ax.set_xlabel("Time (1/Ω_max)", fontsize=12)
    ax.set_ylabel("Pulse Amplitude", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig
