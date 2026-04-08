"""Training visualization utilities.

Functions for visualizing training progress, loss convergence,
and training summaries.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Union


def plot_loss_convergence(
    history: Union[Dict, List],
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Training Loss Convergence",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot loss convergence over training epochs.

    Parameters
    ----------
    history : dict or list
        Training history containing 'loss' key with list of loss values,
        or a simple list of loss values
    save_path : str, optional
        Path to save figure (e.g., 'loss.png', 'training.pdf')
    show : bool, default=True
        Whether to display the plot
    title : str, default="Training Loss Convergence"
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object

    Examples
    --------
    >>> # With trainer history
    >>> history = trainer.train(angles, gate_time, epochs=100)
    >>> plot_loss_convergence(history, save_path='loss.png')

    >>> # With simple list
    >>> losses = [0.9, 0.7, 0.5, 0.3, 0.1]
    >>> plot_loss_convergence(losses, show=True)
    """
    # Extract loss values
    if isinstance(history, dict):
        if "loss" in history:
            losses = history["loss"]
        elif "train_loss" in history:
            losses = history["train_loss"]
        else:
            raise ValueError("History dict must contain 'loss' or 'train_loss' key")
    else:
        losses = history

    # Convert to numpy array
    losses = np.array(losses)
    epochs = np.arange(len(losses))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot loss
    ax.plot(epochs, losses, "b-", linewidth=2, label="Loss")
    ax.scatter(
        epochs[:: max(1, len(epochs) // 20)],
        losses[:: max(1, len(losses) // 20)],
        c="blue",
        s=20,
        alpha=0.5,
    )

    # Formatting
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add initial and final loss as text
    ax.text(
        0.02,
        0.98,
        f"Initial: {losses[0]:.4f}\nFinal: {losses[-1]:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_training_summary(
    history: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Create a comprehensive training summary plot with multiple metrics.

    Parameters
    ----------
    history : dict
        Training history with keys like 'loss', 'infidelity', 'gate_time', etc.
    save_path : str, optional
        Path to save figure
    show : bool, default=True
        Whether to display the plot
    figsize : tuple, default=(14, 5)
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Determine number of subplots based on available metrics
    available_metrics = []
    if "loss" in history:
        available_metrics.append("loss")
    if "infidelity" in history:
        available_metrics.append("infidelity")
    if "gate_time" in history:
        available_metrics.append("gate_time")

    n_metrics = len(available_metrics)
    if n_metrics == 0:
        raise ValueError(
            "History must contain at least one metric (loss, infidelity, or gate_time)"
        )

    # Create subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    epochs = np.arange(len(history[available_metrics[0]]))

    # Plot each metric
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        values = np.array(history[metric])

        ax.plot(epochs, values, linewidth=2)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add initial/final values
        ax.text(
            0.02,
            0.98,
            f"Initial: {values[0]:.4f}\nFinal: {values[-1]:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.suptitle("Training Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig
