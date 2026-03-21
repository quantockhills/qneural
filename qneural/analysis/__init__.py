"""Visualization and analysis tools for quantum control training.

This module provides comprehensive plotting and analysis capabilities
for visualizing quantum control training results, pulse sequences,
and fidelity metrics.

Examples
--------
>>> from qneural.analysis import TrainingVisualizer
>>> viz = TrainingVisualizer()
>>> viz.plot_loss_convergence(history, save_path='loss.png')

>>> from qneural.analysis import FidelityAnalyzer
>>> analyzer = FidelityAnalyzer()
>>> analyzer.plot_fidelity_vs_angle(model, angles)
"""

from .training_plots import plot_loss_convergence, plot_training_summary
from .pulse_plots import plot_pulses_vs_time, plot_pulse_comparison

__all__ = [
    'plot_loss_convergence',
    'plot_training_summary', 
    'plot_pulses_vs_time',
    'plot_pulse_comparison',
]
