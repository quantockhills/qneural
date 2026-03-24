"""Visualization and analysis tools for quantum control training.

This module provides comprehensive plotting and analysis capabilities
for visualizing quantum control training results, pulse sequences,
and fidelity metrics.

Examples
--------
>>> from qneural.analysis import plot_training_progress, plot_detuning_pulses
>>>
>>> # Visualize training progress (works for both CZ and CPHASE)
>>> plot_training_progress(history, rabi_max=25.13, time_bounds=(3.0, 8.5))
>>>
>>> # Plot optimized pulses
>>> plot_detuning_pulses(controller, angles=[0.4*np.pi, 0.5*np.pi, 0.6*np.pi])
>>>
>>> # Create complete summary
>>> figures = create_optimization_summary(
...     history, controller, trainer,
...     angle_range=(0.4*np.pi, 0.6*np.pi),
...     rabi_max=25.13,
...     save_dir='results/'
... )
"""

# Legacy imports (kept for backwards compatibility)
from .training_plots import plot_loss_convergence, plot_training_summary
from .pulse_plots import plot_pulses_vs_time, plot_pulse_comparison

# New unified quantum control visualization
from .quantum_control_plots import (
    plot_training_progress,
    plot_detuning_pulses,
    plot_gate_time_vs_angle,
    plot_fidelity_vs_angle,
    create_optimization_summary
)

__all__ = [
    # Legacy functions
    'plot_loss_convergence',
    'plot_training_summary',
    'plot_pulses_vs_time',
    'plot_pulse_comparison',
    # New unified functions
    'plot_training_progress',
    'plot_detuning_pulses',
    'plot_gate_time_vs_angle',
    'plot_fidelity_vs_angle',
    'create_optimization_summary',
]
