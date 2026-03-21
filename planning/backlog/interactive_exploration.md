# Interactive Jupyter Widgets

**Status:** Backlog | **Priority:** Future | **Estimated Effort:** 2-3 days

---

## Overview

Add interactive Jupyter widgets for exploring quantum control training results. This enhances the static visualization tools with interactive exploration capabilities.

---

## Why This Matters

Static plots are great for publications and reports, but interactive exploration helps with:
- Understanding pulse behavior across different angles
- Debugging training issues
- Teaching and demonstrations
- Quick parameter exploration without re-running code

---

## Proposed Features

### 1. Interactive Angle Selector

```python
from qneural.analysis.interactive import InteractiveExplorer

explorer = InteractiveExplorer(trained_network)
explorer.explore_pulses()  # Opens widget with slider for angle
```

**Widgets:**
- Angle slider (0 to π)
- Real-time pulse plot updates
- Fidelity display
- Gate time readout

### 2. Live Training Dashboard

```python
from qneural.analysis.interactive import TrainingDashboard

dash = TrainingDashboard(trainer)
dash.show()  # Opens live updating dashboard
```

**Widgets:**
- Live loss plot (updates every epoch)
- Current pulse visualization
- Fidelity gauge
- Training controls (pause/resume/stop)

### 3. 3D Interactive Pulse Explorer

```python
explorer.plot_3d_interactive()  # Rotatable 3D plot
```

**Features:**
- Rotate/zoom 3D surface
- Hover for exact values
- Time/angle cross-sections

### 4. Parameter Sweep Interface

```python
explorer.sweep_parameter(
    param='rabi_max',
    range=(20, 30),
    n_points=50
)
```

**Interactive results:**
- See how fidelity changes with parameter
- Compare pulse shapes

---

## Implementation

**Dependencies:**
- `ipywidgets` - Core widget library
- `plotly` - Interactive plots (optional, can use matplotlib widgets)

**Code Structure:**
```python
# qneural/analysis/interactive.py

class InteractiveExplorer:
    """Interactive exploration of trained models."""
    
    def __init__(self, network, gate):
        self.network = network
        self.gate = gate
        
    def explore_pulses(self):
        """Launch interactive pulse explorer."""
        import ipywidgets as widgets
        
        angle_slider = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=np.pi,
            step=0.01,
            description='Angle:',
        )
        
        # Update plot on slider change
        def update_plot(angle):
            # Generate pulses for this angle
            # Update matplotlib plot
            pass
            
        widgets.interact(update_plot, angle=angle_slider)
```

---

## When to Implement

**After:**
- Static visualization tools are working well
- Core training pipeline is stable
- Users are asking for interactive features

**Before:**
- Package release to PyPI
- Comprehensive documentation

---

## Related

- [Visualization Tools](../active/visualization_analysis.md) - Static plots (prerequisite)
- [Animation Training](./animation_training.md) - Related visual feature

---

**Created:** March 21, 2026
