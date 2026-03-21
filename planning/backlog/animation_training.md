# Training Animation

**Status:** Backlog | **Priority:** Low | **Estimated Effort:** 1-2 days

---

## Overview

Animate the evolution of pulse sequences during training to visualize how the neural network learns optimal quantum control.

---

## Why This Matters

- **Educational:** Shows how pulses evolve from random to optimized
- **Debugging:** Reveals if training is stuck or oscillating
- **Presentation:** Great for talks and demos
- **Intuition:** Helps understand the optimization landscape

---

## Proposed Features

### 1. Pulse Evolution Animation

```python
from qneural.analysis.animation import TrainingAnimator

animator = TrainingAnimator(training_history)
animator.animate_pulses(
    angle=np.pi,  # Which angle to animate
    save_path='pulse_evolution.mp4',
    fps=30
)
```

**Animation shows:**
- Frame 0: Random initial pulses
- Frame 50: Pulses starting to take shape
- Frame 100: Nearly converged pulses
- Frame 200: Final optimized pulses

### 2. Loss Landscape Animation

```python
animator.animate_loss_landscape(
    parameter_space=('learning_rate', 'gate_time'),
    save_path='loss_landscape.mp4'
)
```

**Shows:**
- How loss surface changes during training
- Trajectory of optimizer in parameter space

### 3. Multi-Angle Convergence

```python
animator.animate_multi_angle_convergence(
    angles=[0.25*np.pi, 0.5*np.pi, 0.75*np.pi],
    save_path='convergence.mp4'
)
```

**Shows:**
- How different angles converge simultaneously
- Which angles are harder to optimize

---

## Implementation Options

**Option A: Matplotlib Animation**
```python
from matplotlib.animation import FuncAnimation

# Standard, works everywhere
# Save as MP4 or GIF
```

**Option B: Image Sequence**
```python
# Save individual frames
# User can compile into video
# More flexible for editing
```

**Option C: JavaScript (for web)**
```python
# Export to D3.js or similar
# Embed in HTML reports
# Interactive playback
```

**Recommended:** Start with Option A (matplotlib), then Option B for flexibility.

---

## Example Output

**MP4 file:** `cz_training_evolution.mp4`

Frames:
1. Epoch 0: Random noise
2. Epoch 50: Structure emerging
3. Epoch 100: Clear pulse shapes
4. Epoch 200: Optimized pulses
5. Overlay: Final pulses + loss curve

---

## When to Implement

**Nice to have, not critical:**
- After all visualization tools are complete
- When preparing presentations/publications
- If users request it

**Low priority** compared to:
- Checkpoint system
- Static visualization
- Performance optimization

---

## Related

- [Visualization Tools](../active/visualization_analysis.md) - Prerequisite
- [Interactive Widgets](./interactive_exploration.md) - Alternative exploration method

---

**Created:** March 21, 2026
