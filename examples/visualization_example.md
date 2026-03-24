# Using the New Unified Visualization Module

The new `qneural.analysis.quantum_control_plots` module provides unified, intelligent plotting functions that work with both **fixed-time (CZ)** and **time-optimal (CPHASE)** gate optimization.

## Quick Start

### Replace Multiple Plot Cells with One Function Call

**Before (in notebook - lots of code):**
```python
# Cell 1: Plot training progress
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].plot(history['epoch'], history['loss'], 'b-', linewidth=2)
# ... many more lines ...

# Cell 2: Plot detuning pulses
sample_angles = torch.tensor([...])
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for i, angle in enumerate(sample_angles):
    # ... many more lines ...

# Cell 3: Plot gate time vs angle
test_angles = torch.linspace(...)
# ... many more lines ...

# Cell 4: Evaluate fidelity
eval_angles = torch.linspace(...)
eval_results = trainer.evaluate(eval_angles)
# ... many more lines ...
```

**After (clean and simple):**
```python
from qneural.analysis import create_optimization_summary

# One function call creates ALL the plots!
figures = create_optimization_summary(
    history=history,
    controller=controller,
    trainer=trainer,
    angle_range=ANGLE_RANGE,
    rabi_max=rabi_max,
    time_bounds=TIME_BOUNDS_NORMALIZED,  # (3.0, 8.5)
    save_dir='results/',
    show=True
)
```

This creates:
- ✅ Training progress (loss, infidelity, gate time)
- ✅ Detuning pulses for sample angles
- ✅ Gate time vs angle (for time-optimal)
- ✅ Fidelity evaluation across angle range
- ✅ All figures saved to `results/` directory

---

## Individual Plotting Functions

You can also use individual functions for more control:

### 1. Training Progress

```python
from qneural.analysis import plot_training_progress

# Automatically detects if fixed-time or time-optimal!
plot_training_progress(
    history,
    rabi_max=25.13,           # For time-optimal: converts to normalized units
    time_bounds=(3.0, 8.5),   # Shows bounds on gate time plot
    save_path='training.png'
)
```

**Works for both:**
- **Fixed-time (CZ)**: Shows loss and infidelity (2 panels)
- **Time-optimal (CPHASE)**: Shows loss, infidelity, and gate time (3 panels)

### 2. Detuning Pulses

```python
from qneural.analysis import plot_detuning_pulses

# Show pulses for specific angles
angles = torch.tensor([[0.4*np.pi], [0.5*np.pi], [0.6*np.pi]])
plot_detuning_pulses(
    controller,
    angles,
    rabi_max=25.13,
    save_path='pulses.png'
)
```

**Works with:**
- `TimeOptimalController` (CPHASE)
- `PhysicalPulseGenerator` (CZ)

### 3. Gate Time vs Angle (Time-Optimal Only)

```python
from qneural.analysis import plot_gate_time_vs_angle

plot_gate_time_vs_angle(
    controller,
    angle_range=(0.4*np.pi, 0.6*np.pi),
    rabi_max=25.13,
    time_bounds=(3.0, 8.5),
    save_path='time_vs_angle.png'
)
```

### 4. Fidelity Evaluation

```python
from qneural.analysis import plot_fidelity_vs_angle

plot_fidelity_vs_angle(
    trainer,
    angle_range=(0.4*np.pi, 0.6*np.pi),
    n_angles=50,
    target_fidelity=99.0,
    save_path='fidelity.png'
)
```

**Works with both:**
- `FixedRabiTrainer` (CZ)
- `TimeOptimalTrainer` (CPHASE)

---

## Benefits

✅ **Unified API**: Same functions work for both CZ and CPHASE
✅ **Auto-detection**: Automatically detects training type from history
✅ **Less code**: Replace 50+ lines of plotting code with 1 function call
✅ **Consistent styling**: Professional, publication-ready plots
✅ **Flexible**: Use individual functions or `create_optimization_summary()`
✅ **Smart defaults**: Sensible defaults, but fully customizable

---

## Example: Simplified CPHASE Notebook

Replace cells 7, 8, 9, and 10 in `cphase_gate_optimization.ipynb` with:

```python
from qneural.analysis import create_optimization_summary

print("\n" + "="*60)
print("📊 GENERATING VISUALIZATION SUMMARY")
print("="*60)

figures = create_optimization_summary(
    history=history,
    controller=controller,
    trainer=trainer,
    angle_range=ANGLE_RANGE,
    rabi_max=rabi_max,
    time_bounds=TIME_BOUNDS_NORMALIZED,
    n_sample_angles=3,
    save_dir='cphase_results/',
    show=True
)

print(f"\n✓ Generated {len(figures)} figures")
print("  Saved to: cphase_results/")
```

That's it! 4 cells → 1 cell. 🎉
