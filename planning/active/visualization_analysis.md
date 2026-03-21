# Visualization & Analysis Tools

**Status:** Analysis Complete ✓ | **Priority:** High | **Estimated Effort:** 2-3 days

---

## Overview

Port and improve visualization and analysis tools from the original research notebooks. Create a centralized, reusable plotting module for quantum control training analysis.

---

## Analysis of Original Code

### What Exists (Strengths)

1. **Training Progress Visualization** - Real-time loss tracking, gate time scatter plots
2. **3D Pulse Visualization** - Surface plots showing pulses across time and angles  
3. **Multi-Model Comparison** - Loading and comparing multiple trained networks
4. **Verification Tools** - QuTiP integration for independent validation
5. **Fidelity Analysis** - Log-scale infidelity plots, fitted curves

### What Needs Improvement

1. **Scattered Code** - Plotting functions duplicated across notebooks
2. **Hardcoded Paths** - File paths embedded in analysis code
3. **No Centralized Module** - Each notebook has custom plotting
4. **Limited Statistics** - No error bars or confidence intervals
5. **No Automation** - Manual compilation of results

---

## Proposed Implementation

### 1. Module Structure

```
qneural/
├── analysis/
│   ├── __init__.py
│   ├── visualizer.py          # Main visualization class
│   ├── pulse_plots.py         # Pulse visualization tools
│   ├── training_plots.py      # Loss/convergence plots
│   ├── fidelity_analysis.py   # Fidelity comparison tools
│   ├── verification.py        # QuTiP verification integration
│   └── report_generator.py    # Automated report generation
├── examples/
│   ├── 01_basic_cz_gate.py
│   ├── 02_multi_angle_training.py
│   ├── 03_time_optimal_control.py
│   ├── 04_visualize_pulses.py
│   ├── 05_analyze_results.py
│   └── notebooks/
│       ├── tutorial_1_getting_started.ipynb
│       ├── tutorial_2_training_analysis.ipynb
│       └── tutorial_3_advanced_visualization.ipynb
└── utils/
    └── plot_utils.py          # Shared plotting utilities
```

---

### 2. Core Classes

**TrainingVisualizer** - Main interface
```python
from qneural.analysis import TrainingVisualizer

viz = TrainingVisualizer()
viz.plot_loss_convergence(history)                    # Loss vs epoch
viz.plot_gate_time_vs_angle(trainer, angles)         # Gate time scatter
viz.plot_pulses_3d(network, angles)                  # 3D surface
viz.plot_fidelity_landscape(network, angle_range)    # Fidelity heatmap
viz.create_training_report(results, save_path)       # Full report
```

**PulseVisualizer** - Pulse-specific plots
```python
viz.plot_pulses_vs_time(pulses, gate_time)           # Time series
viz.plot_pulse_comparison(models_dict)               # Multiple models
viz.plot_pulse_3d_surface(network, angle_range)      # 3D visualization
```

**FidelityAnalyzer** - Performance analysis
```python
analyzer = FidelityAnalyzer()
analyzer.compute_infidelity_vs_angle(model, angles)  # Fidelity curve
analyzer.compare_models(models_dict)                 # Side-by-side
analyzer.fit_gate_time_model(angles, gate_times)     # Curve fitting
analyzer.verify_with_qutip(model, test_cases)        # Independent validation
```

---

### 3. Features to Port from Original

#### A. Training Progress Plots (from cphase_optim.ipynb)

**What to port:**
- ✅ Loss convergence over epochs
- ✅ Gate time vs angle scatter plots
- ✅ Gradient norm tracking

**Improvements:**
- 🔧 Add shaded confidence intervals (multiple runs)
- 🔧 Moving average smoothing option
- 🔧 Log/linear scale toggle

**Code pattern:**
```python
# Original (in notebook)
plt.scatter(angle_list.detach().numpy(), 
            composite_network.gatetime_prediction.detach().numpy()*cfn.rabi, 
            s=2.4)

# New (in module)
viz.plot_gate_time_vs_angle(trainer, angles, 
    confidence_interval=True, 
    smooth=True,
    save_path='gate_time_plot.png')
```

---

#### B. Pulse Visualizations (from analysis/plot.ipynb)

**What to port:**
- ✅ Detuning/Rabi vs time plots
- ✅ 3D surface plots (time × angle × amplitude)
- ✅ Multiple angles on same plot

**Improvements:**
- 🔧 Interactive sliders for angle selection (future)
- 🔧 Animation of pulse evolution
- 🔧 Color-coded by fidelity

**Code pattern:**
```python
# Original
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(x, y, pred_outputs_det.detach().numpy()/cfn.rabi, color='green')

# New
viz.plot_pulse_3d_surface(network, angle_range=(0, np.pi), 
    control='detuning',
    colormap='viridis',
    save_path='pulse_3d.png')
```

---

#### C. Fidelity Analysis (from analysis/plot.ipynb)

**What to port:**
- ✅ Log-scale infidelity plots
- ✅ Fitted theoretical curves
- ✅ Decay vs no-decay comparison

**Improvements:**
- 🔧 Statistical error bars (bootstrap)
- 🔧 Confidence intervals
- 🔧 Automated curve fitting with multiple models

**Code pattern:**
```python
# Original
plt.yscale('log')
plt.scatter(x_list.T[0], np.array(fidelity_arr), s = 0.3, color = 'red')

# New
analyzer.plot_infidelity_with_fit(angles, infidelities,
    fit_models=['exponential', 'polynomial'],
    confidence_interval=0.95,
    save_path='fidelity_analysis.png')
```

---

#### D. Verification Tools (from verification_*.ipynb)

**What to port:**
- ✅ QuTiP Bloch sphere visualization
- ✅ Bell state preparation test
- ✅ Unitary comparison

**Improvements:**
- 🔧 Automated test suite
- 🔧 Pass/fail reporting
- 🔧 Statistical validation

**Code pattern:**
```python
# Run comprehensive verification
verification = analyzer.verify_with_qutip(
    model=trained_network,
    test_cases=['bell_state', 'ghz_state', 'random_states'],
    n_samples=100
)

print(verification.summary())
# Output: 
# Bell state fidelity: 0.9992 ± 0.0003 ✓
# GHZ state fidelity: 0.9987 ± 0.0005 ✓
# Random states: 0.9876 ± 0.0123 ✓
```

---

### 4. New Features to Add

#### A. Automated Report Generation

```python
from qneural.analysis import ReportGenerator

report = ReportGenerator()
report.add_training_summary(trainer)
report.add_convergence_plots(history)
report.add_pulse_visualizations(best_model)
report.add_fidelity_analysis(test_results)
report.generate_html('training_report.html')
# OR for terminal:
report.generate_ascii()  # Nice tables in terminal
```

**Output formats:**
- HTML dashboard (rich, interactive)
- ASCII tables (terminal-friendly)
- PDF export (publication-ready)

---

#### B. Statistical Analysis

```python
# Bootstrap confidence intervals
analyzer.bootstrap_confidence_interval(n_samples=100)

# Compare multiple training runs
analyzer.compare_training_runs(run_ids=['run1', 'run2', 'run3'])

# Hyperparameter sensitivity
analyzer.hyperparameter_sensitivity_study(
    param='learning_rate',
    values=[1e-4, 5e-4, 1e-3]
)
```

---

### 5. Examples to Create

**Priority: Fixed-Time Examples First**

**Tutorial 1: Getting Started with Fixed-Time Training** (30 min) ✅
```python
# examples/01_basic_cz_gate.py
"""Basic CZ gate optimization with fixed gate time."""
from qneural import CZPhiGate, QuantumTrainer
from qneural.analysis import TrainingVisualizer

# Train with fixed gate time
trainer = QuantumTrainer(...)
history = trainer.train(angles, gate_time=7.0, epochs=200)

# Visualize
viz = TrainingVisualizer()
viz.plot_loss_convergence(history)
viz.plot_fidelity_vs_angle(trainer, angles)  # Works with fixed-time
```

**Tutorial 2: Multi-Angle Training & Analysis** (45 min) ✅
- Training on angle families (fixed-time)
- 3D pulse visualization
- Fidelity landscape analysis
- Works with current fixed-time implementation

**Tutorial 3: Advanced Analysis & Reporting** (45 min) ✅
- Loading saved models
- Comparative analysis
- Generating publication-ready figures
- Statistical analysis

**Tutorial 4: Time-Optimal Control** (60 min) ⏳
- **Requires**: Time-optimal NN chaining implementation
- Variable gate time optimization
- Time vs infidelity tradeoff
- `plot_gate_time_vs_angle()` visualization
- Advanced verification

---

## Improvements Over Original

| Aspect | Original | New Implementation |
|--------|----------|-------------------|
| **Organization** | Scattered in notebooks | Centralized module |
| **Reusability** | Copy-paste code | Importable classes |
| **Documentation** | Minimal | Full docstrings + tutorials |
| **Interactivity** | Static plots | Widgets + animations (future) |
| **Automation** | Manual analysis | Automated reports |
| **Statistics** | Point estimates | Confidence intervals |
| **Testing** | None | Unit tests for plots |

---

## Implementation Checklist

**Legend:**
- ✅ = Works with fixed-time training (current)
- ⏳ = Requires time-optimal training (needs NN chaining)

### Phase 1: Core Visualizations (Fixed-Time) ✅
- [ ] `plot_loss_convergence()` - Basic training curves ✅
- [ ] `plot_pulses_vs_time()` - Time series ✅
- [ ] `plot_fidelity_vs_angle()` - Fidelity curves ✅

### Phase 2: Time-Optimal Specific ⏳
- [ ] `plot_gate_time_vs_angle()` - Scatter plots ⏳
- [ ] `fit_gate_time_model()` - Curve fitting ⏳
- [ ] Time vs infidelity tradeoff analysis ⏳

### Phase 3: Advanced Plots (Both)
- [ ] `plot_pulse_3d_surface()` - 3D visualization ✅
- [ ] `plot_fidelity_landscape()` - Heatmaps ✅
- [ ] `plot_pulse_comparison()` - Multi-model ✅

### Phase 4: Analysis Tools (Both)
- [ ] `FidelityAnalyzer` class ✅
- [ ] QuTiP verification integration ✅
- [ ] Statistical analysis (bootstrap, confidence intervals) ✅

### Phase 5: Reporting (Both)
- [ ] HTML report generator ✅
- [ ] ASCII table generator ✅
- [ ] Automated summary statistics ✅

### Phase 6: Examples
- [ ] 01_basic_cz_gate.py (fixed-time) ✅
- [ ] 02_multi_angle_training.py (fixed-time) ✅
- [ ] 03_time_optimal_control.py (time-optimal) ⏳
- [ ] 04_analysis_and_reporting.py (both) ✅
- [ ] Tutorial notebooks (focus on fixed-time first) ✅

---

## Dependencies

**Required:**
- matplotlib (plotting)
- numpy (numerical)
- torch (tensors)

**Optional:**
- qutip (verification, Bloch spheres)
- plotly (interactive plots - future)
- jupyter (notebooks)

---

## Related Plans

- [Checkpoint System](./checkpoint_resume_plan.md) - Can visualize checkpoint comparisons
- [Interactive Widgets](../backlog/interactive_exploration.md) - Future enhancement

---

**Created:** March 21, 2026  
**Last Updated:** March 21, 2026  
**Author:** Claude (Anthropic) with Julius de Hond
