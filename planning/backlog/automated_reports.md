# Automated Report Generation

**Status:** Backlog | **Priority:** Medium | **Estimated Effort:** 1-2 days

---

## Overview

Generate comprehensive training reports automatically after (or during) quantum control training. Reports include visualizations, statistics, and model summaries in multiple formats.

---

## Use Cases

1. **After Training:** Get a complete summary of what was achieved
2. **Comparison:** Compare multiple training runs side-by-side
3. **Documentation:** Generate publication-ready figures
4. **Debugging:** Understand why training succeeded/failed
5. **Sharing:** Share results with collaborators

---

## Proposed Formats

### 1. HTML Dashboard (Rich)

```python
from qneural.analysis import ReportGenerator

report = ReportGenerator()
report.add_training_summary(trainer)
report.add_visualizations(viz)
report.generate_html('training_report.html')
```

**Features:**
- Interactive plots (Plotly)
- Collapsible sections
- Summary statistics table
- Model architecture diagram
- Embedded pulse animations (if available)
- Download links for model checkpoint

**Example sections:**
1. Executive Summary (loss, fidelity, gate time)
2. Training Curves (loss vs epoch)
3. Pulse Visualizations (2D and 3D)
4. Fidelity Analysis (vs angle)
5. Model Architecture (network diagram)
6. Hyperparameters (table)
7. Hardware Configuration (device, precision)

---

### 2. ASCII Tables (Terminal)

```python
report.generate_ascii()  # Print to terminal
```

**Output:**
```
╔════════════════════════════════════════════════╗
║          qneural Training Report               ║
╠════════════════════════════════════════════════╣
║ Training Time:     45.2 minutes               ║
║ Final Loss:        0.0012                     ║
║ Best Fidelity:     99.87%                     ║
║ Epochs:            1000/1000                  ║
╠════════════════════════════════════════════════╣
║ Pulse Quality                                  ║
╠════════════════════════════════════════════════╣
║ Max Rabi:          24.8 MHz                   ║
║ Max Detuning:      48.3 MHz                   ║
║ Pulse Smoothness:  0.94 (0-1)                 ║
╠════════════════════════════════════════════════╣
║ Gate Performance                               ║
╠════════════════════════════════════════════════╣
║ Angle    Fidelity   Gate Time                 ║
║ 0.0π     99.92%     0.28 μs                  ║
║ 0.5π     99.87%     0.28 μs                  ║
║ 1.0π     99.84%     0.28 μs                  ║
╚════════════════════════════════════════════════╝
```

**Benefits:**
- Works in any terminal
- No dependencies on browsers
- Easy to copy-paste
- Good for logs

---

### 3. PDF Export (Publication)

```python
report.generate_pdf('paper_figures.pdf')
```

**Features:**
- Publication-ready figures (matplotlib)
- LaTeX-style formatting
- Vector graphics
- CMYK color support

---

### 4. Markdown Summary

```python
report.generate_markdown('README.md')
```

**For:**
- GitHub repositories
- Documentation
- Easy editing

---

## Implementation

```python
# qneural/analysis/report_generator.py

class ReportGenerator:
    """Generate training reports in multiple formats."""
    
    def __init__(self, trainer=None, checkpoint_path=None):
        """
        Initialize from trainer or checkpoint.
        
        Args:
            trainer: QuantumTrainer instance
            checkpoint_path: Path to saved checkpoint
        """
        pass
    
    def add_training_summary(self):
        """Add training overview section."""
        pass
    
    def add_convergence_plots(self):
        """Add loss convergence figures."""
        pass
    
    def add_pulse_visualizations(self, angles=None):
        """Add pulse plots."""
        pass
    
    def add_fidelity_analysis(self, test_angles=None):
        """Add fidelity vs angle analysis."""
        pass
    
    def generate_html(self, path, template='modern'):
        """
        Generate HTML report.
        
        Templates:
            - 'modern': Clean, interactive
            - 'minimal': Simple, fast
            - 'paper': Academic style
        """
        pass
    
    def generate_ascii(self, file=None):
        """
        Generate ASCII table report.
        
        Args:
            file: File to write to (default: stdout)
        """
        pass
    
    def generate_pdf(self, path, style='academic'):
        """Generate PDF report."""
        pass
    
    def generate_markdown(self, path):
        """Generate Markdown summary."""
        pass
```

---

## Dependencies

**Required:**
- matplotlib (figures)
- jinja2 (HTML templates)

**Optional:**
- plotly (interactive HTML)
- reportlab (PDF generation)
- weasyprint (HTML to PDF)

---

## When to Implement

**After:**
- Checkpoint system (need to load saved models)
- Visualization tools (need plots to include)

**Before:**
- Package release
- Comprehensive documentation

---

## Related

- [Visualization Tools](../active/visualization_analysis.md) - Prerequisite
- [Checkpoint System](../active/checkpoint_resume_plan.md) - Prerequisite

---

**Created:** March 21, 2026
