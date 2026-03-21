# Checkpoint & Resume Training System

**Status:** Design Complete ✓ | **Priority:** High | **Estimated Effort:** 1-2 days

---

## Overview

Implement comprehensive checkpoint/resume functionality for quantum control training with automatic saves every 200 epochs, graceful interruption handling, and full state restoration.

---

## Problem Statement

Current limitations:
- Training cannot be resumed if interrupted
- No automatic saving during long training runs
- Loss of progress if system crashes or user stops
- No way to compare different training stages

---

## Solution Design

### 1. File Structure (Hybrid Format)

```
checkpoints/
├── checkpoint_epoch_200.pt       # Binary data (network, optimizer, RNG)
├── checkpoint_epoch_200.json     # Metadata (readable without loading)
├── checkpoint_epoch_400.pt
├── checkpoint_epoch_400.json
├── checkpoint_best.pt            # Symlink to best checkpoint
├── checkpoint_latest.pt          # Symlink to most recent
└── training_config.json          # Initial configuration
```

**Why this format:**
- `.pt` files: Fast loading with `torch.load()`, keeps tensors
- `.json` files: Human-readable metadata, easy to browse checkpoints
- Symlinks: Quick access to "best" and "latest"
- Separate from model files: Keeps checkpointing independent from final model export

---

### 2. Checkpoint Data Structure

**`TrainingCheckpoint` class** saves:

```python
checkpoint = {
    # Model state
    'network_state_dict': network.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    
    # Training progress
    'epoch': current_epoch,
    'global_step': total_steps,  # For LR scheduling
    'history': training_history,
    'best_loss': best_loss_achieved,
    'best_epoch': epoch_of_best_loss,
    
    # RNG state for reproducibility
    'torch_rng_state': torch.get_rng_state(),
    'numpy_rng_state': np.random.get_state(),
    'python_rng_state': random.getstate(),
    
    # Training configuration (for resume validation)
    'config': {
        'nqubits': nqubits,
        'network_architecture': {...},
        'angles': angles.tolist(),
        'gate_time': gate_time,
        'rabi_max': rabi_max,
        'loss_fn_type': type(loss_fn).__name__,
        'loss_fn_config': {...},  # Loss-specific params
        'optimizer_type': type(optimizer).__name__,
        'optimizer_config': {...},  # LR, etc.
    },
    
    # Metadata
    'timestamp': ISO8601_timestamp,
    'qneural_version': __version__,
    'git_commit': auto_detected_git_hash,
}
```

---

### 3. CheckpointManager Class

```python
class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        checkpoint_every: int = 200,  # epochs
        keep_last_n: int = 3,         # recent checkpoints
        keep_best_n: int = 2,         # best checkpoints
        save_best: bool = True,
        best_metric: str = 'loss',    # or 'infidelity'
    )
    
    def save_checkpoint(self, trainer, epoch, is_best=False)
    def load_checkpoint(self, path) -> TrainingCheckpoint
    def list_checkpoints(self) -> List[CheckpointInfo]
    def cleanup_old_checkpoints(self)  # Remove excess
    def get_latest_checkpoint(self) -> Optional[str]
    def get_best_checkpoint(self) -> Optional[str]
```

---

### 4. Enhanced QuantumTrainer

**New methods:**

```python
class QuantumTrainer:
    def __init__(..., checkpoint_manager: Optional[CheckpointManager] = None):
        self.checkpoint_manager = checkpoint_manager
    
    def train(self, ..., resume_from: Optional[str] = None):
        if resume_from:
            self.resume_from_checkpoint(resume_from)
        # ... training loop with periodic checkpointing
    
    def resume_from_checkpoint(self, path: str):
        """Resume training from checkpoint."""
        checkpoint = CheckpointManager.load_checkpoint(path)
        
        # Validate config matches
        self._validate_resume_config(checkpoint.config)
        
        # Restore all state
        self.network.load_state_dict(checkpoint.network_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.current_epoch = checkpoint.epoch
        self.history = checkpoint.history
        self.best_loss = checkpoint.best_loss
        
        # Restore RNG
        torch.set_rng_state(checkpoint.torch_rng_state)
        np.random.set_state(checkpoint.numpy_rng_state)
        random.setstate(checkpoint.python_rng_state)
        
        print(f"Resumed from epoch {checkpoint.epoch}")
        return checkpoint.epoch  # Return to continue from here
    
    @classmethod
    def from_checkpoint(cls, path: str, **override_kwargs):
        """Create trainer from checkpoint with optional overrides."""
        # Load config, create trainer, restore state
```

---

### 5. Graceful Interruption Handling

```python
def setup_interrupt_handler(trainer, checkpoint_manager):
    """Setup Ctrl+C handler to save checkpoint before exit."""
    import signal
    
    def signal_handler(signum, frame):
        print("\n\nInterrupt received! Saving checkpoint...")
        checkpoint_manager.save_checkpoint(
            trainer, 
            trainer.current_epoch,
            is_emergency=True
        )
        print(f"Checkpoint saved to {checkpoint_manager.checkpoint_dir}")
        print("Resume with: trainer.resume_from_checkpoint('...')")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
```

**Output on Ctrl+C:**
```
^C

Interrupt received! Saving checkpoint...
Checkpoint saved to checkpoints/my_experiment
Resume with: trainer.resume_from_checkpoint('checkpoints/my_experiment/checkpoint_latest.pt')
```

---

### 6. Cost Function Serialization

Need to make loss functions serializable:

```python
class QuantumLoss(ABC, nn.Module):
    def get_config(self) -> Dict:
        """Return configuration for serialization."""
        raise NotImplementedError
    
    @classmethod
    def from_config(cls, config: Dict) -> 'QuantumLoss':
        """Recreate loss function from config."""
        raise NotImplementedError

class InfidelityLoss(QuantumLoss):
    def get_config(self):
        return {'nqubits': self.nqubits}
    
    @classmethod
    def from_config(cls, config):
        return cls(nqubits=config['nqubits'])
```

---

### 7. Usage Examples

**Basic training with auto-checkpoint:**

```python
from qneural.neural import QuantumTrainer, CheckpointManager

# Setup checkpointing
checkpoint_mgr = CheckpointManager(
    checkpoint_dir='checkpoints/my_experiment',
    checkpoint_every=200,
    keep_last_n=3,
    keep_best_n=2
)

trainer = QuantumTrainer(
    network=network,
    nqubits=2,
    loss_fn=loss_fn,
    checkpoint_manager=checkpoint_mgr
)

# Train with auto-save every 200 epochs
history = trainer.train(angles, gate_time, epochs=1000)
# Saves: epoch_200.pt, epoch_400.pt, ..., epoch_1000.pt, best.pt
```

**Resume training:**

```python
# Method 1: Resume via trainer
trainer = QuantumTrainer(...)  # Create fresh trainer
trainer.train(angles, gate_time, epochs=1000, resume_from='checkpoints/latest.pt')

# Method 2: Create from checkpoint
trainer = QuantumTrainer.from_checkpoint('checkpoints/latest.pt')
trainer.train(angles, gate_time, epochs=1000)  # Continues from epoch 200
```

**Graceful stop:**

```python
# Press Ctrl+C during training
# Output:
# ^C
# Interrupt received! Saving checkpoint...
# Checkpoint saved to checkpoints/my_experiment
# Resume with: trainer.resume_from_checkpoint('checkpoints/my_experiment/checkpoint_latest.pt')
```

---

## Implementation Checklist

### Files to Create
- [ ] `qneural/utils/checkpoint.py` - Checkpoint class
- [ ] `qneural/utils/checkpoint_manager.py` - Manager class
- [ ] `qneural/utils/interrupt_handler.py` - Signal handling

### Files to Modify
- [ ] `qneural/neural/trainer.py` - Integrate checkpointing
- [ ] `qneural/neural/losses.py` - Add get_config/from_config
- [ ] `qneural/__init__.py` - Export checkpoint classes

### Testing
- [ ] Test save/load roundtrip
- [ ] Test resume produces same results
- [ ] Test interrupt handling
- [ ] Test checkpoint cleanup

---

## Open Questions

1. **Checkpoint directory default**: `checkpoints/` in current working directory, or should it be relative to the script?

2. **Emergency checkpoints**: When interrupted, should I save to `checkpoint_interrupt.pt` or just update `latest.pt`?

3. **Loss function serialization**: Should I save the actual loss function class name and recreate it, or just save the config and let user pass a new loss function on resume?

**Status:** Pending user decisions - see questions above.

---

## Related Plans

- [Visualization Tools](./visualization_analysis.md) - Can use checkpoints to compare training runs
- [Performance Optimization](../backlog/performance_optimization.md) - May affect checkpoint frequency

---

**Created:** March 21, 2026  
**Last Updated:** March 21, 2026  
**Author:** Claude (Anthropic) with Julius de Hond
