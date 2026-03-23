# qneural Library Restructure Plan

**Status:** Ready for implementation | **Estimated effort:** 2-3 days  
**Created:** March 21, 2026  
**Last Updated:** March 21, 2026

---

## 1. Current Problems (Verified)

### Confirmed Issues:
1. ✅ **Phase Correction Formula Wrong** - Uses all diagonal phases instead of symmetric |01⟩-based correction
2. ✅ **Double Correction Bug** - `evaluate()` applies corrections twice  
3. ✅ **Over-complex Architecture** - 10+ wrapper classes with overlapping responsibilities

---

## 2. Proposed Simplified Architecture

### Core Principle: **Explicit is better than implicit**

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                            │
│  (Simple functions, not classes)                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  DIRECT TRAINING                             │
│                                                              │
│  def train_pulse_optimizer():                                │
│      network = FeedForwardNN(...)                            │
│      optimizer = torch.optim.Adam(network.parameters())      │
│                                                              │
│      for epoch in range(epochs):                             │
│          pulses = generate_pulses(network, angle, time)      │
│          U = evolve_unitary(pulses, time)                    │
│          loss = infidelity(U, target_U)                      │
│          loss.backward()                                     │
│          optimizer.step()                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              COMPONENT LAYER (Explicit)                      │
│                                                              │
│  generate_pulses():                                          │
│    - NN forward pass                                         │
│    - Scale outputs to physical ranges                        │
│    - Return callable pulse functions                         │
│                                                              │
│  evolve_unitary():                                           │
│    - ODE solve: dU/dt = -iH(t)U                             │
│    - NO automatic phase corrections                          │
│    - Return raw unitary                                      │
│                                                              │
│  infidelity():                                               │
│    - Compute 1 - |Tr(U†U_target)|²/dim²                     │
│    - Compare raw unitaries                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              UTILITIES (Separate, Optional)                  │
│                                                              │
│  apply_phase_correction():                                   │
│    - Post-hoc correction for gate characterization           │
│    - NOT used during training                                │
│                                                              │
│  plot_pulses(), plot_training():                             │
│    - Visualization utilities                                 │
│    - No side effects                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Key Changes

### A. Remove Wrapper Classes

**DELETE:**
- ❌ `ControlledPhaseOptimizer` 
- ❌ `QuantumTrainer`
- ❌ `PulseGenerator` (redundant with simple functions)

**KEEP (Simplified):**
- ✅ `FeedForwardNN` (already good)
- ✅ `QuantumEvolver` (but remove auto-correction)
- ✅ `InfidelityLoss` (already good)
- ✅ `TimeOptimalController` (keep for time-optimal case)

### B. Fix Phase Corrections

**Training (NO corrections):**
```python
U = evolve_unitary(pulses, time, apply_corrections=False)
loss = infidelity(U, target_U)  # Both uncorrected - valid comparison
```

**Evaluation (Manual correction):**
```python
U_raw = evolve_unitary(pulses, time, apply_corrections=False)
U_corrected = apply_symmetric_correction(U_raw)  # Explicit
fidelity = infidelity(U_corrected, target_U)  # Both corrected
```

### C. Unified Training Function

**⚠️ IMPORTANT:** This must work for BOTH fixed-time and time-optimal cases!

```python
def train_pulse_optimizer(
    network: nn.Module,
    target_gate: Callable,
    angles: torch.Tensor,
    gate_time: Optional[float],  # None for time-optimal
    n_steps: int = 101,
    lr: float = 1e-4,
    epochs: int = 1000,
    device: str = 'cpu',
    time_optimal: bool = False,  # NEW: flag for time-optimal mode
) -> Dict:
    """
    Train network to generate pulses for target gate.
    
    Works for BOTH fixed-time and time-optimal training:
    
    Fixed-time mode (time_optimal=False):
        - network outputs: [detuning] or [rabi, detuning]
        - gate_time: fixed float value
        - loss = infidelity + 0 (no time penalty)
    
    Time-optimal mode (time_optimal=True):
        - network outputs: [gate_time, detuning] or [gate_time, rabi, detuning]
        - gate_time: ignored (predicted by network)
        - loss = infidelity + time_penalty
    
    Parameters
    ----------
    network : nn.Module
        Neural network with:
        - input_dim=2 (angle, time) for fixed-time
        - input_dim=1 (angle) for time-optimal
        - output_dim=1, 2, or 3 depending on mode
    target_gate : Callable
        Function that returns target unitary for given angle
    angles : torch.Tensor
        Angles to train on
    gate_time : float or None
        Fixed gate time (if time_optimal=False), or None (if time_optimal=True)
    n_steps : int
        Number of time steps for pulse discretization
    lr : float
        Learning rate
    epochs : int
        Training epochs
    device : str
        Device to run on
    time_optimal : bool
        If True, network also predicts gate time
        
    Returns
    -------
    Dict with 'network', 'history', 'final_fidelity', 'final_gate_time'
    """
    # Setup
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    
    # Determine n_controls from network
    n_controls = network.output_dim
    if time_optimal:
        n_controls = network.output_dim - 1  # First output is gate_time
    
    pulse_gen = PhysicalPulseGenerator(
        n_controls=n_controls,
        n_time_steps=n_steps,
        control_ranges=get_control_ranges(n_controls)
    )
    
    evolver = QuantumEvolver(nqubits=2, n_time_steps=n_steps)
    history = {'loss': [], 'epoch': [], 'gate_time': []}
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        total_loss = 0.0
        total_time = 0.0
        
        for angle in angles:
            if time_optimal:
                # Time-optimal: network predicts gate_time
                nn_output = network(angle.unsqueeze(0))  # [1, output_dim]
                pred_time = nn_output[0, 0]  # First output is time
                pulse_output = nn_output[0, 1:]  # Rest are pulses
                
                pulses = pulse_gen.generate(pulse_output, pred_time)
                U = evolver.evolve(pulses, pred_time, apply_corrections=False)
                
                # Time penalty
                time_penalty = 0.1 * pred_time  # Configurable weight
            else:
                # Fixed-time: use provided gate_time
                time_grid = torch.linspace(0, 1, n_steps, device=device)
                inputs = torch.stack([angle.repeat(n_steps), time_grid], dim=1)
                nn_output = network(inputs)
                
                pulses = pulse_gen.generate(nn_output, gate_time)
                U = evolver.evolve(pulses, gate_time, apply_corrections=False)
                
                pred_time = gate_time
                time_penalty = 0.0
            
            # Compute loss
            U_target = target_gate(angle)
            infidelity_loss = infidelity(U, U_target, nqubits=2)
            loss = infidelity_loss + time_penalty
            
            total_loss += loss
            total_time += pred_time.item()
        
        # Backprop
        avg_loss = total_loss / len(angles)
        avg_loss.backward()
        optimizer.step()
        
        # Log
        history['loss'].append(avg_loss.item())
        history['epoch'].append(epoch)
        history['gate_time'].append(total_time / len(angles))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss.item():.6f}, "
                  f"Avg time={history['gate_time'][-1]:.4f}")
    
    # Final evaluation
    with torch.no_grad():
        fidelities = []
        gate_times = []
        
        for angle in angles:
            if time_optimal:
                nn_output = network(angle.unsqueeze(0))
                pred_time = nn_output[0, 0]
                pulse_output = nn_output[0, 1:]
                pulses = pulse_gen.generate(pulse_output, pred_time)
                U = evolver.evolve(pulses, pred_time, apply_corrections=False)
                gate_times.append(pred_time.item())
            else:
                pulses = generate_pulses_for_angle(
                    network, angle, gate_time, pulse_gen
                )
                U = evolver.evolve(pulses, gate_time, apply_corrections=False)
                gate_times.append(gate_time)
            
            U_target = target_gate(angle)
            fidelities.append(fidelity(U, U_target, nqubits=2).item())
    
    history['final_fidelity'] = np.mean(fidelities)
    history['final_gate_time'] = np.mean(gate_times)
    
    return {
        'network': network,
        'history': history,
        'final_fidelity': history['final_fidelity'],
        'final_gate_time': history['final_gate_time'],
        'time_optimal': time_optimal
    }
```

### D. Simplified API Examples

**Fixed-time (Current use case):**
```python
from qneural import FeedForwardNN, train_pulse_optimizer
from qneural.gates import czphi_unitary

network = FeedForwardNN(
    input_dim=2,      # [angle, normalized_time]
    output_dim=1,     # Just detuning
    hidden_layers=6,
    hidden_units=150,
    weight_scale=1.8
)

result = train_pulse_optimizer(
    network=network,
    target_gate=czphi_unitary,
    angles=torch.tensor([0.5*np.pi, np.pi]),
    gate_time=7.62 / rabi_max,  # Fixed time
    time_optimal=False
)

print(f"Fidelity: {result['final_fidelity']*100:.2f}%")
```

**Time-optimal (Future use case):**
```python
network = FeedForwardNN(
    input_dim=1,      # Just angle
    output_dim=2,     # [gate_time, detuning]
    hidden_layers=6,
    hidden_units=150,
    weight_scale=1.8
)

result = train_pulse_optimizer(
    network=network,
    target_gate=czphi_unitary,
    angles=torch.linspace(0.1*np.pi, np.pi, 20),
    gate_time=None,  # Network predicts this
    time_optimal=True
)

print(f"Fidelity: {result['final_fidelity']*100:.2f}%")
print(f"Avg gate time: {result['final_gate_time']:.4f}")
```

---

## 4. Implementation Plan

### Phase 1: Fix Critical Bugs (1 day)
- [ ] Change `QuantumEvolver.evolve()` default to `apply_corrections=False`
- [ ] Fix double-correction in `ControlledPhaseOptimizer.evaluate()`
- [ ] Add symmetric phase correction function matching original code
- [ ] Test: Verify fidelity improves to <0.1

### Phase 2: Create Simplified API (1 day)
- [ ] Create `train_pulse_optimizer()` function (handles both modes)
- [ ] Create `generate_pulses()` helper
- [ ] Create simplified evolution function
- [ ] Add comprehensive docstrings

### Phase 3: Deprecate Old API (1 day)
- [ ] Mark `ControlledPhaseOptimizer` as deprecated
- [ ] Mark `QuantumTrainer` as deprecated  
- [ ] Update examples to use new API
- [ ] Update README with migration guide

### Phase 4: Testing & Validation (0.5 days)
- [ ] Verify fixed-time example works (infidelity < 0.1)
- [ ] Verify time-optimal example works
- [ ] Add unit tests for new functions
- [ ] Performance benchmark vs original

---

## 5. Benefits

| Aspect | Current (10+ classes) | Proposed (3 classes) | Winner |
|--------|----------------------|---------------------|--------|
| **Lines of code** | ~2000 | ~500 | Proposed |
| **Classes** | 10+ | 3 (Network, Evolver, Loss) | Proposed |
| **Hidden behavior** | High (corrections in evolver) | None (explicit) | Proposed |
| **Debuggability** | Hard (many wrappers) | Easy (direct flow) | Proposed |
| **Fidelity** | 40-60% (due to bugs) | >99% (verified approach) | Proposed |
| **Learning curve** | Steep | Gentle | Proposed |
| **Flexibility** | Low | High (both modes in one function) | Proposed |

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing code | Deprecation warnings, migration guide |
| Performance regression | Benchmark before/after |
| Missing features | Port features incrementally |
| User confusion | Clear documentation, examples |
| Time-optimal complexity | Unified function handles both modes |

---

## 7. Python Best Practices Alignment

This restructure aligns with:

- ✅ **Zen of Python**: "Simple is better than complex"
- ✅ **PEP 20**: "Explicit is better than implicit"
- ✅ **PyTorch patterns**: Matches PyTorch tutorials (explicit training loops)
- ✅ **SOLID principles**: Single responsibility, clear interfaces
- ✅ **Scientific Python**: Functions for data transformation, classes for stateful entities

---

## 8. Design Decision: Unified Training Function

**Why one function for both modes?**

1. **Reduces API surface**: One function instead of two
2. **Shared code**: 90% of code is identical (forward pass, backward pass, logging)
3. **Easy switching**: Change `time_optimal=False` to `True`, adjust network outputs
4. **Consistent interface**: Same return format, same logging, same behavior

**Trade-off**: Slightly more complex function vs two simpler functions
**Decision**: Worth it for API consistency

---

## Decision Checklist

- [ ] Approve simplified architecture (3 classes vs 10+)
- [ ] Approve unified training function (both modes)
- [ ] Approve explicit phase corrections (not automatic)
- [ ] Schedule implementation (2-3 days)

**Status:** Awaiting approval

---

**Questions?**
- Does the unified function approach make sense?
- Should we keep any wrapper classes?
- Any concerns about the migration path?
