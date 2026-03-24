# Archival CPHASE vs qneural TimeOptimalTrainer Differences

## Critical Performance Issues

### 1. **BATCH PROCESSING** ⚠️ MAJOR
**Archival**: Processes entire batch in ONE ODE call
```python
# All angles evolved together using gate_time.max()
time_arr_ = torch.linspace(0, 1.0, time_steps) * neural_model.gatetime_prediction.max()
sol_intrm = reduce_r_dim_2q_vector(tdf.odeint(czphi.instance, init_matrix, time_arr_, method='rk4')[-1], angle_batch=czphi.angle_batch)
```

**qneural**: Loops through angles ONE BY ONE
```python
for angle in angles:  # ← SLOW!
    gate_time, detuning = controller(angle)
    final_unitary = self._evolve(...)  # Separate ODE call per angle
```

**Impact**: ~80x slower for batch_size=80!

### 2. **ANGLE SCALING/OFFSET** ⚠️ IMPORTANT
**Archival**: Uses scale_factor and offset
```python
composite_network.offset = np.mean(composite_network.angle_range)
composite_network.scale_factor = 1e0
angle_input = scale_and_offset(desired_angle, composite_network)
# Where: (angle - offset) * scale_factor
```

**qneural**: No scaling/offset - raw angles used

**Impact**: Training stability, especially for small angles

### 3. **ANGLE RESAMPLING**
**Archival**: Resamples angles every 25 epochs
```python
if epoch_ % 2.5e1 == 0:
    desired_angle = torch.rand(angle_batch, 1) * (range[1] - range[0]) + range[0]
    angle_input = scale_and_offset(desired_angle, composite_network)
```

**qneural**: Fixed angles for entire training

**Impact**: Better exploration of angle space

### 4. **LOSS COMPUTATION**
**Archival**:
```python
infidelity_term = cfn.unitary_infidelity_array(solution, czp_gate_stack(input_tensor/scale_factor + offset))
time_term = multiplier * torch.mean(neural_model.gatetime_prediction)
loss = infidelity_term + time_term
```

**qneural**:
```python
# Per-angle loop
time_penalty = self.time_weight * gate_time.squeeze()
loss = infidelity + time_penalty
# Then average
```

**Impact**: Archival computes mean time penalty, qneural sums then averages

### 5. **CORRECTIONS**
**Archival**: Batched `torch.bmm`
```python
mat0 = corr_1q_rotation_fast_vector(torch.angle(sol_intrm[:, 1, 1]), angle_batch)
return torch.bmm(mat0, sol_intrm)  # Batch matrix multiply
```

**qneural**: Individual corrections in loop

## Recommendations

1. **URGENT**: Implement batched training in TimeOptimalTrainer
   - Process all angles in one forward pass
   - Use `gate_time.max()` for evolution time
   - Batch corrections with bmm

2. **Add angle scaling/offset** to controller or trainer
   - `offset = mean(angle_range)`
   - `scale_factor = 1.0` (configurable)

3. **Add angle resampling** to train() method
   - `resample_every=25` parameter
   - Resample from angle_range

4. **Match loss computation exactly**
   - Use mean of time penalties, not sum

## Speed Test Results
TODO: Run comparison

## Notebook Updates Needed
- Add angle scaling
- Add resampling logic
- Update to use batched trainer (when implemented)
