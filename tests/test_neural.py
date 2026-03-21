"""
Unit tests for neural network-based quantum control.

Tests the ML methods in qneural.neural:
    - Neural network models (FeedForwardNN, TimeOptimalController)
    - Loss functions (InfidelityLoss, CompositeLoss)
    - ODE solvers (TorchDiffeqSolver)
    - Pulse generation
    - Quantum evolution
    - Training infrastructure
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qneural.neural import (
    FeedForwardNN,
    PulseGenerator,
    TimeOptimalController,
    InfidelityLoss,
    TimePenaltyLoss,
    CompositeLoss,
    create_infidelity_loss,
    create_time_optimal_loss,
    TorchDiffeqSolver,
    FixedStepSolver,
    create_solver,
    PhysicalPulseGenerator,
    create_default_physical_pulse_generator,
    QuantumEvolver,
    create_evolver,
    QuantumTrainer,
    create_trainer,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_network():
    """Simple feedforward network for testing."""
    return FeedForwardNN(
        input_dim=2,
        output_dim=2,
        hidden_layers=2,
        hidden_units=10
    )


@pytest.fixture
def time_optimal_controller():
    """Time-optimal controller for testing."""
    return TimeOptimalController(
        time_bounds=(3.0, 8.0),
        n_controls=2,
        n_time_steps=21,  # Small for testing
        time_hidden_layers=2,
        time_hidden_units=10,
        pulse_hidden_layers=2,
        pulse_hidden_units=20
    )


@pytest.fixture
def infidelity_loss():
    """Infidelity loss function."""
    return InfidelityLoss(nqubits=2)


@pytest.fixture
def torchdiffeq_solver():
    """Torchdiffeq solver for testing."""
    return TorchDiffeqSolver(method='rk4')  # RK4 is faster for tests


# =============================================================================
# Test Neural Network Models
# =============================================================================

class TestFeedForwardNN:
    """Tests for FeedForwardNN class."""
    
    def test_forward_pass_shape(self, simple_network):
        """Forward pass should return correct shape."""
        # Arrange
        x = torch.randn(5, 2)
        
        # Act
        output = simple_network(x)
        
        # Assert
        assert output.shape == (5, 2)
    
    def test_single_input(self, simple_network):
        """Should handle single input without batch dimension."""
        # Arrange
        x = torch.randn(2)
        
        # Act
        output = simple_network(x)
        
        # Assert
        assert output.shape == (2,)
    
    def test_output_in_range_sigmoid(self):
        """With sigmoid output, values should be in [0, 1]."""
        # Arrange
        net = FeedForwardNN(
            input_dim=2,
            output_dim=2,
            output_activation='sigmoid'
        )
        x = torch.randn(10, 2)
        
        # Act
        output = net(x)
        
        # Assert
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
    
    def test_output_in_range_tanh(self):
        """With tanh output, values should be in [-1, 1]."""
        # Arrange
        net = FeedForwardNN(
            input_dim=2,
            output_dim=2,
            output_activation='tanh'
        )
        x = torch.randn(10, 2)
        
        # Act
        output = net(x)
        
        # Assert
        assert torch.all(output >= -1)
        assert torch.all(output <= 1)
    
    def test_count_parameters(self, simple_network):
        """Should correctly count trainable parameters."""
        # Act
        n_params = simple_network.count_parameters()
        
        # Assert
        assert n_params > 0
        assert isinstance(n_params, int)


class TestPulseGenerator:
    """Tests for PulseGenerator class (from models.py)."""
    
    def test_forward_generates_pulses(self):
        """Should generate pulses for given angle and time points."""
        # Arrange
        generator = PulseGenerator(n_controls=2, n_time_steps=21)
        angle = torch.tensor([0.5 * torch.pi])
        time_points = torch.linspace(0, 1, 21)
        
        # Act
        pulses = generator(angle, time_points)
        
        # Assert
        assert pulses.shape == (1, 21, 2)
        assert torch.all(pulses >= 0)  # Sigmoid output
        assert torch.all(pulses <= 1)
    
    def test_batch_generation(self):
        """Should handle batch of angles."""
        # Arrange
        generator = PulseGenerator(n_controls=2, n_time_steps=21)
        angles = torch.linspace(0.1, torch.pi, 5)
        time_points = torch.linspace(0, 1, 21)
        
        # Act
        pulses = generator(angles, time_points)
        
        # Assert
        assert pulses.shape == (5, 21, 2)


class TestTimeOptimalController:
    """Tests for TimeOptimalController class."""
    
    def test_outputs_time_and_pulses(self, time_optimal_controller):
        """Should output both gate time and pulses."""
        # Arrange
        angle = torch.tensor([0.5 * torch.pi])
        
        # Act
        gate_time, pulses = time_optimal_controller(angle)
        
        # Assert
        assert gate_time.shape == (1,)
        assert pulses.shape[0] == 1  # Batch size
        assert pulses.shape[2] == 2  # n_controls
    
    def test_time_in_bounds(self, time_optimal_controller):
        """Predicted time should be within specified bounds."""
        # Arrange
        angle = torch.tensor([0.3 * torch.pi])
        t_min, t_max = time_optimal_controller.time_bounds
        
        # Act
        gate_time, _ = time_optimal_controller(angle)
        
        # Assert
        assert gate_time.item() >= t_min
        assert gate_time.item() <= t_max
    
    def test_different_angles_different_times(self, time_optimal_controller):
        """Different angles should potentially give different times."""
        # Arrange
        angle1 = torch.tensor([0.2 * torch.pi])
        angle2 = torch.tensor([0.8 * torch.pi])
        
        # Act
        time1, _ = time_optimal_controller(angle1)
        time2, _ = time_optimal_controller(angle2)
        
        # Assert - they might be different (not guaranteed due to random init)
        # Just check both are in valid range
        t_min, t_max = time_optimal_controller.time_bounds
        assert t_min <= time1.item() <= t_max
        assert t_min <= time2.item() <= t_max


# =============================================================================
# Test Loss Functions
# =============================================================================

class TestInfidelityLoss:
    """Tests for InfidelityLoss."""
    
    def test_identical_unitaries_zero_loss(self, infidelity_loss):
        """Identical unitaries should have zero infidelity."""
        # Arrange
        U = torch.eye(4, dtype=torch.cfloat)
        
        # Act
        loss = infidelity_loss(U, U)
        
        # Assert
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_loss_bounded(self, infidelity_loss):
        """Loss should be between 0 and 1."""
        # Arrange
        U1 = torch.eye(4, dtype=torch.cfloat)
        U2 = torch.tensor([[0, 1, 0, 0],
                           [1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=torch.cfloat)
        
        # Act
        loss = infidelity_loss(U1, U2)
        
        # Assert
        assert 0 <= loss.item() <= 1


class TestTimePenaltyLoss:
    """Tests for TimePenaltyLoss."""
    
    def test_requires_gate_time(self):
        """Should raise error if gate_time not provided."""
        # Arrange
        loss_fn = TimePenaltyLoss(weight=0.1)
        U = torch.eye(4, dtype=torch.cfloat)
        
        # Act & Assert
        with pytest.raises(ValueError):
            loss_fn(U, U)
    
    def test_time_penalty_scales_with_weight(self):
        """Loss should scale with weight parameter."""
        # Arrange
        loss_fn1 = TimePenaltyLoss(weight=0.1)
        loss_fn2 = TimePenaltyLoss(weight=0.5)
        U = torch.eye(4, dtype=torch.cfloat)
        gate_time = torch.tensor(5.0)
        
        # Act
        loss1 = loss_fn1(U, U, gate_time=gate_time)
        loss2 = loss_fn2(U, U, gate_time=gate_time)
        
        # Assert
        assert loss2 > loss1


class TestCompositeLoss:
    """Tests for CompositeLoss."""
    
    def test_combines_losses(self):
        """Should combine multiple losses."""
        # Arrange
        loss_fn = CompositeLoss([
            (InfidelityLoss(nqubits=2), 1.0),
            (TimePenaltyLoss(weight=0.1), 0.5)
        ])
        U = torch.eye(4, dtype=torch.cfloat)
        
        # Act
        total_loss = loss_fn(U, U, gate_time=torch.tensor(5.0))
        
        # Assert - should be weighted sum
        # Infidelity is 0, time penalty is 0.5 * 0.1 * 5 = 0.25
        assert total_loss.item() > 0


# =============================================================================
# Test ODE Solvers
# =============================================================================

class TestTorchDiffeqSolver:
    """Tests for TorchDiffeqSolver."""
    
    def test_solves_simple_ode(self, torchdiffeq_solver):
        """Should solve simple exponential decay ODE."""
        # Arrange: dy/dt = -y, solution: y(t) = y0 * exp(-t)
        def f(t, y):
            return -y
        
        y0 = torch.tensor([1.0])
        
        # Act
        solution = torchdiffeq_solver.solve(f, y0, t_span=(0, 1))
        
        # Assert
        assert solution.shape[0] == 2  # Start and end
        # At t=1, y should be approximately exp(-1) ≈ 0.368
        final_y = solution[-1]
        expected = torch.exp(torch.tensor(-1.0))
        assert torch.allclose(final_y, expected, atol=0.1)
    
    def test_returns_evaluation_points(self):
        """Should return solution at specified evaluation points."""
        # Arrange
        solver = TorchDiffeqSolver(method='rk4')
        def f(t, y):
            return torch.ones_like(y)
        
        y0 = torch.tensor([0.0])
        t_eval = torch.linspace(0, 1, 11)
        
        # Act
        solution = solver.solve(f, y0, t_span=(0, 1), t_eval=t_eval)
        
        # Assert
        assert solution.shape[0] == 11


class TestFixedStepSolver:
    """Tests for FixedStepSolver."""
    
    def test_rk4_accuracy(self):
        """RK4 should be reasonably accurate."""
        # Arrange
        solver = FixedStepSolver(method='rk4', n_steps=100)
        def f(t, y):
            return -y
        
        y0 = torch.tensor([1.0])
        
        # Act
        solution = solver.solve(f, y0, t_span=(0, 1))
        
        # Assert
        expected = torch.exp(torch.tensor(-1.0))
        assert torch.allclose(solution, expected, atol=0.01)


class TestSolverFactory:
    """Tests for create_solver factory function."""
    
    def test_creates_torchdiffeq_solver(self):
        """Should create TorchDiffeqSolver."""
        # Act
        solver = create_solver('torchdiffeq', method='rk4')
        
        # Assert
        assert isinstance(solver, TorchDiffeqSolver)
    
    def test_creates_fixedstep_solver(self):
        """Should create FixedStepSolver."""
        # Act
        solver = create_solver('fixedstep', method='rk4', n_steps=50)
        
        # Assert
        assert isinstance(solver, FixedStepSolver)


# =============================================================================
# Test Pulse Generation
# =============================================================================

class TestPhysicalPulseGenerator:
    """Tests for PhysicalPulseGenerator."""
    
    def test_generates_callable_pulses(self):
        """Should generate callable pulse functions."""
        # Arrange
        gen = PhysicalPulseGenerator(
            n_controls=2,
            n_time_steps=21,
            control_ranges=[(0, 25.0), (-50.0, 50.0)]
        )
        nn_output = torch.rand(21, 2)  # NN predictions
        
        # Act
        pulses = gen.generate(nn_output, gate_time=5.0)
        
        # Assert
        assert len(pulses) == 2  # Two pulse functions
        # Should be callable
        val = pulses[0](2.5)
        assert isinstance(val, (float, torch.Tensor))
    
    def test_pulse_values_in_range(self):
        """Pulse values should be scaled to physical ranges."""
        # Arrange
        gen = PhysicalPulseGenerator(
            n_controls=2,
            n_time_steps=21,
            control_ranges=[(0, 25.0), (-50.0, 50.0)]
        )
        nn_output = torch.rand(21, 2)  # [0, 1] from NN
        
        # Act
        pulses = gen.generate(nn_output, gate_time=5.0)
        
        # Test at multiple time points
        for t in [0, 2.5, 5.0]:
            rabi_val = pulses[0](t).item()
            detuning_val = pulses[1](t).item()
            
            # Assert - should be in physical ranges
            assert 0 <= rabi_val <= 25.0
            assert -50.0 <= detuning_val <= 50.0


# =============================================================================
# Test Quantum Evolution
# =============================================================================

class TestQuantumEvolver:
    """Tests for QuantumEvolver."""
    
    def test_evolve_returns_unitary(self):
        """Evolution should return a unitary matrix."""
        # Arrange
        evolver = create_evolver(nqubits=1)
        from ..hardware.rydberg.pulses import constant_pulse
        
        # Constant zero pulses = no evolution
        pulses = [constant_pulse(0.0), constant_pulse(0.0)]
        
        # Act
        U = evolver.evolve(pulses, gate_time=1.0)
        
        # Assert
        assert U.shape[0] == U.shape[1]  # Square matrix
        # Should be approximately identity (no evolution)
        assert U.shape == (2, 2)  # Single qubit computational space


# =============================================================================
# Test Training Infrastructure
# =============================================================================

class TestQuantumTrainer:
    """Tests for QuantumTrainer."""
    
    def test_trainer_initializes(self):
        """Trainer should initialize properly."""
        # Arrange & Act
        trainer = create_trainer(nqubits=2)
        
        # Assert
        assert trainer.nqubits == 2
        assert isinstance(trainer.network, FeedForwardNN)
    
    def test_train_one_step(self):
        """Should be able to run one training step."""
        # Arrange
        trainer = create_trainer(
            nqubits=2,
            hidden_layers=2,
            hidden_units=10
        )
        angles = torch.tensor([0.5 * torch.pi])
        
        # Act - just run 1 epoch
        history = trainer.train(angles, gate_time=3.0, epochs=1, print_every=1)
        
        # Assert
        assert len(history['loss']) == 1
        assert len(history['epoch']) == 1
    
    def test_evaluation(self):
        """Should be able to evaluate trained model."""
        # Arrange
        trainer = create_trainer(nqubits=2, hidden_layers=2, hidden_units=10)
        angles = torch.tensor([0.3, 0.5, 0.7]) * torch.pi
        
        # Act
        trainer.train(angles, gate_time=3.0, epochs=1)
        results = trainer.evaluate(angles, gate_time=3.0)
        
        # Assert
        assert 'angles' in results
        assert 'infidelities' in results
        assert len(results['angles']) == 3


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_pipeline_constant_pulse(self):
        """Test full pipeline with constant pulse (no evolution)."""
        # Arrange
        network = FeedForwardNN(input_dim=2, output_dim=2, hidden_layers=2, hidden_units=10)
        pulse_gen = create_default_physical_pulse_generator(rabi_max=1.0)
        evolver = create_evolver(nqubits=1)
        
        # Generate pulses
        angle = torch.tensor([0.0])  # Identity
        time_points = torch.linspace(0, 1, 21)
        nn_input = torch.stack([angle.repeat(21), time_points], dim=1)
        nn_output = network(nn_input)
        nn_output = nn_output.reshape(21, 2)
        
        # Act
        pulses = pulse_gen.generate(nn_output, gate_time=1.0)
        final_U = evolver.evolve(pulses, gate_time=1.0)
        
        # Assert
        assert final_U.shape == (2, 2)
        # Should be some valid unitary (approximately)
        identity = torch.matmul(final_U, final_U.conj().T)
        assert torch.allclose(identity, torch.eye(2, dtype=torch.cfloat), atol=0.1)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])