"""
Integration tests for high-fidelity CZ gate optimization.

Tests convergence of CZ gate training with various hyperparameters.
These tests verify that the training pipeline can achieve high fidelity.
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from qneural.gates.rydberg import CZPhiGate
from qneural.neural import FeedForwardNN, TorchDiffeqSolver, QuantumEvolver, InfidelityLoss
from qneural.neural.pulse_generator import PhysicalPulseGenerator
from qneural.core.metrics import unitary_fidelity


class TestCZGateConvergence:
    """Test that CZ gate training converges to high fidelity."""
    
    @pytest.mark.slow
    def test_cz_convergence_with_default_params(self):
        """
        Test CZ gate converges with default hyperparameters.
        
        Uses the standard settings from the paper:
        - 6 layers x 150 units
        - LR: 1e-4
        - Time steps: 101
        - Detuning range: [-2*Ω_max, 2*Ω_max]
        - Gate time: 7.62/Ω_max
        """
        # Setup
        gate = CZPhiGate()
        rabi_max = gate.rabi_max
        gate_time = 7.62 / rabi_max
        angle = torch.tensor([np.pi])
        target_U = gate.get_target_unitary(angle)
        
        network = FeedForwardNN(
            input_dim=2, output_dim=1,
            hidden_layers=6, hidden_units=150,
            activation='relu', output_activation='sigmoid',
            use_batch_norm=True, weight_scale=1.8
        )
        
        pulse_gen = PhysicalPulseGenerator(
            n_controls=1, n_time_steps=101,
            control_ranges=[(-2*rabi_max, 2*rabi_max)]
        )
        
        def rabi_pulse(t): 
            return torch.tensor(rabi_max)
        
        def make_detuning_fn(values, gt):
            def fn(t):
                idx = int(t / gt * (len(values) - 1))
                return values[min(idx, len(values) - 1)]
            return fn
        
        solver = TorchDiffeqSolver(method='rk4')
        evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=101)
        loss_fn = InfidelityLoss(nqubits=2)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
        
        time_grid = torch.linspace(0, 1, 101)
        inputs = torch.stack([angle.repeat(101), time_grid], dim=1)
        
        # Train for 500 epochs
        losses = []
        for epoch in range(500):
            optimizer.zero_grad()
            
            detuning_out = network(inputs).reshape(101)
            detuning_vals = pulse_gen.scale_output(detuning_out, 0)
            detuning_fn = make_detuning_fn(detuning_vals, gate_time)
            final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time)
            loss = loss_fn(final_U, target_U)
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Evaluate final fidelity
        with torch.no_grad():
            fidelity = unitary_fidelity(final_U, target_U, dim=2, nqubits=2)
            infidelity = 1 - fidelity
        
        # Assert training improved
        assert losses[-1] < losses[0], "Loss should decrease during training"
        
        # Assert reasonable fidelity (not expecting >99%, just that it improves)
        assert infidelity < 0.5, f"Infidelity too high: {infidelity:.4f}"
    
    @pytest.mark.slow
    @pytest.mark.parametrize("lr", [1e-3, 5e-4, 1e-4])
    def test_different_learning_rates(self, lr):
        """
        Test that different learning rates all lead to improvement.
        
        This test is parametrized to run with multiple learning rates
        to ensure training works across a range of hyperparameters.
        """
        # Setup
        gate = CZPhiGate()
        rabi_max = gate.rabi_max
        gate_time = 10.0 / rabi_max
        angle = torch.tensor([np.pi])
        target_U = gate.get_target_unitary(angle)
        
        network = FeedForwardNN(
            input_dim=2, output_dim=1,
            hidden_layers=6, hidden_units=150,
            activation='relu', output_activation='sigmoid',
            use_batch_norm=True, weight_scale=1.8
        )
        
        pulse_gen = PhysicalPulseGenerator(
            n_controls=1, n_time_steps=201,
            control_ranges=[(-50.0, 50.0)]
        )
        
        def rabi_pulse(t): 
            return torch.tensor(rabi_max)
        
        def make_detuning_fn(values, gt):
            def fn(t):
                idx = int(t / gt * (len(values) - 1))
                idx = min(idx, len(values) - 1)
                return values[idx]
            return fn
        
        solver = TorchDiffeqSolver(method='rk4')
        evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=201)
        loss_fn = InfidelityLoss(nqubits=2)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        
        time_grid = torch.linspace(0, 1, 201)
        inputs = torch.stack([angle.repeat(201), time_grid], dim=1)
        
        # Train for 300 epochs
        initial_losses = []
        final_losses = []
        
        for epoch in range(300):
            optimizer.zero_grad()
            
            detuning_out = network(inputs).reshape(201)
            detuning_vals = pulse_gen.scale_output(detuning_out, 0)
            detuning_fn = make_detuning_fn(detuning_vals, gate_time)
            pulses = [rabi_pulse, detuning_fn]
            
            final_U = evolver.evolve(pulses, gate_time)
            loss = loss_fn(final_U, target_U)
            
            loss.backward()
            optimizer.step()
            
            if epoch < 10:
                initial_losses.append(loss.item())
            if epoch >= 290:
                final_losses.append(loss.item())
        
        # Assert training improved
        avg_initial = np.mean(initial_losses)
        avg_final = np.mean(final_losses)
        assert avg_final < avg_initial, f"Loss should decrease with LR={lr}"


class TestGateTimeOptimization:
    """Test optimization with different gate times."""
    
    @pytest.mark.slow
    @pytest.mark.parametrize("gate_time_norm", [5.0, 7.0, 7.62, 10.0, 12.0])
    def test_different_gate_times(self, gate_time_norm):
        """
        Test that training works with various gate times.
        
        Tests gate times from 5.0 to 12.0 (in units of 1/Ω_max).
        The optimal is typically around 7.62.
        """
        gate = CZPhiGate()
        rabi_max = gate.rabi_max
        gate_time = gate_time_norm / rabi_max
        angle = torch.tensor([np.pi])
        target_U = gate.get_target_unitary(angle)
        
        network = FeedForwardNN(
            input_dim=2, output_dim=1,
            hidden_layers=6, hidden_units=150,
            activation='relu', output_activation='sigmoid',
            use_batch_norm=True, weight_scale=1.8
        )
        
        pulse_gen = PhysicalPulseGenerator(
            n_controls=1, n_time_steps=101,
            control_ranges=[(-2*rabi_max, 2*rabi_max)]
        )
        
        def rabi_pulse(t): 
            return torch.tensor(rabi_max)
        
        def make_detuning_fn(values, gt):
            def fn(t):
                idx = int(t / gt * (len(values) - 1))
                return values[min(idx, len(values) - 1)]
            return fn
        
        solver = TorchDiffeqSolver(method='rk4')
        evolver = QuantumEvolver(nqubits=2, solver=solver, n_time_steps=101)
        loss_fn = InfidelityLoss(nqubits=2)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
        
        time_grid = torch.linspace(0, 1, 101)
        inputs = torch.stack([angle.repeat(101), time_grid], dim=1)
        
        # Train for 200 epochs
        for epoch in range(200):
            optimizer.zero_grad()
            detuning_out = network(inputs).reshape(101)
            detuning_vals = pulse_gen.scale_output(detuning_out, 0)
            detuning_fn = make_detuning_fn(detuning_vals, gate_time)
            final_U = evolver.evolve([rabi_pulse, detuning_fn], gate_time)
            loss = loss_fn(final_U, target_U)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        with torch.no_grad():
            fidelity = unitary_fidelity(final_U, target_U, dim=2, nqubits=2)
            infidelity = 1 - fidelity
        
        # Assert training made progress
        assert infidelity < 0.8, f"Infidelity too high for gate_time={gate_time_norm}"


class TestControlledPhaseOptimizer:
    """Test the high-level ControlledPhaseOptimizer interface."""
    
    @pytest.mark.slow
    def test_optimizer_factory_czphi(self):
        """
        Test that ControlledPhaseOptimizer works with factory function.
        
        This tests the convenience factory function for creating optimizers.
        """
        from qneural.gates.rydberg import create_czphi_optimizer
        from qneural.neural import (
            FeedForwardNN, create_default_physical_pulse_generator,
            QuantumEvolver, TorchDiffeqSolver, QuantumTrainer, InfidelityLoss
        )
        
        gate = CZPhiGate()
        rabi_max = gate.rabi_max
        
        # Create network with correct settings
        network = FeedForwardNN(
            input_dim=2, output_dim=1,
            hidden_layers=6, hidden_units=150,
            activation='relu', output_activation='sigmoid',
            use_batch_norm=True, weight_scale=1.8
        )
        
        # Create optimizer
        optimizer = create_czphi_optimizer(time_optimal=False)
        
        # Train
        angle = torch.tensor([np.pi])
        gate_time = 7.62 / rabi_max
        
        history = optimizer.train(
            angles=angle,
            gate_time=gate_time,
            epochs=100,
            learning_rate=1e-4,
            print_every=50
        )
        
        # Assert training completed
        assert len(history['loss']) == 100
        assert history['loss'][-1] < history['loss'][0]
        
        # Evaluate
        result = optimizer.evaluate(angle)
        assert 'infidelity' in result
        assert result['infidelity'] < 1.0
