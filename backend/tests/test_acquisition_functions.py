#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for acquisition functions implementations.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock

from bo_engine.acquisition.ei_numpy import ExpectedImprovementNumpy
from bo_engine.acquisition.pi_numpy import ProbabilityOfImprovementNumpy
from bo_engine.acquisition.ucb_numpy import UpperConfidenceBoundNumpy


class TestAcquisitionFunctions(unittest.TestCase):
    """Test cases for various acquisition functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock parameter space
        self.parameter_space = MagicMock()
        
        # Set up a minimization objective by default
        self.parameter_space.objectives = [{"type": "minimize"}]
        
        # Create a mock model
        self.model = MagicMock()
        self.model.is_initialized = True
        
        # Set up test data
        self.X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # 3 test points, 2D
        
        # Best function value observed so far
        self.best_f = 0.5

    def test_ei_minimization(self):
        """Test Expected Improvement for minimization problem."""
        # Set up mock model predictions
        mu = np.array([0.6, 0.4, 0.7])  # Predicted means
        sigma = np.array([0.2, 0.1, 0.3])  # Predicted std deviations
        self.model.predict.return_value = (mu, sigma)
        
        # Create EI acquisition function
        ei = ExpectedImprovementNumpy(self.parameter_space)
        
        # Evaluate EI
        ei_values = ei.evaluate(self.X, self.model, self.best_f)
        
        # Check that EI values are non-negative
        self.assertTrue(np.all(ei_values >= 0))
        
        # Check that the point with mu < best_f has higher EI
        self.assertTrue(ei_values[1] > ei_values[0])
        self.assertTrue(ei_values[1] > ei_values[2])

    def test_ei_maximization(self):
        """Test Expected Improvement for maximization problem."""
        # Change to maximization objective
        self.parameter_space.objectives = [{"type": "maximize"}]
        
        # Set up mock model predictions
        mu = np.array([0.6, 0.7, 0.4])  # Predicted means
        sigma = np.array([0.2, 0.1, 0.3])  # Predicted std deviations
        self.model.predict.return_value = (mu, sigma)
        
        # Create EI acquisition function
        ei = ExpectedImprovementNumpy(self.parameter_space)
        
        # Evaluate EI
        ei_values = ei.evaluate(self.X, self.model, self.best_f)
        
        # Check that EI values are non-negative
        self.assertTrue(np.all(ei_values >= 0))
        
        # Check that the point with mu > best_f has higher EI
        self.assertTrue(ei_values[1] > ei_values[2])

    def test_pi_minimization(self):
        """Test Probability of Improvement for minimization problem."""
        # Set up mock model predictions
        mu = np.array([0.6, 0.4, 0.7])  # Predicted means
        sigma = np.array([0.2, 0.1, 0.3])  # Predicted std deviations
        self.model.predict.return_value = (mu, sigma)
        
        # Create PI acquisition function
        pi = ProbabilityOfImprovementNumpy(self.parameter_space)
        
        # Evaluate PI
        pi_values = pi.evaluate(self.X, self.model, self.best_f)
        
        # Check that PI values are between 0 and 1
        self.assertTrue(np.all(pi_values >= 0) and np.all(pi_values <= 1))
        
        # Check that the point with mu < best_f has higher PI
        self.assertTrue(pi_values[1] > pi_values[0])
        self.assertTrue(pi_values[1] > pi_values[2])

    def test_pi_maximization(self):
        """Test Probability of Improvement for maximization problem."""
        # Change to maximization objective
        self.parameter_space.objectives = [{"type": "maximize"}]
        
        # Set up mock model predictions
        mu = np.array([0.6, 0.7, 0.4])  # Predicted means
        sigma = np.array([0.2, 0.1, 0.3])  # Predicted std deviations
        self.model.predict.return_value = (mu, sigma)
        
        # Create PI acquisition function
        pi = ProbabilityOfImprovementNumpy(self.parameter_space)
        
        # Evaluate PI
        pi_values = pi.evaluate(self.X, self.model, self.best_f)
        
        # Check that PI values are between 0 and 1
        self.assertTrue(np.all(pi_values >= 0) and np.all(pi_values <= 1))
        
        # Check that the point with mu > best_f has higher PI
        self.assertTrue(pi_values[1] > pi_values[2])

    def test_ucb_minimization(self):
        """Test Upper Confidence Bound for minimization problem."""
        # Set up mock model predictions
        mu = np.array([0.5, 0.3, 0.7])  # Predicted means
        sigma = np.array([0.2, 0.1, 0.3])  # Predicted std deviations
        self.model.predict.return_value = (mu, sigma)
        
        # Create UCB acquisition function with kappa=2.0
        ucb = UpperConfidenceBoundNumpy(self.parameter_space, kappa=2.0)
        
        # Evaluate UCB (for minimization, lower is better: mu - kappa*sigma)
        ucb_values = ucb.evaluate(self.X, self.model)
        
        # Expected values: [0.5-2*0.2, 0.3-2*0.1, 0.7-2*0.3] = [0.1, 0.1, 0.1]
        # Check that ucb values match expected pattern
        # For minimization, we want smaller values of mu-kappa*sigma
        self.assertTrue(ucb_values[1] < ucb_values[0])  # 0.1 < 0.1
        self.assertTrue(ucb_values[1] < ucb_values[2])  # 0.1 < 0.1

    def test_ucb_maximization(self):
        """Test Upper Confidence Bound for maximization problem."""
        # Change to maximization objective
        self.parameter_space.objectives = [{"type": "maximize"}]
        
        # Set up mock model predictions
        mu = np.array([0.5, 0.7, 0.3])  # Predicted means
        sigma = np.array([0.2, 0.1, 0.3])  # Predicted std deviations
        self.model.predict.return_value = (mu, sigma)
        
        # Create UCB acquisition function with kappa=2.0
        ucb = UpperConfidenceBoundNumpy(self.parameter_space, kappa=2.0)
        
        # Evaluate UCB (for maximization, higher is better: mu + kappa*sigma)
        ucb_values = ucb.evaluate(self.X, self.model)
        
        # Expected values: [0.5+2*0.2, 0.7+2*0.1, 0.3+2*0.3] = [0.9, 0.9, 0.9]
        # For maximization, we want larger values of mu+kappa*sigma
        # Points with high mean or high uncertainty will have high UCB values
        self.assertTrue(ucb_values[0] >= 0.9)
        self.assertTrue(ucb_values[1] >= 0.9)
        self.assertTrue(ucb_values[2] >= 0.9)

    def test_ucb_parameter_update(self):
        """Test updating UCB parameters."""
        ucb = UpperConfidenceBoundNumpy(self.parameter_space, kappa=1.0)
        self.assertEqual(ucb.kappa, 1.0)
        
        # Update kappa parameter
        ucb.update_parameters(kappa=3.0)
        self.assertEqual(ucb.kappa, 3.0)

    def test_pi_parameter_update(self):
        """Test updating PI parameters."""
        pi = ProbabilityOfImprovementNumpy(self.parameter_space, xi=0.01)
        self.assertEqual(pi.xi, 0.01)
        
        # Update xi parameter
        pi.update_parameters(xi=0.05)
        self.assertEqual(pi.xi, 0.05)

    def test_model_not_initialized(self):
        """Test behavior when model is not initialized."""
        self.model.is_initialized = False
        
        # Create acquisition functions
        ei = ExpectedImprovementNumpy(self.parameter_space)
        pi = ProbabilityOfImprovementNumpy(self.parameter_space)
        ucb = UpperConfidenceBoundNumpy(self.parameter_space)
        
        # Evaluate with uninitialized model
        ei_values = ei.evaluate(self.X, self.model, self.best_f)
        pi_values = pi.evaluate(self.X, self.model, self.best_f)
        ucb_values = ucb.evaluate(self.X, self.model)
        
        # Check that all values are zeros
        self.assertTrue(np.all(ei_values == 0))
        self.assertTrue(np.all(pi_values == 0))
        self.assertTrue(np.all(ucb_values == 0))

    def test_model_no_uncertainty(self):
        """Test behavior when model provides no uncertainty."""
        # Model returns sigma=None
        mu = np.array([0.5, 0.3, 0.7])
        self.model.predict.return_value = (mu, None)
        
        # Create acquisition functions
        ei = ExpectedImprovementNumpy(self.parameter_space)
        pi = ProbabilityOfImprovementNumpy(self.parameter_space)
        ucb = UpperConfidenceBoundNumpy(self.parameter_space)
        
        # Evaluate
        ei_values = ei.evaluate(self.X, self.model, self.best_f)
        pi_values = pi.evaluate(self.X, self.model, self.best_f)
        ucb_values = ucb.evaluate(self.X, self.model)
        
        # Check results
        # EI should return zeros
        self.assertTrue(np.all(ei_values == 0))
        # PI should return zeros
        self.assertTrue(np.all(pi_values == 0))
        # UCB should return mean values
        np.testing.assert_array_equal(ucb_values, mu)


if __name__ == "__main__":
    unittest.main() 
