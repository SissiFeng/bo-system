import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bo_engine.acquisition.ucb_numpy import UpperConfidenceBoundNumpy
from bo_engine.parameter_space import ParameterSpace


class TestUpperConfidenceBoundNumpy(unittest.TestCase):
    """Test cases for the Upper Confidence Bound acquisition function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock parameter space with one objective (maximization)
        self.parameter_space = MagicMock(spec=ParameterSpace)
        self.parameter_space.objectives = [{"type": "maximize"}]
        
        # Create UCB instance
        self.ucb = UpperConfidenceBoundNumpy(
            parameter_space=self.parameter_space,
            kappa=2.0
        )
        
        # Create a mock model
        self.model = MagicMock()
        self.model.is_initialized = True

    def test_initialization(self):
        """Test initialization of the UCB acquisition function."""
        self.assertEqual(self.ucb.kappa, 2.0)
        self.assertEqual(self.ucb.iteration, 0)

    def test_evaluate_maximization(self):
        """Test UCB evaluation for maximization problem."""
        # Mock data and model predictions
        X = np.array([[0.5, 0.5], [0.2, 0.8]])
        
        # For maximization, UCB = mu + kappa*sigma
        mu = np.array([1.0, 2.0])
        sigma = np.array([0.5, 0.3])
        expected_ucb = mu + self.ucb.kappa * sigma  # [2.0, 2.6]
        
        # Configure model mock to return mu and sigma
        self.model.predict.return_value = (mu, sigma)
        
        # Calculate UCB values
        ucb_values = self.ucb.evaluate(X, self.model)
        
        # Check correct calculation of UCB values
        np.testing.assert_almost_equal(ucb_values, expected_ucb)

    def test_evaluate_minimization(self):
        """Test UCB evaluation for minimization problem."""
        # Set parameter space for minimization
        self.parameter_space.objectives = [{"type": "minimize"}]
        
        # Mock data and model predictions
        X = np.array([[0.5, 0.5], [0.2, 0.8]])
        
        # For minimization, UCB = -(mu + kappa*sigma)
        mu = np.array([1.0, 2.0])
        sigma = np.array([0.5, 0.3])
        expected_ucb = -(mu + self.ucb.kappa * sigma)  # [-2.0, -2.6]
        
        # Configure model mock to return mu and sigma
        self.model.predict.return_value = (mu, sigma)
        
        # Calculate UCB values
        ucb_values = self.ucb.evaluate(X, self.model)
        
        # Check correct calculation of UCB values
        np.testing.assert_almost_equal(ucb_values, expected_ucb)

    def test_model_not_initialized(self):
        """Test UCB behavior when model is not initialized."""
        # Set model as not initialized
        self.model.is_initialized = False
        
        # Mock input data
        X = np.array([[0.5, 0.5], [0.2, 0.8]])
        
        # UCB should return zeros when model is not initialized
        ucb_values = self.ucb.evaluate(X, self.model)
        
        # Check correct output
        np.testing.assert_almost_equal(ucb_values, np.zeros(X.shape[0]))

    def test_model_no_uncertainty(self):
        """Test UCB behavior when model does not provide uncertainty."""
        # Mock data and model predictions (without uncertainty)
        X = np.array([[0.5, 0.5], [0.2, 0.8]])
        mu = np.array([1.0, 2.0])
        
        # Configure model mock to return mu and None for sigma
        self.model.predict.return_value = (mu, None)
        
        # UCB should return just the means when sigma is None
        ucb_values = self.ucb.evaluate(X, self.model)
        
        # Check correct output
        np.testing.assert_almost_equal(ucb_values, mu)

    def test_update_kappa(self):
        """Test updating kappa based on iteration count."""
        # Initial kappa value is 2.0
        self.assertEqual(self.ucb.kappa, 2.0)
        
        # Test update_kappa using explicit parameters
        self.ucb.update_kappa(iteration=5, dim=2)
        
        # Kappa should change based on the formula
        self.assertNotEqual(self.ucb.kappa, 2.0)
        self.assertGreater(self.ucb.kappa, 0)
        
        # Test update_kappa using internal counter
        initial_kappa = self.ucb.kappa
        self.ucb.update_kappa()
        
        # Kappa should update again and iteration counter should increment
        self.assertNotEqual(self.ucb.kappa, initial_kappa)
        self.assertEqual(self.ucb.iteration, 6)


if __name__ == '__main__':
    unittest.main() 
