import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch
from scipy.stats import norm

# Add the parent directory to sys.path to import the module being tested
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bo_engine.acquisition.pi_numpy import ProbabilityOfImprovementNumpy


class TestProbabilityOfImprovementNumpy(unittest.TestCase):
    """Test cases for the Probability of Improvement acquisition function."""

    def setUp(self):
        """Set up test fixtures."""
        self.parameter_space = MagicMock()
        self.parameter_space.objectives = [{"type": "minimize"}]  # Default to minimization
        self.xi = 0.01
        self.pi = ProbabilityOfImprovementNumpy(self.parameter_space, xi=self.xi)

    def test_initialization(self):
        """Test correct initialization of PI."""
        self.assertEqual(self.pi.xi, self.xi)
        self.assertEqual(self.pi.parameter_space, self.parameter_space)

    def test_evaluate_minimization(self):
        """Test PI evaluation for minimization problems."""
        X = np.array([[0.5, 0.5], [0.7, 0.3]])
        model = MagicMock()
        model.is_initialized = True
        
        # Mock model predictions
        mu = np.array([0.2, 0.3])
        sigma = np.array([0.1, 0.2])
        model.predict.return_value = (mu, sigma)
        
        # Current best value
        best_f = 0.1
        
        # Expected PI values (calculated manually)
        z_values = (best_f - self.xi - mu) / sigma
        expected_pi = norm.cdf(z_values)
        
        # Get actual PI values
        actual_pi = self.pi.evaluate(X, model, best_f)
        
        # Check results
        np.testing.assert_allclose(actual_pi, expected_pi)
        model.predict.assert_called_once_with(X)

    def test_evaluate_maximization(self):
        """Test PI evaluation for maximization problems."""
        # Change objective to maximization
        self.parameter_space.objectives = [{"type": "maximize"}]
        
        X = np.array([[0.5, 0.5], [0.7, 0.3]])
        model = MagicMock()
        model.is_initialized = True
        
        # Mock model predictions
        mu = np.array([0.8, 0.9])
        sigma = np.array([0.1, 0.2])
        model.predict.return_value = (mu, sigma)
        
        # Current best value
        best_f = 0.7
        
        # Expected PI values (calculated manually)
        z_values = (mu - best_f - self.xi) / sigma
        expected_pi = norm.cdf(z_values)
        
        # Get actual PI values
        actual_pi = self.pi.evaluate(X, model, best_f)
        
        # Check results
        np.testing.assert_allclose(actual_pi, expected_pi)

    def test_model_not_initialized(self):
        """Test PI behavior when model is not initialized."""
        X = np.array([[0.5, 0.5], [0.7, 0.3]])
        model = MagicMock()
        model.is_initialized = False
        
        # PI should return zeros when model is not initialized
        actual_pi = self.pi.evaluate(X, model)
        expected_pi = np.zeros(X.shape[0])
        
        np.testing.assert_array_equal(actual_pi, expected_pi)
        model.predict.assert_not_called()

    def test_model_no_uncertainty(self):
        """Test PI behavior when model does not provide uncertainty."""
        X = np.array([[0.5, 0.5], [0.7, 0.3]])
        model = MagicMock()
        model.is_initialized = True
        
        # Mock model predictions with no uncertainty (sigma=None)
        mu = np.array([0.2, 0.3])
        model.predict.return_value = (mu, None)
        
        # PI should return mean predictions when no uncertainty is provided
        actual_pi = self.pi.evaluate(X, model)
        
        np.testing.assert_array_equal(actual_pi, mu)

    def test_update_parameters(self):
        """Test updating PI parameters."""
        # Initial xi value
        self.assertEqual(self.pi.xi, 0.01)
        
        # Update xi
        new_xi = 0.05
        self.pi.update_parameters(xi=new_xi)
        
        # Check updated value
        self.assertEqual(self.pi.xi, new_xi)


if __name__ == '__main__':
    unittest.main() 
