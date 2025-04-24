import unittest
import numpy as np
import logging
from typing import Tuple

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Mock ParameterSpace (can be shared or redefined)
class MockParameter:
    def __init__(self, name, type, **kwargs):
        self.name = name
        self.type = type
        self.min = kwargs.get('min')
        self.max = kwargs.get('max')
        self.values = kwargs.get('values')

class MockParameterSpace:
    def __init__(self, parameters):
        self.parameters = [MockParameter(**p) for p in parameters]
        self.objectives = {'y': 'minimize'}
        self.constraints = []
        self._internal_dimensions = len(parameters)

    def get_internal_dimensions(self):
        return self._internal_dimensions

    def from_dict(cls, config):
        return cls(config.get('parameters', []))

# Mock Model
# Need to import BaseModel to inherit from it
try:
    from bo_engine.models.base_model import BaseModel
except ImportError:
    # Define a dummy BaseModel if the real one can't be imported
    class BaseModel:
        def __init__(self, parameter_space, **kwargs):
            self._trained = False
        def train(self, X, y):
            self._trained = True
        def predict(self, X):
            raise NotImplementedError
        def is_trained(self):
            return self._trained

class MockModel(BaseModel):
    def __init__(self, parameter_space, mean_val=0.5, variance_val=0.1, **kwargs):
        super().__init__(parameter_space, **kwargs)
        self.mean_val = mean_val
        self.variance_val = variance_val
        self._trained = False # Start as untrained

    def train(self, X: np.ndarray, y: np.ndarray):
        # Mock training just marks the model as trained
        self._trained = True
        logger.debug("MockModel marked as trained.")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained():
            raise RuntimeError("MockModel must be trained before prediction.")
        n_points = X.shape[0]
        # Return constant mean and variance for simplicity
        mean = np.full(n_points, self.mean_val)
        variance = np.full(n_points, self.variance_val)
        logger.debug(f"MockModel predicting mean={self.mean_val}, variance={self.variance_val} for {n_points} points.")
        return mean, variance

# Import the class to be tested
try:
    from bo_engine.acquisition.expected_improvement import ExpectedImprovement
except ImportError as e:
    logger.error(f"Failed to import ExpectedImprovement: {e}")
    ExpectedImprovement = None # Allow tests to be skipped

@unittest.skipIf(ExpectedImprovement is None, "ExpectedImprovement could not be imported")
class TestExpectedImprovement(unittest.TestCase):

    def setUp(self):
        """Set up mocks for testing."""
        self.parameters_config = [
            {'name': 'x1', 'type': 'continuous', 'min': 0, 'max': 1}
        ]
        self.parameter_space = MockParameterSpace(self.parameters_config)
        self.X_test = np.random.rand(5, 1) # Test points in [0, 1] space
        self.mock_model = MockModel(self.parameter_space)
        # Mark the mock model as trained for most tests
        self.mock_model.train(np.array([[0.5]]), np.array([0])) 
        logger.debug("Test setup complete for ExpectedImprovement.")

    def test_initialization(self):
        """Test EI initialization with default and custom xi."""
        logger.info("Running test: test_initialization")
        ei_default = ExpectedImprovement(self.parameter_space)
        self.assertEqual(ei_default.xi, 0.01, "Default xi should be 0.01")
        
        ei_custom = ExpectedImprovement(self.parameter_space, xi=0.1)
        self.assertEqual(ei_custom.xi, 0.1, "Custom xi should be set correctly")
        logger.info("Finished test: test_initialization")

    def test_evaluate_no_best_f(self):
        """Test evaluate returns zeros if best_f is not provided."""
        logger.info("Running test: test_evaluate_no_best_f")
        ei = ExpectedImprovement(self.parameter_space)
        with self.assertLogs(logger='bo_engine.acquisition.expected_improvement', level='WARNING') as cm:
            result = ei.evaluate(self.X_test, self.mock_model, best_f=None)
        self.assertTrue(np.all(result == 0), "Evaluate should return zeros when best_f is None")
        self.assertTrue(any("best_f not provided" in msg for msg in cm.output),
                        "Warning log for missing best_f not found")
        logger.info("Finished test: test_evaluate_no_best_f")
        
    def test_evaluate_model_not_trained(self):
        """Test evaluate returns zeros if model is not trained."""
        logger.info("Running test: test_evaluate_model_not_trained")
        ei = ExpectedImprovement(self.parameter_space)
        untrained_model = MockModel(self.parameter_space) # Create a new untrained model
        self.assertFalse(untrained_model.is_trained())
        with self.assertLogs(logger='bo_engine.acquisition.expected_improvement', level='WARNING') as cm:
             result = ei.evaluate(self.X_test, untrained_model, best_f=0.5)
        self.assertTrue(np.all(result == 0), "Evaluate should return zeros when model is untrained")
        self.assertTrue(any("Model is not trained" in msg for msg in cm.output),
                        "Warning log for untrained model not found")
        logger.info("Finished test: test_evaluate_model_not_trained")

    def test_evaluate_zero_variance(self):
        """Test evaluate returns zero EI when variance is zero."""
        logger.info("Running test: test_evaluate_zero_variance")
        zero_var_model = MockModel(self.parameter_space, variance_val=0.0)
        zero_var_model.train(np.array([[0.5]]), np.array([0])) # Mark as trained
        ei = ExpectedImprovement(self.parameter_space)
        result = ei.evaluate(self.X_test, zero_var_model, best_f=0.5)
        self.assertTrue(np.all(result == 0), "Evaluate should return zeros when variance is zero")
        logger.info("Finished test: test_evaluate_zero_variance")

    def test_evaluate_positive_ei(self):
        """Test evaluate calculates positive EI correctly."""
        logger.info("Running test: test_evaluate_positive_ei")
        # Setup: mean prediction lower than best_f, non-zero variance
        current_best = 0.5
        predict_mean = 0.3
        predict_var = 0.01 # std_dev = 0.1
        xi = 0.0 # Simplify calculation
        
        model = MockModel(self.parameter_space, mean_val=predict_mean, variance_val=predict_var)
        model.train(np.array([[0.5]]), np.array([0])) # Mark as trained
        ei_calc = ExpectedImprovement(self.parameter_space, xi=xi)
        result = ei_calc.evaluate(self.X_test[0:1,:], model, best_f=current_best) # Evaluate single point
        
        # Manual calculation for verification
        imp = current_best - predict_mean - xi
        std_dev = np.sqrt(predict_var)
        Z = imp / std_dev
        from scipy.stats import norm
        expected_ei = imp * norm.cdf(Z) + std_dev * norm.pdf(Z)
        
        self.assertAlmostEqual(result[0], expected_ei, places=6, msg="Calculated EI does not match expected value")
        self.assertGreater(result[0], 0, "EI value should be positive")
        logger.info("Finished test: test_evaluate_positive_ei")
        
    def test_evaluate_zero_ei_due_to_no_improvement(self):
        """Test evaluate gives zero EI when predicted mean >= best_f."""
        logger.info("Running test: test_evaluate_zero_ei_due_to_no_improvement")
        # Setup: mean prediction higher than best_f
        current_best = 0.5
        predict_mean = 0.6 
        predict_var = 0.01
        xi = 0.01
        
        model = MockModel(self.parameter_space, mean_val=predict_mean, variance_val=predict_var)
        model.train(np.array([[0.5]]), np.array([0]))
        ei_calc = ExpectedImprovement(self.parameter_space, xi=xi)
        result = ei_calc.evaluate(self.X_test, model, best_f=current_best)
        
        self.assertTrue(np.all(result == 0), msg="EI should be zero when there is no expected improvement (mean+xi > best_f)")
        logger.info("Finished test: test_evaluate_zero_ei_due_to_no_improvement")
        
    def test_optimize_basic(self):
        """Test the basic functionality of the optimize method."""
        logger.info("Running test: test_optimize_basic")
        ei = ExpectedImprovement(self.parameter_space, xi=0.0)
        current_best = 0.5
        # Use a model where lower mean is better (matches EI formula)
        model = MockModel(self.parameter_space, mean_val=0.3, variance_val=0.01)
        model.train(np.array([[0.5]]), np.array([0])) 

        try:
            # Reduce restarts/samples for faster test
            best_x, max_ei = ei.optimize(model, best_f=current_best, n_restarts=2, raw_samples=50)
            self.assertEqual(best_x.shape, (1, self.parameter_space.get_internal_dimensions()), "Optimized x shape mismatch")
            self.assertTrue(np.all(best_x >= 0) and np.all(best_x <= 1), "Optimized x should be within [0, 1] bounds")
            self.assertIsInstance(max_ei, float, "Max EI value should be a float")
            self.assertGreaterEqual(max_ei, 0, "Max EI value should be non-negative")
        except Exception as e:
            self.fail(f"ei.optimize() failed: {e}")
        logger.info("Finished test: test_optimize_basic")
        
    def test_optimize_no_best_f(self):
        """Test optimize falls back to random if best_f is None."""
        logger.info("Running test: test_optimize_no_best_f")
        ei = ExpectedImprovement(self.parameter_space)
        with self.assertLogs(logger='bo_engine.acquisition.expected_improvement', level='ERROR') as cm:
            best_x, max_ei = ei.optimize(self.mock_model, best_f=None)
        self.assertEqual(best_x.shape, (1, self.parameter_space.get_internal_dimensions()))
        self.assertTrue(np.all(best_x >= 0) and np.all(best_x <= 1))
        self.assertEqual(max_ei, 0.0, "Max EI should be 0.0 when falling back")
        self.assertTrue(any("Cannot optimize EI without best_f" in msg for msg in cm.output))
        logger.info("Finished test: test_optimize_no_best_f")
        
    def test_optimize_model_not_trained(self):
        """Test optimize falls back to random if model is not trained."""
        logger.info("Running test: test_optimize_model_not_trained")
        ei = ExpectedImprovement(self.parameter_space)
        untrained_model = MockModel(self.parameter_space)
        with self.assertLogs(logger='bo_engine.acquisition.expected_improvement', level='ERROR') as cm:
            best_x, max_ei = ei.optimize(untrained_model, best_f=0.5)
        self.assertEqual(best_x.shape, (1, self.parameter_space.get_internal_dimensions()))
        self.assertTrue(np.all(best_x >= 0) and np.all(best_x <= 1))
        self.assertEqual(max_ei, 0.0)
        self.assertTrue(any("Cannot optimize EI when model is not trained" in msg for msg in cm.output))
        logger.info("Finished test: test_optimize_model_not_trained")

if __name__ == '__main__':
    unittest.main() 
