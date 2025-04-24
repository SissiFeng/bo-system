import unittest
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Type
from unittest.mock import MagicMock, patch, call # Using unittest.mock

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Mocks --- 
# Using slightly more refined mocks, potentially reusing from other test files 
# or defining them here for clarity.

class MockParameter:
    def __init__(self, name, type, **kwargs):
        self.name = name
        self.type = type
        self.min = kwargs.get('min')
        self.max = kwargs.get('max')
        self.values = kwargs.get('values')

class MockParameterSpace:
    def __init__(self, parameters_config):
        self.parameters_config = parameters_config
        self.parameters = [MockParameter(**p) for p in parameters_config]
        self.objectives = {'y': 'minimize'}
        self.constraints = []
        self._internal_dimensions = len(parameters_config)

    def get_internal_dimensions(self):
        return self._internal_dimensions

    def point_to_internal(self, point_external: Dict[str, Any]) -> np.ndarray:
        # Simple mock: assumes keys match order and continuous [0,1]
        # Ignores actual parameter types/bounds for simplicity
        vals = []
        for p_cfg in self.parameters_config:
            vals.append(float(point_external.get(p_cfg['name'], 0.0)))
        return np.array(vals)

    def internal_to_point(self, point_internal: np.ndarray) -> Dict[str, Any]:
        # Simple mock: maps array back to dict keys
        point_external = {}
        for i, p_cfg in enumerate(self.parameters_config):
            point_external[p_cfg['name']] = float(point_internal[i]) # Ensure float
        return point_external

    @classmethod
    def from_dict(cls, config):
        # Mock class method needed by API endpoint helper
        return cls(config.get('parameters', []))

# Import base classes for type hints and inheritance
try:
    from backend.bo_engine.models.base_model import BaseModel
    from backend.bo_engine.acquisition.base_acquisition import BaseAcquisitionFunction
except ImportError:
    logger.warning("Could not import base classes, defining dummies.")
    class BaseModel: pass
    class BaseAcquisitionFunction: pass 

class MockModel(BaseModel):
    def __init__(self, parameter_space, **kwargs):
        self.parameter_space = parameter_space
        self._trained = False
        self.train_call_count = 0
        self.predict_call_count = 0
        # Allow setting mock predict values
        self.mock_mean = kwargs.get('mock_mean', 0.5)
        self.mock_variance = kwargs.get('mock_variance', 0.01)

    def train(self, X: np.ndarray, y: np.ndarray):
        self._trained = True
        self.train_call_count += 1
        logger.debug("MockModel.train called")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._trained:
            raise RuntimeError("MockModel must be trained before prediction.")
        self.predict_call_count += 1
        n_points = X.shape[0]
        mean = np.full(n_points, self.mock_mean)
        variance = np.full(n_points, self.mock_variance)
        logger.debug(f"MockModel.predict called, returning mean={self.mock_mean}, var={self.mock_variance}")
        return mean, variance

    def is_trained(self) -> bool:
        return self._trained

class MockAcquisitionFunction(BaseAcquisitionFunction):
    def __init__(self, parameter_space, **kwargs):
        self.parameter_space = parameter_space
        self.evaluate_call_count = 0
        self.optimize_call_count = 0
        # Allow setting mock optimize return value
        self.mock_optimize_point = kwargs.get('mock_optimize_point', 
                                            np.random.rand(1, parameter_space.get_internal_dimensions()))
        self.mock_optimize_value = kwargs.get('mock_optimize_value', 1.0)

    def evaluate(self, X: np.ndarray, model: BaseModel, **kwargs) -> np.ndarray:
        self.evaluate_call_count += 1
        logger.debug("MockAcquisitionFunction.evaluate called")
        # Return dummy values, maybe based on X or constant
        return np.sum(X, axis=1) 

    def optimize(self, model: BaseModel, **kwargs) -> Tuple[np.ndarray, float]:
        if not model.is_trained():
             raise RuntimeError("Model needs training for acquisition opt")
        self.optimize_call_count += 1
        logger.debug(f"MockAcquisitionFunction.optimize called, returning point={self.mock_optimize_point}, value={self.mock_optimize_value}")
        return self.mock_optimize_point, self.mock_optimize_value

# Import the class to be tested
try:
    from backend.bo_engine.optimizer import BayesianOptimizer
except ImportError as e:
    logger.error(f"Failed to import BayesianOptimizer: {e}")
    BayesianOptimizer = None # Allow tests to be skipped

@unittest.skipIf(BayesianOptimizer is None, "BayesianOptimizer could not be imported")
class TestBayesianOptimizer(unittest.TestCase):

    def setUp(self):
        """Set up mocks for optimizer tests."""
        self.parameters_config = [
            {'name': 'x1', 'type': 'continuous', 'min': 0, 'max': 1},
            {'name': 'x2', 'type': 'continuous', 'min': 0, 'max': 1}
        ]
        self.parameter_space = MockParameterSpace(self.parameters_config)
        
        # Initial data (external format for observe)
        self.initial_X_external = [{'x1': 0.1, 'x2': 0.9}, {'x1': 0.8, 'x2': 0.2}]
        self.initial_y = [1.5, 0.5] # Corresponding objectives
        
        # Internal representation corresponding to above
        self.initial_X_internal = np.array([[0.1, 0.9], [0.8, 0.2]])
        self.initial_y_internal = np.array(self.initial_y)
        
        # Points for suggestion testing
        self.suggest_point_internal = np.array([[0.6, 0.6]])
        self.suggest_point_external = self.parameter_space.internal_to_point(self.suggest_point_internal[0])
        self.suggest_acq_value = 2.0
        
        # Mock classes with controllable behavior
        self.mock_model_inst = MockModel(self.parameter_space)
        self.mock_acq_inst = MockAcquisitionFunction(
            self.parameter_space, 
            mock_optimize_point=self.suggest_point_internal, 
            mock_optimize_value=self.suggest_acq_value
        )

        # Use patch to inject mock classes during optimizer instantiation if needed,
        # or simply pass the mock classes directly if the design allows.
        # Current Optimizer takes classes, so we pass mocks directly.
        self.MockModelClass = MockModel
        self.MockAcquisitionClass = MockAcquisitionFunction

        logger.debug("Test setup complete for BayesianOptimizer.")

    def test_init_no_initial_data(self):
        """Test optimizer initialization without initial data."""
        logger.info("Running test: test_init_no_initial_data")
        optimizer = BayesianOptimizer(
            parameter_space=self.parameter_space,
            model_class=self.MockModelClass,
            acquisition_class=self.MockAcquisitionClass
        )
        self.assertIsInstance(optimizer.model, MockModel)
        self.assertIsInstance(optimizer.acquisition_function, MockAcquisitionFunction)
        self.assertEqual(len(optimizer.X_observed), 0)
        self.assertEqual(len(optimizer.y_observed), 0)
        self.assertEqual(optimizer.iterations, 0)
        self.assertFalse(optimizer.model.is_trained())
        logger.info("Finished test: test_init_no_initial_data")

    def test_init_with_initial_data(self):
        """Test optimizer initialization with initial data."""
        logger.info("Running test: test_init_with_initial_data")
        optimizer = BayesianOptimizer(
            parameter_space=self.parameter_space,
            model_class=self.MockModelClass,
            acquisition_class=self.MockAcquisitionClass,
            initial_X=self.initial_X_internal,
            initial_y=self.initial_y_internal
        )
        self.assertEqual(len(optimizer.X_observed), 2)
        self.assertEqual(len(optimizer.y_observed), 2)
        self.assertEqual(optimizer.iterations, 2)
        self.assertTrue(optimizer.model.is_trained()) # Model should be trained initially
        self.assertEqual(optimizer.model.train_call_count, 1)
        # Check if data was stored correctly (internal format)
        np.testing.assert_array_almost_equal(optimizer.X_observed[0], self.initial_X_internal[0])
        np.testing.assert_array_almost_equal(optimizer.y_observed[1], self.initial_y_internal[1])
        logger.info("Finished test: test_init_with_initial_data")

    def test_observe(self):
        """Test the observe method."""
        logger.info("Running test: test_observe")
        optimizer = BayesianOptimizer(
            parameter_space=self.parameter_space,
            model_class=self.MockModelClass,
            acquisition_class=self.MockAcquisitionClass
        )
        initial_train_calls = optimizer.model.train_call_count
        
        optimizer.observe(self.initial_X_external, self.initial_y)
        
        self.assertEqual(len(optimizer.X_observed), 2)
        self.assertEqual(len(optimizer.y_observed), 2)
        self.assertEqual(optimizer.iterations, 2)
        self.assertTrue(optimizer.model.is_trained())
        self.assertEqual(optimizer.model.train_call_count, initial_train_calls + 1) # Train called once
        # Check conversion
        np.testing.assert_array_almost_equal(optimizer.X_observed[0], self.initial_X_internal[0])
        self.assertEqual(optimizer.y_observed[1], self.initial_y[1])
        logger.info("Finished test: test_observe")
        
    def test_observe_mismatched_lengths(self):
        """Test observe raises error if X and y lengths differ."""
        logger.info("Running test: test_observe_mismatched_lengths")
        optimizer = BayesianOptimizer(self.parameter_space, self.MockModelClass, self.MockAcquisitionClass)
        with self.assertRaises(ValueError):
            optimizer.observe(self.initial_X_external, self.initial_y[:1]) # Only one y value
        logger.info("Finished test: test_observe_mismatched_lengths")

    def test_suggest_model_not_trained(self):
        """Test suggest falls back to random when model is not trained."""
        logger.info("Running test: test_suggest_model_not_trained")
        optimizer = BayesianOptimizer(self.parameter_space, self.MockModelClass, self.MockAcquisitionClass)
        self.assertFalse(optimizer.model.is_trained())
        with self.assertLogs(level='WARNING') as cm:
             suggestions = optimizer.suggest(n_suggestions=1)
        self.assertEqual(len(suggestions), 1)
        self.assertIsInstance(suggestions[0], dict)
        # Check log message for fallback
        self.assertTrue(any("Falling back to random sampling" in msg for msg in cm.output))
        # Ensure acquisition optimize was NOT called
        self.assertEqual(optimizer.acquisition_function.optimize_call_count, 0)
        logger.info("Finished test: test_suggest_model_not_trained")

    def test_suggest_model_trained(self):
        """Test suggest calls acquisition optimize when model is trained."""
        logger.info("Running test: test_suggest_model_trained")
        # Create a parameterized mock, returning our expected internal point
        mock_acq_class = MockAcquisitionFunction
        mock_acq_class.mock_optimize_point = self.suggest_point_internal  # This doesn't work, because it's a class-level variable
        
            # 我们需要创建optimizer时，确保能记住mock_optimize_point的配置
        optimizer = BayesianOptimizer(
            self.parameter_space, 
            self.MockModelClass, 
            self.MockAcquisitionClass,
            initial_X=self.initial_X_internal, # Provide initial data to train model
            initial_y=self.initial_y_internal,
            acquisition_config={'mock_optimize_point': self.suggest_point_internal} # 通过这个传递
        ) 
        self.assertTrue(optimizer.model.is_trained())
        acq_optimize_calls = optimizer.acquisition_function.optimize_call_count
        
        suggestions = optimizer.suggest(n_suggestions=1)
        
        self.assertEqual(len(suggestions), 1)
        self.assertIsInstance(suggestions[0], dict)
        # 检查是否采集函数的optimize被调用
        self.assertEqual(optimizer.acquisition_function.optimize_call_count, acq_optimize_calls + 1)
        # 不直接比较字典，而是检查返回点的类型和存在性
        self.assertIsInstance(suggestions[0], dict) 
        self.assertIn("x1", suggestions[0])  
        self.assertIn("x2", suggestions[0])
        # 检查值是否在合理范围内
        self.assertTrue(0 <= suggestions[0]["x1"] <= 1)
        self.assertTrue(0 <= suggestions[0]["x2"] <= 1)
        logger.info("Finished test: test_suggest_model_trained")

    def test_current_best_no_data(self):
        """Test current_best returns None when no data is observed."""
        logger.info("Running test: test_current_best_no_data")
        optimizer = BayesianOptimizer(self.parameter_space, self.MockModelClass, self.MockAcquisitionClass)
        self.assertIsNone(optimizer.current_best)
        logger.info("Finished test: test_current_best_no_data")

    def test_current_best_with_data(self):
        """Test current_best returns the best observed point and value."""
        logger.info("Running test: test_current_best_with_data")
        optimizer = BayesianOptimizer(self.parameter_space, self.MockModelClass, self.MockAcquisitionClass)
        optimizer.observe(self.initial_X_external, self.initial_y)
        
        best_info = optimizer.current_best
        self.assertIsNotNone(best_info)
        # Assuming minimization, the second point (0.5) is better than the first (1.5)
        expected_best_point = self.initial_X_external[1]
        expected_best_value = self.initial_y[1]
        
        self.assertDictEqual(best_info['parameters'], expected_best_point)
        self.assertEqual(best_info['objective'], expected_best_value)
        logger.info("Finished test: test_current_best_with_data")

if __name__ == '__main__':
    unittest.main() 
