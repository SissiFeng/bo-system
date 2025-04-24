import unittest
import numpy as np
import logging

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Mock or simplified ParameterSpace for testing
# In a real scenario, you might use a mock library or a more complex setup fixture
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

    def get_internal_dimensions(self):
        # Simplified: assume one dimension per parameter for testing
        return len(self.parameters)

    def from_dict(cls, config):
        # Mock class method
        return cls(config.get('parameters', []))

# Import the class to be tested AFTER setting up mocks if needed
try:
    from bo_engine.models.gaussian_process import GaussianProcessModel
except ImportError as e:
    logger.error(f"Failed to import GaussianProcessModel: {e}")
    GaussianProcessModel = None # Allow tests to be skipped if import fails

@unittest.skipIf(GaussianProcessModel is None, "GaussianProcessModel could not be imported")
class TestGaussianProcessModel(unittest.TestCase):

    def setUp(self):
        """Set up a simple parameter space and data for tests."""
        self.parameters_config = [
            {'name': 'x1', 'type': 'continuous', 'min': 0, 'max': 1},
            {'name': 'x2', 'type': 'continuous', 'min': 0, 'max': 1}
        ]
        self.parameter_space = MockParameterSpace(self.parameters_config)
        
        # Sample data (in internal [0, 1] space)
        self.X_train = np.random.rand(10, 2) 
        self.y_train = np.sin(self.X_train[:, 0] * 2 * np.pi) + np.cos(self.X_train[:, 1] * np.pi) + np.random.randn(10) * 0.1
        self.y_train = self.y_train.ravel()
        
        self.X_test = np.random.rand(5, 2)

        logger.debug("Test setup complete.")

    def test_initialization_default_kernel(self):
        """Test GPR model initialization with default settings."""
        logger.info("Running test: test_initialization_default_kernel")
        try:
            model = GaussianProcessModel(parameter_space=self.parameter_space)
            self.assertIsNotNone(model.model, "Scikit-learn GPR model should be initialized")
            self.assertFalse(model.is_trained(), "Model should not be trained initially")
            self.assertIsNotNone(model.kernel, "Default kernel should be set")
        except Exception as e:
            self.fail(f"Initialization failed with default kernel: {e}")
        logger.info("Finished test: test_initialization_default_kernel")

    def test_train_basic(self):
        """Test basic model training."""
        logger.info("Running test: test_train_basic")
        model = GaussianProcessModel(parameter_space=self.parameter_space, n_restarts_optimizer=0) # Faster training
        try:
            model.train(self.X_train, self.y_train)
            self.assertTrue(model.is_trained(), "Model should be marked as trained after training")
        except Exception as e:
            self.fail(f"model.train() failed: {e}")
        logger.info("Finished test: test_train_basic")
        
    def test_train_empty_data(self):
        """Test training with empty data."""
        logger.info("Running test: test_train_empty_data")
        model = GaussianProcessModel(parameter_space=self.parameter_space)
        with self.assertLogs(logger='bo_engine.models.gaussian_process', level='WARNING') as cm:
            model.train(np.array([]).reshape(0,2), np.array([]))
        self.assertFalse(model.is_trained(), "Model should not be trained with empty data")
        self.assertTrue(any("Cannot train GPR model with 0 samples" in msg for msg in cm.output),
                        "Warning log for empty data not found")
        logger.info("Finished test: test_train_empty_data")

    def test_predict_before_train(self):
        """Test calling predict before training."""
        logger.info("Running test: test_predict_before_train")
        model = GaussianProcessModel(parameter_space=self.parameter_space)
        with self.assertRaises(RuntimeError):
            model.predict(self.X_test)
        logger.info("Finished test: test_predict_before_train")

    def test_predict_after_train(self):
        """Test prediction after training."""
        logger.info("Running test: test_predict_after_train")
        model = GaussianProcessModel(parameter_space=self.parameter_space, n_restarts_optimizer=0)
        try:
            model.train(self.X_train, self.y_train)
            mean, variance = model.predict(self.X_test)
            
            self.assertEqual(mean.shape, (self.X_test.shape[0],), "Mean prediction shape mismatch")
            self.assertEqual(variance.shape, (self.X_test.shape[0],), "Variance prediction shape mismatch")
            self.assertTrue(np.all(variance >= 0), "Variance should be non-negative")
        except Exception as e:
            self.fail(f"model.predict() failed after training: {e}")
        logger.info("Finished test: test_predict_after_train")
        
    def test_predict_with_scaler(self):
        """Test prediction when using StandardScaler."""
        logger.info("Running test: test_predict_with_scaler")
        # Explicitly enable scaler (it's default, but good to be clear)
        model = GaussianProcessModel(parameter_space=self.parameter_space, use_standard_scaler=True, n_restarts_optimizer=0)
        try:
            model.train(self.X_train, self.y_train)
            self.assertIsNotNone(model.scaler_X, "Scaler should be initialized")
            mean, variance = model.predict(self.X_test)
            
            self.assertEqual(mean.shape, (self.X_test.shape[0],))
            self.assertEqual(variance.shape, (self.X_test.shape[0],))
            self.assertTrue(np.all(variance >= 0))
        except Exception as e:
            self.fail(f"Prediction with scaler failed: {e}")
        logger.info("Finished test: test_predict_with_scaler")
        
    def test_predict_without_scaler(self):
        """Test prediction when StandardScaler is disabled."""
        logger.info("Running test: test_predict_without_scaler")
        model = GaussianProcessModel(parameter_space=self.parameter_space, use_standard_scaler=False, n_restarts_optimizer=0)
        try:
            model.train(self.X_train, self.y_train)
            self.assertIsNone(model.scaler_X, "Scaler should not be initialized")
            mean, variance = model.predict(self.X_test)
            
            self.assertEqual(mean.shape, (self.X_test.shape[0],))
            self.assertEqual(variance.shape, (self.X_test.shape[0],))
            self.assertTrue(np.all(variance >= 0))
        except Exception as e:
            self.fail(f"Prediction without scaler failed: {e}")
        logger.info("Finished test: test_predict_without_scaler")

if __name__ == '__main__':
    unittest.main() 
