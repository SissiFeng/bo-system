import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel
from ..parameter_space import ParameterSpace

logger = logging.getLogger(__name__)

class GaussianProcessModel(BaseModel):
    """
    Surrogate model based on Gaussian Process Regression using scikit-learn.
    """
    def __init__(self, 
                 parameter_space: ParameterSpace, 
                 kernel: Optional[Any] = None, 
                 alpha: float = 1e-10, 
                 n_restarts_optimizer: int = 10, 
                 normalize_y: bool = True,
                 use_standard_scaler: bool = True, # Option to scale input features
                 **kwargs):
        """
        Initialize the Gaussian Process model.
        
        Args:
            parameter_space: The parameter space definition.
            kernel: The kernel specification. If None, a default Matern kernel is used.
                    See scikit-learn documentation for kernel options.
            alpha: Value added to the diagonal of the kernel matrix during fitting.
                   Larger values correspond to increased noise level assumption.
            n_restarts_optimizer: The number of restarts of the optimizer for finding
                                  the kernel's parameters which maximize the log-marginal-likelihood.
            normalize_y: Whether the target values y are normalized by removing the mean and scaling to unit-variance.
            use_standard_scaler: Whether to scale the input features X to zero mean and unit variance.
            **kwargs: Additional configuration arguments (currently unused but kept for flexibility).
        """
        super().__init__(parameter_space, **kwargs)
        
        if kernel is None:
            # Default kernel: Matern nu=2.5 + WhiteKernel for noise
            # length_scale bounds encourage exploring different smoothness levels
            length_scale_bounds = kwargs.get("length_scale_bounds", (1e-2, 1e2))
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=length_scale_bounds, nu=2.5) + WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-5, 1e1))
            logger.info(f"No kernel provided, using default Matern kernel: {kernel}")

        self.kernel = kernel
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.use_standard_scaler = use_standard_scaler
        
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            random_state=kwargs.get("random_state", None) # Allow setting random state
        )
        
        # Scaler for input features X
        self.scaler_X = StandardScaler() if self.use_standard_scaler else None
        self._trained = False # Reset trained status from superclass init

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the Gaussian Process model.
        
        Args:
            X: Observed points, shape (n_samples, n_dims) in the internal [0, 1] space.
            y: Observed objective values, shape (n_samples,). Assumes single objective.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim > 1 and y.shape[1] > 1:
            logger.warning(f"GPR currently supports only single-objective optimization. Received y shape: {y.shape}. Using only the first objective.")
            y = y[:, 0]
        y = y.ravel() # Ensure y is 1D

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) must match.")
        if X.shape[0] == 0:
            logger.warning("Cannot train GPR model with 0 samples.")
            self._trained = False
            return

        # Scale input features if enabled
        if self.scaler_X:
            logger.debug("Fitting StandardScaler for X and transforming X.")
            X_scaled = self.scaler_X.fit_transform(X)
        else:
            X_scaled = X
            
        try:
            logger.info(f"Training GPR model with {X.shape[0]} samples and {X.shape[1]} dimensions...")
            self.model.fit(X_scaled, y)
            self._trained = True
            logger.info("GPR model training completed.")
            logger.info(f"Optimized Kernel: {self.model.kernel_}")
            logger.info(f"Log-Marginal-Likelihood: {self.model.log_marginal_likelihood(self.model.kernel_.theta)}")
        except Exception as e:
            logger.error(f"GPR model training failed: {e}", exc_info=True)
            self._trained = False
            # Consider re-raising or handling specific exceptions if needed
            raise

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained GPR model.
        
        Args:
            X: Points to predict at, shape (n_points, n_dims) in the internal [0, 1] space.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - Mean prediction, shape (n_points,).
                - Variance prediction, shape (n_points,).
        """
        if not self.is_trained():
            raise RuntimeError("GPR Model must be trained before making predictions.")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Scale input features if scaler was fitted
        if self.scaler_X:
            try:
                X_scaled = self.scaler_X.transform(X)
            except Exception as e:
                 logger.error(f"Failed to transform prediction input X using fitted scaler: {e}", exc_info=True)
                 raise
        else:
            X_scaled = X
            
        try:
            # Use return_std=True and square the std dev to get variance
            mean, std_dev = self.model.predict(X_scaled, return_std=True)
            variance = std_dev**2
            # Ensure variance is non-negative (numerical issues might cause small negative values)
            variance = np.maximum(variance, 0)
            return mean.ravel(), variance.ravel()
        except Exception as e:
            logger.error(f"GPR prediction failed: {e}", exc_info=True)
            raise 
