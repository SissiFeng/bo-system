import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import logging
from typing import Tuple, Dict, Any, Optional

from bo_engine.acquisition.base_acquisition import BaseAcquisitionFunction
from bo_engine.models.base_model import BaseModel
from bo_engine.parameter_space import ParameterSpace

logger = logging.getLogger(__name__)

class ExpectedImprovement(BaseAcquisitionFunction):
    """
    Expected Improvement (EI) acquisition function.
    
    Quantifies the expected improvement over the current best observed value,
    considering the model's prediction uncertainty.
    Assumes minimization of the objective function.
    """
    def __init__(self, 
                 parameter_space: ParameterSpace, 
                 xi: float = 0.01, # Exploration-exploitation trade-off parameter
                 **kwargs):
        """
        Initialize the Expected Improvement acquisition function.
        
        Args:
            parameter_space: The parameter space object.
            xi: Controls the trade-off between exploration and exploitation.
                Higher values encourage more exploration.
            **kwargs: Additional configuration arguments.
        """
        super().__init__(parameter_space, **kwargs)
        self.xi = xi

    def evaluate(self, X: np.ndarray, model: BaseModel, best_f: Optional[float] = None, **kwargs) -> np.ndarray:
        """
        Evaluate the Expected Improvement at given points.
        
        Args:
            X: Points to evaluate, shape (n_points, n_dims) in the internal [0, 1] space.
            model: The trained surrogate model (e.g., GaussianProcessModel).
            best_f: The best objective value observed so far (minimum value).
                    If None, EI cannot be calculated and will return zeros.
            **kwargs: Additional arguments (unused by standard EI).
                      
        Returns:
            np.ndarray: Expected Improvement values for each point in X, shape (n_points,).
        """
        if best_f is None:
            logger.warning("best_f not provided to EI evaluate method. Returning zeros.")
            return np.zeros(X.shape[0])
            
        if not model.is_trained():
            logger.warning("Model is not trained. Cannot evaluate EI. Returning zeros.")
            return np.zeros(X.shape[0])

        try:
            mean, variance = model.predict(X)
            std_dev = np.sqrt(np.maximum(variance, 1e-9)) # Avoid sqrt(0) or sqrt(negative)
            
            # Handle cases where standard deviation is effectively zero
            mask_zero_std = std_dev < 1e-9
            ei = np.zeros_like(mean)
            
            # Calculate EI only where std_dev is non-zero
            if not np.all(mask_zero_std):
                non_zero_std = std_dev[~mask_zero_std]
                non_zero_mean = mean[~mask_zero_std]
                
                # Calculate improvement (delta)
                imp = best_f - non_zero_mean - self.xi
                
                Z = imp / non_zero_std
                ei_non_zero = imp * norm.cdf(Z) + non_zero_std * norm.pdf(Z)
                
                # Ensure EI is non-negative
                ei[~mask_zero_std] = np.maximum(0, ei_non_zero)
                
            return ei

        except Exception as e:
            logger.error(f"Error evaluating Expected Improvement: {e}", exc_info=True)
            return np.zeros(X.shape[0]) # Return zeros on failure

    def optimize(self, 
                 model: BaseModel, 
                 best_f: Optional[float] = None, 
                 n_restarts: int = 10, 
                 raw_samples: int = 1000,
                 **kwargs) -> Tuple[np.ndarray, float]:
        """
        Find the point in the parameter space that maximizes the Expected Improvement.
        Uses L-BFGS-B optimization started from multiple random points.
        
        Args:
            model: The trained surrogate model.
            best_f: The best objective value observed so far.
            n_restarts: Number of times to restart the optimizer from different random starting points.
            raw_samples: Number of random points to initially sample for potential starting points.
            **kwargs: Additional arguments passed to the `evaluate` method during optimization.
            
        Returns:
            Tuple[np.ndarray, float]: 
                - The location of the maximum EI, shape (1, n_dims) in the internal [0, 1] space.
                - The maximum value of the EI function.
        """
        if best_f is None:
            logger.error("Cannot optimize EI without best_f. Falling back to random suggestion.")
            dims = self.parameter_space.get_internal_dimensions()
            random_point = np.random.rand(1, dims)
            return random_point, 0.0 # Return random point and zero EI

        if not model.is_trained():
             logger.error("Cannot optimize EI when model is not trained. Falling back to random suggestion.")
             dims = self.parameter_space.get_internal_dimensions()
             random_point = np.random.rand(1, dims)
             return random_point, 0.0

        dims = self.parameter_space.get_internal_dimensions()
        bounds = np.array([[0.0, 1.0]] * dims) # Bounds for the internal [0, 1] space

        # Objective function for the optimizer (minimize negative EI)
        def objective(x):            
            x_reshape = x.reshape(1, -1)
            # Evaluate EI (note the negative sign for minimization)
            ei_value = self.evaluate(x_reshape, model, best_f, **kwargs)
            return -ei_value[0]

        best_x = None
        best_neg_ei = np.inf

        # Generate starting points: sample randomly and choose the best ones
        logger.debug(f"Generating {raw_samples} random samples to find starting points for EI optimization.")
        random_starts = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(raw_samples, dims))
        random_starts_ei = self.evaluate(random_starts, model, best_f, **kwargs)
        # Pick the top `n_restarts` points with the highest EI as starting points
        best_random_indices = np.argsort(random_starts_ei)[-n_restarts:]
        starting_points = random_starts[best_random_indices, :]
        logger.debug(f"Selected {len(starting_points)} starting points for L-BFGS-B.")

        # Run L-BFGS-B from multiple starting points
        for start_point in starting_points:
            try:
                res = minimize(
                    fun=objective,
                    x0=start_point,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200} # Limit iterations
                )
                
                if res.success and res.fun < best_neg_ei:
                    best_neg_ei = res.fun
                    best_x = res.x
                    logger.debug(f"L-BFGS-B found new best point: x={best_x}, EI={-best_neg_ei:.4f}")
                elif not res.success:
                     logger.debug(f"L-BFGS-B optimization failed from start {start_point}: {res.message}")

            except Exception as e:
                logger.warning(f"Optimizer failed for start point {start_point}: {e}", exc_info=False)
                continue # Try next starting point

        if best_x is None:
            logger.warning("EI optimization failed to find a best point. Returning best random start.")
            # Fallback: return the best point found during random sampling
            best_idx = np.argmax(random_starts_ei)
            best_x = random_starts[best_idx]
            best_neg_ei = -random_starts_ei[best_idx]
            
        # Ensure shape is (1, dims)
        best_x = best_x.reshape(1, -1)
        max_ei_value = -best_neg_ei
        
        logger.info(f"EI optimization finished. Best point: {best_x}, Max EI: {max_ei_value:.4f}")
        return best_x, max_ei_value 
