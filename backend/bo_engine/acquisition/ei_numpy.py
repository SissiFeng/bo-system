"""
Expected Improvement (EI) acquisition function implementation using NumPy/SciPy.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import logging
from typing import Tuple, Dict, Any, Optional, Union

# Adjust import based on actual directory structure
try:
    from .base_acquisition import BaseAcquisitionFunction
    from ..models.base_model import BaseModel
    from ..parameter_space import ParameterSpace # Added import
except ImportError:
    # Fallback for different execution contexts
    from base_acquisition import BaseAcquisitionFunction
    from models.base_model import BaseModel
    from parameter_space import ParameterSpace # Added import

logger = logging.getLogger(__name__)

class ExpectedImprovement(BaseAcquisitionFunction):
    """
    Expected Improvement (EI) acquisition function.
    
    For minimization problems:
    EI(x) = (f_best - μ(x)) * Φ(Z) + σ(x) * φ(Z)
    where Z = (f_best - μ(x)) / σ(x)
    
    For maximization problems:
    EI(x) = (μ(x) - f_best) * Φ(Z) + σ(x) * φ(Z)
    where Z = (μ(x) - f_best) / σ(x)
    
    where:
    - μ(x) is the predicted mean at x
    - σ(x) is the predicted standard deviation at x
    - f_best is the best observed value so far
    - Φ is the cumulative distribution function of the standard normal distribution
    - φ is the probability density function of the standard normal distribution
    """
    
    def __init__(
        self, 
        model, 
        parameter_space, 
        xi=0.01, 
        maximize=False, 
        best_f=None
    ):
        """
        Initialize Expected Improvement acquisition function.
        
        Args:
            model: The surrogate model that provides predictions
            parameter_space: The parameter space object defining the search space
            xi: Exploration-exploitation trade-off parameter (default: 0.01)
            maximize: Whether the objective is to be maximized (True) or minimized (False)
            best_f: The best objective value observed so far (optional, will be updated later if None)
        """
        super().__init__(model, parameter_space)
        self.xi = xi
        self.maximize = maximize
        self.best_f = best_f
        self._parameters = {"xi": xi, "maximize": maximize}
        logger.debug(f"Initialized ExpectedImprovement with xi={xi}, maximize={maximize}, best_f={best_f}")
    
    def evaluate(self, X, **kwargs):
        """
        Evaluate the EI acquisition function at points X.
        
        Args:
            X: Points to evaluate, shape (n_points, n_dimensions)
            **kwargs: Additional arguments:
                best_f: Override the best objective value
                
        Returns:
            EI values at X, shape (n_points,)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get the best observed value (can be overridden by kwargs)
        best_f = kwargs.get("best_f", self.best_f)
        if best_f is None:
            raise ValueError("Best objective value (best_f) must be provided")
        
        # Get predictions from model
        mean, std = self.model.predict(X, return_std=True)
        
        # Handle case of zero standard deviation (add small constant)
        std = np.maximum(std, 1e-9)
        
        # Calculate improvement over best observed value
        if self.maximize:
            improvement = mean - best_f - self.xi
            Z = improvement / std
        else:
            improvement = best_f - mean - self.xi
            Z = improvement / std
        
        # Compute expected improvement using normal CDF and PDF
        cdf = norm.cdf(Z)
        pdf = norm.pdf(Z)
        ei = improvement * cdf + std * pdf
        
        # Return negative for minimization in scipy.optimize.minimize
        return -ei if self.maximize else ei
    
    def optimize(self, n_restarts=5, verbose=False):
        """
        Find the point with the optimal EI value in the parameter space.
        
        Args:
            n_restarts: Number of restarts for optimization to avoid local minima
            verbose: Whether to print optimization progress
            
        Returns:
            x_best: Point with optimal acquisition value in the [0,1] internal space
            acq_value: Optimal acquisition value
        """
        # 获取 [0,1] 内部空间的边界
        bounds = [(0.0, 1.0)] * self.parameter_space.get_internal_dimensions()
        dim = len(bounds)
        
        def objective(x):
            # Reshape for evaluation if needed
            x_reshaped = x.reshape(1, -1) if x.ndim == 1 else x
            return self.evaluate(x_reshaped)[0]
        
        # Random starting points for optimization
        x_tries = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            size=(n_restarts, dim)
        )
        
        # Run local optimization from each starting point
        results = []
        for x_try in x_tries:
            res = minimize(
                objective,
                x_try,
                bounds=bounds,
                method="L-BFGS-B"
            )
            results.append((res.fun, res.x))
        
        # Find best result
        best_idx = np.argmin([r[0] for r in results])
        acq_value, x_best = results[best_idx]
        
        if verbose:
            logger.info(f"Optimal EI value: {acq_value} at {x_best}")
        
        return x_best, acq_value
    
    def update_parameters(self, parameters):
        """
        Update the acquisition function parameters.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        if "xi" in parameters:
            self.xi = parameters["xi"]
            self._parameters["xi"] = self.xi
            logger.debug(f"Updated xi to {self.xi}")
        
        if "best_f" in parameters:
            self.best_f = parameters["best_f"]
            logger.debug(f"Updated best_f to {self.best_f}")
            
        if "maximize" in parameters:
            self.maximize = parameters["maximize"]
            self._parameters["maximize"] = self.maximize
            logger.debug(f"Updated maximize to {self.maximize}")
    
    def suggest_next_xi(self, iteration):
        """
        Suggest a new value for xi based on the iteration number.
        Typically, xi decreases over iterations to shift from exploration to exploitation.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            New suggested xi value
        """
        # Simple decay function for xi
        new_xi = max(0.0001, self.xi * (0.9 ** iteration))
        logger.debug(f"Suggesting new xi value: {new_xi} for iteration {iteration}")
        return new_xi
    
    def get_parameters(self):
        """
        Get the current parameters of the acquisition function.
        
        Returns:
            Dictionary of current parameters
        """
        params = self._parameters.copy()
        params["best_f"] = self.best_f
        return params
