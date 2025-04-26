#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Probability of Improvement (PI) acquisition function implemented in NumPy.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import logging
from typing import Dict, Any, Optional, Tuple, Union
from ..models.base_model import BaseModel
from ..parameter_space import ParameterSpace
from .base_acquisition import BaseAcquisitionFunction

logger = logging.getLogger(__name__)


class ProbabilityImprovement(BaseAcquisitionFunction):
    """
    Probability of Improvement (PI) acquisition function.
    
    PI(x) = Φ((f(x_best) - μ(x)) / σ(x)) for minimization
    PI(x) = Φ((μ(x) - f(x_best)) / σ(x)) for maximization
    
    where Φ is the standard normal CDF.
    """
    
    def __init__(self, model, parameter_space, xi=0.01, maximize=False):
        """
        Initialize Probability of Improvement acquisition function.
        
        Args:
            model: The surrogate model that provides predictions
            parameter_space: The parameter space object defining the search space
            xi: Exploration parameter. Higher values encourage more exploration (default: 0.01)
            maximize: Whether the objective is to be maximized (True) or minimized (False)
        """
        super().__init__(model, parameter_space)
        self.xi = xi
        self.maximize = maximize
        self._parameters = {"xi": xi}
        logger.debug(f"Initialized ProbabilityImprovement with xi={xi}, maximize={maximize}")
    
    def evaluate(self, X, best_f=None):
        """
        Evaluate the PI acquisition function at points X.
        
        Args:
            X: Points to evaluate, shape (n_points, n_dimensions)
            best_f: Current best observed objective value. If None, will be determined from model.
                    
        Returns:
            PI values at X, shape (n_points,)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get predictions from model
        mean, std = self.model.predict(X, return_std=True)
        
        # If best_f is not provided, get it from the model
        if best_f is None:
            # Assuming model has training data and targets
            if hasattr(self.model, "y_train_"):
                best_idx = np.argmin(self.model.y_train_) if not self.maximize else np.argmax(self.model.y_train_)
                best_f = self.model.y_train_[best_idx]
            else:
                # If model doesn't have training targets, use the mean prediction's best
                best_idx = np.argmin(mean) if not self.maximize else np.argmax(mean)
                best_f = mean[best_idx]
            logger.debug(f"Using best_f value: {best_f} from model")
        
        # Calculate improvement
        if self.maximize:
            z = (mean - best_f - self.xi) / (std + 1e-9)
        else:
            z = (best_f - mean - self.xi) / (std + 1e-9)
        
        # Calculate probability (CDF of standard normal)
        pi = norm.cdf(z)
        
        # Return negative PI for minimization in optimizers
        return -pi if self.maximize else pi
    
    def optimize(self, n_restarts=5, verbose=False):
        """
        Find the point with the optimal PI value in the parameter space.
        
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
            logger.info(f"Optimal PI value: {acq_value} at {x_best}")
        
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
    
    def suggest_next_xi(self, iteration):
        """
        Suggest a new value for xi based on the iteration number.
        A common strategy is to decrease exploration over time.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            New suggested xi value
        """
        # Simple decay function for xi
        new_xi = max(0.001, self.xi * (0.95 ** iteration))
        logger.debug(f"Suggesting new xi value: {new_xi} for iteration {iteration}")
        return new_xi
    
    def get_parameters(self):
        """
        Get the current parameters of the acquisition function.
        
        Returns:
            Dictionary of current parameters
        """
        return self._parameters.copy()
