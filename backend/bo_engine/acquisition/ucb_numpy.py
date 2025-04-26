#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Upper Confidence Bound (UCB) acquisition function implemented in NumPy.
"""

import numpy as np
from scipy.optimize import minimize
import logging
from typing import Tuple, Dict, Any, Optional, Union

from .base_acquisition import BaseAcquisitionFunction
from ..models.base_model import BaseModel
from ..parameter_space import ParameterSpace

logger = logging.getLogger(__name__)

class UpperConfidenceBound(BaseAcquisitionFunction):
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    UCB(x) = μ(x) - κ * σ(x) for minimization problems
    UCB(x) = μ(x) + κ * σ(x) for maximization problems
    
    where:
    - μ(x) is the predicted mean at x
    - σ(x) is the predicted standard deviation at x
    - κ is the exploration parameter
    """
    
    def __init__(self, model, parameter_space, kappa=2.0, maximize=False):
        """
        Initialize Upper Confidence Bound acquisition function.
        
        Args:
            model: The surrogate model that provides predictions
            parameter_space: The parameter space object defining the search space
            kappa: Exploration parameter. Higher values encourage more exploration (default: 2.0)
            maximize: Whether the objective is to be maximized (True) or minimized (False)
        """
        super().__init__(model, parameter_space)
        self.kappa = kappa
        self.maximize = maximize
        self._parameters = {"kappa": kappa}
        logger.debug(f"Initialized UpperConfidenceBound with kappa={kappa}, maximize={maximize}")
    
    def evaluate(self, X, **kwargs):
        """
        Evaluate the UCB acquisition function at points X.
        
        Args:
            X: Points to evaluate, shape (n_points, n_dimensions)
            **kwargs: Additional arguments (not used)
                    
        Returns:
            UCB values at X, shape (n_points,)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get predictions from model
        mean, std = self.model.predict(X, return_std=True)
        
        # Calculate UCB based on optimization direction
        if self.maximize:
            # For maximization, we want to maximize mean + kappa*std
            ucb = mean + self.kappa * std
            return -ucb  # Negate for minimization in scipy.optimize.minimize
        else:
            # For minimization, we want to minimize mean - kappa*std
            ucb = mean - self.kappa * std
            return ucb
    
    def optimize(self, n_restarts=5, verbose=False):
        """
        Find the point with the optimal UCB value in the parameter space.
        
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
            logger.info(f"Optimal UCB value: {acq_value} at {x_best}")
        
        return x_best, acq_value
    
    def update_parameters(self, parameters):
        """
        Update the acquisition function parameters.
        
        Args:
            parameters: Dictionary of parameters to update
        """
        if "kappa" in parameters:
            self.kappa = parameters["kappa"]
            self._parameters["kappa"] = self.kappa
            logger.debug(f"Updated kappa to {self.kappa}")
    
    def suggest_next_kappa(self, iteration):
        """
        Suggest a new value for kappa based on the iteration number.
        A common strategy is to decrease exploration over time.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            New suggested kappa value
        """
        # Simple decay function for kappa
        # Start with higher exploration, then gradually decrease
        new_kappa = max(0.1, self.kappa * (0.95 ** iteration))
        logger.debug(f"Suggesting new kappa value: {new_kappa} for iteration {iteration}")
        return new_kappa
    
    def get_parameters(self):
        """
        Get the current parameters of the acquisition function.
        
        Returns:
            Dictionary of current parameters
        """
        return self._parameters.copy()
    