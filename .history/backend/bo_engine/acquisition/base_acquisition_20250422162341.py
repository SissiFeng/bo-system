import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional

from bo_engine.parameter_space import ParameterSpace
from bo_engine.models.base_model import BaseModel

class BaseAcquisitionFunction(ABC):
    """
    Abstract base class for acquisition functions.
    Acquisition functions quantify the utility of sampling a particular point.
    """
    def __init__(self, parameter_space: ParameterSpace, **kwargs):
        """
        Initialize the acquisition function.
        
        Args:
            parameter_space: The parameter space object.
            **kwargs: Additional configuration specific to the acquisition function.
        """
        self.parameter_space = parameter_space
        self._config = kwargs

    @abstractmethod
    def evaluate(self, X: np.ndarray, model: BaseModel, **kwargs) -> np.ndarray:
        """
        Evaluate the acquisition function at given points.
        
        Args:
            X: Points to evaluate, shape (n_points, n_dims) in the internal [0, 1] space.
            model: The trained surrogate model.
            **kwargs: Additional arguments needed for specific acquisition functions 
                      (e.g., best observed value `best_f` for EI).
                      
        Returns:
            np.ndarray: Acquisition function values for each point in X, shape (n_points,).
        """
        pass

    def optimize(self, model: BaseModel, **kwargs) -> Tuple[np.ndarray, float]:
        """
        Find the point in the parameter space that maximizes the acquisition function.
        
        This default implementation uses random sampling, which is inefficient.
        Subclasses should override this with a more sophisticated optimization strategy
        (e.g., using L-BFGS-B starting from multiple random points).
        
        Args:
            model: The trained surrogate model.
            **kwargs: Additional arguments passed to the `evaluate` method.
            
        Returns:
            Tuple[np.ndarray, float]: 
                - The location of the maximum, shape (1, n_dims) in the internal [0, 1] space.
                - The maximum value of the acquisition function.
        """
        n_samples = kwargs.get('optimizer_samples', 1000) # Number of random samples
        dims = self.parameter_space.get_internal_dimensions()
        
        # Generate random candidate points within the [0, 1] bounds
        candidate_X = np.random.rand(n_samples, dims)
        
        # Evaluate acquisition function at candidates
        acq_values = self.evaluate(candidate_X, model, **kwargs)
        
        # Find the maximum
        best_idx = np.argmax(acq_values)
        max_acq_value = acq_values[best_idx]
        best_x = candidate_X[best_idx:best_idx+1, :] # Keep shape (1, dims)
        
        return best_x, max_acq_value 
