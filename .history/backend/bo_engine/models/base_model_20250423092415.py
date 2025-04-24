from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging

from ..parameter_space import ParameterSpace

# Setup logger
logger = logging.getLogger("bo_engine.models")

class BaseModel(ABC):
    """
    Abstract base class for surrogate models used in Bayesian optimization.
    
    All model implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, parameter_space: ParameterSpace, **kwargs):
        """
        Initialize the base model.
        
        Args:
            parameter_space: The parameter space definition.
            **kwargs: Model-specific configuration arguments.
        """
        self.parameter_space = parameter_space
        self._config = kwargs
        self._trained = False
        self.X_train = None
        self.y_train = None
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the surrogate model on the observed data.
        
        Args:
            X: Observed points, shape (n_samples, n_dims) in the internal [0, 1] space.
            y: Observed objective values, shape (n_samples,) or (n_samples, n_objectives).
               Implementations should handle single and potentially multi-objective cases.
        """
        self._trained = True
        self.X_train = X
        self.y_train = y
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained model.
        
        Args:
            X: Points to predict at, shape (n_points, n_dims) in the internal [0, 1] space.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - Mean prediction, shape (n_points,) or (n_points, n_objectives).
                - Variance (or standard deviation) prediction, shape (n_points,) or (n_points, n_objectives).
                  Variance is typically preferred for calculations.
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before making predictions.")
        pass
    
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._trained
    
    def get_config(self) -> Dict[str, Any]:
        """Return the model's configuration."""
        return self._config
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate model performance metrics.
        
        Args:
            X: Test inputs, shape (n_samples, n_features)
            y: True outputs, shape (n_samples,) or (n_samples, n_targets)
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        if not self._trained:
            raise ValueError("Model has not been trained yet.")
        
        y_pred, _ = self.predict(X)
        
        # Calculate metrics
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        
        # Calculate R²
        y_mean = np.mean(y, axis=0)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model to
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BaseModel':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            BaseModel: The loaded model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        
        return model
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get the hyperparameters of the model.
        
        Returns:
            Dict[str, Any]: Dictionary of hyperparameter names and values
        """
        return self._config
    
    def set_hyperparameters(self, **kwargs) -> 'BaseModel':
        """
        Set hyperparameters of the model.
        
        Args:
            **kwargs: Hyperparameter names and values
            
        Returns:
            self: The model with updated hyperparameters
        """
        self._config.update(kwargs)
        return self
    
    def __str__(self) -> str:
        """
        String representation of the model.
        
        Returns:
            str: Model description string
        """
        status = "trained" if self._trained else "not trained"
        params = ", ".join(f"{k}={v}" for k, v in self._config.items())
        return f"{self.__class__.__name__}({params}) - {status}" 
