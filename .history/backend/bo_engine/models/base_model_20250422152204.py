from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging

# Setup logger
logger = logging.getLogger("bo_engine.models")

class BaseModel(ABC):
    """
    Abstract base class for surrogate models used in Bayesian optimization.
    
    All model implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the model with hyperparameters.
        
        Args:
            **kwargs: Model-specific hyperparameters
        """
        self.is_fitted = False
        self.hyperparameters = kwargs
        self.X_train = None
        self.y_train = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            X: Training inputs, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,) or (n_samples, n_targets)
            
        Returns:
            self: The fitted model
        """
        self.X_train = X
        self.y_train = y
        self.is_fitted = True
        return self
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the mean output for the given inputs.
        
        Args:
            X: Inputs to predict for, shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted mean outputs, shape (n_samples,) or (n_samples, n_targets)
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Implementations should return predictions
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict both mean and standard deviation of the output.
        
        Args:
            X: Inputs to predict for, shape (n_samples, n_features)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (mean, std) where both have shape (n_samples,) or (n_samples, n_targets)
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Implementations should return (mean, std)
        pass
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate model performance metrics.
        
        Args:
            X: Test inputs, shape (n_samples, n_features)
            y: True outputs, shape (n_samples,) or (n_samples, n_targets)
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        y_pred = self.predict(X)
        
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
        return self.hyperparameters
    
    def set_hyperparameters(self, **kwargs) -> 'BaseModel':
        """
        Set hyperparameters of the model.
        
        Args:
            **kwargs: Hyperparameter names and values
            
        Returns:
            self: The model with updated hyperparameters
        """
        self.hyperparameters.update(kwargs)
        return self
    
    def __str__(self) -> str:
        """
        String representation of the model.
        
        Returns:
            str: Model description string
        """
        status = "fitted" if self.is_fitted else "not fitted"
        params = ", ".join(f"{k}={v}" for k, v in self.hyperparameters.items())
        return f"{self.__class__.__name__}({params}) - {status}" 
