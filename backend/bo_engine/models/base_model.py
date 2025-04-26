"""
定义模型抽象基类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging

# 假设 ParameterSpace 定义在上一级目录
# 如果不是，需要调整导入路径
try:
    from ..parameter_space import ParameterSpace
except ImportError:
    # 处理可能的直接运行脚本或不同目录结构的情况
    from bo_engine.parameter_space import ParameterSpace

# Setup logger
logger = logging.getLogger("bo_engine.models")

class BaseModel(ABC):
    """
    所有代理模型的抽象基类。

    模型内部处理的数据（X, y）应是经过 ParameterSpace 缩放/转换后的。
    预测结果应返回到原始的 y 空间尺度。
    """

    def __init__(self, parameter_space: ParameterSpace, **kwargs):
        """
        初始化模型基类。

        Args:
            parameter_space: 参数空间对象，用于数据转换。
            **kwargs: 特定模型实现的额外参数。
        """
        if parameter_space is None:
            raise ValueError("ParameterSpace 不能为 None")
        self.parameter_space = parameter_space
        self._config = kwargs
        self._model_initialized = False
        self._X_train = None
        self._y_train = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        使用提供的观测数据训练模型。

        Args:
            X: 观测点的参数值（内部表示，通常是 [0, 1] 归一化）。形状为 (n_samples, n_dimensions)。
            y: 观测点的目标值（内部表示，可能经过转换）。形状为 (n_samples,) 或 (n_samples, 1)。
        """
        if X is None or y is None:
             raise ValueError("训练数据 X 和 y 不能为空。")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X 和 y 的样本数量必须一致 ({X.shape[0]} != {y.shape[0]})")

        self._X_train = X
        self._y_train = y
        self._model_initialized = True
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        对新的参数点进行预测。

        Args:
            X: 需要预测的参数点（内部表示，通常是 [0, 1] 归一化）。形状为 (n_predict, n_dimensions)。

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                - 预测均值 (y_mean)，形状为 (n_predict,)。应返回到*原始*y空间尺度。
                - 预测标准差 (y_std)，形状为 (n_predict,)。可选，如果模型不支持则为 None。应返回到*原始*y空间尺度。
        """
        if not self._model_initialized:
            raise RuntimeError("模型尚未训练 (fit)，无法进行预测。")
        if X is None:
            raise ValueError("预测点 X 不能为空。")
        if X.ndim != 2:
             raise ValueError(f"预测点 X 必须是二维数组，当前维度: {X.ndim}")
        pass
    
    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        使用新的观测数据增量更新模型（可选实现）。
        如果模型不支持增量更新，默认行为是重新训练所有数据。

        Args:
            X_new: 新观测点的参数值（内部表示）。形状为 (n_new, n_dimensions)。
            y_new: 新观测点的目标值（内部表示）。形状为 (n_new,) 或 (n_new, 1)。
        """
        if not self._model_initialized:
             raise RuntimeError("模型未初始化，无法更新。请先调用 fit。")
        if X_new is None or y_new is None:
             raise ValueError("新数据 X_new 和 y_new 不能为空。")
        if X_new.shape[0] != y_new.shape[0]:
             raise ValueError(f"X_new 和 y_new 的样本数量必须一致 ({X_new.shape[0]} != {y_new.shape[0]})")

        # 默认实现：合并新旧数据并重新拟合
        if self._X_train is not None and self._y_train is not None:
            X_all = np.vstack((self._X_train, X_new))
            # 确保 y_train 和 y_new 形状一致
            y_old = self._y_train.reshape(-1, 1) if self._y_train.ndim == 1 else self._y_train
            y_new_reshaped = y_new.reshape(-1, 1) if y_new.ndim == 1 else y_new
            y_all = np.vstack((y_old, y_new_reshaped)).flatten() # fit 通常需要 1D y
            self.fit(X_all, y_all)
        else:
             # 如果没有旧数据，直接用新数据拟合
             self.fit(X_new, y_new.flatten())

    @property
    def is_initialized(self) -> bool:
        """检查模型是否已初始化（训练过）。"""
        return self._model_initialized

    def get_training_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取当前用于训练模型的数据。"""
        if self._model_initialized:
            return self._X_train, self._y_train
        return None
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate model performance metrics.
        
        Args:
            X: Test inputs, shape (n_samples, n_features)
            y: True outputs, shape (n_samples,) or (n_samples, n_targets)
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        if not self._model_initialized:
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
        status = "trained" if self._model_initialized else "not trained"
        params = ", ".join(f"{k}={v}" for k, v in self._config.items())
        return f"{self.__class__.__name__}({params}) - {status}" 
