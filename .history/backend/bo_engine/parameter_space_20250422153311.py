from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable
import copy
import re
import logging
from enum import Enum, auto
import random
import json
import os

from bo_engine.utils import (
    validate_expression, 
    evaluate_expression, 
    one_hot_encode, 
    one_hot_decode, 
    scale_parameters, 
    unscale_parameters,
    safe_eval,
    save_to_json,
    load_from_json
)

# Setup logger
logger = logging.getLogger("bo_engine.parameter_space")

class ParameterType(Enum):
    """参数类型枚举"""
    CONTINUOUS = auto()  # 连续参数
    INTEGER = auto()     # 整数参数
    CATEGORICAL = auto() # 分类参数
    ORDINAL = auto()     # 有序分类参数


class ObjectiveType(str, Enum):
    """Enum for objective types."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ConstraintType(str, Enum):
    """Enum for constraint types."""
    SUM_EQUALS = "sum_equals"
    SUM_LESS_THAN = "sum_less_than"
    SUM_GREATER_THAN = "sum_greater_than"
    PRODUCT_EQUALS = "product_equals"
    CUSTOM = "custom"


class Parameter(ABC):
    """
    Abstract base class for all parameter types.
    """
    def __init__(
        self,
        name: str,
        description: str = "",
        parameter_type: ParameterType = None,
        importance: float = 1.0,
        constraints: List[str] = None
    ):
        """
        初始化参数对象
        
        Args:
            name: 参数名称
            description: 参数描述
            parameter_type: 参数类型
            importance: 参数重要性权重（默认为1.0）
            constraints: 参数约束条件列表（表达式字符串）
        """
        self.name = name
        self.description = description
        self.parameter_type = parameter_type
        self.importance = importance
        self.constraints = constraints or []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将参数转换为字典表示
        
        Returns:
            Dict[str, Any]: 参数的字典表示
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameter_type": self.parameter_type.name if self.parameter_type else None,
            "importance": self.importance,
            "constraints": self.constraints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Parameter':
        """
        从字典创建参数对象
        
        Args:
            data: 参数的字典表示
            
        Returns:
            Parameter: 参数对象
        """
        parameter_type = None
        if "parameter_type" in data and data["parameter_type"]:
            parameter_type = ParameterType[data["parameter_type"]]
        
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parameter_type=parameter_type,
            importance=data.get("importance", 1.0),
            constraints=data.get("constraints", [])
        )
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate that the parameter definition is valid."""
        pass
    
    @abstractmethod
    def sample(self) -> Any:
        """Generate a random sample from the parameter space."""
        pass
    
    @abstractmethod
    def contains(self, value: Any) -> bool:
        """Check if a value is within the parameter space."""
        pass
    
    @abstractmethod
    def to_internal(self, value: Any) -> Union[float, List[float]]:
        """Convert value to internal representation (for model training)."""
        pass
    
    @abstractmethod
    def from_internal(self, internal_value: Union[float, List[float]]) -> Any:
        """Convert internal representation to actual value."""
        pass
    
    @abstractmethod
    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """Get bounds for internal representation."""
        pass
    
    @abstractmethod
    def get_dimensionality(self) -> int:
        """Get dimensionality of internal representation."""
        pass


class ContinuousParameter(Parameter):
    """连续参数类"""
    
    def __init__(
        self,
        name: str,
        min_value: float,
        max_value: float,
        description: str = "",
        importance: float = 1.0,
        constraints: List[str] = None,
        log_scale: bool = False,
        prior_mean: Optional[float] = None,
        prior_std: Optional[float] = None
    ):
        """
        初始化连续参数
        
        Args:
            name: 参数名称
            min_value: 最小值
            max_value: 最大值
            description: 参数描述
            importance: 参数重要性权重
            constraints: 参数约束条件列表
            log_scale: 是否使用对数尺度
            prior_mean: 先验分布的均值
            prior_std: 先验分布的标准差
        """
        super().__init__(
            name=name,
            description=description,
            parameter_type=ParameterType.CONTINUOUS,
            importance=importance,
            constraints=constraints
        )
        
        if min_value >= max_value:
            raise ValueError(f"最小值 {min_value} 必须小于最大值 {max_value}")
        
        if log_scale and min_value <= 0:
            raise ValueError(f"对数尺度参数的最小值必须大于0，当前值为 {min_value}")
        
        self.min_value = min_value
        self.max_value = max_value
        self.log_scale = log_scale
        self.prior_mean = prior_mean
        self.prior_std = prior_std
    
    def transform(self, value: float) -> float:
        """
        将标准化值 [0, 1] 转换为实际参数值
        
        Args:
            value: 标准化值 [0, 1]
            
        Returns:
            float: 实际参数值
        """
        if value < 0 or value > 1:
            raise ValueError(f"标准化值必须在 [0, 1] 范围内，当前值为 {value}")
        
        if self.log_scale:
            # 对数尺度转换
            log_min = np.log(self.min_value)
            log_max = np.log(self.max_value)
            return np.exp(log_min + value * (log_max - log_min))
        else:
            # 线性尺度转换
            return self.min_value + value * (self.max_value - self.min_value)
    
    def inverse_transform(self, value: float) -> float:
        """
        将实际参数值转换为标准化值 [0, 1]
        
        Args:
            value: 实际参数值
            
        Returns:
            float: 标准化值 [0, 1]
        """
        if value < self.min_value or value > self.max_value:
            raise ValueError(f"参数值必须在 [{self.min_value}, {self.max_value}] 范围内，当前值为 {value}")
        
        if self.log_scale:
            # 对数尺度逆转换
            log_min = np.log(self.min_value)
            log_max = np.log(self.max_value)
            return (np.log(value) - log_min) / (log_max - log_min)
        else:
            # 线性尺度逆转换
            return (value - self.min_value) / (self.max_value - self.min_value)
    
    def sample(self, n: int = 1, rng: np.random.RandomState = None) -> np.ndarray:
        """
        采样参数值
        
        Args:
            n: 采样数量
            rng: 随机数生成器
            
        Returns:
            np.ndarray: 采样值数组
        """
        if rng is None:
            rng = np.random.RandomState()
        
        # 生成 [0, 1] 范围的均匀分布样本
        samples = rng.uniform(0, 1, n)
        
        # 转换为实际参数值
        return np.array([self.transform(x) for x in samples])
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将参数转换为字典表示
        
        Returns:
            Dict[str, Any]: 参数的字典表示
        """
        base_dict = super().to_dict()
        base_dict.update({
            "min_value": self.min_value,
            "max_value": self.max_value,
            "log_scale": self.log_scale,
            "prior_mean": self.prior_mean,
            "prior_std": self.prior_std
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContinuousParameter':
        """
        从字典创建连续参数对象
        
        Args:
            data: 参数的字典表示
            
        Returns:
            ContinuousParameter: 连续参数对象
        """
        return cls(
            name=data["name"],
            min_value=data["min_value"],
            max_value=data["max_value"],
            description=data.get("description", ""),
            importance=data.get("importance", 1.0),
            constraints=data.get("constraints", []),
            log_scale=data.get("log_scale", False),
            prior_mean=data.get("prior_mean"),
            prior_std=data.get("prior_std")
        )
    
    def to_internal(self, value: float) -> float:
        """
        Scale value to [0, 1] range.
        
        Args:
            value: Original value
            
        Returns:
            float: Scaled value
        """
        return (value - self.min_value) / (self.max_value - self.min_value)
    
    def from_internal(self, internal_value: float) -> float:
        """
        Convert scaled [0, 1] value back to original range.
        
        Args:
            internal_value: Scaled value
            
        Returns:
            float: Original value
        """
        return self.min_value + internal_value * (self.max_value - self.min_value)
    
    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """
        Get bounds for internal representation.
        
        Returns:
            Tuple[List[float], List[float]]: Tuple of (lower_bounds, upper_bounds)
        """
        return ([0.0], [1.0])
    
    def get_dimensionality(self) -> int:
        """
        Get dimensionality of internal representation.
        
        Returns:
            int: Dimensionality (1 for continuous parameters)
        """
        return 1


class IntegerParameter(Parameter):
    """整数参数类"""
    
    def __init__(
        self,
        name: str,
        min_value: int,
        max_value: int,
        description: str = "",
        importance: float = 1.0,
        constraints: List[str] = None,
        log_scale: bool = False
    ):
        """
        初始化整数参数
        
        Args:
            name: 参数名称
            min_value: 最小值
            max_value: 最大值
            description: 参数描述
            importance: 参数重要性权重
            constraints: 参数约束条件列表
            log_scale: 是否使用对数尺度
        """
        super().__init__(
            name=name,
            description=description,
            parameter_type=ParameterType.INTEGER,
            importance=importance,
            constraints=constraints
        )
        
        if min_value >= max_value:
            raise ValueError(f"最小值 {min_value} 必须小于最大值 {max_value}")
        
        if log_scale and min_value <= 0:
            raise ValueError(f"对数尺度参数的最小值必须大于0，当前值为 {min_value}")
        
        self.min_value = min_value
        self.max_value = max_value
        self.log_scale = log_scale
    
    def transform(self, value: float) -> int:
        """
        将标准化值 [0, 1] 转换为实际整数参数值
        
        Args:
            value: 标准化值 [0, 1]
            
        Returns:
            int: 实际整数参数值
        """
        if value < 0 or value > 1:
            raise ValueError(f"标准化值必须在 [0, 1] 范围内，当前值为 {value}")
        
        if self.log_scale:
            # 对数尺度转换
            log_min = np.log(self.min_value) if self.min_value > 0 else np.log(0.5)
            log_max = np.log(self.max_value + 1 - 1e-10)
            cont_value = np.exp(log_min + value * (log_max - log_min))
            return int(np.floor(cont_value))
        else:
            # 线性尺度转换
            return int(np.floor(self.min_value + value * (self.max_value - self.min_value + 1)))
    
    def inverse_transform(self, value: int) -> float:
        """
        将实际整数参数值转换为标准化值 [0, 1]
        
        Args:
            value: 实际整数参数值
            
        Returns:
            float: 标准化值 [0, 1]
        """
        if value < self.min_value or value > self.max_value:
            raise ValueError(f"参数值必须在 [{self.min_value}, {self.max_value}] 范围内，当前值为 {value}")
        
        if self.log_scale:
            # 对数尺度逆转换
            log_min = np.log(self.min_value) if self.min_value > 0 else np.log(0.5)
            log_max = np.log(self.max_value + 1 - 1e-10)
            return (np.log(value + 0.5) - log_min) / (log_max - log_min)
        else:
            # 线性尺度逆转换
            return (value - self.min_value) / (self.max_value - self.min_value + 1)
    
    def sample(self, n: int = 1, rng: np.random.RandomState = None) -> np.ndarray:
        """
        采样参数值
        
        Args:
            n: 采样数量
            rng: 随机数生成器
            
        Returns:
            np.ndarray: 采样值数组
        """
        if rng is None:
            rng = np.random.RandomState()
        
        # 生成 [0, 1] 范围的均匀分布样本
        samples = rng.uniform(0, 1, n)
        
        # 转换为实际参数值
        return np.array([self.transform(x) for x in samples])
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将参数转换为字典表示
        
        Returns:
            Dict[str, Any]: 参数的字典表示
        """
        base_dict = super().to_dict()
        base_dict.update({
            "min_value": self.min_value,
            "max_value": self.max_value,
            "log_scale": self.log_scale
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntegerParameter':
        """
        从字典创建整数参数对象
        
        Args:
            data: 参数的字典表示
            
        Returns:
            IntegerParameter: 整数参数对象
        """
        return cls(
            name=data["name"],
            min_value=data["min_value"],
            max_value=data["max_value"],
            description=data.get("description", ""),
            importance=data.get("importance", 1.0),
            constraints=data.get("constraints", []),
            log_scale=data.get("log_scale", False)
        )


class CategoricalParameter(Parameter):
    """分类参数类"""
    
    def __init__(
        self,
        name: str,
        categories: List[Any],
        description: str = "",
        importance: float = 1.0,
        constraints: List[str] = None,
        probabilities: Optional[List[float]] = None
    ):
        """
        初始化分类参数
        
        Args:
            name: 参数名称
            categories: 分类值列表
            description: 参数描述
            importance: 参数重要性权重
            constraints: 参数约束条件列表
            probabilities: 各分类的概率权重（可选）
        """
        super().__init__(
            name=name,
            description=description,
            parameter_type=ParameterType.CATEGORICAL,
            importance=importance,
            constraints=constraints
        )
        
        if len(categories) == 0:
            raise ValueError("分类列表不能为空")
        
        self.categories = categories
        
        # 检查并设置概率权重
        if probabilities is not None:
            if len(probabilities) != len(categories):
                raise ValueError(f"概率列表长度 {len(probabilities)} 必须等于分类列表长度 {len(categories)}")
            
            if not np.isclose(sum(probabilities), 1.0):
                raise ValueError(f"概率总和必须为1，当前总和为 {sum(probabilities)}")
            
            self.probabilities = probabilities
        else:
            # 默认使用均匀分布
            self.probabilities = [1.0 / len(categories)] * len(categories)
    
    def transform(self, value: float) -> Any:
        """
        将标准化值 [0, 1] 转换为实际分类参数值
        
        Args:
            value: 标准化值 [0, 1]
            
        Returns:
            Any: A 实际分类参数值
        """
        if value < 0 or value > 1:
            raise ValueError(f"标准化值必须在 [0, 1] 范围内，当前值为 {value}")
        
        # 根据概率权重选择对应的分类
        cumsum_probs = np.cumsum(self.probabilities)
        index = np.searchsorted(cumsum_probs, value)
        
        if index >= len(self.categories):
            index = len(self.categories) - 1
            
        return self.categories[index]
    
    def inverse_transform(self, value: Any) -> float:
        """
        将实际分类参数值转换为标准化值 [0, 1]
        
        Args:
            value: 实际分类参数值
            
        Returns:
            float: 标准化值 [0, 1]（返回对应分类的中点概率值）
        """
        try:
            index = self.categories.index(value)
        except ValueError:
            raise ValueError(f"值 {value} 不在分类列表中")
        
        # 计算该分类对应的累积概率范围的中点
        cum_prob_start = sum(self.probabilities[:index])
        cum_prob_end = cum_prob_start + self.probabilities[index]
        
        # 返回区间中点
        return (cum_prob_start + cum_prob_end) / 2
    
    def sample(self, n: int = 1, rng: np.random.RandomState = None) -> np.ndarray:
        """
        采样参数值
        
        Args:
            n: 采样数量
            rng: 随机数生成器
            
        Returns:
            np.ndarray: 采样值数组（对象数组）
        """
        if rng is None:
            rng = np.random.RandomState()
        
        # 根据概率权重进行采样
        indices = rng.choice(len(self.categories), size=n, p=self.probabilities)
        
        # 返回对应的分类值
        return np.array([self.categories[i] for i in indices], dtype=object)
    
    def get_one_hot_dimensions(self) -> int:
        """
        获取One-hot编码的维度
        
        Returns:
            int: One-hot编码的维度
        """
        return len(self.categories)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将参数转换为字典表示
        
        Returns:
            Dict[str, Any]: 参数的字典表示
        """
        base_dict = super().to_dict()
        base_dict.update({
            "categories": self.categories,
            "probabilities": self.probabilities
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CategoricalParameter':
        """
        从字典创建分类参数对象
        
        Args:
            data: 参数的字典表示
            
        Returns:
            CategoricalParameter: 分类参数对象
        """
        return cls(
            name=data["name"],
            categories=data["categories"],
            description=data.get("description", ""),
            importance=data.get("importance", 1.0),
            constraints=data.get("constraints", []),
            probabilities=data.get("probabilities")
        )


class OrdinalParameter(Parameter):
    """有序分类参数类"""
    
    def __init__(
        self,
        name: str,
        categories: List[Any],
        description: str = "",
        importance: float = 1.0,
        constraints: List[str] = None
    ):
        """
        初始化有序分类参数
        
        Args:
            name: 参数名称
            categories: 有序分类值列表（按顺序排列）
            description: 参数描述
            importance: 参数重要性权重
            constraints: 参数约束条件列表
        """
        super().__init__(
            name=name,
            description=description,
            parameter_type=ParameterType.ORDINAL,
            importance=importance,
            constraints=constraints
        )
        
        if len(categories) == 0:
            raise ValueError("分类列表不能为空")
        
        self.categories = categories
    
    def transform(self, value: float) -> Any:
        """
        将标准化值 [0, 1] 转换为实际有序分类参数值
        
        Args:
            value: 标准化值 [0, 1]
            
        Returns:
            Any: 实际有序分类参数值
        """
        if value < 0 or value > 1:
            raise ValueError(f"标准化值必须在 [0, 1] 范围内，当前值为 {value}")
        
        n_categories = len(self.categories)
        
        # 计算索引（线性映射）
        index = min(n_categories - 1, int(np.floor(value * n_categories)))
        
        return self.categories[index]
    
    def inverse_transform(self, value: Any) -> float:
        """
        将实际有序分类参数值转换为标准化值 [0, 1]
        
        Args:
            value: 实际有序分类参数值
            
        Returns:
            float: 标准化值 [0, 1]
        """
        try:
            index = self.categories.index(value)
        except ValueError:
            raise ValueError(f"值 {value} 不在分类列表中")
        
        n_categories = len(self.categories)
        
        # 返回区间中点
        return (index + 0.5) / n_categories
    
    def sample(self, n: int = 1, rng: np.random.RandomState = None) -> np.ndarray:
        """
        采样参数值
        
        Args:
            n: 采样数量
            rng: 随机数生成器
            
        Returns:
            np.ndarray: 采样值数组（对象数组）
        """
        if rng is None:
            rng = np.random.RandomState()
        
        n_categories = len(self.categories)
        
        # 均匀采样索引
        indices = rng.randint(0, n_categories, size=n)
        
        # 返回对应的分类值
        return np.array([self.categories[i] for i in indices], dtype=object)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将参数转换为字典表示
        
        Returns:
            Dict[str, Any]: 参数的字典表示
        """
        base_dict = super().to_dict()
        base_dict.update({
            "categories": self.categories
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrdinalParameter':
        """
        从字典创建有序分类参数对象
        
        Args:
            data: 参数的字典表示
            
        Returns:
            OrdinalParameter: 有序分类参数对象
        """
        return cls(
            name=data["name"],
            categories=data["categories"],
            description=data.get("description", ""),
            importance=data.get("importance", 1.0),
            constraints=data.get("constraints", [])
        )


class Objective:
    """
    Objective function for optimization.
    """
    def __init__(self, name: str, objective_type: ObjectiveType):
        """
        Initialize an objective.
        
        Args:
            name: Objective name
            objective_type: Type of objective (maximize or minimize)
        """
        self.name = name
        self.type = objective_type
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert objective to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "name": self.name,
            "type": self.type.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Objective':
        """
        Create objective from dictionary representation.
        
        Args:
            data: Dictionary with objective definition
            
        Returns:
            Objective: Created objective
        """
        return cls(
            name=data["name"],
            objective_type=ObjectiveType(data["type"])
        )


class Constraint:
    """
    Constraint for optimization.
    """
    def __init__(self, expression: str, constraint_type: ConstraintType, value: float):
        """
        Initialize a constraint.
        
        Args:
            expression: Expression for the constraint
            constraint_type: Type of constraint
            value: Target value for the constraint
        """
        self.expression = expression
        self.type = constraint_type
        self.value = float(value)
    
    def evaluate(self, parameters: Dict[str, Any]) -> bool:
        """
        Evaluate if a point satisfies the constraint.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            bool: True if constraint is satisfied, False otherwise
        """
        # Evaluate the expression
        result = evaluate_expression(self.expression, parameters)
        
        # Check if constraint is satisfied
        if self.type == ConstraintType.SUM_EQUALS:
            return abs(result - self.value) < 1e-6
        elif self.type == ConstraintType.SUM_LESS_THAN:
            return result < self.value
        elif self.type == ConstraintType.SUM_GREATER_THAN:
            return result > self.value
        elif self.type == ConstraintType.PRODUCT_EQUALS:
            return abs(result - self.value) < 1e-6
        else:  # CUSTOM
            return abs(result) < 1e-6
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert constraint to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "expression": self.expression,
            "type": self.type.value,
            "value": self.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Constraint':
        """
        Create constraint from dictionary representation.
        
        Args:
            data: Dictionary with constraint definition
            
        Returns:
            Constraint: Created constraint
        """
        return cls(
            expression=data["expression"],
            constraint_type=ConstraintType(data["type"]),
            value=data["value"]
        )


class ParameterSpace:
    """参数空间类，管理多个参数及其关系"""
    
    def __init__(self, parameters: List[Parameter] = None, constraints: List[str] = None):
        """
        初始化参数空间
        
        Args:
            parameters: 参数列表
            constraints: 全局约束条件列表
        """
        self.parameters = parameters or []
        self.constraints = constraints or []
        self._param_by_name = {param.name: param for param in self.parameters}
    
    def add_parameter(self, parameter: Parameter) -> None:
        """
        添加参数到参数空间
        
        Args:
            parameter: 参数对象
        """
        if parameter.name in self._param_by_name:
            raise ValueError(f"参数名 '{parameter.name}' 已存在")
        
        self.parameters.append(parameter)
        self._param_by_name[parameter.name] = parameter
    
    def get_parameter(self, name: str) -> Parameter:
        """
        根据名称获取参数
        
        Args:
            name: 参数名称
            
        Returns:
            Parameter: 参数对象
        """
        if name not in self._param_by_name:
            raise ValueError(f"参数名 '{name}' 不存在")
        
        return self._param_by_name[name]
    
    def get_parameter_names(self) -> List[str]:
        """
        获取所有参数名称
        
        Returns:
            List[str]: 参数名称列表
        """
        return [param.name for param in self.parameters]
    
    def get_dimensions(self) -> int:
        """
        获取参数空间的维度
        
        Returns:
            int: 参数空间的维度
        """
        return len(self.parameters)
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        获取参数空间的边界（仅适用于连续参数和整数参数）
        
        Returns:
            List[Tuple[float, float]]: 参数边界列表 [(min1, max1), (min2, max2), ...]
        """
        bounds = []
        
        for param in self.parameters:
            if isinstance(param, (ContinuousParameter, IntegerParameter)):
                bounds.append((param.min_value, param.max_value))
            else:
                # 对于分类参数，使用单位区间 [0, 1]
                bounds.append((0.0, 1.0))
        
        return bounds
    
    def transform(self, params_dict: Dict[str, Any]) -> np.ndarray:
        """
        将参数字典转换为标准化的向量表示
        
        Args:
            params_dict: 参数字典 {参数名: 参数值}
            
        Returns:
            np.ndarray: 标准化的向量表示
        """
        # 检查参数是否都存在
        unknown_params = set(params_dict.keys()) - set(self._param_by_name.keys())
        if unknown_params:
            raise ValueError(f"未知参数: {unknown_params}")
        
        # 转换为标准化向量
        x = np.zeros(len(self.parameters))
        
        for i, param in enumerate(self.parameters):
            if param.name in params_dict:
                try:
                    x[i] = param.inverse_transform(params_dict[param.name])
                except Exception as e:
                    logger.error(f"转换参数 '{param.name}' 失败: {e}")
                    raise
        
        return x
    
    def inverse_transform(self, x: np.ndarray) -> Dict[str, Any]:
        """
        将标准化的向量表示转换为参数字典
        
        Args:
            x: 标准化的向量表示
            
        Returns:
            Dict[str, Any]: 参数字典 {参数名: 参数值}
        """
        if len(x) != len(self.parameters):
            raise ValueError(f"向量长度 {len(x)} 与参数数量 {len(self.parameters)} 不匹配")
        
        # 转换为参数字典
        params_dict = {}
        
        for i, param in enumerate(self.parameters):
            try:
                params_dict[param.name] = param.transform(x[i])
            except Exception as e:
                logger.error(f"转换向量位置 {i} 为参数 '{param.name}' 失败: {e}")
                raise
        
        return params_dict
    
    def check_constraints(self, params_dict: Dict[str, Any]) -> bool:
        """
        检查参数是否满足约束条件
        
        Args:
            params_dict: 参数字典 {参数名: 参数值}
            
        Returns:
            bool: 是否满足所有约束条件
        """
        # 检查单个参数的约束条件
        for param_name, param_value in params_dict.items():
            param = self.get_parameter(param_name)
            
            for constraint in param.constraints:
                try:
                    # 创建变量字典用于表达式计算
                    var_dict = {"x": param_value, **params_dict}
                    
                    if not safe_eval(constraint, var_dict):
                        logger.debug(f"参数 '{param_name}' 不满足约束: {constraint}")
                        return False
                except Exception as e:
                    logger.error(f"计算参数 '{param_name}' 的约束 '{constraint}' 时出错: {e}")
                    return False
        
        # 检查全局约束条件
        for constraint in self.constraints:
            try:
                if not safe_eval(constraint, params_dict):
                    logger.debug(f"参数不满足全局约束: {constraint}")
                    return False
            except Exception as e:
                logger.error(f"计算全局约束 '{constraint}' 时出错: {e}")
                return False
        
        return True
    
    def sample_parameters(self, n_samples: int = 1, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        采样参数组合
        
        Args:
            n_samples: 采样数量
            seed: 随机种子
            
        Returns:
            List[Dict[str, Any]]: 参数字典列表
        """
        rng = np.random.RandomState(seed)
        
        # 采样标准化向量
        samples = []
        max_attempts = n_samples * 10  # 最大尝试次数
        attempts = 0
        
        while len(samples) < n_samples and attempts < max_attempts:
            x = np.zeros((1, len(self.parameters)))
            
            # 为每个参数生成随机值
            for i, param in enumerate(self.parameters):
                x[0, i] = rng.uniform(0, 1)
            
            # 转换为参数字典
            sample_dict = self.inverse_transform(x[0])
            
            # 检查约束条件
            if self.check_constraints(sample_dict):
                samples.append(sample_dict)
            
            attempts += 1
        
        if len(samples) < n_samples:
            logger.warning(f"无法生成 {n_samples} 个满足约束条件的样本，仅生成了 {len(samples)} 个")
        
        return samples
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将参数空间转换为字典表示
        
        Returns:
            Dict[str, Any]: 参数空间的字典表示
        """
        return {
            "parameters": [param.to_dict() for param in self.parameters],
            "constraints": self.constraints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterSpace':
        """
        从字典创建参数空间对象
        
        Args:
            data: 参数空间的字典表示
            
        Returns:
            ParameterSpace: 参数空间对象
        """
        parameters = []
        
        for param_data in data.get("parameters", []):
            param_type = param_data.get("parameter_type")
            
            if param_type == ParameterType.CONTINUOUS.name:
                param = ContinuousParameter.from_dict(param_data)
            elif param_type == ParameterType.INTEGER.name:
                param = IntegerParameter.from_dict(param_data)
            elif param_type == ParameterType.CATEGORICAL.name:
                param = CategoricalParameter.from_dict(param_data)
            elif param_type == ParameterType.ORDINAL.name:
                param = OrdinalParameter.from_dict(param_data)
            else:
                param = Parameter.from_dict(param_data)
            
            parameters.append(param)
        
        return cls(
            parameters=parameters,
            constraints=data.get("constraints", [])
        )
    
    def save(self, filepath: str) -> None:
        """
        保存参数空间到文件
        
        Args:
            filepath: 文件路径
        """
        data = self.to_dict()
        save_to_json(data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'ParameterSpace':
        """
        从文件加载参数空间
        
        Args:
            filepath: 文件路径
            
        Returns:
            ParameterSpace: 参数空间对象
        """
        data = load_from_json(filepath)
        return cls.from_dict(data)
