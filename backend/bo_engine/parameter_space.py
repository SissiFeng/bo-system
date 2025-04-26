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

from .utils import (
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
    """Parameter type enumeration"""
    CONTINUOUS = auto()  # Continuous parameter
    INTEGER = auto()     # Integer parameter
    CATEGORICAL = auto() # Categorical parameter
    ORDINAL = auto()     # Ordinal parameter


class ObjectiveDirection(str, Enum):
    """Enum for objective directions."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ConstraintRelation(str, Enum):
    """Enum for constraint relations."""
    LESS_THAN_OR_EQUAL = "<="
    GREATER_THAN_OR_EQUAL = ">="
    EQUAL = "=="
    CUSTOM = "custom"  # For backward compatibility


# For backward compatibility
class ConstraintType(str, Enum):
    """Enum for constraint types (deprecated, use ConstraintRelation instead)."""
    SUM_EQUALS = "sum_equals"
    SUM_LESS_THAN = "sum_less_than"
    SUM_GREATER_THAN = "sum_greater_than"
    PRODUCT_EQUALS = "product_equals"
    CUSTOM = "custom"


# For backward compatibility
class ObjectiveType(str, Enum):
    """Enum for objective types (deprecated, use ObjectiveDirection instead)."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


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
    def __init__(self, name: str, objective_type: ObjectiveDirection, weight: float = 1.0):
        """
        Initialize an objective.

        Args:
            name: Objective name
            objective_type: Type of objective (maximize or minimize)
            weight: Weight for this objective in multi-objective optimization (default: 1.0)
        """
        self.name = name
        self.type = objective_type
        self.weight = weight

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert objective to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "name": self.name,
            "type": self.type.value,
            "weight": self.weight
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
            objective_type=ObjectiveDirection(data["type"]),
            weight=data.get("weight", 1.0)
        )


class Constraint:
    """
    Constraint for optimization.
    """
    def __init__(self, expression: str, constraint_type: ConstraintRelation, value: float):
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
        self._expression_tree = None  # Will be populated by _parse_expression when needed

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
        if self.type == ConstraintRelation.LESS_THAN_OR_EQUAL:
            return result <= self.value
        elif self.type == ConstraintRelation.GREATER_THAN_OR_EQUAL:
            return result >= self.value
        elif self.type == ConstraintRelation.EQUAL:
            return abs(result - self.value) < 1e-6
        else:  # CUSTOM
            return abs(result) < 1e-6

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert constraint to dictionary representation with expression tree for visualization.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        result = {
            "expression": self.expression,
            "type": self.type.value,
            "value": self.value
        }

        # Add expression tree for visualization if needed
        if self._expression_tree is None:
            self._parse_expression()

        if self._expression_tree:
            result["expression_tree"] = self._expression_tree

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Constraint':
        """
        Create constraint from dictionary representation.

        Args:
            data: Dictionary with constraint definition

        Returns:
            Constraint: Created constraint
        """
        constraint = cls(
            expression=data["expression"],
            constraint_type=ConstraintRelation(data["type"]),
            value=data["value"]
        )

        # If expression tree is provided, store it directly
        if "expression_tree" in data:
            constraint._expression_tree = data["expression_tree"]

        return constraint

    def _parse_expression(self):
        """
        Parse the expression into a tree structure for visualization.
        This is a simple implementation that can be enhanced for more complex expressions.
        """
        try:
            # This is a placeholder implementation
            # For a real implementation, you would use a proper parser
            # Below is just a simple structure to demonstrate the concept
            if "<=" in self.expression:
                left, right = self.expression.split("<=")
                op = "<="
            elif ">=" in self.expression:
                left, right = self.expression.split(">=")
                op = ">="
            elif "==" in self.expression:
                left, right = self.expression.split("==")
                op = "=="
            else:
                # Default case or other operators
                self._expression_tree = {"type": "raw", "expression": self.expression}
                return

            # Create a simple tree
            self._expression_tree = {
                "type": "operation",
                "operator": op,
                "left": self._parse_expression_term(left.strip()),
                "right": self._parse_expression_term(right.strip())
            }
        except Exception as e:
            logger.warning(f"Failed to parse expression '{self.expression}' into tree: {str(e)}")
            self._expression_tree = {"type": "raw", "expression": self.expression}

    def _parse_expression_term(self, term: str) -> Dict[str, Any]:
        """
        Parse an expression term into a tree node.

        Args:
            term: Expression term to parse

        Returns:
            Dict[str, Any]: Tree node representing the term
        """
        # Very simple parsing logic - this should be replaced with a proper parser
        if "+" in term:
            parts = term.split("+")
            return {
                "type": "operation",
                "operator": "+",
                "operands": [self._parse_expression_term(p.strip()) for p in parts if p.strip()]
            }
        elif "*" in term:
            parts = term.split("*")
            return {
                "type": "operation",
                "operator": "*",
                "operands": [self._parse_expression_term(p.strip()) for p in parts if p.strip()]
            }
        else:
            # Try to parse as a number
            try:
                value = float(term)
                return {"type": "constant", "value": value}
            except ValueError:
                # It's a variable
                return {"type": "variable", "name": term}


class ParameterGroup(ABC):
    """
    Abstract base class for groups of related parameters.
    """

    def __init__(self, name: str, parameters: List[str], description: str = ""):
        """
        Initialize a parameter group.

        Args:
            name: Name of the parameter group
            parameters: List of parameter names in the group
            description: Description of the parameter group
        """
        self.name = name
        self.parameters = parameters
        self.description = description

    @abstractmethod
    def validate(self, values: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that the parameter values satisfy the group constraints.

        Args:
            values: Dictionary of parameter values

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameter group to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterGroup':
        """
        Create parameter group from dictionary representation.

        Args:
            data: Dictionary with parameter group definition

        Returns:
            ParameterGroup: Created parameter group
        """
        pass


class SimplexGroup(ParameterGroup):
    """
    Group of parameters that must sum to 1 (simplex constraint).
    Useful for representing compositions, e.g., metal alloy components.
    """

    def __init__(self, name: str, parameters: List[str], description: str = "", tolerance: float = 1e-6):
        """
        Initialize a simplex parameter group.

        Args:
            name: Name of the parameter group
            parameters: List of parameter names in the group
            description: Description of the parameter group
            tolerance: Tolerance for the sum to be considered equal to 1
        """
        super().__init__(name, parameters, description)
        self.tolerance = tolerance

    def validate(self, values: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that the parameter values sum to 1 (within tolerance).

        Args:
            values: Dictionary of parameter values

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check if all parameters are present
        for param in self.parameters:
            if param not in values:
                return False, f"Missing parameter '{param}' in simplex group '{self.name}'"

        # Check if the sum is 1 (within tolerance)
        param_sum = sum(values[param] for param in self.parameters)
        if abs(param_sum - 1.0) > self.tolerance:
            return False, f"Parameters in simplex group '{self.name}' sum to {param_sum}, which is not 1.0 (±{self.tolerance})"

        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert simplex group to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "type": "simplex",
            "name": self.name,
            "parameters": self.parameters,
            "description": self.description,
            "tolerance": self.tolerance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimplexGroup':
        """
        Create simplex group from dictionary representation.

        Args:
            data: Dictionary with simplex group definition

        Returns:
            SimplexGroup: Created simplex group
        """
        return cls(
            name=data["name"],
            parameters=data["parameters"],
            description=data.get("description", ""),
            tolerance=data.get("tolerance", 1e-6)
        )


class ParameterSpace:
    """
    Parameter space class for defining optimization parameters and constraints
    """

    def __init__(
        self,
        parameters: Dict[str, Dict[str, Any]],
        constraints: Optional[List[Dict[str, Any]]] = None,
        name: str = "Optimization Task",
        objectives: Optional[List[Dict[str, Any]]] = None,
        description: str = "",
        parameter_groups: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize parameter space

        Args:
            parameters: Parameter definition dictionary
            constraints: List of constraints
            name: Name of the optimization task
            objectives: List of objectives
            description: Description of the parameter space
            parameter_groups: List of parameter groups (optional)
        """
        self.parameters = parameters
        self.constraints = constraints or []
        self.name = name
        self.objectives = objectives or []
        self.description = description
        self.parameter_groups = parameter_groups or []

        # Validate parameter definitions
        self._validate_parameters()

        # Validate constraints
        self._validate_constraints()

        # Validate parameter groups
        self._validate_parameter_groups()

        logger.info(f"Initialized parameter space, parameters: {len(parameters)}, constraints: {len(self.constraints)}, parameter groups: {len(self.parameter_groups)}")

    @classmethod
    def from_api_config(cls, config: Dict[str, Any]) -> 'ParameterSpace':
        """
        Create a parameter space from the API's declarative configuration format

        Args:
            config: API configuration dictionary with the following structure:
                {
                    "name": "Task name",
                    "parameters": [
                        {
                            "name": "param1",
                            "type": "continuous",
                            "bounds": [min, max],
                            "log_scale": false
                        },
                        {
                            "name": "param2",
                            "type": "integer",
                            "bounds": [min, max]
                        },
                        {
                            "name": "param3",
                            "type": "categorical",
                            "categories": ["cat1", "cat2", ...]
                        }
                    ],
                    "objectives": [
                        {
                            "name": "obj1",
                            "direction": "minimize"
                        }
                    ],
                    "constraints": [
                        {
                            "name": "constraint1",
                            "parameters": ["param1", "param2"],
                            "relation": "<=",
                            "threshold": 10.0
                        }
                    ],
                    "description": "Optional description"
                }

        Returns:
            ParameterSpace: Created parameter space object
        """
        # Extract basic info
        name = config.get("name", "Optimization Task")
        description = config.get("description", "")

        # Convert parameters from list to dictionary format
        parameters_dict = {}
        for param_config in config.get("parameters", []):
            param_name = param_config["name"]
            param_type = param_config["type"].lower()

            param_dict = {
                "type": param_type,
                "description": param_config.get("description", "")
            }

            # Add type-specific fields
            if param_type in ["continuous", "integer"]:
                param_dict["min"] = param_config["bounds"][0]
                param_dict["max"] = param_config["bounds"][1]
                if "log_scale" in param_config:
                    param_dict["log_scale"] = param_config["log_scale"]
                if "precision" in param_config and param_type == "continuous":
                    param_dict["precision"] = param_config["precision"]

            elif param_type == "categorical":
                param_dict["categories"] = param_config["categories"]

            parameters_dict[param_name] = param_dict

        # Convert objectives to the internal format
        objectives = []
        for obj_config in config.get("objectives", []):
            objectives.append({
                "name": obj_config["name"],
                "type": obj_config["direction"]  # Use direction as type for compatibility
            })

        # Convert constraints to the internal format
        constraints = []
        # 确保constraints字段存在且为列表
        constraints_list = config.get("constraints", []) or []
        for constraint_config in constraints_list:
            # For simple linear constraints (sum type)
            if constraint_config and "parameters" in constraint_config and len(constraint_config["parameters"]) > 0:
                params = constraint_config["parameters"]
                relation = constraint_config["relation"]
                threshold = constraint_config["threshold"]

                # Build an expression based on parameters
                if len(params) == 1:
                    # Single parameter constraint: param <= threshold
                    expression = f"{params[0]} {relation} {threshold}"
                else:
                    # Multi-parameter constraint: sum(params) <= threshold
                    params_expr = " + ".join(params)
                    expression = f"{params_expr} {relation} {threshold}"

                # Map relation to constraint type
                constraint_type = "nonlinear"  # Default to nonlinear for flexibility
                constraints.append({
                    "type": constraint_type,
                    "expression": expression,
                    "description": constraint_config.get("name", "")
                })

        # Create and return the parameter space
        return cls(
            parameters=parameters_dict,
            constraints=constraints,
            name=name,
            objectives=objectives,
            description=description
        )

    def to_api_config(self) -> Dict[str, Any]:
        """
        Convert parameter space to the API's declarative configuration format

        Returns:
            Dict[str, Any]: API configuration dictionary
        """
        # Convert parameters to list format
        parameters = []
        for param_name, param_config in self.parameters.items():
            param_type = param_config["type"]

            if param_type in ["continuous", "integer"]:
                parameter = {
                    "name": param_name,
                    "type": param_type,
                    "bounds": [param_config["min"], param_config["max"]]
                }

                # Add optional fields if present
                if "log_scale" in param_config:
                    parameter["log_scale"] = param_config["log_scale"]
                if "precision" in param_config and param_type == "continuous":
                    parameter["precision"] = param_config["precision"]

            elif param_type == "categorical":
                parameter = {
                    "name": param_name,
                    "type": param_type,
                    "categories": param_config["categories"]
                }

            # Add description if present
            if "description" in param_config:
                parameter["description"] = param_config["description"]

            parameters.append(parameter)

        # Convert objectives
        objectives = []
        for obj in self.objectives:
            objectives.append({
                "name": obj["name"],
                "direction": obj["type"]  # Use type as direction for compatibility
            })

        # For now, we don't convert constraints back to API format as it's complex
        # This can be improved in the future if needed

        return {
            "name": self.name,
            "parameters": parameters,
            "objectives": objectives,
            "description": self.description
            # constraints are omitted for now
        }

    def validate(self) -> Tuple[bool, str]:
        """
        验证参数空间配置是否有效

        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        try:
            self._validate_parameters()
            self._validate_constraints()
            self._validate_parameter_groups()
            return True, ""
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            return False, f"验证参数空间时发生未知错误: {str(e)}"

    def _validate_parameters(self):
        """
        验证参数定义是否有效
        """
        for param_name, param_config in self.parameters.items():
            # 检查参数类型
            if "type" not in param_config:
                raise ValueError(f"参数 {param_name} 缺少 'type' 字段")

            param_type = param_config["type"].lower()
            if param_type not in ["continuous", "integer", "categorical"]:
                raise ValueError(f"参数 {param_name} 的类型 '{param_type}' 无效，有效类型为: continuous, integer, categorical")

            # 根据参数类型验证必要字段
            if param_type in ["continuous", "integer"]:
                if "min" not in param_config:
                    raise ValueError(f"参数 {param_name} 缺少 'min' 字段")
                if "max" not in param_config:
                    raise ValueError(f"参数 {param_name} 缺少 'max' 字段")

                min_val = param_config["min"]
                max_val = param_config["max"]

                if min_val > max_val:
                    raise ValueError(f"参数 {param_name} 的最小值 {min_val} 大于最大值 {max_val}")

                if param_type == "integer":
                    # 整数参数的值应该是整数
                    if not isinstance(min_val, int):
                        logger.warning(f"参数 {param_name} 的最小值 {min_val} 不是整数，将被转换为整数")
                        param_config["min"] = int(min_val)

                    if not isinstance(max_val, int):
                        logger.warning(f"参数 {param_name} 的最大值 {max_val} 不是整数，将被转换为整数")
                        param_config["max"] = int(max_val)

            elif param_type == "categorical":
                if "categories" not in param_config:
                    raise ValueError(f"参数 {param_name} 缺少 'categories' 字段")

                categories = param_config["categories"]
                if not isinstance(categories, list) or len(categories) == 0:
                    raise ValueError(f"参数 {param_name} 的类别必须是非空列表")

                # 检查类别是否唯一
                if len(categories) != len(set(map(str, categories))):
                    raise ValueError(f"参数 {param_name} 的类别包含重复值")

    def _validate_constraints(self):
        """
        验证约束条件是否有效
        """
        for i, constraint in enumerate(self.constraints):
            # 检查约束类型
            if "type" not in constraint:
                raise ValueError(f"约束条件 #{i} 缺少 'type' 字段")

            constraint_type = constraint["type"]
            if constraint_type not in [ConstraintRelation.LESS_THAN_OR_EQUAL, ConstraintRelation.GREATER_THAN_OR_EQUAL, ConstraintRelation.EQUAL]:
                raise ValueError(f"约束条件 #{i} 的类型 '{constraint_type}' 无效，有效类型为: {ConstraintRelation.LESS_THAN_OR_EQUAL}, {ConstraintRelation.GREATER_THAN_OR_EQUAL}, {ConstraintRelation.EQUAL}")

            # 检查约束表达式
            if "expression" not in constraint:
                raise ValueError(f"约束条件 #{i} 缺少 'expression' 字段")

            # 存储小写的约束类型
            constraint["type"] = constraint_type

    def _validate_parameter_groups(self):
        """
        Validate parameter groups to ensure all referenced parameters exist
        """
        for i, group_config in enumerate(self.parameter_groups):
            # Check group type
            if "type" not in group_config:
                raise ValueError(f"Parameter group #{i} is missing 'type' field")

            group_type = group_config["type"]
            if group_type not in ["simplex"]:  # Add more types as implemented
                raise ValueError(f"Parameter group #{i} has invalid type '{group_type}'")

            # Check if parameters exist
            if "parameters" not in group_config:
                raise ValueError(f"Parameter group #{i} is missing 'parameters' field")

            for param_name in group_config["parameters"]:
                if param_name not in self.parameters:
                    raise ValueError(f"Parameter group #{i} references non-existent parameter '{param_name}'")

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        获取参数定义

        Returns:
            Dict[str, Dict[str, Any]]: 参数定义字典
        """
        return self.parameters

    def get_parameter_names(self) -> List[str]:
        """
        获取参数名称列表

        Returns:
            List[str]: 参数名称列表
        """
        return list(self.parameters.keys())

    def get_parameter_types(self) -> Dict[str, str]:
        """
        获取参数类型字典

        Returns:
            Dict[str, str]: 参数类型字典，键为参数名称，值为参数类型
        """
        return {param_name: param_config["type"] for param_name, param_config in self.parameters.items()}

    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        获取连续参数和整数参数的取值范围

        Returns:
            Dict[str, Tuple[float, float]]: 参数范围字典，键为参数名称，值为(min, max)元组
        """
        ranges = {}
        for param_name, param_config in self.parameters.items():
            if param_config["type"] in ["continuous", "integer"]:
                ranges[param_name] = (param_config["min"], param_config["max"])
        return ranges

    def get_parameter_categories(self) -> Dict[str, List[Any]]:
        """
        获取分类参数的类别列表

        Returns:
            Dict[str, List[Any]]: 参数类别字典，键为参数名称，值为类别列表
        """
        categories = {}
        for param_name, param_config in self.parameters.items():
            if param_config["type"] == "categorical":
                categories[param_name] = param_config["categories"]
        return categories

    def get_constraints(self) -> List[Dict[str, Any]]:
        """
        获取约束条件列表

        Returns:
            List[Dict[str, Any]]: 约束条件列表
        """
        return self.constraints

    def has_constraints(self) -> bool:
        """
        检查是否存在约束条件

        Returns:
            bool: 是否存在约束条件
        """
        return len(self.constraints) > 0

    def check_constraints(self, design: Dict[str, Any]) -> bool:
        """
        检查设计方案是否满足约束条件

        Args:
            design: 设计方案

        Returns:
            bool: 是否满足约束条件
        """
        if not self.constraints:
            return True

        # 首先检查参数取值是否在范围内
        for param_name, param_config in self.parameters.items():
            if param_name not in design:
                logger.warning(f"设计方案缺少参数: {param_name}")
                return False

            value = design[param_name]
            param_type = param_config["type"]

            if param_type == "continuous":
                min_val = param_config["min"]
                max_val = param_config["max"]
                if value < min_val or value > max_val:
                    logger.debug(f"参数 {param_name} 的值 {value} 超出范围 [{min_val}, {max_val}]")
                    return False

            elif param_type == "integer":
                min_val = param_config["min"]
                max_val = param_config["max"]
                if not isinstance(value, int) or value < min_val or value > max_val:
                    logger.debug(f"参数 {param_name} 的值 {value} 不是有效的整数或超出范围 [{min_val}, {max_val}]")
                    return False

            elif param_type == "categorical":
                categories = param_config["categories"]
                if value not in categories:
                    logger.debug(f"参数 {param_name} 的值 {value} 不在有效类别中: {categories}")
                    return False

        # 然后检查约束条件
        for constraint in self.constraints:
            constraint_type = constraint["type"]
            expression = constraint["expression"]

            if constraint_type == ConstraintRelation.LESS_THAN_OR_EQUAL:
                # 线性约束，格式为: a1*x1 + a2*x2 + ... + an*xn <= b
                result = self._evaluate_linear_constraint(expression, design)
                if not result:
                    logger.debug(f"设计方案不满足线性约束: {expression}")
                    return False

            elif constraint_type == ConstraintRelation.GREATER_THAN_OR_EQUAL:
                # 线性约束，格式为: a1*x1 + a2*x2 + ... + an*xn >= b
                result = self._evaluate_linear_constraint(expression, design)
                if not result:
                    logger.debug(f"设计方案不满足线性约束: {expression}")
                    return False

            elif constraint_type == ConstraintRelation.EQUAL:
                # 线性约束，格式为: a1*x1 + a2*x2 + ... + an*xn == b
                result = self._evaluate_linear_constraint(expression, design)
                if not result:
                    logger.debug(f"设计方案不满足线性约束: {expression}")
                    return False

        return True

    def _evaluate_linear_constraint(self, expression: str, design: Dict[str, Any]) -> bool:
        """
        评估线性约束

        Args:
            expression: 约束表达式
            design: 设计方案

        Returns:
            bool: 是否满足约束条件
        """
        # 解析表达式
        if "<=" in expression:
            left, right = expression.split("<=")
            op = "<="
        elif ">=" in expression:
            left, right = expression.split(">=")
            op = ">="
        elif "<" in expression:
            left, right = expression.split("<")
            op = "<"
        elif ">" in expression:
            left, right = expression.split(">")
            op = ">"
        elif "==" in expression:
            left, right = expression.split("==")
            op = "=="
        else:
            logger.warning(f"无效的线性约束表达式: {expression}")
            return False

        # 计算左侧表达式的值
        left_value = 0
        terms = left.strip().replace(" ", "").replace("-", "+-").split("+")
        for term in terms:
            if not term:  # 处理连续的+号
                continue

            if "*" in term:
                coef, param = term.split("*")
                if param in design:
                    left_value += float(coef) * design[param]
                else:
                    logger.warning(f"线性约束使用了未知参数: {param}")
                    return False
            else:
                # 常数项
                try:
                    left_value += float(term)
                except ValueError:
                    if term in design:
                        left_value += design[term]
                    else:
                        logger.warning(f"线性约束使用了未知参数: {term}")
                        return False

        # 计算右侧表达式的值
        right_value = 0
        terms = right.strip().replace(" ", "").replace("-", "+-").split("+")
        for term in terms:
            if not term:  # 处理连续的+号
                continue

            if "*" in term:
                coef, param = term.split("*")
                if param in design:
                    right_value += float(coef) * design[param]
                else:
                    logger.warning(f"线性约束使用了未知参数: {param}")
                    return False
            else:
                # 常数项
                try:
                    right_value += float(term)
                except ValueError:
                    if term in design:
                        right_value += design[term]
                    else:
                        logger.warning(f"线性约束使用了未知参数: {term}")
                        return False

        # 比较大小
        if op == "<=":
            return left_value <= right_value
        elif op == ">=":
            return left_value >= right_value
        elif op == "<":
            return left_value < right_value
        elif op == ">":
            return left_value > right_value
        elif op == "==":
            return left_value == right_value

        return False

    def _evaluate_nonlinear_constraint(self, expression: str, design: Dict[str, Any]) -> bool:
        """
        评估非线性约束

        Args:
            expression: 约束表达式
            design: 设计方案

        Returns:
            bool: 是否满足约束条件
        """
        # 创建一个局部命名空间，包含设计参数和一些基本函数
        namespace = design.copy()
        namespace.update({
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "max": np.maximum,
            "min": np.minimum,
            "pi": np.pi,
            "e": np.e
        })

        try:
            # 评估表达式
            result = eval(expression, {"__builtins__": {}}, namespace)
            return bool(result)
        except Exception as e:
            logger.warning(f"评估非线性约束表达式出错: {expression}, 错误: {str(e)}")
            return False

    def sample_random(self, n_samples: int = 1, rng: Optional[np.random.RandomState] = None) -> List[Dict[str, Any]]:
        """
        从参数空间中随机采样

        Args:
            n_samples: 样本数量
            rng: 随机数生成器

        Returns:
            List[Dict[str, Any]]: 随机样本列表
        """
        if rng is None:
            rng = np.random.RandomState()

        samples = []
        for _ in range(n_samples):
            sample = {}

            for param_name, param_config in self.parameters.items():
                param_type = param_config["type"]

                if param_type == "continuous":
                    min_val = param_config["min"]
                    max_val = param_config["max"]
                    value = rng.uniform(min_val, max_val)
                    sample[param_name] = float(value)

                elif param_type == "integer":
                    min_val = param_config["min"]
                    max_val = param_config["max"]
                    value = rng.randint(min_val, max_val + 1)
                    sample[param_name] = int(value)

                elif param_type == "categorical":
                    categories = param_config["categories"]
                    value = rng.choice(categories)
                    sample[param_name] = value

            samples.append(sample)

        return samples

    def sample_random_batch(self, n_samples: int = 1, rng: Optional[np.random.RandomState] = None) -> List[Dict[str, Any]]:
        """
        从参数空间中随机采样一批样本
        这个方法是为了与DesignGenerator接口兼容

        Args:
            n_samples: 样本数量
            rng: 随机数生成器

        Returns:
            List[Dict[str, Any]]: 随机样本列表
        """
        return self.sample_random(n_samples, rng)

    def get_dimensions(self) -> int:
        """
        获取参数空间的维度

        Returns:
            int: 参数空间维度
        """
        dimensions = 0
        for param_name, param_config in self.parameters.items():
            param_type = param_config["type"]

            if param_type in ["continuous", "integer"]:
                dimensions += 1
            elif param_type == "categorical":
                categories = param_config["categories"]
                if len(categories) > 1:
                    # 对于分类参数，使用one-hot编码
                    dimensions += len(categories)
                else:
                    # 只有一个类别的分类参数不增加维度
                    pass

        return dimensions

    def get_internal_dimensions(self) -> int:
        """
        获取参数空间的内部表示维度，用于设计生成器

        Returns:
            int: 内部表示的维度数
        """
        # 默认情况下，内部维度与普通维度相同
        # 如果未来实现更复杂的表示，可以在此处修改
        return self.get_dimensions()

    def point_to_internal(self, point: Dict[str, Any]) -> np.ndarray:
        """
        将参数点转换为内部表示（归一化的向量）

        Args:
            point: 参数值字典

        Returns:
            np.ndarray: 内部表示向量
        """
        return self.transform(point)

    def internal_to_point(self, internal_vector: np.ndarray) -> Dict[str, Any]:
        """
        将内部表示向量转换为参数点

        Args:
            internal_vector: 内部表示向量

        Returns:
            Dict[str, Any]: 参数值字典
        """
        return self.inverse_transform(internal_vector)

    def is_valid_point(self, point: Dict[str, Any]) -> bool:
        """
        检查点是否满足所有参数约束

        Args:
            point: 参数点

        Returns:
            bool: 如果点满足所有约束则返回True
        """
        return self.check_constraints(point)

    def validate_point(self, point: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that a point is valid according to parameter definitions, constraints, and parameter groups

        Args:
            point: Point to validate

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check that all required parameters are present
        for param_name in self.parameters:
            if param_name not in point:
                return False, f"Missing parameter: {param_name}"

        # Check individual parameter constraints
        for param_name, value in point.items():
            if param_name not in self.parameters:
                return False, f"Unknown parameter: {param_name}"

            param_config = self.parameters[param_name]
            param_type = param_config["type"]

            # Check type-specific constraints
            if param_type == "continuous":
                if not isinstance(value, (int, float)):
                    return False, f"Parameter {param_name} expects a number, got {type(value).__name__}"

                min_val = param_config["min"]
                max_val = param_config["max"]
                if value < min_val or value > max_val:
                    return False, f"Parameter {param_name} value {value} is outside bounds [{min_val}, {max_val}]"

            elif param_type == "integer":
                if not isinstance(value, int):
                    return False, f"Parameter {param_name} expects an integer, got {type(value).__name__}"

                min_val = param_config["min"]
                max_val = param_config["max"]
                if value < min_val or value > max_val:
                    return False, f"Parameter {param_name} value {value} is outside bounds [{min_val}, {max_val}]"

            elif param_type == "categorical":
                if value not in param_config["categories"]:
                    categories_str = ", ".join(str(c) for c in param_config["categories"])
                    return False, f"Parameter {param_name} value '{value}' is not one of the allowed categories: {categories_str}"

        # Check parameter group constraints
        for group_config in self.parameter_groups:
            group_type = group_config["type"]

            if group_type == "simplex":
                group = SimplexGroup.from_dict(group_config)
                is_valid, error_msg = group.validate(point)
                if not is_valid:
                    return False, error_msg

        # Check general constraints
        # (This is already implemented in the check_constraints method)
        if not self.check_constraints(point):
            return False, "Point does not satisfy one or more constraints"

        return True, ""

    def transform(self, design: Dict[str, Any]) -> np.ndarray:
        """
        将设计方案转换为标准化的数值向量（用于优化算法）

        Args:
            design: 设计方案

        Returns:
            np.ndarray: 标准化的数值向量
        """
        # 创建一个空的列表来存储转换后的值
        transformed_values = []

        for param_name, param_config in self.parameters.items():
            if param_name not in design:
                raise ValueError(f"设计方案缺少参数: {param_name}")

            value = design[param_name]
            param_type = param_config["type"]

            if param_type == "continuous":
                min_val = param_config["min"]
                max_val = param_config["max"]

                # 将连续值标准化到[0, 1]范围
                if max_val > min_val:
                    normalized_value = (value - min_val) / (max_val - min_val)
                else:
                    normalized_value = 0.5  # 如果最大值等于最小值，使用0.5

                transformed_values.append(normalized_value)

            elif param_type == "integer":
                min_val = param_config["min"]
                max_val = param_config["max"]

                # 将整数值标准化到[0, 1]范围
                if max_val > min_val:
                    normalized_value = (value - min_val) / (max_val - min_val)
                else:
                    normalized_value = 0.5  # 如果最大值等于最小值，使用0.5

                transformed_values.append(normalized_value)

            elif param_type == "categorical":
                categories = param_config["categories"]

                if len(categories) > 1:
                    # 使用one-hot编码
                    for category in categories:
                        transformed_values.append(1.0 if value == category else 0.0)
                else:
                    # 只有一个类别的分类参数不需要变换
                    pass

        return np.array(transformed_values)

    def inverse_transform(self, vec: np.ndarray) -> Dict[str, Any]:
        """
        将标准化的数值向量转换为设计方案

        Args:
            vec: 标准化的数值向量

        Returns:
            Dict[str, Any]: 设计方案
        """
        if len(vec) == 0:
            raise ValueError("输入向量不能为空")

        # 创建一个空的字典来存储转换后的设计方案
        design = {}

        # 跟踪向量索引
        vec_idx = 0

        for param_name, param_config in self.parameters.items():
            param_type = param_config["type"]

            if param_type == "continuous":
                # 从[0, 1]范围转换为参数范围
                min_val = param_config["min"]
                max_val = param_config["max"]
                value = min_val + vec[vec_idx] * (max_val - min_val)
                design[param_name] = float(value)
                vec_idx += 1

            elif param_type == "integer":
                # 从[0, 1]范围转换为参数范围，并四舍五入为整数
                min_val = param_config["min"]
                max_val = param_config["max"]
                value = min_val + vec[vec_idx] * (max_val - min_val)
                design[param_name] = int(round(value))
                vec_idx += 1

            elif param_type == "categorical":
                categories = param_config["categories"]

                if len(categories) > 1:
                    # 从one-hot编码转换回类别
                    category_values = vec[vec_idx:vec_idx + len(categories)]
                    category_idx = np.argmax(category_values)
                    design[param_name] = categories[category_idx]
                    vec_idx += len(categories)
                else:
                    # 只有一个类别的分类参数直接使用该类别
                    design[param_name] = categories[0]

        return design

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameter space to dictionary representation

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        result = {
            "name": self.name,
            "parameters": self.parameters,
            "constraints": self.constraints,
            "objectives": self.objectives,
            "description": self.description
        }

        # Add parameter groups if present
        if self.parameter_groups:
            result["parameter_groups"] = self.parameter_groups

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterSpace':
        """
        Create parameter space from dictionary representation

        Args:
            data: Dictionary with parameter space definition

        Returns:
            ParameterSpace: Created parameter space
        """
        return cls(
            parameters=data["parameters"],
            constraints=data.get("constraints", []),
            name=data.get("name", "Optimization Task"),
            objectives=data.get("objectives", []),
            description=data.get("description", ""),
            parameter_groups=data.get("parameter_groups", [])
        )

    def save(self, filepath: str):
        """
        保存参数空间到文件

        Args:
            filepath: 文件路径
        """
        data = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"参数空间保存到文件: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ParameterSpace':
        """
        从文件加载参数空间

        Args:
            filepath: 文件路径

        Returns:
            ParameterSpace: 参数空间对象
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"从文件加载参数空间: {filepath}")
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"加载参数空间文件出错: {str(e)}")
            raise ValueError(f"无法加载参数空间文件: {filepath}, 错误: {str(e)}")
