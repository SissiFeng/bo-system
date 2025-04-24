from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import copy
import re
import logging
from enum import Enum
import random

from bo_engine.utils import (
    validate_expression, 
    evaluate_expression, 
    one_hot_encode, 
    one_hot_decode, 
    scale_parameters, 
    unscale_parameters
)

# Setup logger
logger = logging.getLogger("bo_engine.parameter_space")

class ParameterType(str, Enum):
    """Enum for parameter types."""
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    DISCRETE = "discrete"


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
    def __init__(self, name: str, parameter_type: ParameterType):
        """
        Initialize a parameter.
        
        Args:
            name: Parameter name
            parameter_type: Type of parameter
        """
        self.name = name
        self.type = parameter_type
    
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
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter to dictionary representation."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Parameter':
        """Create parameter from dictionary representation."""
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
    """
    Parameter with continuous (real) values within a range.
    """
    def __init__(self, name: str, min_value: float, max_value: float):
        """
        Initialize a continuous parameter.
        
        Args:
            name: Parameter name
            min_value: Minimum value
            max_value: Maximum value
        """
        super().__init__(name, ParameterType.CONTINUOUS)
        self.min = float(min_value)
        self.max = float(max_value)
    
    def validate(self) -> bool:
        """
        Validate that min is less than max.
        
        Returns:
            bool: True if valid, False otherwise
        """
        return self.min < self.max
    
    def sample(self) -> float:
        """
        Generate a random sample from uniform distribution.
        
        Returns:
            float: Random value between min and max
        """
        return random.uniform(self.min, self.max)
    
    def contains(self, value: float) -> bool:
        """
        Check if a value is within the parameter range.
        
        Args:
            value: Value to check
            
        Returns:
            bool: True if min <= value <= max
        """
        return self.min <= value <= self.max
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameter to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "name": self.name,
            "type": self.type.value,
            "min": self.min,
            "max": self.max
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContinuousParameter':
        """
        Create parameter from dictionary representation.
        
        Args:
            data: Dictionary with parameter definition
            
        Returns:
            ContinuousParameter: Created parameter
        """
        return cls(
            name=data["name"],
            min_value=data["min"],
            max_value=data["max"]
        )
    
    def to_internal(self, value: float) -> float:
        """
        Scale value to [0, 1] range.
        
        Args:
            value: Original value
            
        Returns:
            float: Scaled value
        """
        return (value - self.min) / (self.max - self.min)
    
    def from_internal(self, internal_value: float) -> float:
        """
        Convert scaled [0, 1] value back to original range.
        
        Args:
            internal_value: Scaled value
            
        Returns:
            float: Original value
        """
        return self.min + internal_value * (self.max - self.min)
    
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


class DiscreteParameter(Parameter):
    """
    Parameter with discrete integer values within a range.
    """
    def __init__(self, name: str, min_value: int, max_value: int, step: int = 1):
        """
        Initialize a discrete parameter.
        
        Args:
            name: Parameter name
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            step: Step size between values
        """
        super().__init__(name, ParameterType.DISCRETE)
        self.min = int(min_value)
        self.max = int(max_value)
        self.step = int(step)
    
    def validate(self) -> bool:
        """
        Validate that parameter definition is valid.
        
        Returns:
            bool: True if valid, False otherwise
        """
        return (
            self.min < self.max and 
            self.step > 0 and 
            (self.max - self.min) % self.step == 0
        )
    
    def get_values(self) -> List[int]:
        """
        Get all possible values for this parameter.
        
        Returns:
            List[int]: List of all possible values
        """
        return list(range(self.min, self.max + 1, self.step))
    
    def sample(self) -> int:
        """
        Generate a random sample.
        
        Returns:
            int: Random value from possible values
        """
        values = self.get_values()
        return random.choice(values)
    
    def contains(self, value: int) -> bool:
        """
        Check if a value is valid for this parameter.
        
        Args:
            value: Value to check
            
        Returns:
            bool: True if value is valid
        """
        if not isinstance(value, (int, float)) or int(value) != value:
            return False
        
        value = int(value)
        if not (self.min <= value <= self.max):
            return False
        
        return (value - self.min) % self.step == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameter to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "name": self.name,
            "type": self.type.value,
            "min": self.min,
            "max": self.max,
            "step": self.step
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiscreteParameter':
        """
        Create parameter from dictionary representation.
        
        Args:
            data: Dictionary with parameter definition
            
        Returns:
            DiscreteParameter: Created parameter
        """
        return cls(
            name=data["name"],
            min_value=data["min"],
            max_value=data["max"],
            step=data.get("step", 1)
        )
    
    def to_internal(self, value: int) -> float:
        """
        Convert discrete value to continuous internal representation.
        
        Args:
            value: Original discrete value
            
        Returns:
            float: Internal representation in [0, 1] range
        """
        # Normalize to [0, 1] range
        num_steps = (self.max - self.min) // self.step
        step_index = (value - self.min) // self.step
        return step_index / num_steps if num_steps > 0 else 0.5
    
    def from_internal(self, internal_value: float) -> int:
        """
        Convert internal continuous representation to discrete value.
        
        Args:
            internal_value: Internal value in [0, 1] range
            
        Returns:
            int: Original discrete value
        """
        # Clip to [0, 1] range
        internal_value = max(0.0, min(1.0, internal_value))
        
        # Convert to step index
        num_steps = (self.max - self.min) // self.step
        step_index = round(internal_value * num_steps)
        
        # Convert to actual value
        value = self.min + step_index * self.step
        
        # Ensure value is within bounds
        return min(self.max, max(self.min, value))
    
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
            int: Dimensionality (1 for discrete parameters)
        """
        return 1


class CategoricalParameter(Parameter):
    """
    Parameter with categorical values from a fixed set.
    """
    def __init__(self, name: str, values: List[Any]):
        """
        Initialize a categorical parameter.
        
        Args:
            name: Parameter name
            values: List of possible values
        """
        super().__init__(name, ParameterType.CATEGORICAL)
        self.values = list(values)  # Make a copy
    
    def validate(self) -> bool:
        """
        Validate that parameter definition is valid.
        
        Returns:
            bool: True if valid, False otherwise
        """
        return len(self.values) > 0 and len(set(self.values)) == len(self.values)
    
    def sample(self) -> Any:
        """
        Generate a random sample.
        
        Returns:
            Any: Random value from possible values
        """
        return random.choice(self.values)
    
    def contains(self, value: Any) -> bool:
        """
        Check if a value is valid for this parameter.
        
        Args:
            value: Value to check
            
        Returns:
            bool: True if value is in possible values
        """
        return value in self.values
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameter to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "name": self.name,
            "type": self.type.value,
            "values": self.values
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CategoricalParameter':
        """
        Create parameter from dictionary representation.
        
        Args:
            data: Dictionary with parameter definition
            
        Returns:
            CategoricalParameter: Created parameter
        """
        return cls(
            name=data["name"],
            values=data["values"]
        )
    
    def to_internal(self, value: Any) -> List[float]:
        """
        Convert categorical value to one-hot encoded representation.
        
        Args:
            value: Original categorical value
            
        Returns:
            List[float]: One-hot encoded representation
        """
        if not self.contains(value):
            raise ValueError(f"Value '{value}' is not valid for parameter '{self.name}'")
        
        encoding = [0.0] * len(self.values)
        encoding[self.values.index(value)] = 1.0
        return encoding
    
    def from_internal(self, internal_value: List[float]) -> Any:
        """
        Convert one-hot encoded representation to categorical value.
        
        Args:
            internal_value: One-hot encoded representation
            
        Returns:
            Any: Original categorical value
        """
        if len(internal_value) != len(self.values):
            raise ValueError(f"Internal value has wrong length: expected {len(self.values)}, got {len(internal_value)}")
        
        # Find index of maximum value
        max_index = np.argmax(internal_value)
        return self.values[max_index]
    
    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """
        Get bounds for internal representation.
        
        Returns:
            Tuple[List[float], List[float]]: Tuple of (lower_bounds, upper_bounds)
        """
        dim = len(self.values)
        return ([0.0] * dim, [1.0] * dim)
    
    def get_dimensionality(self) -> int:
        """
        Get dimensionality of internal representation.
        
        Returns:
            int: Dimensionality (number of categories)
        """
        return len(self.values)


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
    """
    Parameter space for optimization, including parameters, objectives, and constraints.
    """
    def __init__(
        self, 
        name: str,
        parameters: List[Parameter] = None, 
        objectives: List[Objective] = None, 
        constraints: List[Constraint] = None
    ):
        """
        Initialize a parameter space.
        
        Args:
            name: Name of the parameter space
            parameters: List of parameters
            objectives: List of objectives
            constraints: List of constraints
        """
        self.name = name
        self.parameters = parameters or []
        self.objectives = objectives or []
        self.constraints = constraints or []
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate that the parameter space is valid.
        
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Check parameter names are unique
        parameter_names = [p.name for p in self.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            return False, "Parameter names must be unique"
        
        # Check objective names are unique
        objective_names = [o.name for o in self.objectives]
        if len(objective_names) != len(set(objective_names)):
            return False, "Objective names must be unique"
        
        # Check all parameters are valid
        for param in self.parameters:
            if not param.validate():
                return False, f"Invalid parameter: {param.name}"
        
        # Check all constraints reference valid parameters
        for constraint in self.constraints:
            if not validate_expression(constraint.expression, parameter_names):
                return False, f"Invalid constraint expression: {constraint.expression}"
        
        # Need at least one parameter and one objective
        if not self.parameters:
            return False, "At least one parameter required"
        if not self.objectives:
            return False, "At least one objective required"
        
        return True, None
    
    def get_parameter(self, name: str) -> Optional[Parameter]:
        """
        Get a parameter by name.
        
        Args:
            name: Parameter name
            
        Returns:
            Optional[Parameter]: Parameter with given name, or None if not found
        """
        for param in self.parameters:
            if param.name == name:
                return param
        return None
    
    def get_objective(self, name: str) -> Optional[Objective]:
        """
        Get an objective by name.
        
        Args:
            name: Objective name
            
        Returns:
            Optional[Objective]: Objective with given name, or None if not found
        """
        for obj in self.objectives:
            if obj.name == name:
                return obj
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameter space to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "name": self.name,
            "parameters": [p.to_dict() for p in self.parameters],
            "objectives": [o.to_dict() for o in self.objectives],
            "constraints": [c.to_dict() for c in self.constraints]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterSpace':
        """
        Create parameter space from dictionary representation.
        
        Args:
            data: Dictionary with parameter space definition
            
        Returns:
            ParameterSpace: Created parameter space
        """
        parameters = []
        for param_data in data["parameters"]:
            param_type = ParameterType(param_data["type"])
            
            if param_type == ParameterType.CONTINUOUS:
                parameter = ContinuousParameter.from_dict(param_data)
            elif param_type == ParameterType.DISCRETE:
                parameter = DiscreteParameter.from_dict(param_data)
            elif param_type == ParameterType.CATEGORICAL:
                parameter = CategoricalParameter.from_dict(param_data)
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
            
            parameters.append(parameter)
        
        objectives = [Objective.from_dict(obj_data) for obj_data in data["objectives"]]
        
        constraints = []
        if "constraints" in data and data["constraints"]:
            constraints = [Constraint.from_dict(c_data) for c_data in data["constraints"]]
        
        return cls(
            name=data["name"],
            parameters=parameters,
            objectives=objectives,
            constraints=constraints
        )
    
    def is_valid_point(self, point: Dict[str, Any]) -> bool:
        """
        Check if a point is valid in this parameter space.
        
        Args:
            point: Dictionary mapping parameter names to values
            
        Returns:
            bool: True if point is valid, False otherwise
        """
        # Check all parameters are present and valid
        for param in self.parameters:
            if param.name not in point:
                return False
            if not param.contains(point[param.name]):
                return False
        
        # Check all constraints are satisfied
        for constraint in self.constraints:
            if not constraint.evaluate(point):
                return False
        
        return True
    
    def sample_random(self, max_attempts: int = 100) -> Optional[Dict[str, Any]]:
        """
        Sample a random valid point from the parameter space.
        
        Args:
            max_attempts: Maximum number of attempts to find valid point
            
        Returns:
            Optional[Dict[str, Any]]: Valid random point, or None if couldn't find one
        """
        for _ in range(max_attempts):
            # Sample random values for each parameter
            point = {param.name: param.sample() for param in self.parameters}
            
            # Check if point satisfies all constraints
            if all(constraint.evaluate(point) for constraint in self.constraints):
                return point
        
        logger.warning(f"Could not find valid point after {max_attempts} attempts")
        return None
    
    def sample_random_batch(self, n: int, max_attempts_per_point: int = 100) -> List[Dict[str, Any]]:
        """
        Sample multiple random valid points from the parameter space.
        
        Args:
            n: Number of points to sample
            max_attempts_per_point: Maximum attempts per point
            
        Returns:
            List[Dict[str, Any]]: List of valid random points
        """
        points = []
        for _ in range(n):
            point = self.sample_random(max_attempts=max_attempts_per_point)
            if point is not None:
                points.append(point)
        
        return points
    
    def point_to_internal(self, point: Dict[str, Any]) -> np.ndarray:
        """
        Convert a point to internal representation for the surrogate model.
        
        Args:
            point: Dictionary mapping parameter names to values
            
        Returns:
            np.ndarray: Internal representation as a flat array
        """
        internal_values = []
        
        for param in self.parameters:
            if param.name not in point:
                raise ValueError(f"Missing parameter: {param.name}")
            
            value = point[param.name]
            internal = param.to_internal(value)
            
            # Add to flat array
            if isinstance(internal, list):
                internal_values.extend(internal)
            else:
                internal_values.append(internal)
        
        return np.array(internal_values)
    
    def internal_to_point(self, internal_values: np.ndarray) -> Dict[str, Any]:
        """
        Convert internal representation back to a point.
        
        Args:
            internal_values: Internal representation as a flat array
            
        Returns:
            Dict[str, Any]: Dictionary mapping parameter names to values
        """
        point = {}
        index = 0
        
        for param in self.parameters:
            dim = param.get_dimensionality()
            
            if index + dim > len(internal_values):
                raise ValueError(f"Internal values array too short: expected at least {index + dim}, got {len(internal_values)}")
            
            # Get the slice of internal values for this parameter
            if dim == 1:
                value = param.from_internal(internal_values[index])
            else:
                value = param.from_internal(internal_values[index:index+dim].tolist())
            
            point[param.name] = value
            index += dim
        
        return point
    
    def get_internal_dimensions(self) -> int:
        """
        Get the dimensionality of the internal representation.
        
        Returns:
            int: Total dimensionality
        """
        return sum(param.get_dimensionality() for param in self.parameters)
    
    def get_internal_bounds(self) -> Tuple[List[float], List[float]]:
        """
        Get bounds for the internal representation.
        
        Returns:
            Tuple[List[float], List[float]]: Tuple of (lower_bounds, upper_bounds)
        """
        lower_bounds = []
        upper_bounds = []
        
        for param in self.parameters:
            lower, upper = param.get_bounds()
            lower_bounds.extend(lower)
            upper_bounds.extend(upper)
        
        return lower_bounds, upper_bounds 
