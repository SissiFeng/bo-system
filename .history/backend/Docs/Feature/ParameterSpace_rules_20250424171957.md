# Parameter Space - Design and Implementation Rules

## Module Overview

The Parameter Space module is responsible for defining and managing the search space for Bayesian optimization. It provides a structured way to define continuous, integer, and categorical parameters, along with their bounds, constraints, and transformations. This module serves as the foundation for generating experimental designs and navigating the optimization landscape.

## Core Design Principles

1. **Type Safety**: Ensure parameters are strictly typed and properly validated
2. **Representation Conversion**: Maintain separate internal and external representations
3. **Validation Mechanisms**: Implement robust validation for parameter values and constraints
4. **Sampling Strategies**: Support diverse sampling methods for different parameter types
5. **Constraint Handling**: Allow defining and enforcing constraints across parameters
6. **Serialization Support**: Enable seamless conversion to/from JSON for API interactions

## Enumerations

### ParameterType

```python
class ParameterType(str, Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
```

### ObjectiveType

```python
class ObjectiveType(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
```

### ConstraintType

```python
class ConstraintType(str, Enum):
    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than"
    EQUAL_TO = "equal_to"
```

## Class Hierarchy

### Parameter (Abstract Base Class)

```python
class Parameter(ABC):
    def __init__(self, name, description=None):
        self.name = name
        self.description = description or ""
        
    @abstractmethod
    def validate(self, value):
        """Validate if a value is valid for this parameter"""
        pass
        
    @abstractmethod
    def sample(self, n=1, rng=None):
        """Sample n values from the parameter's domain"""
        pass
        
    @abstractmethod
    def to_internal(self, external_value):
        """Convert external representation to internal representation"""
        pass
        
    @abstractmethod
    def to_external(self, internal_value):
        """Convert internal representation to external representation"""
        pass
        
    @abstractmethod
    def to_dict(self):
        """Convert parameter to a dictionary representation"""
        pass
```

### ContinuousParameter

```python
class ContinuousParameter(Parameter):
    def __init__(self, name, lower_bound, upper_bound, 
                 log_scale=False, description=None):
        super().__init__(name, description)
        self.type = ParameterType.CONTINUOUS
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.log_scale = log_scale
        
    def validate(self, value):
        """Check if value is within bounds"""
        if not isinstance(value, (int, float)):
            return False
        return self.lower_bound <= value <= self.upper_bound
        
    def sample(self, n=1, rng=None):
        """Sample n values uniformly from the range"""
        rng = rng or np.random.default_rng()
        if self.log_scale:
            return np.exp(rng.uniform(
                np.log(self.lower_bound), 
                np.log(self.upper_bound), 
                size=n
            ))
        else:
            return rng.uniform(self.lower_bound, self.upper_bound, size=n)
```

### IntegerParameter

```python
class IntegerParameter(Parameter):
    def __init__(self, name, lower_bound, upper_bound, log_scale=False, description=None):
        super().__init__(name, description)
        self.type = ParameterType.INTEGER
        self.lower_bound = int(lower_bound)
        self.upper_bound = int(upper_bound)
        self.log_scale = log_scale
        
    def validate(self, value):
        """Check if value is an integer within bounds"""
        if not isinstance(value, int):
            return False
        return self.lower_bound <= value <= self.upper_bound
        
    def sample(self, n=1, rng=None):
        """Sample n integer values from the range"""
        rng = rng or np.random.default_rng()
        if self.log_scale:
            log_lower = np.log(max(1, self.lower_bound))
            log_upper = np.log(self.upper_bound)
            samples = np.exp(rng.uniform(log_lower, log_upper, size=n))
            return np.round(samples).astype(int)
        else:
            return rng.integers(self.lower_bound, self.upper_bound + 1, size=n)
```

### CategoricalParameter

```python
class CategoricalParameter(Parameter):
    def __init__(self, name, categories, description=None):
        super().__init__(name, description)
        self.type = ParameterType.CATEGORICAL
        self.categories = list(categories)
        self.category_map = {v: i for i, v in enumerate(self.categories)}
        
    def validate(self, value):
        """Check if value is in the list of categories"""
        return value in self.categories
        
    def sample(self, n=1, rng=None):
        """Sample n values from the categories"""
        rng = rng or np.random.default_rng()
        indices = rng.integers(0, len(self.categories), size=n)
        return [self.categories[i] for i in indices]
        
    def to_internal(self, external_value):
        """Convert category to its integer index"""
        if not self.validate(external_value):
            raise ValueError(f"Invalid value {external_value} for parameter {self.name}")
        return self.category_map[external_value]
        
    def to_external(self, internal_value):
        """Convert integer index to category"""
        if not isinstance(internal_value, int) or not (0 <= internal_value < len(self.categories)):
            raise ValueError(f"Invalid internal value {internal_value} for parameter {self.name}")
        return self.categories[internal_value]
```

### ParameterSpace

```python
class ParameterSpace:
    def __init__(self):
        self.parameters = {}
        self.objectives = {}
        self.constraints = []
        
    def add_parameter(self, parameter):
        """Add a parameter to the space"""
        if parameter.name in self.parameters:
            raise ValueError(f"Parameter {parameter.name} already exists")
        self.parameters[parameter.name] = parameter
        return self
        
    def add_objective(self, name, objective_type=ObjectiveType.MINIMIZE):
        """Add an objective to be optimized"""
        self.objectives[name] = objective_type
        return self
        
    def add_constraint(self, constraint_type, expression, value):
        """Add a constraint to the parameter space"""
        self.constraints.append({
            "type": constraint_type,
            "expression": expression,
            "value": value
        })
        return self
        
    def validate_point(self, point):
        """Check if a point satisfies all parameters and constraints"""
        # Validate individual parameters
        for name, parameter in self.parameters.items():
            if name not in point:
                return False, f"Missing parameter {name}"
            if not parameter.validate(point[name]):
                return False, f"Invalid value for parameter {name}: {point[name]}"
                
        # Validate constraints
        for constraint in self.constraints:
            # Evaluate constraint expression
            # Return False if any constraint is violated
            pass
            
        return True, ""
        
    def sample(self, n=1, method="random", rng=None):
        """Sample n points from the parameter space"""
        rng = rng or np.random.default_rng()
        result = []
        for _ in range(n):
            point = {}
            for name, param in self.parameters.items():
                point[name] = param.sample(1, rng)[0]
            result.append(point)
        return result
        
    def to_dict(self):
        """Convert parameter space to dictionary"""
        return {
            "parameters": {
                name: param.to_dict() 
                for name, param in self.parameters.items()
            },
            "objectives": {
                name: obj_type.value
                for name, obj_type in self.objectives.items()
            },
            "constraints": self.constraints
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create parameter space from dictionary"""
        space = cls()
        
        # Add parameters
        for name, param_data in data.get("parameters", {}).items():
            param_type = param_data.get("type")
            if param_type == ParameterType.CONTINUOUS:
                space.add_parameter(ContinuousParameter(
                    name=name,
                    lower_bound=param_data["lower_bound"],
                    upper_bound=param_data["upper_bound"],
                    log_scale=param_data.get("log_scale", False),
                    description=param_data.get("description", "")
                ))
            elif param_type == ParameterType.INTEGER:
                space.add_parameter(IntegerParameter(
                    name=name,
                    lower_bound=param_data["lower_bound"],
                    upper_bound=param_data["upper_bound"],
                    log_scale=param_data.get("log_scale", False),
                    description=param_data.get("description", "")
                ))
            elif param_type == ParameterType.CATEGORICAL:
                space.add_parameter(CategoricalParameter(
                    name=name,
                    categories=param_data["categories"],
                    description=param_data.get("description", "")
                ))
                
        # Add objectives
        for name, obj_type in data.get("objectives", {}).items():
            space.add_objective(name, ObjectiveType(obj_type))
            
        # Add constraints
        for constraint in data.get("constraints", []):
            space.add_constraint(
                ConstraintType(constraint["type"]),
                constraint["expression"],
                constraint["value"]
            )
            
        return space
```

## Key Methods

### Parameter Management

```python
def add_parameter(self, parameter):
    """Add a parameter to the space"""
    if parameter.name in self.parameters:
        raise ValueError(f"Parameter {parameter.name} already exists")
    self.parameters[parameter.name] = parameter
    return self
    
def get_parameter(self, name):
    """Get a parameter by name"""
    if name not in self.parameters:
        raise ValueError(f"Parameter {name} does not exist")
    return self.parameters[name]
    
def get_parameter_names(self):
    """Get all parameter names"""
    return list(self.parameters.keys())
    
def get_dimension(self):
    """Get the dimensionality of the parameter space"""
    return len(self.parameters)
```

### Objective Management

```python
def add_objective(self, name, objective_type=ObjectiveType.MINIMIZE):
    """Add an objective to be optimized"""
    self.objectives[name] = objective_type
    return self
    
def get_objectives(self):
    """Get all objectives"""
    return self.objectives
    
def is_multi_objective(self):
    """Check if this is a multi-objective optimization problem"""
    return len(self.objectives) > 1
```

### Constraint Management

```python
def add_constraint(self, constraint_type, expression, value):
    """Add a constraint to the parameter space"""
    self.constraints.append({
        "type": constraint_type,
        "expression": expression,
        "value": value
    })
    return self
    
def evaluate_constraints(self, point):
    """Evaluate all constraints for a given point"""
    results = []
    for constraint in self.constraints:
        # Evaluate constraint expression for the point
        # Check if constraint is satisfied
        pass
    return results
```

### Sampling and Validation

```python
def sample(self, n=1, method="random", rng=None):
    """
    Sample n points from the parameter space
    
    Parameters:
    -----------
    n : int
        Number of points to sample
    method : str
        Sampling method ('random', 'lhs', etc.)
    rng : numpy.random.Generator
        Random number generator
        
    Returns:
    --------
    List[Dict]
        List of parameter points
    """
    
def validate_point(self, point):
    """
    Check if a point satisfies all parameters and constraints
    
    Parameters:
    -----------
    point : Dict
        Dictionary of parameter name -> value pairs
        
    Returns:
    --------
    bool, str
        (True, "") if valid, (False, error_message) if invalid
    """
```

### Representation Conversion

```python
def to_internal_representation(self, external_point):
    """
    Convert a point from external to internal representation
    
    Parameters:
    -----------
    external_point : Dict
        Dictionary of parameter name -> external value pairs
        
    Returns:
    --------
    Dict
        Dictionary of parameter name -> internal value pairs
    """
    internal = {}
    for name, param in self.parameters.items():
        if name in external_point:
            internal[name] = param.to_internal(external_point[name])
    return internal
    
def to_external_representation(self, internal_point):
    """
    Convert a point from internal to external representation
    
    Parameters:
    -----------
    internal_point : Dict
        Dictionary of parameter name -> internal value pairs
        
    Returns:
    --------
    Dict
        Dictionary of parameter name -> external value pairs
    """
    external = {}
    for name, param in self.parameters.items():
        if name in internal_point:
            external[name] = param.to_external(internal_point[name])
    return external
```

## Data Flow

1. **Parameter Space Creation**:
   - User defines parameters, objectives, and constraints via API
   - System validates the definitions
   - ParameterSpace object is constructed and stored

2. **Design Point Generation**:
   - System generates design points using the parameter space
   - Points are validated against parameter bounds and constraints
   - Valid points are returned for experimentation

3. **Result Validation**:
   - Experimental results are validated against the parameter space
   - System checks that all parameters are present and valid
   - Objectives are recorded for the optimization process

4. **Surrogate Model Input/Output**:
   - Parameters are converted to internal representation for model training
   - Model outputs are validated against objective definitions
   - Predictions are converted back to external representation for API

## Code Validation Rules

1. **Type Checking**: Ensure correct types for all parameter values
2. **Bounds Validation**: Verify values are within specified bounds
3. **Categorical Validation**: Check categorical values against allowed categories
4. **Constraint Validation**: Validate all constraints are satisfied
5. **Dimensionality Consistency**: Ensure consistent dimensionality in operations
6. **Transformation Validity**: Verify transformations (e.g., log) are valid for the domain

## Future Expansion Plans

1. **Derived Parameters**: Add support for parameters derived from other parameters
2. **Hierarchical Parameters**: Implement conditional parameters that depend on others
3. **Complex Constraints**: Support more complex constraint types and expressions
4. **Custom Transformations**: Allow user-defined transformations for parameters
5. **Advanced Sampling**: Integrate more sophisticated sampling strategies
6. **Visualization**: Add parameter space visualization capabilities
7. **Covariance Structures**: Add support for parameter covariance information
8. **Mixed-Type Optimization**: Better handling of mixed continuous/discrete spaces 
