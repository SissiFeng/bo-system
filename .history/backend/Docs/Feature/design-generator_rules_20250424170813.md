# Design Generator Design and Implementation Rules

## Module Overview

The Design Generator module is responsible for generating initial design points and candidate points within the parameter space, serving as an important component of the Bayesian optimization system. This module provides various experimental design methods, including random design, Latin hypercube design, factorial design, and Sobol sequence design, to meet the needs of different optimization scenarios. The Design Generator ensures that the generated design points comply with parameter constraints and dependency relationships through interaction with the Parameter Space module.

## Core Design Principles

1. **Algorithm Diversity**:
   - Support multiple experimental design algorithms to adapt to different optimization requirements
   - Provide optimal sampling strategies for different parameter types and dimensions

2. **Parameter Space Compatibility**:
   - Tightly integrate with the Parameter Space module to ensure design point validity
   - Support mixed designs for all parameter types (continuous, integer, categorical)

3. **Extensibility**:
   - Implement design generator creation based on the factory pattern
   - Facilitate the addition of new design generation algorithms

4. **Randomness Control**:
   - Provide random seed settings to ensure experiment reproducibility
   - Support deterministic generation modes

## Class Hierarchy and Structure

### Enumeration Types

```python
class DesignType(Enum):
    """Design type enumeration"""
    RANDOM = "random"
    LATIN_HYPERCUBE = "latin_hypercube"
    FACTORIAL = "factorial"
    SOBOL = "sobol"
    CUSTOM = "custom"
```

### Base Class Definition

```python
class DesignGenerator(ABC):
    """Abstract base class for design generators, defining common interfaces for all design generators"""
    
    def __init__(self, parameter_space):
        self.parameter_space = parameter_space
        
    @abstractmethod
    def generate(self, num_points, random_state=None):
        """Generate a set of design points"""
        pass
```

### Concrete Design Generators

```python
class RandomDesignGenerator(DesignGenerator):
    """Random design generator"""
    
    def __init__(self, parameter_space):
        super().__init__(parameter_space)
        
    def generate(self, num_points, random_state=None):
        """Generate random design points"""
        # Use parameter space sampling methods to generate random points
        # Check the validity of points
        # Return the list of generated design points
```

```python
class LatinHypercubeDesignGenerator(DesignGenerator):
    """Latin hypercube design generator"""
    
    def __init__(self, parameter_space, criterion="maximin"):
        super().__init__(parameter_space)
        self.criterion = criterion
        
    def generate(self, num_points, random_state=None):
        """Generate Latin hypercube design points"""
        # Create Latin hypercube sampling based on parameter space dimensions
        # Convert sampling results to actual values in the parameter space
        # Handle categorical parameters
        # Return the list of generated design points
```

```python
class FactorialDesignGenerator(DesignGenerator):
    """Factorial design generator"""
    
    def __init__(self, parameter_space, levels=None):
        super().__init__(parameter_space)
        self.levels = levels or {}
        
    def generate(self, num_points=None, random_state=None):
        """Generate factorial design points"""
        # Determine the number of levels for each parameter
        # Generate complete factorial combinations
        # Filter results to ensure they meet constraint conditions
        # Return the list of generated design points
```

```python
class SobolDesignGenerator(DesignGenerator):
    """Sobol sequence design generator"""
    
    def __init__(self, parameter_space, scramble=True):
        super().__init__(parameter_space)
        self.scramble = scramble
        
    def generate(self, num_points, random_state=None):
        """Generate Sobol sequence design points"""
        # Create a Sobol sequence generator
        # Generate a specified number of low-discrepancy sequences
        # Convert sequences to actual values in the parameter space
        # Return the list of generated design points
```

```python
class CustomDesignGenerator(DesignGenerator):
    """Custom design generator"""
    
    def __init__(self, parameter_space, designs=None):
        super().__init__(parameter_space)
        self.designs = designs or []
        
    def generate(self, num_points=None, random_state=None):
        """Return custom design points"""
        # Validate the validity of custom design points
        # Convert to internal representation
        # Return the list of custom design points
        
    def add_design(self, design):
        """Add a custom design point"""
        # Validate the design point
        # Add to the design collection
```

### Factory Function

```python
def create_design_generator(design_type, parameter_space, **kwargs):
    """Design generator factory function"""
    if design_type == DesignType.RANDOM:
        return RandomDesignGenerator(parameter_space)
    elif design_type == DesignType.LATIN_HYPERCUBE:
        return LatinHypercubeDesignGenerator(parameter_space, **kwargs)
    elif design_type == DesignType.FACTORIAL:
        return FactorialDesignGenerator(parameter_space, **kwargs)
    elif design_type == DesignType.SOBOL:
        return SobolDesignGenerator(parameter_space, **kwargs)
    elif design_type == DesignType.CUSTOM:
        return CustomDesignGenerator(parameter_space, **kwargs)
    else:
        raise ValueError(f"Unknown design type: {design_type}")
```

## Key Method Implementations

### Random Design Generation

```python
def generate_random_design(self, num_points, random_state=None):
    """Generate random design points"""
    # Set random seed
    rng = np.random.RandomState(random_state)
    
    # Sample for each parameter
    designs = []
    for _ in range(num_points):
        design = {}
        for name, param in self.parameter_space.parameters.items():
            design[name] = param.sample(1, rng)[0]
        
        # Validate design point validity
        if self.parameter_space.validate_design(design):
            designs.append(design)
    
    return designs
```

### Latin Hypercube Design Generation

```python
def generate_lhs_design(self, num_points, random_state=None):
    """Generate Latin hypercube design points"""
    # Set random seed
    rng = np.random.RandomState(random_state)
    
    # Get the number of continuous and integer parameters
    continuous_params = [p for p in self.parameter_space.parameters.values() 
                         if p.parameter_type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]]
    n_continuous = len(continuous_params)
    
    if n_continuous > 0:
        # Create Latin hypercube sampling
        lhs_samples = lhs(n_continuous, samples=num_points, criterion=self.criterion, random_state=rng)
        
        # Convert to actual parameter values
        designs = []
        for i in range(num_points):
            design = {}
            continuous_idx = 0
            
            for name, param in self.parameter_space.parameters.items():
                if param.parameter_type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]:
                    # Use LHS sampling for continuous and integer parameters
                    value = param.from_unit_interval(lhs_samples[i, continuous_idx])
                    design[name] = value
                    continuous_idx += 1
                else:
                    # Use random sampling for categorical parameters
                    design[name] = param.sample(1, rng)[0]
            
            # Validate design point validity
            if self.parameter_space.validate_design(design):
                designs.append(design)
        
        return designs
    else:
        # If there are no continuous or integer parameters, fall back to random sampling
        return self.generate_random_design(num_points, random_state)
```

### Grid Design Generation

```python
def generate_grid_design(self, levels=None):
    """Generate grid design points"""
    if levels is None:
        levels = self.levels
    
    # Determine the number of levels for each parameter
    param_levels = {}
    for name, param in self.parameter_space.parameters.items():
        if name in levels:
            param_levels[name] = levels[name]
        elif param.parameter_type == ParameterType.CATEGORICAL:
            param_levels[name] = len(param.categories)
        elif param.parameter_type == ParameterType.INTEGER:
            range_size = param.upper_bound - param.lower_bound + 1
            param_levels[name] = min(range_size, 5)  # Default maximum of 5 levels
        else:
            param_levels[name] = 5  # Default 5 levels for continuous parameters
    
    # Generate grid points
    # Use itertools.product to generate combinations
    
    # Filter invalid points
    # Verify that each point satisfies constraint conditions
    
    return valid_designs
```

### Custom Design Validation and Addition

```python
def add_custom_design(self, design_dict):
    """Add a custom design point"""
    # Validate design point format
    if not isinstance(design_dict, dict):
        raise ValueError("Design must be a dictionary")
        
    # Validate design point parameters
    if not self.parameter_space.validate_design(design_dict):
        raise ValueError("Design violates parameter space constraints")
        
    # Add to custom design list
    self.designs.append(design_dict)
    
    return True
```

## Data Flow Description

1. **Design Generator Initialization**:
   - Receives parameter space object as input
   - Configures the design generator based on design type and additional parameters

2. **Design Point Generation**:
   - Generates design points based on specified quantity and random seed
   - Calls specific algorithms to generate candidate points
   - Validates the validity of generated points
   - Returns the set of design points that meet requirements

3. **Interaction with Parameter Space**:
   - Uses parameter space sampling methods to generate basic random values
   - Calls parameter space validation methods to check design point constraint compliance
   - Uses parameter space transformation methods to convert between internal and external representations

4. **Design Storage and Loading**:
   - Supports saving generated design points to files
   - Able to load predefined design points from files

## Code Validation Rules

1. **Design Point Validity Validation**:
   - All design points must include all parameters defined in the parameter space
   - Parameter values in design points must comply with parameter space constraints
   - The design point collection should not have duplicate points (within a specified tolerance range)

2. **Algorithm Implementation Correctness**:
   - Random design should ensure uniform coverage of the entire parameter space
   - Latin hypercube design should ensure uniform distribution in each dimension
   - Factorial design should generate all possible factor combinations
   - Sobol sequence should possess good low-discrepancy properties

3. **Performance Requirements**:
   - The time complexity of design generation should be proportional to the number of design points and parameter space dimensions
   - Efficient sampling algorithms should be used for high-dimensional parameter spaces
   - Avoid generating a large number of invalid points that are rejected by parameter constraints

## Expansion Plans

1. **Adaptive Design**:
   - Implement design generators that dynamically adjust based on existing observation results
   - Support adaptive sampling strategies with exploration-exploitation balance

2. **Batch Design Optimization**:
   - Optimize batch design generation for parallel evaluation
   - Implement batch design methods that maximize distance between point sets

3. **Constraint-Aware Design**:
   - Develop design generators that effectively handle complex constraints
   - Implement sampling strategies based on constraint boundaries

4. **Multi-Fidelity Design**:
   - Support experimental design at different fidelity levels
   - Implement correlated design generation across fidelity levels

5. **Domain-Specific Design**:
   - Add design generators optimized for specific application domains
   - Support design generation based on expert knowledge

## 🔄 Interaction with Other Modules

- **Input**: Parameter space definition from the `ParameterSpace` module
- **Output**: Design points for use by `BOSystem` and evaluation modules
- **Dependencies**: Depends on helper functions in `utils.py` and parameter definitions in `parameter_space.py`

## 🔮 Future Extensions

1. Add more experimental design methods (Halton sequence, orthogonal arrays, etc.)
2. Support design generation in constrained spaces
3. Increase adaptive design generation strategies
4. Add support for distributed parallel experiments 
