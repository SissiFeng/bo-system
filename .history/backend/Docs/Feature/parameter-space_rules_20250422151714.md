# Parameter Space Feature Design

This document outlines the design principles and implementation details for the parameter space feature in the BO Engine.

## Overview

The parameter space is a core component of Bayesian optimization that defines the variables, their ranges, and constraints. Our implementation aims to support a wide range of parameter types and constraints, making it flexible for various optimization scenarios.

## Key Components

### Parameter Types

The system supports three main parameter types:

1. **Continuous Parameters**:
   - Real-valued parameters within a specified range
   - Example: `{"name": "temperature", "type": "continuous", "min": 20.0, "max": 100.0}`

2. **Discrete Parameters**:
   - Integer-valued parameters within a specified range, with an optional step size
   - Example: `{"name": "batch_size", "type": "discrete", "min": 1, "max": 128, "step": 1}`

3. **Categorical Parameters**:
   - Parameters that take values from a predefined set of options
   - Example: `{"name": "catalyst", "type": "categorical", "values": ["A", "B", "C"]}`

### Objectives

The system supports multiple objective functions, each with a type (maximize or minimize):

- Example: `{"name": "yield", "type": "maximize"}`
- Example: `{"name": "cost", "type": "minimize"}`

### Constraints

The system supports various constraint types:

1. **Sum Equals**: The sum of specified parameters equals a target value
   - Example: `{"expression": "x1 + x2 + x3", "type": "sum_equals", "value": 1.0}`

2. **Sum Less Than**: The sum of specified parameters is less than a target value
   - Example: `{"expression": "x1 + x2", "type": "sum_less_than", "value": 0.5}`

3. **Sum Greater Than**: The sum of specified parameters is greater than a target value
   - Example: `{"expression": "x1 + x2", "type": "sum_greater_than", "value": 0.1}`

4. **Product Equals**: The product of specified parameters equals a target value
   - Example: `{"expression": "x1 * x2", "type": "product_equals", "value": 1.0}`

5. **Custom**: A custom constraint expression
   - Example: `{"expression": "2*x1 + 3*x2 - x3", "type": "custom", "value": 0.0}`

## Implementation Details

### Class Structure

The parameter space implementation will follow a hierarchical structure:

```
ParameterSpace
├── Parameter (abstract base class)
│   ├── ContinuousParameter
│   ├── DiscreteParameter
│   └── CategoricalParameter
├── Objective
└── Constraint
```

### Key Functionality

1. **Parameter Space Validation**:
   - Validate that all parameter names are unique
   - Validate that parameter ranges are valid (min < max)
   - Validate that categorical values are not empty
   - Validate that constraint expressions are valid and refer to existing parameters

2. **Parameter Space Transformation**:
   - Convert categorical parameters to one-hot encoded representations
   - Scale parameters to a standard range (e.g., [0, 1]) for optimization algorithms
   - Convert constraints to a standardized form

3. **Sampling and Evaluation**:
   - Generate random points within the parameter space
   - Check if a point satisfies all constraints
   - Transform between internal representation and user-facing representation

### Integration with Design Generator

The parameter space will be used by the design generator to:

1. Generate initial design points (e.g., using Latin Hypercube Sampling)
2. Ensure generated points satisfy all constraints
3. Transform points between the optimization space and the user-facing space

### Integration with Optimizer

The parameter space will be used by the optimizer to:

1. Validate that experimental results match the defined parameters
2. Transform points for surrogate model training
3. Constrain the acquisition function optimization to valid regions of the space

## API Design

### Creating a Parameter Space

```python
space = ParameterSpace(
    parameters=[
        ContinuousParameter(name="x1", min=0.0, max=1.0),
        DiscreteParameter(name="x2", min=1, max=10, step=1),
        CategoricalParameter(name="x3", values=["A", "B", "C"]),
    ],
    objectives=[
        Objective(name="y1", type=ObjectiveType.MAXIMIZE),
    ],
    constraints=[
        Constraint(expression="x1 + x2/10", type=ConstraintType.SUM_EQUALS, value=1.0),
    ],
)
```

### Checking Point Validity

```python
point = {"x1": 0.5, "x2": 5, "x3": "A"}
is_valid = space.is_valid(point)
```

### Transforming Points

```python
# Transform from user space to internal space
internal_point = space.transform_to_internal(point)

# Transform from internal space to user space
user_point = space.transform_to_user(internal_point)
```

### Sampling Random Points

```python
# Sample a single valid point
random_point = space.sample_random()

# Sample multiple valid points
random_points = space.sample_random_batch(n=10)
```

## Future Enhancements

1. **Conditional Parameters**: Parameters that only exist or have a specific range when certain conditions are met
2. **Complex Constraints**: Support for more complex constraint types (e.g., nonlinear constraints)
3. **Parameter Space Visualization**: Tools to visualize the parameter space and constraints
4. **Constraint Relaxation**: Methods to handle infeasible regions and relax constraints when necessary

## Testing Strategy

1. **Unit Tests**: 
   - Test creation of each parameter type
   - Test validation of parameter ranges
   - Test constraint validation
   - Test transformation between user and internal spaces

2. **Integration Tests**:
   - Test integration with design generator
   - Test integration with optimizer
   - Test end-to-end workflow with parameter space, design generation, and optimization

3. **Performance Tests**:
   - Test performance with large parameter spaces
   - Test performance with many constraints 
