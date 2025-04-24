import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Tuple, Optional
import json
import os
import logging
from pathlib import Path
import re

# Setup logger
logger = logging.getLogger("bo_engine.utils")

def validate_expression(expression: str, parameter_names: List[str]) -> bool:
    """
    Validate that an expression only contains valid parameter names and operations.
    
    Args:
        expression: The constraint expression to validate
        parameter_names: List of valid parameter names
        
    Returns:
        bool: True if the expression is valid, False otherwise
    """
    # Replace all parameter names with 'x' to simplify validation
    sanitized_expr = expression
    for name in parameter_names:
        sanitized_expr = sanitized_expr.replace(name, "x")
    
    # Check if sanitized expression contains only valid characters
    # Valid: x (parameters), numbers, basic operators, whitespace
    valid_chars_pattern = r'^[x0-9\+\-\*\/\(\)\.\s]+$'
    return bool(re.match(valid_chars_pattern, sanitized_expr))


def evaluate_expression(expression: str, parameters: Dict[str, Any]) -> float:
    """
    Safely evaluate a mathematical expression with parameter values.
    
    Args:
        expression: The expression to evaluate
        parameters: Dictionary of parameter names and values
        
    Returns:
        float: The result of evaluating the expression
    
    Raises:
        ValueError: If the expression is invalid or contains unsafe operations
    """
    # Validate parameter names
    if not validate_expression(expression, list(parameters.keys())):
        raise ValueError(f"Invalid expression: {expression}")
    
    # Replace parameter names with their values
    expr = expression
    for name, value in parameters.items():
        expr = expr.replace(name, str(float(value)))
    
    # Evaluate expression safely
    try:
        return float(eval(expr))
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expression}': {str(e)}")


def one_hot_encode(value: Any, categories: List[Any]) -> List[int]:
    """
    Convert a categorical value to one-hot encoding.
    
    Args:
        value: The categorical value to encode
        categories: List of all possible categories
        
    Returns:
        List[int]: One-hot encoded representation
        
    Raises:
        ValueError: If the value is not in the categories list
    """
    if value not in categories:
        raise ValueError(f"Value '{value}' not in categories: {categories}")
    
    encoding = [0] * len(categories)
    encoding[categories.index(value)] = 1
    return encoding


def one_hot_decode(encoding: List[int], categories: List[Any]) -> Any:
    """
    Convert a one-hot encoding back to its categorical value.
    
    Args:
        encoding: One-hot encoded representation
        categories: List of all possible categories
        
    Returns:
        Any: The decoded categorical value
        
    Raises:
        ValueError: If the encoding is invalid
    """
    if len(encoding) != len(categories) or sum(encoding) != 1:
        raise ValueError(f"Invalid one-hot encoding: {encoding}")
    
    index = encoding.index(1)
    return categories[index]


def scale_parameters(parameters: Dict[str, float], ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """
    Scale parameters to [0, 1] range based on their min/max values.
    
    Args:
        parameters: Dictionary of parameter names and values
        ranges: Dictionary of parameter names and (min, max) tuples
        
    Returns:
        Dict[str, float]: Dictionary of scaled parameter values
    """
    scaled = {}
    for name, value in parameters.items():
        if name in ranges:
            min_val, max_val = ranges[name]
            if max_val > min_val:
                scaled[name] = (value - min_val) / (max_val - min_val)
            else:
                scaled[name] = 0.0  # Handle case where min == max
        else:
            # If no range is provided, keep the original value
            scaled[name] = value
    return scaled


def unscale_parameters(scaled_parameters: Dict[str, float], ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """
    Convert scaled [0, 1] parameters back to their original range.
    
    Args:
        scaled_parameters: Dictionary of scaled parameter values
        ranges: Dictionary of parameter names and (min, max) tuples
        
    Returns:
        Dict[str, float]: Dictionary of unscaled parameter values
    """
    unscaled = {}
    for name, scaled_value in scaled_parameters.items():
        if name in ranges:
            min_val, max_val = ranges[name]
            unscaled[name] = min_val + scaled_value * (max_val - min_val)
        else:
            # If no range is provided, keep the scaled value
            unscaled[name] = scaled_value
    return unscaled


def latin_hypercube_sampling(n_samples: int, n_dimensions: int) -> np.ndarray:
    """
    Generate Latin Hypercube samples in [0, 1] range.
    
    Args:
        n_samples: Number of samples to generate
        n_dimensions: Number of dimensions (parameters)
        
    Returns:
        np.ndarray: Array of shape (n_samples, n_dimensions) with samples
    """
    # Initialize result array
    result = np.zeros((n_samples, n_dimensions))
    
    # Generate samples for each dimension
    for i in range(n_dimensions):
        # Generate random permutation of intervals
        perm = np.random.permutation(n_samples)
        
        # Generate uniform samples within each interval
        u = np.random.uniform(0, 1, n_samples)
        
        # Compute samples
        result[:, i] = (perm + u) / n_samples
    
    return result


def save_json(obj: Any, filepath: Union[str, Path], **kwargs) -> None:
    """
    Save an object to a JSON file.
    
    Args:
        obj: Object to save
        filepath: Path to the output file
        **kwargs: Additional arguments for json.dump
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(obj, f, **kwargs)


def load_json(filepath: Union[str, Path], default: Optional[Any] = None) -> Any:
    """
    Load an object from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        default: Default value to return if file doesn't exist
        
    Returns:
        Any: The loaded object, or default if file doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return default
    
    with open(filepath, 'r') as f:
        return json.load(f)


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """
    Convert a list of result dictionaries to a pandas DataFrame.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        pd.DataFrame: DataFrame containing the results
    """
    # Extract parameter and objective values
    rows = []
    for result in results:
        row = {'design_id': result.get('design_id', '')}
        
        # Add parameters
        if 'parameters' in result:
            for param_name, param_value in result['parameters'].items():
                row[param_name] = param_value
        
        # Add objectives
        if 'objectives' in result:
            for obj_name, obj_value in result['objectives'].items():
                row[obj_name] = obj_value
        
        # Add metadata
        if 'metadata' in result:
            for meta_name, meta_value in result['metadata'].items():
                row[f'metadata_{meta_name}'] = meta_value
        
        rows.append(row)
    
    return pd.DataFrame(rows) 
