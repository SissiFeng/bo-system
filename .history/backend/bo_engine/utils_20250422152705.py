import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Tuple, Optional, Callable
import json
import os
import logging
from pathlib import Path
import re
import random

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


def latin_hypercube_sampling(n: int, dim: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate Latin Hypercube samples in [0, 1] space.
    
    Args:
        n: Number of samples to generate
        dim: Dimensionality of the samples
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Array of shape (n, dim) containing the samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate basic LHS samples
    result = np.zeros((n, dim))
    
    for d in range(dim):
        # Generate n evenly spaced points in [0, 1]
        spacing = np.linspace(0, 1, n + 1)
        
        # Get midpoints of the intervals
        points = (spacing[:-1] + spacing[1:]) / 2
        
        # Randomly permute the points for this dimension
        np.random.shuffle(points)
        
        result[:, d] = points
    
    return result


def ensure_directory_exists(directory_path: Union[str, Path]) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    if isinstance(directory_path, str):
        directory_path = Path(directory_path)
    
    if not directory_path.exists():
        try:
            directory_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            raise


def save_to_json(data: Any, filepath: Union[str, Path], ensure_dir: bool = True) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filepath: Path to the JSON file
        ensure_dir: If True, ensure that the directory exists
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    
    if ensure_dir:
        ensure_directory_exists(filepath.parent)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved data to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save data to {filepath}: {e}")
        raise


def load_from_json(filepath: Union[str, Path], default: Any = None) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        default: Default value to return if the file doesn't exist
        
    Returns:
        Any: The loaded data, or the default value if the file doesn't exist
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    
    if not filepath.exists():
        if default is not None:
            return default
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {e}")
        raise


def get_random_seed() -> int:
    """
    Get a random seed that can be used for reproducibility.
    
    Returns:
        int: A random seed
    """
    return random.randint(1, 1000000)


def normalize_array(array: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    """
    Normalize array elements to [0, 1] range.
    
    Args:
        array: Array to normalize
        min_vals: Minimum values for each dimension
        max_vals: Maximum values for each dimension
        
    Returns:
        np.ndarray: Normalized array
    """
    range_vals = max_vals - min_vals
    # Handle potential division by zero (for dimensions with min=max)
    range_vals = np.where(range_vals > 0, range_vals, 1.0)
    
    return (array - min_vals) / range_vals


def denormalize_array(normalized_array: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    """
    Denormalize array elements from [0, 1] range to original range.
    
    Args:
        normalized_array: Normalized array (values in [0, 1])
        min_vals: Minimum values for each dimension
        max_vals: Maximum values for each dimension
        
    Returns:
        np.ndarray: Denormalized array
    """
    return min_vals + normalized_array * (max_vals - min_vals)


def generate_unique_id(prefix: str = "") -> str:
    """
    Generate a unique ID with an optional prefix.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        str: Unique ID
    """
    import uuid
    import time
    
    # Get current timestamp
    timestamp = int(time.time())
    
    # Generate random UUID
    random_uuid = str(uuid.uuid4()).replace('-', '')[:8]
    
    # Combine timestamp and UUID
    unique_id = f"{timestamp}_{random_uuid}"
    
    # Add prefix if provided
    if prefix:
        unique_id = f"{prefix}_{unique_id}"
    
    return unique_id


def safe_eval(expr: str, variables: Dict[str, Any]) -> Any:
    """
    Safely evaluate an expression with given variables.
    
    Args:
        expr: Expression to evaluate
        variables: Dictionary of variable names and values
        
    Returns:
        Any: Result of the evaluation
    """
    # Define allowed functions and operators
    allowed_names = {
        'abs': abs,
        'min': min,
        'max': max,
        'sum': sum,
        'len': len,
        'round': round,
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'pow': pow,
        'True': True,
        'False': False,
        'None': None,
    }
    
    # Update with provided variables
    allowed_names.update(variables)
    
    # Compile the expression
    try:
        code = compile(expr, '<string>', 'eval')
        
        # Check for disallowed names
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of disallowed name: {name}")
        
        # Evaluate the expression
        return eval(code, {"__builtins__": {}}, allowed_names)
    except Exception as e:
        logger.error(f"Error evaluating expression '{expr}': {e}")
        raise ValueError(f"Error evaluating expression '{expr}': {e}")


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
