import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any, Tuple, Optional, Callable
import json
import os
import logging
from pathlib import Path
import re
import random
import uuid

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


def latin_hypercube_sampling(n_samples: int, dimensions: int, seed: Optional[int] = None) -> np.ndarray:
    """
    执行拉丁超立方抽样
    
    Args:
        n_samples: 样本数量
        dimensions: 维度数量
        seed: 随机种子
        
    Returns:
        np.ndarray: 形状为 (n_samples, dimensions) 的样本数组，每个维度的值在 [0, 1] 范围内
    """
    # 设置随机种子
    rng = np.random.RandomState(seed)
    
    # 为每个维度生成随机排列
    result = np.zeros((n_samples, dimensions))
    
    # 对每个维度执行拉丁超立方抽样
    for i in range(dimensions):
        # 将 [0, 1] 区间划分为 n_samples 个等间距部分
        perms = rng.permutation(n_samples)
        # 在每个部分中随机采样一个点
        result[:, i] = (perms + rng.random(n_samples)) / n_samples
    
    return result


def ensure_directory_exists(directory_path: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory_path: 目录路径
    """
    os.makedirs(directory_path, exist_ok=True)


def save_to_json(data: Dict[str, Any], filepath: str) -> None:
    """
    将数据保存为JSON文件
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    # 确保目录存在
    ensure_directory_exists(os.path.dirname(filepath))
    
    # 保存数据
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_from_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    从JSON文件加载数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        Optional[Dict[str, Any]]: 加载的数据，如果文件不存在则返回None
    """
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载JSON文件失败: {filepath}, 错误: {str(e)}")
        return None


def generate_unique_id(length: int = 8) -> str:
    """
    生成唯一ID
    
    Args:
        length: ID长度
        
    Returns:
        str: 唯一ID
    """
    # 使用UUID生成唯一ID
    return str(uuid.uuid4())[:length]


def normalize_value(value: float, min_value: float, max_value: float) -> float:
    """
    将值标准化到 [0, 1] 范围
    
    Args:
        value: 要标准化的值
        min_value: 最小值
        max_value: 最大值
        
    Returns:
        float: 标准化后的值
    """
    if min_value == max_value:
        return 0.5  # 如果最小值等于最大值，返回0.5
    
    return (value - min_value) / (max_value - min_value)


def denormalize_value(normalized_value: float, min_value: float, max_value: float) -> float:
    """
    将 [0, 1] 范围内的值反标准化
    
    Args:
        normalized_value: 标准化的值
        min_value: 最小值
        max_value: 最大值
        
    Returns:
        float: 反标准化后的值
    """
    return min_value + normalized_value * (max_value - min_value)


def compute_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算两个点之间的欧几里得距离
    
    Args:
        x: 第一个点
        y: 第二个点
        
    Returns:
        float: 欧几里得距离
    """
    return np.sqrt(np.sum((x - y) ** 2))


def compute_distance_matrix(points: np.ndarray) -> np.ndarray:
    """
    计算点集的距离矩阵
    
    Args:
        points: 形状为 (n_points, dimensions) 的点集
        
    Returns:
        np.ndarray: 形状为 (n_points, n_points) 的距离矩阵
    """
    n_points = points.shape[0]
    distance_matrix = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            distance = compute_distance(points[i], points[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    return distance_matrix


def find_nearest_neighbors(points: np.ndarray, query_point: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    找到查询点的最近邻
    
    Args:
        points: 形状为 (n_points, dimensions) 的点集
        query_point: 形状为 (dimensions,) 的查询点
        k: 最近邻数量
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (最近邻索引, 最近邻距离)
    """
    # 计算查询点与所有点的距离
    distances = np.array([compute_distance(query_point, point) for point in points])
    
    # 找到最近的k个点
    indices = np.argsort(distances)[:k]
    
    return indices, distances[indices]


def compute_min_distance_to_existing_points(new_point: np.ndarray, existing_points: np.ndarray) -> float:
    """
    计算新点到现有点集的最小距离
    
    Args:
        new_point: 形状为 (dimensions,) 的新点
        existing_points: 形状为 (n_points, dimensions) 的现有点集
        
    Returns:
        float: 最小距离，如果现有点集为空，则返回无穷大
    """
    if len(existing_points) == 0:
        return float('inf')
    
    distances = np.array([compute_distance(new_point, point) for point in existing_points])
    return np.min(distances)


def maximize_min_distance(candidate_points: np.ndarray, n_select: int, existing_points: Optional[np.ndarray] = None) -> np.ndarray:
    """
    从候选点中选择最大化最小距离的点
    
    Args:
        candidate_points: 形状为 (n_candidates, dimensions) 的候选点集
        n_select: 要选择的点数量
        existing_points: 形状为 (n_existing, dimensions) 的现有点集
        
    Returns:
        np.ndarray: 形状为 (n_select, dimensions) 的选择点集
    """
    if existing_points is None or len(existing_points) == 0:
        existing_points = np.zeros((0, candidate_points.shape[1]))
    
    selected_indices = []
    
    for _ in range(n_select):
        # 计算每个候选点到现有点集的最小距离
        min_distances = []
        for i in range(len(candidate_points)):
            if i in selected_indices:
                min_distances.append(-float('inf'))  # 已选择的点不再考虑
            else:
                min_dist = compute_min_distance_to_existing_points(
                    candidate_points[i], 
                    np.vstack([existing_points, candidate_points[selected_indices]])
                )
                min_distances.append(min_dist)
        
        # 选择最大化最小距离的点
        idx = np.argmax(min_distances)
        selected_indices.append(idx)
    
    return candidate_points[selected_indices]


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
