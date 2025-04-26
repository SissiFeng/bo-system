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


class ConstraintHandler:
    """
    高级约束处理工具类，提供约束分析和智能采样功能。
    """
    
    def __init__(self, parameter_space):
        """
        初始化约束处理器
        
        Args:
            parameter_space: 参数空间对象
        """
        self.parameter_space = parameter_space
        self.has_constraints = parameter_space.has_constraints()
        
        # 初始化约束分析缓存
        self.constraint_cache = {
            'valid_regions': None,
            'violation_count': {},
            'feasible_rate': None,
            'analyzed_points': 0
        }
        
        if self.has_constraints:
            logger.info(f"初始化约束处理器，参数空间具有约束条件")
        
    def analyze_constraints(self, sample_points=1000, force=False):
        """
        分析约束条件，评估可行区域特性
        
        Args:
            sample_points: 用于分析的采样点数量
            force: 强制重新分析，即使已有缓存结果
            
        Returns:
            Dict: 约束分析结果
        """
        if not self.has_constraints:
            return {'feasible_rate': 1.0, 'has_constraints': False}
        
        # 如果已有分析结果且不强制重新分析
        if not force and self.constraint_cache['feasible_rate'] is not None:
            return {
                'feasible_rate': self.constraint_cache['feasible_rate'],
                'has_constraints': True,
                'analyzed_points': self.constraint_cache['analyzed_points']
            }
        
        # 生成随机采样点进行分析
        try:
            import numpy as np
            
            # 生成均匀分布的随机点
            valid_count = 0
            violation_counts = {}
            
            # 使用拉丁超立方抽样生成更均匀的点
            try:
                dim = self.parameter_space.get_internal_dimensions()
                # 使用参数空间中的latin_hypercube_sampling函数
                samples = latin_hypercube_sampling(sample_points, dim)
                
                for i in range(sample_points):
                    try:
                        # 将内部表示转换为实际点
                        point = self.parameter_space.internal_to_point(samples[i])
                        
                        # 检查约束条件
                        is_valid, violations = self._check_constraint_details(point)
                        
                        if is_valid:
                            valid_count += 1
                        else:
                            # 记录违反的约束
                            for v in violations:
                                if v not in violation_counts:
                                    violation_counts[v] = 0
                                violation_counts[v] += 1
                    except Exception as e:
                        logger.debug(f"分析约束时生成点出错: {e}")
                        continue
                
            except Exception as e:
                logger.warning(f"使用拉丁超立方抽样分析约束失败: {e}，回退到随机采样")
                
                # 回退到简单随机采样
                for _ in range(sample_points):
                    try:
                        # 随机生成一个点
                        point = {}
                        for param_name, param_config in self.parameter_space.parameters.items():
                            param_type = param_config["type"]
                            
                            if param_type == "continuous":
                                min_val = param_config["min"]
                                max_val = param_config["max"]
                                point[param_name] = np.random.uniform(min_val, max_val)
                            
                            elif param_type == "integer":
                                min_val = param_config["min"]
                                max_val = param_config["max"]
                                point[param_name] = np.random.randint(min_val, max_val + 1)
                            
                            elif param_type == "categorical":
                                categories = param_config["categories"]
                                point[param_name] = np.random.choice(categories)
                        
                        # 检查约束条件
                        is_valid, violations = self._check_constraint_details(point)
                        
                        if is_valid:
                            valid_count += 1
                        else:
                            # 记录违反的约束
                            for v in violations:
                                if v not in violation_counts:
                                    violation_counts[v] = 0
                                violation_counts[v] += 1
                    except Exception as e:
                        logger.debug(f"分析约束时生成随机点出错: {e}")
                        continue
            
            # 计算可行率和记录分析结果
            feasible_rate = valid_count / sample_points if sample_points > 0 else 0
            
            # 更新缓存
            self.constraint_cache.update({
                'feasible_rate': feasible_rate,
                'violation_count': violation_counts,
                'analyzed_points': sample_points
            })
            
            logger.info(f"约束分析结果: 可行率 {feasible_rate:.3f}, 分析了 {sample_points} 个点")
            if violation_counts:
                logger.debug(f"约束违反计数: {violation_counts}")
            
            return {
                'feasible_rate': feasible_rate,
                'violation_counts': violation_counts,
                'has_constraints': True,
                'analyzed_points': sample_points
            }
            
        except Exception as e:
            logger.error(f"分析约束时出错: {e}")
            return {'feasible_rate': None, 'has_constraints': True, 'error': str(e)}
    
    def _check_constraint_details(self, point):
        """
        检查点是否满足约束，并返回违反的约束详情
        
        Args:
            point: 要检查的点
            
        Returns:
            Tuple[bool, List[str]]: (是否满足约束, 违反的约束列表)
        """
        if not self.has_constraints:
            return True, []
        
        violations = []
        is_valid = True
        
        # 检查每个约束条件
        for i, constraint in enumerate(self.parameter_space.constraints):
            constraint_id = f"constraint_{i}"
            if hasattr(constraint, 'description') and constraint['description']:
                constraint_id = constraint['description']
            
            constraint_satisfied = False
            try:
                # 使用参数空间的constraint_type和check_constraints方法
                constraint_satisfied = self.parameter_space.check_constraints(point)
                if not constraint_satisfied:
                    violations.append(constraint_id)
                    is_valid = False
            except Exception as e:
                logger.debug(f"检查约束 {constraint_id} 时出错: {e}")
                violations.append(f"{constraint_id}(error)")
                is_valid = False
        
        return is_valid, violations
    
    def filter_valid_points(self, points):
        """
        过滤出满足所有约束的有效点
        
        Args:
            points: 设计点列表
            
        Returns:
            List: 有效设计点列表
        """
        if not self.has_constraints:
            return points
        
        valid_points = []
        for point in points:
            if self.parameter_space.is_valid_point(point):
                valid_points.append(point)
        
        return valid_points
    
    def adaptive_sampling(self, n_points, generator_func, max_attempts=3, batch_size=None):
        """
        使用自适应采样策略生成满足约束的点
        
        Args:
            n_points: 需要的点数量
            generator_func: 生成候选点的函数，接受n参数
            max_attempts: 最大尝试次数
            batch_size: 每批生成的点数量，默认为n_points的2倍
            
        Returns:
            List: 生成的有效设计点列表
        """
        if not self.has_constraints:
            # 如果没有约束，直接使用生成器函数
            return generator_func(n_points)
        
        # 分析约束
        constraint_analysis = self.analyze_constraints()
        feasible_rate = constraint_analysis.get('feasible_rate', 0.1)
        
        # 确保可行率在有效范围内
        feasible_rate = max(0.01, min(1.0, feasible_rate))
        
        # 根据可行率确定过采样因子
        if feasible_rate > 0.8:
            oversampling_factor = 1.5
        elif feasible_rate > 0.5:
            oversampling_factor = 2
        elif feasible_rate > 0.2:
            oversampling_factor = 3
        elif feasible_rate > 0.05:
            oversampling_factor = 5
        else:
            oversampling_factor = 10
        
        # 确定批处理大小
        if batch_size is None:
            batch_size = min(n_points * 2, 1000)
        
        valid_points = []
        attempts = 0
        total_generated = 0
        
        while len(valid_points) < n_points and attempts < max_attempts:
            # 计算本轮需要生成的点数量
            remaining = n_points - len(valid_points)
            points_to_generate = int(remaining * oversampling_factor)
            
            # 限制单次生成点数，避免内存问题
            points_to_generate = min(points_to_generate, batch_size)
            
            # 生成候选点
            candidate_points = generator_func(points_to_generate)
            total_generated += len(candidate_points)
            
            # 过滤有效点
            for point in candidate_points:
                if self.parameter_space.is_valid_point(point):
                    valid_points.append(point)
                    if len(valid_points) >= n_points:
                        break
            
            # 更新可行率估计并调整过采样因子
            if points_to_generate > 0:
                current_feasible_rate = len(valid_points) / total_generated
                # 平滑更新可行率估计
                feasible_rate = 0.7 * feasible_rate + 0.3 * current_feasible_rate
                
                # 根据新的可行率调整过采样因子
                if feasible_rate > 0.8:
                    oversampling_factor = 1.5
                elif feasible_rate > 0.5:
                    oversampling_factor = 2
                elif feasible_rate > 0.2:
                    oversampling_factor = 3
                elif feasible_rate > 0.05:
                    oversampling_factor = 5
                else:
                    oversampling_factor = 10
            
            attempts += 1
        
        # 如果收集到的有效点不足，记录警告
        if len(valid_points) < n_points:
            logger.warning(f"自适应采样未能生成足够的有效点: 请求 {n_points}，生成 {len(valid_points)}")
        
        return valid_points[:n_points]
    
    def space_filling_sampling(self, n_points, initial_points, generator_func):
        """
        生成空间填充的采样点，尽量分散在可行域中
        
        Args:
            n_points: 需要的点数量
            initial_points: 初始点集合
            generator_func: 生成候选点的函数，接受n参数
            
        Returns:
            List: 生成的设计点列表
        """
        if not self.has_constraints:
            # 如果没有约束，直接使用生成器函数
            return generator_func(n_points)
        
        # 如果初始点集合为空，使用自适应采样生成初始点
        if not initial_points:
            initial_points = self.adaptive_sampling(min(n_points, 10), generator_func)
        
        valid_points = list(initial_points)
        
        # 如果已有足够的点，直接返回
        if len(valid_points) >= n_points:
            return valid_points[:n_points]
        
        try:
            import numpy as np
            from scipy.spatial import distance
            
            # 生成候选点
            oversampling_factor = 5
            candidate_points = self.adaptive_sampling(
                (n_points - len(valid_points)) * oversampling_factor, 
                generator_func
            )
            
            # 如果没有足够的候选点，直接返回所有有效点
            if not candidate_points:
                return valid_points
            
            # 将现有点和候选点转换为数组形式进行距离计算
            existing_array = np.array([list(point.values()) for point in valid_points])
            candidate_array = np.array([list(point.values()) for point in candidate_points])
            
            # 逐个选择最优点
            while len(valid_points) < n_points and candidate_points:
                # 计算每个候选点到现有点集的最小距离
                min_distances = []
                
                for i in range(len(candidate_array)):
                    # 计算候选点到所有现有点的距离
                    if len(existing_array) > 0:
                        dists = distance.cdist([candidate_array[i]], existing_array, 'euclidean')[0]
                        min_dist = np.min(dists)
                    else:
                        # 如果没有现有点，设定一个较大的初始距离
                        min_dist = float('inf')
                    
                    min_distances.append(min_dist)
                
                # 选择最大化最小距离的点
                if min_distances:
                    best_idx = np.argmax(min_distances)
                    
                    # 添加最佳点到结果集
                    valid_points.append(candidate_points[best_idx])
                    
                    # 更新现有点数组
                    existing_array = np.vstack([existing_array, [candidate_array[best_idx]]])
                    
                    # 从候选点中删除已选点
                    candidate_points.pop(best_idx)
                    candidate_array = np.delete(candidate_array, best_idx, axis=0)
                else:
                    break
                
            return valid_points[:n_points]
            
        except Exception as e:
            logger.warning(f"空间填充采样失败: {e}，回退到自适应采样")
            # 回退到自适应采样
            additional_points = self.adaptive_sampling(
                n_points - len(valid_points), 
                generator_func
            )
            valid_points.extend(additional_points)
            return valid_points[:n_points]

# 添加拉丁超立方抽样质量评估函数
def evaluate_lhs_quality(points):
    """
    评估拉丁超立方抽样的质量
    
    Args:
        points: 采样点数组，形状为(n_samples, n_dimensions)
        
    Returns:
        Dict: 质量评估指标
    """
    try:
        import numpy as np
        from scipy.spatial import distance
        
        n_samples, n_dimensions = points.shape
        
        # 计算点之间的最小距离
        dist_matrix = distance.pdist(points)
        min_dist = np.min(dist_matrix) if len(dist_matrix) > 0 else 0
        mean_dist = np.mean(dist_matrix) if len(dist_matrix) > 0 else 0
        
        # 计算空间填充度指标
        if len(dist_matrix) > 0:
            # 标准差/均值比 (smaller is better for uniformity)
            cv = np.std(dist_matrix) / mean_dist if mean_dist > 0 else float('inf')
            
            # 最小距离/均值比 (larger is better for space-filling)
            min_mean_ratio = min_dist / mean_dist if mean_dist > 0 else 0
        else:
            cv = float('inf')
            min_mean_ratio = 0
        
        return {
            'min_distance': min_dist,
            'mean_distance': mean_dist,
            'cv': cv,
            'min_mean_ratio': min_mean_ratio,
            'n_samples': n_samples,
            'n_dimensions': n_dimensions
        }
    except Exception as e:
        logger.debug(f"评估LHS质量时出错: {e}")
        return {
            'error': str(e),
            'n_samples': len(points) if isinstance(points, (list, np.ndarray)) else 0
        } 
