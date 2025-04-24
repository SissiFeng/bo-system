import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
from enum import Enum
import random
from abc import ABC, abstractmethod
import uuid
import os
from pathlib import Path
import json

from .parameter_space import ParameterSpace, Parameter
from .utils import (
    latin_hypercube_sampling,
    ensure_directory_exists,
    save_to_json,
    load_from_json,
    generate_unique_id,
    normalize_value,
    denormalize_value,
    maximize_min_distance
)

# Setup logger
logger = logging.getLogger("bo_engine.design_generator")


class DesignType(str, Enum):
    """Enum for experimental design types."""
    RANDOM = "random"
    LATIN_HYPERCUBE = "latin_hypercube"
    FACTORIAL = "factorial"
    SOBOL = "sobol"
    CUSTOM = "custom"


class DesignGenerator(ABC):
    """
    Abstract base class for all design generators.
    """
    def __init__(self, parameter_space: ParameterSpace):
        """
        Initialize a design generator.
        
        Args:
            parameter_space: Parameter space for generating designs
        """
        self.parameter_space = parameter_space
        
        # Validate parameter space
        valid, error_msg = parameter_space.validate()
        if not valid:
            raise ValueError(f"Invalid parameter space: {error_msg}")
    
    @abstractmethod
    def generate(self, n: int) -> List[Dict[str, Any]]:
        """
        Generate n experimental design points.
        
        Args:
            n: Number of design points to generate
            
        Returns:
            List[Dict[str, Any]]: List of design points as dictionaries
        """
        pass


class RandomDesignGenerator(DesignGenerator):
    """
    Generate random design points by sampling from parameter distributions.
    """
    def generate(self, n: int) -> List[Dict[str, Any]]:
        """
        Generate n random design points.
        
        Args:
            n: Number of design points to generate
            
        Returns:
            List[Dict[str, Any]]: List of design points
        """
        designs = self.parameter_space.sample_random_batch(n)
        
        # If we couldn't generate enough designs, warn
        if len(designs) < n:
            logger.warning(f"Could only generate {len(designs)} out of {n} requested design points")
        
        return designs


class LatinHypercubeDesignGenerator(DesignGenerator):
    """
    Generate design points using Latin Hypercube Sampling (LHS).
    """
    def __init__(self, parameter_space: ParameterSpace, seed: Optional[int] = None):
        """
        Initialize a Latin Hypercube design generator.
        
        Args:
            parameter_space: Parameter space for generating designs
            seed: Random seed for reproducibility
        """
        super().__init__(parameter_space)
        self.seed = seed
        
        # Check if parameter space has any constraints
        self.has_constraints = len(parameter_space.constraints) > 0
        
        if self.has_constraints:
            logger.warning("Parameter space has constraints. LHS designs may not satisfy all constraints.")
    
    def generate(self, n: int) -> List[Dict[str, Any]]:
        """
        Generate n design points using Latin Hypercube Sampling.
        
        Args:
            n: Number of design points to generate
            
        Returns:
            List[Dict[str, Any]]: List of design points
        """
        # Get internal dimensionality
        dim = self.parameter_space.get_internal_dimensions()
        
        # Generate LHS samples in [0, 1] space
        samples = latin_hypercube_sampling(n, dim, seed=self.seed)
        
        # Convert samples to parameter space
        designs = []
        for i in range(n):
            try:
                # Convert internal representation to point
                point = self.parameter_space.internal_to_point(samples[i])
                
                # Check if point satisfies constraints
                if not self.has_constraints or self.parameter_space.is_valid_point(point):
                    designs.append(point)
            except Exception as e:
                logger.warning(f"Error generating design point: {e}")
        
        # If we have constraints and couldn't generate enough designs, fill with random samples
        if len(designs) < n:
            logger.warning(f"LHS generated only {len(designs)} valid points out of {n}. Filling remainder with random samples.")
            
            # Generate additional random points
            random_generator = RandomDesignGenerator(self.parameter_space)
            additional_points = random_generator.generate(n - len(designs))
            designs.extend(additional_points)
        
        return designs


class FactorialDesignGenerator(DesignGenerator):
    """
    Generate design points using factorial design.
    Only practical for small parameter spaces with discrete/categorical parameters.
    """
    def __init__(self, parameter_space: ParameterSpace, levels: Dict[str, int] = None):
        """
        Initialize a factorial design generator.
        
        Args:
            parameter_space: Parameter space for generating designs
            levels: Dictionary mapping parameter names to number of levels
                   For categorical parameters, this is ignored and all levels are used.
                   For continuous/discrete parameters, this controls sampling resolution.
        """
        super().__init__(parameter_space)
        self.levels = levels or {}
        
        # Check if parameter space is too large for factorial design
        self._validate_parameter_space()
    
    def _validate_parameter_space(self):
        """
        Validate that parameter space is suitable for factorial design.
        """
        total_combinations = 1
        
        for param in self.parameter_space.parameters:
            if param.type.value == "categorical":
                level_count = len(param.values)
            elif param.name in self.levels:
                level_count = self.levels[param.name]
            else:
                # Default levels - 5 for continuous, min(10, all values) for discrete
                if param.type.value == "discrete":
                    level_count = min(10, (param.max - param.min) // param.step + 1)
                else:
                    level_count = 5
            
            total_combinations *= level_count
        
        if total_combinations > 10000:
            logger.warning(f"Factorial design will generate {total_combinations} points, which may be excessive.")
        
        return total_combinations
    
    def _get_parameter_levels(self, param, level_count):
        """
        Get specific levels for a parameter based on its type.
        
        Args:
            param: Parameter to get levels for
            level_count: Number of levels to generate
            
        Returns:
            List: List of parameter values at specified levels
        """
        if param.type.value == "categorical":
            return param.values
        
        if param.type.value == "discrete":
            # For discrete, use as many levels as possible up to level_count
            all_values = param.get_values()
            if len(all_values) <= level_count:
                return all_values
            
            # Select evenly spaced values
            indices = np.linspace(0, len(all_values) - 1, level_count, dtype=int)
            return [all_values[i] for i in indices]
        
        # For continuous, use evenly spaced values within range
        return np.linspace(param.min, param.max, level_count).tolist()
    
    def generate(self, n: int = None) -> List[Dict[str, Any]]:
        """
        Generate design points using factorial design.
        Note: For factorial design, the number of points is determined by the levels
        and n is ignored. It's only kept for API consistency.
        
        Args:
            n: Ignored for factorial design
            
        Returns:
            List[Dict[str, Any]]: List of design points
        """
        # Initialize with empty design
        design_points = [{}]
        
        # For each parameter, expand the design
        for param in self.parameter_space.parameters:
            # Determine number of levels for this parameter
            if param.type.value == "categorical":
                level_count = len(param.values)
            elif param.name in self.levels:
                level_count = self.levels[param.name]
            else:
                # Default levels - 5 for continuous, min(10, all values) for discrete
                if param.type.value == "discrete":
                    level_count = min(10, (param.max - param.min) // param.step + 1)
                else:
                    level_count = 5
            
            # Get specific levels
            levels = self._get_parameter_levels(param, level_count)
            
            # Create new expanded design
            new_design_points = []
            for point in design_points:
                for level in levels:
                    new_point = point.copy()
                    new_point[param.name] = level
                    new_design_points.append(new_point)
            
            design_points = new_design_points
        
        # Filter out points that don't satisfy constraints
        if self.parameter_space.constraints:
            valid_points = []
            for point in design_points:
                if self.parameter_space.is_valid_point(point):
                    valid_points.append(point)
            design_points = valid_points
        
        # If n is specified and less than total points, sample randomly
        if n is not None and n < len(design_points) and n > 0:
            design_points = random.sample(design_points, n)
        
        logger.info(f"Generated {len(design_points)} factorial design points")
        return design_points


class SobolDesignGenerator(DesignGenerator):
    """
    Generate design points using Sobol sequences.
    Requires scikit-learn or scipy for Sobol sequence generation.
    """
    def __init__(self, parameter_space: ParameterSpace, seed: Optional[int] = None):
        """
        Initialize a Sobol sequence design generator.
        
        Args:
            parameter_space: Parameter space for generating designs
            seed: Random seed for reproducibility
        """
        super().__init__(parameter_space)
        self.seed = seed
        
        # Check if required libraries are available
        try:
            from scipy.stats import qmc
            self.qmc = qmc
        except ImportError:
            try:
                from sklearn.experimental import enable_halton_sequences_
                from sklearn.preprocessing import QuantileTransformer
                self.sklearn_available = True
            except ImportError:
                raise ImportError("Either scipy or scikit-learn is required for Sobol sequence generation")
            
            self.sklearn_available = True
        else:
            self.sklearn_available = False
        
        # Check if parameter space has any constraints
        self.has_constraints = len(parameter_space.constraints) > 0
        
        if self.has_constraints:
            logger.warning("Parameter space has constraints. Sobol designs may not satisfy all constraints.")
    
    def generate(self, n: int) -> List[Dict[str, Any]]:
        """
        Generate n design points using Sobol sequences.
        
        Args:
            n: Number of design points to generate
            
        Returns:
            List[Dict[str, Any]]: List of design points
        """
        # Get internal dimensionality
        dim = self.parameter_space.get_internal_dimensions()
        
        # Generate Sobol samples in [0, 1] space
        if not self.sklearn_available:
            # Use scipy's implementation
            sampler = self.qmc.Sobol(d=dim, seed=self.seed)
            samples = sampler.random(n)
        else:
            # Fallback to less ideal method using sklearn
            import numpy as np
            from sklearn.preprocessing import QuantileTransformer
            
            # Generate random samples and transform to uniform distribution
            rng = np.random.RandomState(self.seed)
            X = rng.normal(size=(n, dim))
            
            qt = QuantileTransformer(output_distribution='uniform', random_state=self.seed)
            samples = qt.fit_transform(X)
        
        # Convert samples to parameter space
        designs = []
        for i in range(n):
            try:
                # Convert internal representation to point
                point = self.parameter_space.internal_to_point(samples[i])
                
                # Check if point satisfies constraints
                if not self.has_constraints or self.parameter_space.is_valid_point(point):
                    designs.append(point)
            except Exception as e:
                logger.warning(f"Error generating design point: {e}")
        
        # If we have constraints and couldn't generate enough designs, fill with random samples
        if len(designs) < n:
            logger.warning(f"Sobol generated only {len(designs)} valid points out of {n}. Filling remainder with random samples.")
            
            # Generate additional random points
            random_generator = RandomDesignGenerator(self.parameter_space)
            additional_points = random_generator.generate(n - len(designs))
            designs.extend(additional_points)
        
        return designs


class CustomDesignGenerator(DesignGenerator):
    """
    Generator for custom design points provided by the user.
    """
    def __init__(self, parameter_space: ParameterSpace, design_points: List[Dict[str, Any]]):
        """
        Initialize a custom design generator.
        
        Args:
            parameter_space: Parameter space for validating designs
            design_points: List of design points to use
        """
        super().__init__(parameter_space)
        
        # Validate design points
        self.validated_points = []
        
        for point in design_points:
            if self.parameter_space.is_valid_point(point):
                self.validated_points.append(point)
            else:
                logger.warning(f"Invalid design point: {point}")
        
        if not self.validated_points:
            raise ValueError("No valid design points provided")
    
    def generate(self, n: int) -> List[Dict[str, Any]]:
        """
        Return user-provided design points.
        If n is greater than available points, points will be reused.
        If n is less than available points, a subset will be returned.
        
        Args:
            n: Number of design points to generate
            
        Returns:
            List[Dict[str, Any]]: List of design points
        """
        if n <= len(self.validated_points):
            # Return first n points
            return self.validated_points[:n]
        
        # Need to generate more points than we have
        # Reuse points by cycling through them
        designs = []
        
        for i in range(n):
            idx = i % len(self.validated_points)
            designs.append(self.validated_points[idx])
        
        return designs


def create_design_generator(
    parameter_space: ParameterSpace, 
    design_type: DesignType,
    **kwargs
) -> DesignGenerator:
    """
    Factory function to create a design generator.
    
    Args:
        parameter_space: Parameter space for generating designs
        design_type: Type of design generator to create
        **kwargs: Additional parameters for specific design generators
    
    Returns:
        DesignGenerator: Design generator instance
    """
    if design_type == DesignType.RANDOM:
        return RandomDesignGenerator(parameter_space)
    
    elif design_type == DesignType.LATIN_HYPERCUBE:
        seed = kwargs.get("seed")
        return LatinHypercubeDesignGenerator(parameter_space, seed=seed)
    
    elif design_type == DesignType.FACTORIAL:
        levels = kwargs.get("levels")
        return FactorialDesignGenerator(parameter_space, levels=levels)
    
    elif design_type == DesignType.SOBOL:
        seed = kwargs.get("seed")
        return SobolDesignGenerator(parameter_space, seed=seed)
    
    elif design_type == DesignType.CUSTOM:
        design_points = kwargs.get("design_points")
        if not design_points:
            raise ValueError("Custom design generator requires design_points")
        return CustomDesignGenerator(parameter_space, design_points)
    
    else:
        raise ValueError(f"Unknown design type: {design_type}")


class BasicDesignGenerator:
    """
    设计生成器类，用于基于参数空间生成设计方案
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        seed: Optional[int] = None,
        output_dir: str = "designs"
    ):
        """
        初始化设计生成器
        
        Args:
            parameter_space: 参数空间对象
            seed: 随机种子，用于可重复性
            output_dir: 输出目录，用于保存生成的设计方案
        """
        self.parameter_space = parameter_space
        self.seed = seed
        self.output_dir = output_dir
        self.rng = np.random.RandomState(seed)
        
        # 确保输出目录存在
        ensure_directory_exists(self.output_dir)
        
        # 设置日志记录器
        logger.info(f"初始化设计生成器，参数空间维度: {len(parameter_space.get_parameters())}")

    def generate_random_designs(
        self,
        n_designs: int,
        ensure_constraints: bool = True,
        max_attempts: int = 100,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        生成随机设计方案
        
        Args:
            n_designs: 设计方案数量
            ensure_constraints: 是否确保满足约束条件
            max_attempts: 最大尝试次数（当ensure_constraints=True时）
            batch_size: 批处理大小，一次生成多少个设计方案
            
        Returns:
            List[Dict[str, Any]]: 生成的设计方案列表
        """
        logger.info(f"生成 {n_designs} 个随机设计方案")
        
        designs = []
        attempts = 0
        
        while len(designs) < n_designs and attempts < max_attempts:
            # 生成一批随机设计方案
            batch_size_current = min(batch_size, n_designs - len(designs))
            batch_designs = []
            
            for _ in range(batch_size_current):
                design = {}
                
                # 为每个参数生成随机值
                for param_name, param_config in self.parameter_space.get_parameters().items():
                    param_type = param_config["type"]
                    
                    if param_type == "continuous":
                        min_val = param_config["min"]
                        max_val = param_config["max"]
                        value = self.rng.uniform(min_val, max_val)
                        design[param_name] = float(value)
                    
                    elif param_type == "integer":
                        min_val = param_config["min"]
                        max_val = param_config["max"]
                        value = self.rng.randint(min_val, max_val + 1)
                        design[param_name] = int(value)
                    
                    elif param_type == "categorical":
                        categories = param_config["categories"]
                        value = self.rng.choice(categories)
                        design[param_name] = value
                
                batch_designs.append(design)
            
            # 检查约束条件
            if ensure_constraints and self.parameter_space.has_constraints():
                valid_designs = []
                for design in batch_designs:
                    if self.parameter_space.check_constraints(design):
                        valid_designs.append(design)
                batch_designs = valid_designs
            
            # 添加到设计方案列表
            designs.extend(batch_designs)
            attempts += 1
        
        # 如果未生成足够的设计方案，记录警告
        if len(designs) < n_designs:
            logger.warning(f"未能生成足够的设计方案，请求: {n_designs}，生成: {len(designs)}")
        
        # 保存设计方案
        saved_designs = []
        for design in designs[:n_designs]:
            design_id = self._save_design(design)
            design["id"] = design_id
            saved_designs.append(design)
        
        return saved_designs[:n_designs]

    def generate_lhs_designs(
        self,
        n_designs: int,
        ensure_constraints: bool = True,
        max_attempts: int = 100,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        使用拉丁超立方抽样生成设计方案
        
        Args:
            n_designs: 设计方案数量
            ensure_constraints: 是否确保满足约束条件
            max_attempts: 最大尝试次数（当ensure_constraints=True时）
            batch_size: 批处理大小，一次生成多少个设计方案
            
        Returns:
            List[Dict[str, Any]]: 生成的设计方案列表
        """
        logger.info(f"使用拉丁超立方抽样生成 {n_designs} 个设计方案")
        
        # 检查是否包含分类参数
        has_categorical = False
        for param_name, param_config in self.parameter_space.get_parameters().items():
            if param_config["type"] == "categorical":
                has_categorical = True
                logger.warning(f"参数 {param_name} 是分类参数，拉丁超立方抽样可能不是最佳选择")
        
        # 获取连续参数和整数参数的数量
        continuous_params = []
        for param_name, param_config in self.parameter_space.get_parameters().items():
            if param_config["type"] in ["continuous", "integer"]:
                continuous_params.append(param_name)
        
        num_continuous = len(continuous_params)
        
        if num_continuous == 0:
            logger.warning("没有连续参数或整数参数，拉丁超立方抽样可能不是最佳选择")
            return self.generate_random_designs(n_designs, ensure_constraints, max_attempts, batch_size)
        
        designs = []
        attempts = 0
        
        while len(designs) < n_designs and attempts < max_attempts:
            # 生成拉丁超立方样本
            batch_size_current = min(batch_size, n_designs - len(designs))
            
            # 对连续参数和整数参数使用拉丁超立方抽样
            lhs_samples = latin_hypercube_sampling(batch_size_current, num_continuous, self.seed)
            
            batch_designs = []
            for i in range(batch_size_current):
                design = {}
                
                # 转换拉丁超立方样本为参数值
                for j, param_name in enumerate(continuous_params):
                    param_config = self.parameter_space.get_parameters()[param_name]
                    param_type = param_config["type"]
                    min_val = param_config["min"]
                    max_val = param_config["max"]
                    
                    # 从[0, 1]范围转换为参数范围
                    value = min_val + lhs_samples[i, j] * (max_val - min_val)
                    
                    if param_type == "integer":
                        value = int(round(value))
                        # 确保值在范围内
                        value = max(min_val, min(max_val, value))
                    
                    design[param_name] = value
                
                # 对分类参数随机取值
                for param_name, param_config in self.parameter_space.get_parameters().items():
                    if param_config["type"] == "categorical":
                        categories = param_config["categories"]
                        value = self.rng.choice(categories)
                        design[param_name] = value
                
                batch_designs.append(design)
            
            # 检查约束条件
            if ensure_constraints and self.parameter_space.has_constraints():
                valid_designs = []
                for design in batch_designs:
                    if self.parameter_space.check_constraints(design):
                        valid_designs.append(design)
                batch_designs = valid_designs
            
            # 添加到设计方案列表
            designs.extend(batch_designs)
            attempts += 1
        
        # 如果未生成足够的设计方案，记录警告
        if len(designs) < n_designs:
            logger.warning(f"未能生成足够的设计方案，请求: {n_designs}，生成: {len(designs)}")
        
        # 保存设计方案
        saved_designs = []
        for design in designs[:n_designs]:
            design_id = self._save_design(design)
            design["id"] = design_id
            saved_designs.append(design)
        
        return saved_designs[:n_designs]

    def generate_grid_designs(
        self,
        n_divisions: int = 5,
        ensure_constraints: bool = True
    ) -> List[Dict[str, Any]]:
        """
        使用网格搜索生成设计方案
        
        Args:
            n_divisions: 每个维度的划分数量
            ensure_constraints: 是否确保满足约束条件
            
        Returns:
            List[Dict[str, Any]]: 生成的设计方案列表
        """
        logger.info(f"使用网格搜索生成设计方案，每个维度划分数量: {n_divisions}")
        
        # 检查是否包含分类参数
        for param_name, param_config in self.parameter_space.get_parameters().items():
            if param_config["type"] == "categorical":
                logger.warning(f"参数 {param_name} 是分类参数，网格搜索将考虑所有分类值")
        
        # 为每个参数生成网格点
        param_grids = {}
        
        for param_name, param_config in self.parameter_space.get_parameters().items():
            param_type = param_config["type"]
            
            if param_type == "continuous":
                min_val = param_config["min"]
                max_val = param_config["max"]
                grid = np.linspace(min_val, max_val, n_divisions)
                param_grids[param_name] = grid.tolist()
            
            elif param_type == "integer":
                min_val = param_config["min"]
                max_val = param_config["max"]
                # 对于整数参数，确保网格点是整数
                grid = np.linspace(min_val, max_val, min(n_divisions, max_val - min_val + 1))
                param_grids[param_name] = [int(round(x)) for x in grid]
            
            elif param_type == "categorical":
                categories = param_config["categories"]
                param_grids[param_name] = categories
        
        # 计算网格点总数
        total_grid_points = 1
        for param_name, grid in param_grids.items():
            total_grid_points *= len(grid)
        
        logger.info(f"网格搜索将生成 {total_grid_points} 个设计方案")
        
        if total_grid_points > 10000:
            logger.warning(f"网格搜索将生成大量设计方案 ({total_grid_points})，可能需要较长时间")
        
        # 生成所有组合
        designs = []
        
        # 递归生成所有组合
        def generate_combinations(current_design, params_left):
            if not params_left:
                # 所有参数都已设置，添加到设计方案列表
                designs.append(current_design.copy())
                return
            
            # 取出一个参数
            param_name = list(params_left.keys())[0]
            param_values = params_left[param_name]
            remaining_params = {k: v for k, v in params_left.items() if k != param_name}
            
            # 对该参数的每个值生成组合
            for value in param_values:
                current_design[param_name] = value
                generate_combinations(current_design, remaining_params)
        
        # 开始生成组合
        generate_combinations({}, param_grids)
        
        # 检查约束条件
        if ensure_constraints and self.parameter_space.has_constraints():
            valid_designs = []
            for design in designs:
                if self.parameter_space.check_constraints(design):
                    valid_designs.append(design)
            designs = valid_designs
        
        # 保存设计方案
        saved_designs = []
        for design in designs:
            design_id = self._save_design(design)
            design["id"] = design_id
            saved_designs.append(design)
        
        return saved_designs

    def generate_custom_designs(
        self,
        designs: List[Dict[str, Any]],
        validate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        生成自定义设计方案
        
        Args:
            designs: 自定义设计方案列表
            validate: 是否验证设计方案
            
        Returns:
            List[Dict[str, Any]]: 生成的设计方案列表
        """
        logger.info(f"生成 {len(designs)} 个自定义设计方案")
        
        valid_designs = []
        
        for design in designs:
            # 验证设计方案
            if validate:
                is_valid = True
                
                # 检查参数是否完整
                for param_name in self.parameter_space.get_parameters():
                    if param_name not in design:
                        logger.warning(f"设计方案缺少参数: {param_name}")
                        is_valid = False
                        break
                
                # 检查参数值是否有效
                if is_valid:
                    for param_name, value in design.items():
                        if param_name not in self.parameter_space.get_parameters():
                            logger.warning(f"未知参数: {param_name}")
                            is_valid = False
                            break
                        
                        param_config = self.parameter_space.get_parameters()[param_name]
                        param_type = param_config["type"]
                        
                        if param_type == "continuous":
                            if not isinstance(value, (int, float)):
                                logger.warning(f"参数 {param_name} 应为数值类型，但得到 {type(value)}")
                                is_valid = False
                                break
                            
                            min_val = param_config["min"]
                            max_val = param_config["max"]
                            if value < min_val or value > max_val:
                                logger.warning(f"参数 {param_name} 的值 {value} 超出范围 [{min_val}, {max_val}]")
                                is_valid = False
                                break
                        
                        elif param_type == "integer":
                            if not isinstance(value, int):
                                logger.warning(f"参数 {param_name} 应为整数类型，但得到 {type(value)}")
                                is_valid = False
                                break
                            
                            min_val = param_config["min"]
                            max_val = param_config["max"]
                            if value < min_val or value > max_val:
                                logger.warning(f"参数 {param_name} 的值 {value} 超出范围 [{min_val}, {max_val}]")
                                is_valid = False
                                break
                        
                        elif param_type == "categorical":
                            categories = param_config["categories"]
                            if value not in categories:
                                logger.warning(f"参数 {param_name} 的值 {value} 不在有效类别中: {categories}")
                                is_valid = False
                                break
                
                # 检查约束条件
                if is_valid and self.parameter_space.has_constraints():
                    if not self.parameter_space.check_constraints(design):
                        logger.warning("设计方案不满足约束条件")
                        is_valid = False
                
                if not is_valid:
                    continue
            
            # 保存设计方案
            design_id = self._save_design(design)
            design["id"] = design_id
            valid_designs.append(design)
        
        return valid_designs

    def _save_design(self, design: Dict[str, Any]) -> str:
        """
        保存设计方案到文件
        
        Args:
            design: 设计方案
            
        Returns:
            str: 设计方案ID
        """
        # 生成唯一ID
        design_id = generate_unique_id()
        
        # 添加ID到设计方案
        design["id"] = design_id
        
        # 保存设计方案
        filepath = os.path.join(self.output_dir, f"{design_id}.json")
        save_to_json(design, filepath)
        
        return design_id

    def load_design(self, design_id: str) -> Optional[Dict[str, Any]]:
        """
        加载设计方案
        
        Args:
            design_id: 设计方案ID
            
        Returns:
            Optional[Dict[str, Any]]: 加载的设计方案，如果不存在则返回None
        """
        filepath = os.path.join(self.output_dir, f"{design_id}.json")
        return load_from_json(filepath)

    def load_all_designs(self) -> List[Dict[str, Any]]:
        """
        加载所有设计方案
        
        Returns:
            List[Dict[str, Any]]: 加载的设计方案列表
        """
        designs = []
        
        # 获取输出目录中的所有JSON文件
        json_files = [f for f in os.listdir(self.output_dir) if f.endswith(".json")]
        
        for json_file in json_files:
            filepath = os.path.join(self.output_dir, json_file)
            design = load_from_json(filepath)
            if design is not None:
                designs.append(design)
        
        return designs

    def filter_designs(
        self,
        designs: List[Dict[str, Any]],
        filter_fn: Callable[[Dict[str, Any]], bool]
    ) -> List[Dict[str, Any]]:
        """
        过滤设计方案
        
        Args:
            designs: 设计方案列表
            filter_fn: 过滤函数，接收设计方案并返回布尔值
            
        Returns:
            List[Dict[str, Any]]: 过滤后的设计方案列表
        """
        return [design for design in designs if filter_fn(design)] 
