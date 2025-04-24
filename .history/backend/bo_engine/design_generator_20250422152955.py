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

from bo_engine.parameter_space import ParameterSpace, Parameter
from bo_engine.utils import (
    latin_hypercube_sampling,
    ensure_directory_exists,
    save_to_json,
    load_from_json,
    generate_unique_id
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


class DesignGenerator:
    """设计生成器类，用于根据参数空间生成设计方案"""

    def __init__(
        self,
        parameter_space: ParameterSpace,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None
    ):
        """
        初始化设计生成器
        
        Args:
            parameter_space: 参数空间
            seed: 随机种子
            output_dir: 输出目录
        """
        self.parameter_space = parameter_space
        self.seed = seed
        self.output_dir = output_dir or os.path.join(os.getcwd(), "outputs")
        
        # 创建输出目录
        ensure_directory_exists(self.output_dir)
        
        # 创建随机数生成器
        self.rng = np.random.RandomState(seed)
    
    def generate_random_designs(
        self,
        n_designs: int,
        ensure_constraints: bool = True,
        max_attempts: int = 10,
        save_designs: bool = True,
        design_id_prefix: str = "design_"
    ) -> List[Dict[str, Any]]:
        """
        生成随机设计方案
        
        Args:
            n_designs: 设计方案数量
            ensure_constraints: 是否确保满足约束条件
            max_attempts: 最大尝试次数（对于每个设计方案）
            save_designs: 是否保存设计方案
            design_id_prefix: 设计方案ID前缀
            
        Returns:
            List[Dict[str, Any]]: 设计方案列表
        """
        designs = []
        
        # 生成随机设计方案
        attempts = 0
        while len(designs) < n_designs and attempts < max_attempts * n_designs:
            # 为每个参数生成随机值
            design_dict = {}
            for param in self.parameter_space.parameters:
                # 根据参数类型采样
                sample = param.sample(n=1, rng=self.rng)[0]
                design_dict[param.name] = sample
            
            # 检查约束条件
            if not ensure_constraints or self.parameter_space.check_constraints(design_dict):
                # 添加设计方案ID
                design_id = f"{design_id_prefix}{generate_unique_id()}"
                design_dict["design_id"] = design_id
                designs.append(design_dict)
            
            attempts += 1
        
        if len(designs) < n_designs:
            logger.warning(f"未能生成 {n_designs} 个设计方案，仅生成了 {len(designs)} 个")
        
        # 保存设计方案
        if save_designs and designs:
            for design in designs:
                self._save_design(design)
        
        return designs
    
    def generate_lhs_designs(
        self,
        n_designs: int,
        ensure_constraints: bool = True,
        max_attempts: int = 10,
        save_designs: bool = True,
        design_id_prefix: str = "design_"
    ) -> List[Dict[str, Any]]:
        """
        使用拉丁超立方抽样生成设计方案
        
        Args:
            n_designs: 设计方案数量
            ensure_constraints: 是否确保满足约束条件
            max_attempts: 最大尝试次数
            save_designs: 是否保存设计方案
            design_id_prefix: 设计方案ID前缀
            
        Returns:
            List[Dict[str, Any]]: 设计方案列表
        """
        dimensions = self.parameter_space.get_dimensions()
        
        # 生成拉丁超立方样本 [0, 1]^d
        lhs_samples = latin_hypercube_sampling(n_designs, dimensions, seed=self.seed)
        
        designs = []
        attempts = 0
        max_total_attempts = max_attempts * n_designs
        
        while len(designs) < n_designs and attempts < max_total_attempts:
            # 如果尝试次数超过原始样本数，重新生成样本
            if attempts >= n_designs:
                new_seed = None if self.seed is None else self.seed + attempts
                lhs_samples = latin_hypercube_sampling(
                    n_designs, dimensions, seed=new_seed
                )
            
            # 获取当前尝试的样本索引
            idx = attempts % n_designs
            
            # 转换标准化样本为实际参数值
            sample = lhs_samples[idx]
            design_dict = self.parameter_space.inverse_transform(sample)
            
            # 检查约束条件
            if not ensure_constraints or self.parameter_space.check_constraints(design_dict):
                # 添加设计方案ID
                design_id = f"{design_id_prefix}{generate_unique_id()}"
                design_dict["design_id"] = design_id
                designs.append(design_dict)
            
            attempts += 1
        
        if len(designs) < n_designs:
            logger.warning(f"未能生成 {n_designs} 个设计方案，仅生成了 {len(designs)} 个")
        
        # 保存设计方案
        if save_designs and designs:
            for design in designs:
                self._save_design(design)
        
        return designs
    
    def generate_grid_designs(
        self,
        n_points_per_dim: int = 5,
        ensure_constraints: bool = True,
        save_designs: bool = True,
        design_id_prefix: str = "design_"
    ) -> List[Dict[str, Any]]:
        """
        使用网格搜索生成设计方案（仅适用于连续参数和整数参数）
        
        Args:
            n_points_per_dim: 每个维度的点数
            ensure_constraints: 是否确保满足约束条件
            save_designs: 是否保存设计方案
            design_id_prefix: 设计方案ID前缀
            
        Returns:
            List[Dict[str, Any]]: 设计方案列表
        """
        # 检查是否存在分类参数
        has_categorical = False
        for param in self.parameter_space.parameters:
            if param.parameter_type.name in ["CATEGORICAL", "ORDINAL"]:
                has_categorical = True
                break
        
        if has_categorical:
            logger.warning("网格搜索不适用于包含分类参数的参数空间，将使用拉丁超立方抽样代替")
            return self.generate_lhs_designs(
                n_designs=n_points_per_dim**2,  # 使用平方作为样本数
                ensure_constraints=ensure_constraints,
                save_designs=save_designs,
                design_id_prefix=design_id_prefix
            )
        
        # 为每个参数创建网格点
        grid_points = []
        for param in self.parameter_space.parameters:
            if param.parameter_type.name == "CONTINUOUS":
                # 连续参数使用均匀间隔
                points = np.linspace(0, 1, n_points_per_dim)
            elif param.parameter_type.name == "INTEGER":
                # 整数参数使用离散点
                n_values = min(n_points_per_dim, param.max_value - param.min_value + 1)
                points = np.linspace(0, 1, n_values)
            else:
                # 不应该到达这里
                points = np.linspace(0, 1, n_points_per_dim)
            
            grid_points.append(points)
        
        # 生成网格
        mesh = np.meshgrid(*grid_points)
        grid_coords = np.column_stack([m.flatten() for m in mesh])
        
        # 转换为设计方案
        designs = []
        for coords in grid_coords:
            design_dict = self.parameter_space.inverse_transform(coords)
            
            # 检查约束条件
            if not ensure_constraints or self.parameter_space.check_constraints(design_dict):
                # 添加设计方案ID
                design_id = f"{design_id_prefix}{generate_unique_id()}"
                design_dict["design_id"] = design_id
                designs.append(design_dict)
        
        logger.info(f"网格搜索生成了 {len(designs)} 个设计方案")
        
        # 保存设计方案
        if save_designs and designs:
            for design in designs:
                self._save_design(design)
        
        return designs
    
    def generate_custom_designs(
        self,
        design_points: List[Dict[str, Any]],
        ensure_constraints: bool = True,
        save_designs: bool = True,
        design_id_prefix: str = "design_"
    ) -> List[Dict[str, Any]]:
        """
        根据自定义点生成设计方案
        
        Args:
            design_points: 自定义设计点列表
            ensure_constraints: 是否确保满足约束条件
            save_designs: 是否保存设计方案
            design_id_prefix: 设计方案ID前缀
            
        Returns:
            List[Dict[str, Any]]: 设计方案列表
        """
        designs = []
        
        for design_dict in design_points:
            # 检查所有参数是否存在
            missing_params = set(self.parameter_space.get_parameter_names()) - set(design_dict.keys())
            if missing_params:
                logger.warning(f"设计方案缺少参数: {missing_params}，跳过")
                continue
            
            # 检查约束条件
            if not ensure_constraints or self.parameter_space.check_constraints(design_dict):
                # 添加设计方案ID（如果不存在）
                if "design_id" not in design_dict:
                    design_dict["design_id"] = f"{design_id_prefix}{generate_unique_id()}"
                designs.append(design_dict)
        
        logger.info(f"从 {len(design_points)} 个自定义点中生成了 {len(designs)} 个有效设计方案")
        
        # 保存设计方案
        if save_designs and designs:
            for design in designs:
                self._save_design(design)
        
        return designs
    
    def _save_design(self, design: Dict[str, Any]) -> str:
        """
        保存设计方案到文件
        
        Args:
            design: 设计方案字典
            
        Returns:
            str: 保存的文件路径
        """
        design_id = design.get("design_id")
        if not design_id:
            design_id = f"design_{generate_unique_id()}"
            design["design_id"] = design_id
        
        # 创建设计方案目录
        designs_dir = os.path.join(self.output_dir, "designs")
        ensure_directory_exists(designs_dir)
        
        # 保存设计方案
        filepath = os.path.join(designs_dir, f"{design_id}.json")
        save_to_json(design, filepath)
        
        return filepath
    
    def load_design(self, design_id: str) -> Optional[Dict[str, Any]]:
        """
        从文件加载设计方案
        
        Args:
            design_id: 设计方案ID
            
        Returns:
            Optional[Dict[str, Any]]: 设计方案字典，如果不存在则返回None
        """
        designs_dir = os.path.join(self.output_dir, "designs")
        filepath = os.path.join(designs_dir, f"{design_id}.json")
        
        if not os.path.exists(filepath):
            # 尝试添加前缀
            filepath = os.path.join(designs_dir, f"design_{design_id}.json")
            if not os.path.exists(filepath):
                logger.warning(f"设计方案文件不存在: {filepath}")
                return None
        
        return load_from_json(filepath)
    
    def load_all_designs(self) -> List[Dict[str, Any]]:
        """
        加载所有设计方案
        
        Returns:
            List[Dict[str, Any]]: 设计方案列表
        """
        designs_dir = os.path.join(self.output_dir, "designs")
        
        if not os.path.exists(designs_dir):
            logger.warning(f"设计方案目录不存在: {designs_dir}")
            return []
        
        designs = []
        for filename in os.listdir(designs_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(designs_dir, filename)
                design = load_from_json(filepath)
                if design:
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
            filter_fn: 过滤函数，返回True表示保留
            
        Returns:
            List[Dict[str, Any]]: 过滤后的设计方案列表
        """
        return [design for design in designs if filter_fn(design)] 
