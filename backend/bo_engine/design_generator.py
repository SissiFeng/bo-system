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
    Generate design points using random sampling.
    """
    def __init__(self, parameter_space: ParameterSpace, seed: Optional[int] = None):
        """
        Initialize a random design generator.
        
        Args:
            parameter_space: Parameter space for generating designs
            seed: Random seed for reproducibility
        """
        super().__init__(parameter_space)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Initialize constraint handler
        from .utils import ConstraintHandler
        self.constraint_handler = ConstraintHandler(parameter_space)
    
    def generate(self, n: int) -> List[Dict[str, Any]]:
        """
        Generate n random design points.
        
        Args:
            n: Number of design points to generate
            
        Returns:
            List[Dict[str, Any]]: List of design points
        """
        if n <= 0:
            return []
        
        # Use adaptive sampling with constraint handler if constraints exist
        if self.parameter_space.has_constraints():
            # Define a generator function for random sampling
            def random_generator(num_points):
                points = []
                for _ in range(num_points):
                    point = {}
                    for param_name, param_config in self.parameter_space.parameters.items():
                        param_type = param_config["type"]
                        
                        if param_type == "continuous":
                            min_val = param_config["min"]
                            max_val = param_config["max"]
                            point[param_name] = float(self.rng.uniform(min_val, max_val))
                        
                        elif param_type == "integer":
                            min_val = param_config["min"]
                            max_val = param_config["max"]
                            point[param_name] = int(self.rng.randint(min_val, max_val + 1))
                        
                        elif param_type == "categorical":
                            categories = param_config["categories"]
                            point[param_name] = self.rng.choice(categories)
                    
                    points.append(point)
                return points
            
            # Use adaptive sampling to get valid points
            logger.info(f"Using adaptive sampling to generate {n} random design points with constraints")
            return self.constraint_handler.adaptive_sampling(n, random_generator)
        
        # If no constraints, use simple random sampling
        designs = []
        for _ in range(n):
            design = {}
            
            for param_name, param_config in self.parameter_space.parameters.items():
                param_type = param_config["type"]
                
                if param_type == "continuous":
                    min_val = param_config["min"]
                    max_val = param_config["max"]
                    design[param_name] = float(self.rng.uniform(min_val, max_val))
                
                elif param_type == "integer":
                    min_val = param_config["min"]
                    max_val = param_config["max"]
                    design[param_name] = int(self.rng.randint(min_val, max_val + 1))
                
                elif param_type == "categorical":
                    categories = param_config["categories"]
                    design[param_name] = self.rng.choice(categories)
            
            designs.append(design)
        
        logger.info(f"Generated {len(designs)} random design points")
        return designs


class LatinHypercubeDesignGenerator(DesignGenerator):
    """
    Generate design points using Latin Hypercube Sampling (LHS).
    """
    def __init__(
        self, 
        parameter_space: ParameterSpace, 
        seed: Optional[int] = None,
        strength: int = 1,
        optimization: Optional[str] = None
    ):
        """
        Initialize a Latin Hypercube design generator.
        
        Args:
            parameter_space: Parameter space for generating designs
            seed: Random seed for reproducibility
            strength: Correlation control strength (1=no optimization, 2=stronger)
            optimization: Optional optimization method:
                         - "correlation": Minimize correlation between columns
                         - "maximin": Maximize minimum distance between points
                         - "centermaximin": Center points within cells and maximize min distance
                         - None: No optimization
        """
        super().__init__(parameter_space)
        self.seed = seed
        self.strength = strength
        self.optimization = optimization
        
        # Initialize constraint handler
        from .utils import ConstraintHandler
        self.constraint_handler = ConstraintHandler(parameter_space)
        
        # Import scipy for better Latin Hypercube Sampling if available
        try:
            from scipy.stats import qmc
            self.qmc = qmc
            self.has_scipy_qmc = True
        except ImportError:
            self.has_scipy_qmc = False
        
        # Check if parameter space has any constraints
        self.has_constraints = parameter_space.has_constraints()
        
        if self.has_constraints:
            logger.info("Parameter space has constraints. Using advanced constraint handling.")
    
    def _generate_lhs_points_scipy(self, n, dim):
        """
        Generate Latin hypercube points using scipy's implementation
        
        Args:
            n: Number of points
            dim: Dimensionality
            
        Returns:
            np.ndarray: LHS samples in [0,1] space
        """
        try:
            # Use scipy's more advanced LHS implementation
            sampler = self.qmc.LatinHypercube(
                d=dim, 
                strength=self.strength,
                optimization=self.optimization,
                seed=self.seed
            )
            samples = sampler.random(n)
            
            # Use a second optimization step if requested
            if self.optimization == "correlation":
                # Reduce correlation between parameters
                self.qmc.discrepancy(samples, iterative=True, method="L2-centered")
            elif self.optimization == "maximin":
                # Try to maximize the minimum distance
                try:
                    from .utils import maximize_min_distance
                    # Generate multiple candidate sets and pick the best one
                    candidates = []
                    for i in range(3):
                        candidates.append(sampler.random(n))
                    
                    # Select the best candidate
                    best_idx = 0
                    best_min_dist = 0
                    for i, candidate in enumerate(candidates):
                        from scipy.spatial.distance import pdist
                        min_dist = np.min(pdist(candidate))
                        if min_dist > best_min_dist:
                            best_min_dist = min_dist
                            best_idx = i
                    
                    samples = candidates[best_idx]
                except Exception as e:
                    logger.warning(f"Maximin optimization failed: {e}")
            
            return samples
            
        except Exception as e:
            logger.warning(f"Error using scipy's LHS implementation: {e}")
            # Fall back to simple LHS implementation
            from .utils import latin_hypercube_sampling
            return latin_hypercube_sampling(n, dim, self.seed)
    
    def _generate_lhs_points_internal(self, n, dim):
        """
        Generate Latin hypercube points using internal implementation
        
        Args:
            n: Number of points
            dim: Dimensionality
            
        Returns:
            np.ndarray: LHS samples in [0,1] space
        """
        from .utils import latin_hypercube_sampling
        return latin_hypercube_sampling(n, dim, self.seed)
    
    def generate(self, n: int) -> List[Dict[str, Any]]:
        """
        Generate n design points using Latin Hypercube Sampling.
        
        Args:
            n: Number of design points to generate
            
        Returns:
            List[Dict[str, Any]]: List of design points
        """
        if n <= 0:
            return []
        
        # Get internal dimensionality
        dim = self.parameter_space.get_internal_dimensions()
        
        # Define LHS generator function for constraint handler
        def lhs_generator(num_points):
            # Generate LHS samples in [0, 1] space
            if self.has_scipy_qmc:
                samples = self._generate_lhs_points_scipy(num_points, dim)
            else:
                samples = self._generate_lhs_points_internal(num_points, dim)
            
            # Convert samples to parameter space
            designs = []
            for i in range(num_points):
                try:
                    # Convert internal representation to point
                    point = self.parameter_space.internal_to_point(samples[i])
                    designs.append(point)
                except Exception as e:
                    logger.warning(f"Error generating LHS design point: {e}")
            
            return designs
        
        # Use constraint handler if constraints exist
        if self.has_constraints:
            logger.info(f"Using space-filling sampling to generate {n} LHS points with constraints")
            # Use space-filling sampling for better distribution with constraints
            return self.constraint_handler.space_filling_sampling(n, [], lhs_generator)
        
        # If no constraints, generate LHS points directly
        designs = lhs_generator(n)
        
        # Evaluate design quality 
        try:
            if designs and len(designs) > 1:
                from .utils import evaluate_lhs_quality
                points_array = np.array([list(point.values()) for point in designs])
                quality = evaluate_lhs_quality(points_array)
                logger.info(f"LHS design quality: min distance = {quality['min_distance']:.4f}, "
                           f"uniformity (CV) = {quality['cv']:.4f}")
        except Exception as e:
            logger.debug(f"Could not evaluate LHS quality: {e}")
        
        logger.info(f"Generated {len(designs)} LHS design points")
        return designs


class FactorialDesignGenerator(DesignGenerator):
    """
    Generate design points using factorial design.
    Only practical for small parameter spaces with discrete/categorical parameters.
    Includes options for layered processing to handle larger parameter spaces.
    """
    def __init__(
        self, 
        parameter_space: ParameterSpace, 
        levels: Dict[str, int] = None,
        max_combinations: int = 10000,
        adaptive_sampling: bool = False,
        parameter_groups: List[List[str]] = None,
        importance_based: bool = False
    ):
        """
        Initialize a factorial design generator.
        
        Args:
            parameter_space: Parameter space for generating designs
            levels: Dictionary mapping parameter names to number of levels
                   For categorical parameters, this is ignored and all levels are used.
                   For continuous/discrete parameters, this controls sampling resolution.
            max_combinations: Maximum number of combinations to generate before using layered approach
            adaptive_sampling: If True, use adaptive sampling for high-dimensional spaces
            parameter_groups: List of parameter groups to process in layers
                             If None, parameters will be automatically grouped if needed
            importance_based: If True, use parameter importance to determine sampling priority
        """
        super().__init__(parameter_space)
        self.levels = levels or {}
        self.max_combinations = max_combinations
        self.adaptive_sampling = adaptive_sampling
        self.parameter_groups = parameter_groups
        self.importance_based = importance_based
        
        # Check if parameter space is too large for factorial design
        self.total_combinations = self._validate_parameter_space()
        self.layered_approach = self.total_combinations > self.max_combinations
        
        # Automatically compute parameter groups if needed and not provided
        if self.layered_approach and not self.parameter_groups:
            self._compute_parameter_groups()
    
    def _validate_parameter_space(self):
        """
        Validate that parameter space is suitable for factorial design.
        
        Returns:
            int: Total number of combinations
        """
        total_combinations = 1
        self.parameter_level_counts = {}
        
        for param_name, param_config in self.parameter_space.parameters.items():
            param_type = param_config["type"]
            
            if param_type == "categorical":
                level_count = len(param_config["categories"])
            elif param_name in self.levels:
                level_count = self.levels[param_name]
            else:
                # Default levels - 5 for continuous, min(10, all values) for discrete
                if param_type == "integer":
                    # For discrete integer parameters
                    min_val = param_config["min"]
                    max_val = param_config["max"]
                    level_count = min(10, max_val - min_val + 1)
                else:
                    level_count = 5
            
            self.parameter_level_counts[param_name] = level_count
            total_combinations *= level_count
        
        if total_combinations > self.max_combinations:
            logger.warning(f"Factorial design would generate {total_combinations} points, "
                          f"which exceeds the maximum of {self.max_combinations}. "
                          f"Using layered approach instead.")
        
        return total_combinations
    
    def _compute_parameter_groups(self):
        """
        Compute parameter groups for layered approach.
        This splits parameters into groups that can be processed separately.
        """
        # Sort parameters by importance if specified, otherwise by number of levels
        if self.importance_based:
            # Try to get importance from parameter metadata
            params_with_importance = []
            for param_name, param_config in self.parameter_space.parameters.items():
                importance = param_config.get("importance", 1.0)
                params_with_importance.append((param_name, importance))
            
            # Sort by importance (higher first)
            sorted_params = [p[0] for p in sorted(params_with_importance, 
                                                 key=lambda x: x[1], 
                                                 reverse=True)]
        else:
            # Sort by level count (fewer levels first to minimize combinations)
            sorted_params = [p[0] for p in sorted(self.parameter_level_counts.items(), 
                                                key=lambda x: x[1])]
        
        # Create groups with manageable combination counts
        groups = []
        current_group = []
        current_combinations = 1
        
        for param_name in sorted_params:
            level_count = self.parameter_level_counts[param_name]
            new_combinations = current_combinations * level_count
            
            if new_combinations <= self.max_combinations:
                current_group.append(param_name)
                current_combinations = new_combinations
            else:
                if current_group:  # Only add non-empty groups
                    groups.append(current_group)
                current_group = [param_name]
                current_combinations = level_count
        
        if current_group:  # Add the last group if not empty
            groups.append(current_group)
        
        self.parameter_groups = groups
        logger.info(f"Created {len(groups)} parameter groups for layered factorial design")
        for i, group in enumerate(groups):
            logger.debug(f"Group {i+1}: {group}")
    
    def _get_parameter_levels(self, param_name, param_config, level_count):
        """
        Get specific levels for a parameter based on its type.
        
        Args:
            param_name: Name of the parameter
            param_config: Parameter configuration dictionary
            level_count: Number of levels to generate
            
        Returns:
            List: List of parameter values at specified levels
        """
        param_type = param_config["type"]
        
        if param_type == "categorical":
            return param_config["categories"]
        
        if param_type == "integer":
            # For discrete integer parameters
            min_val = param_config["min"]
            max_val = param_config["max"]
            
            # For discrete, use as many levels as possible up to level_count
            if max_val - min_val + 1 <= level_count:
                return list(range(min_val, max_val + 1))
            
            # Select evenly spaced values
            indices = np.linspace(min_val, max_val, level_count, dtype=int)
            return sorted(set(indices.tolist()))  # Remove duplicates if any
        
        # For continuous, use evenly spaced values within range
        min_val = param_config["min"]
        max_val = param_config["max"]
        return np.linspace(min_val, max_val, level_count).tolist()
    
    def _generate_for_group(self, param_group, n=None):
        """
        Generate factorial design points for a specific parameter group.
        
        Args:
            param_group: List of parameter names to include
            n: Optional limit on number of points to generate
            
        Returns:
            List[Dict[str, Any]]: List of design points for this group
        """
        # Start with an empty design point
        design_points = [{}]
        
        # For each parameter in the group, expand the design
        for param_name in param_group:
            param_config = self.parameter_space.parameters[param_name]
            
            # Determine number of levels for this parameter
            if param_config["type"] == "categorical":
                level_count = len(param_config["categories"])
            elif param_name in self.levels:
                level_count = self.levels[param_name]
            else:
                # Default levels - 5 for continuous, min(10, all values) for discrete
                if param_config["type"] == "integer":
                    min_val = param_config["min"]
                    max_val = param_config["max"]
                    level_count = min(10, max_val - min_val + 1)
                else:
                    level_count = 5
            
            # Get specific levels
            levels = self._get_parameter_levels(param_name, param_config, level_count)
            
            # Create new expanded design
            new_design_points = []
            for point in design_points:
                for level in levels:
                    new_point = point.copy()
                    new_point[param_name] = level
                    new_design_points.append(new_point)
            
            design_points = new_design_points
            
            # Early stopping if we already have enough points
            if n is not None and len(design_points) >= n:
                # Randomly sample to exactly n points
                design_points = random.sample(design_points, n)
                break
        
        return design_points
    
    def _merge_design_groups(self, group_designs, n=None):
        """
        Merge design points from different parameter groups.
        
        Args:
            group_designs: List of design point lists for each group
            n: Optional limit on final number of points
            
        Returns:
            List[Dict[str, Any]]: Merged design points
        """
        if not group_designs:
            return []
        
        # Start with the first group's designs
        merged_designs = group_designs[0]
        
        # Merge with each additional group
        for designs in group_designs[1:]:
            if not merged_designs or not designs:
                continue
                
            # If designs are too many, sample from both groups
            if n is not None and len(merged_designs) * len(designs) > n:
                # Use Cartesian product with sampling
                n_samples = min(n, len(merged_designs) * len(designs))
                
                # Determine sample indices for Cartesian product
                all_indices = [(i, j) for i in range(len(merged_designs)) for j in range(len(designs))]
                selected_indices = random.sample(all_indices, n_samples)
                
                new_designs = []
                for i, j in selected_indices:
                    new_point = {**merged_designs[i], **designs[j]}
                    new_designs.append(new_point)
                
                merged_designs = new_designs
            else:
                # Use full Cartesian product
                new_designs = []
                for d1 in merged_designs:
                    for d2 in designs:
                        new_point = {**d1, **d2}
                        new_designs.append(new_point)
                
                merged_designs = new_designs
                
                # Limit if we now have too many designs
                if n is not None and len(merged_designs) > n:
                    merged_designs = random.sample(merged_designs, n)
        
        return merged_designs
    
    def generate(self, n: int = None) -> List[Dict[str, Any]]:
        """
        Generate design points using factorial design.
        For large parameter spaces, uses a layered approach to avoid combinatorial explosion.
        
        Args:
            n: Desired number of design points. If None, generates all combinations,
               but with layered approach for very large spaces.
            
        Returns:
            List[Dict[str, Any]]: List of design points
        """
        # For small parameter spaces, use traditional full factorial design
        if not self.layered_approach:
            return self._generate_traditional_factorial(n)
        
        # For large parameter spaces, use layered approach
        return self._generate_layered_factorial(n)
    
    def _generate_traditional_factorial(self, n: int = None) -> List[Dict[str, Any]]:
        """
        Generate design points using traditional full factorial design.
        
        Args:
            n: Number of design points to generate (optional)
            
        Returns:
            List[Dict[str, Any]]: List of design points
        """
        # Initialize with empty design
        design_points = [{}]
        
        # For each parameter, expand the design
        for param_name, param_config in self.parameter_space.parameters.items():
            # Determine number of levels for this parameter
            if param_config["type"] == "categorical":
                level_count = len(param_config["categories"])
            elif param_name in self.levels:
                level_count = self.levels[param_name]
            else:
                # Default levels - 5 for continuous, min(10, all values) for discrete
                if param_config["type"] == "integer":
                    min_val = param_config["min"]
                    max_val = param_config["max"]
                    level_count = min(10, max_val - min_val + 1)
                else:
                    level_count = 5
            
            # Get specific levels
            levels = self._get_parameter_levels(param_name, param_config, level_count)
            
            # Create new expanded design
            new_design_points = []
            for point in design_points:
                for level in levels:
                    new_point = point.copy()
                    new_point[param_name] = level
                    new_design_points.append(new_point)
            
            design_points = new_design_points
        
        # Filter out points that don't satisfy constraints
        if self.parameter_space.has_constraints():
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
    
    def _generate_layered_factorial(self, n: int = None) -> List[Dict[str, Any]]:
        """
        Generate design points using layered factorial design for large parameter spaces.
        
        Args:
            n: Number of design points to generate
            
        Returns:
            List[Dict[str, Any]]: List of design points
        """
        # If n is not specified, use a reasonable default
        if n is None:
            n = min(self.max_combinations, 100)
            logger.info(f"No sample size specified for large parameter space. Using default n={n}")
        
        # Calculate points to generate per group
        points_per_group = max(int(n ** (1/len(self.parameter_groups))), 2)
        logger.info(f"Using layered approach with {len(self.parameter_groups)} groups, "
                   f"generating ~{points_per_group} points per group")
        
        # Generate designs for each parameter group
        group_designs = []
        for group in self.parameter_groups:
            group_design = self._generate_for_group(group, points_per_group)
            group_designs.append(group_design)
        
        # Merge designs from all groups
        merged_designs = self._merge_design_groups(group_designs, n)
        
        # Filter out points that don't satisfy constraints
        if self.parameter_space.has_constraints():
            valid_points = []
            for point in merged_designs:
                if self.parameter_space.is_valid_point(point):
                    valid_points.append(point)
            
            # If we lost too many points due to constraints, generate more
            if len(valid_points) < n * 0.5 and len(valid_points) < len(merged_designs) * 0.5:
                logger.warning(f"Only {len(valid_points)} out of {len(merged_designs)} points "
                              f"satisfy constraints. Generating additional points.")
                
                # Try to generate more points with higher sampling rate
                additional_needed = n - len(valid_points)
                if additional_needed > 0:
                    # Use random sampling as fallback to meet the requested number
                    random_generator = RandomDesignGenerator(self.parameter_space)
                    random_points = random_generator.generate(additional_needed)
                    valid_points.extend(random_points)
            
            merged_designs = valid_points
        
        # If we still need to limit the number of points
        if n is not None and len(merged_designs) > n:
            merged_designs = random.sample(merged_designs, n)
        
        logger.info(f"Generated {len(merged_designs)} layered factorial design points")
        return merged_designs


class SobolDesignGenerator(DesignGenerator):
    """
    Generate design points using Sobol sequences.
    Requires scipy for optimal Sobol sequence generation.
    """
    def __init__(
        self, 
        parameter_space: ParameterSpace, 
        seed: Optional[int] = None,
        scramble: bool = True,
        bits: Optional[int] = None,
        optimization: Optional[str] = None
    ):
        """
        Initialize a Sobol sequence design generator.
        
        Args:
            parameter_space: Parameter space for generating designs
            seed: Random seed for reproducibility
            scramble: Whether to use scrambling (default: True). Scrambling makes the sequence
                     suitable for singular integrands and can improve convergence rate.
            bits: Number of bits for the Sobol sequence generator (max value 64). 
                  Controls maximum number of points (2**bits). Default (None) uses scipy's default.
            optimization: Optional optimization scheme to improve quality after sampling:
                         - "random-cd": Random permutations to lower centered discrepancy
                         - "lloyd": Perturb samples using modified Lloyd-Max algorithm
                         - None: No optimization
        """
        super().__init__(parameter_space)
        self.seed = seed
        self.scramble = scramble
        self.bits = bits
        self.optimization = optimization
        
        # Check if scipy is available
        try:
            from scipy.stats import qmc
            self.qmc = qmc
        except ImportError:
            raise ImportError("scipy is required for Sobol sequence generation")
        
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
        
        # Create Sobol generator
        try:
            # Use modern scipy.stats.qmc API with rng parameter
            sampler = self.qmc.Sobol(
                d=dim, 
                scramble=self.scramble, 
                bits=self.bits,
                rng=self.seed,
                optimization=self.optimization
            )
        except TypeError:
            # Fallback for older scipy versions using seed parameter
            try:
                sampler = self.qmc.Sobol(
                    d=dim, 
                    scramble=self.scramble,
                    seed=self.seed
                )
            except Exception as e:
                logger.error(f"Failed to create Sobol generator: {e}")
                raise
        
        # Determine how many points to generate 
        # (Sobol sequences work best with powers of 2)
        if n <= 0:
            return []
        
        # For best quality, use random_base2 when n is a power of 2
        is_power_of_two = (n & (n-1) == 0) and n != 0
        
        try:
            if is_power_of_two:
                # Calculate m such that 2^m = n
                m = n.bit_length() - 1
                logger.info(f"Using random_base2 with m={m} to generate {n} Sobol points")
                samples = sampler.random_base2(m=m)
            else:
                logger.info(f"Using random to generate {n} Sobol points")
                # For non-power-of-2 sizes, can still use regular random method
                samples = sampler.random(n=n)
        except Exception as e:
            logger.error(f"Error generating Sobol samples: {e}")
            raise
        
        # Convert samples to parameter space
        designs = []
        valid_count = 0
        
        for i in range(n):
            try:
                # Convert internal representation to point
                point = self.parameter_space.internal_to_point(samples[i])
                
                # Check if point satisfies constraints
                if not self.has_constraints or self.parameter_space.is_valid_point(point):
                    designs.append(point)
                    valid_count += 1
            except Exception as e:
                logger.warning(f"Error generating design point: {e}")
        
        # If we have constraints and couldn't generate enough designs, fill with smart sampling
        if valid_count < n:
            logger.warning(f"Sobol generated only {valid_count} valid points out of {n}. Using adaptive sampling to fill remainder.")
            
            # Calculate how many more points we need
            remaining = n - valid_count
            
            # Generate oversampled additional points to increase chances of finding valid ones
            # Use 3x oversampling as a heuristic
            oversampling_factor = 3
            additional_samples = sampler.random(n=remaining * oversampling_factor)
            
            additional_valid_points = []
            for i in range(len(additional_samples)):
                try:
                    point = self.parameter_space.internal_to_point(additional_samples[i])
                    if self.parameter_space.is_valid_point(point):
                        additional_valid_points.append(point)
                        if len(additional_valid_points) >= remaining:
                            break
                except Exception as e:
                    continue
            
            # If we still don't have enough valid points, use RandomDesignGenerator as fallback
            if len(additional_valid_points) < remaining:
                logger.warning(f"Still missing {remaining - len(additional_valid_points)} valid points. Using random sampling as fallback.")
                random_generator = RandomDesignGenerator(self.parameter_space)
                random_points = random_generator.generate(remaining - len(additional_valid_points))
                additional_valid_points.extend(random_points)
            
            # Add the additional valid points to our design
            designs.extend(additional_valid_points[:remaining])
        
        # Calculate space-filling quality metric if we have the scipy.spatial package
        if valid_count > 1:
            try:
                import numpy as np
                from scipy.spatial import distance
                points_array = np.array([list(point.values()) for point in designs])
                if points_array.shape[1] > 1:  # Only meaningful for multi-dimensional spaces
                    min_dist = np.min(distance.pdist(points_array))
                    mean_dist = np.mean(distance.pdist(points_array))
                    logger.info(f"Sobol design quality: min distance between points = {min_dist:.4f}, mean distance = {mean_dist:.4f}")
            except (ImportError, Exception) as e:
                pass  # Skip quality calculation if not available
        
        logger.info(f"Generated {len(designs)} Sobol design points")
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
            - For RANDOM: seed (optional)
            - For LATIN_HYPERCUBE: seed (optional), strength (default 1),
                                  optimization (default None, options: "correlation", "maximin", "centermaximin")
            - For FACTORIAL: levels (optional), max_combinations (default 10000),
                           adaptive_sampling (default False), parameter_groups (optional),
                           importance_based (default False)
            - For SOBOL: seed (optional), scramble (default True), bits (optional),
                        optimization (optional)
            - For CUSTOM: design_points (required)
    
    Returns:
        DesignGenerator: Design generator instance
    """
    if design_type == DesignType.RANDOM:
        seed = kwargs.get("seed")
        return RandomDesignGenerator(parameter_space, seed=seed)
    
    elif design_type == DesignType.LATIN_HYPERCUBE:
        seed = kwargs.get("seed")
        strength = kwargs.get("strength", 1)
        optimization = kwargs.get("optimization")
        return LatinHypercubeDesignGenerator(
            parameter_space, 
            seed=seed,
            strength=strength,
            optimization=optimization
        )
    
    elif design_type == DesignType.FACTORIAL:
        levels = kwargs.get("levels")
        max_combinations = kwargs.get("max_combinations", 10000)
        adaptive_sampling = kwargs.get("adaptive_sampling", False)
        parameter_groups = kwargs.get("parameter_groups")
        importance_based = kwargs.get("importance_based", False)
        return FactorialDesignGenerator(
            parameter_space, 
            levels=levels,
            max_combinations=max_combinations,
            adaptive_sampling=adaptive_sampling,
            parameter_groups=parameter_groups,
            importance_based=importance_based
        )
    
    elif design_type == DesignType.SOBOL:
        seed = kwargs.get("seed")
        scramble = kwargs.get("scramble", True)
        bits = kwargs.get("bits")
        optimization = kwargs.get("optimization")
        return SobolDesignGenerator(
            parameter_space, 
            seed=seed, 
            scramble=scramble,
            bits=bits,
            optimization=optimization
        )
    
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
