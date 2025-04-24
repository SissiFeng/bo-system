import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from enum import Enum
import random
from abc import ABC, abstractmethod

from bo_engine.parameter_space import ParameterSpace
from bo_engine.utils import latin_hypercube_sampling

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
