import unittest
import numpy as np
import os
import json
from tempfile import TemporaryDirectory
import shutil  # for cleanup   

from backend.bo_engine.parameter_space import ParameterSpace
from backend.bo_engine.design_generator import DesignGenerator, BasicDesignGenerator, DesignType, LatinHypercubeDesignGenerator
from backend.bo_engine.design_generator import create_design_generator

class TestDesignGenerator(unittest.TestCase):
    """Unit tests for the design generator module"""

    def setUp(self):
        """Set up the basic parameter space and output directory for testing"""
        self.parameters = {
            "x1": {
                "type": "continuous",
                "min": 0.0,
                "max": 10.0
            },
            "x2": {
                "type": "integer",
                "min": 1,
                "max": 5
            },
            "x3": {
                "type": "categorical",
                "categories": ["A", "B", "C"]
            }
        }
        self.param_space = ParameterSpace(self.parameters)
        # Use a temporary directory for testing
        self.test_dir = TemporaryDirectory()
        self.output_dir = os.path.join(self.test_dir.name, "test_designs")

    def tearDown(self):
        """Clean up the temporary directory for testing"""
        self.test_dir.cleanup()

    def test_lhs_generator_init(self):
        """Test the initialization of the Latin Hypercube design generator"""
        # Add a mock validate method to ParameterSpace
        self.param_space.validate = lambda: (True, "")
        
        # Check the basic DesignGenerator initialization
        generator = BasicDesignGenerator(self.param_space, seed=42, output_dir=self.output_dir)
        self.assertEqual(generator.parameter_space, self.param_space)
        
        # Create a LatinHypercubeDesignGenerator using the from_list method
        # Since LatinHypercubeDesignGenerator cannot be directly instantiated, use create_design_generator
        lhs_gen = create_design_generator(self.param_space, DesignType.LATIN_HYPERCUBE, seed=42)
        self.assertIsNotNone(lhs_gen)
        # Basic checks
        self.assertTrue(hasattr(lhs_gen, 'parameter_space'))

    def test_lhs_generator_generate(self):
        """Test the Latin Hypercube design generator to generate design points"""
        n_designs = 10
        # BasicDesignGenerator now manages all methods
        generator = BasicDesignGenerator(
            self.param_space,
            seed=42,
            output_dir=self.output_dir
        )

        designs = generator.generate_lhs_designs(n_designs=n_designs, ensure_constraints=True)

        self.assertEqual(len(designs), n_designs)

        # Check the structure and validity of each design point
        for design in designs:
            self.assertIsInstance(design, dict)
            self.assertIn("id", design)  # Check if there is an ID

            # Check if the parameters exist and are valid
            self.assertIn("x1", design)
            self.assertIsInstance(design["x1"], float)
            self.assertTrue(0.0 <= design["x1"] <= 10.0)

            self.assertIn("x2", design)
            self.assertIsInstance(design["x2"], int)
            self.assertTrue(1 <= design["x2"] <= 5)

            self.assertIn("x3", design)
            self.assertIsInstance(design["x3"], str)
            self.assertIn(design["x3"], ["A", "B", "C"])

            # Check if the file is saved
            design_file = os.path.join(self.output_dir, f"{design['id']}.json")
            self.assertTrue(os.path.exists(design_file))

    def test_lhs_generator_with_constraints(self):
        """Test the Latin Hypercube design generator with constraints"""
        constraints = [
            {
                "type": "linear",
                "expression": "x1 + x2 <= 8"
            }
        ]
        constrained_param_space = ParameterSpace(self.parameters, constraints)

        n_designs = 20 # Request more points to increase the chance of finding valid points
        generator = BasicDesignGenerator(
            constrained_param_space,
            seed=43,
            output_dir=self.output_dir
        )

        designs = generator.generate_lhs_designs(n_designs=n_designs, ensure_constraints=True)

        # 可能无法生成所有点，但至少应该生成一些
        self.assertTrue(len(designs) > 0, f"Expected > 0 designs, but got {len(designs)}")
        self.assertTrue(len(designs) <= n_designs)

        # 验证生成的设计点满足约束条件
        for design in designs:
            # 手动检查约束，因为 check_constraints 可能需要 design 包含 id
            self.assertTrue(constrained_param_space.check_constraints(design))
            self.assertTrue(design["x1"] + design["x2"] <= 8)

    # TODO: 添加对其他 DesignGenerator 方法的测试
    # 例如 generate_random_designs, generate_grid_designs 等


if __name__ == "__main__":
    unittest.main()
