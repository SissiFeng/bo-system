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
    """设计生成器模块的单元测试"""

    def setUp(self):
        """设置测试所需的基本参数空间和输出目录"""
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
        # 使用临时目录进行测试
        self.test_dir = TemporaryDirectory()
        self.output_dir = os.path.join(self.test_dir.name, "test_designs")

    def tearDown(self):
        """清理测试生成的临时目录"""
        self.test_dir.cleanup()

    def test_lhs_generator_init(self):
        """测试拉丁超立方生成器的初始化"""
        # 给ParameterSpace添加mock validate方法
        self.param_space.validate = lambda: (True, "")
        
        # 检查基本DesignGenerator初始化
        generator = BasicDesignGenerator(self.param_space, seed=42, output_dir=self.output_dir)
        self.assertEqual(generator.parameter_space, self.param_space)
        
        # 使用from_list方法创建LatinHypercubeDesignGenerator
        # 由于无法直接实例化LatinHypercubeDesignGenerator，可以使用create_design_generator
        lhs_gen = create_design_generator(self.param_space, DesignType.LATIN_HYPERCUBE, seed=42)
        self.assertIsNotNone(lhs_gen)
        # 基本检查
        self.assertTrue(hasattr(lhs_gen, 'parameter_space'))

    def test_lhs_generator_generate(self):
        """测试拉丁超立方生成器生成设计点"""
        n_designs = 10
        # BasicDesignGenerator 现在管理所有方法
        generator = BasicDesignGenerator(
            self.param_space,
            seed=42,
            output_dir=self.output_dir
        )

        designs = generator.generate_lhs_designs(n_designs=n_designs, ensure_constraints=True)

        self.assertEqual(len(designs), n_designs)

        # 检查每个设计点的结构和有效性
        for design in designs:
            self.assertIsInstance(design, dict)
            self.assertIn("id", design)  # 检查是否有 ID

            # 检查参数是否存在且值有效
            self.assertIn("x1", design)
            self.assertIsInstance(design["x1"], float)
            self.assertTrue(0.0 <= design["x1"] <= 10.0)

            self.assertIn("x2", design)
            self.assertIsInstance(design["x2"], int)
            self.assertTrue(1 <= design["x2"] <= 5)

            self.assertIn("x3", design)
            self.assertIsInstance(design["x3"], str)
            self.assertIn(design["x3"], ["A", "B", "C"])

            # 检查文件是否保存
            design_file = os.path.join(self.output_dir, f"{design['id']}.json")
            self.assertTrue(os.path.exists(design_file))

    def test_lhs_generator_with_constraints(self):
        """测试带约束的拉丁超立方生成器"""
        constraints = [
            {
                "type": "linear",
                "expression": "x1 + x2 <= 8"
            }
        ]
        constrained_param_space = ParameterSpace(self.parameters, constraints)

        n_designs = 20 # 请求更多点以增加找到有效点的机会
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
