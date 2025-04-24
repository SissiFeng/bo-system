import os
import json
import unittest
import numpy as np
from tempfile import TemporaryDirectory

from bo_engine.parameter_space import ParameterSpace


class TestParameterSpace(unittest.TestCase):
    """参数空间类的单元测试"""

    def test_init_with_valid_parameters(self):
        """测试使用有效参数初始化"""
        parameters = {
            "x1": {
                "type": "continuous",
                "min": 0.0,
                "max": 10.0,
                "units": "mm",
                "description": "长度"
            },
            "x2": {
                "type": "integer",
                "min": 1,
                "max": 5,
                "description": "等级"
            },
            "x3": {
                "type": "categorical",
                "categories": ["红", "绿", "蓝"],
                "description": "颜色"
            }
        }

        constraints = [
            {
                "type": "linear",
                "expression": "x1 + x2 <= 12",
                "description": "总和不超过12"
            }
        ]

        param_space = ParameterSpace(parameters, constraints)
        
        self.assertEqual(len(param_space.get_parameter_names()), 3)
        self.assertEqual(param_space.get_parameter_names(), ["x1", "x2", "x3"])
        self.assertTrue(param_space.has_constraints())

    def test_init_with_invalid_parameters(self):
        """测试使用无效参数初始化时的异常"""
        # 缺少类型
        parameters1 = {
            "x1": {
                "min": 0.0,
                "max": 10.0
            }
        }
        
        with self.assertRaises(ValueError):
            ParameterSpace(parameters1)
        
        # 无效类型
        parameters2 = {
            "x1": {
                "type": "unknown",
                "min": 0.0,
                "max": 10.0
            }
        }
        
        with self.assertRaises(ValueError):
            ParameterSpace(parameters2)
        
        # 连续参数缺少最小值
        parameters3 = {
            "x1": {
                "type": "continuous",
                "max": 10.0
            }
        }
        
        with self.assertRaises(ValueError):
            ParameterSpace(parameters3)
        
        # 最小值大于最大值
        parameters4 = {
            "x1": {
                "type": "continuous",
                "min": 10.0,
                "max": 5.0
            }
        }
        
        with self.assertRaises(ValueError):
            ParameterSpace(parameters4)
        
        # 分类参数缺少类别
        parameters5 = {
            "x1": {
                "type": "categorical"
            }
        }
        
        with self.assertRaises(ValueError):
            ParameterSpace(parameters5)
        
        # 分类参数类别为空
        parameters6 = {
            "x1": {
                "type": "categorical",
                "categories": []
            }
        }
        
        with self.assertRaises(ValueError):
            ParameterSpace(parameters6)
        
        # 分类参数类别重复
        parameters7 = {
            "x1": {
                "type": "categorical",
                "categories": ["a", "a", "b"]
            }
        }
        
        with self.assertRaises(ValueError):
            ParameterSpace(parameters7)

    def test_init_with_invalid_constraints(self):
        """测试使用无效约束初始化时的异常"""
        parameters = {
            "x1": {
                "type": "continuous",
                "min": 0.0,
                "max": 10.0
            }
        }
        
        # 缺少类型
        constraints1 = [
            {
                "expression": "x1 <= 5"
            }
        ]
        
        with self.assertRaises(ValueError):
            ParameterSpace(parameters, constraints1)
        
        # 无效类型
        constraints2 = [
            {
                "type": "unknown",
                "expression": "x1 <= 5"
            }
        ]
        
        with self.assertRaises(ValueError):
            ParameterSpace(parameters, constraints2)
        
        # 缺少表达式
        constraints3 = [
            {
                "type": "linear"
            }
        ]
        
        with self.assertRaises(ValueError):
            ParameterSpace(parameters, constraints3)

    def test_parameter_access_methods(self):
        """测试参数访问方法"""
        parameters = {
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
                "categories": ["红", "绿", "蓝"]
            }
        }
        
        param_space = ParameterSpace(parameters)
        
        # 测试获取参数名称
        self.assertEqual(set(param_space.get_parameter_names()), {"x1", "x2", "x3"})
        
        # 测试获取参数类型
        param_types = param_space.get_parameter_types()
        self.assertEqual(param_types["x1"], "continuous")
        self.assertEqual(param_types["x2"], "integer")
        self.assertEqual(param_types["x3"], "categorical")
        
        # 测试获取参数范围
        param_ranges = param_space.get_parameter_ranges()
        self.assertEqual(param_ranges["x1"], (0.0, 10.0))
        self.assertEqual(param_ranges["x2"], (1, 5))
        self.assertNotIn("x3", param_ranges)
        
        # 测试获取参数类别
        param_categories = param_space.get_parameter_categories()
        self.assertEqual(param_categories["x3"], ["红", "绿", "蓝"])
        self.assertNotIn("x1", param_categories)
        self.assertNotIn("x2", param_categories)

    def test_constraint_checking(self):
        """测试约束条件检查"""
        parameters = {
            "x1": {
                "type": "continuous",
                "min": 0.0,
                "max": 10.0
            },
            "x2": {
                "type": "integer",
                "min": 1,
                "max": 5
            }
        }
        
        # 线性约束
        constraints1 = [
            {
                "type": "linear",
                "expression": "x1 + x2 <= 12"
            }
        ]
        
        param_space1 = ParameterSpace(parameters, constraints1)
        
        # 满足约束的设计
        design1 = {"x1": 5.0, "x2": 3}
        self.assertTrue(param_space1.check_constraints(design1))
        
        # 不满足约束的设计
        design2 = {"x1": 9.0, "x2": 5}
        self.assertFalse(param_space1.check_constraints(design2))
        
        # 非线性约束
        constraints2 = [
            {
                "type": "nonlinear",
                "expression": "x1**2 + x2 <= 30"
            }
        ]
        
        param_space2 = ParameterSpace(parameters, constraints2)
        
        # 满足约束的设计
        design3 = {"x1": 5.0, "x2": 3}
        self.assertTrue(param_space2.check_constraints(design3))
        
        # 不满足约束的设计
        design4 = {"x1": 6.0, "x2": 5}
        self.assertFalse(param_space2.check_constraints(design4))
        
        # 参数范围检查
        design5 = {"x1": 11.0, "x2": 3}  # x1超出范围
        self.assertFalse(param_space2.check_constraints(design5))
        
        design6 = {"x1": 5.0, "x2": 6}  # x2超出范围
        self.assertFalse(param_space2.check_constraints(design6))
        
        # 缺少参数
        design7 = {"x1": 5.0}  # 缺少x2
        self.assertFalse(param_space2.check_constraints(design7))

    def test_random_sampling(self):
        """测试随机采样"""
        parameters = {
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
                "categories": ["红", "绿", "蓝"]
            }
        }
        
        param_space = ParameterSpace(parameters)
        
        # 测试采样
        n_samples = 10
        rng = np.random.RandomState(42)
        samples = param_space.sample_random(n_samples, rng)
        
        self.assertEqual(len(samples), n_samples)
        
        # 检查样本是否符合参数范围
        for sample in samples:
            self.assertTrue(0.0 <= sample["x1"] <= 10.0)
            self.assertTrue(1 <= sample["x2"] <= 5)
            self.assertTrue(sample["x3"] in ["红", "绿", "蓝"])
            
            # 检查类型
            self.assertIsInstance(sample["x1"], float)
            self.assertIsInstance(sample["x2"], int)

    def test_transform_and_inverse_transform(self):
        """测试向量转换和逆转换"""
        parameters = {
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
                "categories": ["红", "绿", "蓝"]
            }
        }
        
        param_space = ParameterSpace(parameters)
        
        # 测试转换
        design = {"x1": 5.0, "x2": 3, "x3": "蓝"}
        vec = param_space.transform(design)
        
        # 连续参数标准化到[0, 1]
        self.assertEqual(vec[0], 0.5)  # (5.0 - 0.0) / (10.0 - 0.0) = 0.5
        # 整数参数标准化到[0, 1]
        self.assertEqual(vec[1], 0.5)  # (3 - 1) / (5 - 1) = 0.5
        # 分类参数使用one-hot编码
        self.assertEqual(vec[2], 0.0)  # "红"
        self.assertEqual(vec[3], 0.0)  # "绿"
        self.assertEqual(vec[4], 1.0)  # "蓝"
        
        # 测试逆转换
        design2 = param_space.inverse_transform(vec)
        
        self.assertEqual(design2["x1"], 5.0)
        self.assertEqual(design2["x2"], 3)
        self.assertEqual(design2["x3"], "蓝")

    def test_dimensions(self):
        """测试参数空间维度"""
        parameters = {
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
                "categories": ["红", "绿", "蓝"]
            },
            "x4": {
                "type": "categorical",
                "categories": ["单一"]
            }
        }
        
        param_space = ParameterSpace(parameters)
        
        # 计算维度：1(连续) + 1(整数) + 3(分类，3个类别) + 0(分类，1个类别) = 5
        self.assertEqual(param_space.get_dimensions(), 5)

    def test_serialization(self):
        """测试序列化和反序列化"""
        parameters = {
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
                "categories": ["红", "绿", "蓝"]
            }
        }
        
        constraints = [
            {
                "type": "linear",
                "expression": "x1 + x2 <= 12"
            }
        ]
        
        param_space = ParameterSpace(parameters, constraints)
        
        # 测试转换为字典
        data = param_space.to_dict()
        self.assertIn("parameters", data)
        self.assertIn("constraints", data)
        
        # 测试从字典创建
        param_space2 = ParameterSpace.from_dict(data)
        self.assertEqual(param_space2.get_parameter_names(), param_space.get_parameter_names())
        self.assertEqual(param_space2.get_constraints(), param_space.get_constraints())
        
        # 测试保存和加载
        with TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "param_space.json")
            
            # 保存
            param_space.save(filepath)
            self.assertTrue(os.path.exists(filepath))
            
            # 加载
            param_space3 = ParameterSpace.load(filepath)
            self.assertEqual(param_space3.get_parameter_names(), param_space.get_parameter_names())
            self.assertEqual(param_space3.get_constraints(), param_space.get_constraints())


if __name__ == "__main__":
    unittest.main() 
