#!/usr/bin/env python
"""
高级设计生成器使用示例

此脚本展示了如何使用改进后的设计生成器API创建和生成各种类型的设计点。
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bo_engine.design_generator import (
    DesignType, 
    create_design_generator,
    RandomDesignGenerator,
    LatinHypercubeDesignGenerator,
    SobolDesignGenerator,
    FactorialDesignGenerator
)
from bo_engine.parameter_space import ParameterSpace, ConstraintRelation
from bo_engine.utils import evaluate_lhs_quality

# 设置随机种子以保证可重复性
np.random.seed(42)


def create_parameter_space():
    """创建示例参数空间"""
    # 创建参数定义
    parameters = {
        "x1": {
            "type": "continuous",
            "min": 0.0,
            "max": 10.0,
            "description": "First parameter"
        },
        "x2": {
            "type": "continuous",
            "min": 0.0,
            "max": 10.0,
            "description": "Second parameter"
        },
        "x3": {
            "type": "integer",
            "min": 1,
            "max": 5,
            "description": "Third parameter",
            "importance": 0.5
        },
        "x4": {
            "type": "categorical",
            "categories": ["red", "green", "blue"],
            "description": "Fourth parameter",
            "importance": 0.3
        }
    }
    
    # 创建约束条件
    constraints = [
        {
            "type": ConstraintRelation.LESS_THAN_OR_EQUAL,
            "expression": "x1 + x2 <= 15",
            "description": "Sum constraint"
        },
        {
            "type": ConstraintRelation.GREATER_THAN_OR_EQUAL,
            "expression": "x1 * x2 >= 5",
            "description": "Product constraint"
        }
    ]
    
    # 创建参数空间
    return ParameterSpace(
        parameters=parameters,
        constraints=constraints,
        name="Advanced Design Example",
        description="Example parameter space for demonstrating advanced design generators"
    )


def plot_2d_design(designs, title):
    """绘制2D设计点可视化"""
    plt.figure(figsize=(10, 8))
    
    # 提取x1和x2坐标
    x1 = [design["x1"] for design in designs]
    x2 = [design["x2"] for design in designs]
    
    # 根据x3的值确定点大小
    sizes = [20 + 5 * design.get("x3", 3) for design in designs]
    
    # 根据x4的值确定点颜色
    colors = []
    for design in designs:
        if design.get("x4") == "red":
            colors.append("red")
        elif design.get("x4") == "green":
            colors.append("green")
        elif design.get("x4") == "blue":
            colors.append("blue")
        else:
            colors.append("gray")
    
    # 绘制设计点
    plt.scatter(x1, x2, s=sizes, c=colors, alpha=0.7)
    
    # 绘制约束边界
    x = np.linspace(0, 10, 100)
    y1 = 15 - x  # x1 + x2 <= 15 的边界
    y2 = 5 / (x + 0.001)  # x1 * x2 >= 5 的边界
    plt.plot(x, y1, 'k--', label='x1 + x2 = 15')
    plt.plot(x, y2, 'k-.', label='x1 * x2 = 5')
    
    # 添加有效区域阴影
    xx, yy = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    condition1 = xx + yy <= 15
    condition2 = xx * yy >= 5
    valid_region = condition1 & condition2
    plt.contourf(xx, yy, valid_region.astype(int), levels=[-0.5, 0.5], alpha=0.1, colors=['red', 'green'])
    
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    return plt


def compare_designs():
    """比较不同设计生成器生成的设计点"""
    # 创建参数空间
    parameter_space = create_parameter_space()
    
    # 创建各种设计生成器
    generators = {
        "Random": create_design_generator(
            parameter_space=parameter_space,
            design_type=DesignType.RANDOM,
            seed=42
        ),
        "Latin Hypercube": create_design_generator(
            parameter_space=parameter_space,
            design_type=DesignType.LATIN_HYPERCUBE,
            seed=42,
            strength=2,
            optimization="random-cd"
        ),
        "Sobol": create_design_generator(
            parameter_space=parameter_space,
            design_type=DesignType.SOBOL,
            seed=42,
            scramble=True,
            optimization="random-cd"
        ),
        "Factorial (Layered)": create_design_generator(
            parameter_space=parameter_space,
            design_type=DesignType.FACTORIAL,
            max_combinations=100,
            importance_based=True
        )
    }
    
    # 生成设计点
    n_designs = 32
    designs = {}
    for name, generator in generators.items():
        print(f"生成 {n_designs} 个 {name} 设计点...")
        designs[name] = generator.generate(n_designs)
        print(f"  实际生成点数: {len(designs[name])}")
    
    # 可视化结果
    for name, design_points in designs.items():
        plt = plot_2d_design(design_points, f"{name} Design (n={len(design_points)})")
        plt.savefig(f"{name}_design.png")
        print(f"已保存 {name}_design.png")
    
    # 比较设计质量
    quality_metrics = {}
    for name, design_points in designs.items():
        # 创建2D点数组用于质量评估
        points_array = np.array([[d["x1"], d["x2"]] for d in design_points])
        quality = evaluate_lhs_quality(points_array)
        quality_metrics[name] = quality
    
    # 创建质量指标表格
    metrics_df = pd.DataFrame(quality_metrics).T
    print("\n设计质量指标比较:")
    print(metrics_df[["min_distance", "mean_distance", "cv", "min_mean_ratio"]])
    
    return designs, quality_metrics


if __name__ == "__main__":
    print("高级设计生成器示例\n")
    
    # 确保存在输出目录
    os.makedirs("output", exist_ok=True)
    
    # 切换到输出目录保存图片
    os.chdir("output")
    
    # 比较不同设计生成器
    designs, quality_metrics = compare_designs()
    
    print("\n示例完成，图片已保存到 output 目录。") 
