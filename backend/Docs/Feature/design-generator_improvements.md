# Design Generator Improvements

## 概述

本文档总结了对设计生成器模块的改进，包括增强Sobol序列生成、优化FactorialDesignGenerator以处理大型参数空间，以及添加智能约束处理机制。

## 主要改进

### 1. Sobol序列生成器增强

改进了`SobolDesignGenerator`类以更好地利用scipy的Sobol序列实现：

- 添加了对scipy.stats.qmc.Sobol的完整支持，包括所有配置参数
- 增加了对scrambling（扰动）参数的支持，使用户可以控制序列的随机性
- 添加了bits参数，控制生成器可生成的最大点数（2^bits）
- 添加了对optimization参数的支持，允许使用后处理优化方法
- 添加对2的幂次样本大小的特殊处理，通过random_base2方法生成更高质量的点
- 改进了采样点质量评估，添加了空间分布指标的计算和日志输出

### 2. 全因子设计生成器优化

增强了`FactorialDesignGenerator`以更好地处理大型参数空间：

- 添加了分层处理方法，使全因子设计可以处理大型参数空间
- 实现了自动参数分组算法，根据参数空间大小自动划分参数组
- 添加了最大组合数限制，超过此限制时自动使用分层方法
- 实现了基于参数重要性的分组策略，优先考虑重要参数
- 优化了索引采样，在大型空间中高效生成点
- 改进了约束处理，使生成的设计点尽可能满足约束条件

### 3. 智能约束处理机制

添加了`ConstraintHandler`类处理约束条件：

- 实现了约束分析功能，评估可行区域占参数空间的比例
- 添加了自适应采样策略，根据约束密度动态调整过采样因子
- 实现了空间填充采样方法，在满足约束的情况下最大化点之间的距离
- 添加了约束情况记录功能，统计各约束的违反频率
- 提供约束处理器API供设计生成器使用

### 4. 其他优化

- 改进了拉丁超立方抽样，添加了对scipy.stats.qmc.LatinHypercube的支持
- 优化了RandomDesignGenerator的实现，提升了随机采样效率
- 添加了设计质量评估功能，计算多种空间填充指标
- 改进了create_design_generator工厂函数，支持所有新增参数
- 增强了异常处理，提供更详细的错误信息和恢复策略

## 测试

添加了全面的单元测试，覆盖了：

- 所有设计生成器类和约束处理器的基本功能
- 各种边界条件和特殊情况的处理
- 约束条件检查和组合约束的处理
- 各种随机种子和配置选项的行为
- API兼容性验证

## 使用示例

### Sobol序列生成

```python
from bo_engine.design_generator import create_design_generator, DesignType

# 创建一个具有高级选项的Sobol设计生成器
generator = create_design_generator(
    parameter_space=parameter_space,
    design_type=DesignType.SOBOL,
    seed=42,
    scramble=True,
    bits=30,
    optimization="random-cd"
)

# 生成16个点（2的幂次，最佳质量）
designs = generator.generate(16)
```

### 分层全因子设计

```python
# 创建一个使用分层策略的全因子设计生成器
generator = create_design_generator(
    parameter_space=parameter_space,
    design_type=DesignType.FACTORIAL,
    levels={"x1": 3, "x2": 3, "x3": 2},
    max_combinations=1000,
    adaptive_sampling=True,
    importance_based=True
)

# 生成设计点
designs = generator.generate(50)
```

### 拉丁超立方设计

```python
# 创建一个优化的拉丁超立方设计生成器
generator = create_design_generator(
    parameter_space=parameter_space,
    design_type=DesignType.LATIN_HYPERCUBE,
    seed=42,
    strength=2,
    optimization="random-cd"
)

# 生成设计点
designs = generator.generate(15)
```

## 后续计划

1. 考虑添加更多类型的设计生成器，如Halton序列和CVT (Centroidal Voronoi Tessellation)
2. 探索多保真度设计生成方法，支持不同精度级别的模拟
3. 实现自适应设计生成策略，根据先前结果智能选择下一批设计点
4. 优化大维度空间（>100维）的处理能力 
