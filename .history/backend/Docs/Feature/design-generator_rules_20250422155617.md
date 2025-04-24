# 设计生成器（Design Generator）功能规则

## 📋 功能概述

设计生成器模块负责在参数空间中生成初始实验设计点。该模块支持多种常用的实验设计方法，包括随机采样、拉丁超立方体设计、全因子设计、Sobol序列设计以及自定义设计。

## 🏗️ 设计思路

### 核心设计原则

1. **策略模式**：通过抽象基类`DesignGenerator`定义统一接口，允许不同实现策略可互换
2. **工厂模式**：使用`create_design_generator`函数根据设计类型实例化合适的设计生成器
3. **扩展性**：易于添加新的设计生成策略，只需扩展基类并实现必要方法
4. **参数空间一致性**：所有生成的设计点都经过验证，确保符合参数空间约束

### 类层次结构

```
DesignGenerator (抽象基类)
├── RandomDesignGenerator
├── LatinHypercubeDesignGenerator
├── FactorialDesignGenerator 
├── SobolDesignGenerator
└── CustomDesignGenerator
```

### 设计类型枚举

```python
class DesignType(str, Enum):
    RANDOM = "random"
    LHS = "lhs"
    FACTORIAL = "factorial"
    SOBOL = "sobol"
    CUSTOM = "custom"
```

## 🔍 实现细节

### 基类：`DesignGenerator`

抽象基类，定义所有设计生成器必须实现的接口：

```python
class DesignGenerator(ABC):
    def __init__(self, parameter_space):
        self.parameter_space = parameter_space
        
    @abstractmethod
    def generate(self, n_samples, seed=None):
        """生成设计点"""
        pass
```

### 具体实现

#### 1. 随机设计生成器（RandomDesignGenerator）

- 功能：从参数空间中随机均匀采样
- 实现：利用参数空间各参数的采样方法随机生成符合约束的点
- 适用场景：探索性研究、基准测试、小规模实验

#### 2. 拉丁超立方体设计生成器（LatinHypercubeDesignGenerator）

- 功能：生成拉丁超立方体设计，确保参数空间均匀覆盖
- 实现：使用`pyDOE`库生成拉丁超立方体，然后转换到实际参数范围
- 特性：支持多种优化准则（maximin, correlation等）
- 适用场景：中等维度参数空间的均匀探索

#### 3. 因子设计生成器（FactorialDesignGenerator）

- 功能：生成网格化的全因子或部分因子设计
- 实现：对每个参数取一定数量的均匀点，然后生成所有可能组合
- 特性：对于k个因子，每个因子取n个水平，生成n^k个点
- 适用场景：低维参数空间的详细探索，交互效应分析

#### 4. Sobol序列设计生成器（SobolDesignGenerator）

- 功能：生成Sobol准随机序列，具有低偏差特性
- 实现：使用`scipy`库的`qmc.Sobol`生成器
- 特性：比纯随机采样有更好的覆盖性，适合高维空间
- 适用场景：高维参数空间的有效探索

#### 5. 自定义设计生成器（CustomDesignGenerator）

- 功能：支持用户提供预定义的设计点
- 实现：验证提供的设计点是否符合参数空间约束
- 适用场景：专家知识引导、已有实验数据复用、特定实验设计

### 工厂函数

```python
def create_design_generator(parameter_space, design_type, **kwargs):
    """创建适当类型的设计生成器"""
    if design_type == DesignType.RANDOM:
        return RandomDesignGenerator(parameter_space)
    elif design_type == DesignType.LHS:
        return LatinHypercubeDesignGenerator(parameter_space, **kwargs)
    # ... 其他类型
```

## 🔄 数据流

1. 用户指定参数空间和设计类型
2. 系统创建相应的设计生成器
3. 设计生成器在参数空间中生成设计点
4. 设计点经过验证确保符合所有约束
5. 设计点可以保存为CSV/JSON格式或直接用于后续实验

## 📊 使用示例

```python
# 创建参数空间
params = [
    ContinuousParameter("x1", 0.0, 10.0),
    ContinuousParameter("x2", -5.0, 5.0),
    CategoricalParameter("x3", ["A", "B", "C"])
]
parameter_space = ParameterSpace(params)

# 创建LHS设计生成器
generator = create_design_generator(parameter_space, DesignType.LHS)

# 生成10个设计点
design = generator.generate(10, seed=42)

# 保存设计
design.save("my_design.csv")
```

## ⚠️ 约束与限制

1. 类别参数在拉丁超立方体设计中通过均匀分布处理
2. 高维空间（>20维）可能需要较大的样本量才能有效覆盖
3. 全因子设计在参数数量增加时会呈指数级增长
4. 自定义设计必须符合参数空间的所有约束

## 🔄 与其他模块的交互

- **输入**：来自`ParameterSpace`模块的参数空间定义
- **输出**：设计点供`BOSystem`和评估模块使用
- **依赖**：依赖`utils.py`中的辅助函数和`parameter_space.py`中的参数定义

## 🔮 未来扩展

1. 添加更多实验设计方法（Halton序列、正交阵等）
2. 支持约束空间中的设计生成
3. 增加自适应设计生成策略
4. 添加对分布式并行实验的支持 
