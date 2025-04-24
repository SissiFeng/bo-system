# 参数空间（Parameter Space）功能规则

## 📋 功能概述

参数空间模块是贝叶斯优化系统的基础组件，负责定义和管理实验中的各种参数类型、取值范围、约束和目标函数。该模块为其他组件（如设计生成器、优化算法）提供统一的参数处理接口。

## 🏗️ 设计思路

### 核心设计原则

1. **继承层次结构**：使用抽象基类`Parameter`定义通用接口，各种参数类型通过继承实现具体功能
2. **类型安全**：严格的类型检查和验证，确保参数值符合定义的约束
3. **灵活性**：支持多种参数类型，包括连续型、整数型、类别型等
4. **内外表示分离**：区分内部表示（用于优化算法）和外部表示（用于用户交互）

### 类层次结构

```
Parameter (抽象基类)
├── ContinuousParameter
├── IntegerParameter
├── CategoricalParameter
├── OrdinalParameter
└── CompositeParameter
```

### 类型枚举

```python
class ParameterType(str, Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer" 
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    COMPOSITE = "composite"

class ObjectiveType(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

class ConstraintType(str, Enum):
    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than"
    EQUAL_TO = "equal_to"
```

## 🔍 实现细节

### 基类：`Parameter`

抽象基类，定义所有参数类型共有的接口和属性：

```python
class Parameter(ABC):
    def __init__(self, name, parameter_type):
        self.name = name
        self.parameter_type = parameter_type
        
    @abstractmethod
    def validate(self, value):
        """验证参数值是否有效"""
        pass
        
    @abstractmethod
    def sample(self, n=1, seed=None):
        """从参数空间采样"""
        pass
        
    @abstractmethod
    def to_internal(self, value):
        """转换为内部表示"""
        pass
        
    @abstractmethod
    def to_external(self, internal_value):
        """转换为外部表示"""
        pass
```

### 具体实现

#### 1. 连续参数（ContinuousParameter）

- 功能：表示在指定区间上的连续实数值
- 属性：名称、下界、上界、可选变换函数（如对数变换）
- 内部表示：[0,1]区间上的归一化值
- 采样方法：均匀分布、正态分布等

#### 2. 整数参数（IntegerParameter）

- 功能：表示在指定区间上的整数值
- 属性：名称、下界、上界
- 内部表示：[0,1]区间上的归一化值
- 采样方法：均匀整数采样

#### 3. 类别参数（CategoricalParameter）

- 功能：表示离散的、无序的选项集合
- 属性：名称、可能值列表
- 内部表示：独热编码或嵌入表示
- 采样方法：均匀随机选择

#### 4. 有序参数（OrdinalParameter）

- 功能：表示有序的离散选项集合
- 属性：名称、有序可能值列表
- 内部表示：整数索引的归一化值
- 采样方法：均匀随机选择

#### 5. 复合参数（CompositeParameter）

- 功能：组合多个相关参数，实现条件参数和层级结构
- 属性：名称、子参数集合、条件逻辑
- 内部表示：子参数内部表示的组合
- 采样方法：基于条件逻辑的分层采样

### 参数空间

```python
class ParameterSpace:
    def __init__(self, parameters, objectives=None, constraints=None):
        self.parameters = parameters
        self.objectives = objectives or []
        self.constraints = constraints or []
        
    def validate_point(self, point):
        """验证一个设计点是否有效"""
        pass
        
    def sample(self, n=1, seed=None):
        """采样n个有效的设计点"""
        pass
        
    def to_internal(self, external_point):
        """将外部表示转换为内部表示"""
        pass
        
    def to_external(self, internal_point):
        """将内部表示转换为外部表示"""
        pass
```

### 目标函数和约束

```python
class Objective:
    def __init__(self, name, objective_type=ObjectiveType.MINIMIZE):
        self.name = name
        self.objective_type = objective_type

class Constraint:
    def __init__(self, expression, constraint_type, threshold):
        self.expression = expression
        self.constraint_type = constraint_type
        self.threshold = threshold
        
    def evaluate(self, point):
        """评估约束条件"""
        pass
```

## 🔄 数据流

1. 用户定义参数、目标函数和约束条件
2. 系统创建参数空间对象
3. 参数空间提供接口用于：
   - 验证设计点的有效性
   - 采样有效的设计点
   - 转换设计点的内部和外部表示
4. 优化算法和设计生成器通过这些接口与参数空间交互

## 📊 使用示例

```python
# 创建参数
params = [
    ContinuousParameter("x1", 0.0, 10.0),
    ContinuousParameter("x2", -5.0, 5.0, log_scale=True),
    IntegerParameter("x3", 1, 100),
    CategoricalParameter("x4", ["red", "green", "blue"])
]

# 创建目标函数
objectives = [
    Objective("y1", ObjectiveType.MINIMIZE),
    Objective("y2", ObjectiveType.MAXIMIZE)
]

# 创建约束
constraints = [
    Constraint("x1 + x2", ConstraintType.LESS_THAN, 5.0)
]

# 创建参数空间
space = ParameterSpace(params, objectives, constraints)

# 采样设计点
samples = space.sample(10, seed=42)
```

## ⚠️ 约束与限制

1. 复合参数和层级结构增加了采样和优化的复杂性
2. 类别参数的内部表示可能导致优化算法效率下降
3. 非线性约束可能导致可行域形状复杂，影响采样效率
4. 高维参数空间可能存在维度灾难问题

## 🔄 与其他模块的交互

- **输出**：为`DesignGenerator`模块提供参数定义和采样功能
- **输出**：为`BOSystem`模块提供参数处理接口
- **依赖**：依赖`utils.py`中的辅助函数

## 🔮 未来扩展

1. 支持更多参数类型（周期性参数、概率分布参数等）
2. 增强条件参数和依赖关系的支持
3. 改进高维空间中的采样效率
4. 添加参数重要性分析功能
5. 支持更复杂的约束表达式
