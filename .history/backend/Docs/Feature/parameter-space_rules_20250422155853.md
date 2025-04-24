# 参数空间 (ParameterSpace) 设计与实现规则

## 模块概述

参数空间模块是贝叶斯优化系统的基础组件，负责定义和管理优化过程中的参数、目标和约束条件。该模块提供了灵活的参数类型定义，支持连续、整数和类别参数，同时实现了参数验证、采样和转换等核心功能。参数空间为优化算法提供了统一的接口，确保参数配置的一致性和正确性。

## 核心设计原则

1. **类型安全**：
   - 使用枚举类型明确定义参数、目标和约束的类型
   - 强类型检查确保参数定义和操作的一致性

2. **表示转换**：
   - 提供内部表示（用于优化算法）和外部表示（用于用户接口）之间的无缝转换
   - 支持不同参数类型的特定转换规则

3. **验证机制**：
   - 全面的参数边界和类型验证
   - 参数间依赖关系和约束条件验证

4. **采样策略**：
   - 针对不同参数类型的优化采样策略
   - 支持考虑参数间依赖关系的条件采样

## 类层次与结构

### 枚举类型

```python
class ParameterType(Enum):
    """参数类型枚举"""
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"

class ObjectiveType(Enum):
    """目标函数类型枚举"""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

class ConstraintType(Enum):
    """约束条件类型枚举"""
    EQUAL = "equal"
    LESS_EQUAL = "less_equal"
    GREATER_EQUAL = "greater_equal"
```

### 基类定义

```python
class Parameter(ABC):
    """参数抽象基类，定义所有参数类型共有的接口"""
    
    def __init__(self, name, parameter_type):
        self.name = name
        self.parameter_type = parameter_type
        
    @abstractmethod
    def validate(self, value):
        """验证参数值是否有效"""
        pass
        
    @abstractmethod
    def sample(self, num_samples=1, random_state=None):
        """从参数空间采样"""
        pass
        
    @abstractmethod
    def to_internal(self, value):
        """将外部表示转换为内部表示"""
        pass
        
    @abstractmethod
    def to_external(self, value):
        """将内部表示转换为外部表示"""
        pass
```

### 具体参数类型

```python
class ContinuousParameter(Parameter):
    """连续参数类，表示浮点数范围"""
    
    def __init__(self, name, lower_bound, upper_bound, log_scale=False):
        super().__init__(name, ParameterType.CONTINUOUS)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.log_scale = log_scale
```

```python
class IntegerParameter(Parameter):
    """整数参数类，表示整数范围"""
    
    def __init__(self, name, lower_bound, upper_bound, step=1):
        super().__init__(name, ParameterType.INTEGER)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.step = step
```

```python
class CategoricalParameter(Parameter):
    """类别参数类，表示离散选项集合"""
    
    def __init__(self, name, categories):
        super().__init__(name, ParameterType.CATEGORICAL)
        self.categories = categories
```

### 参数空间主类

```python
class ParameterSpace:
    """参数空间类，管理参数、目标和约束的集合"""
    
    def __init__(self):
        self.parameters = {}
        self.objectives = {}
        self.constraints = {}
```

## 关键方法实现

### 参数管理

```python
def add_parameter(self, parameter):
    """添加参数到参数空间"""
    # 验证参数的有效性
    # 确保参数名称唯一
    # 将参数添加到参数字典
```

### 目标管理

```python
def add_objective(self, name, objective_type=ObjectiveType.MINIMIZE):
    """添加优化目标到参数空间"""
    # 验证目标名称和类型
    # 将目标添加到目标字典
```

### 约束管理

```python
def add_constraint(self, name, constraint_type, value):
    """添加约束条件到参数空间"""
    # 验证约束名称、类型和值
    # 将约束添加到约束字典
```

### 参数采样

```python
def sample(self, num_samples=1, random_state=None):
    """从参数空间采样生成设计点"""
    # 为每个参数生成随机样本
    # 组合成完整的设计点
    # 返回内部表示的样本
```

### 表示转换

```python
def dict_to_internal(self, design_dict):
    """将外部表示的设计点字典转换为内部表示"""
    # 遍历每个参数
    # 调用参数的to_internal方法
    # 返回转换后的内部表示
```

```python
def internal_to_dict(self, design_internal):
    """将内部表示的设计点转换为外部表示字典"""
    # 遍历每个参数
    # 调用参数的to_external方法
    # 返回转换后的外部表示字典
```

### 设计点验证

```python
def validate_design(self, design_dict):
    """验证设计点的有效性"""
    # 检查所有必需参数是否存在
    # 对每个参数值进行验证
    # 验证参数间的相互约束
```

## 数据流描述

1. **参数空间创建**：
   - 实例化 `ParameterSpace` 对象
   - 添加各种类型的参数定义
   - 设置优化目标和约束条件

2. **设计点生成**：
   - 通过采样方法生成候选设计点
   - 将内部表示转换为外部表示供用户使用

3. **设计点验证**：
   - 接收外部设计点定义
   - 验证参数值是否在有效范围内
   - 检查是否满足所有约束条件

4. **表示转换流程**：
   - 外部表示 (用户输入) → 内部表示 (优化算法使用)
   - 内部表示 (算法输出) → 外部表示 (返回给用户)

5. **与优化系统集成**：
   - 为设计生成器提供参数定义和采样接口
   - 为代理模型提供标准化的参数表示
   - 为优化算法提供约束条件验证

## 代码验证规则

1. **参数合法性验证**：
   - 连续参数的下界必须小于上界
   - 整数参数必须有有效的步长
   - 类别参数必须有唯一的选项值

2. **命名规则**：
   - 参数、目标和约束名称必须是字符串
   - 名称在各自的集合中必须唯一
   - 名称不得包含特殊字符（限定为字母、数字和下划线）

3. **值域验证**：
   - 连续参数值必须在[lower_bound, upper_bound]范围内
   - 整数参数值必须是整数并在指定范围内
   - 类别参数值必须是预定义类别中的一个

4. **约束一致性**：
   - 约束条件必须引用已定义的参数
   - 约束类型必须是预定义的ConstraintType枚举值之一

## 扩展计划

1. **条件参数**：
   - 实现基于其他参数值激活的条件参数
   - 支持复杂的参数依赖关系图

2. **高级约束**：
   - 支持多参数约束表达式
   - 添加非线性约束支持

3. **层次参数**：
   - 实现嵌套参数组支持
   - 提供层次参数的专用采样和转换方法

4. **自定义分布**：
   - 允许为连续参数指定自定义概率分布
   - 支持基于先验知识的带权采样

5. **参数组合规则**：
   - 添加参数组合的有效性规则
   - 实现组合规则的高效验证方法
