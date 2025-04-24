# 设计生成器 (DesignGenerator) 设计与实现规则

## 模块概述

设计生成器模块负责在参数空间中生成初始设计点和候选点，是贝叶斯优化系统的重要组成部分。该模块提供多种实验设计方法，包括随机设计、拉丁超立方设计、因子设计和索伯序列设计等，以满足不同优化场景的需求。设计生成器通过与参数空间模块的交互，确保生成的设计点符合参数约束和依赖关系。

## 核心设计原则

1. **算法多样性**：
   - 支持多种实验设计算法，适应不同的优化需求
   - 为不同参数类型和维度提供最优的采样策略

2. **参数空间兼容**：
   - 与参数空间模块紧密集成，确保设计点的有效性
   - 支持所有参数类型（连续、整数、类别）的混合设计

3. **可扩展性**：
   - 基于工厂模式实现设计生成器的创建
   - 便于添加新的设计生成算法

4. **随机性控制**：
   - 提供随机种子设置，确保实验的可重复性
   - 支持确定性生成模式

## 类层次与结构

### 枚举类型

```python
class DesignType(Enum):
    """设计类型枚举"""
    RANDOM = "random"
    LATIN_HYPERCUBE = "latin_hypercube"
    FACTORIAL = "factorial"
    SOBOL = "sobol"
    CUSTOM = "custom"
```

### 基类定义

```python
class DesignGenerator(ABC):
    """设计生成器抽象基类，定义所有设计生成器共有的接口"""
    
    def __init__(self, parameter_space):
        self.parameter_space = parameter_space
        
    @abstractmethod
    def generate(self, num_points, random_state=None):
        """生成设计点集合"""
        pass
```

### 具体设计生成器

```python
class RandomDesignGenerator(DesignGenerator):
    """随机设计生成器"""
    
    def __init__(self, parameter_space):
        super().__init__(parameter_space)
        
    def generate(self, num_points, random_state=None):
        """生成随机设计点"""
        # 使用参数空间的采样方法生成随机点
        # 检查点的有效性
        # 返回生成的设计点列表
```

```python
class LatinHypercubeDesignGenerator(DesignGenerator):
    """拉丁超立方设计生成器"""
    
    def __init__(self, parameter_space, criterion="maximin"):
        super().__init__(parameter_space)
        self.criterion = criterion
        
    def generate(self, num_points, random_state=None):
        """生成拉丁超立方设计点"""
        # 根据参数空间维度创建拉丁超立方采样
        # 将采样结果转换为参数空间中的实际值
        # 处理类别参数
        # 返回生成的设计点列表
```

```python
class FactorialDesignGenerator(DesignGenerator):
    """因子设计生成器"""
    
    def __init__(self, parameter_space, levels=None):
        super().__init__(parameter_space)
        self.levels = levels or {}
        
    def generate(self, num_points=None, random_state=None):
        """生成因子设计点"""
        # 确定每个参数的水平数
        # 生成完整的因子组合
        # 对结果进行过滤，确保符合约束条件
        # 返回生成的设计点列表
```

```python
class SobolDesignGenerator(DesignGenerator):
    """索伯序列设计生成器"""
    
    def __init__(self, parameter_space, scramble=True):
        super().__init__(parameter_space)
        self.scramble = scramble
        
    def generate(self, num_points, random_state=None):
        """生成索伯序列设计点"""
        # 创建Sobol序列生成器
        # 生成指定数量的低差异序列
        # 将序列转换为参数空间中的实际值
        # 返回生成的设计点列表
```

```python
class CustomDesignGenerator(DesignGenerator):
    """自定义设计生成器"""
    
    def __init__(self, parameter_space, designs=None):
        super().__init__(parameter_space)
        self.designs = designs or []
        
    def generate(self, num_points=None, random_state=None):
        """返回自定义设计点"""
        # 验证自定义设计点的有效性
        # 转换为内部表示
        # 返回自定义设计点列表
        
    def add_design(self, design):
        """添加自定义设计点"""
        # 验证设计点
        # 添加到设计集合
```

### 工厂函数

```python
def create_design_generator(design_type, parameter_space, **kwargs):
    """设计生成器工厂函数"""
    if design_type == DesignType.RANDOM:
        return RandomDesignGenerator(parameter_space)
    elif design_type == DesignType.LATIN_HYPERCUBE:
        return LatinHypercubeDesignGenerator(parameter_space, **kwargs)
    elif design_type == DesignType.FACTORIAL:
        return FactorialDesignGenerator(parameter_space, **kwargs)
    elif design_type == DesignType.SOBOL:
        return SobolDesignGenerator(parameter_space, **kwargs)
    elif design_type == DesignType.CUSTOM:
        return CustomDesignGenerator(parameter_space, **kwargs)
    else:
        raise ValueError(f"Unknown design type: {design_type}")
```

## 关键方法实现

### 随机设计生成

```python
def generate_random_design(self, num_points, random_state=None):
    """生成随机设计点"""
    # 设置随机种子
    rng = np.random.RandomState(random_state)
    
    # 为每个参数采样
    designs = []
    for _ in range(num_points):
        design = {}
        for name, param in self.parameter_space.parameters.items():
            design[name] = param.sample(1, rng)[0]
        
        # 验证设计点有效性
        if self.parameter_space.validate_design(design):
            designs.append(design)
    
    return designs
```

### 拉丁超立方设计生成

```python
def generate_lhs_design(self, num_points, random_state=None):
    """生成拉丁超立方设计点"""
    # 设置随机种子
    rng = np.random.RandomState(random_state)
    
    # 获取连续和整数参数的数量
    continuous_params = [p for p in self.parameter_space.parameters.values() 
                         if p.parameter_type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]]
    n_continuous = len(continuous_params)
    
    if n_continuous > 0:
        # 创建拉丁超立方采样
        lhs_samples = lhs(n_continuous, samples=num_points, criterion=self.criterion, random_state=rng)
        
        # 转换为实际参数值
        designs = []
        for i in range(num_points):
            design = {}
            continuous_idx = 0
            
            for name, param in self.parameter_space.parameters.items():
                if param.parameter_type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]:
                    # 连续和整数参数使用LHS采样
                    value = param.from_unit_interval(lhs_samples[i, continuous_idx])
                    design[name] = value
                    continuous_idx += 1
                else:
                    # 类别参数使用随机采样
                    design[name] = param.sample(1, rng)[0]
            
            # 验证设计点有效性
            if self.parameter_space.validate_design(design):
                designs.append(design)
        
        return designs
    else:
        # 如果没有连续或整数参数，回退到随机采样
        return self.generate_random_design(num_points, random_state)
```

### 网格设计生成

```python
def generate_grid_design(self, levels=None):
    """生成网格设计点"""
    if levels is None:
        levels = self.levels
    
    # 确定每个参数的水平数
    param_levels = {}
    for name, param in self.parameter_space.parameters.items():
        if name in levels:
            param_levels[name] = levels[name]
        elif param.parameter_type == ParameterType.CATEGORICAL:
            param_levels[name] = len(param.categories)
        elif param.parameter_type == ParameterType.INTEGER:
            range_size = param.upper_bound - param.lower_bound + 1
            param_levels[name] = min(range_size, 5)  # 默认最多5个水平
        else:
            param_levels[name] = 5  # 连续参数默认5个水平
    
    # 生成网格点
    # 使用itertools.product生成组合
    
    # 过滤无效点
    # 验证每个点是否满足约束条件
    
    return valid_designs
```

### 自定义设计验证与添加

```python
def add_custom_design(self, design_dict):
    """添加自定义设计点"""
    # 验证设计点格式
    if not isinstance(design_dict, dict):
        raise ValueError("Design must be a dictionary")
        
    # 验证设计点参数
    if not self.parameter_space.validate_design(design_dict):
        raise ValueError("Design violates parameter space constraints")
        
    # 添加到自定义设计列表
    self.designs.append(design_dict)
    
    return True
```

## 数据流描述

1. **设计器初始化**：
   - 接收参数空间对象作为输入
   - 根据设计类型和附加参数配置设计生成器

2. **设计点生成**：
   - 根据指定的数量和随机种子生成设计点
   - 调用特定算法生成候选点
   - 验证生成点的有效性
   - 返回符合要求的设计点集合

3. **与参数空间交互**：
   - 使用参数空间的采样方法生成基础随机值
   - 调用参数空间的验证方法检查设计点的约束合规性
   - 使用参数空间的转换方法在内部和外部表示之间转换

4. **设计存储与加载**：
   - 支持将生成的设计点保存到文件
   - 能够从文件加载预定义设计点

## 代码验证规则

1. **设计点合法性验证**：
   - 所有设计点必须包含参数空间中定义的所有参数
   - 设计点的参数值必须符合参数空间的约束条件
   - 设计点集合中不应有重复点（在指定的容差范围内）

2. **算法实现正确性**：
   - 随机设计应确保均匀覆盖整个参数空间
   - 拉丁超立方设计应确保每个维度的均匀分布
   - 因子设计应生成所有可能的因子组合
   - 索伯序列应具备良好的低差异性质

3. **性能要求**：
   - 设计生成的时间复杂度应与设计点数量和参数空间维度成正比
   - 对于高维参数空间，应采用高效的采样算法
   - 应避免生成大量被参数约束拒绝的无效点

## 扩展计划

1. **自适应设计**：
   - 实现基于现有观测结果动态调整的设计生成器
   - 支持探索-利用平衡的自适应采样策略

2. **批量设计优化**：
   - 为并行评估优化批量设计生成
   - 实现点集间最大化距离的批量设计方法

3. **约束感知设计**：
   - 开发能够有效处理复杂约束的设计生成器
   - 实现基于约束边界的采样策略

4. **多保真度设计**：
   - 支持不同保真度级别的实验设计
   - 实现跨保真度的相关设计生成

5. **领域特定设计**：
   - 添加针对特定应用领域优化的设计生成器
   - 支持基于专家知识的设计生成

## 🔄 与其他模块的交互

- **输入**：来自`ParameterSpace`模块的参数空间定义
- **输出**：设计点供`BOSystem`和评估模块使用
- **依赖**：依赖`utils.py`中的辅助函数和`parameter_space.py`中的参数定义

## 🔮 未来扩展

1. 添加更多实验设计方法（Halton序列、正交阵等）
2. 支持约束空间中的设计生成
3. 增加自适应设计生成策略
4. 添加对分布式并行实验的支持 
