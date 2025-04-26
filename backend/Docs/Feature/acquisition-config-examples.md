# 采集函数配置示例

本文档提供各种采集函数的配置示例，用于贝叶斯优化过程中指导采样策略。

## 采集函数类型

贝叶斯优化系统支持以下几种采集函数，每种采集函数都有不同的特点和适用场景：

1. **期望改进 (Expected Improvement, EI)**  
   平衡探索和利用，适用于大多数场景
   
2. **改进概率 (Probability of Improvement, PI)**  
   更偏向利用，在需要快速收敛的场景中使用
   
3. **置信上界 (Upper Confidence Bound, UCB)**  
   更加灵活的探索-利用平衡，通过 kappa 参数控制
   
4. **随机采样 (Random)**  
   在模型训练失败或初始阶段使用

## JSON 配置示例

### 期望改进 (EI) 采集函数

```json
{
  "acquisition_function": "expected_improvement",
  "parameters": {
    "xi": 0.01,
    "maximize": false
  }
}
```

参数说明：
- `xi`: 探索-利用平衡参数，值越大越偏向探索，范围通常为 0.0001~0.1
- `maximize`: 是否为最大化问题，true 表示最大化，false 表示最小化

### 改进概率 (PI) 采集函数

```json
{
  "acquisition_function": "probability_improvement",
  "parameters": {
    "xi": 0.05,
    "maximize": false
  }
}
```

参数说明：
- `xi`: 改进阈值，值越大需要更显著的改进才会被选中，范围通常为 0.001~0.1
- `maximize`: 是否为最大化问题

### 置信上界 (UCB) 采集函数

```json
{
  "acquisition_function": "upper_confidence_bound",
  "parameters": {
    "kappa": 2.0,
    "maximize": false
  }
}
```

参数说明：
- `kappa`: 探索参数，值越大越偏向探索未知区域，范围通常为 0.1~5.0
- `maximize`: 是否为最大化问题

### 随机采样 Fallback

```json
{
  "acquisition_function": "random",
  "parameters": {}
}
```

## 动态调整采集函数参数

在优化过程中，可以根据迭代次数动态调整采集函数参数。例如，随着迭代进行，可以逐渐减小探索参数：

```json
{
  "acquisition_function": "expected_improvement",
  "parameters": {
    "xi": 0.05,
    "maximize": false
  },
  "schedule": {
    "xi": {
      "type": "exponential_decay",
      "initial_value": 0.05,
      "decay_rate": 0.9,
      "min_value": 0.001
    }
  }
}
```

## 推荐使用场景

1. **期望改进 (EI)**
   - 默认首选采集函数
   - 平衡的探索-利用特性
   - 适合大多数优化问题

2. **改进概率 (PI)**
   - 更关注较小的确定性改进
   - 适合噪声较小且需要精确收敛的场景
   - 在优化后期使用效果更好

3. **置信上界 (UCB)**
   - 可通过 kappa 参数灵活调整探索程度
   - 适合需要更多探索的复杂响应面
   - 在高维空间中表现良好

4. **随机采样**
   - 初始探索阶段
   - 当代理模型训练失败时的 fallback 策略
   - 用于跳出局部最优 
