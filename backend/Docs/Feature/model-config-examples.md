# 代理模型配置示例

本文档提供各种代理模型的配置示例，用于贝叶斯优化过程中预测未知点的目标函数值。

## 支持的代理模型类型

贝叶斯优化系统支持以下几种代理模型，每种模型都有不同的特点和适用场景：

1. **高斯过程 (Gaussian Process, GP)**  
   提供最佳的不确定性估计，适用于小规模问题和平滑目标函数
   
2. **随机森林 (Random Forest, RF)**  
   适用于处理分类特征和不平滑目标函数
   
3. **主动多输出高斯过程 (MOGP)**  
   用于处理多目标优化问题

## JSON 配置示例

### 高斯过程模型

```json
{
  "model_type": "gaussian_process",
  "parameters": {
    "kernel": "matern",
    "nu": 2.5,
    "length_scale_bounds": [1e-5, 1e5],
    "alpha": 1e-10,
    "normalize_y": true,
    "n_restarts_optimizer": 5
  }
}
```

参数说明：
- `kernel`: 核函数类型，可选: "rbf", "matern", "rq" (rational quadratic)
- `nu`: Matern核的平滑参数 (仅当kernel="matern"时使用)，常用值: 0.5, 1.5, 2.5
- `length_scale_bounds`: 长度尺度边界，决定GP的平滑度
- `alpha`: 观测噪声水平，值越大表示数据越嘈杂
- `normalize_y`: 是否对目标值进行标准化
- `n_restarts_optimizer`: 超参数优化的重启次数

### 随机森林模型

```json
{
  "model_type": "random_forest",
  "parameters": {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": true,
    "random_state": 42
  }
}
```

参数说明：
- `n_estimators`: 森林中树的数量，值越大模型越精确但训练更慢
- `max_depth`: 树的最大深度，控制模型复杂度
- `min_samples_split`: 分裂内部节点所需的最小样本数
- `min_samples_leaf`: 叶节点所需的最小样本数
- `bootstrap`: 是否使用自助抽样
- `random_state`: 随机种子，用于结果可重现性

### 多输出高斯过程模型 (多目标优化)

```json
{
  "model_type": "multi_output_gp",
  "parameters": {
    "kernel": "matern",
    "nu": 2.5,
    "length_scale_bounds": [1e-5, 1e5],
    "alpha": 1e-6,
    "normalize_y": true,
    "n_restarts_optimizer": 5,
    "output_correlations": "learn"
  }
}
```

参数说明：
- 与GP模型参数相同，但增加了：
- `output_correlations`: 如何处理输出之间的相关性，可选: "learn", "independent", "fixed"

## 推荐使用场景

1. **高斯过程模型**
   - 样本数量较少 (<1000) 的场景
   - 连续、平滑的目标函数
   - 需要精确的不确定性估计
   - 适合低维到中等维度 (通常 <20) 的问题

2. **随机森林模型**
   - 包含分类特征的场景
   - 不平滑或有噪声的目标函数
   - 样本数量中等 (>1000) 的场景
   - 适合高维问题 (>20 维)
   - 当高斯过程模型训练时间过长时可以作为替代

3. **多输出高斯过程模型**
   - 多目标优化问题
   - 需要考虑输出之间的相关性
   - 适合低维到中等维度的问题

## 模型选择指南

选择合适的代理模型是贝叶斯优化成功的关键因素之一。以下是一些选择指南：

- **样本数量**: 对于小样本 (<100)，高斯过程通常表现最佳；对于大样本 (>1000)，随机森林更高效。
- **维度**: 对于低维问题，高斯过程通常优于随机森林；对于高维问题 (>20维)，随机森林往往更有效。
- **特征类型**: 如果有分类特征，随机森林更适合；如果全是连续特征，高斯过程可能更好。
- **计算资源**: 高斯过程在样本增加时计算成本增长很快；随机森林更具有可扩展性。

## 注意事项

1. 代理模型的选择和配置应根据具体问题特点调整。
2. 模型超参数可以在优化过程中动态调整，以适应优化的不同阶段。
3. 对于特别复杂的问题，可以考虑集成多个模型的预测结果。 
