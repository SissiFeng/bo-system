# 贝叶斯优化系统 - 任务管理与策略配置设计规则

## 1. 概述

任务管理模块负责在贝叶斯优化系统中管理多个优化任务的创建、状态跟踪、持久化和策略配置。本文档详细说明了相关的设计规则和实现策略。

## 2. 核心设计原则

1. **持久化优先**：所有任务相关数据都持久化到文件系统，确保系统崩溃或重启后能够完整恢复任务状态。
2. **懒加载机制**：任务数据在需要时才从文件系统加载到内存，减少内存占用。
3. **健壮性**：各种异常情况（文件缺失、数据格式不正确等）都有相应的处理机制。
4. **状态一致性**：确保内存中的任务状态与文件系统中的状态保持一致。
5. **进度跟踪**：提供任务进度的实时更新。

## 3. 文件结构

每个任务都有一个唯一的 `task_id`，其数据存储在以下文件结构中：

```
/data/tasks/{task_id}/
  ├── task_info.json       # 任务元数据
  ├── parameter_space.json # 参数空间定义
  ├── strategy.json        # 优化策略配置
  ├── initial_designs.json # 初始设计点
  ├── results.json         # 实验结果
  ├── error.log            # 错误日志(可选)
  └── export.{format}      # 导出文件(按需生成)
```

## 4. 数据模型

### 4.1 任务信息 (task_info.json)

```json
{
  "task_id": "uuid-string",
  "name": "任务名称",
  "status": "created|running|paused|completed|failed",
  "created_at": "ISO时间",
  "updated_at": "ISO时间",
  "progress": 0.0-100.0,
  "description": "任务描述"
}
```

### 4.2 优化策略 (strategy.json)

```json
{
  "algorithm": "gaussian_process",
  "acquisition_function": "ei",
  "batch_size": 1,
  "settings": {
    "exploration_weight": 0.5,
    "kernel": "matern",
    "noise_level": 0.01,
    "iterations": 50
  },
  "task_id": "uuid-string",
  "created_at": "ISO时间",
  "updated_at": "ISO时间"
}
```

## 5. API 端点

### 5.1 任务查询

- **GET /api/tasks**：获取所有任务列表，懒加载文件系统中的任务。
- **GET /api/tasks/{task_id}/status**：获取特定任务的状态，计算进度，查找最佳结果。

### 5.2 任务配置

- **POST /api/strategy/{task_id}**：设置优化策略，保存到 strategy.json。
- **GET /api/strategy/{task_id}**：获取当前优化策略。

### 5.3 任务管理

- **POST /api/tasks/{task_id}/restart**：重启任务，可选是否保留历史数据。
- **GET /api/tasks/{task_id}/export**：导出任务数据为 JSON 或 CSV 格式。
- **GET /api/diagnostics/{task_id}**：获取任务诊断信息，检查文件和状态一致性。

### 5.4 WebSocket

- **WS /ws/tasks/{task_id}**：实时接收任务状态更新。

## 6. 任务状态流转

```
创建(CREATED) --> 运行(RUNNING) --> 暂停(PAUSED) --> 运行(RUNNING) --> 完成(COMPLETED)
                         |                               |
                         v                               v
                      失败(FAILED) ---------------------->
```

## 7. 优化器管理

`get_or_create_optimizer` 函数负责：

1. 首先检查内存中是否已有优化器实例。
2. 如果没有，从文件系统加载参数空间、策略配置和已有结果。
3. 实例化优化器并根据策略配置正确设置模型和采集函数。
4. 缓存优化器实例到内存以备后续使用。

## 8. 进度计算规则

任务进度计算依据：

1. 如果策略中指定了总迭代次数(`iterations`)，进度 = min(100%, 当前结果数 / 总迭代数 * 100%)
2. 如果没有指定总迭代数：
   - CREATED 状态：0%
   - RUNNING 状态：根据结果数估算，最多到 80%
   - PAUSED 状态：70%
   - COMPLETED 状态：100%
   - FAILED 状态：30%

## 9. 错误处理

1. **文件不存在**：尝试从其他可用数据重建，或返回404错误。
2. **数据格式错误**：记录到错误日志，尝试使用默认值或返回500错误。
3. **优化器错误**：记录到任务的 error.log 文件，返回500错误。

## 10. 未来扩展

1. **数据库存储**：将文件系统存储替换为数据库存储，以支持更高的并发和更复杂的查询。
2. **任务依赖**：支持任务之间的依赖关系，实现串行或并行执行多个相关优化任务。
3. **用户权限**：增加用户管理和权限控制，支持多用户场景。
4. **任务模板**：支持将常用任务配置保存为模板，方便快速创建新任务。

## 11. 实现注意事项

1. **内存管理**：大量任务时实现LRU缓存，防止内存溢出。
2. **文件锁**：并发访问同一任务时使用文件锁防止数据竞争。
3. **事务**：多文件更新时实现简单的事务机制，确保数据一致性。
4. **异步**：耗时操作使用异步处理，不阻塞API响应。 
