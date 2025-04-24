# Bayesian Optimization System - Task Management and Strategy Configuration Design Rules

## 1. Overview

The task management module is responsible for managing the creation, status tracking, persistence, and strategy configuration of multiple optimization tasks in the Bayesian optimization system. This document details the relevant design rules and implementation strategies.

## 2. Core Design Principles

1. **Persistence Priority**: All task-related data is persisted to the file system, ensuring complete recovery of task status after system crashes or restarts.
2. **Lazy Loading Mechanism**: Task data is loaded from the file system into memory only when needed, reducing memory usage.
3. **Robustness**: Various exceptional situations (missing files, incorrect data formats, etc.) have corresponding handling mechanisms.
4. **State Consistency**: Ensures that the task state in memory is consistent with the state in the file system.
5. **Progress Tracking**: Provides real-time updates of task progress.

## 3. File Structure

Each task has a unique `task_id`, with its data stored in the following file structure:

```
/data/tasks/{task_id}/
  ├── task_info.json       # Task metadata
  ├── parameter_space.json # Parameter space definition
  ├── strategy.json        # Optimization strategy configuration
  ├── initial_designs.json # Initial design points
  ├── results.json         # Experiment results
  ├── error.log            # Error log (optional)
  └── export.{format}      # Export file (generated as needed)
```

## 4. Data Models

### 4.1 Task Information (task_info.json)

```json
{
  "task_id": "uuid-string",
  "name": "Task Name",
  "status": "created|running|paused|completed|failed",
  "created_at": "ISO timestamp",
  "updated_at": "ISO timestamp",
  "progress": 0.0-100.0,
  "description": "Task description"
}
```

### 4.2 Optimization Strategy (strategy.json)

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
  "created_at": "ISO timestamp",
  "updated_at": "ISO timestamp"
}
```

## 5. API Endpoints

### 5.1 Task Queries

- **GET /api/tasks**: Get a list of all tasks, lazy loading tasks from the file system.
- **GET /api/tasks/{task_id}/status**: Get the status of a specific task, calculate progress, find the best result.

### 5.2 Task Configuration

- **POST /api/strategy/{task_id}**: Set the optimization strategy, save to strategy.json.
- **GET /api/strategy/{task_id}**: Get the current optimization strategy.

### 5.3 Task Management

- **POST /api/tasks/{task_id}/restart**: Restart a task, optionally preserving historical data.
- **GET /api/tasks/{task_id}/export**: Export task data in JSON or CSV format.
- **GET /api/diagnostics/{task_id}**: Get task diagnostic information, check file and state consistency.

### 5.4 WebSocket

- **WS /ws/tasks/{task_id}**: Receive real-time task status updates.

## 6. Task State Transitions

```
CREATED --> RUNNING --> PAUSED --> RUNNING --> COMPLETED
              |                      |
              v                      v
           FAILED ------------------>
```

## 7. Optimizer Management

The `get_or_create_optimizer` function is responsible for:

1. First checking if an optimizer instance already exists in memory.
2. If not, loading the parameter space, strategy configuration, and existing results from the file system.
3. Instantiating the optimizer and correctly setting the model and acquisition function according to the strategy configuration.
4. Caching the optimizer instance in memory for subsequent use.

## 8. Progress Calculation Rules

Task progress calculation basis:

1. If the total number of iterations (`iterations`) is specified in the strategy, progress = min(100%, current result count / total iterations * 100%)
2. If the total number of iterations is not specified:
   - CREATED status: 0%
   - RUNNING status: Estimated based on result count, up to 80%
   - PAUSED status: 70%
   - COMPLETED status: 100%
   - FAILED status: 30%

## 9. Error Handling

1. **File Does Not Exist**: Attempt to rebuild from other available data, or return a 404 error.
2. **Data Format Error**: Record to the error log, attempt to use default values or return a 500 error.
3. **Optimizer Error**: Record to the task's error.log file, return a 500 error.

## 10. Future Extensions

1. **Database Storage**: Replace file system storage with database storage to support higher concurrency and more complex queries.
2. **Task Dependencies**: Support dependencies between tasks, implementing serial or parallel execution of multiple related optimization tasks.
3. **User Permissions**: Add user management and permission control to support multi-user scenarios.
4. **Task Templates**: Support saving common task configurations as templates for easy creation of new tasks.

## 11. Implementation Considerations

1. **Memory Management**: Implement LRU caching for a large number of tasks to prevent memory overflow.
2. **File Locks**: Use file locks to prevent data races when concurrently accessing the same task.
3. **Transactions**: Implement a simple transaction mechanism for multi-file updates to ensure data consistency.
4. **Asynchrony**: Use asynchronous processing for time-consuming operations to avoid blocking API responses. 
