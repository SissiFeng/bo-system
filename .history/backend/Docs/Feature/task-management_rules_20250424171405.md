# Bayesian Optimization System - Task Management and Strategy Configuration Design Rules

## Overview

The Task Management module is responsible for creating, retrieving, updating, and tracking optimization tasks. It provides a unified interface for task lifecycle management, including task creation, status tracking, and strategy configuration. This module serves as the bridge between the API layer and the core BO engine.

## Core Design Principles

1. **Task Isolation**: Each optimization task is isolated and contains its own parameter space, strategy, and results.
2. **Status Tracking**: Clear task status transitions with proper validation at each step.
3. **Persistent Storage**: All task data is persistently stored for reliability and recovery.
4. **Asynchronous Processing**: Long-running operations can be executed asynchronously with proper status updates.
5. **Idempotent Operations**: API operations are designed to be idempotent when appropriate.

## Data Models

### Task

```python
class TaskStatus(str, Enum):
    CREATED = "created"
    STRATEGY_SET = "strategy_set"
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Task:
    def __init__(self, task_id: str, name: str = None):
        self.task_id = task_id
        self.name = name or f"Task-{task_id[:8]}"
        self.status = TaskStatus.CREATED
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.parameter_space = None
        self.strategy = None
        self.results = []
        self.metadata = {}
```

### Strategy

```python
class OptimizationStrategy:
    def __init__(
        self,
        acquisition_function: str,
        acquisition_params: Dict[str, Any] = None,
        initial_design: Dict[str, Any] = None,
        model_params: Dict[str, Any] = None
    ):
        self.acquisition_function = acquisition_function
        self.acquisition_params = acquisition_params or {}
        self.initial_design = initial_design or {"type": "random", "num_samples": 10}
        self.model_params = model_params or {}
```

## Key Components

### TaskManager

```python
class TaskManager:
    def __init__(self, storage_manager):
        self.storage_manager = storage_manager
        
    def create_task(self, name=None):
        """Create a new optimization task."""
        # Implementation details
        
    def get_task(self, task_id):
        """Retrieve a task by its ID."""
        # Implementation details
        
    def list_tasks(self, filter_by=None, sort_by=None):
        """List all tasks with optional filtering and sorting."""
        # Implementation details
        
    def delete_task(self, task_id):
        """Delete a task and all its associated data."""
        # Implementation details
```

### StrategyManager

```python
class StrategyManager:
    def __init__(self, task_manager):
        self.task_manager = task_manager
        
    def set_strategy(self, task_id, strategy):
        """Set the optimization strategy for a task."""
        # Implementation details
        
    def get_strategy(self, task_id):
        """Get the current strategy for a task."""
        # Implementation details
```

## Key Method Implementations

### create_task()
- Generates a unique task ID
- Creates a new task with initial status CREATED
- Persists the task to storage
- Returns the task ID and basic task info

### set_strategy()
- Validates the strategy configuration
- Updates task status to STRATEGY_SET
- Persists the strategy to storage
- Validates that the task is in the correct state for setting a strategy

### get_initial_design()
- Validates that the task has a strategy set
- Creates a design generator based on the strategy's initial design configuration
- Generates initial design points
- Updates task status to INITIALIZED
- Returns the generated design points

### submit_results()
- Validates that results match the parameter space
- Adds results to the task's result collection
- Updates task status to RUNNING if this is the first batch of results
- Persists results to storage

### get_next_design()
- Ensures the task has results to train a model
- Uses the BO engine to recommend next points
- Returns the recommended points

## State Transitions

```
CREATED → STRATEGY_SET → INITIALIZED → RUNNING → COMPLETED
                                           ↓
                                        FAILED
```

Transitions occur as follows:
- CREATED → STRATEGY_SET: When a valid strategy is set
- STRATEGY_SET → INITIALIZED: After generating initial design points
- INITIALIZED → RUNNING: After submitting the first batch of results
- RUNNING → COMPLETED: When optimization goal is reached or user marks as completed
- Any state → FAILED: When an unrecoverable error occurs

## Storage Management

The TaskManager interfaces with a StorageManager that handles persistent storage of tasks:

```python
class StorageManager:
    def __init__(self, base_path):
        self.base_path = base_path
        
    def save_task(self, task):
        """Save task data to persistent storage."""
        # Implementation details
        
    def load_task(self, task_id):
        """Load task data from persistent storage."""
        # Implementation details
        
    def list_tasks(self):
        """List all tasks from storage."""
        # Implementation details
        
    def delete_task(self, task_id):
        """Delete a task from storage."""
        # Implementation details
```

## API Integration

The task management module is exposed through API endpoints:

```python
@router.post("/tasks", response_model=TaskCreationResponse)
def create_task(task_request: TaskCreationRequest):
    task_id = task_manager.create_task(task_request.name)
    return {"task_id": task_id}

@router.post("/strategy/{task_id}", response_model=StrategyResponse)
def set_strategy(task_id: str, strategy: StrategyRequest):
    strategy_manager.set_strategy(task_id, strategy)
    return {"task_id": task_id, "status": "strategy_set"}
```

## Error Handling

1. **Not Found Errors**: When a task ID doesn't exist
   ```python
   class TaskNotFoundError(Exception):
       """Raised when a task is not found."""
       pass
   ```

2. **State Transition Errors**: When an operation is incompatible with the task's current state
   ```python
   class InvalidTaskStateError(Exception):
       """Raised when an operation can't be performed in the current task state."""
       pass
   ```

3. **Validation Errors**: When input data doesn't meet requirements
   ```python
   class ValidationError(Exception):
       """Raised when input data doesn't meet validation requirements."""
       pass
   ```

## Performance Requirements

1. Task creation should complete in under 500ms
2. Task retrieval should complete in under 200ms
3. Strategy setting should validate and persist in under 1s
4. The system should support at least 1000 concurrent tasks

## Future Expansion

1. **Task Templates**: Predefined task templates for common optimization scenarios
2. **Batch Operations**: APIs for managing multiple tasks simultaneously
3. **Advanced Filtering**: More complex filters for task listing
4. **Collaborative Tasks**: Shared optimization tasks with role-based access control
5. **Automated Strategy Selection**: Intelligent selection of strategies based on problem characteristics

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
