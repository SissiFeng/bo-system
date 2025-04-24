# Task Management Module - Design and Implementation Rules

## Overview

The Task Management module is responsible for managing the lifecycle of optimization tasks, providing a consistent interface for task creation, status tracking, persistence, and retrieval. This module serves as the central coordination component for the entire Bayesian optimization process.

## Core Design Principles

1. **State Management**: Implement robust state management for optimization tasks, ensuring that task state is consistent and recoverable.
2. **Persistence**: Provide reliable persistence mechanisms to save and load task data, allowing for system restarts and long-running optimization processes.
3. **Isolation**: Ensure each task is isolated from others, preventing cross-task contamination while enabling concurrent optimization processes.
4. **Observability**: Implement comprehensive logging and event tracking to monitor task progress and diagnose issues.
5. **Recovery**: Include mechanisms for error recovery and task resumption after failures.

## Enumerations

### TaskStatus
```python
class TaskStatus(str, Enum):
    CREATED = "created"           # Task created but not yet started
    INITIALIZING = "initializing" # Initial design points being generated
    ACTIVE = "active"             # Task is actively being optimized
    PAUSED = "paused"             # Task temporarily paused
    COMPLETED = "completed"       # Optimization completed successfully
    FAILED = "failed"             # Task failed due to an error
```

## Class Definitions

### Task
```python
class Task:
    """
    Represents a Bayesian optimization task with its associated state and data.
    """
    def __init__(self, task_id: str, name: str = None, description: str = None):
        self.task_id = task_id
        self.name = name or f"Task-{task_id}"
        self.description = description
        self.status = TaskStatus.CREATED
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.parameter_space = None
        self.strategy = None
        self.designs = []
        self.results = []
        self.model = None
        self.metadata = {}
```

### TaskManager
```python
class TaskManager:
    """
    Manages the lifecycle of optimization tasks, providing methods
    for task creation, retrieval, updating, and persistence.
    """
    def __init__(self, storage_backend):
        self.tasks = {}  # In-memory task storage
        self.storage_backend = storage_backend
        
    def create_task(self, name=None, description=None) -> Task:
        """Create a new optimization task"""
        
    def get_task(self, task_id: str) -> Task:
        """Retrieve a task by its ID"""
        
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks, with optional filtering"""
        
    def update_task_status(self, task_id: str, status: TaskStatus) -> Task:
        """Update the status of a task"""
        
    def set_parameter_space(self, task_id: str, parameter_space: ParameterSpace) -> Task:
        """Set the parameter space for a task"""
        
    def set_strategy(self, task_id: str, strategy: Dict) -> Task:
        """Set the optimization strategy for a task"""
        
    def add_design_points(self, task_id: str, design_points: List[Dict]) -> Task:
        """Add design points to a task"""
        
    def add_results(self, task_id: str, results: List[Dict]) -> Task:
        """Add experimental results to a task"""
        
    def export_task(self, task_id: str) -> Dict:
        """Export complete task data"""
        
    def persist_task(self, task_id: str) -> bool:
        """Save task state to the storage backend"""
        
    def load_task(self, task_id: str) -> Task:
        """Load task state from the storage backend"""
        
    def delete_task(self, task_id: str) -> bool:
        """Delete a task and its associated data"""
```

### StorageBackend (Abstract)
```python
class StorageBackend(ABC):
    """
    Abstract base class for task storage backends.
    Implementations may include FileStorage, DatabaseStorage, etc.
    """
    @abstractmethod
    def save(self, task_id: str, data: Dict) -> bool:
        """Save task data to storage"""
        pass
    
    @abstractmethod
    def load(self, task_id: str) -> Dict:
        """Load task data from storage"""
        pass
    
    @abstractmethod
    def list_tasks(self) -> List[str]:
        """List all available task IDs"""
        pass
    
    @abstractmethod
    def delete(self, task_id: str) -> bool:
        """Delete task data from storage"""
        pass
```

### FileStorageBackend
```python
class FileStorageBackend(StorageBackend):
    """
    Implementation of StorageBackend that uses the file system.
    Organizes task data in a directory structure:
    data/tasks/{task_id}/
        - metadata.json        # Task metadata
        - parameter_space.json # Parameter space definition
        - strategy.json        # Optimization strategy
        - designs/            # Directory for design points
            - initial.json    # Initial design points
            - batch_1.json    # First batch of recommended points
            - ...
        - results/            # Directory for experimental results
            - batch_1.json    # Results for first batch
            - ...
        - models/             # Directory for serialized models
            - latest.pkl      # Latest model state
            - history/        # Historical model snapshots
                - iteration_1.pkl
                - ...
    """
```

## Key Method Implementations

### Task Creation and Retrieval
```python
def create_task(self, name=None, description=None) -> Task:
    """Create a new optimization task with a unique ID"""
    task_id = str(uuid.uuid4())
    task = Task(task_id, name, description)
    self.tasks[task_id] = task
    self.persist_task(task_id)
    return task

def get_task(self, task_id: str) -> Task:
    """Retrieve a task by its ID, loading from storage if necessary"""
    if task_id not in self.tasks:
        try:
            task_data = self.storage_backend.load(task_id)
            self.tasks[task_id] = self._deserialize_task(task_data)
        except Exception as e:
            raise TaskNotFoundError(f"Task {task_id} not found: {str(e)}")
    return self.tasks[task_id]
```

### Task State Management
```python
def update_task_status(self, task_id: str, status: TaskStatus) -> Task:
    """Update the status of a task and record the update timestamp"""
    task = self.get_task(task_id)
    task.status = status
    task.updated_at = datetime.now()
    self.persist_task(task_id)
    return task

def add_results(self, task_id: str, results: List[Dict]) -> Task:
    """
    Add experimental results to a task
    
    Parameters:
    -----------
    task_id : str
        ID of the task to update
    results : List[Dict]
        List of result dictionaries, each containing:
        - design_id: ID of the design point
        - values: Dict mapping objective names to measured values
        - timestamp: When the result was obtained (optional)
        - metadata: Additional result metadata (optional)
    
    Returns:
    --------
    Task
        Updated task object
    """
    task = self.get_task(task_id)
    for result in results:
        # Validate result against parameter space and design
        if not self._validate_result(task, result):
            raise InvalidResultError(f"Invalid result format: {result}")
        
        # Add result to task
        task.results.append(result)
    
    task.updated_at = datetime.now()
    self.persist_task(task_id)
    return task
```

### Task Persistence
```python
def persist_task(self, task_id: str) -> bool:
    """Save task state to the storage backend"""
    task = self.tasks.get(task_id)
    if not task:
        return False
    
    task_data = self._serialize_task(task)
    return self.storage_backend.save(task_id, task_data)

def _serialize_task(self, task: Task) -> Dict:
    """Convert a Task object to a serializable dictionary"""
    return {
        "task_id": task.task_id,
        "name": task.name,
        "description": task.description,
        "status": task.status,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "parameter_space": task.parameter_space.to_dict() if task.parameter_space else None,
        "strategy": task.strategy,
        "designs": task.designs,
        "results": task.results,
        "metadata": task.metadata
    }
```

## Data Flow

1. **Task Creation**:
   - Client requests task creation via API
   - `TaskManager` creates a new `Task` with a unique ID
   - Task is persisted to storage
   - Task ID is returned to client

2. **Parameter Space Configuration**:
   - Client submits parameter space definition
   - API endpoint validates the definition
   - `TaskManager` updates the task with the parameter space
   - Changes are persisted

3. **Optimization Cycle**:
   - Initial design points are generated and added to task
   - Client retrieves design points via API
   - Client submits experimental results
   - Results are validated and added to task
   - Next batch of design points is generated using the BO engine
   - Cycle continues until completion or failure

4. **Task Completion**:
   - Client marks task as complete or system automatically detects completion
   - Final task state is persisted
   - Complete results can be exported

## Code Validation Rules

1. **Error Handling**: All methods should use proper exception handling with specific exception types.
2. **Validation**: Input data should be validated before processing to ensure consistency.
3. **Atomicity**: State changes should be atomic - either complete successfully or fail without partial updates.
4. **Recovery**: The system should be able to recover task state after a crash or restart.
5. **Idempotence**: Methods should be idempotent where possible to prevent duplicate operations.

## Future Expansion Plans

1. **Database Backend**: Implement a database storage backend for improved scalability and querying.
2. **Task Versioning**: Add support for versioning task definitions and results.
3. **User Authentication**: Integrate with authentication system to associate tasks with users.
4. **Task Templates**: Allow creation of task templates for common optimization scenarios.
5. **Distributed Tasks**: Support distributed task execution across multiple worker nodes.
6. **Automated Cleanup**: Implement policy-based cleanup of old or inactive tasks.
7. **Task Archiving**: Support archiving completed tasks to cold storage for long-term retention.
8. **Notifications**: Implement an event notification system for task state changes.
