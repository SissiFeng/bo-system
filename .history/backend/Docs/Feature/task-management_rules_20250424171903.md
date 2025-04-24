# Task Management - Design and Implementation Rules

## Module Overview

The Task Management module is responsible for creating, tracking, storing, and managing optimization tasks within the Bayesian Optimization system. It serves as a central component for managing the lifecycle of optimization tasks, from creation to completion, and provides interfaces for task state transitions, persistence, and retrieval.

## Core Design Principles

1. **State Management**: Implement a clear state machine for task lifecycle management with well-defined transitions
2. **Persistence**: Ensure reliable storage and retrieval of task information
3. **Concurrency Safety**: Handle concurrent access to tasks safely
4. **Query Efficiency**: Optimize for efficient task querying and filtering
5. **Event Propagation**: Implement an event system for task state changes
6. **Extensibility**: Design for future extension of task properties and states

## Class Hierarchy

### TaskStatus Enum

```python
class TaskStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
```

### Task Class

```python
class Task:
    def __init__(self, task_id: str = None, name: str = None, description: str = None):
        self.task_id = task_id or str(uuid.uuid4())
        self.name = name or f"Task_{self.task_id[:8]}"
        self.description = description or ""
        self.status = TaskStatus.CREATED
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.parameter_space = None
        self.strategy = None
        self.initial_designs = []
        self.results = []
        self.next_points = []
        self.model_state = None
        self.metadata = {}
```

### TaskManager Class

```python
class TaskManager:
    def __init__(self, storage_provider=None):
        self.storage_provider = storage_provider or InMemoryStorageProvider()
        self.tasks = {}
        self._load_tasks()
        
    def create_task(self, name=None, description=None, parameter_space=None):
        # Create a new task and persist it
        
    def get_task(self, task_id):
        # Retrieve a task by ID
        
    def update_task(self, task_id, **updates):
        # Update task properties and persist changes
        
    def delete_task(self, task_id):
        # Delete a task
        
    def list_tasks(self, status=None, limit=None, offset=None):
        # List tasks with optional filtering
        
    def transition_status(self, task_id, new_status):
        # Change task status with validation
```

### StorageProvider Interface

```python
class StorageProvider(ABC):
    @abstractmethod
    def save_task(self, task):
        pass
        
    @abstractmethod
    def load_task(self, task_id):
        pass
        
    @abstractmethod
    def delete_task(self, task_id):
        pass
        
    @abstractmethod
    def list_tasks(self, **filters):
        pass
```

### InMemoryStorageProvider Implementation

```python
class InMemoryStorageProvider(StorageProvider):
    def __init__(self):
        self.tasks = {}
        
    def save_task(self, task):
        self.tasks[task.task_id] = copy.deepcopy(task)
        return task.task_id
        
    def load_task(self, task_id):
        if task_id not in self.tasks:
            return None
        return copy.deepcopy(self.tasks[task_id])
        
    def delete_task(self, task_id):
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False
        
    def list_tasks(self, **filters):
        # Filter tasks based on provided criteria
        result = list(self.tasks.values())
        # Apply filters
        return result
```

### FileStorageProvider Implementation

```python
class FileStorageProvider(StorageProvider):
    def __init__(self, storage_dir="./data/tasks"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
    def save_task(self, task):
        task_path = os.path.join(self.storage_dir, f"{task.task_id}.json")
        with open(task_path, 'w') as f:
            json.dump(task.__dict__, f, default=self._json_serializer)
        return task.task_id
        
    def load_task(self, task_id):
        task_path = os.path.join(self.storage_dir, f"{task_id}.json")
        if not os.path.exists(task_path):
            return None
        with open(task_path, 'r') as f:
            task_dict = json.load(f)
        return self._dict_to_task(task_dict)
```

## Key Methods Implementation

### Task Creation

```python
def create_task(self, name=None, description=None, parameter_space=None):
    task = Task(name=name, description=description)
    if parameter_space:
        task.parameter_space = parameter_space
    task_id = self.storage_provider.save_task(task)
    self.tasks[task_id] = task
    self._publish_event(TaskEvent.CREATED, task)
    return task
```

### Task State Transition

```python
def transition_status(self, task_id, new_status):
    if task_id not in self.tasks:
        raise TaskNotFoundError(f"Task {task_id} not found")
        
    task = self.tasks[task_id]
    old_status = task.status
    
    # Validate state transition
    if not self._is_valid_transition(old_status, new_status):
        raise InvalidStatusTransitionError(
            f"Cannot transition from {old_status} to {new_status}"
        )
        
    # Update task status
    task.status = new_status
    task.updated_at = datetime.now()
    
    # Persist changes
    self.storage_provider.save_task(task)
    
    # Publish event
    self._publish_event(TaskEvent.STATUS_CHANGED, task, {
        'old_status': old_status,
        'new_status': new_status
    })
    
    return task
```

### Task Update

```python
def update_task(self, task_id, **updates):
    if task_id not in self.tasks:
        raise TaskNotFoundError(f"Task {task_id} not found")
        
    task = self.tasks[task_id]
    
    # Apply updates to task
    for key, value in updates.items():
        if hasattr(task, key):
            setattr(task, key, value)
    
    task.updated_at = datetime.now()
    
    # Persist changes
    self.storage_provider.save_task(task)
    
    # Publish event
    self._publish_event(TaskEvent.UPDATED, task)
    
    return task
```

### Task Querying

```python
def list_tasks(self, status=None, limit=None, offset=None):
    filters = {}
    if status:
        filters['status'] = status
        
    tasks = self.storage_provider.list_tasks(**filters)
    
    # Apply pagination
    if offset is not None:
        tasks = tasks[offset:]
    if limit is not None:
        tasks = tasks[:limit]
        
    return tasks
```

## Data Flow

1. **Task Creation Flow**:
   - Client requests task creation via API
   - TaskManager creates a new Task instance
   - Task is saved to storage
   - Task creation event is published
   - Task ID is returned to client

2. **Task Update Flow**:
   - Client requests task update via API
   - TaskManager retrieves the task
   - Updates are applied to the task
   - Task is saved to storage
   - Task update event is published
   - Updated task is returned to client

3. **State Transition Flow**:
   - Client requests status change via API
   - TaskManager validates the transition
   - Status is updated and timestamp is refreshed
   - Task is saved to storage
   - Status change event is published
   - Updated task is returned to client

4. **Task Querying Flow**:
   - Client requests task list with filters
   - TaskManager applies filters to storage query
   - Pagination is applied if requested
   - Filtered task list is returned to client

## Code Validation Rules

1. **State Transition Validation**: Only allow valid state transitions according to the state machine
2. **Concurrency Control**: Use locks or transactions to prevent race conditions during task updates
3. **Error Handling**: Properly handle and report task-related errors with meaningful messages
4. **Event Consistency**: Ensure events are published for all task state changes
5. **Data Validation**: Validate task properties before persistence
6. **Transactional Operations**: Ensure operations that modify multiple tasks are atomic

## Future Expansion

1. **Tagging System**: Add support for task tagging and tag-based querying
2. **Task Prioritization**: Implement priority queues for task execution
3. **Batch Operations**: Support batch creation and updates of tasks
4. **Hierarchical Tasks**: Support parent-child relationships between tasks
5. **Advanced Filtering**: Implement more sophisticated query capabilities
6. **History Tracking**: Add versioning and history tracking for task changes
7. **User Association**: Associate tasks with users for access control
8. **Notification Rules**: Allow configuration of notification rules per task
