from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import re

# Common Base Models

class ObjectId(str):
    """MongoDB ObjectId field."""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_id
    
    @field_validator('ObjectId')
    @classmethod
    def validate_id(cls, v):
        """Validate that the id is a valid MongoDB ObjectId."""
        if not BsonObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return str(v)


# Parameter Type Enums

class ParameterType(str, Enum):
    """Enum for parameter types."""
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    INTEGER = "integer"  # integer parameter type


class ObjectiveDirection(str, Enum):
    """Enum for optimization direction."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ConstraintRelation(str, Enum):
    """Enum for constraint relations."""
    LESS_THAN_OR_EQUAL = "<="
    GREATER_THAN_OR_EQUAL = ">="
    EQUAL = "=="


class SamplingMethod(str, Enum):
    """Enum for initial sampling methods."""
    LHS = "lhs"
    RANDOM = "random"
    GRID = "grid"
    FACTORIAL = "factorial"
    SOBOL = "sobol"
    CUSTOM = "custom"


class AcquisitionFunction(str, Enum):
    """Enum for acquisition functions."""
    EI = "ei"
    UCB = "ucb"
    PI = "pi"
    EHVI = "ehvi"
    KNOWLEDGE_GRADIENT = "kg"
    ENTROPY = "entropy"
    CUSTOM = "custom"


class KernelType(str, Enum):
    """Enum for kernel types in Gaussian Processes."""
    RBF = "rbf"
    MATERN = "matern"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    CUSTOM = "custom"


class ModelType(str, Enum):
    """Enum for surrogate model types."""
    GP = "gp"
    RF = "rf"
    SVM = "svm"
    MLP = "mlp"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class TaskStatus(str, Enum):
    """Enum for optimization task status."""
    PENDING = "pending"  # initial state, waiting for configuration
    CONFIGURED = "configured"  # parameter space is configured
    GENERATING_INITIAL = "generating_initial"  # generating initial experimental points
    READY_FOR_RESULTS = "ready_for_results"  # waiting for experimental results
    OPTIMIZING = "optimizing"  # executing optimization loop
    PAUSED = "paused"  # paused
    COMPLETED = "completed"  # completed
    FAILED = "failed"  # failed


class TaskStatusResponse(BaseModel):
    """Model for task status response."""
    status: TaskStatus = Field(..., description="Task status")
    current_iteration: Optional[int] = Field(None, description="Current iteration")
    total_evaluations: Optional[int] = Field(None, description="Total evaluations")
    message: Optional[str] = Field(None, description="Status message")
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        """Validate task status is one of the allowed values."""
        allowed = [status.value for status in TaskStatus]
        if v not in allowed:
            raise ValueError(f"Status must be one of: {', '.join(allowed)}")
        return v


# Parameter Space Models

class ParameterConfig(BaseModel):
    """Configuration model for a single parameter."""
    name: str = Field(..., description="Unique name of the parameter")
    type: ParameterType = Field(..., description="Type of the parameter")
    bounds: Optional[List[Union[float, int]]] = Field(None, description="Bounds [min, max] for continuous or integer parameters")
    categories: Optional[List[Any]] = Field(None, description="List of possible values for categorical parameters")
    log_scale: Optional[bool] = Field(False, description="Whether the parameter should be treated on a log scale")
    precision: Optional[int] = Field(None, description="Number of decimal places for continuous parameters")

    @model_validator(mode='after')
    def check_parameter_consistency(self):
        """validate the consistency of parameter configuration"""
        if self.type == ParameterType.CONTINUOUS or self.type == ParameterType.INTEGER:
            if self.bounds is None:
                raise ValueError(f"parameter '{self.name}' type is '{self.type}', must specify 'bounds'")
            if len(self.bounds) != 2:
                raise ValueError(f"parameter '{self.name}' 'bounds' must contain exactly two elements [min, max]")
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError(f"parameter '{self.name}' 'bounds' must have min < max")
            if self.type == ParameterType.INTEGER:
                if not all(isinstance(b, int) for b in self.bounds):
                    raise ValueError(f"parameter '{self.name}' type is '{self.type}', must be integer")
            if self.type == ParameterType.CONTINUOUS:
                if not all(isinstance(b, (int, float)) for b in self.bounds):
                    raise ValueError(f"parameter '{self.name}' type is '{self.type}', must be numeric")

            if self.categories is not None:
                raise ValueError(f"parameter '{self.name}' type is '{self.type}', should not define 'categories'")

        elif self.type == ParameterType.CATEGORICAL:
            if self.categories is None:
                raise ValueError(f"parameter '{self.name}' type is '{self.type}', must specify 'categories'")
            if not self.categories:
                raise ValueError(f"parameter '{self.name}' 'categories' list cannot be empty")
            if self.bounds is not None:
                raise ValueError(f"parameter '{self.name}' type is '{self.type}', should not define 'bounds'")
            if self.log_scale:
                raise ValueError(f"parameter '{self.name}' type is '{self.type}', categorical parameters cannot use log scale")
            if self.precision is not None:
                raise ValueError(f"parameter '{self.name}' type is '{self.type}', categorical parameters do not use precision setting")

        return self


class ObjectiveConfig(BaseModel):
    """Configuration model for a single objective."""
    name: str = Field(..., description="Unique name of the objective")
    direction: ObjectiveDirection = Field(..., description="Optimization direction (minimize or maximize)")
    target: Optional[float] = Field(None, description="Optional target value for the objective")


class ConstraintConfig(BaseModel):
    """Configuration model for a single constraint."""
    name: Optional[str] = Field(None, description="Optional unique name for the constraint")
    parameters: List[str] = Field(..., description="List of parameter names involved in the constraint")
    relation: ConstraintRelation = Field(..., description="Constraint relation (e.g., <=, >=, ==)")
    threshold: float = Field(..., description="Threshold value for the constraint")

    @model_validator(mode='after')
    def check_constraint_definition(self):
        """validate the constraint definition"""
        if not self.parameters:
            raise ValueError("constraint must involve at least one parameter")
        return self


class ParameterSpaceConfig(BaseModel):
    """declarative configuration model for the entire parameter space, as input for the API"""
    name: str = Field(..., description="name of the optimization task")
    parameters: List[ParameterConfig] = Field(..., description="list of parameter configurations")
    objectives: List[ObjectiveConfig] = Field(..., description="list of objective configurations")
    constraints: Optional[List[ConstraintConfig]] = Field(None, description="optional list of constraint configurations")
    description: Optional[str] = Field(None, description="参数空间的可选描述")

    @model_validator(mode='after')
    def check_names_uniqueness_and_references(self):
        """validate the uniqueness of names and the validity of references"""
        param_names = [p.name for p in self.parameters]
        obj_names = [o.name for o in self.objectives]

        # check for duplicate parameter names
        if len(param_names) != len(set(param_names)):
            raise ValueError("duplicate parameter names found")

        # check for duplicate objective names
        if len(obj_names) != len(set(obj_names)):
            raise ValueError("duplicate objective names found")

        # check for duplicate constraint names
        if self.constraints:
            constraint_names = [c.name for c in self.constraints if c.name]
            if len(constraint_names) != len(set(constraint_names)):
                raise ValueError("duplicate constraint names found")

            all_param_names = set(param_names)
            for i, constraint in enumerate(self.constraints):
                for param_name in constraint.parameters:
                    if param_name not in all_param_names:
                        constraint_id = constraint.name if constraint.name else f"index {i}"
                        raise ValueError(f"constraint '{constraint_id}' references undefined parameter '{param_name}'")
        return self


class ParameterSpaceCreateResponse(BaseModel):
    """response model after creating the parameter space"""
    task_id: str = Field(..., description="ID of the created optimization task")
    message: str = Field("Parameter space configured successfully", description="operation message")


class ParameterSpaceReadResponse(ParameterSpaceConfig):
    """model for reading the parameter space configuration"""
    task_id: str = Field(..., description="ID of the optimization task that this configuration belongs to")
    created_at: Optional[datetime] = Field(None, description="task creation time")
    updated_at: Optional[datetime] = Field(None, description="configuration last update time")


# Strategy Models

class StrategyCreate(BaseModel):
    """model for creating the optimization strategy"""
    model_type: str = Field("gp", description="proxy model type (e.g., 'gp', 'rf')")
    acquisition_type: str = Field("ei", description="acquisition function type (e.g., 'ei', 'ucb')")
    batch_size: Optional[int] = Field(1, description="number of points to suggest per batch", ge=1)
    n_initial_points: Optional[int] = Field(10, description="number of initial design points to generate", ge=1)
    initial_design_type: Optional[str] = Field("lhs", description="initial design type (e.g., 'lhs', 'random', 'sobol')")
    settings: Optional[Dict[str, Any]] = Field({}, description="other strategy settings")


class StrategyReadResponse(StrategyCreate):
    """model for reading the optimization strategy"""
    task_id: str = Field(..., description="ID of the optimization task that this strategy belongs to")
    created_at: Optional[datetime] = Field(None, description="strategy creation time")
    updated_at: Optional[datetime] = Field(None, description="strategy last update time")


# Design Models

class DesignParameters(BaseModel):
    """model for design parameters"""
    parameters: Dict[str, Any] = Field(..., description="design parameters")


class DesignMetadata(BaseModel):
    """model for experimental metadata"""
    timestamp: Optional[datetime] = Field(None, description="experiment timestamp")
    experimenter: Optional[str] = Field(None, description="experimenter name")
    notes: Optional[str] = Field(None, description="notes about the experiment")


class Prediction(BaseModel):
    """model for prediction with uncertainty"""
    mean: float = Field(..., description="average prediction value")
    std: float = Field(..., description="prediction standard deviation")


class PredictionResponse(BaseModel):
    """model for prediction response"""
    predictions: Dict[str, Prediction] = Field(..., description="predictions for each objective")


class Design(BaseModel):
    """model for experimental design point"""
    id: str = Field(..., description="unique identifier of the design")
    parameters: Dict[str, Any] = Field(..., description="design parameters")
    predictions: Optional[Dict[str, Prediction]] = Field(None, description="predictions")
    uncertainty: Optional[float] = Field(None, description="overall uncertainty measure")
    reason: Optional[str] = Field(None, description="reason for recommending this design")


class DesignResponse(BaseModel):
    """model for design response"""
    designs: List[Design] = Field(..., description="list of design points")


# Result Models

class ResultSubmit(BaseModel):
    """model for submitting results"""
    parameters: Dict[str, Any] = Field(..., description="parameter values")
    objectives: Dict[str, float] = Field(..., description="objective values")
    constraints: Optional[Dict[str, float]] = Field({}, description="constraint values")
    metadata: Optional[Dict[str, Any]] = Field(None, description="additional metadata")
    
    @model_validator(mode='after')
    def validate_result(self):
        """validate the result contains all required parameters and objectives"""
        # 这里可以实现额外的验证
        return self


class ResultsSubmission(BaseModel):
    """model for batch submitting results"""
    results: List[ResultSubmit] = Field(..., description="list of experimental results")


# Prediction Models

class PredictionRequest(BaseModel):
    """model for prediction request"""
    parameters: Dict[str, Any] = Field(..., description="parameter values to predict")
    
    @model_validator(mode='after')
    def validate_parameters(self):
        """validate the parameters exist"""
        # 这是额外验证的占位符
        return self


class ModelPerformance(BaseModel):
    """model for current model performance"""
    model_type: str = Field(..., description="used proxy model type")
    metrics: Dict[str, float] = Field(..., description="performance metrics (e.g., R², MAE, RMSE)")
    cross_validation_results: Optional[Dict[str, List[float]]] = Field(None, description="cross-validation results")
    
    @field_validator('metrics')
    @classmethod
    def validate_metrics(cls, v):
        """validate the metrics contain required fields"""
        required_metrics = ["r2", "mae", "rmse"]
        for metric in required_metrics:
            if metric not in v:
                raise ValueError(f"missing required metric '{metric}'")
        return v


# Pareto Front Models

class ParetoSolution(BaseModel):
    """model for Pareto front solution"""
    id: str = Field(..., description="design ID")
    parameters: Dict[str, Any] = Field(..., description="parameter values")
    objectives: Dict[str, float] = Field(..., description="objective values")
    uncertainty: Optional[float] = Field(None, description="uncertainty measure")


class ParetoFront(BaseModel):
    """model for Pareto front"""
    points: List[Dict[str, Any]] = Field(..., description="list of non-dominated points")
    objective_names: List[str] = Field(..., description="names of objectives used for Pareto front")
    
    @model_validator(mode='after')
    def validate_points(self):
        """validate the points contain all objective values"""
        for point in self.points:
            for obj in self.objective_names:
                if obj not in point:
                    raise ValueError(f"point missing objective value '{obj}'")
        return self


# Uncertainty Analysis Models

class UncertaintyAnalysis(BaseModel):
    """model for uncertainty analysis results"""
    parameter_name: str = Field(..., description="name of the analyzed parameter")
    uncertainty_values: List[float] = Field(..., description="uncertainty values in the parameter range")
    parameter_values: List[float] = Field(..., description="parameter values used for uncertainty evaluation")
    
    @model_validator(mode='after')
    def validate_data_length(self):
        """validate the uncertainty and parameter values have the same length"""
        if len(self.uncertainty_values) != len(self.parameter_values):
            raise ValueError("uncertainty_values and parameter_values must have the same length")
        return self


# Task Management Models

class TaskBasicInfo(BaseModel):
    """model for basic task information"""
    task_id: str = Field(..., description="task ID")
    name: str = Field(..., description="task name")
    status: TaskStatus = Field(..., description="task status")
    created_at: datetime = Field(..., description="creation timestamp")
    updated_at: datetime = Field(..., description="last update timestamp")
    progress: float = Field(0.0, ge=0, le=100, description="progress percentage")


class TaskListResponse(BaseModel):
    """model for task list response"""
    tasks: List[TaskBasicInfo] = Field(..., description="list of all optimization tasks")
    count: int = Field(..., description="total number of tasks")


class TaskDetails(BaseModel):
    """model for detailed task information response"""
    task_id: str = Field(..., description="task ID")
    name: str = Field(..., description="task name") 
    status: TaskStatus = Field(..., description="current task status")
    created_at: datetime = Field(..., description="task creation time")
    updated_at: datetime = Field(..., description="last update time")
    progress: float = Field(..., ge=0, le=100, description="progress percentage")
    parameter_space: Optional[ParameterSpaceReadResponse] = Field(None, description="parameter space definition")
    strategy: Optional[StrategyReadResponse] = Field(None, description="optimization strategy")
    initial_designs: Optional[List[Design]] = Field(None, description="initial design points")
    results: Optional[List[ResultSubmit]] = Field(None, description="submitted results")
    next_points: Optional[List[Design]] = Field(None, description="next recommended points")
    
    @model_validator(mode='after')
    def check_task_consistency(self):
        """ensure the task details are consistent with the task status"""
        # 已配置的任务应该有参数空间定义
        if self.status == TaskStatus.CONFIGURED and not self.parameter_space:
            raise ValueError("configured task must have parameter space definition")
        
        # 优化中的任务应该有策略和初始设计
        if self.status == TaskStatus.OPTIMIZING:
            if not self.strategy:
                raise ValueError("optimization task must have defined strategy")
            if not self.initial_designs:
                raise ValueError("optimization task must have generated initial designs")
        
        # 已完成的任务应该有结果
        if self.status == TaskStatus.COMPLETED and not self.results:
            raise ValueError("completed task must have submitted results")
        
        return self


class TaskStatusUpdateResponse(BaseModel):
    """model for detailed task status response"""
    status: str = Field(..., description="task status")
    current_iteration: Optional[int] = Field(None, description="current iteration")
    total_iterations: Optional[int] = Field(None, description="total iterations")
    progress: float = Field(..., ge=0, le=100, description="progress percentage")
    last_updated: datetime = Field(..., description="last update timestamp")
    message: Optional[str] = Field(None, description="Status message")
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        """Validate task status is one of the allowed values."""
        allowed = [status.value for status in TaskStatus]
        if v not in allowed:
            raise ValueError(f"Status must be one of: {', '.join(allowed)}")
        return v


# Task Restart Model

class TaskRestart(BaseModel):
    """Model for task restart request."""
    strategy: str = Field(..., description="Restart strategy (reuse_last or reset)")
    preserve_history: bool = Field(True, description="Whether to preserve experiment history")


# Diagnostics Model

class Diagnostics(BaseModel):
    """Model for diagnostics response."""
    parameter_space: str = Field(..., description="Parameter space status")
    model_trained: bool = Field(..., description="Whether model is trained")
    recent_exception: Optional[str] = Field(None, description="Recent exception if any")
    pending_experiments: List[str] = Field(..., description="List of pending experiment IDs")
    last_recommendation_time: Optional[datetime] = Field(None, description="Timestamp of last recommendation")


class TaskCreate(BaseModel):
    """Model for creating a new optimization task."""
    name: str = Field(..., description="Name of the task")
    description: Optional[str] = Field(None, description="Description of the task")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate that the name is not empty."""
        if not v or v.strip() == "":
            raise ValueError("name cannot be empty")
        return v


class DiagnosticsResponse(BaseModel):
    """Model for system diagnostics response."""
    system_health: str = Field(..., description="Overall system health")
    cpu_usage: float = Field(..., ge=0, le=100, description="Current CPU usage percentage")
    memory_usage: float = Field(..., ge=0, description="Current memory usage in MB")
    active_tasks: int = Field(..., ge=0, description="Number of active tasks")
    storage_usage: float = Field(..., ge=0, description="Storage usage in MB")
    
    @field_validator('system_health')
    @classmethod
    def validate_system_health(cls, v):
        """Validate system health is one of the allowed values."""
        allowed = ["healthy", "warning", "critical"]
        if v not in allowed:
            raise ValueError(f"System health must be one of: {', '.join(allowed)}")
        return v
        
    @model_validator(mode='after')
    def validate_health_consistency(self):
        """Validate system health is consistent with resource usage."""
        cpu_threshold = 90
        memory_threshold = 85
        
        if self.cpu_usage > cpu_threshold or self.memory_usage > memory_threshold:
            if self.system_health == "healthy":
                raise ValueError("System health cannot be 'healthy' when resource usage exceeds thresholds")
                
        return self
 