from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import RootModel
from datetime import datetime
import re
from bson import ObjectId as BsonObjectId

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
    DISCRETE = "discrete"


class ObjectiveType(str, Enum):
    """Enum for objective types."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ConstraintType(str, Enum):
    """Enum for constraint types."""
    SUM_EQUALS = "sum_equals"
    SUM_LESS_THAN = "sum_less_than"
    SUM_GREATER_THAN = "sum_greater_than"
    PRODUCT_EQUALS = "product_equals"
    CUSTOM = "custom"


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
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatusResponse(BaseModel):
    """Model for task status response."""
    id: str = Field(..., description="Task ID")
    name: str = Field(..., description="Task name")
    status: str = Field(..., description="Current status of the task")
    created_at: datetime = Field(..., description="Task creation time")
    updated_at: datetime = Field(..., description="Last update time")
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        """Validate task status is one of the allowed values."""
        allowed = [status.value for status in TaskStatus]
        if v not in allowed:
            raise ValueError(f"Status must be one of: {', '.join(allowed)}")
        return v


# Parameter Space Models

class ParameterBase(BaseModel):
    """Base model for all parameter types."""
    name: str = Field(..., description="Name of the parameter")
    type: ParameterType = Field(..., description="Type of the parameter")


class ContinuousParameter(ParameterBase):
    """Model for continuous parameters."""
    type: Literal[ParameterType.CONTINUOUS] = ParameterType.CONTINUOUS
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    
    @field_validator('max')
    @classmethod
    def validate_min_max(cls, v, info):
        """Validate that max is greater than min."""
        data = info.data
        if 'min' in data and v <= data['min']:
            raise ValueError("max must be greater than min")
        return v


class DiscreteParameter(ParameterBase):
    """Model for discrete parameters."""
    type: Literal[ParameterType.DISCRETE] = ParameterType.DISCRETE
    min: int = Field(..., description="Minimum value")
    max: int = Field(..., description="Maximum value")
    step: int = Field(1, description="Step size")
    
    @field_validator('max')
    @classmethod
    def validate_min_max(cls, v, info):
        """Validate that max is greater than min."""
        data = info.data
        if 'min' in data and v <= data['min']:
            raise ValueError("max must be greater than min")
        return v
    
    @field_validator('step')
    @classmethod
    def validate_step(cls, v):
        """Validate that step is positive."""
        if v <= 0:
            raise ValueError("step must be positive")
        return v


class CategoricalParameter(ParameterBase):
    """Model for categorical parameters."""
    type: Literal[ParameterType.CATEGORICAL] = ParameterType.CATEGORICAL
    values: List[Any] = Field(..., description="Possible values")
    
    @field_validator('values')
    @classmethod
    def validate_values(cls, v):
        """Validate that values is not empty."""
        if len(v) == 0:
            raise ValueError("values must not be empty")
        return v


class ObjectiveBase(BaseModel):
    """Base model for objectives."""
    name: str = Field(..., description="Name of the objective")
    type: ObjectiveType = Field(..., description="Type of the objective")
    bounds: Optional[List[float]] = Field(None, description="Bounds for the objective")
    
    @field_validator('bounds')
    @classmethod
    def validate_bounds(cls, v):
        """Validate that bounds is a list of two floats."""
        if v is not None:
            if len(v) != 2:
                raise ValueError("bounds must be a list of two values")
            if v[0] >= v[1]:
                raise ValueError("lower bound must be less than upper bound")
        return v


class ConstraintBase(BaseModel):
    """Base model for constraints."""
    name: str = Field(..., description="Name of the constraint")
    type: ConstraintType = Field(..., description="Type of the constraint")
    threshold: float = Field(..., description="Threshold value for the constraint")


class ParameterSpaceCreate(BaseModel):
    """Model for creating a parameter space."""
    name: str = Field(..., description="Name of the optimization task")
    parameters: Dict[str, Union[ContinuousParameter, DiscreteParameter, CategoricalParameter]] = Field(
        ..., description="Dictionary of parameters"
    )
    objectives: Dict[str, ObjectiveBase] = Field(..., description="Dictionary of objectives")
    constraints: Dict[str, ConstraintBase] = Field({}, description="Dictionary of constraints")
    
    @model_validator(mode='after')
    def validate_parameter_space(self):
        """Validate that the parameter space has at least one parameter and one objective."""
        if len(self.parameters) == 0:
            raise ValueError("parameter space must have at least one parameter")
        if len(self.objectives) == 0:
            raise ValueError("parameter space must have at least one objective")
        return self


class ParameterSpaceCreationResponse(BaseModel):
    """Response model after creating a parameter space."""
    task_id: str = Field(..., description="ID of the created optimization task")
    status: str = Field(..., description="Status of the task")
    message: str = Field(..., description="Message about the operation")
    
    @field_validator('status')
    @classmethod 
    def validate_status(cls, v):
        """Validate task status is one of the allowed values."""
        allowed = [status.value for status in TaskStatus]
        if v not in allowed:
            raise ValueError(f"Status must be one of: {', '.join(allowed)}")
        return v


class ParameterSpaceRead(ParameterSpaceCreate):
    """Model for reading a parameter space."""
    task_id: str = Field(..., description="ID of the optimization task")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


# Strategy Models

class InitialSampling(BaseModel):
    """Model for initial sampling configuration."""
    method: SamplingMethod = Field(..., description="Sampling method")
    samples: int = Field(..., description="Number of initial samples")


class StrategyConfig(BaseModel):
    """Model for advanced strategy configuration."""
    acquisition_function: AcquisitionFunction = Field(..., description="Acquisition function")
    kernel: Optional[KernelType] = Field(None, description="Kernel for GP models")
    exploration_weight: Optional[float] = Field(None, description="Exploration weight parameter")
    noise_level: Optional[float] = Field(None, description="Assumed noise level")
    multi_objective: Optional[bool] = Field(False, description="Enable multi-objective optimization")
    moo_acquisition: Optional[str] = Field(None, description="Multi-objective acquisition function")
    noisy_moo: Optional[bool] = Field(False, description="Handle noisy evaluations in MOO")


class StrategyCreate(BaseModel):
    """Model for creating an optimization strategy."""
    algorithm: str = Field(..., description="Algorithm to use for optimization")
    settings: Dict[str, Any] = Field({}, description="Additional settings for the algorithm")
    acquisition_function: str = Field("EI", description="Acquisition function to use")
    batch_size: int = Field(1, description="Number of points to suggest at once")
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        """Validate that batch_size is positive."""
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v


class StrategyRead(StrategyCreate):
    """Model for reading an optimization strategy."""
    task_id: str = Field(..., description="ID of the task this strategy belongs to")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


# Design Models

class DesignParameters(RootModel):
    """Model for design parameters."""
    root: Dict[str, Any] = Field(..., description="Parameter values for this design")


class DesignMetadata(BaseModel):
    """Model for experiment metadata."""
    timestamp: Optional[datetime] = Field(None, description="Timestamp of the experiment")
    experimenter: Optional[str] = Field(None, description="Name of the experimenter")
    notes: Optional[str] = Field(None, description="Notes about the experiment")


class Prediction(BaseModel):
    """Model for prediction output with uncertainty."""
    mean: float = Field(..., description="Mean prediction")
    std: float = Field(..., description="Standard deviation of prediction")


class Predictions(RootModel):
    """Model for objective predictions."""
    root: Dict[str, Prediction] = Field(..., description="Predictions for each objective")


class Design(BaseModel):
    """Model for an experiment design point."""
    id: str = Field(..., description="Unique identifier for this design")
    parameters: Dict[str, Any] = Field(..., description="Parameter values for this design")
    predictions: Optional[Dict[str, Prediction]] = Field(None, description="Predicted outcomes")
    uncertainty: Optional[float] = Field(None, description="Overall uncertainty measure")
    reason: Optional[str] = Field(None, description="Reason this design was recommended")


class DesignResponse(BaseModel):
    """Response model for design endpoints."""
    designs: List[Design] = Field(..., description="List of design points")


# Result Models

class ObjectiveResults(RootModel):
    """Model for objective function results."""
    root: Dict[str, float] = Field(..., description="Values for each objective")


class ResultSubmit(BaseModel):
    """Model for submitting results."""
    parameters: Dict[str, Any] = Field(..., description="Parameter values")
    objectives: Dict[str, float] = Field(..., description="Objective values")
    constraints: Dict[str, float] = Field({}, description="Constraint values")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @model_validator(mode='after')
    def validate_result(self):
        """Validate that the result has all required parameters and objectives."""
        # Additional validation can be implemented here
        return self


class ResultsSubmission(BaseModel):
    """Model for batch submission of results."""
    results: List[ResultSubmit] = Field(..., description="List of experiment results")


# Prediction Models

class PredictionRequest(BaseModel):
    """Model for prediction requests."""
    parameters: Dict[str, Any] = Field(..., description="Parameter values to predict for")
    
    @model_validator(mode='after')
    def validate_parameters(self):
        """Validate that all required parameters are present."""
        # This is a placeholder for additional validation
        return self


class PredictionResponse(BaseModel):
    """Response model for prediction requests."""
    predictions: List[Dict] = Field(..., description="Predictions for each parameter combination")


# Model Performance Models

class ModelMetrics(BaseModel):
    """Model for regression performance metrics."""
    r2: float = Field(..., description="R-squared score")
    rmse: float = Field(..., description="Root mean squared error")
    mae: float = Field(..., description="Mean absolute error")


class CrossValidation(BaseModel):
    """Model for cross-validation results."""
    cv_scores: List[float] = Field(..., description="Cross-validation scores")
    mean_score: float = Field(..., description="Mean of cross-validation scores")
    std_score: float = Field(..., description="Standard deviation of cross-validation scores")


class ModelPerformance(BaseModel):
    """Model for the current model performance."""
    model_type: str = Field(..., description="Type of surrogate model used")
    metrics: Dict[str, float] = Field(..., description="Performance metrics (e.g., R², MAE, RMSE)")
    cross_validation_results: Optional[Dict[str, List[float]]] = Field(None, description="Cross-validation results")
    
    @field_validator('metrics')
    @classmethod
    def validate_metrics(cls, v):
        """Validate that metrics contains required fields."""
        required_metrics = ["r2", "mae", "rmse"]
        for metric in required_metrics:
            if metric not in v:
                raise ValueError(f"Required metric '{metric}' missing")
        return v


# Pareto Front Models

class ParetoSolution(BaseModel):
    """Model for a solution on the Pareto front."""
    id: str = Field(..., description="Design ID")
    parameters: Dict[str, Any] = Field(..., description="Parameter values")
    objectives: Dict[str, float] = Field(..., description="Objective values")
    uncertainty: Optional[float] = Field(None, description="Uncertainty measure")


class ParetoFront(BaseModel):
    """Model for the current Pareto front."""
    points: List[Dict[str, Any]] = Field(..., description="List of non-dominated points")
    objective_names: List[str] = Field(..., description="Names of objectives used for Pareto front")
    
    @model_validator(mode='after')
    def validate_points(self):
        """Validate that points contain values for all objectives."""
        for point in self.points:
            for obj in self.objective_names:
                if obj not in point:
                    raise ValueError(f"Point missing value for objective '{obj}'")
        return self


# Uncertainty Analysis Models

class PredictionActual(BaseModel):
    """Model for prediction vs actual comparison."""
    design_id: str = Field(..., description="Design ID")
    predicted: Prediction = Field(..., description="Predicted value with uncertainty")
    actual: float = Field(..., description="Actual measured value")
    error: float = Field(..., description="Absolute error")
    within_confidence: bool = Field(..., description="Whether actual is within confidence interval")


class CalibrationMetrics(BaseModel):
    """Model for uncertainty calibration metrics."""
    coverage_probability: float = Field(..., description="Empirical coverage probability")
    sharpness: float = Field(..., description="Sharpness of predictions")


class UncertaintyAnalysis(BaseModel):
    """Model for uncertainty analysis results."""
    parameter_name: str = Field(..., description="Name of the parameter analyzed")
    uncertainty_values: List[float] = Field(..., description="Uncertainty values across the parameter range")
    parameter_values: List[float] = Field(..., description="Parameter values at which uncertainty was evaluated")
    
    @model_validator(mode='after')
    def validate_data_length(self):
        """Validate that uncertainty and parameter values have same length."""
        if len(self.uncertainty_values) != len(self.parameter_values):
            raise ValueError("uncertainty_values and parameter_values must have the same length")
        return self


# Task Management Models

class TaskInfo(BaseModel):
    """Basic task information model."""
    task_id: str = Field(..., description="Task ID")
    name: str = Field(..., description="Task name")
    status: TaskStatusResponse = Field(..., description="Task status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class TaskList(BaseModel):
    """Model for task listing response."""
    tasks: List[TaskInfo] = Field(..., description="List of tasks")


class TaskListResponse(BaseModel):
    """Model for task list response."""
    tasks: List[Dict] = Field(..., description="List of all optimization tasks")
    count: int = Field(..., description="Total number of tasks")


class TaskSummary(BaseModel):
    """Model for task summary."""
    task_id: str = Field(..., description="Task ID")
    name: str = Field(..., description="Task name")
    status: str = Field(..., description="Task status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        """Validate task status is one of the allowed values."""
        allowed = [status.value for status in TaskStatus]
        if v not in allowed:
            raise ValueError(f"Status must be one of: {', '.join(allowed)}")
        return v


class TaskDetails(BaseModel):
    """Detailed task information response model."""
    task_id: str = Field(..., description="Task ID")
    name: str = Field(..., description="Task name") 
    status: str = Field(..., description="Current status of the task")
    created_at: datetime = Field(..., description="Task creation time")
    updated_at: datetime = Field(..., description="Last update time")
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    parameter_space: Optional[ParameterSpaceRead] = Field(None, description="Parameter space definition")
    strategy: Optional[Dict] = Field(None, description="Optimization strategy")
    initial_designs: Optional[List[Dict]] = Field(None, description="Initial design points")
    results: Optional[List[Dict]] = Field(None, description="Submitted results")
    next_points: Optional[List[Dict]] = Field(None, description="Next recommended points")
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        """Validate task status is one of the allowed values."""
        allowed = [status.value for status in TaskStatus]
        if v not in allowed:
            raise ValueError(f"Status must be one of: {', '.join(allowed)}")
        return v
    
    @model_validator(mode='after')
    def check_task_consistency(self):
        """Ensure task details are consistent with the task status."""
        # Created tasks should have parameter_space defined
        if self.status == TaskStatus.CREATED.value and not self.parameter_space:
            raise ValueError("Created tasks must have a parameter space defined")
        
        # Running tasks should have strategy, initial_designs
        if self.status == TaskStatus.RUNNING.value:
            if not self.strategy:
                raise ValueError("Running tasks must have a strategy defined")
            if not self.initial_designs:
                raise ValueError("Running tasks must have initial designs generated")
        
        # Completed tasks should have results
        if self.status == TaskStatus.COMPLETED.value and not self.results:
            raise ValueError("Completed tasks must have results submitted")
        
        return self


class TaskStatusUpdateResponse(BaseModel):
    """Detailed task status response model."""
    status: str = Field(..., description="Task status")
    current_iteration: Optional[int] = Field(None, description="Current iteration")
    total_iterations: Optional[int] = Field(None, description="Total iterations")
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    last_updated: datetime = Field(..., description="Last update timestamp")
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
 