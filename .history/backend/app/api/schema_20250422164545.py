from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
from pydantic import RootModel
from datetime import datetime
import re

# Common Base Models

class ObjectId(BaseModel):
    """Model for MongoDB ObjectId serialization/deserialization."""
    id: str

    @validator('id')
    def validate_id(cls, v):
        """Validate that id is in proper format."""
        if not re.match(r'^[a-f0-9]{24}$', v):
            raise ValueError("ObjectId must be a 24-character hex string")
        return v


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
    
    @validator('max')
    def validate_min_max(cls, v, values):
        """Validate that max is greater than min."""
        if 'min' in values and v <= values['min']:
            raise ValueError("max must be greater than min")
        return v


class DiscreteParameter(ParameterBase):
    """Model for discrete parameters."""
    type: Literal[ParameterType.DISCRETE] = ParameterType.DISCRETE
    min: int = Field(..., description="Minimum value")
    max: int = Field(..., description="Maximum value")
    step: int = Field(1, description="Step size")
    
    @validator('max')
    def validate_min_max(cls, v, values):
        """Validate that max is greater than min."""
        if 'min' in values and v <= values['min']:
            raise ValueError("max must be greater than min")
        return v
    
    @validator('step')
    def validate_step(cls, v):
        """Validate that step is positive."""
        if v <= 0:
            raise ValueError("step must be positive")
        return v


class CategoricalParameter(ParameterBase):
    """Model for categorical parameters."""
    type: Literal[ParameterType.CATEGORICAL] = ParameterType.CATEGORICAL
    values: List[Any] = Field(..., description="Possible values")
    
    @validator('values')
    def validate_values(cls, v):
        """Validate that values is not empty."""
        if len(v) == 0:
            raise ValueError("values must not be empty")
        return v


class Objective(BaseModel):
    """Model for objective function."""
    name: str = Field(..., description="Name of the objective")
    type: ObjectiveType = Field(..., description="Type of the objective (maximize/minimize)")


class Constraint(BaseModel):
    """Model for constraints."""
    expression: str = Field(..., description="Expression for the constraint")
    type: ConstraintType = Field(..., description="Type of the constraint")
    value: float = Field(..., description="Target value for the constraint")


# Parameter Space Request/Response Models

class ParameterSpaceCreate(BaseModel):
    """Model for creating a parameter space."""
    name: str = Field(..., description="Name of the optimization task")
    parameters: List[Union[ContinuousParameter, DiscreteParameter, CategoricalParameter]] = Field(
        ..., description="List of parameters defining the search space"
    )
    objectives: List[Objective] = Field(..., description="List of objectives")
    constraints: Optional[List[Constraint]] = Field(None, description="List of constraints")


class ParameterSpaceResponse(BaseModel):
    """Response model after creating a parameter space."""
    task_id: str = Field(..., description="ID of the created optimization task")
    status: TaskStatus = Field(..., description="Status of the task")
    message: str = Field(..., description="Message about the operation")


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
    algorithm: str = Field(..., description="Optimization algorithm")
    config: StrategyConfig = Field(..., description="Algorithm configuration")
    initial_sampling: InitialSampling = Field(..., description="Initial sampling strategy")
    batch_size: Optional[int] = Field(1, description="Batch size for recommendations")
    iterations: Optional[int] = Field(None, description="Maximum number of iterations")


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


class ResultCreate(BaseModel):
    """Model for submitting experiment results."""
    design_id: str = Field(..., description="ID of the design point")
    objectives: Dict[str, float] = Field(..., description="Measured objective values")
    metadata: Optional[DesignMetadata] = Field(None, description="Additional metadata")


class ResultsSubmission(BaseModel):
    """Model for batch submission of results."""
    results: List[ResultCreate] = Field(..., description="List of experiment results")


# Prediction Models

class PredictionRequest(BaseModel):
    """Model for requesting predictions for new points."""
    parameters: List[Dict[str, Any]] = Field(..., description="Parameter combinations to predict")


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
    """Model for overall model performance."""
    metrics: ModelMetrics = Field(..., description="Performance metrics")
    cross_validation: Optional[CrossValidation] = Field(None, description="Cross-validation results")


# Pareto Front Models

class ParetoSolution(BaseModel):
    """Model for a solution on the Pareto front."""
    id: str = Field(..., description="Design ID")
    parameters: Dict[str, Any] = Field(..., description="Parameter values")
    objectives: Dict[str, float] = Field(..., description="Objective values")
    uncertainty: Optional[float] = Field(None, description="Uncertainty measure")


class ParetoFront(BaseModel):
    """Model for the Pareto front response."""
    pareto_front: List[ParetoSolution] = Field(..., description="Solutions on the Pareto front")
    dominated_solutions: List[ParetoSolution] = Field(..., description="Dominated solutions")
    ideal_point: Dict[str, float] = Field(..., description="Ideal point (best possible for each objective)")
    nadir_point: Dict[str, float] = Field(..., description="Nadir point (worst values on Pareto front)")


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
    """Model for uncertainty analysis response."""
    prediction_vs_actual: List[PredictionActual] = Field(..., description="Prediction vs actual comparisons")
    calibration_metrics: CalibrationMetrics = Field(..., description="Calibration metrics")


# Task Management Models

class TaskInfo(BaseModel):
    """Basic task information model."""
    task_id: str = Field(..., description="Task ID")
    name: str = Field(..., description="Task name")
    status: TaskStatus = Field(..., description="Task status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class TaskList(BaseModel):
    """Model for task listing response."""
    tasks: List[TaskInfo] = Field(..., description="List of tasks")


class TaskStatusResponse(BaseModel):
    """Detailed task status response model."""
    status: TaskStatus = Field(..., description="Task status")
    current_iteration: Optional[int] = Field(None, description="Current iteration")
    total_iterations: Optional[int] = Field(None, description="Total iterations")
    best_result: Optional[Dict] = Field(None, description="Best result so far")
    last_updated: datetime = Field(..., description="Last update timestamp")


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
