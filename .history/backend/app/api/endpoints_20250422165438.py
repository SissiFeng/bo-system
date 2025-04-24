from fastapi import APIRouter, HTTPException, Depends, Query, Path, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Any, Optional
import uuid
import json
import time
from datetime import datetime
import logging
import os
from pathlib import Path as PathLib
import asyncio
import numpy as np

from app.core.config import get_settings
from app.core.logger import setup_logger
from app.api import schema
from bo_engine.parameter_space import ParameterSpace
from bo_engine.design_generator import create_design_generator, DesignType
from bo_engine.utils import generate_unique_id
from bo_engine.optimizer import BayesianOptimizer
from bo_engine.models import GaussianProcessModel
from bo_engine.acquisition import ExpectedImprovement

# Import core BO engine components
# These will be implemented in Phase 2-4
# from bo_engine.parameter_space import ParameterSpace
# from bo_engine.design_generator import DesignGenerator
# from bo_engine.optimizer import Optimizer

# Setup
settings = get_settings()
logger = setup_logger("api")

# Create router
router = APIRouter()

# In-memory storage for tasks (temporary, will be replaced with proper persistence)
# In a production system, this would be a database
tasks = {}
parameter_spaces = {}
strategies = {}
designs = {}
results = {}
optimizers: Dict[str, BayesianOptimizer] = {}

# Utility functions
def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())

def generate_error_response(status_code: int, message: str) -> JSONResponse:
    """Generate an error response."""
    return JSONResponse(
        status_code=status_code,
        content={"error": message},
    )

# ----- Helper Function to Get/Create Optimizer -----

def get_or_create_optimizer(task_id: str) -> BayesianOptimizer:
    """Gets the optimizer instance for a task_id, creating it if it doesn't exist."""
    if task_id in optimizers:
        logger.debug(f"Optimizer instance found in memory for task {task_id}.")
        return optimizers[task_id]

    logger.info(f"Optimizer instance not found for task {task_id}. Creating new instance.")
    task_dir = PathLib(settings.TASK_DIR) / task_id

    # 1. Load Parameter Space
    ps_file = task_dir / "parameter_space.json"
    if not ps_file.exists():
        logger.error(f"Cannot create optimizer: Parameter space file not found at {ps_file}")
        raise HTTPException(status_code=404, detail="Parameter space config not found for task")
    try:
        with open(ps_file, "r") as f:
            ps_config = json.load(f)
        # Assuming ParameterSpace.from_dict exists and works
        parameter_space = ParameterSpace.from_dict(ps_config)
    except Exception as e:
        logger.error(f"Failed to load/reconstruct parameter space from {ps_file}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load parameter space config")

    # 2. Load Existing Results (Initial X and y)
    initial_X_internal_list = []
    initial_y_list = []
    results_file = task_dir / "results.json"
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                results_data = json.load(f)
            logger.info(f"Found {len(results_data)} existing results in {results_file}.")
            for res in results_data:
                # Ensure result format is as expected (dict with 'parameters' and 'objectives')
                if isinstance(res, dict) and 'parameters' in res and 'objectives' in res:
                    try:
                        # Convert external point dict to internal numpy array
                        internal_point = parameter_space.point_to_internal(res['parameters'])
                        initial_X_internal_list.append(internal_point)
                        # Assuming single objective - find the first objective value
                        if res['objectives']:
                            first_objective_key = next(iter(res['objectives']))
                            initial_y_list.append(float(res['objectives'][first_objective_key]))
                        else:
                             logger.warning(f"Skipping result with empty objectives: {res}")
                    except Exception as conversion_err:
                        logger.warning(f"Skipping result due to conversion error: {conversion_err}. Result: {res}", exc_info=False) # Reduce noise
                else:
                     logger.warning(f"Skipping invalid result format: {res}")

        except Exception as e:
            logger.error(f"Failed to load or parse results from {results_file}: {e}", exc_info=True)
            initial_X_internal_list = []
            initial_y_list = []
    else:
         logger.info(f"No results file found at {results_file}. Initializing optimizer without prior data.")

    # Prepare initial data for optimizer (can be None if no results)
    initial_X_np = np.array(initial_X_internal_list) if initial_X_internal_list else None
    initial_y_np = np.array(initial_y_list) if initial_y_list else None

    # 3. Instantiate Optimizer
    try:
        # Using the implemented concrete classes for now
        # TODO: Load model_class, acquisition_class, and configs from strategy.json
        optimizer = BayesianOptimizer(
            parameter_space=parameter_space,
            model_class=GaussianProcessModel,
            acquisition_class=ExpectedImprovement,
            initial_X=initial_X_np,
            initial_y=initial_y_np,
            # model_config={}, # Load from strategy later
            # acquisition_config={} # Load from strategy later
        )
        optimizers[task_id] = optimizer # Store the instance
        logger.info(f"Successfully created and stored optimizer instance for task {task_id}.")
        return optimizer
    except Exception as e:
        logger.error(f"Failed to instantiate BayesianOptimizer for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to initialize optimization engine")

# ----- 1. Parameter Space Configuration API -----

@router.post("/parameter-space", response_model=schema.ParameterSpaceCreationResponse)
async def create_parameter_space(data: schema.ParameterSpaceCreate):
    """
    Create a new optimization task by defining a parameter space.
    """
    logger.info(f"Creating parameter space: {data.name}")
    
    # Generate a unique task ID
    task_id = generate_id()
    
    # Store the parameter space (in-memory for now)
    now = datetime.now()
    parameter_spaces[task_id] = schema.ParameterSpaceRead(
        **data.dict(),
        task_id=task_id,
        created_at=now,
        updated_at=now,
    )
    
    # Create a task entry
    tasks[task_id] = {
        "task_id": task_id,
        "name": data.name,
        "status": schema.TaskStatus.CREATED.value,
        "created_at": now,
        "updated_at": now,
    }
    
    # Create task directory
    task_dir = PathLib(settings.TASK_DIR) / task_id
    os.makedirs(task_dir, exist_ok=True)
    
    # Save parameter space to file (temporary persistence)
    with open(task_dir / "parameter_space.json", "w") as f:
        json.dump(parameter_spaces[task_id].dict(), f, default=str)
    
    return schema.ParameterSpaceCreationResponse(
        task_id=task_id,
        status=schema.TaskStatus.CREATED.value,
        message=f"Parameter space '{data.name}' created successfully",
    )


@router.get("/parameter-space/{task_id}", response_model=schema.ParameterSpaceRead)
async def get_parameter_space(task_id: str = Path(..., description="Task ID")):
    """
    Get the parameter space for a specific task.
    """
    if task_id not in parameter_spaces:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return parameter_spaces[task_id]


@router.put("/parameter-space/{task_id}", response_model=schema.ParameterSpaceCreationResponse)
async def update_parameter_space(
    data: schema.ParameterSpaceCreate,
    task_id: str = Path(..., description="Task ID"),
):
    """
    Update an existing parameter space.
    """
    if task_id not in parameter_spaces:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Update parameter space
    now = datetime.now()
    parameter_spaces[task_id] = schema.ParameterSpaceRead(
        **data.dict(),
        task_id=task_id,
        created_at=parameter_spaces[task_id].created_at,
        updated_at=now,
    )
    
    # Update task
    tasks[task_id]["name"] = data.name
    tasks[task_id]["updated_at"] = now
    
    # Update file
    task_dir = PathLib(settings.TASK_DIR) / task_id
    with open(task_dir / "parameter_space.json", "w") as f:
        json.dump(parameter_spaces[task_id].dict(), f, default=str)
    
    return schema.ParameterSpaceCreationResponse(
        task_id=task_id,
        status=schema.TaskStatus.CREATED.value,
        message=f"Parameter space updated successfully",
    )

# ----- 2. Optimization Strategy API -----

@router.post("/strategy/{task_id}", response_model=dict)
async def set_strategy(
    strategy: schema.StrategyCreate,
    task_id: str = Path(..., description="Task ID"),
):
    """
    Configure the optimization algorithm and strategy.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Save strategy
    now = datetime.now()
    strategies[task_id] = schema.StrategyRead(
        **strategy.dict(),
        task_id=task_id,
        created_at=now,
        updated_at=now,
    )
    
    # Save to file
    task_dir = PathLib(settings.TASK_DIR) / task_id
    with open(task_dir / "strategy.json", "w") as f:
        json.dump(strategies[task_id].dict(), f, default=str)
    
    return {"message": "Strategy set successfully"}


@router.get("/strategy/{task_id}", response_model=schema.StrategyRead)
async def get_strategy(task_id: str = Path(..., description="Task ID")):
    """
    Get the current optimization strategy.
    """
    if task_id not in strategies:
        raise HTTPException(status_code=404, detail=f"No strategy set for task {task_id}")
    
    return strategies[task_id]

# ----- 3. Experiment Design API -----

@router.get("/designs/{task_id}/initial", response_model=schema.DesignResponse)
async def get_initial_designs(
    task_id: str = Path(..., description="Task ID"),
    samples: int = Query(None, description="Number of samples to generate"),
):
    """
    Get initial design points for a task. Generates them if they don't exist.
    """
    # --- Load Parameter Space ---
    task_dir = PathLib(settings.TASK_DIR) / task_id
    ps_file = task_dir / "parameter_space.json"
    if not ps_file.exists():
        logger.error(f"Parameter space config not found for task {task_id} at {ps_file}")
        raise HTTPException(status_code=404, detail=f"Parameter space config not found for task {task_id}")

    try:
        with open(ps_file, "r") as f:
            ps_config = json.load(f)
        # Note: Assumes ParameterSpace has a class method `from_dict` to reconstruct from JSON config
        parameter_space_obj = ParameterSpace.from_dict(ps_config)
        logger.info(f"Successfully loaded parameter space for task {task_id}")
    except Exception as e:
        logger.error(f"Failed to load or reconstruct parameter space for task {task_id} from {ps_file}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load parameter space configuration")

    # --- Check for Existing Designs ---
    designs_file = task_dir / "initial_designs.json"
    if designs_file.exists():
        try:
            with open(designs_file, "r") as f:
                stored_designs_data = json.load(f)
            # Validate and parse stored designs using Pydantic models
            validated_designs = [schema.Design(**d) for d in stored_designs_data]
            logger.info(f"Found {len(validated_designs)} existing initial designs for task {task_id}")
            return schema.DesignResponse(designs=validated_designs)
        except Exception as e:
            logger.warning(f"Failed to load or parse existing designs file {designs_file} for task {task_id}: {e}. Regenerating.", exc_info=True)
            # If loading fails, proceed to generate new ones

    # --- Generate Initial Designs ---
    n_samples = samples or settings.DEFAULT_INITIAL_SAMPLES
    # For now, default to LHS. This could be made configurable later.
    design_type = DesignType.LATIN_HYPERCUBE

    try:
        logger.info(f"Generating {n_samples} initial designs using {design_type.value} for task {task_id}")
        # Create the design generator (seed handling might be internal or added here)
        design_generator = create_design_generator(parameter_space_obj, design_type)

        # Generate design points (list of dictionaries: {param_name: value})
        generated_points = design_generator.generate(n=n_samples)
        logger.info(f"Successfully generated {len(generated_points)} points for task {task_id}")

        # Convert to schema.Design format, adding unique IDs
        final_designs = [
            schema.Design(id=generate_unique_id(), parameters=point)
            for point in generated_points
        ]
        logger.info(f"Formatted {len(final_designs)} designs into schema for task {task_id}")

        if len(final_designs) < n_samples:
             logger.warning(f"Only generated {len(final_designs)} designs out of {n_samples} requested for task {task_id}, possibly due to constraints.")


    except Exception as e:
        logger.error(f"Failed to generate initial designs for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate initial designs: {str(e)}")

    # --- Save Generated Designs ---
    try:
        with open(designs_file, "w") as f:
            # Convert Pydantic models to dicts for JSON serialization
            json.dump([d.dict() for d in final_designs], f, indent=4)
        logger.info(f"Saved {len(final_designs)} initial designs to {designs_file} for task {task_id}")
    except Exception as e:
        logger.error(f"Failed to save initial designs to {designs_file} for task {task_id}: {e}", exc_info=True)
        # Proceed even if saving fails, but log the error

    return schema.DesignResponse(designs=final_designs)


@router.post("/results/{task_id}", response_model=dict)
async def submit_results(
    results_data: schema.ResultsSubmission,
    task_id: str = Path(..., description="Task ID"),
):
    """
    Submit experiment results and update the optimizer state.
    """
    if task_id not in tasks:
        logger.error(f"Task not found for ID: {task_id} during results submission.")
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    try:
        # Get or create the optimizer instance for this task
        optimizer = get_or_create_optimizer(task_id)
    except HTTPException as e:
         raise e # Propagate known HTTP errors
    except Exception as e:
        logger.error(f"Unexpected error getting/creating optimizer for task {task_id} during results submission: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error initializing optimizer")

    # Prepare data for the optimizer's observe method
    observed_points_external = []
    observed_objectives = []
    valid_results_count = 0
    for result in results_data.results: # results_data.results is List[ResultCreate]
        if isinstance(result.parameters, dict) and isinstance(result.objectives, dict) and result.objectives:
             observed_points_external.append(result.parameters)
             # Assuming single objective - get the first value
             first_objective_key = next(iter(result.objectives))
             observed_objectives.append(float(result.objectives[first_objective_key]))
             valid_results_count += 1
        else:
            logger.warning(f"Skipping invalid result format during submission: {result.dict()}")

    if not valid_results_count:
        logger.warning(f"No valid results found in submission for task {task_id}.")
        return {"message": "No valid results submitted."}

    try:
        # Call the optimizer's observe method
        logger.info(f"Observing {valid_results_count} new results for task {task_id}.")
        optimizer.observe(X=observed_points_external, y=observed_objectives)
        logger.info(f"Optimizer model retrained successfully for task {task_id}.")

        # Update task status in the main tasks dictionary
        tasks[task_id]["status"] = schema.TaskStatus.RUNNING.value
        tasks[task_id]["updated_at"] = datetime.now()
        logger.info(f"Task {task_id} status updated to RUNNING.")

        # Persist results to file (append or overwrite depending on strategy)
        # Current logic overwrites, which might be okay if get_or_create re-reads it
        task_dir = PathLib(settings.TASK_DIR) / task_id
        results_file = task_dir / "results.json"
        all_results = []
        if results_file.exists():
             try:
                 with open(results_file, "r") as f:
                      all_results = json.load(f)
             except Exception as e:
                 logger.warning(f"Could not read existing results file {results_file}: {e}. Overwriting.")
        # Append new valid results
        all_results.extend([res.dict() for res in results_data.results if isinstance(res.parameters, dict) and isinstance(res.objectives, dict) and res.objectives])
        try:
             with open(results_file, "w") as f:
                 json.dump(all_results, f, indent=4, default=str)
             logger.info(f"Saved updated results ({len(all_results)} total) to {results_file}")
        except Exception as e:
             logger.error(f"Failed to save updated results to {results_file}: {e}")

    except Exception as e:
        logger.error(f"Failed to observe results or retrain model for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing results: {str(e)}")

    return {"message": f"{valid_results_count} results submitted and processed successfully for task {task_id}"}


@router.get("/designs/{task_id}/next", response_model=schema.DesignResponse)
async def get_next_designs(
    task_id: str = Path(..., description="Task ID"),
    batch_size: int = Query(1, description="Number of designs to generate"),
    strategy: Optional[str] = Query(None, description="Batch strategy (unused for now)"),
):
    """
    Get the next batch of recommended design points using Bayesian Optimization.
    """
    if task_id not in tasks:
        logger.error(f"Task not found for ID: {task_id}")
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    try:
        # Get or create the optimizer instance for this task
        optimizer = get_or_create_optimizer(task_id)
    except HTTPException as e:
         # Propagate known HTTP errors (e.g., 404, 500 during loading)
         raise e
    except Exception as e:
        # Catch unexpected errors during optimizer creation/retrieval
        logger.error(f"Unexpected error getting/creating optimizer for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error initializing optimizer")

    # Check if there are any observations to base suggestions on
    if not optimizer.y_observed: # Check if y_observed list is empty
        logger.warning(f"No results observed yet for task {task_id}. Cannot suggest next points via BO.")
        # Instead of raising 400, maybe return empty list or a specific status?
        # For now, raise 400 as per previous logic.
        raise HTTPException(status_code=400, detail="No results submitted yet. Cannot suggest next points via BO. Please submit results for initial designs.")

    try:
        logger.info(f"Suggesting {batch_size} next design(s) for task {task_id} using BO.")
        # Call the optimizer's suggest method
        suggested_points_external = optimizer.suggest(n_suggestions=batch_size) # Returns List[Dict]

        # Convert suggestions to schema.Design format
        final_suggestions = []
        for point in suggested_points_external:
            design_id = generate_unique_id()
            predictions_dict = None
            # Optionally add predictions for the suggested point
            if optimizer.model.is_trained():
                try:
                    internal_point = optimizer.parameter_space.point_to_internal(point)
                    mean, variance = optimizer.model.predict(internal_point.reshape(1,-1))
                    # Assuming single objective for now - get name from param space
                    obj_name = next(iter(optimizer.parameter_space.objectives)) # Get the first objective name
                    predictions_dict = {
                         obj_name: schema.Prediction(mean=mean[0], std=np.sqrt(np.maximum(variance[0], 0))) # Ensure std is non-negative
                     }
                except Exception as pred_err:
                    logger.warning(f"Could not get prediction for suggested point {point}: {pred_err}", exc_info=False) # Reduce noise

            final_suggestions.append(
                schema.Design(
                    id=design_id,
                    parameters=point,
                    predictions=predictions_dict,
                    # uncertainty=?, # Could be acquisition value or model variance
                    # reason="Suggested by Bayesian Optimizer (EI)" # Add reason later
                )
            )

        logger.info(f"Successfully generated {len(final_suggestions)} suggestions for task {task_id}")

    except Exception as e:
        logger.error(f"Failed to suggest next designs for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to suggest next designs: {str(e)}")

    # Optional: Save suggestions to a file (e.g., suggestions_iter_N.json)
    # ... (saving logic similar to initial designs if needed) ...

    return schema.DesignResponse(designs=final_suggestions)

# ----- 4. Model & Analysis API -----

@router.post("/predict/{task_id}", response_model=schema.PredictionResponse)
async def predict(
    prediction_request: schema.PredictionRequest,
    task_id: str = Path(..., description="Task ID"),
):
    """
    Get model predictions for specific parameter combinations.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    if task_id not in results or not results[task_id]:
        raise HTTPException(status_code=400, detail="No results submitted yet, model not trained")
    
    # Placeholder for prediction
    # This will be implemented in Phase 4
    predictions = []
    for params in prediction_request.parameters:
        predictions.append({
            "parameters": params,
            "objectives": {
                "y1": {"mean": 0.8, "std": 0.05},
                "y2": {"mean": 0.2, "std": 0.03},
            }
        })
    
    return schema.PredictionResponse(predictions=predictions)


@router.get("/model/{task_id}/performance", response_model=schema.ModelPerformance)
async def get_model_performance(task_id: str = Path(..., description="Task ID")):
    """
    Get the current model's performance metrics.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    if task_id not in results or not results[task_id]:
        raise HTTPException(status_code=400, detail="No results submitted yet, model not trained")
    
    # Placeholder for model performance
    # This will be implemented in Phase 4
    return schema.ModelPerformance(
        metrics=schema.ModelMetrics(
            r2=0.92,
            rmse=0.08,
            mae=0.06,
        ),
        cross_validation=schema.CrossValidation(
            cv_scores=[0.91, 0.93, 0.90, 0.94, 0.92],
            mean_score=0.92,
            std_score=0.015,
        ),
    )


@router.get("/pareto/{task_id}", response_model=schema.ParetoFront)
async def get_pareto_front(task_id: str = Path(..., description="Task ID")):
    """
    Get the Pareto front for multi-objective optimization.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    if task_id not in results or not results[task_id]:
        raise HTTPException(status_code=400, detail="No results submitted yet")
    
    # Placeholder for Pareto front
    # This will be implemented in Phase 4
    pareto_solutions = [
        schema.ParetoSolution(
            id=f"design_5",
            parameters={"x1": 0.3, "x2": "B", "x3": 4},
            objectives={"y1": 0.9, "y2": 0.3},
            uncertainty=0.02,
        ),
        schema.ParetoSolution(
            id=f"design_8",
            parameters={"x1": 0.7, "x2": "A", "x3": 6},
            objectives={"y1": 0.7, "y2": 0.1},
            uncertainty=0.03,
        ),
    ]
    
    dominated_solutions = [
        schema.ParetoSolution(
            id=f"design_2",
            parameters={"x1": 0.2, "x2": "C", "x3": 3},
            objectives={"y1": 0.6, "y2": 0.4},
            uncertainty=0.04,
        ),
    ]
    
    return schema.ParetoFront(
        pareto_front=pareto_solutions,
        dominated_solutions=dominated_solutions,
        ideal_point={"y1": 1.0, "y2": 0.0},
        nadir_point={"y1": 0.5, "y2": 0.8},
    )


@router.get("/uncertainty/{task_id}", response_model=schema.UncertaintyAnalysis)
async def get_uncertainty_analysis(task_id: str = Path(..., description="Task ID")):
    """
    Get uncertainty analysis comparing predictions to actual values.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    if task_id not in results or not results[task_id]:
        raise HTTPException(status_code=400, detail="No results submitted yet")
    
    # Placeholder for uncertainty analysis
    # This will be implemented in Phase 4
    prediction_vs_actual = [
        schema.PredictionActual(
            design_id="design_1",
            predicted=schema.Prediction(mean=0.8, std=0.05),
            actual=0.75,
            error=0.05,
            within_confidence=True,
        ),
        schema.PredictionActual(
            design_id="design_2",
            predicted=schema.Prediction(mean=0.7, std=0.04),
            actual=0.72,
            error=0.02,
            within_confidence=True,
        ),
    ]
    
    return schema.UncertaintyAnalysis(
        prediction_vs_actual=prediction_vs_actual,
        calibration_metrics=schema.CalibrationMetrics(
            coverage_probability=0.95,
            sharpness=0.08,
        ),
    )

# ----- 5. Task Management API -----

@router.get("/tasks", response_model=schema.TaskList)
async def get_tasks():
    """
    Get a list of all optimization tasks.
    """
    task_list = [
        schema.TaskInfo(**task) for task in tasks.values()
    ]
    return schema.TaskList(tasks=task_list)


@router.get("/tasks/{task_id}/status", response_model=schema.TaskStatusResponse)
async def get_task_status(task_id: str = Path(..., description="Task ID")):
    """
    Get the status of a specific task.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Get best result (if any)
    best_result = None
    if task_id in results and results[task_id]:
        # Placeholder for finding best result
        # This will be implemented in Phase 5
        best_result = {
            "parameters": {"x1": 0.4, "x2": "C", "x3": 7},
            "objectives": {"y1": 0.92, "y2": 0.18},
        }
    
    return schema.TaskStatusResponse(
        status=schema.TaskStatus(tasks[task_id]["status"]),
        current_iteration=len(results.get(task_id, [])),
        total_iterations=strategies.get(task_id, {}).get("iterations", None),
        best_result=best_result,
        last_updated=tasks[task_id]["updated_at"],
    )


@router.get("/tasks/{task_id}/export")
async def export_task_data(
    task_id: str = Path(..., description="Task ID"),
    format: str = Query("json", description="Export format (json or csv)"),
):
    """
    Export complete task data.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task_dir = PathLib(settings.TASK_DIR) / task_id
    export_file = task_dir / f"export.{format}"
    
    # Placeholder for data export
    # This will be implemented in Phase 5
    if format == "json":
        export_data = {
            "task_info": tasks[task_id],
            "parameter_space": parameter_spaces.get(task_id, {}).dict() if task_id in parameter_spaces else {},
            "strategy": strategies.get(task_id, {}).dict() if task_id in strategies else {},
            "results": results.get(task_id, []),
        }
        
        with open(export_file, "w") as f:
            json.dump(export_data, f, default=str)
    else:
        # CSV export would be implemented here
        # For now, just create a dummy file
        with open(export_file, "w") as f:
            f.write("task_id,parameter,value\n")
            f.write(f"{task_id},dummy,data\n")
    
    return FileResponse(
        path=export_file,
        filename=f"task_{task_id}_export.{format}",
        media_type="application/json" if format == "json" else "text/csv",
    )

# ----- 6. WebSocket API -----

@router.websocket("/ws/tasks/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket connection for real-time task updates.
    """
    if task_id not in tasks:
        await websocket.close(code=1008, reason=f"Task {task_id} not found")
        return
    
    await websocket.accept()
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "data": {
                "task_id": task_id,
                "status": tasks[task_id]["status"],
                "timestamp": datetime.now().isoformat(),
            }
        })
        
        # Keep connection alive with periodic pings
        while True:
            # In a real implementation, this would be event-driven
            # For now, just sleep and send a dummy update
            await asyncio.sleep(settings.WS_PING_INTERVAL)
            
            # Send ping to keep connection alive
            await websocket.send_json({
                "type": "ping",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                }
            })
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
    finally:
        logger.info(f"WebSocket connection closed for task {task_id}")

# ----- 7. Additional Endpoints (from supplementary suggestions) -----

@router.post("/tasks/{task_id}/restart", response_model=dict)
async def restart_task(
    restart_config: schema.TaskRestart,
    task_id: str = Path(..., description="Task ID"),
):
    """
    Restart a BO experiment task after interruption.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Update task status
    tasks[task_id]["status"] = schema.TaskStatus.RUNNING.value
    tasks[task_id]["updated_at"] = datetime.now()
    
    # Handle history based on restart strategy
    if restart_config.strategy == "reset" and not restart_config.preserve_history:
        # Clear results
        if task_id in results:
            results[task_id] = []
        
        # Save empty results
        task_dir = PathLib(settings.TASK_DIR) / task_id
        with open(task_dir / "results.json", "w") as f:
            json.dump([], f)
    
    return {
        "message": f"Task {task_id} restarted with strategy: {restart_config.strategy}",
        "preserve_history": restart_config.preserve_history,
    }


@router.get("/diagnostics/{task_id}", response_model=schema.Diagnostics)
async def get_diagnostics(task_id: str = Path(..., description="Task ID")):
    """
    Get diagnostics information for debugging.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # This would contain actual diagnostics in a real implementation
    # For now, return dummy data
    return schema.Diagnostics(
        parameter_space="valid" if task_id in parameter_spaces else "not_defined",
        model_trained=task_id in results and len(results[task_id]) > 0,
        recent_exception=None,
        pending_experiments=[],
        last_recommendation_time=datetime.now() if task_id in results and results[task_id] else None,
    ) 
