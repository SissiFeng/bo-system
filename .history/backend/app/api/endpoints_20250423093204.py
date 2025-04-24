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

from backend.app.core.config import get_settings
from backend.app.core.logger import setup_logger
from backend.app.api import schema
from backend.bo_engine.parameter_space import ParameterSpace
from backend.bo_engine.design_generator import create_design_generator, DesignType
from backend.bo_engine.utils import generate_unique_id
from backend.bo_engine.optimizer import BayesianOptimizer
from backend.bo_engine.models import GaussianProcessModel
from backend.bo_engine.acquisition import ExpectedImprovement

# Import core BO engine components
# These will be implemented in Phase 2-4
# from backend.bo_engine.parameter_space import ParameterSpace
# from backend.bo_engine.design_generator import DesignGenerator
# from backend.bo_engine.optimizer import Optimizer

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

    # 2. Load Strategy Configuration if available
    model_class = GaussianProcessModel  # Default model class
    acquisition_class = ExpectedImprovement  # Default acquisition class
    model_config = {}  # Default empty model config
    acquisition_config = {}  # Default empty acquisition config
    
    strategy_file = task_dir / "strategy.json"
    if strategy_file.exists():
        try:
            with open(strategy_file, "r") as f:
                strategy_config = json.load(f)
            logger.info(f"Successfully loaded strategy config from {strategy_file}")
            
            # Map strategy.algorithm to model class if applicable
            algorithm = strategy_config.get("algorithm", "").lower()
            if algorithm == "gaussian_process" or algorithm == "gp":
                model_class = GaussianProcessModel
            # Add mappings for other algorithms as they are implemented
            
            # Extract acquisition function
            acq_function = strategy_config.get("acquisition_function", "").lower()
            if acq_function == "ei" or acq_function == "expected_improvement":
                acquisition_class = ExpectedImprovement
            # Add mappings for other acquisition functions as they are implemented
            
            # Extract settings for model and acquisition function
            settings = strategy_config.get("settings", {})
            # Separate settings into model_config and acquisition_config
            # This is a simple example, might need more sophisticated mapping
            if "exploration_weight" in settings:
                acquisition_config["xi"] = float(settings["exploration_weight"])
            if "kernel" in settings:
                model_config["kernel"] = settings["kernel"]
            if "noise_level" in settings:
                model_config["noise_level"] = float(settings["noise_level"])
                
            logger.debug(f"Using model_config: {model_config}, acquisition_config: {acquisition_config}")
            
        except Exception as e:
            logger.warning(f"Failed to load strategy from {strategy_file}: {e}. Using defaults.", exc_info=True)
    else:
        logger.info(f"No strategy file found at {strategy_file}. Using default optimization strategy.")

    # 3. Load Existing Results (Initial X and y)
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

    # 4. Instantiate Optimizer
    try:
        optimizer = BayesianOptimizer(
            parameter_space=parameter_space,
            model_class=model_class,
            acquisition_class=acquisition_class,
            initial_X=initial_X_np,
            initial_y=initial_y_np,
            model_config=model_config,
            acquisition_config=acquisition_config
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
    task_info = {
        "task_id": task_id,
        "name": data.name,
        "status": schema.TaskStatus.CREATED.value,
        "created_at": now,
        "updated_at": now,
        "progress": 0.0,  # Initial progress is 0%
        "description": data.dict().get("description", ""),  # Optional description
    }
    tasks[task_id] = task_info
    
    # Create task directory
    task_dir = PathLib(settings.TASK_DIR) / task_id
    os.makedirs(task_dir, exist_ok=True)
    
    # Save parameter space to file
    with open(task_dir / "parameter_space.json", "w") as f:
        json.dump(parameter_spaces[task_id].dict(), f, default=str)
    
    # Save task info to file
    with open(task_dir / "task_info.json", "w") as f:
        json.dump(task_info, f, default=str)
    
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
    task_dir = PathLib(settings.TASK_DIR) / task_id
    
    # Check if task exists, try to load if not in memory
    if task_id not in tasks:
        task_info_file = task_dir / "task_info.json"
        if task_info_file.exists():
            try:
                with open(task_info_file, "r") as f:
                    tasks[task_id] = json.load(f)
                logger.info(f"Loaded task {task_id} from disk for results submission")
            except Exception as e:
                logger.error(f"Failed to load task {task_id} for results submission: {e}", exc_info=True)
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be loaded")
        else:
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

        # Update task status and time
        now = datetime.now()
        tasks[task_id]["status"] = schema.TaskStatus.RUNNING.value
        tasks[task_id]["updated_at"] = now

        # Update progress based on results count
        results_file = task_dir / "results.json"
        results_count = 0
        all_results = []
        
        if results_file.exists():
             try:
                 with open(results_file, "r") as f:
                      all_results = json.load(f)
                 results_count = len(all_results)
             except Exception as e:
                 logger.warning(f"Could not read existing results file {results_file}: {e}. Overwriting.")
        
        # Append new valid results
        all_results.extend([res.dict() for res in results_data.results if isinstance(res.parameters, dict) and isinstance(res.objectives, dict) and res.objectives])
        results_count = len(all_results)
        
        # Calculate progress
        strategy_file = task_dir / "strategy.json"
        total_iterations = None
        if strategy_file.exists():
            try:
                with open(strategy_file, "r") as f:
                    strategy_data = json.load(f)
                total_iterations = strategy_data.get("iterations")
            except Exception as e:
                logger.warning(f"Failed to read strategy file for task {task_id}: {e}", exc_info=False)
        
        # Update progress in task info
        if total_iterations is not None and total_iterations > 0:
            tasks[task_id]["progress"] = min(100.0, (results_count / total_iterations) * 100.0)
        else:
            # Simple progress estimation
            if results_count > 0:
                tasks[task_id]["progress"] = min(80.0, results_count * 5.0)  # Arbitrary progress scale
        
        # Save updated task info
        task_info_file = task_dir / "task_info.json"
        with open(task_info_file, "w") as f:
            task_info = tasks[task_id].copy()
            # Convert datetime objects to ISO strings for JSON serialization
            if isinstance(task_info.get("created_at"), datetime):
                task_info["created_at"] = task_info["created_at"].isoformat()
            if isinstance(task_info.get("updated_at"), datetime):
                task_info["updated_at"] = task_info["updated_at"].isoformat()
            json.dump(task_info, f, default=str)
        
        # Save results to file
        try:
             with open(results_file, "w") as f:
                 json.dump(all_results, f, indent=4, default=str)
             logger.info(f"Saved updated results ({len(all_results)} total) to {results_file}")
        except Exception as e:
             logger.error(f"Failed to save updated results to {results_file}: {e}")

    except Exception as e:
        logger.error(f"Failed to observe results or retrain model for task {task_id}: {e}", exc_info=True)
        
        # Record error in error log
        try:
            error_log_file = task_dir / "error.log"
            with open(error_log_file, "a") as f:
                f.write(f"[{datetime.now().isoformat()}] Error submitting results: {str(e)}\n")
        except Exception:
            pass  # Silently ignore error log write failures
            
        raise HTTPException(status_code=500, detail=f"Error processing results: {str(e)}")

    return {
        "message": f"{valid_results_count} results submitted and processed successfully for task {task_id}",
        "results_count": results_count,
        "progress": tasks[task_id]["progress"]
    }


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
    
    # 获取或创建优化器实例，其中包含已训练的模型
    try:
        optimizer = get_or_create_optimizer(task_id)
    except HTTPException as e:
        raise e  # 传递已知的 HTTP 异常
    except Exception as e:
        logger.error(f"获取优化器时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取优化器失败: {str(e)}")
    
    # 检查模型是否已训练
    if not optimizer.model.is_trained():
        raise HTTPException(status_code=400, detail="模型尚未训练，请先提交实验结果")
    
    # 准备预测的点
    parameter_combinations = prediction_request.parameters
    if not isinstance(parameter_combinations, list):
        parameter_combinations = [parameter_combinations]
    
    # 进行预测
    predictions = []
    for params in parameter_combinations:
        try:
            # 将参数转换为内部表示
            internal_point = optimizer.parameter_space.point_to_internal(params)
            
            # 使用模型进行预测
            mean, variance = optimizer.model.predict(internal_point.reshape(1, -1))
            std_dev = np.sqrt(np.maximum(variance, 0))  # 确保标准差非负
            
            # 获取目标名称
            objective_names = list(optimizer.parameter_space.objectives.keys())
            if not objective_names:
                objective_names = ["objective"]  # 默认目标名称
            
            # 创建预测响应
            objectives_dict = {}
            for i, obj_name in enumerate(objective_names):
                if i < len(mean):
                    objectives_dict[obj_name] = {
                        "mean": float(mean[i]),
                        "std": float(std_dev[i])
                    }
                else:
                    # 如果只有一个预测但有多个目标，使用相同的预测
                    objectives_dict[obj_name] = {
                        "mean": float(mean[0]),
                        "std": float(std_dev[0])
                    }
            
            predictions.append({
                "parameters": params,
                "objectives": objectives_dict
            })
            
        except Exception as e:
            logger.error(f"预测参数 {params} 时出错: {e}", exc_info=True)
            # 跳过失败的预测点而不是整个请求失败
            continue
    
    if not predictions:
        raise HTTPException(status_code=422, detail="无法为任何提供的参数组合生成预测")
    
    return schema.PredictionResponse(predictions=predictions)


@router.get("/model/{task_id}/performance", response_model=schema.ModelPerformance)
async def get_model_performance(task_id: str = Path(..., description="Task ID")):
    """
    Get the current model's performance metrics.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # 获取或创建优化器实例
    try:
        optimizer = get_or_create_optimizer(task_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"获取优化器时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取优化器失败: {str(e)}")
    
    # 检查模型是否已训练
    if not optimizer.model.is_trained():
        raise HTTPException(status_code=400, detail="模型尚未训练，请先提交实验结果")
    
    # 从优化器和模型中获取性能指标
    try:
        # 获取训练数据（已转换为内部表示的数据）
        X_train = np.array(optimizer.X_observed)
        y_train = np.array(optimizer.y_observed)
        
        if len(X_train) < 5:  # 数据点太少，无法进行可靠的交叉验证
            # 直接使用训练数据计算性能指标
            metrics = optimizer.model.score(X_train, y_train)
            
            return schema.ModelPerformance(
                model_type=optimizer.model.__class__.__name__,
                metrics=metrics,
                cross_validation_results=None
            )
        else:
            # 使用交叉验证计算更可靠的性能指标
            from sklearn.model_selection import KFold
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            import numpy as np
            
            # 使用5折交叉验证（或调整为适合数据大小的折数）
            n_splits = min(5, len(X_train))
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            r2_scores = []
            rmse_scores = []
            mae_scores = []
            
            # 执行交叉验证
            for train_idx, test_idx in kf.split(X_train):
                # 分割数据
                X_cv_train, X_cv_test = X_train[train_idx], X_train[test_idx]
                y_cv_train, y_cv_test = y_train[train_idx], y_train[test_idx]
                
                # 创建一个新的模型实例进行训练
                temp_model = optimizer.model.__class__(optimizer.parameter_space, **optimizer.model._config)
                temp_model.train(X_cv_train, y_cv_train)
                
                # 预测测试集
                y_pred, _ = temp_model.predict(X_cv_test)
                
                # 计算性能指标
                r2 = r2_score(y_cv_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_cv_test, y_pred))
                mae = mean_absolute_error(y_cv_test, y_pred)
                
                r2_scores.append(r2)
                rmse_scores.append(rmse)
                mae_scores.append(mae)
            
            # 计算平均性能指标
            mean_r2 = float(np.mean(r2_scores))
            mean_rmse = float(np.mean(rmse_scores))
            mean_mae = float(np.mean(mae_scores))
            
            # 返回性能指标和交叉验证结果
            metrics = {
                "r2": mean_r2,
                "rmse": mean_rmse,
                "mae": mean_mae
            }
            
            cv_results = {
                "r2_scores": [float(score) for score in r2_scores],
                "rmse_scores": [float(score) for score in rmse_scores],
                "mae_scores": [float(score) for score in mae_scores],
                "mean_r2": mean_r2,
                "mean_rmse": mean_rmse,
                "mean_mae": mean_mae,
                "std_r2": float(np.std(r2_scores)),
                "std_rmse": float(np.std(rmse_scores)),
                "std_mae": float(np.std(mae_scores))
            }
            
            return schema.ModelPerformance(
                model_type=optimizer.model.__class__.__name__,
                metrics=metrics,
                cross_validation_results=cv_results
            )
            
    except Exception as e:
        logger.error(f"计算模型性能指标时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"计算模型性能指标失败: {str(e)}")


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
    # Scan task directory to get all tasks - even those not loaded in memory
    task_list = []
    
    # First, add tasks that are already in memory
    for task_id, task_info in tasks.items():
        task_list.append(schema.TaskInfo(**task_info))
    
    # Then scan the task directory for any tasks not yet loaded in memory
    task_dir_base = PathLib(settings.TASK_DIR)
    if task_dir_base.exists():
        for task_dir in task_dir_base.iterdir():
            if task_dir.is_dir():
                task_id = task_dir.name
                
                # Skip if already in memory
                if task_id in tasks:
                    continue
                
                # Try to load task_info.json
                task_info_file = task_dir / "task_info.json"
                if task_info_file.exists():
                    try:
                        with open(task_info_file, "r") as f:
                            task_info = json.load(f)
                        
                        # Convert string dates to datetime objects if needed
                        if isinstance(task_info.get("created_at"), str):
                            task_info["created_at"] = datetime.fromisoformat(task_info["created_at"].replace('Z', '+00:00'))
                        if isinstance(task_info.get("updated_at"), str):
                            task_info["updated_at"] = datetime.fromisoformat(task_info["updated_at"].replace('Z', '+00:00'))
                        
                        # Add to in-memory tasks dictionary
                        tasks[task_id] = task_info
                        
                        # Add to the response list
                        task_list.append(schema.TaskInfo(**task_info))
                        logger.debug(f"Loaded task {task_id} from disk")
                    except Exception as e:
                        logger.error(f"Failed to load task info for {task_id}: {e}", exc_info=True)
                else:
                    # Task directory exists but no task_info.json, try to reconstruct minimal info
                    ps_file = task_dir / "parameter_space.json"
                    if ps_file.exists():
                        try:
                            with open(ps_file, "r") as f:
                                ps_config = json.load(f)
                            
                            # Create minimal task info from parameter space
                            name = ps_config.get("name", f"Task {task_id}")
                            minimal_task_info = {
                                "task_id": task_id,
                                "name": name,
                                "status": schema.TaskStatus.CREATED.value,
                                "created_at": datetime.now(),  # Use current time as fallback
                                "updated_at": datetime.now(),
                                "progress": 0.0,
                            }
                            tasks[task_id] = minimal_task_info
                            task_list.append(schema.TaskInfo(**minimal_task_info))
                            
                            # Create task_info.json for future reference
                            with open(task_info_file, "w") as f:
                                json.dump(minimal_task_info, f, default=str)
                                
                            logger.warning(f"Reconstructed minimal task info for {task_id}")
                        except Exception as e:
                            logger.error(f"Failed to reconstruct task info for {task_id}: {e}", exc_info=True)
    
    # Sort tasks by creation date, newest first
    task_list.sort(key=lambda x: x.created_at, reverse=True)
    
    return schema.TaskList(tasks=task_list)


@router.get("/tasks/{task_id}/status", response_model=schema.TaskStatusResponse)
async def get_task_status(task_id: str = Path(..., description="Task ID")):
    """
    Get the status of a specific task.
    """
    # Check if task exists in memory or try to load it
    if task_id not in tasks:
        # Try to load task from disk
        task_dir = PathLib(settings.TASK_DIR) / task_id
        task_info_file = task_dir / "task_info.json"
        
        if task_info_file.exists():
            try:
                with open(task_info_file, "r") as f:
                    task_info = json.load(f)
                
                # Convert string dates to datetime objects if needed
                if isinstance(task_info.get("created_at"), str):
                    task_info["created_at"] = datetime.fromisoformat(task_info["created_at"].replace('Z', '+00:00'))
                if isinstance(task_info.get("updated_at"), str):
                    task_info["updated_at"] = datetime.fromisoformat(task_info["updated_at"].replace('Z', '+00:00'))
                
                # Add to in-memory tasks dictionary
                tasks[task_id] = task_info
                logger.debug(f"Loaded task {task_id} from disk for status check")
            except Exception as e:
                logger.error(f"Failed to load task info for {task_id}: {e}", exc_info=True)
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        else:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Get task info
    task_info = tasks[task_id]
    
    # Get number of iterations (results submitted)
    results_count = 0
    task_dir = PathLib(settings.TASK_DIR) / task_id
    results_file = task_dir / "results.json"
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                results_data = json.load(f)
            results_count = len(results_data)
        except Exception as e:
            logger.warning(f"Failed to read results file for task {task_id}: {e}", exc_info=False)
    
    # Get best result (if any)
    best_result = None
    if results_count > 0:
        try:
            # Try to get optimizer to find the best result
            optimizer = get_or_create_optimizer(task_id)
            if optimizer.current_best is not None:
                best_result = optimizer.current_best
                logger.debug(f"Found best result for task {task_id} using optimizer")
        except Exception as e:
            logger.warning(f"Could not use optimizer to find best result for task {task_id}: {e}", exc_info=False)
            
            # Fallback: find best result manually from results file
            if results_file.exists():
                try:
                    with open(results_file, "r") as f:
                        results_data = json.load(f)
                    
                    # Load parameter space to get objective type (min/max)
                    ps_file = task_dir / "parameter_space.json"
                    if ps_file.exists():
                        with open(ps_file, "r") as f:
                            ps_config = json.load(f)
                        
                        # Get first objective and its type
                        objectives = ps_config.get("objectives", {})
                        if objectives:
                            first_obj_name = next(iter(objectives))
                            minimize = objectives[first_obj_name].get("type") == "minimize"
                            
                            # Find best result
                            best_idx = None
                            best_val = float('inf') if minimize else float('-inf')
                            
                            for i, res in enumerate(results_data):
                                if 'objectives' in res and first_obj_name in res['objectives']:
                                    val = float(res['objectives'][first_obj_name])
                                    if (minimize and val < best_val) or (not minimize and val > best_val):
                                        best_val = val
                                        best_idx = i
                            
                            if best_idx is not None:
                                best_result = {
                                    "parameters": results_data[best_idx]['parameters'],
                                    "objectives": results_data[best_idx]['objectives']
                                }
                                logger.debug(f"Found best result for task {task_id} manually")
                except Exception as e:
                    logger.warning(f"Failed to manually find best result for task {task_id}: {e}", exc_info=False)
    
    # Get total iterations from strategy if available
    total_iterations = None
    strategy_file = task_dir / "strategy.json"
    if strategy_file.exists():
        try:
            with open(strategy_file, "r") as f:
                strategy_data = json.load(f)
            total_iterations = strategy_data.get("iterations")
        except Exception as e:
            logger.warning(f"Failed to read strategy file for task {task_id}: {e}", exc_info=False)
    
    # Calculate progress percentage
    progress = 0.0
    if total_iterations is not None and total_iterations > 0:
        progress = min(100.0, (results_count / total_iterations) * 100.0)
    else:
        # If total_iterations not set, use a simple scale: 0% -> 100% based on status
        status = task_info["status"]
        if status == schema.TaskStatus.CREATED.value:
            progress = 0.0
        elif status == schema.TaskStatus.RUNNING.value:
            progress = 50.0  # Halfway
        elif status == schema.TaskStatus.PAUSED.value:
            progress = 70.0  # More than halfway
        elif status == schema.TaskStatus.COMPLETED.value:
            progress = 100.0  # Complete
        elif status == schema.TaskStatus.FAILED.value:
            progress = 30.0  # Something went wrong
    
    # Update task progress in memory and on disk
    tasks[task_id]["progress"] = progress
    # Save updated task info to file
    with open(task_dir / "task_info.json", "w") as f:
        task_info_copy = task_info.copy()
        # Convert datetime objects to ISO strings for JSON serialization
        if isinstance(task_info_copy.get("created_at"), datetime):
            task_info_copy["created_at"] = task_info_copy["created_at"].isoformat()
        if isinstance(task_info_copy.get("updated_at"), datetime):
            task_info_copy["updated_at"] = task_info_copy["updated_at"].isoformat()
        json.dump(task_info_copy, f, default=str)
    
    # Create response
    response = schema.TaskStatusResponse(
        id=task_id,
        name=task_info["name"],
        status=task_info["status"],
        created_at=task_info["created_at"],
        updated_at=task_info["updated_at"],
        progress=progress,
        current_iteration=results_count,
        total_iterations=total_iterations,
        best_result=best_result
    )
    
    return response


@router.get("/tasks/{task_id}/export")
async def export_task_data(
    task_id: str = Path(..., description="Task ID"),
    format: str = Query("json", description="Export format (json or csv)"),
):
    """
    Export complete task data.
    """
    # Check if task exists
    if task_id not in tasks:
        # Try to load task from disk
        task_dir = PathLib(settings.TASK_DIR) / task_id
        task_info_file = task_dir / "task_info.json"
        
        if task_info_file.exists():
            try:
                with open(task_info_file, "r") as f:
                    task_info = json.load(f)
                
                # Add to in-memory tasks dictionary
                tasks[task_id] = task_info
                logger.debug(f"Loaded task {task_id} from disk for export")
            except Exception as e:
                logger.error(f"Failed to load task info for {task_id}: {e}", exc_info=True)
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        else:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Create export data structure
    task_dir = PathLib(settings.TASK_DIR) / task_id
    export_file = task_dir / f"export.{format}"
    
    # Export data in requested format
    if format.lower() == "json":
        # Load all task data
        export_data = {
            "task_info": tasks[task_id],
            "parameter_space": None,
            "strategy": None,
            "initial_designs": None,
            "results": None,
        }
        
        # Load parameter space
        ps_file = task_dir / "parameter_space.json"
        if ps_file.exists():
            try:
                with open(ps_file, "r") as f:
                    export_data["parameter_space"] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load parameter space for export: {e}", exc_info=False)
        
        # Load strategy
        strategy_file = task_dir / "strategy.json"
        if strategy_file.exists():
            try:
                with open(strategy_file, "r") as f:
                    export_data["strategy"] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load strategy for export: {e}", exc_info=False)
        
        # Load initial designs
        designs_file = task_dir / "initial_designs.json"
        if designs_file.exists():
            try:
                with open(designs_file, "r") as f:
                    export_data["initial_designs"] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load initial designs for export: {e}", exc_info=False)
        
        # Load results
        results_file = task_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    export_data["results"] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load results for export: {e}", exc_info=False)
        
        # Save to export file
        with open(export_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
    
    elif format.lower() == "csv":
        # CSV export
        import csv
        
        # Load results if they exist
        results = []
        results_file = task_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    results = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load results for CSV export: {e}", exc_info=False)
        
        # Create CSV file
        with open(export_file, "w", newline='') as f:
            if not results:
                # No results, just write a header
                writer = csv.writer(f)
                writer.writerow(["task_id", "No results available"])
                writer.writerow([task_id, ""])
            else:
                # Determine all parameters and objectives from results
                all_params = set()
                all_objectives = set()
                for result in results:
                    if 'parameters' in result:
                        all_params.update(result['parameters'].keys())
                    if 'objectives' in result:
                        all_objectives.update(result['objectives'].keys())
                
                # Create writer and write header
                writer = csv.writer(f)
                param_cols = sorted(all_params)
                obj_cols = sorted(all_objectives)
                
                header = ["design_id"] + [f"param.{p}" for p in param_cols] + [f"obj.{o}" for o in obj_cols]
                writer.writerow(header)
                
                # Write data rows
                for i, result in enumerate(results):
                    row = [result.get('id', f"design_{i+1}")]
                    
                    # Add parameter values
                    params = result.get('parameters', {})
                    for p in param_cols:
                        row.append(params.get(p, ""))
                    
                    # Add objective values
                    objs = result.get('objectives', {})
                    for o in obj_cols:
                        row.append(objs.get(o, ""))
                    
                    writer.writerow(row)
    
    else:
        # Unsupported format
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")
    
    return FileResponse(
        path=export_file,
        filename=f"task_{task_id}_export.{format}",
        media_type="application/json" if format.lower() == "json" else "text/csv",
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
    task_dir = PathLib(settings.TASK_DIR) / task_id
    
    # Check if task exists, try to load if not in memory
    if task_id not in tasks:
        task_info_file = task_dir / "task_info.json"
        if task_info_file.exists():
            try:
                with open(task_info_file, "r") as f:
                    tasks[task_id] = json.load(f)
                logger.info(f"Loaded task {task_id} from disk for restart")
            except Exception as e:
                logger.error(f"Failed to load task {task_id} for restart: {e}", exc_info=True)
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be loaded")
        else:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Update task status and time
    now = datetime.now()
    tasks[task_id]["status"] = schema.TaskStatus.RUNNING.value
    tasks[task_id]["updated_at"] = now
    
    # Clean up optimizer instance if it exists
    if task_id in optimizers:
        del optimizers[task_id]
        logger.info(f"Removed existing optimizer for task {task_id}")
    
    # Handle history based on restart strategy
    if restart_config.strategy == "reset":
        if not restart_config.preserve_history:
            # Clear results
            logger.info(f"Clearing results for task {task_id} (preserve_history=False)")
            results_file = task_dir / "results.json"
            with open(results_file, "w") as f:
                json.dump([], f)
            
            # Reset progress
            tasks[task_id]["progress"] = 0.0
        else:
            logger.info(f"Keeping results for task {task_id} (preserve_history=True)")
            # Keep results but reset progress if needed
            results_file = task_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, "r") as f:
                        results_data = json.load(f)
                    tasks[task_id]["progress"] = min(50.0, len(results_data) * 5)  # Rough estimate
                except Exception as e:
                    logger.warning(f"Failed to read results file for progress estimation: {e}", exc_info=False)
    elif restart_config.strategy == "continue":
        # Just change status to RUNNING
        logger.info(f"Continuing task {task_id} with existing results")
        # No additional action needed
    else:
        # Unknown strategy
        logger.warning(f"Unknown restart strategy: {restart_config.strategy}, defaulting to 'continue'")
        # No additional action needed
    
    # Save updated task info
    task_info_file = task_dir / "task_info.json"
    with open(task_info_file, "w") as f:
        task_info = tasks[task_id].copy()
        # Convert datetime objects to ISO strings for JSON serialization
        if isinstance(task_info.get("created_at"), datetime):
            task_info["created_at"] = task_info["created_at"].isoformat()
        if isinstance(task_info.get("updated_at"), datetime):
            task_info["updated_at"] = task_info["updated_at"].isoformat()
        json.dump(task_info, f, default=str)
    
    return {
        "message": f"Task {task_id} restarted with strategy: {restart_config.strategy}",
        "preserve_history": restart_config.preserve_history,
        "status": tasks[task_id]["status"],
        "updated_at": tasks[task_id]["updated_at"],
    }


@router.get("/diagnostics/{task_id}", response_model=schema.Diagnostics)
async def get_diagnostics(task_id: str = Path(..., description="Task ID")):
    """
    Get diagnostics information for debugging.
    """
    task_dir = PathLib(settings.TASK_DIR) / task_id
    
    # Check if task exists, try to load if not in memory
    if task_id not in tasks:
        task_info_file = task_dir / "task_info.json"
        if task_info_file.exists():
            try:
                with open(task_info_file, "r") as f:
                    tasks[task_id] = json.load(f)
                logger.info(f"Loaded task {task_id} from disk for diagnostics")
            except Exception as e:
                logger.error(f"Failed to load task {task_id} for diagnostics: {e}", exc_info=True)
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be loaded")
        else:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Gather diagnostics
    diagnostics = {}
    
    # 1. Check parameter space
    ps_file = task_dir / "parameter_space.json"
    if ps_file.exists():
        try:
            with open(ps_file, "r") as f:
                ps_data = json.load(f)
            param_count = len(ps_data.get("parameters", {}))
            obj_count = len(ps_data.get("objectives", {}))
            constr_count = len(ps_data.get("constraints", {}))
            diagnostics["parameter_space"] = f"valid ({param_count} params, {obj_count} objs, {constr_count} constrs)"
        except Exception as e:
            diagnostics["parameter_space"] = f"error: {str(e)}"
    else:
        diagnostics["parameter_space"] = "not_defined"
    
    # 2. Check if model is trained (has observed results)
    results_file = task_dir / "results.json"
    model_trained = False
    pending_experiments = []
    
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                results_data = json.load(f)
            if results_data:
                model_trained = True
                diagnostics["model_trained"] = True
                diagnostics["results_count"] = len(results_data)
            else:
                diagnostics["model_trained"] = False
                diagnostics["results_count"] = 0
        except Exception as e:
            diagnostics["model_trained"] = False
            diagnostics["results_count"] = 0
            diagnostics["results_error"] = str(e)
    else:
        diagnostics["model_trained"] = False
        diagnostics["results_count"] = 0
    
    # 3. Check for initial designs
    designs_file = task_dir / "initial_designs.json"
    if designs_file.exists():
        try:
            with open(designs_file, "r") as f:
                designs_data = json.load(f)
            
            # Check if we have results for each design
            result_ids = set()
            if results_file.exists():
                try:
                    with open(results_file, "r") as f:
                        results_data = json.load(f)
                    for r in results_data:
                        if isinstance(r, dict) and 'id' in r:
                            result_ids.add(r['id'])
                except Exception:
                    pass
            
            # Find designs without results
            for design in designs_data:
                if isinstance(design, dict) and 'id' in design and design['id'] not in result_ids:
                    pending_experiments.append(design['id'])
            
            diagnostics["initial_designs_count"] = len(designs_data)
            diagnostics["pending_experiments"] = pending_experiments
        except Exception as e:
            diagnostics["initial_designs_error"] = str(e)
    else:
        diagnostics["initial_designs_count"] = 0
        diagnostics["pending_experiments"] = []
    
    # 4. Check for strategy
    strategy_file = task_dir / "strategy.json"
    if strategy_file.exists():
        try:
            with open(strategy_file, "r") as f:
                strategy_data = json.load(f)
            diagnostics["strategy"] = f"valid ({strategy_data.get('algorithm', 'unknown')}, acq={strategy_data.get('acquisition_function', 'unknown')})"
        except Exception as e:
            diagnostics["strategy"] = f"error: {str(e)}"
    else:
        diagnostics["strategy"] = "not_defined"
    
    # 5. Get recent exception if any
    recent_exception = None
    error_log_file = task_dir / "error.log"
    if error_log_file.exists():
        try:
            with open(error_log_file, "r") as f:
                last_lines = f.readlines()[-10:]  # Get last 10 lines
                if last_lines:
                    recent_exception = "".join(last_lines)
        except Exception:
            pass
    
    # 6. Check optimizer instance
    if task_id in optimizers:
        diagnostics["optimizer_in_memory"] = True
    else:
        diagnostics["optimizer_in_memory"] = False
    
    # 7. Get last recommendation time from log or API access time
    last_recommendation_time = None
    # For now, just use current time as a proxy
    last_recommendation_time = datetime.now()
    
    # Return diagnostics
    return schema.Diagnostics(
        parameter_space=diagnostics.get("parameter_space", "unknown"),
        model_trained=diagnostics.get("model_trained", False),
        recent_exception=recent_exception,
        pending_experiments=diagnostics.get("pending_experiments", []),
        last_recommendation_time=last_recommendation_time,
        **{k: v for k, v in diagnostics.items() if k not in ["parameter_space", "model_trained", "pending_experiments"]}
    ) 
