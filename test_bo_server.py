from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import uuid
import numpy as np
import os
from pathlib import Path as PathLib

app = FastAPI(
    title="Simple BO Test Server",
    description="A simplified server for testing Catalyst BO JSON",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
tasks = {}
parameter_spaces = {}
designs = {}
results = {}

# Create data directory
data_dir = PathLib("./data")
task_dir = data_dir / "tasks"
os.makedirs(task_dir, exist_ok=True)

# Models
class ParameterSpaceConfig(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: List[Dict[str, Any]]
    objectives: List[Dict[str, Any]]
    constraints: Optional[List[Dict[str, Any]]] = []

class ParameterSpaceResponse(BaseModel):
    task_id: str
    message: str

class Design(BaseModel):
    id: str
    parameters: Dict[str, Any]

class DesignResponse(BaseModel):
    designs: List[Design]

class ResultSubmit(BaseModel):
    parameters: Dict[str, Any]
    objectives: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

class ResultsSubmission(BaseModel):
    results: List[ResultSubmit]

# Helper functions
def generate_id():
    return str(uuid.uuid4())

def latin_hypercube_sampling(n_samples, parameter_space):
    """Simple Latin Hypercube Sampling implementation"""
    continuous_params = []
    categorical_params = []
    integer_params = []
    
    # Separate parameters by type
    for param in parameter_space["parameters"]:
        if param["type"] == "continuous":
            continuous_params.append(param)
        elif param["type"] == "categorical":
            categorical_params.append(param)
        elif param["type"] == "integer":
            integer_params.append(param)
    
    # Generate samples
    samples = []
    for _ in range(n_samples):
        sample = {}
        
        # Handle continuous parameters
        for param in continuous_params:
            min_val, max_val = param["bounds"]
            sample[param["name"]] = np.random.uniform(min_val, max_val)
        
        # Handle integer parameters
        for param in integer_params:
            min_val, max_val = param["bounds"]
            sample[param["name"]] = np.random.randint(min_val, max_val + 1)
        
        # Handle categorical parameters
        for param in categorical_params:
            choices = param["choices"]
            sample[param["name"]] = np.random.choice(choices)
        
        samples.append(sample)
    
    # Apply constraints if any
    valid_samples = []
    for sample in samples:
        valid = True
        
        # Check constraints
        for constraint in parameter_space.get("constraints", []):
            if constraint["type"] == "sum":
                param_names = constraint["parameters"]
                relation = constraint["relation"]
                value = constraint["value"]
                
                param_sum = sum(sample[name] for name in param_names)
                
                if relation == "<=" and param_sum > value:
                    valid = False
                elif relation == ">=" and param_sum < value:
                    valid = False
                elif relation == "==" and param_sum != value:
                    valid = False
        
        if valid:
            valid_samples.append(sample)
    
    # If we don't have enough valid samples, generate more
    while len(valid_samples) < n_samples:
        sample = {}
        
        # Handle continuous parameters
        for param in continuous_params:
            min_val, max_val = param["bounds"]
            sample[param["name"]] = np.random.uniform(min_val, max_val)
        
        # Handle integer parameters
        for param in integer_params:
            min_val, max_val = param["bounds"]
            sample[param["name"]] = np.random.randint(min_val, max_val + 1)
        
        # Handle categorical parameters
        for param in categorical_params:
            choices = param["choices"]
            sample[param["name"]] = np.random.choice(choices)
        
        # Check constraints
        valid = True
        for constraint in parameter_space.get("constraints", []):
            if constraint["type"] == "sum":
                param_names = constraint["parameters"]
                relation = constraint["relation"]
                value = constraint["value"]
                
                param_sum = sum(sample[name] for name in param_names)
                
                if relation == "<=" and param_sum > value:
                    valid = False
                elif relation == ">=" and param_sum < value:
                    valid = False
                elif relation == "==" and param_sum != value:
                    valid = False
        
        if valid:
            valid_samples.append(sample)
            if len(valid_samples) >= n_samples:
                break
    
    return valid_samples[:n_samples]

def random_next_designs(n_samples, parameter_space):
    """Generate random next designs"""
    return latin_hypercube_sampling(n_samples, parameter_space)

# API Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/parameter-space", response_model=ParameterSpaceResponse)
async def create_parameter_space(data: ParameterSpaceConfig):
    task_id = generate_id()
    
    # Store parameter space
    parameter_spaces[task_id] = data.dict()
    
    # Create task entry
    tasks[task_id] = {
        "task_id": task_id,
        "name": data.name,
        "status": "configured",
        "progress": 0.0,
    }
    
    # Save to file
    task_path = task_dir / task_id
    os.makedirs(task_path, exist_ok=True)
    
    with open(task_path / "parameter_space.json", "w") as f:
        json.dump(parameter_spaces[task_id], f, indent=2)
    
    with open(task_path / "task_info.json", "w") as f:
        json.dump(tasks[task_id], f, indent=2)
    
    return ParameterSpaceResponse(
        task_id=task_id,
        message=f"Parameter space '{data.name}' created successfully"
    )

@app.get("/api/designs/{task_id}/initial", response_model=DesignResponse)
async def get_initial_designs(
    task_id: str = Path(..., description="Task ID"),
    samples: int = Query(5, description="Number of samples to generate"),
):
    if task_id not in parameter_spaces:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Generate designs using Latin Hypercube Sampling
    parameter_space = parameter_spaces[task_id]
    design_points = latin_hypercube_sampling(samples, parameter_space)
    
    # Convert to Design objects
    design_objects = [
        Design(id=generate_id(), parameters=point)
        for point in design_points
    ]
    
    # Store designs
    designs[task_id] = design_objects
    
    # Save to file
    task_path = task_dir / task_id
    with open(task_path / "initial_designs.json", "w") as f:
        json.dump([d.dict() for d in design_objects], f, indent=2)
    
    # Update task status
    tasks[task_id]["status"] = "ready_for_results"
    tasks[task_id]["progress"] = 10.0
    
    with open(task_path / "task_info.json", "w") as f:
        json.dump(tasks[task_id], f, indent=2)
    
    return DesignResponse(designs=design_objects)

@app.post("/api/results/{task_id}")
async def submit_results(
    results_data: ResultsSubmission,
    task_id: str = Path(..., description="Task ID"),
):
    if task_id not in parameter_spaces:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Store results
    if task_id not in results:
        results[task_id] = []
    
    results[task_id].extend([r.dict() for r in results_data.results])
    
    # Save to file
    task_path = task_dir / task_id
    with open(task_path / "results.json", "w") as f:
        json.dump(results[task_id], f, indent=2)
    
    # Update task status
    tasks[task_id]["status"] = "optimizing"
    tasks[task_id]["progress"] = min(80.0, 10.0 + len(results[task_id]) * 5.0)
    
    with open(task_path / "task_info.json", "w") as f:
        json.dump(tasks[task_id], f, indent=2)
    
    return {
        "message": f"{len(results_data.results)} results submitted successfully",
        "results_count": len(results[task_id]),
        "progress": tasks[task_id]["progress"]
    }

@app.get("/api/designs/{task_id}/next", response_model=DesignResponse)
async def get_next_designs(
    task_id: str = Path(..., description="Task ID"),
    batch_size: int = Query(1, description="Number of designs to generate"),
):
    if task_id not in parameter_spaces:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    if task_id not in results or not results[task_id]:
        raise HTTPException(status_code=400, detail="No results submitted yet")
    
    # Generate next designs (in a real system, this would use Bayesian Optimization)
    parameter_space = parameter_spaces[task_id]
    next_points = random_next_designs(batch_size, parameter_space)
    
    # Convert to Design objects
    next_designs = [
        Design(id=generate_id(), parameters=point)
        for point in next_points
    ]
    
    # Save to file
    task_path = task_dir / task_id
    with open(task_path / "next_designs.json", "w") as f:
        json.dump([d.dict() for d in next_designs], f, indent=2)
    
    return DesignResponse(designs=next_designs)

@app.get("/api/tasks/{task_id}/status")
async def get_task_status(task_id: str = Path(..., description="Task ID")):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task_info = tasks[task_id]
    
    # Get results count
    results_count = 0
    if task_id in results:
        results_count = len(results[task_id])
    
    return {
        "status": task_info["status"],
        "progress": task_info["progress"],
        "results_count": results_count
    }

@app.get("/api/tasks")
async def get_tasks():
    return {"tasks": list(tasks.values())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
