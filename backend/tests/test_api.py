import json
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Test data
test_parameter_space = {
    "name": "Test Optimization",
    "parameters": [
        {
            "name": "x1",
            "type": "continuous",
            "min": 0.0,
            "max": 1.0
        },
        {
            "name": "x2",
            "type": "categorical",
            "values": ["A", "B", "C"]
        }
    ],
    "objectives": [
        {
            "name": "y1",
            "type": "maximize"
        }
    ],
    "constraints": []
}

test_strategy = {
    "algorithm": "bayesian",
    "config": {
        "acquisition_function": "ei",
        "kernel": "matern",
        "exploration_weight": 0.5,
        "noise_level": 0.1,
        "multi_objective": False,
        "moo_acquisition": None,
        "noisy_moo": False
    },
    "initial_sampling": {
        "method": "lhs",
        "samples": 5
    },
    "batch_size": 1,
    "iterations": 20
}

test_results = {
    "results": [
        {
            "design_id": "design_1",
            "objectives": {"y1": 0.75},
            "metadata": {
                "timestamp": "2023-06-15T10:30:00Z",
                "experimenter": "test_user",
                "notes": "Test result"
            }
        }
    ]
}


# Health check test
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# Parameter space tests
def test_create_parameter_space():
    response = client.post("/api/parameter-space", json=test_parameter_space)
    assert response.status_code == 200
    result = response.json()
    assert "task_id" in result
    assert result["status"] == "created"
    return result["task_id"]


def test_get_parameter_space():
    # First create a parameter space
    task_id = test_create_parameter_space()
    
    # Then get it
    response = client.get(f"/api/parameter-space/{task_id}")
    assert response.status_code == 200
    result = response.json()
    assert result["name"] == test_parameter_space["name"]
    assert len(result["parameters"]) == len(test_parameter_space["parameters"])


def test_update_parameter_space():
    # First create a parameter space
    task_id = test_create_parameter_space()
    
    # Then update it
    updated_space = test_parameter_space.copy()
    updated_space["name"] = "Updated Test"
    response = client.put(f"/api/parameter-space/{task_id}", json=updated_space)
    assert response.status_code == 200
    
    # Verify the update
    response = client.get(f"/api/parameter-space/{task_id}")
    assert response.status_code == 200
    result = response.json()
    assert result["name"] == "Updated Test"


# Strategy tests
def test_set_strategy():
    # First create a parameter space
    task_id = test_create_parameter_space()
    
    # Then set a strategy
    response = client.post(f"/api/strategy/{task_id}", json=test_strategy)
    assert response.status_code == 200
    
    # Verify the strategy
    response = client.get(f"/api/strategy/{task_id}")
    assert response.status_code == 200
    result = response.json()
    assert result["algorithm"] == test_strategy["algorithm"]


# Design tests
def test_get_initial_designs():
    # First create a parameter space
    task_id = test_create_parameter_space()
    
    # Then get initial designs
    response = client.get(f"/api/designs/{task_id}/initial")
    assert response.status_code == 200
    result = response.json()
    assert "designs" in result
    assert len(result["designs"]) > 0


# Results tests
def test_submit_results():
    # First create a parameter space
    task_id = test_create_parameter_space()
    
    # Get initial designs
    response = client.get(f"/api/designs/{task_id}/initial")
    assert response.status_code == 200
    
    # Submit results
    response = client.post(f"/api/results/{task_id}", json=test_results)
    assert response.status_code == 200
    result = response.json()
    assert "message" in result


# Next designs tests
def test_get_next_designs():
    # First create a parameter space
    task_id = test_create_parameter_space()
    
    # Get initial designs
    response = client.get(f"/api/designs/{task_id}/initial")
    assert response.status_code == 200
    
    # Submit results
    response = client.post(f"/api/results/{task_id}", json=test_results)
    assert response.status_code == 200
    
    # Get next designs
    response = client.get(f"/api/designs/{task_id}/next")
    assert response.status_code == 200
    result = response.json()
    assert "designs" in result
    assert len(result["designs"]) > 0


# Task management tests
def test_get_tasks():
    # First create a parameter space
    test_create_parameter_space()
    
    # Then get tasks
    response = client.get("/api/tasks")
    assert response.status_code == 200
    result = response.json()
    assert "tasks" in result
    assert len(result["tasks"]) > 0


def test_get_task_status():
    # First create a parameter space
    task_id = test_create_parameter_space()
    
    # Then get status
    response = client.get(f"/api/tasks/{task_id}/status")
    assert response.status_code == 200
    result = response.json()
    assert "status" in result


def test_task_export():
    # First create a parameter space
    task_id = test_create_parameter_space()
    
    # Then export
    response = client.get(f"/api/tasks/{task_id}/export")
    assert response.status_code == 200


# Error handling tests
def test_get_nonexistent_task():
    response = client.get("/api/parameter-space/nonexistent")
    assert response.status_code == 404 
