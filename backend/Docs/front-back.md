# Frontend-Backend Interface Document

## Overview

This document outlines the interface specification between the Frontend BO Optimizer GUI and the Backend FastAPI BO Engine. It details the API endpoints, request/response formats, and communication protocols required for seamless integration.

## API Endpoints

### Parameter Space Configuration

#### Create Parameter Space

- **Endpoint:** `POST /api/parameter-space`
- **Description:** Creates a new optimization task with the defined parameter space
- **Request Body:**

```json
{
  "parameters": {
    "x1": {
      "type": "continuous",
      "lower_bound": 0.0,
      "upper_bound": 10.0,
      "log_scale": false,
      "description": "First parameter"
    },
    "x2": {
      "type": "integer",
      "lower_bound": 1,
      "upper_bound": 10,
      "log_scale": false,
      "description": "Second parameter"
    },
    "x3": {
      "type": "categorical",
      "categories": ["A", "B", "C"],
      "description": "Third parameter"
    }
  },
  "objectives": {
    "y1": "minimize",
    "y2": "maximize"
  },
  "constraints": [
    {
      "type": "less_than",
      "expression": "x1 + x2",
      "value": 15.0
    }
  ]
}
```

- **Response:**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "created",
  "created_at": "2023-04-01T10:30:00Z",
  "message": "Parameter space created successfully"
}
```

#### Get Parameter Space

- **Endpoint:** `GET /api/parameter-space/{task_id}`
- **Description:** Retrieves the defined parameter space configuration
- **Response:** Same format as the request body for creating a parameter space

### Optimization Strategy Configuration

#### Set Strategy

- **Endpoint:** `POST /api/strategy/{task_id}`
- **Description:** Sets the optimization strategy for the given task
- **Request Body:**

```json
{
  "algorithm": "bayesian",
  "acquisition_function": "expected_improvement",
  "surrogate_model": "gaussian_process",
  "acquisition_optimizer": "lbfgs",
  "exploration_weight": 0.1,
  "model_update_interval": 1,
  "batch_size": 1,
  "random_seed": 42,
  "hyperparameters": {
    "kernel": "matern",
    "nu": 2.5,
    "length_scale_bounds": [0.1, 10.0]
  }
}
```

- **Response:**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "strategy_set",
  "message": "Optimization strategy set successfully"
}
```

#### Get Strategy

- **Endpoint:** `GET /api/strategy/{task_id}`
- **Description:** Retrieves the current optimization strategy
- **Response:** Same format as the request body for setting a strategy

### Experiment Design

#### Get Initial Design

- **Endpoint:** `GET /api/designs/{task_id}/initial?n=10&design_type=latin_hypercube`
- **Description:** Gets initial design points for experimentation
- **Query Parameters:**
  - `n`: Number of design points (integer)
  - `design_type`: Type of design ("random", "latin_hypercube", "factorial", "sobol", "custom")
- **Response:**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "design_points": [
    {"x1": 1.2, "x2": 5, "x3": "A"},
    {"x1": 8.7, "x2": 3, "x3": "B"},
    {"x1": 4.5, "x2": 7, "x3": "C"}
  ],
  "design_type": "latin_hypercube",
  "message": "Initial design generated successfully"
}
```

### Results Submission

#### Submit Results

- **Endpoint:** `POST /api/results/{task_id}`
- **Description:** Submits experiment results for one or more design points
- **Request Body:**

```json
{
  "results": [
    {
      "parameters": {"x1": 1.2, "x2": 5, "x3": "A"},
      "objectives": {"y1": 10.5, "y2": 8.3},
      "constraints": {"x1 + x2": 6.2},
      "metadata": {"execution_time": 120.5}
    },
    {
      "parameters": {"x1": 8.7, "x2": 3, "x3": "B"},
      "objectives": {"y1": 5.2, "y2": 12.1},
      "constraints": {"x1 + x2": 11.7},
      "metadata": {"execution_time": 95.3}
    }
  ]
}
```

- **Response:**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "accepted_count": 2,
  "rejected_count": 0,
  "message": "Results submitted successfully"
}
```

### Next Design Point Recommendation

#### Get Next Design Points

- **Endpoint:** `GET /api/designs/{task_id}/next?n=1`
- **Description:** Gets the next batch of recommended design points
- **Query Parameters:**
  - `n`: Number of points to recommend (integer)
- **Response:**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "design_points": [
    {"x1": 6.3, "x2": 4, "x3": "B"}
  ],
  "expected_outcomes": [
    {"y1": 4.2, "y2": 9.7}
  ],
  "acquisition_values": [0.85],
  "message": "Next design points generated successfully"
}
```

### Model Interrogation

#### Make Predictions

- **Endpoint:** `POST /api/predict/{task_id}`
- **Description:** Makes model predictions for specific parameter combinations
- **Request Body:**

```json
{
  "points": [
    {"x1": 5.0, "x2": 6, "x3": "A"},
    {"x1": 7.5, "x2": 2, "x3": "C"}
  ]
}
```

- **Response:**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "predictions": [
    {
      "parameters": {"x1": 5.0, "x2": 6, "x3": "A"},
      "objectives": {"y1": {"mean": 7.2, "std": 0.8}, "y2": {"mean": 9.5, "std": 1.2}}
    },
    {
      "parameters": {"x1": 7.5, "x2": 2, "x3": "C"},
      "objectives": {"y1": {"mean": 6.3, "std": 1.1}, "y2": {"mean": 8.7, "std": 0.9}}
    }
  ],
  "message": "Predictions generated successfully"
}
```

#### Get Model Performance

- **Endpoint:** `GET /api/model/{task_id}/performance`
- **Description:** Returns the current model's performance metrics
- **Response:**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "model_type": "gaussian_process",
  "metrics": {
    "r2_score": 0.92,
    "mean_absolute_error": 0.43,
    "root_mean_squared_error": 0.56
  },
  "cross_validation": {
    "folds": 5,
    "mean_r2_score": 0.90,
    "std_r2_score": 0.03
  },
  "hyperparameters": {
    "kernel": "matern",
    "length_scale": [1.2, 0.8, 0.5],
    "noise_level": 0.1
  },
  "message": "Model performance retrieved successfully"
}
```

#### Get Pareto Front

- **Endpoint:** `GET /api/model/{task_id}/pareto-front`
- **Description:** Retrieves the current Pareto front for multi-objective optimization
- **Response:**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "pareto_points": [
    {
      "parameters": {"x1": 3.2, "x2": 5, "x3": "A"},
      "objectives": {"y1": 4.2, "y2": 12.5}
    },
    {
      "parameters": {"x1": 5.8, "x2": 3, "x3": "B"},
      "objectives": {"y1": 5.1, "y2": 15.2}
    },
    {
      "parameters": {"x1": 7.3, "x2": 2, "x3": "C"},
      "objectives": {"y1": 6.3, "y2": 18.7}
    }
  ],
  "message": "Pareto front retrieved successfully"
}
```

#### Get Uncertainty Analysis

- **Endpoint:** `GET /api/model/{task_id}/uncertainty?resolution=20`
- **Description:** Retrieves uncertainty maps across the parameter space
- **Query Parameters:**
  - `resolution`: Grid resolution for uncertainty mapping (integer)
- **Response:**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "parameter_grids": {
    "x1": [0.0, 0.5, 1.0, ..., 10.0],
    "x2": [1, 2, 3, ..., 10]
  },
  "uncertainty_maps": {
    "y1": [
      [0.5, 0.6, 0.7, ...],
      [0.4, 0.5, 0.6, ...],
      ...
    ],
    "y2": [
      [0.8, 0.9, 1.0, ...],
      [0.7, 0.8, 0.9, ...],
      ...
    ]
  },
  "message": "Uncertainty analysis retrieved successfully"
}
```

### Task Management

#### List Tasks

- **Endpoint:** `GET /api/tasks`
- **Description:** Lists all optimization tasks
- **Response:**

```json
{
  "tasks": [
    {
      "task_id": "123e4567-e89b-12d3-a456-426614174000",
      "status": "running",
      "created_at": "2023-04-01T10:30:00Z",
      "updated_at": "2023-04-01T15:45:00Z",
      "name": "Catalyst Optimization",
      "description": "Optimizing catalyst formulation for higher yield",
      "num_parameters": 3,
      "num_objectives": 2,
      "evaluations_completed": 15,
      "best_objective_values": {"y1": 3.2, "y2": 18.7}
    },
    {
      "task_id": "223e4567-e89b-12d3-a456-426614174001",
      "status": "completed",
      "created_at": "2023-03-15T09:20:00Z",
      "updated_at": "2023-03-16T14:30:00Z",
      "name": "Reactor Temperature",
      "description": "Optimizing reactor temperature profile",
      "num_parameters": 5,
      "num_objectives": 1,
      "evaluations_completed": 25,
      "best_objective_values": {"y1": 2.1}
    }
  ],
  "total_count": 2,
  "message": "Tasks retrieved successfully"
}
```

#### Get Task Details

- **Endpoint:** `GET /api/tasks/{task_id}`
- **Description:** Gets detailed information about a specific task
- **Response:**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "running",
  "created_at": "2023-04-01T10:30:00Z",
  "updated_at": "2023-04-01T15:45:00Z",
  "name": "Catalyst Optimization",
  "description": "Optimizing catalyst formulation for higher yield",
  "parameter_space_summary": {
    "parameters": ["x1", "x2", "x3"],
    "objectives": ["y1", "y2"],
    "constraints": 1
  },
  "strategy_summary": {
    "algorithm": "bayesian",
    "surrogate_model": "gaussian_process",
    "acquisition_function": "expected_improvement"
  },
  "progress": {
    "evaluations_completed": 15,
    "evaluations_planned": 30,
    "best_objective_values": {"y1": 3.2, "y2": 18.7},
    "convergence_metrics": {
      "improvement_rate": 0.12,
      "exploration_vs_exploitation": 0.65
    }
  },
  "message": "Task details retrieved successfully"
}
```

#### Update Task

- **Endpoint:** `PATCH /api/tasks/{task_id}`
- **Description:** Updates task metadata or status
- **Request Body:**

```json
{
  "name": "Updated Catalyst Optimization",
  "description": "Revised description with additional details",
  "status": "paused"
}
```

- **Response:**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "message": "Task updated successfully"
}
```

#### Delete Task

- **Endpoint:** `DELETE /api/tasks/{task_id}`
- **Description:** Deletes a task and all associated data
- **Response:**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "message": "Task deleted successfully"
}
```

### Diagnostics

#### Get Convergence Diagnostics

- **Endpoint:** `GET /api/diagnostics/{task_id}/convergence`
- **Description:** Retrieves convergence metrics over the optimization history
- **Response:**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "iteration_history": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  "objective_history": {
    "y1": [10.2, 9.1, 8.3, 7.5, 6.8, 6.2, 5.7, 5.3, 4.9, 4.5, 4.2, 3.9, 3.6, 3.4, 3.2],
    "y2": [5.3, 6.8, 8.1, 9.5, 10.7, 11.8, 12.9, 13.8, 14.7, 15.5, 16.2, 16.9, 17.5, 18.1, 18.7]
  },
  "hyperparameter_history": {
    "length_scale": [
      [0.8, 0.5, 0.3],
      [0.9, 0.6, 0.4],
      // ... more history entries
      [1.2, 0.8, 0.5]
    ]
  },
  "acquisition_value_history": [0.92, 0.87, 0.83, 0.78, 0.74, 0.70, 0.67, 0.63, 0.60, 0.57, 0.54, 0.51, 0.48, 0.46, 0.43],
  "exploration_exploitation_ratio_history": [0.85, 0.82, 0.79, 0.75, 0.72, 0.69, 0.67, 0.64, 0.62, 0.59, 0.57, 0.55, 0.53, 0.51, 0.49],
  "message": "Convergence diagnostics retrieved successfully"
}
```

#### Export Results

- **Endpoint:** `GET /api/export/{task_id}?format=csv`
- **Description:** Exports all results in the specified format
- **Query Parameters:**
  - `format`: Export format ("csv", "json", "excel")
- **Response:** Binary file download

### Real-time Notifications

#### WebSocket Connection

- **Endpoint:** `WebSocket /ws/tasks/{task_id}`
- **Description:** Establishes a WebSocket connection to receive real-time updates
- **Event Types:**
  - `task_status_changed`: Sent when task status changes
  - `evaluation_completed`: Sent when a new evaluation is completed
  - `model_updated`: Sent when the surrogate model is updated
  - `recommendation_ready`: Sent when new recommendations are available

- **Event Format:**

```json
{
  "event_type": "evaluation_completed",
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "timestamp": "2023-04-01T15:45:00Z",
  "data": {
    "parameters": {"x1": 6.3, "x2": 4, "x3": "B"},
    "objectives": {"y1": 4.2, "y2": 9.7},
    "evaluations_completed": 16,
    "is_best": true
  }
}
```

## Error Handling

All endpoints should follow a consistent error handling pattern:

```json
{
  "error": true,
  "code": 400,
  "message": "Invalid parameter values provided",
  "details": {
    "x1": "Value must be between 0 and 10",
    "x3": "Value must be one of ['A', 'B', 'C']"
  }
}
```

## Authentication and Authorization

- Bearer token authentication
- All API requests must include the Authorization header: `Authorization: Bearer <token>`
- Different permission levels:
  - Read-only: Can view tasks but not modify them
  - Contributor: Can view tasks and submit results
  - Admin: Full access, including creating and deleting tasks
