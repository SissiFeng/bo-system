# Frontend and Backend Interface for Bayesian Optimization System

Remember the BO microservice frontend we've been developing? I want to separate this BO system from the canvas but make it easy for the canvas to call. Below are the interfaces between the backend to be developed and the already developed frontend:

### Summary of Frontend and Backend Interfaces

Based on the provided backend system structure, here are the main interfaces needed between the frontend (BO Optimizer GUI) and the backend (FastAPI BO Engine):

## 1. Parameter Space Configuration Interface

### 1.1 Create/Initialize Parameter Space

```plaintext
POST /api/parameter-space
```

- **Function**: Create a new optimization task, define parameter space
- **Request Body**:

```json
{
  "name": "Optimization Task Name",
  "parameters": [
    {
      "name": "x1",
      "type": "continuous",
      "min": 0,
      "max": 1
    },
    {
      "name": "x2",
      "type": "categorical",
      "values": ["A", "B", "C"]
    },
    {
      "name": "x3",
      "type": "discrete",
      "min": 1,
      "max": 10,
      "step": 1
    }
  ],
  "objectives": [
    {
      "name": "y1",
      "type": "maximize"
    },
    {
      "name": "y2",
      "type": "minimize"
    }
  ],
  "constraints": [
    {
      "expression": "x1 + x2",
      "type": "sum_equals",
      "value": 1
    }
  ]
}
```


- **Response**:

```json
{
  "task_id": "12345",
  "status": "created",
  "message": "Parameter space created successfully"
}
```




### 1.2 Get Parameter Space Configuration

```plaintext
GET /api/parameter-space/{task_id}
```

- **Function**: Get the defined parameter space configuration
- **Response**: Complete parameter space configuration


### 1.3 Update Parameter Space

```plaintext
PUT /api/parameter-space/{task_id}
```

- **Function**: Update existing parameter space configuration
- **Request Body**: Similar to creation, but can be partially updated


## 2. Optimization Strategy Configuration Interface

### 2.1 Set Optimization Strategy

```plaintext
POST /api/strategy/{task_id}
```

- **Function**: Configure optimization algorithm and strategy
- **Request Body**:

```json
{
  "algorithm": "bayesian",
  "config": {
    "acquisition_function": "ei",
    "kernel": "matern",
    "exploration_weight": 0.5,
    "noise_level": 0.1,
    "multi_objective": true,
    "moo_acquisition": "ehvi",
    "noisy_moo": true
  },
  "initial_sampling": {
    "method": "lhs",
    "samples": 10
  },
  "batch_size": 5,
  "iterations": 20
}
```


- **Response**: Confirmation that the strategy has been set


### 2.2 Get Optimization Strategy

```plaintext
GET /api/strategy/{task_id}
```

- **Function**: Get the currently configured optimization strategy
- **Response**: Current strategy configuration


## 3. Experimental Design and Results Interface

### 3.1 Get Initial Experimental Design

```plaintext
GET /api/designs/{task_id}/initial
```

- **Function**: Get initial experimental design points
- **Response**:

```json
{
  "designs": [
    {
      "id": "design_1",
      "parameters": {"x1": 0.2, "x2": "A", "x3": 5}
    },
    {
      "id": "design_2",
      "parameters": {"x1": 0.8, "x2": "B", "x3": 3}
    }
  ]
}
```




### 3.2 Submit Experiment Results

```plaintext
POST /api/results/{task_id}
```

- **Function**: Submit experiment results
- **Request Body**:

```json
{
  "results": [
    {
      "design_id": "design_1",
      "objectives": {"y1": 0.75, "y2": 0.25},
      "metadata": {
        "timestamp": "2023-06-15T10:30:00Z",
        "experimenter": "Username",
        "notes": "Experiment notes"
      }
    }
  ]
}
```


- **Response**: Confirmation that results have been received


### 3.3 Get Next Batch of Experimental Designs

```plaintext
GET /api/designs/{task_id}/next
```

- **Function**: Get the next batch of recommended experimental design points
- **Query Parameters**:

- `batch_size`: Number of design points needed
- `strategy`: Optional batch strategy (such as "qei", "kb", etc.)



- **Response**:

```json
{
  "designs": [
    {
      "id": "design_10",
      "parameters": {"x1": 0.4, "x2": "C", "x3": 7},
      "predictions": {
        "y1": {"mean": 0.85, "std": 0.05},
        "y2": {"mean": 0.15, "std": 0.03}
      },
      "uncertainty": 0.04,
      "reason": "High expected improvement value, low uncertainty"
    }
  ]
}
```




## 4. Model and Analysis Interface

### 4.1 Get Model Predictions

```plaintext
POST /api/predict/{task_id}
```

- **Function**: Get model predictions for specified parameter combinations
- **Request Body**:

```json
{
  "parameters": [
    {"x1": 0.5, "x2": "A", "x3": 5},
    {"x1": 0.6, "x2": "B", "x3": 6}
  ]
}
```


- **Response**:

```json
{
  "predictions": [
    {
      "parameters": {"x1": 0.5, "x2": "A", "x3": 5},
      "objectives": {
        "y1": {"mean": 0.8, "std": 0.05},
        "y2": {"mean": 0.2, "std": 0.03}
      }
    }
  ]
}
```




### 4.2 Get Model Performance Metrics

```plaintext
GET /api/model/{task_id}/performance
```

- **Function**: Get the current model's performance metrics
- **Response**:

```json
{
  "metrics": {
    "r2": 0.92,
    "rmse": 0.08,
    "mae": 0.06
  },
  "cross_validation": {
    "cv_scores": [0.91, 0.93, 0.90, 0.94, 0.92],
    "mean_score": 0.92,
    "std_score": 0.015
  }
}
```




### 4.3 Get Pareto Front

```plaintext
GET /api/pareto/{task_id}
```

- **Function**: Get the Pareto front for multi-objective optimization
- **Response**:

```json
{
  "pareto_front": [
    {
      "id": "design_5",
      "parameters": {"x1": 0.3, "x2": "B", "x3": 4},
      "objectives": {"y1": 0.9, "y2": 0.3},
      "uncertainty": 0.02
    }
  ],
  "dominated_solutions": [...],
  "ideal_point": {"y1": 1.0, "y2": 0.0},
  "nadir_point": {"y1": 0.5, "y2": 0.8}
}
```




### 4.4 Get Uncertainty Analysis

```plaintext
GET /api/uncertainty/{task_id}
```

- **Function**: Get analysis of the relationship between predicted and actual measured values
- **Response**:

```json
{
  "prediction_vs_actual": [
    {
      "design_id": "design_1",
      "predicted": {"mean": 0.8, "std": 0.05},
      "actual": 0.75,
      "error": 0.05,
      "within_confidence": true
    }
  ],
  "calibration_metrics": {
    "coverage_probability": 0.95,
    "sharpness": 0.08
  }
}
```




## 5. Task Management Interface

### 5.1 Get Task List

```plaintext
GET /api/tasks
```

- **Function**: Get a list of all optimization tasks
- **Response**: List of task IDs and basic information


### 5.2 Get Task Status

```plaintext
GET /api/tasks/{task_id}/status
```

- **Function**: Get the current status of a specific task
- **Response**:

```json
{
  "status": "running",
  "current_iteration": 15,
  "total_iterations": 20,
  "best_result": {
    "parameters": {"x1": 0.4, "x2": "C", "x3": 7},
    "objectives": {"y1": 0.92, "y2": 0.18}
  },
  "last_updated": "2023-06-15T14:30:00Z"
}
```




### 5.3 Export Task Data

```plaintext
GET /api/tasks/{task_id}/export
```

- **Function**: Export complete task data
- **Query Parameters**:

- `format`: Export format (such as "json", "csv")



- **Response**: Complete task data file


## 6. Real-time Notification Interface

### 6.1 WebSocket Connection

```plaintext
WebSocket /ws/tasks/{task_id}
```

- **Function**: Establish a WebSocket connection to receive real-time updates for a task
- **Message Types**:

- Model update notifications
- New recommended design points
- Optimization progress updates





These interfaces cover the main interaction functionalities needed between the frontend BO optimizer GUI and the backend BO engine, supporting the complete Bayesian optimization workflow, including parameter space definition, strategy configuration, experimental design generation, result submission, model analysis, and multi-objective optimization.


## ✅ Confirmation Points

| Module | Evaluation |
|------|------|
| 🔧 Parameter Space and Objective Configuration | ✅ Very clear, supports multiple variable types and complex constraints |
| 🧠 Strategy and Algorithm Configuration | ✅ Supports LHS, acquisition function selection, kernel switching, reserved multi-objective optimization configuration |
| 🎯 Experimental Design and Recommendation | ✅ Distinguishes between initial points and recommended points, returning acquisition function explanation (reason) is excellent |
| 📈 Model Evaluation and Visualization | ✅ Considers predicted vs actual value difference analysis, supports R², RMSE, Pareto front, and other metrics |
| 📦 Task Management | ✅ Supports task status tracking, export, WebSocket real-time connection |
| 💬 Extensibility | ✅ Can be used for independent deployment, frontend framework agnostic, can connect to Canvas or be used for CLI/script execution |

---

## 🌱 Additional Suggestions (Optional)

### ✅ 1. Add `/api/tasks/{task_id}/restart` Interface
- **Function**: Restart a BO experiment task after interruption (for Canvas interruption recovery)
- **Optional Fields**:
```json
{
  "strategy": "reuse_last" | "reset",
  "preserve_history": true
}
```

---

### ✅ 2. Assign a Persistent Directory or Database Table for Each Task
Your design assumes `task_id` is the primary key, suggestion:
- All design points, results, model states saved in:
  ```
  /data/tasks/{task_id}/
      ├── parameter_space.json
      ├── strategy.json
      ├── history.csv
      ├── model.pkl
      └── log.json
  ```
- Facilitates debugging + reproducibility + Canvas can load as needed

---

### ✅ 3. Support Asynchronous Computation (recommended with WebSocket)
- Some backend models (like multi-objective + qEI) are computationally intensive, synchronous API responses are not suitable.
- Recommend introducing **task queue system (like Celery + Redis)**, and pushing completion events via WebSocket.

---

### ✅ 4. Add `/api/diagnostics/{task_id}` Interface
For debugging system interruptions or unresponsiveness:
```json
{
  "parameter_space": "valid",
  "model_trained": true,
  "recent_exception": null,
  "pending_experiments": [],
  "last_recommendation_time": "2025-04-22T10:30:00Z"
}
```

---

## ✅ Summary: You already have an excellent microservice architecture specification, convenient for future integration with:

- 📦 Canvas / Streamlit frontend
- 🔁 Prefect / Argo workflow control
- 🔍 Data lake or metadata platform
- ☁️ Cloud deployment (Lambda / ECS)

Correct format for parameter space creation request:
curl -X POST http://localhost:8000/api/parameter-space -H "Content-Type: application/json" -d '{"name":"Test Parameter Space","description":"Test parameter space for optimization","parameters":{"x1":{"name":"x1","type":"continuous","min":0.0,"max":10.0,"description":"Continuous parameter"},"x2":{"name":"x2","type":"discrete","min":1,"max":5,"description":"Integer parameter"},"x3":{"name":"x3","type":"categorical","values":["A","B","C"],"description":"Categorical parameter"}},"objectives":{"y1":{"name":"y1","type":"minimize","bounds":[0,100]}}}'

