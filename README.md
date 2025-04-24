# Bayesian Optimization System

> Efficient Bayesian optimization engine with REST API interface for complex parameter space optimization, built on a microservices architecture.

## 🔍 Project Overview

Bayesian Optimization (BO) is a powerful technique for optimizing expensive-to-evaluate black-box functions. This system provides:

- Definition of complex parameter spaces (continuous, categorical, discrete parameters)
- Configuration of optimization strategies (surrogate models, acquisition functions)
- Initial experimental design generation (random, Latin hypercube, factorial, etc.)
- Continuous learning and recommendation of optimal parameter combinations
- Complete task management and real-time status updates

## 🏗️ System Architecture

The system is built as a set of microservices:

```
bo-system/
├── backend/                # Backend services
│   ├── app/                # FastAPI application
│   │   ├── api/            # API endpoints and schemas
│   │   └── main.py         # Application entry point
│   ├── bo_engine/          # Bayesian optimization engine
│   │   ├── acquisition/    # Acquisition functions
│   │   ├── models/         # Surrogate models
│   │   ├── parameter_space.py  # Parameter space representation
│   │   ├── design_generator.py # Experimental design generation
│   │   └── optimizer.py    # Core optimization logic
│   ├── tests/              # Unit and integration tests
│   └── Docs/               # Documentation
└── frontend/               
```

## 🚀 Quick Start

### Environment Requirements

- Python 3.8+
- FastAPI
- Scikit-learn
- NumPy, SciPy, Pandas

### Installation

```bash
# Clone the repository
git clone https://github.com/username/bo-system.git
cd bo-system

# Install dependencies
pip install -r backend/requirements.txt

# Run the backend service
cd backend
uvicorn app.main:app --reload
```

## 📡 API Interfaces

### Parameter Space Configuration

- `POST /api/parameter-space` - Create a new optimization task
- `GET /api/parameter-space/{task_id}` - Get parameter space configuration

### Optimization Strategy Configuration

- `POST /api/strategy/{task_id}` - Set optimization strategy
- `GET /api/strategy/{task_id}` - Get current optimization strategy

### Experimental Design and Optimization

- `GET /api/designs/{task_id}/initial` - Get initial experimental design points
- `POST /api/results/{task_id}` - Submit experimental results
- `GET /api/designs/{task_id}/next` - Get next suggested design points
- `POST /api/predict/{task_id}` - Predict outcomes for specific parameter combinations
- `GET /api/model/{task_id}/performance` - Get current model performance

### Task Management

- `GET /api/tasks` - List all optimization tasks
- `GET /api/tasks/{task_id}/status` - Get task status
- `POST /api/tasks/{task_id}/restart` - Restart an optimization task
- `GET /api/tasks/{task_id}/export` - Export task data

## 📋 Usage Example

```python
import requests
import json

# Server URL
BASE_URL = "http://localhost:8000/api"

# Create a new optimization task
parameter_space = {
    "parameters": [
        {
            "name": "x1",
            "type": "continuous",
            "bounds": [-5.0, 5.0]
        },
        {
            "name": "x2",
            "type": "continuous",
            "bounds": [-5.0, 5.0]
        }
    ],
    "objectives": [
        {
            "name": "y",
            "type": "minimize"
        }
    ]
}

response = requests.post(f"{BASE_URL}/parameter-space", json=parameter_space)
task_id = response.json()["task_id"]
print(f"Created task: {task_id}")

# Set optimization strategy
strategy = {
    "model_type": "gaussian_process",
    "acquisition_type": "expected_improvement",
    "batch_size": 1
}

requests.post(f"{BASE_URL}/strategy/{task_id}", json=strategy)

# Get initial experimental points
initial_points = requests.get(f"{BASE_URL}/designs/{task_id}/initial").json()
print("Initial points to evaluate:", initial_points)

# Submit results (assuming evaluation of function)
for point in initial_points["design_points"]:
    x1 = point["parameters"]["x1"]
    x2 = point["parameters"]["x2"]
    # Example objective function: f(x1, x2) = x1^2 + x2^2
    y = x1**2 + x2**2
    
    result = {
        "design_id": point["design_id"],
        "objectives": {"y": y}
    }
    requests.post(f"{BASE_URL}/results/{task_id}", json=result)

# Get suggested next points
next_points = requests.get(f"{BASE_URL}/designs/{task_id}/next").json()
print("Next suggested points:", next_points)
```

## 🚦 Development Status

| Component | Status | Notes |
|-----------|--------|-------|
| Parameter Space | ✅ Complete | Supports continuous, categorical, and discrete parameters |
| Initial Design Generation | ✅ Complete | Random, LHS, factorial, Sobol designs |
| Core BO Engine | ✅ Complete | GP model and EI acquisition function implemented |
| Task Management | ✅ Complete | Task creation, status tracking, results recording |
| Real-time Notifications | 🚧 In Progress | WebSocket integration planned |
| Frontend UI | 📅 Planned | Development will begin in next phase |

See [progress.md](progress.md) for detailed development status.

## 📚 Documentation

- [Parameter Space Configuration](backend/Docs/Feature/parameter-space_rules.md)
- [Design Generator](backend/Docs/Feature/design-generator_rules.md)
- [BO System](backend/Docs/Feature/bo-system_rules.md)
- [API Documentation](backend/Docs/api-docs.md)

## 👥 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
