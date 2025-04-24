# BO Engine API

A modular, configurable Bayesian Optimization (BO) engine with FastAPI backend that supports flexible parameter space management and active learning strategies.

## Overview

This system is designed as a microservice that can be used standalone or integrated with other systems via JSON APIs. It provides a complete Bayesian Optimization workflow, including:

- Parameter space definition (continuous, discrete, categorical variables)
- Multiple objective functions (maximize/minimize)
- Constraints handling
- Various sampling strategies (LHS, Factorial, etc.)
- Multiple surrogate models (GP, RF, SVM)
- Acquisition functions (EI, UCB, PI)
- Multi-objective optimization capabilities
- Real-time monitoring and visualization

## Project Structure

```
backend/
│
├── app/                        # FastAPI application
│   ├── main.py                 # Main entry point
│   ├── api/                    # REST API routes
│   │   ├── endpoints.py        # API definitions
│   │   └── schema.py           # Pydantic data models
│   └── core/                   # Core utilities
│       ├── config.py           # Configuration management
│       └── logger.py           # Logging utilities
│
├── bo_engine/                  # Core Bayesian optimization engine
│   ├── parameter_space.py      # Parameter space definition
│   ├── design_generator.py     # Initial sampling strategies
│   ├── models/                 # Surrogate models
│   │   ├── gpr.py              # Gaussian Process Regressor
│   │   ├── rfr.py              # Random Forest Regressor
│   │   ├── svm.py              # Support Vector Machine
│   │   └── base_model.py       # Model interface
│   ├── acquisition.py          # Acquisition functions
│   ├── optimizer.py            # Core optimization process
│   └── utils.py                # Utility functions
│
├── data/                       # Data storage
│   └── tasks/                  # Task-specific data
│
├── notebooks/                  # Analysis notebooks
│
├── tests/                      # Unit tests
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerization
└── .env                        # Environment variables
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the API

```bash
# Development mode
uvicorn app.main:app --reload --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Using Docker

```bash
# Build the Docker image
docker build -t bo-engine-api .

# Run the container
docker run -p 8000:8000 bo-engine-api
```

## API Documentation

Once the server is running, you can access the API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Examples

See the `notebooks/` directory for example usage and demonstrations.

## Contributing

Please see the [Contributing Guide](../Docs/Contributing.md) for details on how to contribute to this project.

## License

[MIT License](LICENSE) 
