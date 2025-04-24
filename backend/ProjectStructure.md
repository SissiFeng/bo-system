# Project Structure

This document outlines the overall structure of the BO Engine API project, including directories, key files, and their purposes.

## Directory Structure

```
backend/
│
├── app/                        # FastAPI application
│   ├── main.py                 # Main entry point, registers routes & starts the API
│   ├── api/                    # REST API routes
│   │   ├── endpoints.py        # API endpoint definitions
│   │   └── schema.py           # Pydantic models for API requests/responses
│   └── core/                   # Core utilities
│       ├── config.py           # Configuration management
│       └── logger.py           # Logging setup and utilities
│
├── bo_engine/                  # Core Bayesian optimization engine
│   ├── parameter_space.py      # Parameter space definition and management
│   ├── design_generator.py     # Initial sampling strategies
│   ├── models/                 # Surrogate models
│   │   ├── base_model.py       # Base model interface
│   │   ├── gpr.py              # Gaussian Process Regression
│   │   └── rfr.py              # Random Forest Regression
│   ├── acquisition.py          # Acquisition functions
│   ├── optimizer.py            # Core optimization workflow
│   └── utils.py                # Utility functions
│
├── data/                       # Data storage
│   ├── tasks/                  # Task-specific data
│   │   └── {task_id}/          # Data for each task
│   │       ├── parameter_space.json
│   │       ├── strategy.json
│   │       ├── designs.json
│   │       ├── results.json
│   │       └── model.pkl
│   └── logs/                   # Log files
│
├── notebooks/                  # Analysis notebooks
│   ├── bo_simulation.ipynb
│   └── model_comparison.ipynb
│
├── tests/                      # Unit and integration tests
│   ├── test_parameter_space.py
│   ├── test_design_generator.py
│   ├── test_models.py
│   ├── test_acquisition.py
│   ├── test_optimizer.py
│   └── test_api.py
│
├── Docs/                       # Project documentation
│   ├── Contributing.md         # Contribution guidelines
│   ├── DevEnvConfig.md         # Development environment setup
│   ├── DevLog/                 # Development logs
│   │   └── 2023-06-15_progress.md
│   ├── Feature/                # Feature-specific documentation
│   │   └── parameter-space_rules.md
│   ├── AskLog/                 # Records of questions and answers
│   ├── FAQ.md                  # Frequently asked questions
│   └── TechDebt.md             # Technical debt tracking
│
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in version control)
├── README.md                   # Project overview
├── FeatureMap.md               # Feature relationship visualization
└── ProjectStructure.md         # This file
```

## Key Components

### FastAPI Application (`app/`)

The FastAPI application handles HTTP requests and responses, defining the API that clients interact with.

- **`main.py`**: Entry point for the application, sets up FastAPI, registers routes, and configures middleware.
- **`api/endpoints.py`**: Defines all API endpoints, their request/response models, and handling logic.
- **`api/schema.py`**: Contains Pydantic models that define the structure of API requests and responses.
- **`core/config.py`**: Manages application configuration, loading from environment variables and providing defaults.
- **`core/logger.py`**: Sets up logging for the application, including console and file outputs.

### Bayesian Optimization Engine (`bo_engine/`)

The core BO engine implements the functionality needed for Bayesian optimization.

- **`parameter_space.py`**: Defines the parameter space, including continuous, discrete, and categorical parameters, as well as constraints.
- **`design_generator.py`**: Implements methods for generating initial design points (e.g., Latin Hypercube Sampling).
- **`models/`**: Contains surrogate model implementations with a common interface.
- **`acquisition.py`**: Implements acquisition functions for selecting next points to evaluate.
- **`optimizer.py`**: Orchestrates the optimization process, maintaining state and managing the workflow.
- **`utils.py`**: Provides utility functions used throughout the BO engine.

### Data Storage (`data/`)

The data directory stores task-specific data and logs.

- **`tasks/{task_id}/`**: Each task has its own directory containing all relevant data.
- **`logs/`**: Contains application logs for monitoring and debugging.

### Documentation (`Docs/`)

Comprehensive documentation for developers and users.

- **`Contributing.md`**: Guidelines for contributing to the project.
- **`DevEnvConfig.md`**: Instructions for setting up the development environment.
- **`Feature/`**: Detailed documentation for specific features.
- **`DevLog/`**: Chronicles development progress and decisions.
- **`FAQ.md`**: Answers to frequently asked questions.
- **`TechDebt.md`**: Tracks technical debt and future improvements.

## Extending the Project

### Adding a New Parameter Type

To add a new parameter type:

1. Extend the base `Parameter` class in `bo_engine/parameter_space.py`
2. Add corresponding schema to `app/api/schema.py`
3. Update validation logic in `ParameterSpace` class
4. Add tests in `tests/test_parameter_space.py`
5. Document in `Docs/Feature/parameter-space_rules.md`

### Adding a New Surrogate Model

To add a new surrogate model:

1. Create a new file in `bo_engine/models/` that extends `BaseModel`
2. Implement required methods: `fit()`, `predict()`, `predict_with_uncertainty()`
3. Add model selection logic in `bo_engine/optimizer.py`
4. Add tests in `tests/test_models.py`
5. Document in appropriate feature documentation

### Adding a New API Endpoint

To add a new API endpoint:

1. Define new Pydantic models in `app/api/schema.py` if needed
2. Add the endpoint function in `app/api/endpoints.py`
3. Implement the required logic, connecting to the BO engine
4. Add tests in `tests/test_api.py`
5. Update API documentation

## Scaling Considerations

As the project grows, consider the following scaling strategies:

1. **Database Storage**: Replace file-based storage with a proper database (e.g., PostgreSQL, MongoDB)
2. **Task Queue**: Add a task queue (e.g., Celery with Redis) for handling long-running operations
3. **Microservices**: Split into smaller microservices (e.g., parameter space service, optimizer service)
4. **API Gateway**: Add an API gateway for routing, authentication, and rate limiting
5. **Caching**: Implement caching for frequently accessed data

The current structure supports easy transition to these scaling strategies without major refactoring. 
