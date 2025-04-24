# Development Environment Configuration

This document provides detailed instructions for setting up the development environment for the BO Engine API.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- git (version control)
- Docker (optional, for containerized development)

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd backend
```

## Step 2: Set Up a Virtual Environment

### For macOS/Linux:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### For Windows:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# If you want to install development dependencies as well
pip install -r requirements-dev.txt  # if available
```

## Step 4: Configure Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# Application
APP_ENV=development
APP_PORT=8000
APP_HOST=0.0.0.0
APP_NAME=BO-Engine-API
APP_VERSION=0.1.0

# Logging
LOG_LEVEL=DEBUG

# Data storage
DATA_DIR=./data
TASK_DIR=${DATA_DIR}/tasks

# Default optimization settings
DEFAULT_RANDOM_SEED=42
DEFAULT_INITIAL_SAMPLES=10
DEFAULT_ACQUISITION_FUNCTION=ei
DEFAULT_KERNEL=matern
DEFAULT_EXPLORATION_WEIGHT=0.5

# WebSocket
WS_PING_INTERVAL=30

# Performance
MAX_WORKERS=4
```

## Step 5: Create Required Directories

```bash
# Create data directories
mkdir -p data/tasks data/logs
```

## Step 6: Run the Application

### For development with auto-reload:

```bash
uvicorn app.main:app --reload --port 8000
```

### For production-like environment:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Step 7: Access the API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Docker Setup (Optional)

If you prefer to use Docker:

```bash
# Build the Docker image
docker build -t bo-engine-api .

# Run the container
docker run -p 8000:8000 -v $(pwd)/data:/app/data bo-engine-api
```

## Setting Up for Testing

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=app --cov=bo_engine

# Run a specific test file
pytest tests/test_parameter_space.py
```

## IDE Configuration

### VS Code

For VS Code users, create a `.vscode/settings.json` file with the following settings:

```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": [
        "--line-length",
        "100"
    ],
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "python.testing.nosetestsEnabled": false,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ]
}
```

### PyCharm

For PyCharm users:

1. Open the project in PyCharm
2. Go to File > Settings > Project > Python Interpreter
3. Add a new interpreter and select the virtual environment you created
4. Configure the project structure to mark the root directory as Sources

## Troubleshooting

### Common Issues

**Issue: Import errors when running the application**

Solution: Make sure you're running the application from the root directory of the project.

**Issue: Module not found errors**

Solution: Ensure that your virtual environment is activated and all dependencies are installed.

**Issue: Permission denied when creating directories or files**

Solution: Check the permissions of the parent directories and adjust as needed.

## Additional Development Tools

- **Black**: Code formatter
  ```
  pip install black
  black .
  ```

- **Flake8**: Linter
  ```
  pip install flake8
  flake8 .
  ```

- **MyPy**: Static type checker
  ```
  pip install mypy
  mypy app bo_engine
  ```

- **isort**: Import sorter
  ```
  pip install isort
  isort .
  ``` 
