#!/bin/bash

# Start integration testing environment for the Bayesian Optimization system

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
  lsof -i:"$1" >/dev/null 2>&1
}

# Check for required commands
if ! command_exists uvicorn; then
  echo "Error: uvicorn is not installed. Please install it with 'pip install uvicorn'."
  exit 1
fi

if ! command_exists node; then
  echo "Error: Node.js is not installed. Please install it from https://nodejs.org/."
  exit 1
fi

# Check if ports are already in use
if port_in_use 8000; then
  echo "Error: Port 8000 is already in use. Please stop any running services on this port."
  exit 1
fi

if port_in_use 3000; then
  echo "Error: Port 3000 is already in use. Please stop any running services on this port."
  exit 1
fi

# Create a .env file for the backend if it doesn't exist
if [ ! -f backend/.env ]; then
  echo "Creating .env file for backend..."
  cat > backend/.env << EOL
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
TASK_DIR=\${DATA_DIR}/tasks

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
EOL
fi

# Create a .env.local file for the frontend if it doesn't exist
if [ ! -f frontend/.env.local ]; then
  echo "Creating .env.local file for frontend..."
  cat > frontend/.env.local << EOL
NEXT_PUBLIC_API_URL=http://localhost:8000/api
EOL
fi

# Create necessary directories
mkdir -p backend/data/tasks
mkdir -p backend/data/logs

# Start the backend server in the background
echo "Starting backend server..."
cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for the backend to start
echo "Waiting for backend server to start..."
sleep 5

# Start the frontend server in the background
echo "Starting frontend server..."
cd frontend && npm run dev &
FRONTEND_PID=$!

# Wait for the frontend to start
echo "Waiting for frontend server to start..."
sleep 5

echo "Both servers are now running."
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Documentation: http://localhost:8000/docs"

# Function to clean up processes on exit
cleanup() {
  echo "Stopping servers..."
  kill $BACKEND_PID $FRONTEND_PID
  wait
  echo "Servers stopped."
}

# Register the cleanup function to be called on exit
trap cleanup EXIT

# Keep the script running
echo "Press Ctrl+C to stop the servers."
wait
