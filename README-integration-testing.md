# Frontend-Backend Integration Testing

This document provides instructions for testing the integration between the frontend and backend components of the Bayesian Optimization system.

## Prerequisites

- Python 3.8+ with pip
- Node.js and npm/pnpm
- Basic understanding of REST APIs and Bayesian Optimization

## Setup

1. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   # or
   pnpm install
   ```

## Running the Integration Tests

### Option 1: Using the Automated Script

We've provided a script that starts both the frontend and backend servers:

```bash
./start-integration-test.sh
```

This script:
- Checks for required dependencies
- Creates necessary configuration files if they don't exist
- Starts the backend server on port 8000
- Starts the frontend server on port 3000
- Sets up proper cleanup on exit

### Option 2: Manual Setup

If you prefer to start the servers manually:

1. Start the backend server:
   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. In a separate terminal, start the frontend server:
   ```bash
   cd frontend
   npm run dev
   # or
   pnpm dev
   ```

## Testing the Integration

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:3000`
2. Use the web interface to:
   - Create a new optimization task
   - Configure optimization strategy
   - Generate initial design points
   - Submit results
   - Get next recommended design points
   - Make predictions

### Using the Integration Test Script

We've provided a JavaScript test script that tests the API endpoints:

```bash
# Install node-fetch if you don't have it
npm install node-fetch

# Run the test script
node integration-test.js
```

This script:
- Creates a parameter space
- Sets an optimization strategy
- Gets initial design points
- Submits results
- Gets next recommended design points
- Makes predictions
- Gets task status
- Exports task data

## API Documentation

The backend API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Troubleshooting

### Common Issues

1. **Port already in use**
   - Make sure ports 8000 and 3000 are not being used by other applications
   - You can change the ports in the configuration files if needed

2. **CORS errors**
   - The backend is configured to allow cross-origin requests from the frontend
   - If you're seeing CORS errors, make sure the frontend is making requests to the correct URL

3. **API endpoint not found**
   - Check the API documentation to ensure you're using the correct endpoint
   - Verify that the backend server is running

4. **Authentication errors**
   - The current implementation doesn't include authentication
   - If you've added authentication, make sure the frontend is sending the correct credentials

## Next Steps

After successful integration testing, you may want to:

1. Implement proper error handling in the frontend
2. Add loading states for API requests
3. Implement WebSocket connection for real-time updates
4. Add authentication if needed
5. Deploy the application to a production environment
