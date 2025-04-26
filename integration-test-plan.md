# Frontend-Backend Integration Test Plan

## Overview
This document outlines the plan for testing the integration between the frontend (Next.js) and backend (FastAPI) components of the Bayesian Optimization system.

## Prerequisites
1. Backend server running
2. Frontend development server running
3. Network connectivity between frontend and backend

## Test Environment Setup

### 1. Start the Backend Server
```bash
# Navigate to the backend directory
cd backend

# Run the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend Development Server
```bash
# Navigate to the frontend directory
cd frontend

# Run the Next.js development server
npm run dev
# or
pnpm dev
```

## Integration Test Cases

### 1. Parameter Space Configuration
- **Test Case**: Create a new optimization task
- **Endpoint**: `POST /api/parameter-space`
- **Frontend Component**: Parameter space configuration form
- **Expected Result**: New task created, task ID returned

### 2. Optimization Strategy Configuration
- **Test Case**: Set optimization strategy for a task
- **Endpoint**: `POST /api/strategy/{task_id}`
- **Frontend Component**: Strategy configuration form
- **Expected Result**: Strategy set successfully

### 3. Initial Design Generation
- **Test Case**: Generate initial design points
- **Endpoint**: `GET /api/designs/{task_id}/initial`
- **Frontend Component**: Initial design display
- **Expected Result**: Initial design points returned

### 4. Results Submission
- **Test Case**: Submit experiment results
- **Endpoint**: `POST /api/results/{task_id}`
- **Frontend Component**: Results submission form
- **Expected Result**: Results accepted, confirmation returned

### 5. Next Design Recommendation
- **Test Case**: Get next recommended design points
- **Endpoint**: `GET /api/designs/{task_id}/next`
- **Frontend Component**: Next design display
- **Expected Result**: Next design points returned

### 6. Model Prediction
- **Test Case**: Make predictions for specific parameter combinations
- **Endpoint**: `POST /api/predict/{task_id}`
- **Frontend Component**: Prediction form/display
- **Expected Result**: Predictions returned

### 7. Task Status Monitoring
- **Test Case**: Get task status
- **Endpoint**: `GET /api/tasks/{task_id}/status`
- **Frontend Component**: Task status display
- **Expected Result**: Current task status returned

## End-to-End Test Scenario

### Complete Optimization Workflow
1. Create a new optimization task with parameter space
2. Configure optimization strategy
3. Generate initial design points
4. Submit results for initial design points
5. Get next recommended design points
6. Submit results for recommended design points
7. Repeat steps 5-6 for multiple iterations
8. Make predictions for new parameter combinations
9. Export task data

## WebSocket Testing (if implemented)
- **Test Case**: Establish WebSocket connection for real-time updates
- **Endpoint**: `WebSocket /ws/tasks/{task_id}`
- **Frontend Component**: Real-time update display
- **Expected Result**: Real-time updates received

## Error Handling Tests
1. Test with invalid parameter space configuration
2. Test with invalid results submission
3. Test with non-existent task ID
4. Test with invalid prediction request

## Performance Considerations
- Test response times for computationally intensive operations
- Test handling of large result sets
- Test concurrent requests

## Documentation
- Document any issues encountered during integration testing
- Update API documentation if discrepancies are found
- Document any workarounds implemented
