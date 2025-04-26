/**
 * Integration test script for the Bayesian Optimization system
 * 
 * This script tests the integration between the frontend and backend
 * by making API calls to the backend endpoints.
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000/api';
let taskId = null;

// Helper function for API requests
async function fetchAPI(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };
  
  console.log(`Making ${options.method || 'GET'} request to ${url}`);
  
  try {
    const response = await fetch(url, {
      ...options,
      headers,
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      console.error(`Error: ${response.status}`, data);
      return { success: false, data };
    }
    
    console.log('Response:', data);
    return { success: true, data };
  } catch (error) {
    console.error('Request failed:', error);
    return { success: false, error };
  }
}

// Test cases
const tests = {
  // 1. Create parameter space
  async createParameterSpace() {
    console.log('\n=== Test: Create Parameter Space ===');
    
    const parameterSpace = {
      name: "Test Optimization Task",
      description: "Integration test for parameter space creation",
      parameters: [
        {
          name: "x1",
          type: "continuous",
          bounds: [0, 10],
          description: "First parameter"
        },
        {
          name: "x2",
          type: "discrete",
          values: [1, 2, 3, 4, 5],
          description: "Second parameter"
        },
        {
          name: "x3",
          type: "categorical",
          categories: ["A", "B", "C"],
          description: "Third parameter"
        }
      ],
      objectives: [
        {
          name: "y1",
          direction: "minimize",
          description: "First objective"
        }
      ]
    };
    
    const result = await fetchAPI('/parameter-space', {
      method: 'POST',
      body: JSON.stringify(parameterSpace),
    });
    
    if (result.success && result.data.task_id) {
      taskId = result.data.task_id;
      console.log(`Task ID: ${taskId}`);
      return true;
    }
    
    return false;
  },
  
  // 2. Set optimization strategy
  async setStrategy() {
    console.log('\n=== Test: Set Optimization Strategy ===');
    
    if (!taskId) {
      console.error('No task ID available. Run createParameterSpace first.');
      return false;
    }
    
    const strategy = {
      algorithm: "bayesian",
      acquisition_function: "ei",
      surrogate_model: "gaussian_process",
      exploration_weight: 0.1,
      random_seed: 42
    };
    
    const result = await fetchAPI(`/strategy/${taskId}`, {
      method: 'POST',
      body: JSON.stringify(strategy),
    });
    
    return result.success;
  },
  
  // 3. Get initial design points
  async getInitialDesigns() {
    console.log('\n=== Test: Get Initial Designs ===');
    
    if (!taskId) {
      console.error('No task ID available. Run createParameterSpace first.');
      return false;
    }
    
    const result = await fetchAPI(`/designs/${taskId}/initial?samples=5`);
    
    return result.success;
  },
  
  // 4. Submit results
  async submitResults() {
    console.log('\n=== Test: Submit Results ===');
    
    if (!taskId) {
      console.error('No task ID available. Run createParameterSpace first.');
      return false;
    }
    
    // Get initial designs first to have valid design points
    const designsResult = await fetchAPI(`/designs/${taskId}/initial?samples=3`);
    
    if (!designsResult.success) {
      console.error('Failed to get designs for result submission');
      return false;
    }
    
    const designs = designsResult.data.designs;
    
    // Create mock results for the designs
    const results = designs.map(design => ({
      parameters: design,
      objectives: { y1: Math.random() * 10 },
      metadata: { timestamp: new Date().toISOString() }
    }));
    
    const result = await fetchAPI(`/results/${taskId}`, {
      method: 'POST',
      body: JSON.stringify({ results }),
    });
    
    return result.success;
  },
  
  // 5. Get next design points
  async getNextDesigns() {
    console.log('\n=== Test: Get Next Designs ===');
    
    if (!taskId) {
      console.error('No task ID available. Run createParameterSpace first.');
      return false;
    }
    
    const result = await fetchAPI(`/designs/${taskId}/next?batch_size=2`);
    
    return result.success;
  },
  
  // 6. Make predictions
  async makePredictions() {
    console.log('\n=== Test: Make Predictions ===');
    
    if (!taskId) {
      console.error('No task ID available. Run createParameterSpace first.');
      return false;
    }
    
    const points = [
      { x1: 5.0, x2: 3, x3: "A" },
      { x1: 7.5, x2: 4, x3: "B" }
    ];
    
    const result = await fetchAPI(`/predict/${taskId}`, {
      method: 'POST',
      body: JSON.stringify({ points }),
    });
    
    return result.success;
  },
  
  // 7. Get task status
  async getTaskStatus() {
    console.log('\n=== Test: Get Task Status ===');
    
    if (!taskId) {
      console.error('No task ID available. Run createParameterSpace first.');
      return false;
    }
    
    const result = await fetchAPI(`/tasks/${taskId}/status`);
    
    return result.success;
  },
  
  // 8. Export task data
  async exportTaskData() {
    console.log('\n=== Test: Export Task Data ===');
    
    if (!taskId) {
      console.error('No task ID available. Run createParameterSpace first.');
      return false;
    }
    
    const result = await fetchAPI(`/tasks/${taskId}/export?format=json`);
    
    return result.success;
  }
};

// Run all tests in sequence
async function runTests() {
  console.log('Starting integration tests...');
  
  const testResults = {};
  
  // Run tests in order
  for (const [testName, testFn] of Object.entries(tests)) {
    try {
      const success = await testFn();
      testResults[testName] = success ? 'PASS' : 'FAIL';
    } catch (error) {
      console.error(`Error in test ${testName}:`, error);
      testResults[testName] = 'ERROR';
    }
  }
  
  // Print summary
  console.log('\n=== Test Results Summary ===');
  for (const [testName, result] of Object.entries(testResults)) {
    console.log(`${testName}: ${result}`);
  }
}

// Run the tests
runTests().catch(console.error);
