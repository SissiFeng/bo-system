/**
 * Automated test script for catalyst optimization experiment
 */

const fs = require('fs');
const path = require('path');

// API base URL
const API_BASE_URL = 'http://localhost:8006/api';

// Load the catalyst optimization configuration
const configPath = path.join(__dirname, 'catalyst_optimization.json');
const configData = JSON.parse(fs.readFileSync(configPath, 'utf8'));

// Global variables to store task ID and designs
let taskId = null;
let designs = null;

// Helper function for API requests
async function fetchAPI(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;

  console.log(`Making ${options.method || 'GET'} request to ${url}`);

  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  console.log(`Response status: ${response.status}`);

  const contentType = response.headers.get('content-type');
  let data;

  if (contentType && contentType.includes('application/json')) {
    data = await response.json();
  } else {
    const text = await response.text();
    try {
      data = JSON.parse(text);
    } catch (e) {
      data = { text };
    }
  }

  if (!response.ok) {
    throw new Error(`API error: ${response.status} - ${JSON.stringify(data)}`);
  }

  return data;
}

// Test functions
async function testCreateParameterSpace() {
  console.log('\n===== Testing Parameter Space Creation =====');

  try {
    const result = await fetchAPI('/parameter-space', {
      method: 'POST',
      body: JSON.stringify(configData),
    });

    console.log('✅ Parameter space created successfully');
    console.log('Task ID:', result.task_id);
    taskId = result.task_id;
    return true;
  } catch (error) {
    console.error('❌ Failed to create parameter space:', error.message);
    return false;
  }
}

async function testGetInitialDesigns() {
  console.log('\n===== Testing Initial Design Generation =====');

  if (!taskId) {
    console.error('❌ Task ID is required');
    return false;
  }

  try {
    const result = await fetchAPI(`/designs/${taskId}/initial?samples=5`);

    console.log(`✅ Generated ${result.designs.length} initial designs`);
    designs = result.designs;

    // Display the first design
    if (result.designs.length > 0) {
      console.log('First design:', JSON.stringify(result.designs[0], null, 2));
    }

    return true;
  } catch (error) {
    console.error('❌ Failed to get initial designs:', error.message);
    return false;
  }
}

async function testSubmitResults() {
  console.log('\n===== Testing Result Submission =====');

  if (!taskId || !designs || designs.length === 0) {
    console.error('❌ Task ID and designs are required');
    return false;
  }

  // Generate mock results for the designs
  const mockResults = designs.map(design => ({
    parameters: design.parameters,
    objectives: {
      LSV_slope: Math.random() * 10,  // Random value between 0 and 10
      CV_stability: Math.random() * 5  // Random value between 0 and 5
    },
    metadata: {
      timestamp: new Date().toISOString(),
      experiment_id: `exp-${Math.floor(Math.random() * 1000)}`
    }
  }));

  try {
    const result = await fetchAPI(`/results/${taskId}`, {
      method: 'POST',
      body: JSON.stringify({ results: mockResults }),
    });

    console.log('✅ Results submitted successfully');
    return true;
  } catch (error) {
    console.error('❌ Failed to submit results:', error.message);
    return false;
  }
}

async function testGetNextDesigns() {
  console.log('\n===== Testing Next Design Recommendation =====');

  if (!taskId) {
    console.error('❌ Task ID is required');
    return false;
  }

  try {
    const result = await fetchAPI(`/designs/${taskId}/next?batch_size=3`);

    console.log(`✅ Received ${result.designs.length} next design recommendations`);

    // Display the first recommended design
    if (result.designs.length > 0) {
      console.log('First recommendation:', JSON.stringify(result.designs[0], null, 2));
    }

    return true;
  } catch (error) {
    console.error('❌ Failed to get next designs:', error.message);
    return false;
  }
}

async function testGetTaskStatus() {
  console.log('\n===== Testing Task Status =====');

  if (!taskId) {
    console.error('❌ Task ID is required');
    return false;
  }

  try {
    const result = await fetchAPI(`/tasks/${taskId}/status`);

    console.log('✅ Task status retrieved successfully');
    console.log('Status:', result.status);
    console.log('Progress:', result.progress);

    return true;
  } catch (error) {
    console.error('❌ Failed to get task status:', error.message);
    return false;
  }
}

// Run all tests
async function runAllTests() {
  console.log('Starting automated tests for catalyst optimization experiment...');

  const tests = [
    { name: 'Create Parameter Space', fn: testCreateParameterSpace },
    { name: 'Get Initial Designs', fn: testGetInitialDesigns },
    { name: 'Submit Results', fn: testSubmitResults },
    { name: 'Get Next Designs', fn: testGetNextDesigns },
    { name: 'Get Task Status', fn: testGetTaskStatus },
  ];

  const results = {};

  for (const test of tests) {
    console.log(`\nRunning test: ${test.name}`);
    const success = await test.fn();
    results[test.name] = success ? '✅ Success' : '❌ Failed';

    if (!success && (test.name === 'Create Parameter Space' || test.name === 'Get Initial Designs')) {
      console.error(`Critical test ${test.name} failed, aborting remaining tests`);
      break;
    }

    // Add a small delay between tests
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  console.log('\n===== Test Results Summary =====');
  for (const [name, result] of Object.entries(results)) {
    console.log(`${name}: ${result}`);
  }
}

// Run the tests
runAllTests().catch(console.error);
