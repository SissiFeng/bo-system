/**
 * Simple test script for catalyst optimization JSON file
 */

const fs = require('fs');
const path = require('path');

// Load the catalyst optimization configuration
const configPath = path.join(__dirname, 'catalyst_optimization.json');
const configData = JSON.parse(fs.readFileSync(configPath, 'utf8'));

// Function to validate the JSON structure
function validateJSON(data) {
  console.log('\n===== Validating JSON Structure =====');
  
  // Check required top-level fields
  const requiredFields = ['name', 'description', 'parameters', 'objectives'];
  for (const field of requiredFields) {
    if (!data[field]) {
      console.error(`❌ Missing required field: ${field}`);
      return false;
    }
  }
  
  // Check parameters
  if (!Array.isArray(data.parameters) || data.parameters.length === 0) {
    console.error('❌ Parameters must be a non-empty array');
    return false;
  }
  
  // Check objectives
  if (!Array.isArray(data.objectives) || data.objectives.length === 0) {
    console.error('❌ Objectives must be a non-empty array');
    return false;
  }
  
  // Check constraints
  if (!Array.isArray(data.constraints)) {
    console.error('❌ Constraints must be an array');
    return false;
  }
  
  console.log('✅ JSON structure is valid');
  return true;
}

// Function to analyze parameters
function analyzeParameters(data) {
  console.log('\n===== Analyzing Parameters =====');
  
  const continuousParams = data.parameters.filter(p => p.type === 'continuous');
  const integerParams = data.parameters.filter(p => p.type === 'integer');
  const categoricalParams = data.parameters.filter(p => p.type === 'categorical');
  
  console.log(`Total parameters: ${data.parameters.length}`);
  console.log(`Continuous parameters: ${continuousParams.length}`);
  console.log(`Integer parameters: ${integerParams.length}`);
  console.log(`Categorical parameters: ${categoricalParams.length}`);
  
  // List all parameters
  console.log('\nParameter details:');
  data.parameters.forEach(param => {
    if (param.type === 'continuous' || param.type === 'integer') {
      console.log(`- ${param.name} (${param.type}): ${param.bounds[0]} to ${param.bounds[1]}`);
    } else if (param.type === 'categorical') {
      console.log(`- ${param.name} (${param.type}): ${param.choices.join(', ')}`);
    }
  });
  
  return true;
}

// Function to analyze objectives
function analyzeObjectives(data) {
  console.log('\n===== Analyzing Objectives =====');
  
  console.log(`Total objectives: ${data.objectives.length}`);
  
  // List all objectives
  console.log('\nObjective details:');
  data.objectives.forEach(obj => {
    console.log(`- ${obj.name} (${obj.direction}): ${obj.description}`);
  });
  
  return true;
}

// Function to analyze constraints
function analyzeConstraints(data) {
  console.log('\n===== Analyzing Constraints =====');
  
  console.log(`Total constraints: ${data.constraints.length}`);
  
  if (data.constraints.length > 0) {
    // List all constraints
    console.log('\nConstraint details:');
    data.constraints.forEach(constraint => {
      console.log(`- Type: ${constraint.type}`);
      console.log(`  Parameters: ${constraint.parameters.join(', ')}`);
      console.log(`  Relation: ${constraint.relation}`);
      console.log(`  Value: ${constraint.value}`);
    });
  } else {
    console.log('No constraints defined.');
  }
  
  return true;
}

// Run all tests
function runAllTests() {
  console.log('Starting validation for catalyst optimization JSON...');
  
  const tests = [
    { name: 'Validate JSON Structure', fn: () => validateJSON(configData) },
    { name: 'Analyze Parameters', fn: () => analyzeParameters(configData) },
    { name: 'Analyze Objectives', fn: () => analyzeObjectives(configData) },
    { name: 'Analyze Constraints', fn: () => analyzeConstraints(configData) },
  ];
  
  const results = {};
  
  for (const test of tests) {
    console.log(`\nRunning test: ${test.name}`);
    const success = test.fn();
    results[test.name] = success ? '✅ Success' : '❌ Failed';
    
    if (!success && test.name === 'Validate JSON Structure') {
      console.error(`Critical test ${test.name} failed, aborting remaining tests`);
      break;
    }
  }
  
  console.log('\n===== Test Results Summary =====');
  for (const [name, result] of Object.entries(results)) {
    console.log(`${name}: ${result}`);
  }
}

// Run the tests
runAllTests();
