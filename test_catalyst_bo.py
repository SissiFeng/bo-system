import json
import numpy as np
import os
from pathlib import Path
import uuid
import matplotlib.pyplot as plt
from datetime import datetime

# Load the catalyst optimization configuration
with open('catalyst_optimization.json', 'r') as f:
    config = json.load(f)

print("Loaded Catalyst Optimization Configuration:")
print(f"Name: {config['name']}")
print(f"Description: {config['description']}")
print(f"Parameters: {len(config['parameters'])}")
print(f"Objectives: {len(config['objectives'])}")
print(f"Constraints: {len(config['constraints'])}")

# Create output directory
output_dir = Path("catalyst_bo_results")
output_dir.mkdir(exist_ok=True)

# Save the configuration
with open(output_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

# Helper functions
def generate_id():
    return str(uuid.uuid4())[:8]

def latin_hypercube_sampling(n_samples, parameter_space):
    """Simple Latin Hypercube Sampling implementation"""
    continuous_params = []
    categorical_params = []
    integer_params = []
    
    # Separate parameters by type
    for param in parameter_space["parameters"]:
        if param["type"] == "continuous":
            continuous_params.append(param)
        elif param["type"] == "categorical":
            categorical_params.append(param)
        elif param["type"] == "integer":
            integer_params.append(param)
    
    # Generate samples
    samples = []
    for _ in range(n_samples):
        sample = {}
        
        # Handle continuous parameters
        for param in continuous_params:
            min_val, max_val = param["bounds"]
            sample[param["name"]] = np.random.uniform(min_val, max_val)
        
        # Handle integer parameters
        for param in integer_params:
            min_val, max_val = param["bounds"]
            sample[param["name"]] = np.random.randint(min_val, max_val + 1)
        
        # Handle categorical parameters
        for param in categorical_params:
            choices = param["choices"]
            sample[param["name"]] = np.random.choice(choices)
        
        samples.append(sample)
    
    # Apply constraints if any
    valid_samples = []
    for sample in samples:
        valid = True
        
        # Check constraints
        for constraint in parameter_space.get("constraints", []):
            if constraint["type"] == "sum":
                param_names = constraint["parameters"]
                relation = constraint["relation"]
                value = constraint["value"]
                
                param_sum = sum(sample[name] for name in param_names)
                
                if relation == "<=" and param_sum > value:
                    valid = False
                elif relation == ">=" and param_sum < value:
                    valid = False
                elif relation == "==" and param_sum != value:
                    valid = False
        
        if valid:
            valid_samples.append(sample)
    
    # If we don't have enough valid samples, generate more
    while len(valid_samples) < n_samples:
        sample = {}
        
        # Handle continuous parameters
        for param in continuous_params:
            min_val, max_val = param["bounds"]
            sample[param["name"]] = np.random.uniform(min_val, max_val)
        
        # Handle integer parameters
        for param in integer_params:
            min_val, max_val = param["bounds"]
            sample[param["name"]] = np.random.randint(min_val, max_val + 1)
        
        # Handle categorical parameters
        for param in categorical_params:
            choices = param["choices"]
            sample[param["name"]] = np.random.choice(choices)
        
        # Check constraints
        valid = True
        for constraint in parameter_space.get("constraints", []):
            if constraint["type"] == "sum":
                param_names = constraint["parameters"]
                relation = constraint["relation"]
                value = constraint["value"]
                
                param_sum = sum(sample[name] for name in param_names)
                
                if relation == "<=" and param_sum > value:
                    valid = False
                elif relation == ">=" and param_sum < value:
                    valid = False
                elif relation == "==" and param_sum != value:
                    valid = False
        
        if valid:
            valid_samples.append(sample)
            if len(valid_samples) >= n_samples:
                break
    
    return valid_samples[:n_samples]

def simulate_experiment(design):
    """Simulate an experiment with the given design parameters"""
    # This is a mock function that simulates experimental results
    # In a real system, this would be replaced by actual experimental data
    
    # Extract parameters
    Ni_ratio = design.get("Ni_ratio", 0)
    Co_ratio = design.get("Co_ratio", 0)
    Fe_ratio = design.get("Fe_ratio", 0)
    sintering_temperature = design.get("sintering_temperature", 600)
    electrolyte_concentration = design.get("electrolyte_concentration", 1.0)
    current_density = design.get("current_density", 20)
    deposition_time = design.get("deposition_time", 300)
    precursor_pH = design.get("precursor_pH", 7.0)
    
    # Categorical parameters (convert to numerical factors for simulation)
    material_system = design.get("material_system", "Ni-Co")
    material_factor = {"Ni-Co": 1.2, "Ni-Fe": 1.0, "Co-Fe": 0.8}.get(material_system, 1.0)
    
    washing_method = design.get("washing_method", "DI_water")
    washing_factor = {"DI_water": 1.0, "Ethanol": 1.1, "None": 0.7}.get(washing_method, 1.0)
    
    ultrasound_treatment = design.get("ultrasound_treatment", "no")
    ultrasound_factor = 1.2 if ultrasound_treatment == "yes" else 1.0
    
    precursor_type = design.get("precursor_type", "nitrate")
    precursor_factor = {"nitrate": 1.0, "chloride": 0.9, "sulfate": 0.8, "acetate": 1.1}.get(precursor_type, 1.0)
    
    # Simulate LSV_slope (higher is better)
    # Complex interaction between parameters
    LSV_base = 5.0
    LSV_composition = (Ni_ratio * 8.0 + Co_ratio * 6.0 + Fe_ratio * 4.0) * material_factor
    LSV_process = (sintering_temperature / 900) * (electrolyte_concentration / 1.0) * (current_density / 30)
    LSV_treatment = washing_factor * ultrasound_factor * precursor_factor
    LSV_pH = 1.0 - abs(precursor_pH - 7.0) / 7.0  # Optimal around pH 7
    LSV_time = min(1.0, deposition_time / 400)  # Saturates after a while
    
    LSV_slope = LSV_base + LSV_composition + LSV_process * LSV_treatment * LSV_pH * LSV_time
    # Add some noise
    LSV_slope *= np.random.normal(1.0, 0.1)
    
    # Simulate CV_stability (lower is better)
    # Different interaction pattern
    CV_base = 2.0
    CV_composition = (Ni_ratio * 1.0 + Co_ratio * 2.0 + Fe_ratio * 3.0) * material_factor
    CV_process = (sintering_temperature / 600) * (current_density / 20)
    CV_treatment = (2.0 - washing_factor) * (2.0 - ultrasound_factor) * precursor_factor
    CV_pH = abs(precursor_pH - 4.0) / 4.0  # Optimal around pH 4
    CV_time = max(0.5, min(1.5, deposition_time / 300))  # U-shaped relationship
    
    CV_stability = CV_base + CV_composition * CV_process * CV_treatment * CV_pH * CV_time
    # Add some noise
    CV_stability *= np.random.normal(1.0, 0.15)
    
    return {
        "LSV_slope": float(LSV_slope),
        "CV_stability": float(CV_stability)
    }

def random_next_designs(n_samples, parameter_space, previous_results=None):
    """Generate random next designs (in a real system, this would use Bayesian Optimization)"""
    return latin_hypercube_sampling(n_samples, parameter_space)

# Generate initial designs
print("\nGenerating initial designs using Latin Hypercube Sampling...")
n_initial = 10
initial_designs = latin_hypercube_sampling(n_initial, config)

# Save initial designs
initial_designs_with_ids = []
for design in initial_designs:
    initial_designs_with_ids.append({
        "id": generate_id(),
        "parameters": design
    })

with open(output_dir / "initial_designs.json", "w") as f:
    json.dump(initial_designs_with_ids, f, indent=2)

print(f"Generated {len(initial_designs_with_ids)} initial designs")

# Simulate experiments for initial designs
print("\nSimulating experiments for initial designs...")
results = []
for design in initial_designs_with_ids:
    objectives = simulate_experiment(design["parameters"])
    results.append({
        "id": design["id"],
        "parameters": design["parameters"],
        "objectives": objectives,
        "timestamp": datetime.now().isoformat()
    })

# Save results
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Simulated {len(results)} experiments")

# Run BO iterations
n_iterations = 3
batch_size = 5
all_results = results.copy()

print("\nRunning Bayesian Optimization iterations...")
for i in range(n_iterations):
    print(f"\nIteration {i+1}/{n_iterations}")
    
    # Generate next designs
    next_designs = random_next_designs(batch_size, config, all_results)
    
    # Add IDs
    next_designs_with_ids = []
    for design in next_designs:
        next_designs_with_ids.append({
            "id": generate_id(),
            "parameters": design
        })
    
    # Save next designs
    with open(output_dir / f"next_designs_iter_{i+1}.json", "w") as f:
        json.dump(next_designs_with_ids, f, indent=2)
    
    print(f"Generated {len(next_designs_with_ids)} next designs")
    
    # Simulate experiments for next designs
    iter_results = []
    for design in next_designs_with_ids:
        objectives = simulate_experiment(design["parameters"])
        iter_results.append({
            "id": design["id"],
            "parameters": design["parameters"],
            "objectives": objectives,
            "timestamp": datetime.now().isoformat()
        })
    
    # Save iteration results
    with open(output_dir / f"results_iter_{i+1}.json", "w") as f:
        json.dump(iter_results, f, indent=2)
    
    # Add to all results
    all_results.extend(iter_results)
    
    print(f"Simulated {len(iter_results)} experiments")

# Save all results
with open(output_dir / "all_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nCompleted {n_iterations} iterations with {len(all_results)} total experiments")

# Analyze results
print("\nAnalyzing results...")

# Extract objective values
LSV_values = [result["objectives"]["LSV_slope"] for result in all_results]
CV_values = [result["objectives"]["CV_stability"] for result in all_results]

# Find best results
best_LSV_idx = np.argmax(LSV_values)
best_CV_idx = np.argmin(CV_values)

print(f"\nBest LSV_slope (maximize): {LSV_values[best_LSV_idx]:.4f}")
print("Parameters:")
for key, value in all_results[best_LSV_idx]["parameters"].items():
    print(f"  {key}: {value}")

print(f"\nBest CV_stability (minimize): {CV_values[best_CV_idx]:.4f}")
print("Parameters:")
for key, value in all_results[best_CV_idx]["parameters"].items():
    print(f"  {key}: {value}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(LSV_values, CV_values, c=range(len(LSV_values)), cmap='viridis', alpha=0.7)
plt.colorbar(label='Experiment Index')
plt.xlabel('LSV_slope (maximize)')
plt.ylabel('CV_stability (minimize)')
plt.title('Pareto Front Approximation')
plt.grid(True, alpha=0.3)
plt.savefig(output_dir / "pareto_front.png")

print("\nResults saved to the 'catalyst_bo_results' directory")
print("Pareto front plot saved as 'pareto_front.png'")
