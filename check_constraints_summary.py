import json

# Load the initial designs
with open('backend/catalyst_bo_results/initial_designs.json', 'r') as f:
    initial_designs = json.load(f)

# Load all results
with open('backend/catalyst_bo_results/all_results.json', 'r') as f:
    all_results = json.load(f)

# Check initial designs summary
print("Initial Designs Summary:")
print(f"Total initial designs: {len(initial_designs)}")

valid_count = 0
for design in initial_designs:
    params = design['parameters']
    total = params['Ni_ratio'] + params['Co_ratio'] + params['Fe_ratio']
    if total <= 1.0:
        valid_count += 1

total_count = len(initial_designs)
print(f"Valid designs: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}% satisfy Ni+Co+Fe <= 1.0)")

# Check all results summary
print("\nAll Results Summary:")
print(f"Total results: {len(all_results)}")

valid_count = 0
for result in all_results:
    params = result['parameters']
    total = params['Ni_ratio'] + params['Co_ratio'] + params['Fe_ratio']
    if total <= 1.0:
        valid_count += 1

total_count = len(all_results)
print(f"Valid results: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}% satisfy Ni+Co+Fe <= 1.0)")
