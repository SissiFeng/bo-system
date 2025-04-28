import json

# Load the initial designs
with open('backend/catalyst_bo_results/initial_designs.json', 'r') as f:
    initial_designs = json.load(f)

# Load all results
with open('backend/catalyst_bo_results/all_results.json', 'r') as f:
    all_results = json.load(f)

# Check metal ratio sum constraint for initial designs
print("Checking metal ratio sum constraint for initial designs:")
for design in initial_designs:
    params = design['parameters']
    total = params['Ni_ratio'] + params['Co_ratio'] + params['Fe_ratio']
    status = "✅" if total <= 1.0 else "❌"
    print(f"Design {design['id']}: Metal ratio sum = {total:.4f} {status}")

# Check initial designs summary
print("\nInitial Designs Summary:")
valid_count = sum(1 for design in initial_designs if (design['parameters']['Ni_ratio'] + design['parameters']['Co_ratio'] + design['parameters']['Fe_ratio']) <= 1.0)
total_count = len(initial_designs)
print(f"Valid designs: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")

# Check metal ratio sum constraint for all results
print("\nChecking metal ratio sum constraint for all results:")
for result in all_results:
    params = result['parameters']
    total = params['Ni_ratio'] + params['Co_ratio'] + params['Fe_ratio']
    status = "✅" if total <= 1.0 else "❌"
    print(f"Result {result['id']}: Metal ratio sum = {total:.4f} {status}")

# Check all results summary
print("\nAll Results Summary:")
valid_count = sum(1 for result in all_results if (result['parameters']['Ni_ratio'] + result['parameters']['Co_ratio'] + result['parameters']['Fe_ratio']) <= 1.0)
total_count = len(all_results)
print(f"Valid results: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
