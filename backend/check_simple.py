import json

# Load all results
with open('catalyst_bo_results/all_results.json', 'r') as f:
    all_results = json.load(f)

# Check constraints
valid_count = 0
for result in all_results:
    params = result['parameters']
    total = params['Ni_ratio'] + params['Co_ratio'] + params['Fe_ratio']
    if total <= 1.0:
        valid_count += 1

total_count = len(all_results)
print(f"Valid results: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
