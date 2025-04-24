# Frequently Asked Questions (FAQ)

## General Questions

### What is the BO Engine API?

The BO Engine API is a microservice that provides Bayesian Optimization (BO) capabilities through a RESTful API. It allows users to define parameter spaces, experimental designs, and optimization strategies, and then receive recommendations for experiments to run in order to optimize objectives.

### What is Bayesian Optimization?

Bayesian Optimization is a sequential design strategy for global optimization of black-box functions. It's particularly useful when:
- Function evaluations are expensive or time-consuming
- The objective function has no known analytical form (black-box)
- The function is non-convex with potentially multiple local optima

The BO approach builds a surrogate model (typically a Gaussian Process) of the objective function and uses an acquisition function to determine the next most promising points to evaluate.

### How does the BO Engine API differ from other optimization tools?

Unlike traditional optimization libraries that require embedding directly into your code, the BO Engine API:
- Provides a language-agnostic REST API interface
- Supports persistent optimization tasks across multiple sessions
- Offers WebSocket capabilities for real-time updates
- Is designed for integration with web applications and UIs
- Enables distributed experimentation workflows

## Technical Questions

### What programming languages and frameworks does the BO Engine API use?

The BO Engine API is built with:
- Python 3.10+
- FastAPI for the web framework
- Pydantic for data validation
- Scikit-learn, GPyTorch, and BoTorch for machine learning capabilities
- Pandas and NumPy for data handling
- Docker for containerization

### How do I run the API locally?

See the [DevEnvConfig.md](DevEnvConfig.md) document for detailed instructions on setting up and running the API locally.

### Can I deploy the BO Engine API to the cloud?

Yes, the API is containerized using Docker, making it easy to deploy to any cloud provider that supports Docker containers, such as:
- AWS (ECS, EKS, Fargate)
- Google Cloud (GKE, Cloud Run)
- Azure (AKS, Container Instances)
- Any Kubernetes cluster

### How does the API handle large-scale optimization tasks?

For large-scale optimization:
- Long-running computations are handled asynchronously
- Results are persisted to disk for reliability
- The system can be scaled horizontally by deploying multiple instances
- For very large workloads, the API can be configured to use distributed computing frameworks

## Usage Questions

### How do I define a parameter space?

You define a parameter space by sending a POST request to the `/api/parameter-space` endpoint with a JSON payload that defines:
- Parameters (continuous, discrete, or categorical)
- Objectives (maximize or minimize)
- Constraints (optional)

See the API documentation for detailed examples and schema information.

### How many parameters can I optimize simultaneously?

While there's no hard limit, Bayesian Optimization typically performs best with:
- Up to ~20 continuous parameters
- Up to ~10 categorical parameters

Performance depends on the nature of the problem, the number of evaluations you can perform, and the complexity of the objective function landscape.

### How do I choose the right acquisition function?

Common acquisition functions include:
- **Expected Improvement (EI)**: Balanced exploration-exploitation, good default choice
- **Upper Confidence Bound (UCB)**: More exploration, good for noisy objectives
- **Probability of Improvement (PI)**: More exploitation, good when you want to improve on a specific target

The best choice depends on your specific problem. The API allows you to configure the acquisition function via the strategy configuration endpoint.

### Can I optimize multiple objectives simultaneously?

Yes, the BO Engine API supports multi-objective optimization, which will:
- Find the Pareto front of non-dominated solutions
- Use specialized acquisition functions for multi-objective problems (like EHVI)
- Provide tools for analyzing trade-offs between different objectives

### How do I handle constraints in my optimization?

The API supports different types of constraints:
- Linear constraints (sum_equals, sum_less_than, etc.)
- Custom constraints through expression evaluation
- Constraints are enforced during the recommendation of new design points

## Troubleshooting

### What should I do if the API returns an error?

Common solutions for API errors:
1. Check that your request follows the correct schema format
2. Verify that any referenced task IDs exist
3. Ensure your parameter space definition is valid (e.g., min < max for continuous parameters)
4. Check the API logs for more detailed error information

### Why aren't my design recommendations improving the objective?

If recommendation quality seems poor:
1. Ensure you've submitted enough initial results for the model to learn from
2. Check if your objective function is very noisy
3. Consider adjusting the acquisition function or exploration parameters
4. Verify that your parameter space is correctly defined

### How can I reset a task if something goes wrong?

Use the `/api/tasks/{task_id}/restart` endpoint to restart a task with options to:
- Reuse or reset the existing model
- Preserve or clear the experiment history

## Performance and Scaling

### How many concurrent optimization tasks can the API handle?

The number of concurrent tasks depends on:
- Server hardware resources
- Complexity of the surrogate models
- Number of parameters and objectives
- Frequency of recommendations

For production deployments, consider:
- Horizontal scaling with multiple API instances
- Database backends instead of file storage
- Task queues for handling computation-intensive operations

### How does the API handle computationally intensive operations?

For intensive operations like:
- Training surrogate models with large datasets
- Optimizing acquisition functions in high-dimensional spaces
- Handling batch recommendation requests

The API uses:
1. Asynchronous processing
2. Optional background workers (with Celery)
3. Efficient numerical libraries
4. Caching of intermediate results

## Data and Privacy

### Where is my optimization data stored?

By default, data is stored:
- In memory during API operation
- In JSON files in the `data/tasks/` directory
- Completely within your deployment environment

### Is my data secure?

Security considerations:
- The API itself does not implement authentication by default (add this at deployment)
- Data is stored locally to your deployment
- For production use, add authentication/authorization middleware
- Consider encrypted storage for sensitive data 
