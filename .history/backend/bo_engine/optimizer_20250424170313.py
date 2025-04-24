# backend/bo_engine/optimizer.py
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Type
import logging
from datetime import datetime

from .parameter_space import ParameterSpace
from .models.base_model import BaseModel
from .acquisition.base_acquisition import BaseAcquisitionFunction
# Import specific implementations later
# from .models.gaussian_process import GaussianProcessModel
# from .acquisition.expected_improvement import ExpectedImprovement

logger = logging.getLogger(__name__)

class BayesianOptimizer:
    """
    Coordinates the Bayesian Optimization process.

    Manages the surrogate model, acquisition function, and optimization loop.
    """
    def __init__(
        self,
        parameter_space: ParameterSpace,
        model_class: Type[BaseModel],
        acquisition_class: Type[BaseAcquisitionFunction],
        model_config: Optional[Dict[str, Any]] = None,
        acquisition_config: Optional[Dict[str, Any]] = None,
        initial_X: Optional[np.ndarray] = None,
        initial_y: Optional[np.ndarray] = None,
    ):
        """
        Initialize the Bayesian Optimizer.

        Args:
            parameter_space: The parameter space for the optimization.
            model_class: The class of the surrogate model to use (e.g., GaussianProcessModel).
            acquisition_class: The class of the acquisition function (e.g., ExpectedImprovement).
            model_config: Configuration dictionary for the surrogate model.
            acquisition_config: Configuration dictionary for the acquisition function.
            initial_X: Initial design points (internal [0, 1] space), shape (n_initial, n_dims).
            initial_y: Initial objective values, shape (n_initial, n_objectives).
                       Assume single objective (n_initial,) for now.
        """
        self.parameter_space = parameter_space
        self.model_config = model_config or {}
        self.acquisition_config = acquisition_config or {}

        # Instantiate model and acquisition function
        self.model = model_class(parameter_space, **self.model_config)
        self.acquisition_function = acquisition_class(parameter_space, **self.acquisition_config)

        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = [] # Assuming single objective for now
        self.iterations = 0
        self.history: List[Dict[str, Any]] = [] # Track optimization history

        if initial_X is not None and initial_y is not None:
            if len(initial_X) != len(initial_y):
                raise ValueError("Initial X and y must have the same number of samples.")
            if initial_X.shape[1] != self.parameter_space.get_internal_dimensions():
                 raise ValueError("Initial X dimensions do not match parameter space internal dimensions.")

            # Convert initial data to list format for internal storage
            self.X_observed.extend(list(initial_X))
            self.y_observed.extend(list(initial_y))
            self.iterations = len(initial_X)
            logger.info(f"Initialized optimizer with {self.iterations} initial points.")

            # Train initial model
            self._train_model()
        else:
             logger.info("Initialized optimizer with no initial data.")

    def _train_model(self):
        """Trains the surrogate model with currently observed data."""
        if not self.X_observed or not self.y_observed:
            logger.warning("No data available to train the model.")
            return

        X_train = np.array(self.X_observed)
        y_train = np.array(self.y_observed)

        try:
            logger.info(f"Training model with {len(X_train)} data points...")
            start_time = datetime.now()
            self.model.train(X_train, y_train)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Model training completed in {duration:.2f} seconds.")
        except Exception as e:
            logger.error(f"Model training failed: {e}", exc_info=True)
            # Decide how to handle training failure (e.g., raise, log and skip suggestion)

    def suggest(self, n_suggestions: int = 1) -> List[Dict[str, Any]]:
        """
        Suggest the next point(s) to evaluate.

        Returns:
            List[Dict[str, Any]]: A list of suggested points in the original parameter space format.
                                  Each dict contains parameter names and their values.
        """
        if not self.model.is_trained():
            logger.error("Model is not trained. Cannot suggest points.")
            # Option 1: Raise error
            # raise RuntimeError("Model must be trained before suggesting points.")
            # Option 2: Fallback to random sampling
            logger.warning("Falling back to random sampling as model is not trained.")
            random_points_internal = np.random.rand(n_suggestions, self.parameter_space.get_internal_dimensions())
            suggested_points = [self.parameter_space.internal_to_point(p) for p in random_points_internal]
            return suggested_points

        suggested_points_internal = []
        for _ in range(n_suggestions): # Simple sequential suggestion for now
            try:
                 # Find the point maximizing the acquisition function
                 # Additional args for acquisition (e.g., best observed y) might be needed
                 best_y = min(self.y_observed) if self.y_observed else None # Assuming minimization
                 # Pass necessary arguments like best_f for EI
                 acq_kwargs = {'best_f': best_y} if best_y is not None else {}

                 logger.info("Optimizing acquisition function...")
                 start_time = datetime.now()
                 next_point_internal, acq_value = self.acquisition_function.optimize(
                     self.model,
                     **acq_kwargs
                     # Pass optimization control parameters if needed: n_restarts=..., raw_samples=...
                 )
                 duration = (datetime.now() - start_time).total_seconds()
                 logger.info(f"Acquisition optimization completed in {duration:.2f} seconds. Max value: {acq_value:.4f}")

                 # TODO: Add logic to handle batch suggestions (e.g., qEI, Kriging Believer)
                 # TODO: Add logic to avoid suggesting already evaluated or very close points
                 suggested_points_internal.append(next_point_internal[0]) # Optimize returns shape (1, dim)

            except Exception as e:
                logger.error(f"Failed to optimize acquisition function: {e}. Falling back to random.", exc_info=True)
                # Fallback: suggest a random point if optimization fails
                suggested_points_internal.append(np.random.rand(self.parameter_space.get_internal_dimensions()))

        # Convert internal points back to the original parameter space format
        suggested_points_external = [self.parameter_space.internal_to_point(p) for p in suggested_points_internal]

        return suggested_points_external

    def observe(self, X: List[Dict[str, Any]], y: List[float]):
        """
        Register new observations.

        Args:
            X: List of observed points in the original parameter space format.
            y: List of corresponding objective values. Assume single objective.
        """
        if len(X) != len(y):
            raise ValueError("Number of points X and objective values y must match.")

        logger.info(f"Observing {len(X)} new data point(s).")

        for i in range(len(X)):
            point_external = X[i]
            value = y[i]

            try:
                # Convert point to internal representation
                point_internal = self.parameter_space.point_to_internal(point_external)

                # Store observation
                self.X_observed.append(point_internal)
                self.y_observed.append(value)
                self.iterations += 1

                # Record history (optional)
                self.history.append({
                    "iteration": self.iterations,
                    "point_external": point_external,
                    "point_internal": point_internal.tolist(), # Convert numpy array for logging/storage
                    "objective": value,
                    "timestamp": datetime.now().isoformat(),
                })

            except Exception as e:
                 logger.error(f"Failed to process observation {i+1}: Point={point_external}, Value={value}. Error: {e}", exc_info=True)

        # Retrain the model with new data
        self._train_model()

    @property
    def current_best(self) -> Optional[Dict[str, Any]]:
        """
        Returns the best observation found so far (point and objective value).
        Assumes minimization.
        """
        if not self.y_observed:
            return None

        best_idx = np.argmin(self.y_observed)
        best_y_val = self.y_observed[best_idx]
        best_x_internal = self.X_observed[best_idx]

        try:
            best_x_external = self.parameter_space.internal_to_point(best_x_internal)
            return {
                "parameters": best_x_external,
                "objective": best_y_val,
                "iteration_found": self.history[best_idx]['iteration'] if best_idx < len(self.history) else None
            }
        except Exception as e:
            logger.error(f"Failed to convert best internal point back to external: {e}", exc_info=True)
            return {"objective": best_y_val, "parameters": "Error converting point"} # Provide partial info


# Example usage structure:
# ps = ParameterSpace(...)
# optimizer = BayesianOptimizer(ps, GaussianProcessModel, ExpectedImprovement, initial_X=..., initial_y=...)
# next_points = optimizer.suggest(n_suggestions=1)
# results = run_experiments(next_points) # Function to evaluate the objective function
# optimizer.observe(next_points, results)
# best = optimizer.current_best 
