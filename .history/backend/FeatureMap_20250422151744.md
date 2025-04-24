# Feature Map

This document visualizes the relationships between features in the BO Engine API, showing dependencies and connections.

## Core Components

```
                                         +---------------------+
                                         |                     |
                                    +--->| Surrogate Models    |
                                    |    | (bo_engine/models/) |
                                    |    |                     |
                                    |    +---------------------+
                                    |
+----------------------+      +---------------------+      +----------------------+
|                      |      |                     |      |                      |
| Parameter Space      |----->| Optimizer           |----->| Acquisition Functions|
| (parameter_space.py) |      | (optimizer.py)      |      | (acquisition.py)     |
|                      |      |                     |      |                      |
+----------------------+      +---------------------+      +----------------------+
        |                            ^   |
        |                            |   |
        v                            |   v
+----------------------+             |  +----------------------+
|                      |             |  |                      |
| Design Generator     |-------------+  | Result Storage       |
| (design_generator.py)|                | (In-Memory/Files)    |
|                      |                |                      |
+----------------------+                +----------------------+
```

## API Endpoints

```
                              +---------------------+
                              |                     |
                              | FastAPI Application |
                              | (app/main.py)       |
                              |                     |
                              +---------------------+
                                        |
                                        |
                                        v
                              +---------------------+
                              |                     |
                              | API Router          |
                              | (app/api/endpoints) |
                              |                     |
                              +---------------------+
                                        |
                +------------------------+------------------------+
                |                        |                        |
                v                        v                        v
    +---------------------+  +---------------------+  +---------------------+
    |                     |  |                     |  |                     |
    | Parameter Space API |  | Experiment API      |  | Task Management API |
    | - Create            |  | - Get designs       |  | - List tasks        |
    | - Update            |  | - Submit results    |  | - Get status        |
    | - Get               |  | - Get predictions   |  | - Export data       |
    |                     |  |                     |  | - Restart           |
    +---------------------+  +---------------------+  +---------------------+
                |               |         |                      |
                |               |         |                      |
                v               v         v                      v
    +---------------------+  +-----+  +-----+  +---------------------+
    |                     |  |     |  |     |  |                     |
    | Strategy API        |  | ... |  | ... |  | WebSocket API       |
    | - Set               |  |     |  |     |  | - Real-time updates |
    | - Get               |  |     |  |     |  |                     |
    |                     |  |     |  |     |  |                     |
    +---------------------+  +-----+  +-----+  +---------------------+
```

## Data Flow

```
   +-----------------+
   | User/Client     |
   | (Frontend)      |
   +-----------------+
           |
           | 1. Define Parameter Space
           v
   +-----------------+
   | Parameter Space |
   | API             |
   +-----------------+
           |
           | 2. Set Strategy
           v
   +-----------------+
   | Strategy API    |
   +-----------------+
           |
           | 3. Get Initial Designs
           v
   +-----------------+
   | Design Generator|<--------+
   +-----------------+         |
           |                   |
           | 4. Return Designs |
           v                   |
   +-----------------+         |
   | User/Client     |         |
   | (Experiments)   |         |
   +-----------------+         |
           |                   |
           | 5. Submit Results |
           v                   |
   +-----------------+         |
   | Results API     |         |
   +-----------------+         |
           |                   |
           | 6. Train Model    |
           v                   |
   +-----------------+         |
   | Surrogate Model |         |
   +-----------------+         |
           |                   |
           | 7. Optimize       |
           v                   |
   +-----------------+         |
   | Acquisition     |         |
   | Function        |---------+
   +-----------------+         |
           |                   |
           | 8. Get Next Designs
           v                   |
   +-----------------+         |
   | User/Client     |---------+
   +-----------------+
```

## Feature Dependencies

| Feature | Depends On | Required By |
|---------|------------|-------------|
| Parameter Space | None | Design Generator, Optimizer |
| Design Generator | Parameter Space | Initial Experiment API |
| Surrogate Models | Parameter Space, Results | Acquisition Functions |
| Acquisition Functions | Surrogate Models, Parameter Space | Optimizer |
| Optimizer | Parameter Space, Surrogate Models, Acquisition Functions | Next Design API |
| Parameter Space API | Parameter Space | Strategy API |
| Strategy API | Parameter Space API | Design API |
| Result Submission API | Parameter Space | Optimizer |
| Next Design API | Optimizer | None |
| Task Management | All other features | None |
| WebSocket API | Task status events | None |

## Implementation Status

| Feature | Status | Priority | Complexity | Notes |
|---------|--------|----------|------------|-------|
| Basic FastAPI Setup | ✅ Complete | High | Low | App structure, config, logging |
| Parameter Space | 🔄 In Progress | High | Medium | Core data structure for BO |
| Design Generator | 📝 Planned | High | Medium | Needed for initial designs |
| Surrogate Models | 📝 Planned | High | High | Core of Bayesian modeling |
| Acquisition Functions | 📝 Planned | High | High | Core of Bayesian optimization |
| Optimizer | 📝 Planned | High | High | Ties everything together |
| Parameter Space API | ✅ Complete | High | Low | Endpoints created, needs core integration |
| Strategy API | ✅ Complete | Medium | Low | Endpoints created, needs core integration |
| Initial Design API | ✅ Complete | High | Low | Endpoints created, needs core integration |
| Result Submission API | ✅ Complete | High | Low | Endpoints created, needs core integration |
| Next Design API | ✅ Complete | High | Low | Endpoints created, needs core integration |
| Model Analysis API | ✅ Complete | Medium | Medium | Endpoints created, needs core integration |
| Task Management API | ✅ Complete | Medium | Low | Basic functionality implemented |
| WebSocket API | ✅ Complete | Low | Medium | Basic implementation done |
| Persistent Storage | 📝 Planned | Medium | Medium | Using file-based storage for now |
| Multi-objective Optimization | 📝 Planned | Low | High | Extension for Phase 6 |
| Batch Optimization | 📝 Planned | Low | High | Extension for Phase 6 | 
