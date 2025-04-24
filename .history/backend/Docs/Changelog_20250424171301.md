# Bayesian Optimization System - Changelog

## [0.5.0] - 2025-04-22

### Added

- Complete implementation of task management module, supporting multi-task persistent management
- Task status tracking and progress calculation functionality
- Optimization strategy configuration support, stored in strategy.json file
- Task export functionality, supporting JSON and CSV formats
- Task restart functionality, with options to retain or clear historical data
- Task diagnostic API, providing detailed diagnostic information
- Lazy loading mechanism, loading task data from the file system as needed

### Improved

- Enhanced get_or_create_optimizer function, supporting model and acquisition function loading from strategy configuration
- Optimized error handling mechanism with detailed logging
- Implemented task_info.json persistence to ensure task state consistency
- Improved task list API, supporting scanning of all tasks in the file system

### Documentation

- Added task management rules documentation, detailing design principles and implementation rules
- Updated development log, recording the completion of Phase 5

## [0.4.0] - 2025-04-15

### Added

- Core Bayesian optimization engine implementation
- Gaussian Process model (GaussianProcessModel) implementation
- Expected Improvement (ExpectedImprovement) acquisition function implementation
- BayesianOptimizer class implementation, supporting observe and suggest functionality

### Improved

- Enhanced parameter space definition functionality, supporting more validation rules
- Enhanced initial experiment design functionality, supporting parameter constraints
- Optimized internal data structures for improved performance

### Fixed

- Fixed crash issues in design generator under special parameter combinations
- Fixed boundary condition handling in parameter validation process

## [0.3.0] - 2025-04-08

### Added

- Initial experiment design API implementation
- Design generator supporting multiple sampling methods (LHS, Random, Grid)
- Design point validation functionality, ensuring parameter space constraints are met

### Improved

- Enhanced parameter space API, supporting more parameter types and validation
- Enhanced FastAPI error handling, providing more user-friendly error messages

## [0.2.0] - 2025-04-01

### Added

- Parameter space definition API implementation
- Parameter type support (continuous, integer, categorical)
- Objective function definition support
- Constraint condition definition support
- Data validation and conversion functionality

## [0.1.0] - 2025-03-25

### Added

- Project basic infrastructure setup
- FastAPI application initialization
- Core data model definition
- Logging and configuration management
- Basic API structure

All notable changes to the BO Engine API will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Basic FastAPI application setup with configuration and logging
- API endpoint stubs with placeholder implementations
- Pydantic models for all API request/response schemas
- Initial documentation structure
- Docker configuration
- WebSocket support for real-time updates
- In-memory and file-based task storage

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - Planned Initial Release

### Added
- Parameter space definition and validation
- Design generator with LHS support
- Basic surrogate models (GPR, RFR)
- Acquisition functions (EI, UCB, PI)
- Core optimizer implementation
- Full API integration with BO engine
- Unit and integration tests
- Complete documentation 
