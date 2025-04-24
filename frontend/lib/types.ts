export enum ParameterType {
  CONTINUOUS = "continuous",
  DISCRETE = "discrete",
  CATEGORICAL = "categorical",
}

export interface Parameter {
  id: string
  name: string
  type: ParameterType
  min?: number
  max?: number
  step?: number
  values?: string[]
}

export enum OptimizationType {
  MAXIMIZE = "maximize",
  MINIMIZE = "minimize",
  TARGET_RANGE = "target_range",
}

export interface Objective {
  id: string
  name: string
  type: OptimizationType
  targetMin?: number
  targetMax?: number
}

export enum ConstraintType {
  SUM_EQUALS = "sum_equals",
  SUM_LESS_THAN = "sum_less_than",
  SUM_GREATER_THAN = "sum_greater_than",
}

export interface Constraint {
  id: string
  expression: string
  type: ConstraintType
  value: number
}
