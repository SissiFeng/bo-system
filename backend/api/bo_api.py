from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
import numpy as np
import os
import json
import logging
from pydantic import BaseModel, Field

from bo_engine.parameter_space import ParameterSpace, ParameterType
from bo_engine.design_generator import DesignType
from bo_engine.bo_system import BOSystem, AcquisitionFunction
from bo_engine.utils import save_to_json, load_from_json, ensure_directory_exists

# 配置日志记录器
logger = logging.getLogger("api.bo_api")

# 创建路由器
router = APIRouter(prefix="/api/bo", tags=["bayesian-optimization"])

# 数据模型定义
class Parameter(BaseModel):
    name: str
    type: ParameterType
    bounds: List[float] = None
    categories: List[str] = None
    integer: bool = False
    log_scale: bool = False
    description: str = ""

class Constraint(BaseModel):
    expression: str
    description: str = ""

class ParameterSpaceConfig(BaseModel):
    parameters: List[Parameter]
    constraints: List[Constraint] = []

class ObjectiveConfig(BaseModel):
    name: str
    direction: str = "minimize"  # 'minimize' 或 'maximize'
    description: str = ""

class ExperimentConfig(BaseModel):
    name: str
    parameter_space: ParameterSpaceConfig
    objectives: List[ObjectiveConfig]
    design_type: DesignType = DesignType.LATIN_HYPERCUBE
    initial_designs: int = 10
    acquisition_function: AcquisitionFunction = AcquisitionFunction.EI
    exploration_weight: float = 0.1
    seed: Optional[int] = None
    description: str = ""

class DesignPoint(BaseModel):
    parameters: Dict[str, Any]
    id: Optional[str] = None

class EvaluationResult(BaseModel):
    design_id: str
    objectives: Dict[str, float]
    constraints_satisfied: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NextDesignsRequest(BaseModel):
    experiment_id: str
    count: int = 1
    acquisition_function: Optional[AcquisitionFunction] = None
    exploration_weight: Optional[float] = None

class ExperimentSummary(BaseModel):
    id: str
    name: str
    description: str
    status: str
    iteration: int
    best_objectives: Dict[str, float] = None
    created_at: str
    updated_at: str

# 新增数据模型定义
class StrategyConfig(BaseModel):
    algorithm: str = "bayesian"
    acquisition_function: AcquisitionFunction = AcquisitionFunction.EI
    surrogate_model: str = "gaussian_process"
    acquisition_optimizer: str = "lbfgs"
    exploration_weight: float = 0.1
    model_update_interval: int = 1
    batch_size: int = 1
    random_seed: Optional[int] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)

class ModelPredictionRequest(BaseModel):
    points: List[Dict[str, Any]]

class ParameterSpaceUpdateRequest(BaseModel):
    parameters: Dict[str, Dict[str, Any]] = None
    constraints: List[Dict[str, Any]] = None
    objectives: Dict[str, str] = None
    description: Optional[str] = None

class TaskRestartRequest(BaseModel):
    strategy: str = "reuse_last"  # "reuse_last" or "reset"
    preserve_history: bool = True

# 系统状态存储
experiments = {}  # 存储所有实验的字典
data_dir = "data/bo_experiments"
ensure_directory_exists(data_dir)

# 工具函数
def get_experiment(experiment_id: str) -> BOSystem:
    """获取实验实例，如果不存在则抛出异常"""
    if experiment_id not in experiments:
        # 尝试从文件加载
        filepath = os.path.join(data_dir, f"{experiment_id}.json")
        if os.path.exists(filepath):
            try:
                experiment = BOSystem.load(filepath)
                experiments[experiment_id] = experiment
                return experiment
            except Exception as e:
                logger.error(f"加载实验 {experiment_id} 失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"加载实验失败: {str(e)}")
        else:
            raise HTTPException(status_code=404, detail=f"实验 {experiment_id} 不存在")
    return experiments[experiment_id]

def save_experiment_state(experiment_id: str):
    """保存实验状态到文件"""
    if experiment_id in experiments:
        experiment = experiments[experiment_id]
        filepath = os.path.join(data_dir, f"{experiment_id}.json")
        experiment.save(filepath)

# API端点实现
@router.post("/experiments", response_model=dict)
async def create_experiment(config: ExperimentConfig):
    """创建新的优化实验"""
    try:
        # 构建参数空间
        parameters = []
        for param in config.parameter_space.parameters:
            if param.type == ParameterType.CONTINUOUS or param.type == ParameterType.INTEGER:
                parameters.append({
                    "name": param.name,
                    "type": param.type,
                    "bounds": param.bounds,
                    "integer": param.integer,
                    "log_scale": param.log_scale,
                    "description": param.description
                })
            elif param.type == ParameterType.CATEGORICAL:
                parameters.append({
                    "name": param.name,
                    "type": param.type,
                    "categories": param.categories,
                    "description": param.description
                })
        
        # 构建约束条件
        constraints = [{"expression": c.expression, "description": c.description} 
                       for c in config.parameter_space.constraints]
        
        # 创建参数空间
        parameter_space = ParameterSpace(parameters=parameters, constraints=constraints)
        
        # 创建实验输出目录
        experiment_dir = os.path.join(data_dir, config.name.replace(" ", "_"))
        ensure_directory_exists(experiment_dir)
        
        # 创建优化系统
        bo_system = BOSystem(
            parameter_space=parameter_space,
            output_dir=experiment_dir,
            seed=config.seed,
            acquisition_function=config.acquisition_function,
            exploration_weight=config.exploration_weight
        )
        
        # 生成初始设计方案
        initial_designs = bo_system.initialize_design(
            n_initial_designs=config.initial_designs,
            design_type=config.design_type
        )
        
        # 保存实验配置
        experiment_id = bo_system.system_id
        experiments[experiment_id] = bo_system
        
        # 保存实验状态
        save_experiment_state(experiment_id)
        
        # 返回实验信息和初始设计方案
        return {
            "experiment_id": experiment_id,
            "name": config.name,
            "initial_designs": [{"id": f"initial_{i}", "parameters": design} 
                                for i, design in enumerate(initial_designs)],
            "message": "实验创建成功"
        }
    
    except Exception as e:
        logger.error(f"创建实验失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建实验失败: {str(e)}")

@router.get("/experiments", response_model=List[ExperimentSummary])
async def list_experiments():
    """获取所有实验的列表"""
    try:
        # 加载所有保存的实验
        experiment_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        experiment_summaries = []
        
        for file in experiment_files:
            try:
                filepath = os.path.join(data_dir, file)
                state = load_from_json(filepath)
                
                # 提取基本信息
                exp_id = state.get("system_id", file.replace(".json", ""))
                
                # 如果实验尚未加载到内存，则从文件加载基本信息
                best_objectives = None
                if "best_observation_id" in state and state["best_observation_id"]:
                    for obs in state.get("observations", []):
                        if obs["id"] == state["best_observation_id"]:
                            best_objectives = obs.get("objectives", {})
                            break
                
                summary = ExperimentSummary(
                    id=exp_id,
                    name=os.path.basename(state.get("output_dir", "")),
                    description=state.get("description", ""),
                    status="active",
                    iteration=state.get("iteration", 0),
                    best_objectives=best_objectives,
                    created_at=state.get("created_at", ""),
                    updated_at=state.get("timestamp", "")
                )
                experiment_summaries.append(summary)
            except Exception as e:
                logger.error(f"加载实验摘要 {file} 失败: {str(e)}")
        
        return experiment_summaries
    
    except Exception as e:
        logger.error(f"获取实验列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取实验列表失败: {str(e)}")

@router.get("/experiments/{experiment_id}", response_model=dict)
async def get_experiment_details(experiment_id: str):
    """获取特定实验的详细信息"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 获取实验详情
        observations = experiment.get_optimization_history()
        best_design = experiment.get_best_design()
        best_objectives = experiment.get_best_objectives()
        
        return {
            "experiment_id": experiment_id,
            "parameter_space": experiment.parameter_space.to_dict(),
            "iteration": experiment.iteration,
            "observations": observations,
            "best_design": best_design,
            "best_objectives": best_objectives
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取实验 {experiment_id} 详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取实验详情失败: {str(e)}")

@router.post("/experiments/{experiment_id}/evaluate", response_model=dict)
async def add_evaluation(experiment_id: str, evaluation: EvaluationResult, background_tasks: BackgroundTasks):
    """添加评估结果到实验中"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 获取设计方案
        design_id = evaluation.design_id
        design = None
        
        # 如果是初始设计方案，从评估结果中提取
        if design_id.startswith("initial_"):
            # 这里需要确保前端发送完整的设计方案参数
            if not hasattr(evaluation, "design") or not evaluation.design:
                raise HTTPException(status_code=400, detail="初始设计方案需要提供完整参数")
            design = evaluation.design
        else:
            # 从历史记录中查找
            for obs in experiment.get_optimization_history():
                if obs["id"] == design_id:
                    design = obs["design"]
                    break
            
            if design is None:
                raise HTTPException(status_code=404, detail=f"找不到ID为 {design_id} 的设计方案")
        
        # 添加观测结果
        observation = experiment.add_observation(
            design=design,
            objectives=evaluation.objectives,
            constraints_satisfied=evaluation.constraints_satisfied,
            metadata=evaluation.metadata
        )
        
        # 异步保存实验状态
        background_tasks.add_task(save_experiment_state, experiment_id)
        
        # WebSocket通知 - 添加到background_tasks避免阻塞API响应
        if experiment_id in manager.active_connections:
            is_best = (experiment.best_observation and observation["id"] == experiment.best_observation["id"])
            
            notification = {
                "event_type": "evaluation_completed",
                "task_id": experiment_id,
                "timestamp": observation["timestamp"],
                "data": {
                    "parameters": observation["design"],
                    "objectives": observation["objectives"],
                    "constraints_satisfied": observation["constraints_satisfied"],
                    "evaluations_completed": len(experiment.observations),
                    "is_best": is_best
                }
            }
            
            background_tasks.add_task(manager.send_json, notification, experiment_id)
        
        return {
            "observation_id": observation["id"],
            "message": "评估结果已添加"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加评估结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"添加评估结果失败: {str(e)}")

@router.post("/experiments/{experiment_id}/next", response_model=List[DesignPoint])
async def get_next_designs(experiment_id: str, request: NextDesignsRequest, background_tasks: BackgroundTasks):
    """获取下一批推荐设计方案"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 获取下一批设计方案建议
        next_designs = experiment.get_next_design(
            n_designs=request.count,
            acquisition_function=request.acquisition_function,
            exploration_weight=request.exploration_weight
        )
        
        # 异步保存实验状态
        background_tasks.add_task(save_experiment_state, experiment_id)
        
        # 准备返回的设计点
        design_points = [DesignPoint(parameters=design, id=f"design_{i}") 
                         for i, design in enumerate(next_designs)]
        
        # WebSocket通知 - 添加到background_tasks避免阻塞API响应
        if experiment_id in manager.active_connections:
            notification = {
                "event_type": "recommendation_ready",
                "task_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "designs": [design.dict() for design in design_points],
                    "iteration": experiment.iteration
                }
            }
            
            background_tasks.add_task(manager.send_json, notification, experiment_id)
        
        return design_points
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取下一批设计方案失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取下一批设计方案失败: {str(e)}")

@router.delete("/experiments/{experiment_id}", response_model=dict)
async def delete_experiment(experiment_id: str):
    """删除指定的实验"""
    try:
        # 检查实验是否存在
        if experiment_id not in experiments:
            filepath = os.path.join(data_dir, f"{experiment_id}.json")
            if not os.path.exists(filepath):
                raise HTTPException(status_code=404, detail=f"实验 {experiment_id} 不存在")
        
        # 删除实验文件
        filepath = os.path.join(data_dir, f"{experiment_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # 从内存中移除
        if experiment_id in experiments:
            del experiments[experiment_id]
        
        return {"message": f"实验 {experiment_id} 已删除"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除实验 {experiment_id} 失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除实验失败: {str(e)}")

@router.get("/experiments/{experiment_id}/optimization-history", response_model=dict)
async def get_optimization_history(experiment_id: str):
    """获取优化历史数据，用于前端可视化"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 获取历史数据
        history = experiment.get_optimization_history()
        
        # 提取迭代数和目标值
        iterations = []
        objectives = []
        parameter_values = {}
        
        for observation in history:
            iterations.append(observation["iteration"])
            obj_values = observation["objectives"]
            objectives.append(obj_values)
            
            # 记录参数值
            for param_name, param_value in observation["design"].items():
                if param_name not in parameter_values:
                    parameter_values[param_name] = []
                parameter_values[param_name].append(param_value)
        
        return {
            "iterations": iterations,
            "objectives": objectives,
            "parameter_values": parameter_values,
            "best_objectives": experiment.get_best_objectives()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取优化历史数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取优化历史数据失败: {str(e)}")

# 新增API端点 - 策略配置相关

@router.post("/strategy/{experiment_id}", response_model=dict)
async def set_strategy(experiment_id: str, strategy: StrategyConfig, background_tasks: BackgroundTasks):
    """设置优化策略"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 将策略配置保存到文件
        strategy_file = os.path.join(experiment.output_dir, "strategy.json")
        
        # 将策略配置转换为字典并保存
        strategy_dict = strategy.dict()
        save_to_json(strategy_dict, strategy_file)
        
        # 更新BOSystem的策略参数
        if strategy.acquisition_function:
            experiment.acquisition_function = strategy.acquisition_function
        
        if strategy.exploration_weight:
            experiment.exploration_weight = strategy.exploration_weight
        
        if strategy.random_seed is not None:
            experiment.seed = strategy.random_seed
            # 如果需要重新初始化随机数生成器
            experiment.rng = np.random.RandomState(strategy.random_seed)
        
        # 异步保存实验状态
        background_tasks.add_task(save_experiment_state, experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "status": "strategy_set",
            "message": "优化策略设置成功"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"设置优化策略失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"设置优化策略失败: {str(e)}")

@router.get("/strategy/{experiment_id}", response_model=Dict[str, Any])
async def get_strategy(experiment_id: str):
    """获取当前优化策略"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 尝试从文件加载策略配置
        strategy_file = os.path.join(experiment.output_dir, "strategy.json")
        
        if os.path.exists(strategy_file):
            strategy_dict = load_from_json(strategy_file)
        else:
            # 如果没有单独的策略文件，从BOSystem对象构建默认策略
            strategy_dict = {
                "algorithm": "bayesian",
                "acquisition_function": experiment.acquisition_function.value,
                "surrogate_model": "gaussian_process",  # 默认值
                "acquisition_optimizer": "lbfgs",  # 默认值
                "exploration_weight": experiment.exploration_weight,
                "model_update_interval": 1,  # 默认值
                "batch_size": 1,  # 默认值
                "random_seed": experiment.seed,
                "hyperparameters": {}  # 默认空字典
            }
        
        return strategy_dict
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取优化策略失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取优化策略失败: {str(e)}")

# 新增API端点 - 参数空间更新

@router.put("/parameter-space/{experiment_id}", response_model=dict)
async def update_parameter_space(experiment_id: str, update: ParameterSpaceUpdateRequest, background_tasks: BackgroundTasks):
    """更新参数空间配置"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 获取当前参数空间
        current_params = experiment.parameter_space.get_parameters()
        current_constraints = experiment.parameter_space.get_constraints() if experiment.parameter_space.has_constraints() else []
        
        # 更新参数（如果提供）
        if update.parameters:
            # 检查参数是否可以更新
            if experiment.iteration > 0:
                # 只允许更新现有参数的某些属性，不允许添加/删除参数
                for param_name, param_config in update.parameters.items():
                    if param_name not in current_params:
                        raise HTTPException(status_code=400, detail=f"实验已经开始，不能添加新参数 {param_name}")
                
                # 更新参数描述、重要性等非结构性属性
                for param_name, param_config in update.parameters.items():
                    if "description" in param_config:
                        current_params[param_name]["description"] = param_config["description"]
                    if "importance" in param_config:
                        current_params[param_name]["importance"] = param_config["importance"]
            else:
                # 实验尚未开始，可以完全重新定义参数
                current_params = update.parameters
        
        # 更新约束（如果提供）
        if update.constraints:
            if experiment.iteration > 0:
                # 只允许更新现有约束的描述，不允许添加/删除约束
                for i, constraint in enumerate(update.constraints):
                    if i < len(current_constraints):
                        if "description" in constraint:
                            current_constraints[i]["description"] = constraint["description"]
            else:
                # 实验尚未开始，可以完全重新定义约束
                current_constraints = update.constraints
        
        # 更新目标（如果提供，且实验尚未开始）
        if update.objectives and experiment.iteration == 0:
            # 重建参数空间
            new_parameter_space = ParameterSpace(
                parameters=current_params,
                constraints=current_constraints,
                objectives=update.objectives,
                description=update.description if update.description else experiment.parameter_space.description
            )
            
            # 替换实验中的参数空间
            experiment.parameter_space = new_parameter_space
        
        # 异步保存实验状态
        background_tasks.add_task(save_experiment_state, experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "status": "parameter_space_updated",
            "message": "参数空间已更新"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新参数空间失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新参数空间失败: {str(e)}")

# 新增API端点 - 模型预测相关

@router.post("/predict/{experiment_id}", response_model=Dict[str, Any])
async def predict_points(experiment_id: str, request: ModelPredictionRequest):
    """使用当前模型预测指定参数组合的目标值"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 检查是否有足够的观测数据来构建模型
        if len(experiment.observations) < 2:
            raise HTTPException(status_code=400, detail="观测数据不足，无法进行预测")
        
        # 确保模型已更新
        if not experiment._update_surrogate_model():
            raise HTTPException(status_code=500, detail="模型更新失败，无法进行预测")
        
        # 预测结果列表
        predictions = []
        
        # 对每个参数点进行预测
        for point in request.points:
            try:
                # 将参数点转换为内部表示
                X_point = experiment.parameter_space.transform(point)
                X = np.array([X_point])
                
                # 获取预测
                mean, std = experiment.surrogate_model.predict(X, return_std=True)
                
                # 将预测结果添加到列表中
                # 假设只有一个目标，如果有多个目标需要修改
                objective_name = list(experiment.parameter_space.objectives[0].values())[0]
                predictions.append({
                    "parameters": point,
                    "objectives": {
                        objective_name: {
                            "mean": float(mean[0]),
                            "std": float(std[0])
                        }
                    }
                })
            except Exception as e:
                logger.warning(f"预测参数点 {point} 失败: {str(e)}")
                # 继续预测下一个点
        
        return {
            "experiment_id": experiment_id,
            "predictions": predictions,
            "message": "预测生成成功"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成预测失败: {str(e)}")

@router.get("/model/{experiment_id}/performance", response_model=Dict[str, Any])
async def get_model_performance(experiment_id: str):
    """获取当前模型的性能指标"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 检查是否有足够的观测数据来评估模型
        if len(experiment.observations) < 3:
            raise HTTPException(status_code=400, detail="观测数据不足，无法评估模型性能")
        
        # 确保模型已更新
        if not experiment._update_surrogate_model():
            raise HTTPException(status_code=500, detail="模型更新失败，无法评估性能")
        
        # 留一交叉验证
        X = []
        y = []
        
        # 准备训练数据
        for obs in experiment.observations:
            if obs["constraints_satisfied"]:
                X_point = experiment.parameter_space.transform(obs["design"])
                X.append(X_point)
                
                # 简单起见，只考虑第一个目标
                objective_name = list(obs["objectives"].keys())[0]
                y.append(obs["objectives"][objective_name])
        
        X = np.array(X)
        y = np.array(y)
        
        # 留一交叉验证
        n = len(X)
        errors = []
        predictions = []
        actuals = []
        
        for i in range(n):
            # 训练集
            X_train = np.concatenate([X[:i], X[i+1:]])
            y_train = np.concatenate([y[:i], y[i+1:]])
            
            # 测试点
            X_test = X[i].reshape(1, -1)
            y_test = y[i]
            
            # 使用相同类型的模型进行训练
            cv_model = type(experiment.surrogate_model)()
            cv_model.fit(X_train, y_train)
            
            # 预测
            y_pred, y_std = cv_model.predict(X_test, return_std=True)
            
            # 记录结果
            errors.append(float(y_test - y_pred[0]))
            predictions.append(float(y_pred[0]))
            actuals.append(float(y_test))
        
        # 计算性能指标
        mae = np.mean(np.abs(errors))
        mse = np.mean(np.square(errors))
        rmse = np.sqrt(mse)
        
        # 计算R^2
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum(np.square(errors))
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        # 构建响应
        performance = {
            "experiment_id": experiment_id,
            "model_type": type(experiment.surrogate_model).__name__,
            "metrics": {
                "r2_score": float(r2),
                "mean_absolute_error": float(mae),
                "root_mean_squared_error": float(rmse)
            },
            "cross_validation": {
                "folds": n,
                "predictions": predictions,
                "actuals": actuals,
                "errors": errors
            },
            "message": "模型性能评估成功"
        }
        
        return performance
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"评估模型性能失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"评估模型性能失败: {str(e)}")

@router.get("/pareto/{experiment_id}", response_model=Dict[str, Any])
async def get_pareto_front(experiment_id: str):
    """获取多目标优化问题的帕累托前沿"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 获取观测历史
        observations = experiment.get_optimization_history()
        
        # 检查是否有多个目标
        if len(experiment.parameter_space.objectives) < 2:
            raise HTTPException(status_code=400, detail="单目标优化问题不存在帕累托前沿")
        
        # 提取目标名称
        objective_names = [obj.get("name") for obj in experiment.parameter_space.objectives]
        
        # 确定每个目标的优化方向（最大化/最小化）
        directions = [obj.get("type", "minimize").lower() == "maximize" for obj in experiment.parameter_space.objectives]
        
        # 提取每个观测的目标值
        evaluated_points = []
        for obs in observations:
            if obs["constraints_satisfied"]:
                point = {
                    "id": obs["id"],
                    "parameters": obs["design"],
                    "objectives": {}
                }
                
                # 提取所有目标值
                for obj_name in objective_names:
                    if obj_name in obs["objectives"]:
                        point["objectives"][obj_name] = obs["objectives"][obj_name]
                
                # 只有包含所有目标的观测才被考虑
                if len(point["objectives"]) == len(objective_names):
                    evaluated_points.append(point)
        
        # 识别帕累托前沿上的点
        pareto_front = []
        dominated_solutions = []
        
        for i, point_i in enumerate(evaluated_points):
            is_dominated = False
            
            for j, point_j in enumerate(evaluated_points):
                if i != j:
                    dominates = True
                    
                    for k, obj_name in enumerate(objective_names):
                        # 根据优化方向比较目标值
                        if directions[k]:  # 最大化
                            if point_j["objectives"][obj_name] < point_i["objectives"][obj_name]:
                                dominates = False
                                break
                        else:  # 最小化
                            if point_j["objectives"][obj_name] > point_i["objectives"][obj_name]:
                                dominates = False
                                break
                    
                    if dominates:
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_front.append(point_i)
            else:
                dominated_solutions.append(point_i)
        
        # 计算理想点和非理想点
        ideal_point = {}
        nadir_point = {}
        
        for obj_name in objective_names:
            if directions[objective_names.index(obj_name)]:  # 最大化
                ideal_point[obj_name] = max([p["objectives"][obj_name] for p in evaluated_points])
                nadir_point[obj_name] = min([p["objectives"][obj_name] for p in evaluated_points])
            else:  # 最小化
                ideal_point[obj_name] = min([p["objectives"][obj_name] for p in evaluated_points])
                nadir_point[obj_name] = max([p["objectives"][obj_name] for p in evaluated_points])
        
        return {
            "experiment_id": experiment_id,
            "pareto_front": pareto_front,
            "dominated_solutions": dominated_solutions,
            "ideal_point": ideal_point,
            "nadir_point": nadir_point,
            "message": "帕累托前沿计算成功"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"计算帕累托前沿失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"计算帕累托前沿失败: {str(e)}")

# 新增API端点 - 不确定性分析

@router.get("/uncertainty/{experiment_id}", response_model=Dict[str, Any])
async def get_uncertainty_analysis(experiment_id: str, resolution: int = Query(20, ge=5, le=50)):
    """获取参数空间中的不确定性地图"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 检查是否有足够的观测数据来构建模型
        if len(experiment.observations) < 3:
            raise HTTPException(status_code=400, detail="观测数据不足，无法进行不确定性分析")
        
        # 确保模型已更新
        if not experiment._update_surrogate_model():
            raise HTTPException(status_code=500, detail="模型更新失败，无法进行不确定性分析")
        
        # 获取参数类型和范围
        param_ranges = {}
        param_types = {}
        
        for param_name, param_config in experiment.parameter_space.get_parameters().items():
            param_type = param_config.get("type", "").lower()
            param_types[param_name] = param_type
            
            if param_type in ["continuous", "integer"]:
                param_ranges[param_name] = {
                    "min": param_config.get("min", 0),
                    "max": param_config.get("max", 1)
                }
            elif param_type == "categorical":
                param_ranges[param_name] = {
                    "categories": param_config.get("categories", [])
                }
        
        # 只为连续参数创建网格（最多2个）
        continuous_params = [name for name, type_ in param_types.items() if type_ in ["continuous", "integer"]]
        if len(continuous_params) > 2:
            continuous_params = continuous_params[:2]  # 只取前两个连续参数
        
        # 无连续参数时无法创建网格
        if not continuous_params:
            raise HTTPException(status_code=400, detail="参数空间中没有连续参数，无法创建不确定性地图")
        
        # 创建网格
        grid_points = []
        parameter_grids = {}
        
        if len(continuous_params) == 1:
            # 一维网格
            param_name = continuous_params[0]
            param_min = param_ranges[param_name]["min"]
            param_max = param_ranges[param_name]["max"]
            
            if param_types[param_name] == "integer":
                grid = np.linspace(param_min, param_max, min(resolution, param_max - param_min + 1), dtype=int)
            else:
                grid = np.linspace(param_min, param_max, resolution)
            
            parameter_grids[param_name] = grid.tolist()
            
            # 创建参数点
            for val in grid:
                point = {}
                for name in param_types:
                    if name == param_name:
                        point[name] = int(val) if param_types[name] == "integer" else float(val)
                    elif param_types[name] == "categorical":
                        point[name] = param_ranges[name]["categories"][0]  # 默认使用第一个类别
                    else:
                        point[name] = param_ranges[name]["min"]  # 默认使用最小值
                
                grid_points.append(point)
        
        elif len(continuous_params) == 2:
            # 二维网格
            param1, param2 = continuous_params
            
            if param_types[param1] == "integer":
                grid1 = np.linspace(param_ranges[param1]["min"], param_ranges[param1]["max"], 
                                    min(resolution, param_ranges[param1]["max"] - param_ranges[param1]["min"] + 1), dtype=int)
            else:
                grid1 = np.linspace(param_ranges[param1]["min"], param_ranges[param1]["max"], resolution)
            
            if param_types[param2] == "integer":
                grid2 = np.linspace(param_ranges[param2]["min"], param_ranges[param2]["max"], 
                                    min(resolution, param_ranges[param2]["max"] - param_ranges[param2]["min"] + 1), dtype=int)
            else:
                grid2 = np.linspace(param_ranges[param2]["min"], param_ranges[param2]["max"], resolution)
            
            parameter_grids[param1] = grid1.tolist()
            parameter_grids[param2] = grid2.tolist()
            
            # 创建参数点
            for val1 in grid1:
                for val2 in grid2:
                    point = {}
                    for name in param_types:
                        if name == param1:
                            point[name] = int(val1) if param_types[name] == "integer" else float(val1)
                        elif name == param2:
                            point[name] = int(val2) if param_types[name] == "integer" else float(val2)
                        elif param_types[name] == "categorical":
                            point[name] = param_ranges[name]["categories"][0]  # 默认使用第一个类别
                        else:
                            point[name] = param_ranges[name]["min"]  # 默认使用最小值
                    
                    grid_points.append(point)
        
        # 预测网格点的不确定性
        uncertainty_maps = {}
        mean_maps = {}
        
        # 对每个点进行预测
        X_grid = []
        for point in grid_points:
            X_grid.append(experiment.parameter_space.transform(point))
        
        X_grid = np.array(X_grid)
        
        # 批量预测
        means, stds = experiment.surrogate_model.predict(X_grid, return_std=True)
        
        # 重塑预测结果到网格形状
        for obj_idx, obj in enumerate(experiment.parameter_space.objectives):
            obj_name = obj.get("name")
            
            if len(continuous_params) == 1:
                uncertainty_maps[obj_name] = stds.tolist()
                mean_maps[obj_name] = means.tolist()
            else:  # 2D
                grid_shape = (len(parameter_grids[continuous_params[0]]), len(parameter_grids[continuous_params[1]]))
                uncertainty_maps[obj_name] = stds.reshape(grid_shape).tolist()
                mean_maps[obj_name] = means.reshape(grid_shape).tolist()
        
        return {
            "experiment_id": experiment_id,
            "parameter_grids": parameter_grids,
            "uncertainty_maps": uncertainty_maps,
            "mean_maps": mean_maps,
            "grid_parameters": continuous_params,
            "message": "不确定性分析完成"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"不确定性分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"不确定性分析失败: {str(e)}")

# 新增API端点 - 任务状态和管理

@router.get("/tasks/{experiment_id}/status", response_model=Dict[str, Any])
async def get_task_status(experiment_id: str):
    """获取实验任务的当前状态"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 从实验中提取状态信息
        status = "running"  # 默认状态
        
        # 确定实验的实际状态
        if experiment.iteration == 0 and not experiment.observations:
            status = "initialized"
        elif experiment.iteration >= 20:  # 假设20次迭代为完成条件
            status = "completed"
        
        # 创建状态响应
        status_response = {
            "experiment_id": experiment_id,
            "status": status,
            "current_iteration": experiment.iteration,
            "total_observations": len(experiment.observations),
            "last_updated": experiment.observations[-1]["timestamp"] if experiment.observations else None,
            "best_result": None
        }
        
        # 添加最佳结果（如果有）
        if experiment.best_observation:
            status_response["best_result"] = {
                "parameters": experiment.best_observation["design"],
                "objectives": experiment.best_observation["objectives"],
                "iteration": experiment.best_observation["iteration"]
            }
        
        return status_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")

@router.get("/tasks/{experiment_id}/export", response_model=None)
async def export_task_data(experiment_id: str, format: str = Query("json", regex="^(json|csv|excel)$")):
    """导出实验任务的完整数据"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 准备导出数据
        export_data = {
            "experiment_id": experiment_id,
            "name": os.path.basename(experiment.output_dir),
            "parameter_space": experiment.parameter_space.to_dict(),
            "observations": experiment.observations,
            "best_observation": experiment.best_observation,
            "iteration": experiment.iteration,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 根据格式导出
        if format == "json":
            # 返回JSON格式
            return export_data
        
        elif format == "csv":
            # 将观测数据转换为CSV格式
            observations = experiment.observations
            
            if not observations:
                raise HTTPException(status_code=404, detail="没有观测数据可导出")
            
            # 提取所有参数和目标名称
            param_names = list(observations[0]["design"].keys())
            obj_names = list(observations[0]["objectives"].keys())
            
            # 创建CSV内容
            csv_rows = []
            csv_header = ["iteration", "id", "timestamp", "constraints_satisfied"] + param_names + obj_names
            
            for obs in observations:
                row = [
                    obs["iteration"],
                    obs["id"],
                    obs["timestamp"],
                    obs["constraints_satisfied"]
                ]
                
                # 添加参数值
                for param in param_names:
                    row.append(obs["design"].get(param, ""))
                
                # 添加目标值
                for obj in obj_names:
                    row.append(obs["objectives"].get(obj, ""))
                
                csv_rows.append(row)
            
            # 返回CSV格式
            csv_content = ",".join(csv_header) + "\n"
            for row in csv_rows:
                csv_content += ",".join(str(x) for x in row) + "\n"
            
            # 自定义响应，返回CSV文件
            from fastapi.responses import Response
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={experiment_id}_export.csv"
                }
            )
        
        elif format == "excel":
            # Excel导出需要pandas
            try:
                import pandas as pd
                from io import BytesIO
                from fastapi.responses import Response
                
                # 将观测数据转换为DataFrame
                observations = experiment.observations
                
                if not observations:
                    raise HTTPException(status_code=404, detail="没有观测数据可导出")
                
                # 展平观测数据
                flat_data = []
                for obs in observations:
                    row = {
                        "iteration": obs["iteration"],
                        "id": obs["id"],
                        "timestamp": obs["timestamp"],
                        "constraints_satisfied": obs["constraints_satisfied"]
                    }
                    
                    # 添加参数
                    for param_name, param_value in obs["design"].items():
                        row[f"param_{param_name}"] = param_value
                    
                    # 添加目标
                    for obj_name, obj_value in obs["objectives"].items():
                        row[f"objective_{obj_name}"] = obj_value
                    
                    flat_data.append(row)
                
                # 创建DataFrame
                df = pd.DataFrame(flat_data)
                
                # 创建Excel文件
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, sheet_name="Observations", index=False)
                    
                    # 创建参数空间表格
                    param_df = pd.DataFrame([{
                        "name": name, 
                        "type": config.get("type", ""),
                        "min": config.get("min", ""),
                        "max": config.get("max", ""),
                        "categories": str(config.get("categories", ""))
                    } for name, config in experiment.parameter_space.get_parameters().items()])
                    
                    param_df.to_excel(writer, sheet_name="Parameters", index=False)
                
                excel_buffer.seek(0)
                
                # 返回Excel文件
                return Response(
                    content=excel_buffer.getvalue(),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={
                        "Content-Disposition": f"attachment; filename={experiment_id}_export.xlsx"
                    }
                )
            
            except ImportError:
                raise HTTPException(status_code=400, detail="Excel导出功能需要安装pandas和xlsxwriter库")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出任务数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"导出任务数据失败: {str(e)}")

@router.post("/tasks/{experiment_id}/restart", response_model=Dict[str, Any])
async def restart_task(experiment_id: str, request: TaskRestartRequest, background_tasks: BackgroundTasks):
    """重启优化任务"""
    try:
        # 检查实验是否存在
        experiment = get_experiment(experiment_id)
        
        if request.strategy == "reset":
            # 完全重置实验
            # 保留参数空间和输出目录
            parameter_space = experiment.parameter_space
            output_dir = experiment.output_dir
            seed = experiment.seed
            acquisition_function = experiment.acquisition_function
            exploration_weight = experiment.exploration_weight
            
            # 创建新的BOSystem实例
            new_experiment = BOSystem(
                parameter_space=parameter_space,
                output_dir=output_dir,
                seed=seed,
                acquisition_function=acquisition_function,
                exploration_weight=exploration_weight
            )
            
            # 替换实验
            experiments[experiment_id] = new_experiment
            
        else:  # "reuse_last"
            # 重用现有实验，但重置迭代计数
            if not request.preserve_history:
                # 清除观测历史，但保留其他设置
                experiment.observations = []
                experiment.best_observation = None
            
            # 重置迭代计数
            experiment.iteration = 0
        
        # 异步保存实验状态
        background_tasks.add_task(save_experiment_state, experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "status": "restarted",
            "strategy": request.strategy,
            "preserve_history": request.preserve_history,
            "message": "实验任务已重启"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重启任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重启任务失败: {str(e)}")

# 新增API端点 - 诊断接口

@router.get("/diagnostics/{experiment_id}", response_model=Dict[str, Any])
async def get_diagnostics(experiment_id: str):
    """获取实验任务的诊断信息"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 收集诊断信息
        diagnostics = {
            "experiment_id": experiment_id,
            "parameter_space": "valid" if experiment.parameter_space else "invalid",
            "model_trained": hasattr(experiment, "surrogate_model") and experiment.surrogate_model is not None,
            "recent_exception": None,  # 可以在其他地方补充异常记录
            "observations_count": len(experiment.observations),
            "pending_experiments": [],  # 可在需要时补充
            "last_recommendation_time": None
        }
        
        # 获取最后推荐的时间
        if experiment.observations:
            last_obs = sorted(experiment.observations, key=lambda x: x["timestamp"], reverse=True)[0]
            diagnostics["last_recommendation_time"] = last_obs["timestamp"]
        
        # 检查参数空间有效性
        valid, msg = experiment.parameter_space.validate()
        if not valid:
            diagnostics["parameter_space"] = f"invalid: {msg}"
        
        # 添加内存使用等系统信息
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            
            diagnostics["system_info"] = {
                "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(interval=0.1),
                "threads_count": process.num_threads()
            }
        except ImportError:
            diagnostics["system_info"] = "psutil library not available"
        
        return diagnostics
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取诊断信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取诊断信息失败: {str(e)}")

@router.get("/diagnostics/{experiment_id}/convergence", response_model=Dict[str, Any])
async def get_convergence_diagnostics(experiment_id: str):
    """获取实验任务的收敛诊断信息"""
    try:
        experiment = get_experiment(experiment_id)
        
        # 检查是否有足够的观测数据
        if len(experiment.observations) < 2:
            raise HTTPException(status_code=400, detail="观测数据不足，无法进行收敛分析")
        
        # 提取迭代历史
        iterations = []
        objective_history = {}
        acquisition_values = []
        
        # 收集历史数据
        for obs in sorted(experiment.observations, key=lambda x: x["iteration"]):
            iterations.append(obs["iteration"])
            
            # 收集每个目标的历史值
            for obj_name, obj_value in obs["objectives"].items():
                if obj_name not in objective_history:
                    objective_history[obj_name] = []
                objective_history[obj_name].append(obj_value)
        
        # 创建响应
        convergence_data = {
            "experiment_id": experiment_id,
            "iteration_history": iterations,
            "objective_history": objective_history,
            "message": "收敛诊断数据获取成功"
        }
        
        # 如果可用，添加采集函数值历史
        if acquisition_values:
            convergence_data["acquisition_value_history"] = acquisition_values
        
        return convergence_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取收敛诊断信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取收敛诊断信息失败: {str(e)}")

# 新增WebSocket通知模块

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
import asyncio
from datetime import datetime

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        # 存储活跃连接：experiment_id -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, experiment_id: str):
        await websocket.accept()
        
        if experiment_id not in self.active_connections:
            self.active_connections[experiment_id] = set()
        
        self.active_connections[experiment_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, experiment_id: str):
        if experiment_id in self.active_connections:
            self.active_connections[experiment_id].discard(websocket)
            
            # 如果没有活跃连接，清理键
            if not self.active_connections[experiment_id]:
                del self.active_connections[experiment_id]
    
    async def send_json(self, message: dict, experiment_id: str):
        if experiment_id not in self.active_connections:
            return
        
        disconnected = set()
        for connection in self.active_connections[experiment_id]:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                disconnected.add(connection)
        
        # 清理断开的连接
        for conn in disconnected:
            self.disconnect(conn, experiment_id)
    
    async def broadcast(self, message: dict):
        """向所有连接发送消息"""
        all_experiment_ids = list(self.active_connections.keys())
        
        for experiment_id in all_experiment_ids:
            await self.send_json(message, experiment_id)

# 创建连接管理器实例
manager = ConnectionManager()

# WebSocket路由
@router.websocket("/ws/tasks/{experiment_id}")
async def websocket_endpoint(websocket: WebSocket, experiment_id: str):
    """为特定实验建立WebSocket连接，接收实时更新"""
    try:
        # 尝试获取实验，确认它存在
        get_experiment(experiment_id)
        
        await manager.connect(websocket, experiment_id)
        
        # 发送初始连接成功消息
        await websocket.send_json({
            "event_type": "connection_established",
            "task_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "message": "WebSocket连接已建立"
        })
        
        try:
            # 保持连接，等待客户端消息
            while True:
                # 等待客户端消息，但我们主要是用来推送通知，不一定需要处理客户端消息
                data = await websocket.receive_text()
                
                # 简单的echo响应
                await websocket.send_json({
                    "event_type": "message_received",
                    "task_id": experiment_id,
                    "timestamp": datetime.now().isoformat(),
                    "data": data
                })
        
        except WebSocketDisconnect:
            # 客户端断开连接
            manager.disconnect(websocket, experiment_id)
    
    except HTTPException as e:
        # 实验不存在或其他错误
        await websocket.accept()
        await websocket.send_json({
            "event_type": "error",
            "error": str(e.detail),
            "code": e.status_code,
            "timestamp": datetime.now().isoformat()
        })
        await websocket.close()
    
    except Exception as e:
        # 其他错误
        logger.error(f"WebSocket连接错误: {str(e)}")
        try:
            await websocket.accept()
            await websocket.send_json({
                "event_type": "error",
                "error": "内部服务器错误",
                "timestamp": datetime.now().isoformat()
            })
            await websocket.close()
        except:
            pass 
