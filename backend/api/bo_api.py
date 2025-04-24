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
        
        # 返回设计方案
        return [DesignPoint(parameters=design, id=f"design_{i}") 
                for i, design in enumerate(next_designs)]
    
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
