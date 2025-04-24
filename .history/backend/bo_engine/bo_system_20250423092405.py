import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
import os
import json

from .parameter_space import ParameterSpace
from .design_generator import DesignGenerator, DesignType
from .utils import ensure_directory_exists, generate_unique_id, save_to_json, load_from_json

# 设置日志记录器
logger = logging.getLogger("bo_engine.bo_system")

class AcquisitionFunction(str, Enum):
    """采集函数类型枚举"""
    EI = "expected_improvement"  # 期望改进
    PI = "probability_improvement"  # 改进概率
    UCB = "upper_confidence_bound"  # 置信上界
    LCB = "lower_confidence_bound"  # 置信下界

class BOSystem:
    """贝叶斯优化系统类，整合参数空间、设计生成器和优化算法"""
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        output_dir: str = "bo_results",
        seed: Optional[int] = None,
        acquisition_function: AcquisitionFunction = AcquisitionFunction.EI,
        exploration_weight: float = 0.1
    ):
        """
        初始化贝叶斯优化系统
        
        Args:
            parameter_space: 参数空间对象
            output_dir: 输出目录路径
            seed: 随机种子，用于结果可重现性
            acquisition_function: 采集函数类型
            exploration_weight: 探索权重参数（用于UCB/LCB）
        """
        self.parameter_space = parameter_space
        self.output_dir = output_dir
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.acquisition_function = acquisition_function
        self.exploration_weight = exploration_weight
        
        # 初始化设计生成器
        self.design_generator = DesignGenerator(
            parameter_space=parameter_space,
            seed=seed,
            output_dir=os.path.join(output_dir, "designs")
        )
        
        # 确保输出目录存在
        ensure_directory_exists(output_dir)
        ensure_directory_exists(os.path.join(output_dir, "models"))
        ensure_directory_exists(os.path.join(output_dir, "designs"))
        ensure_directory_exists(os.path.join(output_dir, "evaluations"))
        
        # 初始化优化状态
        self.observations = []  # 观测结果列表
        self.best_observation = None  # 当前最佳观测
        self.iteration = 0  # 当前迭代次数
        self.surrogate_model = None  # 代理模型（将在需要时初始化）
        
        # 系统ID，用于保存/加载
        self.system_id = generate_unique_id()
        
        logger.info(f"初始化贝叶斯优化系统，参数空间维度: {parameter_space.get_dimensions()}")
    
    def initialize_design(
        self,
        n_initial_designs: int = 10,
        design_type: DesignType = DesignType.LATIN_HYPERCUBE
    ) -> List[Dict[str, Any]]:
        """
        生成初始设计方案
        
        Args:
            n_initial_designs: 初始设计方案数量
            design_type: 设计方案生成类型
            
        Returns:
            List[Dict[str, Any]]: 生成的初始设计方案列表
        """
        logger.info(f"生成{n_initial_designs}个初始设计方案，类型: {design_type}")
        
        if design_type == DesignType.LATIN_HYPERCUBE:
            designs = self.design_generator.generate_lhs_designs(
                n_designs=n_initial_designs,
                ensure_constraints=True
            )
        elif design_type == DesignType.RANDOM:
            designs = self.design_generator.generate_random_designs(
                n_designs=n_initial_designs,
                ensure_constraints=True
            )
        elif design_type == DesignType.FACTORIAL:
            designs = self.design_generator.generate_grid_designs(
                n_divisions=max(2, int(n_initial_designs ** (1 / self.parameter_space.get_dimensions()))),
                ensure_constraints=True
            )
            # 如果网格采样产生的设计方案过多，随机选择子集
            if len(designs) > n_initial_designs:
                self.rng.shuffle(designs)
                designs = designs[:n_initial_designs]
        else:
            raise ValueError(f"不支持的初始设计类型: {design_type}")
        
        logger.info(f"成功生成{len(designs)}个初始设计方案")
        return designs
    
    def add_observation(
        self,
        design: Dict[str, Any],
        objectives: Dict[str, float],
        constraints_satisfied: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        添加观测结果
        
        Args:
            design: 设计方案参数
            objectives: 目标函数值字典
            constraints_satisfied: 是否满足约束条件
            metadata: 其他元数据信息
            
        Returns:
            Dict[str, Any]: 添加的观测记录
        """
        # 验证设计方案是否有效
        valid, msg = self.parameter_space.validate_point(design)
        if not valid:
            raise ValueError(f"无效的设计方案: {msg}")
        
        # 创建观测记录
        observation = {
            "id": generate_unique_id(),
            "iteration": self.iteration,
            "design": design,
            "objectives": objectives,
            "constraints_satisfied": constraints_satisfied,
            "metadata": metadata or {},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 添加到观测列表
        self.observations.append(observation)
        
        # 更新最佳观测
        if self._is_better_observation(observation, self.best_observation):
            self.best_observation = observation
            logger.info(f"发现新的最佳设计方案，目标值: {objectives}")
        
        # 保存观测记录
        self._save_observation(observation)
        
        return observation
    
    def _is_better_observation(
        self,
        observation: Dict[str, Any],
        current_best: Optional[Dict[str, Any]]
    ) -> bool:
        """
        判断是否为更好的观测结果
        
        Args:
            observation: 新的观测记录
            current_best: 当前最佳观测记录
            
        Returns:
            bool: 如果新的观测更好则返回True
        """
        # 如果当前没有最佳观测或新观测满足约束而当前最佳不满足约束
        if current_best is None:
            return True
        if observation["constraints_satisfied"] and not current_best["constraints_satisfied"]:
            return True
        if not observation["constraints_satisfied"] and current_best["constraints_satisfied"]:
            return False
        
        # 比较目标函数值（假设是单目标优化）
        # 如果是多目标优化，这里需要考虑帕累托前沿
        objective_name = list(observation["objectives"].keys())[0]
        objective_value = observation["objectives"][objective_name]
        best_objective_value = current_best["objectives"][objective_name]
        
        # 假设是最小化问题（如果是最大化问题，需要修改比较逻辑）
        return objective_value < best_objective_value
    
    def _save_observation(self, observation: Dict[str, Any]) -> None:
        """
        保存观测记录到文件
        
        Args:
            observation: 观测记录
        """
        filepath = os.path.join(
            self.output_dir, 
            "evaluations", 
            f"observation_{observation['id']}.json"
        )
        save_to_json(observation, filepath)
    
    def get_next_design(
        self,
        n_designs: int = 1,
        acquisition_function: Optional[AcquisitionFunction] = None,
        exploration_weight: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        基于当前观测，推荐下一批设计方案
        
        Args:
            n_designs: 需要生成的设计方案数量
            acquisition_function: 可选，覆盖默认采集函数
            exploration_weight: 可选，覆盖默认探索权重
            
        Returns:
            List[Dict[str, Any]]: 推荐的设计方案列表
        """
        # 如果没有足够的观测数据，不能构建代理模型
        if len(self.observations) < 2:
            logger.warning("观测数据不足，无法构建代理模型，使用随机采样")
            return self.design_generator.generate_random_designs(n_designs)
        
        # 使用指定的采集函数或默认采集函数
        acq_func = acquisition_function or self.acquisition_function
        exp_weight = exploration_weight or self.exploration_weight
        
        # 更新代理模型
        self._update_surrogate_model()
        
        # 使用采集函数优化采样
        # 这里是简化实现，实际上需要根据代理模型预测和不确定性进行优化
        # 实际实现时，可以使用各种优化器来最大化采集函数
        
        # 生成候选设计
        n_candidates = max(100, n_designs * 10)
        candidates = self.design_generator.generate_random_designs(n_candidates)
        
        # 计算每个候选设计的采集函数值
        acq_values = []
        for design in candidates:
            acq_value = self._compute_acquisition_value(design, acq_func, exp_weight)
            acq_values.append(acq_value)
        
        # 选择采集函数值最高的设计
        indices = np.argsort(-np.array(acq_values))[:n_designs]
        selected_designs = [candidates[i] for i in indices]
        
        # 增加迭代计数
        self.iteration += 1
        
        return selected_designs
    
    def _update_surrogate_model(self) -> None:
        """
        更新代理模型
        
        注意：实际实现时，这里应该使用高斯过程、随机森林等模型
        这个方法是一个简化的占位符
        """
        logger.info("更新代理模型")
        
        # 实际实现时，应该根据观测数据训练代理模型
        # self.surrogate_model = ...
        
        # 这里使用一个占位符，表示模型已更新
        self.surrogate_model = {"updated": True}
    
    def _compute_acquisition_value(
        self,
        design: Dict[str, Any],
        acquisition_function: AcquisitionFunction,
        exploration_weight: float
    ) -> float:
        """
        计算给定设计方案的采集函数值
        
        Args:
            design: 设计方案
            acquisition_function: 采集函数类型
            exploration_weight: 探索权重
            
        Returns:
            float: 采集函数值
        """
        # 注意：这是一个简化的实现
        # 实际实现时，应该根据代理模型的预测和不确定性计算采集函数值
        
        # 对于测试目的，暂时返回一个随机值
        return self.rng.random()
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        获取优化历史记录
        
        Returns:
            List[Dict[str, Any]]: 优化历史记录列表
        """
        return self.observations
    
    def get_best_design(self) -> Optional[Dict[str, Any]]:
        """
        获取当前最佳设计方案
        
        Returns:
            Optional[Dict[str, Any]]: 最佳设计方案，如果没有则返回None
        """
        if self.best_observation is None:
            return None
        return self.best_observation["design"]
    
    def get_best_objectives(self) -> Optional[Dict[str, float]]:
        """
        获取当前最佳目标函数值
        
        Returns:
            Optional[Dict[str, float]]: 最佳目标函数值，如果没有则返回None
        """
        if self.best_observation is None:
            return None
        return self.best_observation["objectives"]
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        保存优化系统状态
        
        Args:
            filepath: 保存文件路径，如果为None则使用默认路径
            
        Returns:
            str: 保存的文件路径
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, f"bo_system_{self.system_id}.json")
        
        # 创建系统状态数据
        state = {
            "system_id": self.system_id,
            "iteration": self.iteration,
            "parameter_space": self.parameter_space.to_dict(),
            "observations": self.observations,
            "best_observation_id": self.best_observation["id"] if self.best_observation else None,
            "acquisition_function": self.acquisition_function.value,
            "exploration_weight": self.exploration_weight,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 保存状态
        save_to_json(state, filepath)
        logger.info(f"保存优化系统状态到: {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'BOSystem':
        """
        从文件加载优化系统状态
        
        Args:
            filepath: 状态文件路径
            
        Returns:
            BOSystem: 加载的优化系统实例
        """
        # 加载状态数据
        state = load_from_json(filepath)
        if state is None:
            raise ValueError(f"无法加载状态文件: {filepath}")
        
        # 重建参数空间
        parameter_space = ParameterSpace.from_dict(state["parameter_space"])
        
        # 创建系统实例
        system = cls(
            parameter_space=parameter_space,
            output_dir=state["output_dir"],
            seed=state["seed"],
            acquisition_function=AcquisitionFunction(state["acquisition_function"]),
            exploration_weight=state["exploration_weight"]
        )
        
        # 恢复系统状态
        system.system_id = state["system_id"]
        system.iteration = state["iteration"]
        system.observations = state["observations"]
        
        # 恢复最佳观测
        if state["best_observation_id"]:
            for obs in system.observations:
                if obs["id"] == state["best_observation_id"]:
                    system.best_observation = obs
                    break
        
        logger.info(f"从 {filepath} 加载优化系统状态")
        return system 
