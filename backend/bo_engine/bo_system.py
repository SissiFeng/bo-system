import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
import os
import json

from .parameter_space import ParameterSpace
from .design_generator import BasicDesignGenerator, DesignType
from .utils import ensure_directory_exists, generate_unique_id, save_to_json, load_from_json
from .models.base_model import BaseModel
from .models.gp_model import GaussianProcessModel
from .acquisition.ei_numpy import ExpectedImprovement
from .acquisition.pi_numpy import ProbabilityImprovement
from .acquisition.ucb_numpy import UpperConfidenceBound

# 设置日志记录器
logger = logging.getLogger("bo_engine.bo_system")

class AcquisitionFunction(str, Enum):
    """采集函数类型枚举"""
    EI = "expected_improvement"  # 期望改进
    PI = "probability_improvement"  # 改进概率
    UCB = "upper_confidence_bound"  # 置信上界
    LCB = "lower_confidence_bound"  # 置信下界
    RANDOM = "random"  # 随机采样（作为 fallback）

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
        self.design_generator = BasicDesignGenerator(
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
        # 如果没有足够的观测数据，不能构建代理模型，使用随机采样
        if len(self.observations) < 2:
            logger.warning("观测数据不足，无法构建代理模型，使用随机采样")
            return self.design_generator.generate_random_designs(n_designs)
        
        # 使用指定的采集函数或默认采集函数
        acq_func = acquisition_function or self.acquisition_function
        exp_weight = exploration_weight or self.exploration_weight
        
        # 更新代理模型
        model_updated = self._update_surrogate_model()
        
        # 如果模型更新失败，使用随机采样作为 fallback
        if not model_updated or self.surrogate_model is None:
            logger.warning("代理模型更新失败或未初始化，使用随机采样作为 fallback")
            return self.design_generator.generate_random_designs(n_designs)
        
        try:
            # 获取最佳观测值（用于 EI 和 PI 采集函数）
            best_f = None
            if self.best_observation is not None:
                objective_name = list(self.best_observation["objectives"].keys())[0]
                best_f = self.best_observation["objectives"][objective_name]
            
            # 根据请求的设计方案数量决定策略
            if n_designs == 1:
                # 单点推荐：使用采集函数优化
                next_point_internal, _ = self._optimize_acquisition(acq_func, exp_weight, best_f)
                
                # 将内部表示转换回原始参数空间
                next_point = self.parameter_space.inverse_transform(next_point_internal)
                return [next_point]
            else:
                # 多点推荐：简单方法是生成候选集，根据采集函数值排序选择前 n_designs 个
                # 更复杂的方法可以使用批量贝叶斯优化技术，如 qEI, DPP
                n_candidates = max(100, n_designs * 10)
                candidates_internal = np.random.uniform(0, 1, size=(n_candidates, self.parameter_space.get_internal_dimensions()))
                
                # 计算每个候选点的采集函数值
                acq_values = []
                for i in range(len(candidates_internal)):
                    candidate = candidates_internal[i].reshape(1, -1)
                    acq_value = self._compute_acquisition_value(candidate, acq_func, exp_weight, best_f)
                    acq_values.append(acq_value)
                
                # 选择采集函数值最高的点
                indices = np.argsort(-np.array(acq_values))[:n_designs]
                selected_points_internal = candidates_internal[indices]
                
                # 将内部表示转换回原始参数空间
                selected_designs = []
                for point_internal in selected_points_internal:
                    point = self.parameter_space.inverse_transform(point_internal)
                    if self.parameter_space.check_constraints(point):
                        selected_designs.append(point)
                
                # 如果有约束导致某些点被过滤，补充随机点
                if len(selected_designs) < n_designs:
                    logger.warning(f"采集函数推荐的点部分不满足约束，补充随机点")
                    random_designs = self.design_generator.generate_random_designs(n_designs - len(selected_designs))
                    selected_designs.extend(random_designs)
                
                return selected_designs
        except Exception as e:
            logger.error(f"采集函数优化过程出错，使用随机采样作为 fallback: {str(e)}", exc_info=True)
            return self.design_generator.generate_random_designs(n_designs)
        
        # 增加迭代计数
        self.iteration += 1
    
    def _update_surrogate_model(self) -> bool:
        """
        更新代理模型
        
        Returns:
            bool: 模型是否成功更新
        """
        logger.info("更新代理模型")
        
        try:
            if self.surrogate_model is None:
                # 初始化代理模型（这里使用高斯过程，实际应用中可根据需求选择不同模型）
                self.surrogate_model = GaussianProcessModel()
            
            # 准备训练数据
            X = []
            y = []
            for obs in self.observations:
                # 只考虑满足约束的观测
                if obs["constraints_satisfied"]:
                    # 将设计点转换为内部表示（标准化到 [0, 1] 空间）
                    X_point = self.parameter_space.transform(obs["design"])
                    X.append(X_point)
                    
                    # 简单起见，只考虑第一个目标
                    objective_name = list(obs["objectives"].keys())[0]
                    y.append(obs["objectives"][objective_name])
            
            if len(X) < 2:
                logger.warning("有效观测数据不足，无法训练代理模型")
                return False
            
            # 转换为数组
            X = np.array(X)
            y = np.array(y)
            
            # 训练模型
            self.surrogate_model.fit(X, y)
            
            return self.surrogate_model.is_trained()
        except Exception as e:
            logger.error(f"更新代理模型失败: {str(e)}", exc_info=True)
            return False
    
    def _optimize_acquisition(
        self,
        acquisition_function: AcquisitionFunction,
        exploration_weight: float,
        best_f: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        优化采集函数，寻找下一个采样点
        
        Args:
            acquisition_function: 采集函数类型
            exploration_weight: 探索权重（用于 UCB/LCB）
            best_f: 当前最佳目标函数值（用于 EI/PI）
            
        Returns:
            Tuple[np.ndarray, float]: (最优点的内部表示, 采集函数值)
        """
        # 确定优化方向（默认为最小化）
        maximize = False
        if self.parameter_space.objectives:
            objective_direction = self.parameter_space.objectives[0].get('type', 'minimize').lower()
            maximize = (objective_direction == 'maximize')
        
        # 根据采集函数类型创建相应的采集函数实例
        acq_instance = None
        
        if acquisition_function == AcquisitionFunction.EI:
            acq_instance = ExpectedImprovement(
                model=self.surrogate_model,
                parameter_space=self.parameter_space,
                xi=0.01,
                maximize=maximize,
                best_f=best_f
            )
        elif acquisition_function == AcquisitionFunction.PI:
            acq_instance = ProbabilityImprovement(
                model=self.surrogate_model,
                parameter_space=self.parameter_space,
                xi=0.01,
                maximize=maximize,
                best_f=best_f
            )
        elif acquisition_function == AcquisitionFunction.UCB:
            acq_instance = UpperConfidenceBound(
                model=self.surrogate_model,
                parameter_space=self.parameter_space,
                kappa=exploration_weight,
                maximize=maximize
            )
        elif acquisition_function == AcquisitionFunction.RANDOM or acquisition_function == AcquisitionFunction.LCB:
            # 对于 RANDOM 或尚未实现的 LCB，返回随机点
            dim = self.parameter_space.get_internal_dimensions()
            random_point = np.random.uniform(0, 1, size=dim)
            return random_point.reshape(1, -1), 0.0
        else:
            # 默认使用 EI
            logger.warning(f"未知的采集函数类型 {acquisition_function}，使用 EI 作为默认")
            acq_instance = ExpectedImprovement(
                model=self.surrogate_model,
                parameter_space=self.parameter_space,
                xi=0.01,
                maximize=maximize,
                best_f=best_f
            )
        
        # 通过采集函数的 optimize 方法找到最优点
        x_best, acq_value = acq_instance.optimize(n_restarts=5, verbose=False)
        
        return x_best.reshape(1, -1), acq_value
    
    def _compute_acquisition_value(
        self,
        X: np.ndarray,
        acquisition_function: AcquisitionFunction,
        exploration_weight: float,
        best_f: Optional[float] = None
    ) -> float:
        """
        计算给定设计方案的采集函数值
        
        Args:
            X: 设计方案的内部表示 ([0, 1] 空间)，形状为 (1, n_dims)
            acquisition_function: 采集函数类型
            exploration_weight: 探索权重
            best_f: 当前最佳目标函数值
            
        Returns:
            float: 采集函数值
        """
        # 确定优化方向（默认为最小化）
        maximize = False
        if self.parameter_space.objectives:
            objective_direction = self.parameter_space.objectives[0].get('type', 'minimize').lower()
            maximize = (objective_direction == 'maximize')
        
        try:
            # 根据采集函数类型计算采集函数值
            if acquisition_function == AcquisitionFunction.EI:
                if best_f is None:
                    logger.warning("EI 采集函数需要 best_f 值，但未提供")
                    return 0.0
                
                ei = ExpectedImprovement(
                    model=self.surrogate_model,
                    parameter_space=self.parameter_space,
                    xi=0.01,
                    maximize=maximize,
                    best_f=best_f
                )
                return -ei.evaluate(X)[0]  # 返回负值，使得较大的值被优先选择
            
            elif acquisition_function == AcquisitionFunction.PI:
                if best_f is None:
                    logger.warning("PI 采集函数需要 best_f 值，但未提供")
                    return 0.0
                
                pi = ProbabilityImprovement(
                    model=self.surrogate_model,
                    parameter_space=self.parameter_space,
                    xi=0.01,
                    maximize=maximize,
                    best_f=best_f
                )
                return -pi.evaluate(X)[0]  # 返回负值，使得较大的值被优先选择
            
            elif acquisition_function == AcquisitionFunction.UCB:
                ucb = UpperConfidenceBound(
                    model=self.surrogate_model,
                    parameter_space=self.parameter_space,
                    kappa=exploration_weight,
                    maximize=maximize
                )
                return -ucb.evaluate(X)[0]  # 返回负值，使得较小的值（更好的 UCB）被优先选择
            
            elif acquisition_function == AcquisitionFunction.LCB:
                # LCB 可以通过调整 UCB 的 maximize 参数实现
                lcb = UpperConfidenceBound(
                    model=self.surrogate_model,
                    parameter_space=self.parameter_space,
                    kappa=exploration_weight,
                    maximize=not maximize  # 反转优化方向
                )
                return -lcb.evaluate(X)[0]
            
            elif acquisition_function == AcquisitionFunction.RANDOM:
                # 随机采集函数返回随机值
                return np.random.rand()
            
            else:
                logger.warning(f"未知的采集函数类型 {acquisition_function}，使用随机值")
                return np.random.rand()
        
        except Exception as e:
            logger.error(f"计算采集函数值时出错: {str(e)}", exc_info=True)
            return np.random.rand()  # 出错时返回随机值作为 fallback
    
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
