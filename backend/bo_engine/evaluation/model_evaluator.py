import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    评估模型性能的工具类，用于计算和跟踪各种评估指标。
    
    支持常见的回归模型评估指标，例如均方误差(MSE)、
    均方根误差(RMSE)、平均绝对误差(MAE)、R²分数等。
    同时跟踪推理时间和训练时间。
    """
    
    def __init__(self):
        """初始化模型评估器"""
        self.reset()
        
    def reset(self):
        """重置所有评估指标和时间记录"""
        self.metrics_history = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'r2': [],
            'neg_log_likelihood': [],
            'training_time_ms': [],
            'inference_time_ms': []
        }
        self.latest_metrics = {}
        self.training_start_time = None
        self.training_end_time = None
        
    def start_training_timer(self):
        """开始计时模型训练过程"""
        self.training_start_time = time.time()
        
    def stop_training_timer(self):
        """停止计时模型训练过程并记录耗时"""
        if self.training_start_time is None:
            logger.warning("Training timer was not started before stopping")
            return 0
            
        self.training_end_time = time.time()
        training_time_ms = (self.training_end_time - self.training_start_time) * 1000
        self.metrics_history['training_time_ms'].append(training_time_ms)
        self.latest_metrics['training_time_ms'] = training_time_ms
        
        return training_time_ms
        
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_std: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        评估模型预测性能
        
        Args:
            y_true: 真实目标值
            y_pred: 模型预测均值
            y_std: 模型预测标准差 (可选，仅用于计算负对数似然)
            
        Returns:
            包含各项指标的字典
        """
        inference_start = time.time()
        
        # 计算各种指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 如果有标准差信息，计算负对数似然
        neg_log_likelihood = None
        if y_std is not None and np.all(y_std > 0):
            # 高斯分布的负对数似然
            neg_log_likelihood = np.mean(
                0.5 * np.log(2 * np.pi * y_std**2) + 
                0.5 * ((y_true - y_pred) / y_std)**2
            )
            
        inference_time_ms = (time.time() - inference_start) * 1000
        
        # 更新指标历史
        self.metrics_history['mse'].append(mse)
        self.metrics_history['rmse'].append(rmse)
        self.metrics_history['mae'].append(mae)
        self.metrics_history['r2'].append(r2)
        self.metrics_history['inference_time_ms'].append(inference_time_ms)
        
        if neg_log_likelihood is not None:
            self.metrics_history['neg_log_likelihood'].append(neg_log_likelihood)
            
        # 更新最新指标
        self.latest_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'inference_time_ms': inference_time_ms
        }
        
        if neg_log_likelihood is not None:
            self.latest_metrics['neg_log_likelihood'] = neg_log_likelihood
            
        return self.latest_metrics
        
    def get_latest_metrics(self) -> Dict[str, float]:
        """获取最新的评估指标"""
        return self.latest_metrics
        
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """获取所有评估指标的历史记录"""
        return self.metrics_history
        
    def get_metric_trend(self, metric_name: str) -> List[float]:
        """
        获取特定指标的历史趋势
        
        Args:
            metric_name: 指标名称（例如 'mse', 'rmse', 'r2'等）
            
        Returns:
            指定指标的所有历史值列表
        """
        if metric_name not in self.metrics_history:
            logger.warning(f"Metric '{metric_name}' not found in history")
            return []
            
        return self.metrics_history[metric_name]
        
    def print_summary(self):
        """打印最新的评估指标摘要"""
        if not self.latest_metrics:
            logger.warning("No metrics available to print")
            return
            
        print("\n=== 模型性能评估 ===")
        for metric, value in self.latest_metrics.items():
            if metric == 'training_time_ms' or metric == 'inference_time_ms':
                print(f"{metric}: {value:.2f} ms")
            else:
                print(f"{metric}: {value:.6f}")
        print("=====================\n")
        
    def calculate_cross_validation_metrics(self, cv_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        计算交叉验证的平均指标和标准差
        
        Args:
            cv_results: 包含每次交叉验证折的指标结果的列表
            
        Returns:
            包含每个指标的平均值和标准差的嵌套字典
        """
        if not cv_results:
            logger.warning("No cross-validation results provided")
            return {}
            
        # 提取所有可用指标
        all_metrics = set()
        for result in cv_results:
            all_metrics.update(result.keys())
            
        # 计算每个指标的平均值和标准差
        cv_summary = {}
        for metric in all_metrics:
            # 收集所有折的该指标值
            values = [result.get(metric, np.nan) for result in cv_results]
            values = [v for v in values if not np.isnan(v)]
            
            if not values:
                continue
                
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            cv_summary[metric] = {
                'mean': mean_value,
                'std': std_value
            }
            
        return cv_summary 
