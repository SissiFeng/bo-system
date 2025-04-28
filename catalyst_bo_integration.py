# catalyst_bo_integration.py

import requests
import json
import time
import logging
import os
from datetime import datetime

# 配置
BO_API_URL = os.environ.get("BO_API_URL", "http://localhost:8005/api")
TASK_ID = os.environ.get("TASK_ID", "")  # 需要设置任务ID
LOG_FILE = os.environ.get("LOG_FILE", "catalyst_bo_integration.log")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("catalyst_bo_integration")

class BOIntegrationClient:
    """BO系统集成客户端"""
    
    def __init__(self, api_url, task_id=None):
        self.api_url = api_url
        self.task_id = task_id
        logger.info(f"初始化BO集成客户端: API URL={api_url}, Task ID={task_id}")
    
    def create_task(self, config_file):
        """创建新的优化任务"""
        logger.info(f"创建新的优化任务，配置文件: {config_file}")
        
        try:
            # 读取配置文件
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # 发送请求
            response = requests.post(
                f"{self.api_url}/parameter-space",
                json=config,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # 更新任务ID
            self.task_id = result.get("task_id")
            logger.info(f"任务创建成功，Task ID: {self.task_id}")
            
            return result
        except Exception as e:
            logger.error(f"创建任务失败: {str(e)}")
            raise
    
    def get_initial_designs(self, samples=5):
        """获取初始设计点"""
        if not self.task_id:
            raise ValueError("Task ID未设置")
        
        logger.info(f"获取初始设计点，Task ID: {self.task_id}, Samples: {samples}")
        
        try:
            response = requests.get(
                f"{self.api_url}/designs/{self.task_id}/initial",
                params={"samples": samples},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            designs = result.get("designs", [])
            logger.info(f"获取到{len(designs)}个初始设计点")
            
            return designs
        except Exception as e:
            logger.error(f"获取初始设计点失败: {str(e)}")
            raise
    
    def get_next_designs(self, batch_size=1):
        """获取下一批推荐设计点"""
        if not self.task_id:
            raise ValueError("Task ID未设置")
        
        logger.info(f"获取下一批设计点，Task ID: {self.task_id}, Batch Size: {batch_size}")
        
        try:
            response = requests.get(
                f"{self.api_url}/designs/{self.task_id}/next",
                params={"batch_size": batch_size},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            designs = result.get("designs", [])
            logger.info(f"获取到{len(designs)}个推荐设计点")
            
            return designs
        except Exception as e:
            logger.error(f"获取推荐设计点失败: {str(e)}")
            raise
    
    def submit_results(self, results):
        """提交实验结果"""
        if not self.task_id:
            raise ValueError("Task ID未设置")
        
        logger.info(f"提交实验结果，Task ID: {self.task_id}, Results Count: {len(results)}")
        
        try:
            payload = {"results": results}
            response = requests.post(
                f"{self.api_url}/results/{self.task_id}",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"结果提交成功: {result}")
            return result
        except Exception as e:
            logger.error(f"提交结果失败: {str(e)}")
            raise
    
    def get_task_status(self):
        """获取任务状态"""
        if not self.task_id:
            raise ValueError("Task ID未设置")
        
        logger.info(f"获取任务状态，Task ID: {self.task_id}")
        
        try:
            response = requests.get(
                f"{self.api_url}/tasks/{self.task_id}/status",
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"任务状态: {result}")
            return result
        except Exception as e:
            logger.error(f"获取任务状态失败: {str(e)}")
            raise

# 实验数据处理函数
def process_experiment_data(experiment_file):
    """处理实验数据文件，提取目标函数值"""
    logger.info(f"处理实验数据文件: {experiment_file}")
    
    try:
        # 读取实验数据文件
        with open(experiment_file, 'r') as f:
            data = json.load(f)
        
        # 提取LSV数据
        lsv_data = data.get("lsv_data", [])
        if not lsv_data:
            logger.warning("LSV数据为空")
            return None
        
        # 提取CV数据
        cv_data = data.get("cv_data", [])
        if not cv_data:
            logger.warning("CV数据为空")
            return None
        
        # 计算LSV斜率（示例计算，需要根据实际数据格式调整）
        lsv_slope = calculate_lsv_slope(lsv_data)
        
        # 计算CV稳定性（示例计算，需要根据实际数据格式调整）
        cv_stability = calculate_cv_stability(cv_data)
        
        # 返回目标函数值
        return {
            "LSV_slope": lsv_slope,
            "CV_stability": cv_stability
        }
    except Exception as e:
        logger.error(f"处理实验数据失败: {str(e)}")
        return None

def calculate_lsv_slope(lsv_data):
    """计算LSV斜率（示例函数）"""
    # 这里是简化的计算，实际计算需要根据数据格式调整
    try:
        # 假设lsv_data是电流-电压数据点的列表
        # 例如: [{"voltage": 0.1, "current": 0.5}, {"voltage": 0.2, "current": 1.0}, ...]
        
        # 提取电压和电流数据
        voltages = [point["voltage"] for point in lsv_data]
        currents = [point["current"] for point in lsv_data]
        
        # 计算斜率（简化为线性拟合）
        n = len(voltages)
        if n < 2:
            return 0
        
        # 简单线性回归
        sum_x = sum(voltages)
        sum_y = sum(currents)
        sum_xy = sum(x*y for x, y in zip(voltages, currents))
        sum_xx = sum(x*x for x in voltages)
        
        # 计算斜率
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        return slope
    except Exception as e:
        logger.error(f"计算LSV斜率失败: {str(e)}")
        return 0

def calculate_cv_stability(cv_data):
    """计算CV稳定性（示例函数）"""
    # 这里是简化的计算，实际计算需要根据数据格式调整
    try:
        # 假设cv_data包含多个循环的数据
        # 例如: {"cycle1": [...], "cycle2": [...], ...}
        
        # 获取第一个循环和最后一个循环
        cycles = sorted(cv_data.keys())
        if len(cycles) < 2:
            return 0
        
        first_cycle = cv_data[cycles[0]]
        last_cycle = cv_data[cycles[-1]]
        
        # 计算循环面积
        first_area = calculate_cv_area(first_cycle)
        last_area = calculate_cv_area(last_cycle)
        
        # 计算面积变化百分比
        if first_area == 0:
            return 0
        
        stability = abs(last_area - first_area) / first_area
        return stability
    except Exception as e:
        logger.error(f"计算CV稳定性失败: {str(e)}")
        return 0

def calculate_cv_area(cycle_data):
    """计算CV循环的面积（示例函数）"""
    # 这里是简化的计算，实际计算需要根据数据格式调整
    try:
        # 假设cycle_data是电压-电流数据点的列表
        # 例如: [{"voltage": 0.1, "current": 0.5}, {"voltage": 0.2, "current": 1.0}, ...]
        
        # 使用梯形法则计算面积
        area = 0
        for i in range(len(cycle_data) - 1):
            v1 = cycle_data[i]["voltage"]
            v2 = cycle_data[i+1]["voltage"]
            c1 = cycle_data[i]["current"]
            c2 = cycle_data[i+1]["current"]
            
            area += 0.5 * (v2 - v1) * (c1 + c2)
        
        return abs(area)
    except Exception as e:
        logger.error(f"计算CV面积失败: {str(e)}")
        return 0

# 命令行接口
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Catalyst BO Integration Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # 创建任务命令
    create_parser = subparsers.add_parser("create", help="Create a new optimization task")
    create_parser.add_argument("config", help="Path to the configuration JSON file")
    
    # 获取初始设计点命令
    initial_parser = subparsers.add_parser("initial", help="Get initial design points")
    initial_parser.add_argument("--task-id", help="Task ID (if not set in environment)")
    initial_parser.add_argument("--samples", type=int, default=5, help="Number of samples to generate")
    initial_parser.add_argument("--output", help="Output JSON file for designs")
    
    # 获取下一批设计点命令
    next_parser = subparsers.add_parser("next", help="Get next recommended design points")
    next_parser.add_argument("--task-id", help="Task ID (if not set in environment)")
    next_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    next_parser.add_argument("--output", help="Output JSON file for designs")
    
    # 提交结果命令
    submit_parser = subparsers.add_parser("submit", help="Submit experiment results")
    submit_parser.add_argument("--task-id", help="Task ID (if not set in environment)")
    submit_parser.add_argument("--design-id", required=True, help="Design ID")
    submit_parser.add_argument("--experiment-file", required=True, help="Experiment data JSON file")
    submit_parser.add_argument("--metadata", help="Additional metadata JSON file")
    
    # 获取任务状态命令
    status_parser = subparsers.add_parser("status", help="Get task status")
    status_parser.add_argument("--task-id", help="Task ID (if not set in environment)")
    
    args = parser.parse_args()
    
    # 初始化客户端
    task_id = args.task_id if hasattr(args, "task_id") and args.task_id else TASK_ID
    client = BOIntegrationClient(BO_API_URL, task_id)
    
    # 执行命令
    if args.command == "create":
        result = client.create_task(args.config)
        print(json.dumps(result, indent=2))
    
    elif args.command == "initial":
        designs = client.get_initial_designs(args.samples)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"designs": designs}, f, indent=2)
            print(f"Designs saved to {args.output}")
        else:
            print(json.dumps({"designs": designs}, indent=2))
    
    elif args.command == "next":
        designs = client.get_next_designs(args.batch_size)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"designs": designs}, f, indent=2)
            print(f"Designs saved to {args.output}")
        else:
            print(json.dumps({"designs": designs}, indent=2))
    
    elif args.command == "submit":
        # 处理实验数据
        objectives = process_experiment_data(args.experiment_file)
        if not objectives:
            print("Error: Failed to process experiment data")
            return 1
        
        # 准备元数据
        metadata = {}
        if args.metadata:
            with open(args.metadata, 'r') as f:
                metadata = json.load(f)
        
        # 添加时间戳
        metadata["timestamp"] = datetime.now().isoformat()
        
        # 构建结果对象
        result = {
            "parameters": {},  # 这里需要从设计点或实验文件中获取参数
            "objectives": objectives,
            "metadata": metadata
        }
        
        # 提交结果
        submission_result = client.submit_results([result])
        print(json.dumps(submission_result, indent=2))
    
    elif args.command == "status":
        status = client.get_task_status()
        print(json.dumps(status, indent=2))
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
