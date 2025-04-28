# test_bo_integration.py

import os
import json
import subprocess
import time
import argparse

def run_command(command):
    """运行命令并返回输出"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return result.stdout

def test_end_to_end(config_file, api_url, iterations=3):
    """测试端到端流程"""
    print("=== Starting End-to-End Test ===")
    
    # 创建输出目录
    os.makedirs("test_output", exist_ok=True)
    
    # 1. 创建优化任务
    print("\n1. Creating optimization task...")
    create_output = run_command(f"python catalyst_bo_integration.py create {config_file}")
    if not create_output:
        print("Failed to create task")
        return False
    
    create_result = json.loads(create_output)
    task_id = create_result.get("task_id")
    if not task_id:
        print("No task ID returned")
        return False
    
    print(f"Task created with ID: {task_id}")
    
    # 2. 获取初始设计点
    print("\n2. Getting initial design points...")
    initial_output = run_command(
        f"python catalyst_bo_integration.py initial --task-id {task_id} --samples 5 --output test_output/initial_designs.json"
    )
    if not initial_output:
        print("Failed to get initial designs")
        return False
    
    print("Initial designs saved to test_output/initial_designs.json")
    
    # 3. 运行多次迭代
    for i in range(iterations):
        print(f"\n=== Iteration {i+1}/{iterations} ===")
        
        # 3.1 获取下一个设计点
        print(f"3.1 Getting next design point...")
        next_output = run_command(
            f"python catalyst_bo_integration.py next --task-id {task_id} --batch-size 1 --output test_output/next_design_{i+1}.json"
        )
        if not next_output:
            print(f"Failed to get next design for iteration {i+1}")
            continue
        
        print(f"Next design saved to test_output/next_design_{i+1}.json")
        
        # 3.2 模拟实验
        print(f"3.2 Simulating experiment...")
        sim_output = run_command(
            f"python experiment_simulator.py --design test_output/next_design_{i+1}.json --output test_output/experiment_data_{i+1}.json"
        )
        if not sim_output:
            print(f"Failed to simulate experiment for iteration {i+1}")
            continue
        
        print(f"Experiment data saved to test_output/experiment_data_{i+1}.json")
        
        # 3.3 提交结果
        print(f"3.3 Submitting results...")
        # 读取设计ID
        with open(f"test_output/next_design_{i+1}.json", 'r') as f:
            design_data = json.load(f)
            design_id = design_data["designs"][0]["id"]
        
        submit_output = run_command(
            f"python catalyst_bo_integration.py submit --task-id {task_id} --design-id {design_id} --experiment-file test_output/experiment_data_{i+1}.json"
        )
        if not submit_output:
            print(f"Failed to submit results for iteration {i+1}")
            continue
        
        print(f"Results submitted for iteration {i+1}")
        
        # 3.4 检查任务状态
        print(f"3.4 Checking task status...")
        status_output = run_command(
            f"python catalyst_bo_integration.py status --task-id {task_id}"
        )
        if not status_output:
            print(f"Failed to get task status for iteration {i+1}")
            continue
        
        print(f"Task status: {status_output}")
        
        # 等待一段时间
        time.sleep(2)
    
    print("\n=== End-to-End Test Completed ===")
    return True

def main():
    parser = argparse.ArgumentParser(description="Test BO Integration")
    parser.add_argument("--config", default="catalyst_optimization.json", help="Configuration file")
    parser.add_argument("--api-url", default="http://localhost:8005/api", help="BO API URL")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations to run")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["BO_API_URL"] = args.api_url
    
    # 运行测试
    test_end_to_end(args.config, args.api_url, args.iterations)
    
    return 0

if __name__ == "__main__":
    exit(main())