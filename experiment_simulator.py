# experiment_simulator.py

import json
import random
import math
import argparse
from datetime import datetime
import os

def generate_lsv_data(parameters):
    """生成模拟的LSV数据"""
    # 提取参数
    ni_ratio = parameters.get("Ni_ratio", 0.3)
    co_ratio = parameters.get("Co_ratio", 0.2)
    fe_ratio = parameters.get("Fe_ratio", 0.4)
    sintering_temp = parameters.get("sintering_temperature", 600)
    electrolyte_conc = parameters.get("electrolyte_concentration", 1.0)
    current_density = parameters.get("current_density", 20)
    deposition_time = parameters.get("deposition_time", 300)
    precursor_ph = parameters.get("precursor_pH", 7.0)
    
    # 计算基础斜率
    base_slope = 5.0 + ni_ratio * 8.0 + co_ratio * 6.0 + fe_ratio * 4.0
    
    # 添加其他参数的影响
    temp_factor = sintering_temp / 600
    conc_factor = electrolyte_conc / 1.0
    current_factor = current_density / 20
    time_factor = min(1.0, deposition_time / 400)
    ph_factor = 1.0 - abs(precursor_ph - 7.0) / 7.0
    
    # 计算最终斜率
    slope = base_slope * temp_factor * conc_factor * current_factor * time_factor * ph_factor
    
    # 添加随机噪声
    slope *= random.uniform(0.9, 1.1)
    
    # 生成电压-电流数据点
    voltage_start = 0.0
    voltage_end = 1.0
    num_points = 100
    
    lsv_data = []
    for i in range(num_points):
        voltage = voltage_start + (voltage_end - voltage_start) * i / (num_points - 1)
        # 使用斜率生成电流值，添加一些非线性和噪声
        current = slope * voltage + 0.1 * math.sin(voltage * 10) + random.uniform(-0.05, 0.05)
        lsv_data.append({"voltage": voltage, "current": current})
    
    return lsv_data, slope

def generate_cv_data(parameters):
    """生成模拟的CV数据"""
    # 提取参数
    ni_ratio = parameters.get("Ni_ratio", 0.3)
    co_ratio = parameters.get("Co_ratio", 0.2)
    fe_ratio = parameters.get("Fe_ratio", 0.4)
    sintering_temp = parameters.get("sintering_temperature", 600)
    current_density = parameters.get("current_density", 20)
    deposition_time = parameters.get("deposition_time", 300)
    precursor_ph = parameters.get("precursor_pH", 7.0)
    
    # 计算基础稳定性
    base_stability = 2.0 + ni_ratio * 1.0 + co_ratio * 2.0 + fe_ratio * 3.0
    
    # 添加其他参数的影响
    temp_factor = sintering_temp / 600
    current_factor = current_density / 20
    time_factor = max(0.5, min(1.5, deposition_time / 300))
    ph_factor = abs(precursor_ph - 4.0) / 4.0
    
    # 计算最终稳定性
    stability = base_stability * temp_factor * current_factor * time_factor * ph_factor
    
    # 添加随机噪声
    stability *= random.uniform(0.85, 1.15)
    
    # 生成多个循环的CV数据
    num_cycles = 5
    cv_data = {}
    
    for cycle in range(1, num_cycles + 1):
        cycle_key = f"cycle{cycle}"
        
        # 生成电压-电流数据点
        voltage_start = 0.0
        voltage_end = 1.0
        num_points = 200
        
        # 循环衰减因子
        decay_factor = 1.0 - (stability / 10.0) * (cycle - 1) / (num_cycles - 1)
        
        cycle_data = []
        # 正向扫描
        for i in range(num_points // 2):
            voltage = voltage_start + (voltage_end - voltage_start) * i / (num_points // 2 - 1)
            # 使用正弦函数模拟CV曲线，添加衰减和噪声
            current = decay_factor * (2.0 * math.sin(voltage * math.pi) + random.uniform(-0.1, 0.1))
            cycle_data.append({"voltage": voltage, "current": current})
        
        # 反向扫描
        for i in range(num_points // 2):
            voltage = voltage_end - (voltage_end - voltage_start) * i / (num_points // 2 - 1)
            # 使用正弦函数模拟CV曲线，添加衰减、滞后和噪声
            current = decay_factor * (1.8 * math.sin(voltage * math.pi - 0.2) + random.uniform(-0.1, 0.1))
            cycle_data.append({"voltage": voltage, "current": current})
        
        cv_data[cycle_key] = cycle_data
    
    return cv_data, stability

def generate_experiment_data(parameters, output_file):
    """生成完整的模拟实验数据"""
    # 生成LSV数据
    lsv_data, lsv_slope = generate_lsv_data(parameters)
    
    # 生成CV数据
    cv_data, cv_stability = generate_cv_data(parameters)
    
    # 构建完整的实验数据
    experiment_data = {
        "parameters": parameters,
        "lsv_data": lsv_data,
        "cv_data": cv_data,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": f"sim_{random.randint(10000, 99999)}",
            "operator": "Simulation"
        }
    }
    
    # 保存到文件
    with open(output_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"Generated experiment data saved to {output_file}")
    print(f"LSV Slope: {lsv_slope:.4f}")
    print(f"CV Stability: {cv_stability:.4f}")
    
    return experiment_data

def main():
    parser = argparse.ArgumentParser(description="Catalyst Experiment Simulator")
    parser.add_argument("--design", required=True, help="Design parameters JSON file")
    parser.add_argument("--output", default="experiment_data.json", help="Output file for experiment data")
    
    args = parser.parse_args()
    
    # 读取设计参数
    with open(args.design, 'r') as f:
        design = json.load(f)
    
    # 如果设计文件包含多个设计点，使用第一个
    if "designs" in design:
        parameters = design["designs"][0]["parameters"]
    elif "parameters" in design:
        parameters = design["parameters"]
    else:
        parameters = design
    
    # 生成实验数据
    generate_experiment_data(parameters, args.output)
    
    return 0

if __name__ == "__main__":
    exit(main())