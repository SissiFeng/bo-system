# mock_bo_server.py (继续)

def latin_hypercube_sampling(n_samples, parameter_space):
    """简单的拉丁超立方采样实现"""
    continuous_params = []
    categorical_params = []
    integer_params = []
    
    # 按类型分离参数
    for param in parameter_space["parameters"]:
        if param["type"] == "continuous":
            continuous_params.append(param)
        elif param["type"] == "categorical":
            categorical_params.append(param)
        elif param["type"] == "integer":
            integer_params.append(param)
    
    # 生成样本
    samples = []
    for _ in range(n_samples):
        sample = {}
        
        # 处理连续参数
        for param in continuous_params:
            min_val, max_val = param["bounds"]
            sample[param["name"]] = np.random.uniform(min_val, max_val)
        
        # 处理整数参数
        for param in integer_params:
            min_val, max_val = param["bounds"]
            sample[param["name"]] = np.random.randint(min_val, max_val + 1)
        
        # 处理分类参数
        for param in categorical_params:
            choices = param["choices"]
            sample[param["name"]] = np.random.choice(choices)
        
        samples.append(sample)
    
    # 应用约束条件（如果有）
    valid_samples = []
    for sample in samples:
        valid = True
        
        # 检查约束条件
        for constraint in parameter_space.get("constraints", []):
            if constraint["type"] == "sum":
                param_names = constraint["parameters"]
                relation = constraint["relation"]
                value = constraint["value"]
                
                param_sum = sum(sample[name] for name in param_names)
                
                if relation == "<=" and param_sum > value:
                    valid = False
                elif relation == ">=" and param_sum < value:
                    valid = False
                elif relation == "==" and param_sum != value:
                    valid = False
        
        if valid:
            valid_samples.append(sample)
    
    # 如果没有足够的有效样本，生成更多
    while len(valid_samples) < n_samples:
        sample = {}
        
        # 处理连续参数
        for param in continuous_params:
            min_val, max_val = param["bounds"]
            sample[param["name"]] = np.random.uniform(min_val, max_val)
        
        # 处理整数参数
        for param in integer_params:
            min_val, max_val = param["bounds"]
            sample[param["name"]] = np.random.randint(min_val, max_val + 1)
        
        # 处理分类参数
        for param in categorical_params:
            choices = param["choices"]
            sample[param["name"]] = np.random.choice(choices)
        
        # 检查约束条件
        valid = True
        for constraint in parameter_space.get("constraints", []):
            if constraint["type"] == "sum":
                param_names = constraint["parameters"]
                relation = constraint["relation"]
                value = constraint["value"]
                
                param_sum = sum(sample[name] for name in param_names)
                
                if relation == "<=" and param_sum > value:
                    valid = False
                elif relation == ">=" and param_sum < value:
                    valid = False
                elif relation == "==" and param_sum != value:
                    valid = False
        
        if valid:
            valid_samples.append(sample)
            if len(valid_samples) >= n_samples:
                break
    
    return valid_samples[:n_samples]

def random_next_designs(n_samples, parameter_space, previous_results=None):
    """生成随机的下一批设计点（在真实系统中，这将使用贝叶斯优化）"""
    # 在这个模拟服务器中，我们只是使用拉丁超立方采样
    # 在真实系统中，这将使用贝叶斯优化算法
    return latin_hypercube_sampling(n_samples, parameter_space)

# API端点
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/parameter-space", response_model=ParameterSpaceResponse)
async def create_parameter_space(data: ParameterSpaceConfig):
    task_id = generate_id()
    
    # 存储参数空间
    parameter_spaces[task_id] = data.dict()
    
    # 创建任务条目
    tasks[task_id] = {
        "task_id": task_id,
        "name": data.name,
        "status": "configured",
        "progress": 0.0,
    }
    
    # 保存到文件
    task_path = task_dir / task_id
    os.makedirs(task_path, exist_ok=True)
    
    with open(task_path / "parameter_space.json", "w") as f:
        json.dump(parameter_spaces[task_id], f, indent=2)
    
    with open(task_path / "task_info.json", "w") as f:
        json.dump(tasks[task_id], f, indent=2)
    
    return ParameterSpaceResponse(
        task_id=task_id,
        message=f"Parameter space '{data.name}' created successfully"
    )

@app.get("/api/designs/{task_id}/initial", response_model=DesignResponse)
async def get_initial_designs(
    task_id: str = Path(..., description="Task ID"),
    samples: int = Query(5, description="Number of samples to generate"),
):
    if task_id not in parameter_spaces:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # 使用拉丁超立方采样生成设计点
    parameter_space = parameter_spaces[task_id]
    design_points = latin_hypercube_sampling(samples, parameter_space)
    
    # 转换为Design对象
    design_objects = [
        Design(id=generate_id(), parameters=point)
        for point in design_points
    ]
    
    # 存储设计点
    designs[task_id] = design_objects
    
    # 保存到文件
    task_path = task_dir / task_id
    with open(task_path / "initial_designs.json", "w") as f:
        json.dump([d.dict() for d in design_objects], f, indent=2)
    
    # 更新任务状态
    tasks[task_id]["status"] = "ready_for_results"
    tasks[task_id]["progress"] = 10.0
    
    with open(task_path / "task_info.json", "w") as f:
        json.dump(tasks[task_id], f, indent=2)
    
    return DesignResponse(designs=design_objects)

@app.post("/api/results/{task_id}")
async def submit_results(
    results_data: ResultsSubmission,
    task_id: str = Path(..., description="Task ID"),
):
    if task_id not in parameter_spaces:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # 存储结果
    if task_id not in results:
        results[task_id] = []
    
    results[task_id].extend([r.dict() for r in results_data.results])
    
    # 保存到文件
    task_path = task_dir / task_id
    with open(task_path / "results.json", "w") as f:
        json.dump(results[task_id], f, indent=2)
    
    # 更新任务状态
    tasks[task_id]["status"] = "optimizing"
    tasks[task_id]["progress"] = min(80.0, 10.0 + len(results[task_id]) * 5.0)
    
    with open(task_path / "task_info.json", "w") as f:
        json.dump(tasks[task_id], f, indent=2)
    
    return {
        "message": f"{len(results_data.results)} results submitted successfully",
        "results_count": len(results[task_id]),
        "progress": tasks[task_id]["progress"]
    }

@app.get("/api/designs/{task_id}/next", response_model=DesignResponse)
async def get_next_designs(
    task_id: str = Path(..., description="Task ID"),
    batch_size: int = Query(1, description="Number of designs to generate"),
):
    if task_id not in parameter_spaces:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    if task_id not in results or not results[task_id]:
        raise HTTPException(status_code=400, detail="No results submitted yet")
    
    # 生成下一批设计点（在真实系统中，这将使用贝叶斯优化）
    parameter_space = parameter_spaces[task_id]
    next_points = random_next_designs(batch_size, parameter_space)
    
    # 转换为Design对象
    next_designs = [
        Design(id=generate_id(), parameters=point)
        for point in next_points
    ]
    
    # 保存到文件
    task_path = task_dir / task_id
    with open(task_path / "next_designs.json", "w") as f:
        json.dump([d.dict() for d in next_designs], f, indent=2)
    
    return DesignResponse(designs=next_designs)

@app.get("/api/tasks/{task_id}/status")
async def get_task_status(task_id: str = Path(..., description="Task ID")):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task_info = tasks[task_id]
    
    # 获取结果数量
    results_count = 0
    if task_id in results:
        results_count = len(results[task_id])
    
    return {
        "status": task_info["status"],
        "progress": task_info["progress"],
        "results_count": results_count
    }

@app.get("/api/tasks")
async def get_tasks():
    return {"tasks": list(tasks.values())}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock BO Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8005, help="Port to bind")
    
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
