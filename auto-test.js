/**
 * 自动测试脚本，用于测试所有API端点
 */

const API_BASE_URL = 'http://localhost:8005/api';
let taskId = null;
let designs = null;

// 测试参数空间创建
async function testCreateParameterSpace() {
  console.log('\n===== 测试创建参数空间 =====');
  
  const data = {
    name: "自动测试优化任务",
    description: "自动测试脚本创建的任务",
    parameters: [
      {
        name: "x1",
        type: "continuous",
        bounds: [0, 10],
        description: "连续参数"
      },
      {
        name: "x2",
        type: "integer",
        bounds: [1, 5],
        description: "整数参数"
      },
      {
        name: "x3",
        type: "categorical",
        categories: ["A", "B", "C"],
        description: "分类参数"
      }
    ],
    objectives: [
      {
        name: "y1",
        direction: "minimize",
        description: "目标函数"
      }
    ]
  };
  
  try {
    const response = await fetch(`${API_BASE_URL}/parameter-space`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    
    const result = await response.json();
    
    if (response.ok) {
      console.log('✅ 参数空间创建成功');
      console.log('任务ID:', result.task_id);
      taskId = result.task_id;
      return true;
    } else {
      console.error('❌ 参数空间创建失败:', result);
      return false;
    }
  } catch (error) {
    console.error('❌ 请求错误:', error);
    return false;
  }
}

// 测试获取任务列表
async function testGetTasks() {
  console.log('\n===== 测试获取任务列表 =====');
  
  try {
    const response = await fetch(`${API_BASE_URL}/tasks`);
    const result = await response.json();
    
    if (response.ok) {
      console.log('✅ 获取任务列表成功');
      console.log(`共有 ${result.tasks.length} 个任务`);
      return true;
    } else {
      console.error('❌ 获取任务列表失败:', result);
      return false;
    }
  } catch (error) {
    console.error('❌ 请求错误:', error);
    return false;
  }
}

// 测试获取初始设计点
async function testGetInitialDesigns() {
  console.log('\n===== 测试获取初始设计点 =====');
  
  if (!taskId) {
    console.error('❌ 需要先创建任务获取任务ID');
    return false;
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}/designs/${taskId}/initial?samples=3`);
    const result = await response.json();
    
    if (response.ok) {
      console.log('✅ 获取初始设计点成功');
      console.log(`获取了 ${result.designs.length} 个设计点`);
      designs = result.designs;
      return true;
    } else {
      console.error('❌ 获取初始设计点失败:', result);
      return false;
    }
  } catch (error) {
    console.error('❌ 请求错误:', error);
    return false;
  }
}

// 测试提交结果
async function testSubmitResults() {
  console.log('\n===== 测试提交结果 =====');
  
  if (!taskId) {
    console.error('❌ 需要先创建任务获取任务ID');
    return false;
  }
  
  if (!designs || designs.length === 0) {
    console.error('❌ 需要先获取设计点');
    return false;
  }
  
  const data = {
    results: designs.map(design => ({
      parameters: design.parameters,
      objectives: { y1: Math.random() * 10 },
      metadata: { timestamp: new Date().toISOString() }
    }))
  };
  
  try {
    const response = await fetch(`${API_BASE_URL}/results/${taskId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    
    const result = await response.json();
    
    if (response.ok) {
      console.log('✅ 提交结果成功');
      return true;
    } else {
      console.error('❌ 提交结果失败:', result);
      return false;
    }
  } catch (error) {
    console.error('❌ 请求错误:', error);
    return false;
  }
}

// 测试获取下一个设计点
async function testGetNextDesigns() {
  console.log('\n===== 测试获取下一个设计点 =====');
  
  if (!taskId) {
    console.error('❌ 需要先创建任务获取任务ID');
    return false;
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}/designs/${taskId}/next?batch_size=1`);
    const result = await response.json();
    
    if (response.ok) {
      console.log('✅ 获取下一个设计点成功');
      console.log(`获取了 ${result.designs.length} 个设计点`);
      return true;
    } else {
      console.error('❌ 获取下一个设计点失败:', result);
      return false;
    }
  } catch (error) {
    console.error('❌ 请求错误:', error);
    return false;
  }
}

// 测试获取任务状态
async function testGetTaskStatus() {
  console.log('\n===== 测试获取任务状态 =====');
  
  if (!taskId) {
    console.error('❌ 需要先创建任务获取任务ID');
    return false;
  }
  
  try {
    const response = await fetch(`${API_BASE_URL}/tasks/${taskId}/status`);
    const result = await response.json();
    
    if (response.ok) {
      console.log('✅ 获取任务状态成功');
      console.log('任务状态:', result.status);
      console.log('进度:', result.progress);
      return true;
    } else {
      console.error('❌ 获取任务状态失败:', result);
      return false;
    }
  } catch (error) {
    console.error('❌ 请求错误:', error);
    return false;
  }
}

// 运行所有测试
async function runAllTests() {
  console.log('开始运行所有测试...');
  
  const tests = [
    { name: '创建参数空间', fn: testCreateParameterSpace },
    { name: '获取任务列表', fn: testGetTasks },
    { name: '获取初始设计点', fn: testGetInitialDesigns },
    { name: '提交结果', fn: testSubmitResults },
    { name: '获取下一个设计点', fn: testGetNextDesigns },
    { name: '获取任务状态', fn: testGetTaskStatus },
  ];
  
  const results = {};
  
  for (const test of tests) {
    console.log(`\n正在运行测试: ${test.name}`);
    const success = await test.fn();
    results[test.name] = success ? '✅ 成功' : '❌ 失败';
    
    // 如果是关键测试失败，可能需要中断后续测试
    if (!success && (test.name === '创建参数空间' || test.name === '获取初始设计点')) {
      console.error(`关键测试 ${test.name} 失败，中断后续测试`);
      break;
    }
    
    // 添加延迟，避免请求过快
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  console.log('\n===== 测试结果摘要 =====');
  for (const [name, result] of Object.entries(results)) {
    console.log(`${name}: ${result}`);
  }
}

// 运行测试
runAllTests().catch(console.error);
