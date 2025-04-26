import { useState, useEffect } from 'react';

export default function ApiTest() {
  const [apiStatus, setApiStatus] = useState('未测试');
  const [apiResponse, setApiResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8005/api';

  const testEndpoints = [
    { name: '创建参数空间', endpoint: '/parameter-space', method: 'POST', data: {
      name: "测试优化任务",
      description: "前后端联调测试",
      parameters: [
        {
          name: "x1",
          type: "continuous",
          bounds: [0, 10],
          description: "第一个参数"
        },
        {
          name: "x2",
          type: "integer",
          bounds: [1, 5],
          description: "第二个参数"
        },
        {
          name: "x3",
          type: "categorical",
          categories: ["A", "B", "C"],
          description: "第三个参数"
        }
      ],
      objectives: [
        {
          name: "y1",
          direction: "minimize",
          description: "目标函数"
        }
      ]
    }},
    { name: '获取任务列表', endpoint: '/tasks', method: 'GET' },
    { name: '获取初始设计点', endpoint: '/designs/{task_id}/initial?samples=3', method: 'GET', needsTaskId: true },
    { name: '提交结果', endpoint: '/results/{task_id}', method: 'POST', needsTaskId: true, needsDesigns: true,
      dataFn: (designs) => ({
        results: designs.map(design => ({
          parameters: design.parameters,
          objectives: { y1: Math.random() * 10 },
          metadata: { timestamp: new Date().toISOString() }
        }))
      })
    },
    { name: '获取下一个设计点', endpoint: '/designs/{task_id}/next?batch_size=1', method: 'GET', needsTaskId: true },
    { name: '获取任务状态', endpoint: '/tasks/{task_id}/status', method: 'GET', needsTaskId: true },
  ];

  const [taskId, setTaskId] = useState('');
  const [designs, setDesigns] = useState([]);
  const [testResults, setTestResults] = useState({});
  const [currentTest, setCurrentTest] = useState(null);

  const fetchAPI = async (endpoint, options = {}) => {
    const url = `${API_BASE_URL}${endpoint}`;

    try {
      setLoading(true);
      setError(null);

      console.log(`Making ${options.method || 'GET'} request to ${url}`, options.body ? JSON.parse(options.body) : '');

      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        // 添加这些选项以确保跨域请求正常工作
        mode: 'cors',
        credentials: 'same-origin',
      });

      console.log(`Response status: ${response.status}`);

      let data;
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
        console.log('Response data:', data);
      } else {
        const text = await response.text();
        console.log('Response text:', text);
        try {
          data = JSON.parse(text);
        } catch (e) {
          data = { text };
        }
      }

      if (!response.ok) {
        throw new Error(`API错误: ${response.status} - ${data.detail || JSON.stringify(data)}`);
      }

      return { success: true, data };
    } catch (error) {
      console.error('请求失败:', error);
      setError(error.message);
      return { success: false, error: error.message };
    } finally {
      setLoading(false);
    }
  };

  const runTest = async (test) => {
    setCurrentTest(test.name);

    let endpoint = test.endpoint;
    let data = test.data;

    // 替换路径中的任务ID
    if (test.needsTaskId) {
      if (!taskId) {
        setError('需要先创建任务获取任务ID');
        setTestResults({
          ...testResults,
          [test.name]: { success: false, error: '需要先创建任务获取任务ID' }
        });
        return;
      }
      endpoint = endpoint.replace('{task_id}', taskId);
    }

    // 如果需要设计点数据
    if (test.needsDesigns) {
      if (!designs || designs.length === 0) {
        setError('需要先获取设计点');
        setTestResults({
          ...testResults,
          [test.name]: { success: false, error: '需要先获取设计点' }
        });
        return;
      }

      if (test.dataFn) {
        data = test.dataFn(designs);
      }
    }

    const options = {
      method: test.method,
      ...(data && { body: JSON.stringify(data) }),
    };

    const result = await fetchAPI(endpoint, options);

    if (result.success) {
      setApiResponse(result.data);
      setApiStatus('成功');

      // 保存任务ID
      if (test.name === '创建参数空间' && result.data.task_id) {
        setTaskId(result.data.task_id);
      }

      // 保存设计点
      if (test.name === '获取初始设计点' && result.data.designs) {
        setDesigns(result.data.designs);
      }
    } else {
      setApiStatus('失败');
      setApiResponse(null);
    }

    setTestResults({
      ...testResults,
      [test.name]: result
    });
  };

  const runAllTests = async () => {
    setTestResults({});

    for (const test of testEndpoints) {
      await runTest(test);
      // 添加延迟，避免请求过快
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    setCurrentTest(null);
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">前后端API联调测试</h1>

      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">API状态</h2>
        <p>
          <span className="font-medium">后端API地址: </span>
          <code className="bg-gray-100 px-2 py-1 rounded">{API_BASE_URL}</code>
        </p>
        {taskId && (
          <p className="mt-2">
            <span className="font-medium">当前任务ID: </span>
            <code className="bg-gray-100 px-2 py-1 rounded">{taskId}</code>
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div>
          <h2 className="text-xl font-semibold mb-2">测试操作</h2>
          <div className="space-y-2">
            <button
              onClick={runAllTests}
              disabled={loading}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
            >
              运行所有测试
            </button>

            <div className="space-y-2 mt-4">
              {testEndpoints.map((test) => (
                <button
                  key={test.name}
                  onClick={() => runTest(test)}
                  disabled={loading || (test.needsTaskId && !taskId) || (test.needsDesigns && designs.length === 0)}
                  className="block w-full px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-50 text-left"
                >
                  {test.name}
                  {testResults[test.name] && (
                    <span className={`float-right ${testResults[test.name].success ? 'text-green-500' : 'text-red-500'}`}>
                      {testResults[test.name].success ? '✓' : '✗'}
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div>
          <h2 className="text-xl font-semibold mb-2">测试结果</h2>
          {loading ? (
            <div className="p-4 border rounded bg-gray-50">
              <p>正在测试 {currentTest}...</p>
            </div>
          ) : error ? (
            <div className="p-4 border rounded bg-red-50 text-red-700">
              <p className="font-bold">错误:</p>
              <p>{error}</p>
            </div>
          ) : apiResponse ? (
            <div className="p-4 border rounded bg-green-50">
              <p className="font-bold text-green-700 mb-2">API响应 ({apiStatus}):</p>
              <pre className="bg-white p-2 rounded overflow-auto max-h-80 text-sm">
                {JSON.stringify(apiResponse, null, 2)}
              </pre>
            </div>
          ) : (
            <div className="p-4 border rounded bg-gray-50">
              <p>点击左侧按钮开始测试</p>
            </div>
          )}
        </div>
      </div>

      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">测试结果摘要</h2>
        <div className="border rounded overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">测试名称</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">状态</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">详情</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {Object.entries(testResults).map(([name, result]) => (
                <tr key={name}>
                  <td className="px-6 py-4 whitespace-nowrap">{name}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      result.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}>
                      {result.success ? '成功' : '失败'}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    {result.success
                      ? (result.data && result.data.task_id
                          ? `任务ID: ${result.data.task_id}`
                          : '请求成功')
                      : result.error}
                  </td>
                </tr>
              ))}
              {Object.keys(testResults).length === 0 && (
                <tr>
                  <td colSpan="3" className="px-6 py-4 text-center text-sm text-gray-500">
                    尚未运行任何测试
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
