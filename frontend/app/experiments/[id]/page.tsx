"use client"

import { useState, useEffect } from "react"
import { useParams, useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Play, Pause, RotateCw, Settings, Download, FileSpreadsheet } from "lucide-react"
import Link from "next/link"
import { DesignPointsTable } from "@/components/experiment/design-points-table"
import { ExperimentSummary } from "@/components/experiment/experiment-summary"
import { ResultsSubmissionForm } from "@/components/experiment/results-submission-form"
import { OptimizationHistory } from "@/components/experiment/optimization-history"

// 模拟实验数据，实际应用中应从API获取
const mockExperiment = {
  id: "exp-001",
  name: "催化剂优化实验-A",
  description: "通过贝叶斯优化寻找最佳催化剂配方，优化转化率和选择性",
  status: "running",
  created_at: "2023-06-15T10:30:00Z",
  updated_at: "2023-06-16T14:20:00Z",
  parameters: [
    { name: "温度", type: "continuous", min: 50, max: 100, unit: "°C" },
    { name: "压力", type: "continuous", min: 1, max: 10, unit: "bar" },
    { name: "催化剂A", type: "continuous", min: 0, max: 0.5, unit: "mol%" },
    { name: "催化剂B", type: "continuous", min: 0, max: 0.5, unit: "mol%" },
  ],
  objectives: [
    { name: "转化率", type: "maximize", unit: "%" },
    { name: "选择性", type: "maximize", unit: "%" },
  ],
  constraints: [{ expression: "催化剂A + 催化剂B", type: "sum_equals", value: 0.5 }],
  algorithm: {
    type: "bayesian",
    acquisition_function: "ei",
    kernel: "matern",
    exploration_weight: 0.5,
  },
  iterations_completed: 12,
  max_iterations: 25,
  best_result: {
    parameters: { 温度: 75.3, 压力: 5.2, 催化剂A: 0.3, 催化剂B: 0.2 },
    objectives: { 转化率: 85.2, 选择性: 92.1 },
  },
  initial_designs: [
    { id: "design-001", parameters: { 温度: 60, 压力: 3, 催化剂A: 0.2, 催化剂B: 0.3 } },
    { id: "design-002", parameters: { 温度: 80, 压力: 5, 催化剂A: 0.3, 催化剂B: 0.2 } },
    { id: "design-003", parameters: { 温度: 70, 压力: 7, 催化剂A: 0.1, 催化剂B: 0.4 } },
  ],
  results: [
    {
      design_id: "design-001",
      parameters: { 温度: 60, 压力: 3, 催化剂A: 0.2, 催化剂B: 0.3 },
      objectives: { 转化率: 72.5, 选择性: 85.3 },
      timestamp: "2023-06-15T11:30:00Z",
    },
    {
      design_id: "design-002",
      parameters: { 温度: 80, 压力: 5, 催化剂A: 0.3, 催化剂B: 0.2 },
      objectives: { 转化率: 85.2, 选择性: 92.1 },
      timestamp: "2023-06-15T14:45:00Z",
    },
    {
      design_id: "design-003",
      parameters: { 温度: 70, 压力: 7, 催化剂A: 0.1, 催化剂B: 0.4 },
      objectives: { 转化率: 78.9, 选择性: 88.7 },
      timestamp: "2023-06-16T09:20:00Z",
    },
  ],
  next_designs: [
    {
      id: "design-004",
      parameters: { 温度: 75, 压力: 6, 催化剂A: 0.25, 催化剂B: 0.25 },
      predictions: {
        转化率: { mean: 83.5, std: 2.1 },
        选择性: { mean: 90.2, std: 1.8 },
      },
      uncertainty: 0.05,
      reason: "平衡探索与利用，预期有较高的改进空间",
    },
  ],
}

export default function ExperimentDetailPage() {
  const params = useParams()
  const router = useRouter()
  const { id } = params
  const [experiment, setExperiment] = useState(mockExperiment)
  const [activeTab, setActiveTab] = useState("summary")
  const [isLoading, setIsLoading] = useState(false)

  // 实际应用中应从API获取实验数据
  useEffect(() => {
    // 模拟API调用
    // setIsLoading(true)
    // fetch(`/api/bo/experiments/${id}`)
    //   .then(response => response.json())
    //   .then(data => {
    //     setExperiment(data)
    //     setIsLoading(false)
    //   })
    //   .catch(error => {
    //     console.error("Error fetching experiment:", error)
    //     setIsLoading(false)
    //   })
  }, [id])

  // 更改实验状态
  const handleStatusChange = (newStatus: string) => {
    // 实际应用中应调用API
    // fetch(`/api/bo/experiments/${id}/status`, {
    //   method: 'PUT',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({ status: newStatus })
    // })
    //   .then(response => response.json())
    //   .then(data => {
    //     setExperiment({ ...experiment, status: newStatus })
    //   })

    // 模拟状态更改
    setExperiment({ ...experiment, status: newStatus })
  }

  // 获取下一批设计点
  const handleGetNextDesigns = () => {
    // 实际应用中应调用API
    // setIsLoading(true)
    // fetch(`/api/bo/experiments/${id}/next`, {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({ batch_size: 1 })
    // })
    //   .then(response => response.json())
    //   .then(data => {
    //     setExperiment({ ...experiment, next_designs: data.designs })
    //     setIsLoading(false)
    //   })
    //   .catch(error => {
    //     console.error("Error fetching next designs:", error)
    //     setIsLoading(false)
    //   })

    // 模拟获取新设计点
    const newDesign = {
      id: `design-00${experiment.next_designs.length + 4}`,
      parameters: {
        温度: 72 + Math.random() * 10,
        压力: 4 + Math.random() * 4,
        催化剂A: 0.2 + Math.random() * 0.2,
        催化剂B: 0.3 - Math.random() * 0.2,
      },
      predictions: {
        转化率: { mean: 80 + Math.random() * 10, std: 1.5 + Math.random() },
        选择性: { mean: 88 + Math.random() * 5, std: 1.2 + Math.random() },
      },
      uncertainty: 0.03 + Math.random() * 0.05,
      reason: "基于当前模型预测，此点有较高的期望改进",
    }

    setExperiment({
      ...experiment,
      next_designs: [...experiment.next_designs, newDesign],
    })
  }

  // 提交实验结果
  const handleSubmitResults = (designId: string, results: any) => {
    // 实际应用中应调用API
    // fetch(`/api/bo/experiments/${id}/evaluate`, {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({
    //     design_id: designId,
    //     objectives: results
    //   })
    // })
    //   .then(response => response.json())
    //   .then(data => {
    //     // 更新实验数据
    //     const updatedResults = [...experiment.results, {
    //       design_id: designId,
    //       parameters: experiment.next_designs.find(d => d.id === designId)?.parameters || {},
    //       objectives: results,
    //       timestamp: new Date().toISOString()
    //     }]
    //
    //     // 从next_designs中移除已提交的设计点
    //     const updatedNextDesigns = experiment.next_designs.filter(d => d.id !== designId)
    //
    //     setExperiment({
    //       ...experiment,
    //       results: updatedResults,
    //       next_designs: updatedNextDesigns,
    //       iterations_completed: experiment.iterations_completed + 1
    //     })
    //   })

    // 模拟提交结果
    const designToSubmit = experiment.next_designs.find((d) => d.id === designId)
    if (!designToSubmit) return

    const updatedResults = [
      ...experiment.results,
      {
        design_id: designId,
        parameters: designToSubmit.parameters,
        objectives: results,
        timestamp: new Date().toISOString(),
      },
    ]

    // 从next_designs中移除已提交的设计点
    const updatedNextDesigns = experiment.next_designs.filter((d) => d.id !== designId)

    // 检查是否是新的最佳结果
    let newBestResult = experiment.best_result
    const isBetter = Object.entries(results).every(([key, value]) => {
      const objective = experiment.objectives.find((o) => o.name === key)
      if (!objective) return false

      if (objective.type === "maximize") {
        return value > (experiment.best_result.objectives[key] || 0)
      } else {
        return value < (experiment.best_result.objectives[key] || Number.POSITIVE_INFINITY)
      }
    })

    if (isBetter) {
      newBestResult = {
        parameters: designToSubmit.parameters,
        objectives: results,
      }
    }

    setExperiment({
      ...experiment,
      results: updatedResults,
      next_designs: updatedNextDesigns,
      iterations_completed: experiment.iterations_completed + 1,
      best_result: newBestResult,
    })
  }

  // 获取状态徽章样式
  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return <Badge className="bg-green-100 text-green-800">已完成</Badge>
      case "running":
        return <Badge className="bg-blue-100 text-blue-800">运行中</Badge>
      case "paused":
        return <Badge className="bg-yellow-100 text-yellow-800">已暂停</Badge>
      case "draft":
        return <Badge className="bg-gray-100 text-gray-800">草稿</Badge>
      default:
        return <Badge variant="outline">{status}</Badge>
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <Button variant="outline" size="icon" asChild className="mr-4">
            <Link href="/experiments">
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </Button>
          <div>
            <h1 className="text-3xl font-bold">{experiment.name}</h1>
            <div className="flex items-center mt-1 space-x-2">
              {getStatusBadge(experiment.status)}
              <span className="text-sm text-muted-foreground">
                迭代: {experiment.iterations_completed}/{experiment.max_iterations}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {experiment.status === "running" ? (
            <Button variant="outline" onClick={() => handleStatusChange("paused")}>
              <Pause className="mr-2 h-4 w-4" />
              暂停
            </Button>
          ) : (
            <Button variant="outline" onClick={() => handleStatusChange("running")}>
              <Play className="mr-2 h-4 w-4" />
              继续
            </Button>
          )}
          <Button variant="outline" asChild>
            <Link href={`/experiments/${id}/settings`}>
              <Settings className="mr-2 h-4 w-4" />
              设置
            </Link>
          </Button>
          <Button variant="outline">
            <Download className="mr-2 h-4 w-4" />
            导出
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList>
          <TabsTrigger value="summary">实验概览</TabsTrigger>
          <TabsTrigger value="designs">设计点</TabsTrigger>
          <TabsTrigger value="results">提交结果</TabsTrigger>
          <TabsTrigger value="history">优化历史</TabsTrigger>
          <TabsTrigger value="visualization" asChild>
            <Link href={`/experiments/${id}/visualization`}>可视化分析</Link>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="summary">
          <ExperimentSummary experiment={experiment} />
        </TabsContent>

        <TabsContent value="designs">
          <Card>
            <CardHeader>
              <CardTitle>实验设计点</CardTitle>
              <CardDescription>查看初始设计点和推荐的下一批设计点</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium mb-3">初始设计点</h3>
                  <DesignPointsTable
                    designs={experiment.initial_designs.map((d) => ({
                      id: d.id,
                      parameters: d.parameters,
                      isCompleted: experiment.results.some((r) => r.design_id === d.id),
                    }))}
                    parameters={experiment.parameters}
                    showActions={false}
                  />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-medium">推荐的下一批设计点</h3>
                    <Button onClick={handleGetNextDesigns} disabled={isLoading}>
                      <RotateCw className="mr-2 h-4 w-4" />
                      获取新设计点
                    </Button>
                  </div>

                  {experiment.next_designs.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">暂无推荐的设计点，点击上方按钮获取</div>
                  ) : (
                    <DesignPointsTable
                      designs={experiment.next_designs.map((d) => ({
                        id: d.id,
                        parameters: d.parameters,
                        predictions: d.predictions,
                        uncertainty: d.uncertainty,
                        reason: d.reason,
                        isCompleted: false,
                      }))}
                      parameters={experiment.parameters}
                      showPredictions={true}
                      showActions={true}
                      onViewDetails={(designId) => setActiveTab("results")}
                    />
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>提交实验结果</CardTitle>
                <CardDescription>为推荐的设计点提交实验测量结果</CardDescription>
              </CardHeader>
              <CardContent>
                {experiment.next_designs.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">暂无待提交结果的设计点</div>
                ) : (
                  <ResultsSubmissionForm
                    designs={experiment.next_designs}
                    objectives={experiment.objectives}
                    onSubmit={handleSubmitResults}
                  />
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>已提交的结果</CardTitle>
                <CardDescription>查看所有已提交的实验结果</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {experiment.results.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">暂无已提交的实验结果</div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full border-collapse">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2 px-2">设计点ID</th>
                            {experiment.objectives.map((obj) => (
                              <th key={obj.name} className="text-left py-2 px-2">
                                {obj.name} ({obj.unit || ""})
                              </th>
                            ))}
                            <th className="text-left py-2 px-2">提交时间</th>
                          </tr>
                        </thead>
                        <tbody>
                          {experiment.results.map((result, index) => (
                            <tr key={index} className="border-b">
                              <td className="py-2 px-2 font-mono text-xs">{result.design_id}</td>
                              {experiment.objectives.map((obj) => (
                                <td key={obj.name} className="py-2 px-2">
                                  {result.objectives[obj.name]?.toFixed(2) || "-"}
                                </td>
                              ))}
                              <td className="py-2 px-2 text-sm">{new Date(result.timestamp).toLocaleString()}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  <div className="flex justify-end">
                    <Button variant="outline" size="sm">
                      <FileSpreadsheet className="mr-2 h-4 w-4" />
                      导出结果
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>优化历史</CardTitle>
              <CardDescription>查看优化过程中的参数和目标值变化</CardDescription>
            </CardHeader>
            <CardContent>
              <OptimizationHistory
                results={experiment.results}
                objectives={experiment.objectives}
                parameters={experiment.parameters}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
