"use client"

import { useState, useEffect } from "react"
import { useParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ArrowLeft, Download } from "lucide-react"
import Link from "next/link"
import { OptimizationProgressChart } from "@/components/visualization/optimization-progress-chart"
import { ParameterVariationChart } from "@/components/visualization/parameter-variation-chart"
import { ParetoFrontVisualization } from "@/components/pareto-front-visualization"

// 模拟实验数据，实际应用中应从API获取
const mockExperiment = {
  id: "exp-001",
  name: "催化剂优化实验-A",
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
  results: [
    {
      design_id: "design-001",
      parameters: { 温度: 60, 压力: 3, 催化剂A: 0.2, 催化剂B: 0.3 },
      objectives: { 转化率: 72.5, 选择性: 85.3 },
      timestamp: "2023-06-15T11:30:00Z",
      iteration: 1,
    },
    {
      design_id: "design-002",
      parameters: { 温度: 80, 压力: 5, 催化剂A: 0.3, 催化剂B: 0.2 },
      objectives: { 转化率: 85.2, 选择性: 92.1 },
      timestamp: "2023-06-15T14:45:00Z",
      iteration: 2,
    },
    {
      design_id: "design-003",
      parameters: { 温度: 70, 压力: 7, 催化剂A: 0.1, 催化剂B: 0.4 },
      objectives: { 转化率: 78.9, 选择性: 88.7 },
      timestamp: "2023-06-16T09:20:00Z",
      iteration: 3,
    },
    {
      design_id: "design-004",
      parameters: { 温度: 75, 压力: 6, 催化剂A: 0.25, 催化剂B: 0.25 },
      objectives: { 转化率: 83.5, 选择性: 90.2 },
      timestamp: "2023-06-16T14:10:00Z",
      iteration: 4,
    },
    {
      design_id: "design-005",
      parameters: { 温度: 78, 压力: 5.5, 催化剂A: 0.28, 催化剂B: 0.22 },
      objectives: { 转化率: 84.8, 选择性: 91.5 },
      timestamp: "2023-06-17T10:30:00Z",
      iteration: 5,
    },
    {
      design_id: "design-006",
      parameters: { 温度: 77, 压力: 5.8, 催化剂A: 0.32, 催化剂B: 0.18 },
      objectives: { 转化率: 86.1, 选择性: 93.2 },
      timestamp: "2023-06-17T15:45:00Z",
      iteration: 6,
    },
    {
      design_id: "design-007",
      parameters: { 温度: 76, 压力: 5.7, 催化剂A: 0.31, 催化剂B: 0.19 },
      objectives: { 转化率: 85.7, 选择性: 92.8 },
      timestamp: "2023-06-18T09:20:00Z",
      iteration: 7,
    },
    {
      design_id: "design-008",
      parameters: { 温度: 79, 压力: 5.9, 催化剂A: 0.33, 催化剂B: 0.17 },
      objectives: { 转化率: 86.5, 选择性: 93.5 },
      timestamp: "2023-06-18T14:10:00Z",
      iteration: 8,
    },
    {
      design_id: "design-009",
      parameters: { 温度: 78.5, 压力: 5.8, 催化剂A: 0.32, 催化剂B: 0.18 },
      objectives: { 转化率: 86.3, 选择性: 93.4 },
      timestamp: "2023-06-19T10:30:00Z",
      iteration: 9,
    },
    {
      design_id: "design-010",
      parameters: { 温度: 79.2, 压力: 6.0, 催化剂A: 0.34, 催化剂B: 0.16 },
      objectives: { 转化率: 86.8, 选择性: 93.7 },
      timestamp: "2023-06-19T15:45:00Z",
      iteration: 10,
    },
    {
      design_id: "design-011",
      parameters: { 温度: 79.5, 压力: 6.1, 催化剂A: 0.35, 催化剂B: 0.15 },
      objectives: { 转化率: 87.0, 选择性: 93.9 },
      timestamp: "2023-06-20T09:20:00Z",
      iteration: 11,
    },
    {
      design_id: "design-012",
      parameters: { 温度: 79.8, 压力: 6.2, 催化剂A: 0.36, 催化剂B: 0.14 },
      objectives: { 转化率: 87.2, 选择性: 94.0 },
      timestamp: "2023-06-20T14:10:00Z",
      iteration: 12,
    },
  ],
  paretoFront: [
    { id: 8, x: 0.865, y: 0.935, uncertainty: 0.02 },
    { id: 10, x: 0.868, y: 0.937, uncertainty: 0.018 },
    { id: 11, x: 0.87, y: 0.939, uncertainty: 0.015 },
    { id: 12, x: 0.872, y: 0.94, uncertainty: 0.012 },
  ],
  nonDominatedSolutions: [
    { id: 6, x: 0.861, y: 0.932, uncertainty: 0.025 },
    { id: 7, x: 0.857, y: 0.928, uncertainty: 0.028 },
  ],
  dominatedSolutions: [
    { id: 1, x: 0.725, y: 0.853, uncertainty: 0.05 },
    { id: 2, x: 0.852, y: 0.921, uncertainty: 0.04 },
    { id: 3, x: 0.789, y: 0.887, uncertainty: 0.045 },
    { id: 4, x: 0.835, y: 0.902, uncertainty: 0.035 },
    { id: 5, x: 0.848, y: 0.915, uncertainty: 0.03 },
    { id: 9, x: 0.863, y: 0.934, uncertainty: 0.022 },
  ],
}

export default function ExperimentVisualizationPage() {
  const params = useParams()
  const { id } = params
  const [experiment, setExperiment] = useState(mockExperiment)
  const [activeTab, setActiveTab] = useState("progress")
  const [selectedObjective, setSelectedObjective] = useState<string>(experiment.objectives[0]?.name || "")
  const [selectedParameter, setSelectedParameter] = useState<string>(experiment.parameters[0]?.name || "")

  // 实际应用中应从API获取实验数据
  useEffect(() => {
    // 模拟API调用
    // fetch(`/api/bo/experiments/${id}`)
    //   .then(response => response.json())
    //   .then(data => {
    //     setExperiment(data)
    //     setSelectedObjective(data.objectives[0]?.name || "")
    //     setSelectedParameter(data.parameters[0]?.name || "")
    //   })
  }, [id])

  // 获取优化进度数据
  const getProgressData = () => {
    return experiment.results.map((result) => ({
      iteration: result.iteration,
      value: result.objectives[selectedObjective] || 0,
    }))
  }

  // 获取参数变化数据
  const getParameterVariationData = () => {
    return experiment.results.map((result) => ({
      iteration: result.iteration,
      value: result.parameters[selectedParameter] || 0,
    }))
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <Button variant="outline" size="icon" asChild className="mr-4">
            <Link href={`/experiments/${id}`}>
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </Button>
          <div>
            <h1 className="text-3xl font-bold">{experiment.name}</h1>
            <p className="text-muted-foreground">可视化分析</p>
          </div>
        </div>

        <Button variant="outline">
          <Download className="mr-2 h-4 w-4" />
          导出图表
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList>
          <TabsTrigger value="progress">优化进度</TabsTrigger>
          <TabsTrigger value="parameters">参数变化</TabsTrigger>
          <TabsTrigger value="pareto">Pareto前沿</TabsTrigger>
        </TabsList>

        <TabsContent value="progress">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>优化进度折线图</CardTitle>
                  <CardDescription>目标值随迭代次数的变化趋势</CardDescription>
                </div>
                <div className="w-[200px]">
                  <Select value={selectedObjective} onValueChange={setSelectedObjective}>
                    <SelectTrigger>
                      <SelectValue placeholder="选择目标" />
                    </SelectTrigger>
                    <SelectContent>
                      {experiment.objectives.map((obj) => (
                        <SelectItem key={obj.name} value={obj.name}>
                          {obj.name} ({obj.unit || ""})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent className="h-[500px]">
              <OptimizationProgressChart
                data={getProgressData()}
                objective={experiment.objectives.find((obj) => obj.name === selectedObjective)}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="parameters">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>参数变化曲线</CardTitle>
                  <CardDescription>参数值随迭代次数的变化趋势</CardDescription>
                </div>
                <div className="w-[200px]">
                  <Select value={selectedParameter} onValueChange={setSelectedParameter}>
                    <SelectTrigger>
                      <SelectValue placeholder="选择参数" />
                    </SelectTrigger>
                    <SelectContent>
                      {experiment.parameters.map((param) => (
                        <SelectItem key={param.name} value={param.name}>
                          {param.name} ({param.unit || ""})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent className="h-[500px]">
              <ParameterVariationChart
                data={getParameterVariationData()}
                parameter={experiment.parameters.find((param) => param.name === selectedParameter)}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="pareto">
          <Card>
            <CardHeader>
              <CardTitle>Pareto前沿可视化</CardTitle>
              <CardDescription>多目标优化的Pareto最优解集</CardDescription>
            </CardHeader>
            <CardContent className="h-[500px]">
              <ParetoFrontVisualization
                paretoFront={experiment.paretoFront}
                nonDominatedSolutions={experiment.nonDominatedSolutions}
                dominatedSolutions={experiment.dominatedSolutions}
                objective1={{
                  name: experiment.objectives[0]?.name || "目标1",
                  type: "maximize",
                }}
                objective2={{
                  name: experiment.objectives[1]?.name || "目标2",
                  type: "maximize",
                }}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
