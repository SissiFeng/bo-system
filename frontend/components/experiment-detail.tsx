"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { ArrowLeft, Trash2, Copy, Play, Pause } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

interface Parameter {
  id: string
  name: string
  type: "continuous" | "discrete" | "categorical"
  min?: number
  max?: number
  options?: string[]
  value?: number | string
}

interface Objective {
  id: string
  name: string
  direction: "maximize" | "minimize"
  value?: number
}

interface Constraint {
  id: string
  name: string
  expression: string
  isSatisfied?: boolean
}

interface ExperimentResult {
  id: string
  parameters: Record<string, number | string>
  objectives: Record<string, number>
  constraints: Record<string, boolean>
  timestamp: string
  isRecommended: boolean
}

interface Experiment {
  id: string
  name: string
  description: string
  status: "running" | "completed" | "failed" | "draft" | "paused"
  parameters: Parameter[]
  objectives: Objective[]
  constraints: Constraint[]
  results: ExperimentResult[]
  createdAt: string
  updatedAt: string
  currentIteration: number
  maxIterations: number
}

export default function ExperimentDetail({ experimentId }: { experimentId: string }) {
  const [experiment, setExperiment] = useState<Experiment | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('overview')
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const router = useRouter()
  const { toast } = useToast()

  // Mock data for demonstration
  useEffect(() => {
    // In a real application, this would be an API call
    // GET /api/bo/experiments/{id}
    const fetchExperiment = async () => {
      try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 800))
        
        // Mock experiment data
        const mockExperiment: Experiment = {
          id: experimentId,
          name: '催化剂优化实验',
          description: '优化催化剂配方以提高转化率和选择性',
          status: 'running',
          parameters: [
            { id: 'p1', name: '温度 (°C)', type: 'continuous', min: 50, max: 150, value: 95 },
            { id: 'p2', name: '压力 (bar)', type: 'continuous', min: 1, max: 10, value: 5.5 },
            { id: 'p3', name: '催化剂类型', type: 'categorical', options: ['A型', 'B型', 'C型'], value: 'B型' },
            { id: 'p4', name: '反应时间 (min)', type: 'continuous', min: 30, max: 180, value: 120 },
            { id: 'p5', name: '搅拌速度 (rpm)', type: 'continuous', min: 200, max: 800, value: 450 }
          ],
          objectives: [
            { id: 'o1', name: '转化率 (%)', direction: 'maximize', value: 78.5 },
            { id: 'o2', name: '选择性 (%)', direction: 'maximize', value: 92.3 }
          ],
          constraints: [
            { id: 'c1', name: '能耗限制', expression: '温度 * 反应时间 < 15000', isSatisfied: true },
            { id: 'c2', name: '安全限制', expression: '温度 < 160 && 压力 < 12', isSatisfied: true }
          ],
          results: generateMockResults(),
          createdAt: '2023-04-15T10:30:00Z',
          updatedAt: '2023-04-18T14:20:00Z',
          currentIteration: 12,
          maxIterations: 30
        }
        
        setExperiment(mockExperiment)
      } catch (error) {
        toast({
          title: "加载失败",
          description: "无法加载实验详情，请重试",
          variant: "destructive"
        })
      } finally {
        setLoading(false)
      }
    }
    
    fetchExperiment()
  }, [experimentId, toast])

  function generateMockResults(): ExperimentResult[] {
    const results: ExperimentResult[] = []
    
    for (let i = 1; i <= 12; i++) {
      results.push({
        id: `r${i}`,
        parameters: {
          '温度 (°C)': 80 + Math.random() * 40,
          '压力 (bar)': 2 + Math.random() * 6,
          '催化剂类型': ['A型', 'B型', 'C型'][Math.floor(Math.random() * 3)],
          '反应时间 (min)': 60 + Math.random() * 90,
          '搅拌速度 (rpm)': 300 + Math.random() * 400
        },
        objectives: {
          '转化率 (%)': 60 + Math.random() * 30,
          '选择性 (%)': 80 + Math.random() * 15
        },
        constraints: {
          '能耗限制': Math.random() > 0.1,
          '安全限制': true
        },
        timestamp: new Date(Date.now() - (12 - i) * 3600000).toISOString(),
        isRecommended: i === 12
      })
    }
    
    return results
  }

  const handleDelete = async () => {
    // In a real application, this would be an API call
    // DELETE /api/bo/experiments/{id}
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500))
      
      toast({
        title: "实验已删除",
        description: "实验已成功删除",
      })
      
      router.push('/experiments')
    } catch (error) {
      toast({
        title: "删除失败",
        description: "删除实验时出错，请重试",
        variant: "destructive"
      })
    }
  }

  const handleClone = async () => {
    // In a real application, this would be a combination of API calls
    // GET + POST to clone the experiment
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500))
      
      toast({
        title: "实验已复制",
        description: "实验已成功复制，您可以进行编辑",
      })
      
      // Redirect to the new experiment (in a real app, we'd get the new ID from the API)
      router.push('/experiments/new-cloned-id')
    } catch (error) {
      toast({
        title: "复制失败",
        description: "复制实验时出错，请重试",
        variant: "destructive"
      })
    }
  }

  const handleToggleRunning = async () => {
    if (!experiment) return
    
    const newStatus = experiment.status === 'running' ? 'paused' : 'running'
    
    // In a real application, this would be an API call
    // PUT /api/bo/experiments/{id} with { status: newStatus }
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 300))
      
      setExperiment({
        ...experiment,
        status: newStatus
      })
      
      toast({
        title: newStatus === 'running' ? "实验已启动" : "实验已暂停",
        description: newStatus === 'running' ? "实验正在运行中" : "实验已暂停运行",
      })
    } catch (error) {
      toast({
        title: "操作失败",
        description: "更改实验状态时出错，请重试",
        variant: "destructive"
      })
    }
  }

  const handleSubmitResult = () => {
    router.push(`/experiments/${experimentId}/submit-result`)
  }

  const handleGetNextPoint = async () => {
    // In a real application, this would be an API call
    // POST /api/bo/experiments/{id}/next
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 800))
      
      toast({
        title: "已生成新的设计点",
        description: "系统已推荐下一个实验设计点",
      })
      
      router.push(`/experiments/${experimentId}/next-point`)
    } catch (error) {
      toast({
        title: "生成失败",
        description: "生成下一个设计点时出错，请重试",
        variant: "destructive"
      })
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'running':
        return <Badge variant="default" className="bg-blue-500">进行中</Badge>
      case 'completed':
        return <Badge variant="default" className="bg-green-500">已完成</Badge>
      case 'failed':
        return <Badge variant="destructive">失败</Badge>
      case 'draft':
        return <Badge variant="outline">草稿</Badge>
      case 'paused':
        return <Badge variant="secondary">已暂停</Badge>
      default:
        return <Badge variant="secondary">{status}</Badge>
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('zh-CN', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  if (loading) {
    return (
      <Card className="w-full">
        <CardContent className="pt-6">
          <div className="flex justify-center items-center h-64">
            <p>加载中...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!experiment) {
    return (
      <Card className="w-full">
        <CardContent className="pt-6">
          <div className="flex justify-center items-center h-64">
            <p>未找到实验</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <Button 
          variant="ghost" 
          onClick={() => router.push('/experiments')}
          className="flex items-center gap-1"
        >
          <ArrowLeft size={16} />
          <span>返回实验列表</span>
        </Button>
        
        <div className="flex items-center gap-2">
          <Button 
            variant={experiment.status === 'running' ? "outline" : "default"}
            onClick={handleToggleRunning}
            className="flex items-center gap-1"
          >
            {experiment.status === 'running' ? (
              <>
                <Pause size={16} />
                <span>暂停实验</span>
              </>
            ) : (
              <>
                <Play size={16} />
                <span>启动实验</span>
              </>
            )}
          </Button>
          
          <Button 
            variant="outline" 
            onClick={handleClone}
            className="flex items-center gap-1"
          >
            <Copy size={16} />
            <span>复制实验</span>
          </Button>
          
          <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
            <AlertDialogTrigger asChild>
              <Button 
                variant="outline" 
                className="flex items-center gap-1 text-red-600 border-red-200 hover:bg-red-50"
              >
                <Trash2 size={16} />
                <span>删除实验</span>
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>确认删除实验</AlertDialogTitle>
                <AlertDialogDescription>
                  您确定要删除此实验吗？此操作无法撤销，所有相关数据将被永久删除。
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>取消</AlertDialogCancel>
                <AlertDialogAction onClick={handleDelete} className="bg-red-600 hover:bg-red-700">
                  确认删除
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>
      
      <Card className="w-full">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl">{experiment.name}</CardTitle>
              <CardDescription className="mt-2">{experiment.description}</CardDescription>
            </div>
            <div className="flex flex-col items-end gap-1">
              {getStatusBadge(experiment.status)}
              <span className="text-sm text-muted-foreground">
                迭代: {experiment.currentIteration} / {experiment.maxIterations}
              </span>
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid grid-cols-4 mb-6">
              <TabsTrigger value="overview">概览</TabsTrigger>
              <TabsTrigger value="results">实验结果</TabsTrigger>
              <TabsTrigger value="visualization">数据可视化</TabsTrigger>
              <TabsTrigger value="settings">设置</TabsTrigger>
            </TabsList>
            
            <TabsContent value="overview" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">实验信息</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <dl className="space-y-4">
                      <div className="flex justify-between">
                        <dt className="font-medium text-muted-foreground">创建时间</dt>
                        <dd>{formatDate(experiment.createdAt)}</dd>
                      </div>
                      <div className="flex justify-between">
                        <dt className="font-medium text-muted-foreground">最后更新</dt>
                        <dd>{formatDate(experiment.updatedAt)}</dd>
                      </div>
                      <div className="flex justify-between">
                        <dt className="font-medium text-muted-foreground">参数数量</dt>
                        <dd>{experiment.parameters.length}</dd>
                      </div>
                      <div\
