"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ArrowLeft, ArrowRight, Check } from "lucide-react"
import Link from "next/link"
import { useToast } from "@/components/ui/use-toast"
import { ParameterList } from "@/components/parameter-list"
import { ParameterForm } from "@/components/parameter-form"
import { ObjectiveList } from "@/components/objective-list"
import { ObjectiveForm } from "@/components/objective-form"
import { ConstraintList } from "@/components/constraint-list"
import { ConstraintForm } from "@/components/constraint-form"
import { AlgorithmSelector } from "@/components/algorithm-selector"

export default function NewExperimentPage() {
  const router = useRouter()
  const { toast } = useToast()
  const [activeTab, setActiveTab] = useState("basic")
  const [isSubmitting, setIsSubmitting] = useState(false)

  // 基本信息
  const [basicInfo, setBasicInfo] = useState({
    name: "",
    description: "",
  })

  // 参数空间
  const [parameters, setParameters] = useState<any[]>([])
  const [editingParameter, setEditingParameter] = useState<any | null>(null)

  // 目标函数
  const [objectives, setObjectives] = useState<any[]>([])
  const [editingObjective, setEditingObjective] = useState<any | null>(null)

  // 约束条件
  const [constraints, setConstraints] = useState<any[]>([])
  const [editingConstraint, setEditingConstraint] = useState<any | null>(null)

  // 算法配置
  const [algorithm, setAlgorithm] = useState<any>(null)

  // 处理基本信息变更
  const handleBasicInfoChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target
    setBasicInfo((prev) => ({ ...prev, [name]: value }))
  }

  // 处理参数添加/编辑
  const handleAddParameter = (parameter: any) => {
    if (editingParameter && editingParameter.id) {
      setParameters(parameters.map((p) => (p.id === parameter.id ? parameter : p)))
      setEditingParameter(null)
    } else {
      setParameters([...parameters, { ...parameter, id: Date.now().toString() }])
    }
  }

  // 处理目标添加/编辑
  const handleAddObjective = (objective: any) => {
    if (editingObjective && editingObjective.id) {
      setObjectives(objectives.map((o) => (o.id === objective.id ? objective : o)))
      setEditingObjective(null)
    } else {
      setObjectives([...objectives, { ...objective, id: Date.now().toString() }])
    }
  }

  // 处理约束添加/编辑
  const handleAddConstraint = (constraint: any) => {
    if (editingConstraint && editingConstraint.id) {
      setConstraints(constraints.map((c) => (c.id === constraint.id ? constraint : c)))
      setEditingConstraint(null)
    } else {
      setConstraints([...constraints, { ...constraint, id: Date.now().toString() }])
    }
  }

  // 处理算法选择
  const handleAlgorithmSelect = (selectedAlgorithm: any) => {
    setAlgorithm(selectedAlgorithm)
    toast({
      title: "算法已选择",
      description: `已选择 ${selectedAlgorithm.type} 算法`,
    })
  }

  // 验证当前步骤
  const validateCurrentStep = () => {
    switch (activeTab) {
      case "basic":
        return basicInfo.name.trim() !== ""
      case "parameters":
        return parameters.length > 0
      case "objectives":
        return objectives.length > 0
      case "constraints":
        return true // 约束是可选的
      case "algorithm":
        return algorithm !== null
      default:
        return false
    }
  }

  // 处理下一步
  const handleNextStep = () => {
    if (!validateCurrentStep()) {
      toast({
        title: "请完成当前步骤",
        description: "请填写所有必要信息后再继续",
        variant: "destructive",
      })
      return
    }

    switch (activeTab) {
      case "basic":
        setActiveTab("parameters")
        break
      case "parameters":
        setActiveTab("objectives")
        break
      case "objectives":
        setActiveTab("constraints")
        break
      case "constraints":
        setActiveTab("algorithm")
        break
      case "algorithm":
        handleSubmit()
        break
    }
  }

  // 处理上一步
  const handlePreviousStep = () => {
    switch (activeTab) {
      case "parameters":
        setActiveTab("basic")
        break
      case "objectives":
        setActiveTab("parameters")
        break
      case "constraints":
        setActiveTab("objectives")
        break
      case "algorithm":
        setActiveTab("constraints")
        break
    }
  }

  // 提交表单
  const handleSubmit = async () => {
    if (!validateCurrentStep()) {
      toast({
        title: "请完成所有步骤",
        description: "请填写所有必要信息后再提交",
        variant: "destructive",
      })
      return
    }

    setIsSubmitting(true)

    // 构建实验配置
    const experimentConfig = {
      name: basicInfo.name,
      description: basicInfo.description,
      parameters,
      objectives,
      constraints,
      algorithm,
    }

    // 实际应用中应调用API
    // try {
    //   const response = await fetch('/api/bo/experiments', {
    //     method: 'POST',
    //     headers: { 'Content-Type': 'application/json' },
    //     body: JSON.stringify(experimentConfig)
    //   })
    //
    //   const data = await response.json()
    //
    //   if (response.ok) {
    //     toast({
    //       title: "实验创建成功",
    //       description: `实验 "${basicInfo.name}" 已成功创建`
    //     })
    //     router.push(`/experiments/${data.id}`)
    //   } else {
    //     throw new Error(data.message || '创建实验失败')
    //   }
    // } catch (error) {
    //   toast({
    //     title: "创建失败",
    //     description: error.message,
    //     variant: "destructive"
    //   })
    // } finally {
    //   setIsSubmitting(false)
    // }

    // 模拟API调用
    setTimeout(() => {
      toast({
        title: "实验创建成功",
        description: `实验 "${basicInfo.name}" 已成功创建`,
      })
      router.push("/experiments")
      setIsSubmitting(false)
    }, 1500)
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center mb-6">
        <Button variant="outline" size="icon" asChild className="mr-4">
          <Link href="/experiments">
            <ArrowLeft className="h-4 w-4" />
          </Link>
        </Button>
        <h1 className="text-3xl font-bold">创建新实验</h1>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="basic">基本信息</TabsTrigger>
          <TabsTrigger value="parameters">参数空间</TabsTrigger>
          <TabsTrigger value="objectives">目标函数</TabsTrigger>
          <TabsTrigger value="constraints">约束条件</TabsTrigger>
          <TabsTrigger value="algorithm">算法配置</TabsTrigger>
        </TabsList>
        <TabsContent value="basic" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>基本信息</CardTitle>
              <CardDescription>填写实验的基本信息</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="grid gap-2">
                <label htmlFor="name">实验名称</label>
                <Input id="name" name="name" value={basicInfo.name} onChange={handleBasicInfoChange} />
              </div>
              <div className="grid gap-2">
                <label htmlFor="description">实验描述</label>
                <Textarea
                  id="description"
                  name="description"
                  value={basicInfo.description}
                  onChange={handleBasicInfoChange}
                />
              </div>
            </CardContent>
            <CardFooter className="flex justify-end">
              <Button onClick={() => setActiveTab("parameters")}>
                下一步 <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
        <TabsContent value="parameters" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>参数空间</CardTitle>
              <CardDescription>定义实验的参数空间</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <ParameterList
                parameters={parameters}
                setEditingParameter={setEditingParameter}
                setParameters={setParameters}
              />
              <ParameterForm
                onSubmit={handleAddParameter}
                editingParameter={editingParameter}
                setEditingParameter={setEditingParameter}
              />
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={handlePreviousStep}>
                <ArrowLeft className="mr-2 h-4 w-4" /> 上一步
              </Button>
              <Button onClick={handleNextStep}>
                下一步 <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
        <TabsContent value="objectives" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>目标函数</CardTitle>
              <CardDescription>定义实验的目标函数</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <ObjectiveList
                objectives={objectives}
                setEditingObjective={setEditingObjective}
                setObjectives={setObjectives}
              />
              <ObjectiveForm
                onSubmit={handleAddObjective}
                editingObjective={editingObjective}
                setEditingObjective={setEditingObjective}
              />
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={handlePreviousStep}>
                <ArrowLeft className="mr-2 h-4 w-4" /> 上一步
              </Button>
              <Button onClick={handleNextStep}>
                下一步 <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
        <TabsContent value="constraints" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>约束条件</CardTitle>
              <CardDescription>定义实验的约束条件 (可选)</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <ConstraintList
                constraints={constraints}
                setEditingConstraint={setEditingConstraint}
                setConstraints={setConstraints}
              />
              <ConstraintForm
                onSubmit={handleAddConstraint}
                editingConstraint={editingConstraint}
                setEditingConstraint={setEditingConstraint}
              />
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={handlePreviousStep}>
                <ArrowLeft className="mr-2 h-4 w-4" /> 上一步
              </Button>
              <Button onClick={handleNextStep}>
                下一步 <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
        <TabsContent value="algorithm" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>算法配置</CardTitle>
              <CardDescription>选择优化算法</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <AlgorithmSelector onAlgorithmSelect={handleAlgorithmSelect} />
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={handlePreviousStep}>
                <ArrowLeft className="mr-2 h-4 w-4" /> 上一步
              </Button>
              <Button onClick={handleNextStep} disabled={isSubmitting}>
                {isSubmitting ? "提交中..." : "提交"}
                {isSubmitting ? null : <Check className="ml-2 h-4 w-4" />}
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
