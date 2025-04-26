"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination"
import { useToast } from "@/components/ui/use-toast"
import { Copy, FileEdit, MoreHorizontal, Plus, Search, Trash2, ArrowUpDown, BarChart } from "lucide-react"
import Link from "next/link"

// 模拟实验数据，实际应用中应从API获取
const mockExperiments = [
  {
    id: "exp-001",
    name: "催化剂优化实验-A",
    status: "completed",
    created_at: "2023-06-15T10:30:00Z",
    updated_at: "2023-06-16T14:20:00Z",
    best_kpi: { 转化率: 0.85, 选择性: 0.92 },
    iterations: 25,
    parameters_count: 4,
    objectives_count: 2,
  },
  {
    id: "exp-002",
    name: "反应温度优化",
    status: "running",
    created_at: "2023-06-18T09:15:00Z",
    updated_at: "2023-06-18T15:45:00Z",
    best_kpi: { 转化率: 0.78, 选择性: 0.88 },
    iterations: 12,
    parameters_count: 3,
    objectives_count: 2,
  },
  {
    id: "exp-003",
    name: "材料配比测试",
    status: "paused",
    created_at: "2023-06-10T11:20:00Z",
    updated_at: "2023-06-12T16:30:00Z",
    best_kpi: { 硬度: 72.5, 韧性: 0.65 },
    iterations: 8,
    parameters_count: 5,
    objectives_count: 2,
  },
  {
    id: "exp-004",
    name: "电池配方优化",
    status: "completed",
    created_at: "2023-05-20T08:45:00Z",
    updated_at: "2023-05-25T17:10:00Z",
    best_kpi: { 容量: 3450, 循环寿命: 850 },
    iterations: 30,
    parameters_count: 6,
    objectives_count: 3,
  },
  {
    id: "exp-005",
    name: "光催化剂筛选",
    status: "draft",
    created_at: "2023-06-19T14:25:00Z",
    updated_at: "2023-06-19T14:25:00Z",
    best_kpi: {},
    iterations: 0,
    parameters_count: 4,
    objectives_count: 2,
  },
]

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState(mockExperiments)
  const [searchTerm, setSearchTerm] = useState("")
  const [currentPage, setCurrentPage] = useState(1)
  const [sortField, setSortField] = useState("updated_at")
  const [sortDirection, setSortDirection] = useState("desc")
  const { toast } = useToast()

  const itemsPerPage = 10
  const filteredExperiments = experiments.filter(
    (exp) =>
      exp.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      exp.id.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  // 排序实验
  const sortedExperiments = [...filteredExperiments].sort((a, b) => {
    if (sortField === "updated_at" || sortField === "created_at") {
      return sortDirection === "asc"
        ? new Date(a[sortField]).getTime() - new Date(b[sortField]).getTime()
        : new Date(b[sortField]).getTime() - new Date(a[sortField]).getTime()
    }

    if (sortField === "name") {
      return sortDirection === "asc" ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name)
    }

    return sortDirection === "asc" ? a[sortField] - b[sortField] : b[sortField] - a[sortField]
  })

  const paginatedExperiments = sortedExperiments.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage)

  const totalPages = Math.ceil(filteredExperiments.length / itemsPerPage)

  // 处理排序
  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortField(field)
      setSortDirection("asc")
    }
  }

  // 删除实验
  const handleDelete = (id: string) => {
    // 实际应用中应调用API
    // fetch(`/api/bo/experiments/${id}`, { method: 'DELETE' })
    //   .then(response => {
    //     if (response.ok) {
    //       setExperiments(experiments.filter(exp => exp.id !== id))
    //       toast({ title: "实验已删除", description: `实验 ${id} 已成功删除` })
    //     }
    //   })

    // 模拟删除
    setExperiments(experiments.filter((exp) => exp.id !== id))
    toast({
      title: "实验已删除",
      description: `实验 ${id} 已成功删除`,
    })
  }

  // 克隆实验
  const handleClone = (id: string) => {
    const expToClone = experiments.find((exp) => exp.id === id)
    if (!expToClone) return

    // 实际应用中应调用API
    // 1. 获取原实验详情
    // 2. 创建新实验，复制原实验配置

    // 模拟克隆
    const clonedExp = {
      ...expToClone,
      id: `exp-${Math.floor(Math.random() * 1000)
        .toString()
        .padStart(3, "0")}`,
      name: `${expToClone.name} (复制)`,
      status: "draft",
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      iterations: 0,
      best_kpi: {},
    }

    setExperiments([...experiments, clonedExp])
    toast({
      title: "实验已克隆",
      description: `已创建实验 ${clonedExp.name}`,
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

  // 格式化日期
  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleString("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    })
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">实验管理</h1>
        <Button asChild>
          <Link href="/experiments/new">
            <Plus className="mr-2 h-4 w-4" />
            创建实验
          </Link>
        </Button>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle>实验列表</CardTitle>
          <CardDescription>管理所有贝叶斯优化实验</CardDescription>
          <div className="flex items-center mt-2">
            <div className="relative flex-1 max-w-sm">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="搜索实验..."
                className="pl-8"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[100px]">ID</TableHead>
                <TableHead className="cursor-pointer" onClick={() => handleSort("name")}>
                  <div className="flex items-center">
                    实验名称
                    <ArrowUpDown className="ml-2 h-4 w-4" />
                  </div>
                </TableHead>
                <TableHead>状态</TableHead>
                <TableHead className="cursor-pointer" onClick={() => handleSort("updated_at")}>
                  <div className="flex items-center">
                    更新时间
                    <ArrowUpDown className="ml-2 h-4 w-4" />
                  </div>
                </TableHead>
                <TableHead>最佳KPI</TableHead>
                <TableHead className="cursor-pointer" onClick={() => handleSort("iterations")}>
                  <div className="flex items-center">
                    迭代次数
                    <ArrowUpDown className="ml-2 h-4 w-4" />
                  </div>
                </TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {paginatedExperiments.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-8 text-muted-foreground">
                    未找到实验数据
                  </TableCell>
                </TableRow>
              ) : (
                paginatedExperiments.map((experiment) => (
                  <TableRow key={experiment.id}>
                    <TableCell className="font-mono text-xs">{experiment.id}</TableCell>
                    <TableCell>
                      <Link href={`/experiments/${experiment.id}`} className="font-medium hover:underline">
                        {experiment.name}
                      </Link>
                      <div className="text-xs text-muted-foreground">
                        {experiment.parameters_count}个参数 · {experiment.objectives_count}个目标
                      </div>
                    </TableCell>
                    <TableCell>{getStatusBadge(experiment.status)}</TableCell>
                    <TableCell>
                      <div className="text-sm">{formatDate(experiment.updated_at)}</div>
                      <div className="text-xs text-muted-foreground">创建: {formatDate(experiment.created_at)}</div>
                    </TableCell>
                    <TableCell>
                      {Object.entries(experiment.best_kpi).length > 0 ? (
                        <div className="space-y-1">
                          {Object.entries(experiment.best_kpi).map(([key, value]) => (
                            <div key={key} className="text-sm">
                              {key}: <span className="font-medium">{value}</span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <span className="text-muted-foreground text-sm">暂无数据</span>
                      )}
                    </TableCell>
                    <TableCell>{experiment.iterations}</TableCell>
                    <TableCell className="text-right">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon">
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem asChild>
                            <Link href={`/experiments/${experiment.id}`}>
                              <FileEdit className="mr-2 h-4 w-4" />
                              查看详情
                            </Link>
                          </DropdownMenuItem>
                          <DropdownMenuItem asChild>
                            <Link href={`/experiments/${experiment.id}/visualization`}>
                              <BarChart className="mr-2 h-4 w-4" />
                              查看分析
                            </Link>
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => handleClone(experiment.id)}>
                            <Copy className="mr-2 h-4 w-4" />
                            克隆实验
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={() => handleDelete(experiment.id)}
                            className="text-red-600 focus:text-red-600"
                          >
                            <Trash2 className="mr-2 h-4 w-4" />
                            删除实验
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>

          {totalPages > 1 && (
            <Pagination className="mt-4">
              <PaginationContent>
                <PaginationItem>
                  <PaginationPrevious
                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                    className={currentPage === 1 ? "pointer-events-none opacity-50" : ""}
                  />
                </PaginationItem>
                {Array.from({ length: totalPages }).map((_, i) => (
                  <PaginationItem key={i}>
                    <PaginationLink isActive={currentPage === i + 1} onClick={() => setCurrentPage(i + 1)}>
                      {i + 1}
                    </PaginationLink>
                  </PaginationItem>
                ))}
                <PaginationItem>
                  <PaginationNext
                    onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                    className={currentPage === totalPages ? "pointer-events-none opacity-50" : ""}
                  />
                </PaginationItem>
              </PaginationContent>
            </Pagination>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
