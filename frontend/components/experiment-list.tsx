"use client"

import { useState, useEffect } from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
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
import { Search, MoreVertical, Plus, Copy, Trash2, ArrowUpDown } from "lucide-react"
import { useRouter } from "next/navigation"
import { useToast } from "@/hooks/use-toast"

interface Experiment {
  id: string
  name: string
  status: "running" | "completed" | "failed" | "draft"
  objectives: number
  parameters: number
  createdAt: string
  bestKPI?: number
}

export default function ExperimentList() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [currentPage, setCurrentPage] = useState(1)
  const [sortField, setSortField] = useState<keyof Experiment>("createdAt")
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc")
  const router = useRouter()
  const { toast } = useToast()
  const itemsPerPage = 10

  // Mock data for demonstration
  useEffect(() => {
    // In a real application, this would be an API call
    // GET /api/bo/experiments
    const mockExperiments: Experiment[] = [
      {
        id: "1",
        name: "催化剂优化实验",
        status: "completed",
        objectives: 2,
        parameters: 5,
        createdAt: "2023-04-15T10:30:00Z",
        bestKPI: 0.89,
      },
      {
        id: "2",
        name: "反应温度优化",
        status: "running",
        objectives: 1,
        parameters: 3,
        createdAt: "2023-04-18T14:20:00Z",
        bestKPI: 0.76,
      },
      {
        id: "3",
        name: "材料配方测试",
        status: "draft",
        objectives: 3,
        parameters: 8,
        createdAt: "2023-04-20T09:15:00Z",
      },
      {
        id: "4",
        name: "电池寿命优化",
        status: "failed",
        objectives: 2,
        parameters: 4,
        createdAt: "2023-04-10T16:45:00Z",
        bestKPI: 0.45,
      },
      {
        id: "5",
        name: "药物配方筛选",
        status: "completed",
        objectives: 4,
        parameters: 10,
        createdAt: "2023-04-05T11:20:00Z",
        bestKPI: 0.92,
      },
    ]

    setExperiments(mockExperiments)
    setLoading(false)
  }, [])

  const handleSort = (field: keyof Experiment) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortField(field)
      setSortDirection("asc")
    }
  }

  const handleDelete = async (id: string) => {
    // In a real application, this would be an API call
    // DELETE /api/bo/experiments/{id}
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 500))

      setExperiments(experiments.filter((exp) => exp.id !== id))
      toast({
        title: "实验已删除",
        description: "实验已成功删除",
      })
    } catch (error) {
      toast({
        title: "删除失败",
        description: "删除实验时出错，请重试",
        variant: "destructive",
      })
    }
  }

  const handleClone = async (id: string) => {
    // In a real application, this would be a combination of API calls
    // GET + POST to clone the experiment
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 500))

      const originalExperiment = experiments.find((exp) => exp.id === id)
      if (!originalExperiment) return

      const newId = (Math.max(...experiments.map((e) => Number.parseInt(e.id))) + 1).toString()
      const clonedExperiment: Experiment = {
        ...originalExperiment,
        id: newId,
        name: `${originalExperiment.name} (复制)`,
        status: "draft",
        createdAt: new Date().toISOString(),
      }

      setExperiments([...experiments, clonedExperiment])
      toast({
        title: "实验已复制",
        description: "实验已成功复制，您可以进行编辑",
      })
    } catch (error) {
      toast({
        title: "复制失败",
        description: "复制实验时出错，请重试",
        variant: "destructive",
      })
    }
  }

  // Filter experiments based on search query
  const filteredExperiments = experiments.filter((exp) => exp.name.toLowerCase().includes(searchQuery.toLowerCase()))

  // Sort experiments
  const sortedExperiments = [...filteredExperiments].sort((a, b) => {
    if (a[sortField] < b[sortField]) return sortDirection === "asc" ? -1 : 1
    if (a[sortField] > b[sortField]) return sortDirection === "asc" ? 1 : -1
    return 0
  })

  // Paginate experiments
  const totalPages = Math.ceil(sortedExperiments.length / itemsPerPage)
  const paginatedExperiments = sortedExperiments.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage)

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "running":
        return (
          <Badge variant="default" className="bg-blue-500">
            进行中
          </Badge>
        )
      case "completed":
        return (
          <Badge variant="default" className="bg-green-500">
            已完成
          </Badge>
        )
      case "failed":
        return <Badge variant="destructive">失败</Badge>
      case "draft":
        return <Badge variant="outline">草稿</Badge>
      default:
        return <Badge variant="secondary">{status}</Badge>
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("zh-CN", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    })
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>实验列表</CardTitle>
            <CardDescription>管理您的所有优化实验</CardDescription>
          </div>
          <Button onClick={() => router.push("/experiments/new")} className="flex items-center gap-1">
            <Plus size={16} />
            <span>创建实验</span>
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-center mb-4">
          <div className="relative flex-1">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="搜索实验..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8"
            />
          </div>
        </div>

        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[250px]">
                  <div className="flex items-center cursor-pointer" onClick={() => handleSort("name")}>
                    实验名称
                    <ArrowUpDown size={16} className="ml-1" />
                  </div>
                </TableHead>
                <TableHead>状态</TableHead>
                <TableHead className="text-center">目标数</TableHead>
                <TableHead className="text-center">参数数</TableHead>
                <TableHead>
                  <div className="flex items-center cursor-pointer" onClick={() => handleSort("createdAt")}>
                    创建时间
                    <ArrowUpDown size={16} className="ml-1" />
                  </div>
                </TableHead>
                <TableHead className="text-center">最佳KPI</TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-10">
                    加载中...
                  </TableCell>
                </TableRow>
              ) : paginatedExperiments.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-10">
                    没有找到实验。{searchQuery ? "尝试不同的搜索词或" : ""}
                    <Button variant="link" onClick={() => router.push("/experiments/new")} className="px-1">
                      创建一个新实验
                    </Button>
                  </TableCell>
                </TableRow>
              ) : (
                paginatedExperiments.map((experiment) => (
                  <TableRow key={experiment.id}>
                    <TableCell
                      className="font-medium cursor-pointer hover:underline"
                      onClick={() => router.push(`/experiments/${experiment.id}`)}
                    >
                      {experiment.name}
                    </TableCell>
                    <TableCell>{getStatusBadge(experiment.status)}</TableCell>
                    <TableCell className="text-center">{experiment.objectives}</TableCell>
                    <TableCell className="text-center">{experiment.parameters}</TableCell>
                    <TableCell>{formatDate(experiment.createdAt)}</TableCell>
                    <TableCell className="text-center">
                      {experiment.bestKPI !== undefined ? experiment.bestKPI.toFixed(2) : "-"}
                    </TableCell>
                    <TableCell className="text-right">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon">
                            <MoreVertical size={16} />
                            <span className="sr-only">打开菜单</span>
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => router.push(`/experiments/${experiment.id}`)}>
                            查看详情
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => handleClone(experiment.id)}>
                            <Copy size={16} className="mr-2" />
                            复制实验
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => handleDelete(experiment.id)} className="text-red-600">
                            <Trash2 size={16} className="mr-2" />
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
        </div>

        {totalPages > 1 && (
          <Pagination className="mt-4">
            <PaginationContent>
              <PaginationItem>
                <PaginationPrevious
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  className={currentPage === 1 ? "pointer-events-none opacity-50" : "cursor-pointer"}
                />
              </PaginationItem>

              {Array.from({ length: totalPages }).map((_, i) => (
                <PaginationItem key={i}>
                  <PaginationLink onClick={() => setCurrentPage(i + 1)} isActive={currentPage === i + 1}>
                    {i + 1}
                  </PaginationLink>
                </PaginationItem>
              ))}

              <PaginationItem>
                <PaginationNext
                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                  className={currentPage === totalPages ? "pointer-events-none opacity-50" : "cursor-pointer"}
                />
              </PaginationItem>
            </PaginationContent>
          </Pagination>
        )}
      </CardContent>
    </Card>
  )
}
