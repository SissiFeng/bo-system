"use client"

import { memo } from "react"
import { Handle, Position } from "reactflow"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { BarChart } from "lucide-react"

export const AnalysisNode = memo(({ data, isConnectable }: any) => {
  const getTypeColor = () => {
    switch (data.type) {
      case "pareto":
        return "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-300"
      case "sensitivity":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300"
      case "uncertainty":
        return "bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-300"
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300"
    }
  }

  return (
    <Card className="w-64 shadow-md">
      <CardHeader className="p-3 pb-0">
        <CardTitle className="text-sm flex items-center">
          <BarChart className="mr-2 h-4 w-4 text-cyan-500" />
          {data.name || "Result Analysis"}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3">
        <div className="flex flex-col gap-1">
          <div className="flex justify-between items-center">
            <Badge variant="outline" className={getTypeColor()}>
              {data.type || "pareto"}
            </Badge>
            <span className="text-xs">{data.dimensions || 2}D visualization</span>
          </div>
        </div>
      </CardContent>

      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        style={{ background: "#555", width: 8, height: 8 }}
        isConnectable={isConnectable}
      />

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        style={{ background: "#555", width: 8, height: 8 }}
        isConnectable={isConnectable}
      />
    </Card>
  )
})

AnalysisNode.displayName = "AnalysisNode"
