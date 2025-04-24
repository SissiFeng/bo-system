"use client"

import { memo } from "react"
import { Handle, Position } from "reactflow"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Database } from "lucide-react"

export const DataProcessingNode = memo(({ data, isConnectable }: any) => {
  const getMethodColor = () => {
    switch (data.method) {
      case "min-max":
        return "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300"
      case "z-score":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300"
      case "log":
        return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300"
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300"
    }
  }

  return (
    <Card className="w-64 shadow-md">
      <CardHeader className="p-3 pb-0">
        <CardTitle className="text-sm flex items-center">
          <Database className="mr-2 h-4 w-4 text-amber-500" />
          {data.name || "Data Processing"}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3">
        <div className="flex flex-col gap-1">
          <div className="flex justify-between items-center">
            <Badge variant="outline" className={getMethodColor()}>
              {data.method || "min-max"}
            </Badge>
            <span className="text-xs">{data.type || "normalization"}</span>
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

DataProcessingNode.displayName = "DataProcessingNode"
