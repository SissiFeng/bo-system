"use client"

import { memo } from "react"
import { Handle, Position } from "reactflow"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Settings } from "lucide-react"

export const ParameterNode = memo(({ data, isConnectable }: any) => {
  const getTypeColor = () => {
    switch (data.type) {
      case "continuous":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300"
      case "discrete":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300"
      case "categorical":
        return "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300"
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300"
    }
  }

  return (
    <Card className="w-64 shadow-md">
      <CardHeader className="p-3 pb-0">
        <CardTitle className="text-sm flex items-center">
          <Settings className="mr-2 h-4 w-4 text-primary" />
          {data.name || "Parameter"}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3">
        <div className="flex flex-col gap-1">
          <div className="flex justify-between items-center">
            <Badge variant="outline" className={getTypeColor()}>
              {data.type || "continuous"}
            </Badge>
            {data.type !== "categorical" ? (
              <span className="text-xs">
                {data.min} to {data.max}
                {data.unit ? ` ${data.unit}` : ""}
              </span>
            ) : (
              <span className="text-xs truncate max-w-[120px]">{data.values?.join(", ") || "No values"}</span>
            )}
          </div>
        </div>
      </CardContent>

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

ParameterNode.displayName = "ParameterNode"
