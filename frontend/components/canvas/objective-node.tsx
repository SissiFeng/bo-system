"use client"

import { memo } from "react"
import { Handle, Position } from "reactflow"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { LineChart } from "lucide-react"

export const ObjectiveNode = memo(({ data, isConnectable }: any) => {
  const getTypeColor = () => {
    switch (data.type) {
      case "maximize":
        return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300"
      case "minimize":
        return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300"
      case "target_range":
        return "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300"
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300"
    }
  }

  return (
    <Card className="w-64 shadow-md">
      <CardHeader className="p-3 pb-0">
        <CardTitle className="text-sm flex items-center">
          <LineChart className="mr-2 h-4 w-4 text-primary" />
          {data.name || "Objective"}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3">
        <div className="flex flex-col gap-1">
          <div className="flex justify-between items-center">
            <Badge variant="outline" className={getTypeColor()}>
              {data.type || "maximize"}
            </Badge>
            {data.type === "target_range" ? (
              <span className="text-xs">
                {data.targetMin} to {data.targetMax}
                {data.unit ? ` ${data.unit}` : ""}
              </span>
            ) : (
              <span className="text-xs">{data.type === "maximize" ? "Higher is better" : "Lower is better"}</span>
            )}
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

ObjectiveNode.displayName = "ObjectiveNode"
