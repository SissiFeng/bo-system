"use client"

import { memo } from "react"
import { Handle, Position } from "reactflow"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Box } from "lucide-react"

export const OutputNode = memo(({ data, isConnectable }: any) => {
  const getOutputTypeName = () => {
    switch (data.type) {
      case "json":
        return "JSON"
      case "csv":
        return "CSV"
      case "python":
        return "Python Script"
      default:
        return "JSON"
    }
  }

  return (
    <Card className="w-64 shadow-md">
      <CardHeader className="p-3 pb-0">
        <CardTitle className="text-sm flex items-center">
          <Box className="mr-2 h-4 w-4 text-primary" />
          {data.name || "Output"}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3">
        <div className="flex flex-col gap-1">
          <Badge variant="outline" className="self-start">
            {getOutputTypeName()}
          </Badge>
          <p className="text-xs text-muted-foreground">Click to view or export results</p>
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
    </Card>
  )
})

OutputNode.displayName = "OutputNode"
