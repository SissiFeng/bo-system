"use client"

import { memo } from "react"
import { Handle, Position } from "reactflow"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Calculator } from "lucide-react"

export const ConstraintNode = memo(({ data, isConnectable }: any) => {
  const getConstraintSymbol = () => {
    switch (data.type) {
      case "sum_equals":
        return "="
      case "sum_less_than":
        return "≤"
      case "sum_greater_than":
        return "≥"
      default:
        return "="
    }
  }

  const getConstraintTypeName = () => {
    switch (data.type) {
      case "sum_equals":
        return "Sum equals"
      case "sum_less_than":
        return "Sum less than"
      case "sum_greater_than":
        return "Sum greater than"
      default:
        return "Sum equals"
    }
  }

  return (
    <Card className="w-64 shadow-md">
      <CardHeader className="p-3 pb-0">
        <CardTitle className="text-sm flex items-center">
          <Calculator className="mr-2 h-4 w-4 text-primary" />
          Constraint
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3">
        <div className="flex flex-col gap-1">
          <Badge variant="outline" className="mb-1 self-start">
            {getConstraintTypeName()}
          </Badge>
          <div className="text-xs font-mono bg-muted p-1 rounded">
            {data.expression || "x1 + x2"} {getConstraintSymbol()} {data.value || 1}
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

ConstraintNode.displayName = "ConstraintNode"
