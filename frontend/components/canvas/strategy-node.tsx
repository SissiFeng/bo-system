"use client"

import { memo } from "react"
import { Handle, Position } from "reactflow"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Beaker } from "lucide-react"

export const StrategyNode = memo(({ data, isConnectable }: any) => {
  const getStrategyName = () => {
    switch (data.type) {
      case "lhs":
        return "Latin Hypercube Sampling"
      case "sobol":
        return "Sobol Sequence"
      case "random":
        return "Random Sampling"
      case "bo":
        return "Bayesian Optimization"
      case "bo_ei":
        return "BO Expected Improvement"
      case "bo_ucb":
        return "BO Upper Confidence Bound"
      case "bo_multi":
        return "Multi-Objective BO"
      default:
        return "Bayesian Optimization"
    }
  }

  const getAcquisitionName = () => {
    switch (data.acquisitionFunction) {
      case "ei":
        return "Expected Improvement"
      case "ucb":
        return "Upper Confidence Bound"
      case "pi":
        return "Probability of Improvement"
      default:
        return "Expected Improvement"
    }
  }

  const getKernelName = () => {
    switch (data.kernelType) {
      case "rbf":
        return "RBF"
      case "matern":
        return "Matérn"
      case "linear":
        return "Linear"
      default:
        return "Matérn"
    }
  }

  return (
    <Card className="w-64 shadow-md">
      <CardHeader className="p-3 pb-0">
        <CardTitle className="text-sm flex items-center">
          <Beaker className="mr-2 h-4 w-4 text-purple-500" />
          {data.name || "Strategy"}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3">
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center">
            <Badge variant="outline" className="bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-300">
              {getStrategyName()}
            </Badge>
          </div>

          <div className="grid grid-cols-2 gap-1 text-xs">
            <div className="flex items-center">
              <span className="text-muted-foreground mr-1">Batch:</span>
              <span>{data.batchSize || 5}</span>
            </div>
            <div className="flex items-center">
              <span className="text-muted-foreground mr-1">Explore:</span>
              <span>{(data.explorationWeight || 0.5) * 100}%</span>
            </div>
            <div className="flex items-center">
              <span className="text-muted-foreground mr-1">Kernel:</span>
              <span>{getKernelName()}</span>
            </div>
            <div className="flex items-center">
              <span className="text-muted-foreground mr-1">Acq:</span>
              <span>{getAcquisitionName()}</span>
            </div>
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

StrategyNode.displayName = "StrategyNode"
