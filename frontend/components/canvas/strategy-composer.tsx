"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Plus, Trash2, MoveUp, MoveDown, ArrowRight } from "lucide-react"
import { strategyTemplates } from "@/lib/templates"

interface StrategyStage {
  id: string
  type: string
  name: string
  iterations: number
  explorationWeight?: number
  acquisitionFunction?: string
  batchSize?: number
}

interface StrategyComposerProps {
  onUpdate: (stages: StrategyStage[]) => void
}

export function StrategyComposer({ onUpdate }: StrategyComposerProps) {
  const [stages, setStages] = useState<StrategyStage[]>([
    {
      id: "stage-1",
      type: "lhs",
      name: "Initial Exploration",
      iterations: 10,
    },
    {
      id: "stage-2",
      type: "bo",
      name: "Bayesian Optimization",
      iterations: 20,
      explorationWeight: 0.5,
      acquisitionFunction: "ei",
      batchSize: 5,
    },
  ])

  const addStage = () => {
    const newStage: StrategyStage = {
      id: `stage-${Date.now()}`,
      type: "bo",
      name: "New Stage",
      iterations: 10,
      explorationWeight: 0.5,
      acquisitionFunction: "ei",
      batchSize: 5,
    }
    setStages([...stages, newStage])
    onUpdate([...stages, newStage])
  }

  const removeStage = (id: string) => {
    const updatedStages = stages.filter((stage) => stage.id !== id)
    setStages(updatedStages)
    onUpdate(updatedStages)
  }

  const moveStage = (id: string, direction: "up" | "down") => {
    const index = stages.findIndex((stage) => stage.id === id)
    if ((direction === "up" && index === 0) || (direction === "down" && index === stages.length - 1)) {
      return
    }

    const newIndex = direction === "up" ? index - 1 : index + 1
    const updatedStages = [...stages]
    const [movedStage] = updatedStages.splice(index, 1)
    updatedStages.splice(newIndex, 0, movedStage)

    setStages(updatedStages)
    onUpdate(updatedStages)
  }

  const updateStage = (id: string, updates: Partial<StrategyStage>) => {
    const updatedStages = stages.map((stage) => (stage.id === id ? { ...stage, ...updates } : stage))
    setStages(updatedStages)
    onUpdate(updatedStages)
  }

  const getTotalIterations = () => {
    return stages.reduce((sum, stage) => sum + stage.iterations, 0)
  }

  const getStageColor = (type: string) => {
    switch (type) {
      case "lhs":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300"
      case "sobol":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300"
      case "random":
        return "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300"
      case "bo":
        return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300"
      case "bo_ei":
        return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-300"
      case "bo_ucb":
        return "bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-300"
      case "bo_multi":
        return "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-300"
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300"
    }
  }

  const getStrategyName = (type: string) => {
    const template = strategyTemplates.find((t) => t.id === type)
    return template?.name || type
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Multi-Strategy Composition</CardTitle>
        <CardDescription>Combine multiple optimization strategies into a sequential workflow</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Strategy Timeline Visualization */}
          <div className="relative py-6">
            <div className="absolute top-1/2 left-0 right-0 h-1 bg-muted -translate-y-1/2"></div>
            <div className="flex justify-between relative">
              {stages.map((stage, index) => (
                <div key={stage.id} className="flex flex-col items-center relative z-10">
                  <Badge className={`${getStageColor(stage.type)} mb-2`}>Stage {index + 1}</Badge>
                  <div className="w-6 h-6 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-xs">
                    {index + 1}
                  </div>
                  <div className="mt-2 text-xs text-center max-w-[80px] truncate">{stage.name}</div>
                  <div className="text-xs text-muted-foreground">{stage.iterations} iter</div>
                  {index < stages.length - 1 && (
                    <ArrowRight className="absolute -right-8 top-0 h-4 w-4 text-muted-foreground" />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Strategy Stages */}
          <div className="space-y-4">
            {stages.map((stage, index) => (
              <Card key={stage.id} className="relative">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base flex items-center">
                      <Badge className={`${getStageColor(stage.type)} mr-2`}>Stage {index + 1}</Badge>
                      {stage.name}
                    </CardTitle>
                    <div className="flex items-center space-x-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => moveStage(stage.id, "up")}
                        disabled={index === 0}
                      >
                        <MoveUp className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => moveStage(stage.id, "down")}
                        disabled={index === stages.length - 1}
                      >
                        <MoveDown className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => removeStage(stage.id)}
                        disabled={stages.length <= 1}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor={`name-${stage.id}`}>Stage Name</Label>
                        <Input
                          id={`name-${stage.id}`}
                          value={stage.name}
                          onChange={(e) => updateStage(stage.id, { name: e.target.value })}
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor={`type-${stage.id}`}>Strategy Type</Label>
                        <Select value={stage.type} onValueChange={(value) => updateStage(stage.id, { type: value })}>
                          <SelectTrigger id={`type-${stage.id}`}>
                            <SelectValue placeholder="Select strategy type" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="lhs">Latin Hypercube Sampling</SelectItem>
                            <SelectItem value="sobol">Sobol Sequence</SelectItem>
                            <SelectItem value="random">Random Sampling</SelectItem>
                            <SelectItem value="bo">Bayesian Optimization</SelectItem>
                            <SelectItem value="bo_ei">BO (Expected Improvement)</SelectItem>
                            <SelectItem value="bo_ucb">BO (Upper Confidence Bound)</SelectItem>
                            <SelectItem value="bo_multi">Multi-Objective BO</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor={`iterations-${stage.id}`}>Iterations: {stage.iterations}</Label>
                        <Slider
                          id={`iterations-${stage.id}`}
                          min={1}
                          max={50}
                          step={1}
                          value={[stage.iterations]}
                          onValueChange={(value) => updateStage(stage.id, { iterations: value[0] })}
                        />
                      </div>

                      {(stage.type === "bo" || stage.type.startsWith("bo_")) && (
                        <div className="space-y-4">
                          <div className="space-y-2">
                            <Label htmlFor={`exploration-${stage.id}`}>
                              Exploration Weight: {stage.explorationWeight?.toFixed(2)}
                            </Label>
                            <Slider
                              id={`exploration-${stage.id}`}
                              min={0}
                              max={1}
                              step={0.05}
                              value={[stage.explorationWeight || 0.5]}
                              onValueChange={(value) => updateStage(stage.id, { explorationWeight: value[0] })}
                            />
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor={`batch-${stage.id}`}>Batch Size: {stage.batchSize}</Label>
                            <Slider
                              id={`batch-${stage.id}`}
                              min={1}
                              max={10}
                              step={1}
                              value={[stage.batchSize || 5]}
                              onValueChange={(value) => updateStage(stage.id, { batchSize: value[0] })}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="flex justify-between items-center">
            <Button onClick={addStage} variant="outline">
              <Plus className="mr-2 h-4 w-4" />
              Add Strategy Stage
            </Button>
            <div className="text-sm text-muted-foreground">Total: {getTotalIterations()} iterations</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
