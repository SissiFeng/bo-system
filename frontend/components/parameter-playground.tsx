"use client"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { ParameterType, type Parameter } from "@/lib/types"
import { Plus, Trash2, GripVertical, LinkIcon, Settings } from "lucide-react"
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors } from "@dnd-kit/core"
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable"
import { CSS } from "@dnd-kit/utilities"

interface ParameterConstraint {
  id: string
  type: "equality" | "inequality" | "sum" | "custom"
  parameters: string[]
  operator: "=" | "<" | ">" | "<=" | ">=" | "sum"
  value: number
  expression?: string
}

interface ParameterPlaygroundProps {
  initialParameters?: Parameter[]
  onUpdate?: (parameters: Parameter[], constraints: ParameterConstraint[]) => void
}

// Sortable parameter item component
function SortableParameterItem({
  parameter,
  onEdit,
  onDelete,
  onDuplicate,
}: {
  parameter: Parameter
  onEdit: (id: string, updates: Partial<Parameter>) => void
  onDelete: (id: string) => void
  onDuplicate: (parameter: Parameter) => void
}) {
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id: parameter.id })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      className="flex items-center space-x-2 bg-background border rounded-md p-3 mb-2"
    >
      <div {...attributes} {...listeners} className="cursor-grab">
        <GripVertical className="h-5 w-5 text-muted-foreground" />
      </div>

      <div className="flex-1 grid grid-cols-5 gap-2 items-center">
        <div className="col-span-2">
          <Input
            value={parameter.name}
            onChange={(e) => onEdit(parameter.id, { name: e.target.value })}
            placeholder="Parameter name"
            className="h-8"
          />
        </div>

        <div>
          <Select
            value={parameter.type}
            onValueChange={(value) => onEdit(parameter.id, { type: value as ParameterType })}
          >
            <SelectTrigger className="h-8">
              <SelectValue placeholder="Type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ParameterType.CONTINUOUS}>Continuous</SelectItem>
              <SelectItem value={ParameterType.DISCRETE}>Discrete</SelectItem>
              <SelectItem value={ParameterType.CATEGORICAL}>Categorical</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="col-span-2">
          {parameter.type === ParameterType.CATEGORICAL ? (
            <Input
              value={parameter.values?.join(", ") || ""}
              onChange={(e) => onEdit(parameter.id, { values: e.target.value.split(",").map((v) => v.trim()) })}
              placeholder="Values (comma separated)"
              className="h-8"
            />
          ) : (
            <div className="flex items-center space-x-2">
              <Input
                type="number"
                value={parameter.min}
                onChange={(e) => onEdit(parameter.id, { min: Number(e.target.value) })}
                placeholder="Min"
                className="h-8 w-20"
              />
              <span className="text-muted-foreground">to</span>
              <Input
                type="number"
                value={parameter.max}
                onChange={(e) => onEdit(parameter.id, { max: Number(e.target.value) })}
                placeholder="Max"
                className="h-8 w-20"
              />
              {parameter.type === ParameterType.DISCRETE && (
                <>
                  <span className="text-muted-foreground">step</span>
                  <Input
                    type="number"
                    value={parameter.step}
                    onChange={(e) => onEdit(parameter.id, { step: Number(e.target.value) })}
                    placeholder="Step"
                    className="h-8 w-16"
                  />
                </>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="flex space-x-1">
        <Button variant="ghost" size="icon" onClick={() => onDuplicate(parameter)}>
          <Plus className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" onClick={() => onDelete(parameter.id)}>
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}

// Constraint component
function ConstraintItem({
  constraint,
  parameters,
  onUpdate,
  onDelete,
}: {
  constraint: ParameterConstraint
  parameters: Parameter[]
  onUpdate: (id: string, updates: Partial<ParameterConstraint>) => void
  onDelete: (id: string) => void
}) {
  return (
    <div className="flex items-center space-x-2 bg-background border rounded-md p-3 mb-2">
      <div className="flex-1 grid grid-cols-6 gap-2 items-center">
        <div className="col-span-2">
          <Select
            value={constraint.type}
            onValueChange={(value) =>
              onUpdate(constraint.id, { type: value as "equality" | "inequality" | "sum" | "custom" })
            }
          >
            <SelectTrigger className="h-8">
              <SelectValue placeholder="Type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="equality">Equality</SelectItem>
              <SelectItem value="inequality">Inequality</SelectItem>
              <SelectItem value="sum">Sum Constraint</SelectItem>
              <SelectItem value="custom">Custom Expression</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {constraint.type === "custom" ? (
          <div className="col-span-3">
            <Input
              value={constraint.expression || ""}
              onChange={(e) => onUpdate(constraint.id, { expression: e.target.value })}
              placeholder="e.g., x1 + x2 <= 1"
              className="h-8"
            />
          </div>
        ) : (
          <>
            <div className="col-span-2">
              <Select
                value={constraint.parameters[0] || ""}
                onValueChange={(value) => {
                  const newParams = [...constraint.parameters]
                  newParams[0] = value
                  onUpdate(constraint.id, { parameters: newParams })
                }}
              >
                <SelectTrigger className="h-8">
                  <SelectValue placeholder="Parameter" />
                </SelectTrigger>
                <SelectContent>
                  {parameters.map((param) => (
                    <SelectItem key={param.id} value={param.id}>
                      {param.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Select
                value={constraint.operator}
                onValueChange={(value) => onUpdate(constraint.id, { operator: value as any })}
              >
                <SelectTrigger className="h-8">
                  <SelectValue placeholder="Op" />
                </SelectTrigger>
                <SelectContent>
                  {constraint.type === "equality" && <SelectItem value="=">= (Equal)</SelectItem>}
                  {constraint.type === "inequality" && (
                    <>
                      <SelectItem value="<">{"<"} (Less than)</SelectItem>
                      <SelectItem value=">">{">"} (Greater than)</SelectItem>
                      <SelectItem value="<=">{"≤"} (Less or equal)</SelectItem>
                      <SelectItem value=">=">{"≥"} (Greater or equal)</SelectItem>
                    </>
                  )}
                  {constraint.type === "sum" && <SelectItem value="sum">Sum</SelectItem>}
                </SelectContent>
              </Select>
            </div>

            {constraint.type === "sum" ? (
              <div>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button variant="outline" size="sm" className="h-8">
                      {constraint.parameters.length} parameters
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-80">
                    <div className="space-y-2">
                      <h4 className="font-medium">Select Parameters</h4>
                      <div className="space-y-1">
                        {parameters.map((param) => (
                          <div key={param.id} className="flex items-center space-x-2">
                            <Switch
                              checked={constraint.parameters.includes(param.id)}
                              onCheckedChange={(checked) => {
                                const newParams = checked
                                  ? [...constraint.parameters, param.id]
                                  : constraint.parameters.filter((p) => p !== param.id)
                                onUpdate(constraint.id, { parameters: newParams })
                              }}
                            />
                            <Label>{param.name}</Label>
                          </div>
                        ))}
                      </div>
                    </div>
                  </PopoverContent>
                </Popover>
              </div>
            ) : null}

            <div>
              <Input
                type="number"
                value={constraint.value}
                onChange={(e) => onUpdate(constraint.id, { value: Number(e.target.value) })}
                placeholder="Value"
                className="h-8"
              />
            </div>
          </>
        )}
      </div>

      <Button variant="ghost" size="icon" onClick={() => onDelete(constraint.id)}>
        <Trash2 className="h-4 w-4" />
      </Button>
    </div>
  )
}

export function ParameterPlayground({ initialParameters = [], onUpdate }: ParameterPlaygroundProps) {
  const [parameters, setParameters] = useState<Parameter[]>(initialParameters)
  const [constraints, setConstraints] = useState<ParameterConstraint[]>([])
  const [activeTab, setActiveTab] = useState("parameters")
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // DnD sensors
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  )

  // Handle DnD end
  const handleDragEnd = (event: any) => {
    const { active, over } = event

    if (active.id !== over.id) {
      setParameters((items) => {
        const oldIndex = items.findIndex((item) => item.id === active.id)
        const newIndex = items.findIndex((item) => item.id === over.id)
        return arrayMove(items, oldIndex, newIndex)
      })
    }
  }

  // Parameter CRUD operations
  const addParameter = () => {
    const newParameter: Parameter = {
      id: `param-${Date.now()}`,
      name: `Parameter ${parameters.length + 1}`,
      type: ParameterType.CONTINUOUS,
      min: 0,
      max: 1,
    }
    setParameters([...parameters, newParameter])
  }

  const updateParameter = (id: string, updates: Partial<Parameter>) => {
    setParameters(
      parameters.map((param) => {
        if (param.id === id) {
          return { ...param, ...updates }
        }
        return param
      }),
    )
  }

  const deleteParameter = (id: string) => {
    setParameters(parameters.filter((param) => param.id !== id))
    // Also remove any constraints that reference this parameter
    setConstraints(
      constraints.filter((constraint) => {
        if (constraint.type === "custom") return true
        return !constraint.parameters.includes(id)
      }),
    )
  }

  const duplicateParameter = (parameter: Parameter) => {
    const newParameter: Parameter = {
      ...parameter,
      id: `param-${Date.now()}`,
      name: `${parameter.name} (copy)`,
    }
    setParameters([...parameters, newParameter])
  }

  // Constraint CRUD operations
  const addConstraint = () => {
    const newConstraint: ParameterConstraint = {
      id: `constraint-${Date.now()}`,
      type: "equality",
      parameters: parameters.length > 0 ? [parameters[0].id] : [],
      operator: "=",
      value: 0,
    }
    setConstraints([...constraints, newConstraint])
  }

  const updateConstraint = (id: string, updates: Partial<ParameterConstraint>) => {
    setConstraints(
      constraints.map((constraint) => {
        if (constraint.id === id) {
          return { ...constraint, ...updates }
        }
        return constraint
      }),
    )
  }

  const deleteConstraint = (id: string) => {
    setConstraints(constraints.filter((constraint) => constraint.id !== id))
  }

  // Notify parent component of updates
  useEffect(() => {
    if (onUpdate) {
      onUpdate(parameters, constraints)
    }
  }, [parameters, constraints, onUpdate])

  // Draw parameter space visualization
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Set dimensions
    const width = canvas.width
    const height = canvas.height
    const padding = 40

    // If we have less than 2 parameters, show a message
    if (parameters.length < 2) {
      ctx.fillStyle = "#888"
      ctx.font = "14px sans-serif"
      ctx.textAlign = "center"
      ctx.fillText("Add at least 2 parameters to visualize parameter space", width / 2, height / 2)
      return
    }

    // Get the first two continuous or discrete parameters
    const continuousParams = parameters.filter(
      (p) => p.type === ParameterType.CONTINUOUS || p.type === ParameterType.DISCRETE,
    )

    if (continuousParams.length < 2) {
      ctx.fillStyle = "#888"
      ctx.font = "14px sans-serif"
      ctx.textAlign = "center"
      ctx.fillText("Need at least 2 continuous/discrete parameters", width / 2, height / 2)
      return
    }

    const param1 = continuousParams[0]
    const param2 = continuousParams[1]

    // Draw axes
    ctx.beginPath()
    ctx.moveTo(padding, height - padding)
    ctx.lineTo(width - padding, height - padding)
    ctx.strokeStyle = "#888"
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(padding, height - padding)
    ctx.lineTo(padding, padding)
    ctx.strokeStyle = "#888"
    ctx.stroke()

    // Draw axis labels
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(param1.name, width / 2, height - 10)

    ctx.save()
    ctx.translate(15, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = "center"
    ctx.fillText(param2.name, 0, 0)
    ctx.restore()

    // Draw ticks and values for X axis
    const min1 = param1.min ?? 0
    const max1 = param1.max ?? 1
    const range1 = max1 - min1

    for (let i = 0; i <= 5; i++) {
      const x = padding + ((width - 2 * padding) * i) / 5
      const value = min1 + (range1 * i) / 5

      // Draw tick
      ctx.beginPath()
      ctx.moveTo(x, height - padding)
      ctx.lineTo(x, height - padding + 5)
      ctx.strokeStyle = "#888"
      ctx.stroke()

      // Draw value
      ctx.fillStyle = "#888"
      ctx.font = "10px sans-serif"
      ctx.textAlign = "center"
      ctx.fillText(value.toFixed(1), x, height - padding + 15)
    }

    // Draw ticks and values for Y axis
    const min2 = param2.min ?? 0
    const max2 = param2.max ?? 1
    const range2 = max2 - min2

    for (let i = 0; i <= 5; i++) {
      const y = height - padding - ((height - 2 * padding) * i) / 5
      const value = min2 + (range2 * i) / 5

      // Draw tick
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(padding - 5, y)
      ctx.strokeStyle = "#888"
      ctx.stroke()

      // Draw value
      ctx.fillStyle = "#888"
      ctx.font = "10px sans-serif"
      ctx.textAlign = "right"
      ctx.fillText(value.toFixed(1), padding - 8, y + 3)
    }

    // Draw parameter space grid
    ctx.strokeStyle = "#ddd"
    ctx.setLineDash([2, 2])

    // Vertical grid lines
    for (let i = 1; i < 5; i++) {
      const x = padding + ((width - 2 * padding) * i) / 5
      ctx.beginPath()
      ctx.moveTo(x, padding)
      ctx.lineTo(x, height - padding)
      ctx.stroke()
    }

    // Horizontal grid lines
    for (let i = 1; i < 5; i++) {
      const y = padding + ((height - 2 * padding) * i) / 5
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(width - padding, y)
      ctx.stroke()
    }
    ctx.setLineDash([])

    // Draw constraints
    constraints.forEach((constraint, index) => {
      // Skip constraints that don't involve our visualized parameters
      if (
        constraint.type === "custom" ||
        (!constraint.parameters.includes(param1.id) && !constraint.parameters.includes(param2.id))
      ) {
        return
      }

      const colors = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6"]
      const color = colors[index % colors.length]

      if (constraint.parameters.includes(param1.id) && constraint.parameters.includes(param2.id)) {
        // Both parameters are involved
        if (constraint.type === "sum" && constraint.operator === "sum") {
          // Draw a line for sum constraint (x + y = c)
          const value = constraint.value
          const x1 = padding
          const y1 = height - padding - ((height - 2 * padding) * (value - min1)) / range2
          const x2 = padding + ((width - 2 * padding) * (value - min2)) / range1
          const y2 = height - padding

          ctx.beginPath()
          ctx.moveTo(x1, y1)
          ctx.lineTo(x2, y2)
          ctx.strokeStyle = color
          ctx.lineWidth = 2
          ctx.stroke()
          ctx.lineWidth = 1

          // Fill the feasible region
          ctx.beginPath()
          ctx.moveTo(padding, height - padding)
          ctx.lineTo(x1, y1)
          ctx.lineTo(x2, y2)
          ctx.closePath()
          ctx.fillStyle = `${color}20` // 20% opacity
          ctx.fill()
        } else if (constraint.type === "equality" && constraint.operator === "=") {
          // Draw a point for equality constraint
          const x = padding + ((width - 2 * padding) * (constraint.value - min1)) / range1
          const y = height - padding - ((height - 2 * padding) * (constraint.value - min2)) / range2

          ctx.beginPath()
          ctx.arc(x, y, 5, 0, Math.PI * 2)
          ctx.fillStyle = color
          ctx.fill()
        }
      } else if (constraint.parameters.includes(param1.id)) {
        // Only x-axis parameter is involved
        const value = constraint.value
        const x = padding + ((width - 2 * padding) * (value - min1)) / range1

        ctx.beginPath()
        ctx.moveTo(x, padding)
        ctx.lineTo(x, height - padding)
        ctx.strokeStyle = color
        ctx.setLineDash([5, 5])
        ctx.stroke()
        ctx.setLineDash([])

        // Fill the feasible region based on the operator
        if (constraint.operator === "<" || constraint.operator === "<=") {
          ctx.beginPath()
          ctx.rect(padding, padding, x - padding, height - 2 * padding)
          ctx.fillStyle = `${color}20` // 20% opacity
          ctx.fill()
        } else if (constraint.operator === ">" || constraint.operator === ">=") {
          ctx.beginPath()
          ctx.rect(x, padding, width - padding - x, height - 2 * padding)
          ctx.fillStyle = `${color}20` // 20% opacity
          ctx.fill()
        }
      } else if (constraint.parameters.includes(param2.id)) {
        // Only y-axis parameter is involved
        const value = constraint.value
        const y = height - padding - ((height - 2 * padding) * (value - min2)) / range2

        ctx.beginPath()
        ctx.moveTo(padding, y)
        ctx.lineTo(width - padding, y)
        ctx.strokeStyle = color
        ctx.setLineDash([5, 5])
        ctx.stroke()
        ctx.setLineDash([])

        // Fill the feasible region based on the operator
        if (constraint.operator === "<" || constraint.operator === "<=") {
          ctx.beginPath()
          ctx.rect(padding, y, width - 2 * padding, height - padding - y)
          ctx.fillStyle = `${color}20` // 20% opacity
          ctx.fill()
        } else if (constraint.operator === ">" || constraint.operator === ">=") {
          ctx.beginPath()
          ctx.rect(padding, padding, width - 2 * padding, y - padding)
          ctx.fillStyle = `${color}20` // 20% opacity
          ctx.fill()
        }
      }
    })

    // If we have more than 2 parameters, visualize them as points
    if (parameters.length > 2) {
      // Generate some sample points in the parameter space
      const pointCount = Math.min(50, 5 * parameters.length)
      const points = []

      for (let i = 0; i < pointCount; i++) {
        const x = padding + Math.random() * (width - 2 * padding)
        const y = padding + Math.random() * (height - 2 * padding)
        const size = 3 + Math.random() * 5
        points.push({ x, y, size })
      }

      // Draw points
      points.forEach((point) => {
        ctx.beginPath()
        ctx.arc(point.x, height - point.y, point.size, 0, Math.PI * 2)
        ctx.fillStyle = `hsla(${Math.random() * 360}, 70%, 60%, 0.7)`
        ctx.fill()
      })

      // Add legend for additional parameters
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "left"
      ctx.fillText(`+ ${parameters.length - 2} more parameters`, padding, padding - 10)
    }
  }, [parameters, constraints])

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Parameter Space Playground</CardTitle>
        <CardDescription>Define and visualize your parameter space and constraints</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="parameters">Parameters</TabsTrigger>
            <TabsTrigger value="constraints">Constraints</TabsTrigger>
            <TabsTrigger value="visualization">Visualization</TabsTrigger>
          </TabsList>

          <TabsContent value="parameters">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-medium">Parameter Definitions</h3>
                <Button onClick={addParameter} size="sm">
                  <Plus className="mr-2 h-4 w-4" />
                  Add Parameter
                </Button>
              </div>

              {parameters.length === 0 ? (
                <div className="flex flex-col items-center justify-center p-8 text-center border border-dashed rounded-lg">
                  <Settings className="h-8 w-8 text-muted-foreground mb-2" />
                  <p className="text-muted-foreground">No parameters defined yet</p>
                  <Button onClick={addParameter} variant="outline" className="mt-4">
                    <Plus className="mr-2 h-4 w-4" />
                    Add Parameter
                  </Button>
                </div>
              ) : (
                <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
                  <SortableContext items={parameters.map((p) => p.id)} strategy={verticalListSortingStrategy}>
                    {parameters.map((parameter) => (
                      <SortableParameterItem
                        key={parameter.id}
                        parameter={parameter}
                        onEdit={updateParameter}
                        onDelete={deleteParameter}
                        onDuplicate={duplicateParameter}
                      />
                    ))}
                  </SortableContext>
                </DndContext>
              )}
            </div>
          </TabsContent>

          <TabsContent value="constraints">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-medium">Parameter Constraints</h3>
                <Button onClick={addConstraint} size="sm" disabled={parameters.length === 0}>
                  <Plus className="mr-2 h-4 w-4" />
                  Add Constraint
                </Button>
              </div>

              {constraints.length === 0 ? (
                <div className="flex flex-col items-center justify-center p-8 text-center border border-dashed rounded-lg">
                  <LinkIcon className="h-8 w-8 text-muted-foreground mb-2" />
                  <p className="text-muted-foreground">No constraints defined yet</p>
                  <Button onClick={addConstraint} variant="outline" className="mt-4" disabled={parameters.length === 0}>
                    <Plus className="mr-2 h-4 w-4" />
                    Add Constraint
                  </Button>
                </div>
              ) : (
                <div className="space-y-2">
                  {constraints.map((constraint) => (
                    <ConstraintItem
                      key={constraint.id}
                      constraint={constraint}
                      parameters={parameters}
                      onUpdate={updateConstraint}
                      onDelete={deleteConstraint}
                    />
                  ))}
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="visualization">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-medium">Parameter Space Visualization</h3>
                <Badge variant="outline" className="font-mono">
                  {parameters.length} parameters, {constraints.length} constraints
                </Badge>
              </div>

              <div className="border rounded-md p-4 bg-white">
                <canvas ref={canvasRef} width={600} height={400} className="w-full h-auto" />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader className="py-3">
                    <CardTitle className="text-sm">Parameter Space</CardTitle>
                  </CardHeader>
                  <CardContent className="py-2">
                    <div className="space-y-1 text-sm">
                      {parameters.map((param) => (
                        <div key={param.id} className="flex justify-between">
                          <span className="font-medium">{param.name}</span>
                          <span className="text-muted-foreground">
                            {param.type === ParameterType.CATEGORICAL
                              ? param.values?.join(", ")
                              : `${param.min} to ${param.max}${
                                  param.type === ParameterType.DISCRETE && param.step ? ` (step: ${param.step})` : ""
                                }`}
                          </span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="py-3">
                    <CardTitle className="text-sm">Feasible Region</CardTitle>
                  </CardHeader>
                  <CardContent className="py-2">
                    <div className="space-y-1 text-sm">
                      {constraints.length === 0 ? (
                        <p className="text-muted-foreground">No constraints defined</p>
                      ) : (
                        constraints.map((constraint) => (
                          <div key={constraint.id} className="flex justify-between">
                            <span className="font-medium">
                              {constraint.type === "custom"
                                ? constraint.expression
                                : constraint.type === "sum"
                                  ? `Sum of ${constraint.parameters.length} params`
                                  : parameters.find((p) => p.id === constraint.parameters[0])?.name}
                            </span>
                            <span className="text-muted-foreground">
                              {constraint.type === "custom" ? "" : `${constraint.operator} ${constraint.value}`}
                            </span>
                          </div>
                        ))
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
