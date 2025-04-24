"use client"

import { useCallback, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import * as z from "zod"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts"
import { Button } from "@/components/ui/button"

// Define schemas for different node types
const parameterSchema = z.object({
  name: z.string().min(1, "Name is required"),
  type: z.string(),
  min: z.number().optional(),
  max: z.number().optional(),
  step: z.number().optional(),
  values: z.string().optional(),
  unit: z.string().optional(),
  description: z.string().optional(),
  dimensions: z.number().optional(),
})

const objectiveSchema = z.object({
  name: z.string().min(1, "Name is required"),
  type: z.string(),
  expression: z.string().optional(),
  targetMin: z.number().optional(),
  targetMax: z.number().optional(),
  unit: z.string().optional(),
  description: z.string().optional(),
})

const constraintSchema = z.object({
  expression: z.string().min(1, "Expression is required"),
  type: z.string(),
  value: z.number(),
  description: z.string().optional(),
})

const strategySchema = z.object({
  name: z.string().min(1, "Name is required"),
  type: z.string(),
  batchSize: z.number().min(1),
  explorationWeight: z.number().min(0).max(1),
  kernelType: z.string(),
  acquisitionFunction: z.string(),
  description: z.string().optional(),
})

const dataProcessingSchema = z.object({
  name: z.string().min(1, "Name is required"),
  type: z.string(),
  method: z.string(),
  description: z.string().optional(),
})

const analysisSchema = z.object({
  name: z.string().min(1, "Name is required"),
  type: z.string(),
  dimensions: z.number().min(2).max(3),
  description: z.string().optional(),
})

export function NodePanel({ selectedNode, setNodes }: any) {
  // If no node is selected, show empty state
  if (!selectedNode) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Properties</CardTitle>
          <CardDescription>Select a node to edit its properties</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64 text-muted-foreground">No node selected</div>
        </CardContent>
      </Card>
    )
  }

  // Determine which form to show based on node type
  switch (selectedNode.type) {
    case "parameterNode":
      return <ParameterPanel selectedNode={selectedNode} setNodes={setNodes} />
    case "objectiveNode":
      return <ObjectivePanel selectedNode={selectedNode} setNodes={setNodes} />
    case "constraintNode":
      return <ConstraintPanel selectedNode={selectedNode} setNodes={setNodes} />
    case "strategyNode":
      return <StrategyPanel selectedNode={selectedNode} setNodes={setNodes} />
    case "dataProcessingNode":
      return <DataProcessingPanel selectedNode={selectedNode} setNodes={setNodes} />
    case "analysisNode":
      return <AnalysisPanel selectedNode={selectedNode} setNodes={setNodes} />
    case "outputNode":
      return <OutputPanel selectedNode={selectedNode} setNodes={setNodes} />
    default:
      return (
        <Card>
          <CardHeader>
            <CardTitle>Properties</CardTitle>
            <CardDescription>Unknown node type</CardDescription>
          </CardHeader>
        </Card>
      )
  }
}

function ParameterPanel({ selectedNode, setNodes }: any) {
  const form = useForm({
    resolver: zodResolver(parameterSchema),
    defaultValues: {
      name: selectedNode.data.name || "",
      type: selectedNode.data.type || "continuous",
      min: selectedNode.data.min || 0,
      max: selectedNode.data.max || 1,
      step: selectedNode.data.step || undefined,
      values: selectedNode.data.values?.join(", ") || "",
      unit: selectedNode.data.unit || "",
      description: selectedNode.data.description || "",
      dimensions: selectedNode.data.dimensions || 2,
    },
  })

  const parameterType = form.watch("type")
  const dimensions = form.watch("dimensions")

  const onSubmit = useCallback(
    (values: any) => {
      setNodes((nds: any[]) =>
        nds.map((node) => {
          if (node.id === selectedNode.id) {
            // Process values based on type
            const processedValues = { ...values }
            if (values.type === "categorical" && typeof values.values === "string") {
              processedValues.values = values.values.split(",").map((v: string) => v.trim())
            }

            return {
              ...node,
              data: {
                ...node.data,
                ...processedValues,
              },
            }
          }
          return node
        }),
      )
    },
    [selectedNode.id, setNodes],
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle>Parameter Space Properties</CardTitle>
        <CardDescription>Configure your parameter space</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="basic">
          <TabsList className="grid grid-cols-2 mb-4">
            <TabsTrigger value="basic">Basic</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>

          <TabsContent value="basic">
            <Form {...form}>
              <form className="space-y-4">
                <FormField
                  control={form.control}
                  name="name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Name</FormLabel>
                      <FormControl>
                        <Input placeholder="e.g., Parameter Space" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="type"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Type</FormLabel>
                      <Select
                        onValueChange={(value) => {
                          field.onChange(value)
                          setTimeout(() => form.handleSubmit(onSubmit)(), 0)
                        }}
                        defaultValue={field.value}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select parameter type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="continuous">Continuous</SelectItem>
                          <SelectItem value="discrete">Discrete</SelectItem>
                          <SelectItem value="categorical">Categorical</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="dimensions"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Dimensions</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          min={1}
                          max={10}
                          {...field}
                          onChange={(e) => field.onChange(Number.parseInt(e.target.value))}
                          onBlur={() => form.handleSubmit(onSubmit)()}
                          value={field.value || 2}
                        />
                      </FormControl>
                      <FormDescription>Number of parameters in the space</FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <Button type="button" onClick={form.handleSubmit(onSubmit)} className="w-full mt-2">
                  Apply Changes
                </Button>
              </form>
            </Form>
          </TabsContent>

          <TabsContent value="advanced">
            <Form {...form}>
              <form className="space-y-4">
                {(parameterType === "continuous" || parameterType === "discrete") && (
                  <>
                    <div className="grid grid-cols-2 gap-4">
                      <FormField
                        control={form.control}
                        name="min"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Min</FormLabel>
                            <FormControl>
                              <Input
                                type="number"
                                {...field}
                                onChange={(e) => field.onChange(Number.parseFloat(e.target.value))}
                                value={field.value || ""}
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                      <FormField
                        control={form.control}
                        name="max"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Max</FormLabel>
                            <FormControl>
                              <Input
                                type="number"
                                {...field}
                                onChange={(e) => field.onChange(Number.parseFloat(e.target.value))}
                                value={field.value || ""}
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    </div>

                    {parameterType === "discrete" && (
                      <FormField
                        control={form.control}
                        name="step"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Step Size</FormLabel>
                            <FormControl>
                              <Input
                                type="number"
                                {...field}
                                onChange={(e) => field.onChange(Number.parseFloat(e.target.value))}
                                value={field.value || ""}
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    )}
                  </>
                )}

                {parameterType === "categorical" && (
                  <FormField
                    control={form.control}
                    name="values"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Values</FormLabel>
                        <FormControl>
                          <Input placeholder="value1, value2, value3" {...field} />
                        </FormControl>
                        <FormDescription>Comma-separated list of possible values</FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                )}

                <FormField
                  control={form.control}
                  name="description"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Description (Optional)</FormLabel>
                      <FormControl>
                        <Textarea placeholder="Brief description of this parameter space" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <Button type="button" onClick={form.handleSubmit(onSubmit)} className="w-full mt-2">
                  Apply Changes
                </Button>
              </form>
            </Form>
          </TabsContent>
        </Tabs>

        {/* Parameter Space Visualization */}
        <div className="mt-6 pt-4 border-t">
          <h3 className="text-sm font-medium mb-2">Parameter Space Preview</h3>
          <div className="bg-muted rounded-md p-2 h-32 flex items-center justify-center">
            <div className="text-xs text-muted-foreground">
              {dimensions}D Parameter Space ({parameterType})
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function ObjectivePanel({ selectedNode, setNodes }: any) {
  const form = useForm({
    resolver: zodResolver(objectiveSchema),
    defaultValues: {
      name: selectedNode.data.name || "",
      type: selectedNode.data.type || "maximize",
      expression: selectedNode.data.expression || "",
      targetMin: selectedNode.data.targetMin || 0,
      targetMax: selectedNode.data.targetMax || 1,
      unit: selectedNode.data.unit || "",
      description: selectedNode.data.description || "",
    },
  })

  const objectiveType = form.watch("type")
  const expression = form.watch("expression")

  const onSubmit = useCallback(
    (values: any) => {
      setNodes((nds: any[]) =>
        nds.map((node) => {
          if (node.id === selectedNode.id) {
            return {
              ...node,
              data: {
                ...node.data,
                ...values,
              },
            }
          }
          return node
        }),
      )
    },
    [selectedNode.id, setNodes],
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle>Objective Function Properties</CardTitle>
        <CardDescription>Configure your optimization objective</CardDescription>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form className="space-y-4">
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Name</FormLabel>
                  <FormControl>
                    <Input placeholder="e.g., Objective Function" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="type"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Optimization Type</FormLabel>
                  <Select
                    onValueChange={(value) => {
                      field.onChange(value)
                      setTimeout(() => form.handleSubmit(onSubmit)(), 0)
                    }}
                    defaultValue={field.value}
                  >
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select optimization type" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value="maximize">Maximize</SelectItem>
                      <SelectItem value="minimize">Minimize</SelectItem>
                      <SelectItem value="target_range">Target Range</SelectItem>
                    </SelectContent>
                  </Select>
                  <FormDescription>How this objective should be optimized</FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="expression"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Function Expression</FormLabel>
                  <FormControl>
                    <Input placeholder="e.g., f(x, y) = y - x^2" {...field} />
                  </FormControl>
                  <FormDescription>Mathematical expression of the objective function</FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            {objectiveType === "target_range" && (
              <div className="grid grid-cols-2 gap-4">
                <FormField
                  control={form.control}
                  name="targetMin"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Target Minimum</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          placeholder="0"
                          {...field}
                          onChange={(e) => field.onChange(Number.parseFloat(e.target.value))}
                          value={field.value ?? ""}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="targetMax"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Target Maximum</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          placeholder="1"
                          {...field}
                          onChange={(e) => field.onChange(Number.parseFloat(e.target.value))}
                          value={field.value ?? ""}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            )}

            <Button type="button" onClick={form.handleSubmit(onSubmit)} className="w-full mt-2">
              Apply Changes
            </Button>
          </form>
        </Form>

        {/* Objective Function Visualization */}
        {expression && (
          <div className="mt-6 pt-4 border-t">
            <h3 className="text-sm font-medium mb-2">Function Visualization</h3>
            <div className="bg-white rounded-md p-2 h-40 border">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={[
                    { x: 0, y: 0 },
                    { x: 0.2, y: 0.3 },
                    { x: 0.4, y: 0.5 },
                    { x: 0.6, y: 0.8 },
                    { x: 0.8, y: 0.6 },
                    { x: 1, y: 1 },
                  ]}
                  margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="x" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="y" stroke="#8884d8" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="text-xs text-center mt-2 text-muted-foreground">{expression}</div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function ConstraintPanel({ selectedNode, setNodes }: any) {
  const form = useForm({
    resolver: zodResolver(constraintSchema),
    defaultValues: {
      expression: selectedNode.data.expression || "",
      type: selectedNode.data.type || "sum_equals",
      value: selectedNode.data.value || 1,
      description: selectedNode.data.description || "",
    },
  })

  const onSubmit = useCallback(
    (values: any) => {
      setNodes((nds: any[]) =>
        nds.map((node) => {
          if (node.id === selectedNode.id) {
            return {
              ...node,
              data: {
                ...node.data,
                ...values,
              },
            }
          }
          return node
        }),
      )
    },
    [selectedNode.id, setNodes],
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle>Constraint Properties</CardTitle>
        <CardDescription>Configure your constraint</CardDescription>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form className="space-y-4">
            <FormField
              control={form.control}
              name="expression"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Expression</FormLabel>
                  <FormControl>
                    <Input placeholder="e.g., x1 + x2 + x3" {...field} />
                  </FormControl>
                  <FormDescription>Variables in your parameter space (e.g., x1, x2)</FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="type"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Constraint Type</FormLabel>
                  <Select
                    onValueChange={(value) => {
                      field.onChange(value)
                      setTimeout(() => form.handleSubmit(onSubmit)(), 0)
                    }}
                    defaultValue={field.value}
                  >
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select constraint type" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value="sum_equals">Sum Equals (∑xᵢ = value)</SelectItem>
                      <SelectItem value="sum_less_than">Sum Less Than (∑xᵢ ≤ value)</SelectItem>
                      <SelectItem value="sum_greater_than">Sum Greater Than (∑xᵢ ≥ value)</SelectItem>
                    </SelectContent>
                  </Select>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="value"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Value</FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      placeholder="1"
                      {...field}
                      onChange={(e) => field.onChange(Number.parseFloat(e.target.value))}
                      value={field.value || ""}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="description"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Description (Optional)</FormLabel>
                  <FormControl>
                    <Textarea placeholder="Brief description of this constraint" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <Button type="button" onClick={form.handleSubmit(onSubmit)} className="w-full mt-2">
              Apply Changes
            </Button>
          </form>
        </Form>
      </CardContent>
    </Card>
  )
}

function StrategyPanel({ selectedNode, setNodes }: any) {
  const [visualizationData, setVisualizationData] = useState(() => generateVisualizationData(0.5, "ei"))

  const form = useForm({
    resolver: zodResolver(strategySchema),
    defaultValues: {
      name: selectedNode.data.name || "",
      type: selectedNode.data.type || "bo",
      batchSize: selectedNode.data.batchSize || 5,
      explorationWeight: selectedNode.data.explorationWeight || 0.5,
      kernelType: selectedNode.data.kernelType || "matern",
      acquisitionFunction: selectedNode.data.acquisitionFunction || "ei",
      description: selectedNode.data.description || "",
    },
  })

  const explorationWeight = form.watch("explorationWeight")
  const acquisitionFunction = form.watch("acquisitionFunction")

  // Update visualization when parameters change
  const updateVisualization = useCallback(() => {
    setVisualizationData(generateVisualizationData(explorationWeight, acquisitionFunction))
  }, [explorationWeight, acquisitionFunction])

  const onSubmit = useCallback(
    (values: any) => {
      setNodes((nds: any[]) =>
        nds.map((node) => {
          if (node.id === selectedNode.id) {
            return {
              ...node,
              data: {
                ...node.data,
                ...values,
              },
            }
          }
          return node
        }),
      )
      updateVisualization()
    },
    [selectedNode.id, setNodes, updateVisualization],
  )

  // Generate visualization data based on parameters
  function generateVisualizationData(exploration: number, acqFunc: string) {
    const data = []
    const points = 50

    for (let i = 0; i < points; i++) {
      const x = i / (points - 1)

      // Base function (posterior mean)
      const mean = Math.sin(x * Math.PI * 2) * 0.5 + 0.5

      // Uncertainty (posterior variance)
      const variance = 0.1 + 0.2 * Math.sin(x * Math.PI * 4 + 1) * Math.sin(x * Math.PI * 2.5)

      // Acquisition function value depends on type and exploration weight
      let acquisition
      if (acqFunc === "ei") {
        // Expected Improvement
        acquisition = variance * exploration + (mean - 0.5) * (1 - exploration)
      } else if (acqFunc === "ucb") {
        // Upper Confidence Bound
        acquisition = mean + variance * exploration * 2
      } else {
        // Probability of Improvement
        acquisition = mean + variance * exploration
      }

      data.push({
        x,
        mean,
        variance,
        acquisition: Math.max(0, Math.min(1, acquisition)),
      })
    }

    return data
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Strategy Properties</CardTitle>
        <CardDescription>Configure your optimization strategy</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="basic">
          <TabsList className="grid grid-cols-2 mb-4">
            <TabsTrigger value="basic">Basic</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>

          <TabsContent value="basic">
            <Form {...form}>
              <form className="space-y-4">
                <FormField
                  control={form.control}
                  name="name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Name</FormLabel>
                      <FormControl>
                        <Input placeholder="e.g., Bayesian Optimization" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="type"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Strategy Type</FormLabel>
                      <Select
                        onValueChange={(value) => {
                          field.onChange(value)
                          setTimeout(() => form.handleSubmit(onSubmit)(), 0)
                        }}
                        defaultValue={field.value}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select strategy type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="bo">Bayesian Optimization</SelectItem>
                          <SelectItem value="lhs">Latin Hypercube Sampling</SelectItem>
                          <SelectItem value="sobol">Sobol Sequence</SelectItem>
                          <SelectItem value="random">Random Sampling</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="batchSize"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Batch Size</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          min={1}
                          max={20}
                          {...field}
                          onChange={(e) => field.onChange(Number.parseInt(e.target.value))}
                          value={field.value || ""}
                        />
                      </FormControl>
                      <FormDescription>Number of experiments per iteration</FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <Button type="button" onClick={form.handleSubmit(onSubmit)} className="w-full mt-2">
                  Apply Changes
                </Button>
              </form>
            </Form>
          </TabsContent>

          <TabsContent value="advanced">
            <Form {...form}>
              <form className="space-y-4">
                <FormField
                  control={form.control}
                  name="acquisitionFunction"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Acquisition Function</FormLabel>
                      <Select
                        onValueChange={(value) => {
                          field.onChange(value)
                          setTimeout(() => form.handleSubmit(onSubmit)(), 0)
                        }}
                        defaultValue={field.value}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select acquisition function" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="ei">Expected Improvement (EI)</SelectItem>
                          <SelectItem value="ucb">Upper Confidence Bound (UCB)</SelectItem>
                          <SelectItem value="pi">Probability of Improvement (PI)</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormDescription>How to balance exploration vs. exploitation</FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="explorationWeight"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Exploration Weight: {field.value}</FormLabel>
                      <FormControl>
                        <Input
                          type="range"
                          min={0}
                          max={1}
                          step={0.01}
                          {...field}
                          onChange={(e) => {
                            const value = Number.parseFloat(e.target.value)
                            field.onChange(value)
                            // Update visualization when slider changes
                            setVisualizationData(generateVisualizationData(value, acquisitionFunction))
                          }}
                          value={field.value || 0.5}
                          className="w-full"
                        />
                      </FormControl>
                      <FormDescription>
                        Low: Focus on exploitation (use what we know). High: Focus on exploration (try new areas).
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="kernelType"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Kernel Type</FormLabel>
                      <Select
                        onValueChange={(value) => {
                          field.onChange(value)
                          setTimeout(() => form.handleSubmit(onSubmit)(), 0)
                        }}
                        defaultValue={field.value}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select kernel type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="rbf">Radial Basis Function (RBF)</SelectItem>
                          <SelectItem value="matern">Matérn</SelectItem>
                          <SelectItem value="linear">Linear</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormDescription>Kernel function for Gaussian Process</FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <Button type="button" onClick={form.handleSubmit(onSubmit)} className="w-full mt-2">
                  Apply Changes
                </Button>
              </form>
            </Form>

            {/* Real-time Visualization */}
            <div className="mt-6 pt-4 border-t">
              <h3 className="text-sm font-medium mb-2">Acquisition Function Visualization</h3>
              <div className="bg-white rounded-md p-2 h-40 border">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={visualizationData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="x" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="mean" stroke="#8884d8" name="Mean" />
                    <Line type="monotone" dataKey="acquisition" stroke="#82ca9d" name="Acquisition" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="text-xs text-center mt-2 text-muted-foreground">
                {acquisitionFunction === "ei"
                  ? "Expected Improvement"
                  : acquisitionFunction === "ucb"
                    ? "Upper Confidence Bound"
                    : "Probability of Improvement"}
                {" with "}
                {explorationWeight < 0.3 ? "low" : explorationWeight > 0.7 ? "high" : "balanced"} exploration
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

function DataProcessingPanel({ selectedNode, setNodes }: any) {
  const form = useForm({
    resolver: zodResolver(dataProcessingSchema),
    defaultValues: {
      name: selectedNode.data.name || "",
      type: selectedNode.data.type || "filter",
      method: selectedNode.data.method || "gaussian",
      description: selectedNode.data.description || "",
    },
  })

  const onSubmit = useCallback(
    (values: any) => {
      setNodes((nds: any[]) =>
        nds.map((node) => {
          if (node.id === selectedNode.id) {
            return {
              ...node,
              data: {
                ...node.data,
                ...values,
              },
            }
          }
          return node
        }),
      )
    },
    [selectedNode.id, setNodes],
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle>Data Processing Properties</CardTitle>
        <CardDescription>Configure your data processing method</CardDescription>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form className="space-y-4">
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Name</FormLabel>
                  <FormControl>
                    <Input placeholder="e.g., Gaussian Filter" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="type"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Processing Type</FormLabel>
                  <Select
                    onValueChange={(value) => {
                      field.onChange(value)
                      setTimeout(() => form.handleSubmit(onSubmit)(), 0)
                    }}
                    defaultValue={field.value}
                  >
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select processing type" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value="filter">Filter</SelectItem>
                      <SelectItem value="normalize">Normalize</SelectItem>
                      <SelectItem value="aggregate">Aggregate</SelectItem>
                    </SelectContent>
                  </Select>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="method"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Method</FormLabel>
                  <FormControl>
                    <Input placeholder="e.g., Gaussian" {...field} />
                  </FormControl>
                  <FormDescription>Specific method for data processing</FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="description"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Description (Optional)</FormLabel>
                  <FormControl>
                    <Textarea placeholder="Brief description of this data processing step" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <Button type="button" onClick={form.handleSubmit(onSubmit)} className="w-full mt-2">
              Apply Changes
            </Button>
          </form>
        </Form>
      </CardContent>
    </Card>
  )
}

function AnalysisPanel({ selectedNode, setNodes }: any) {
  const form = useForm({
    resolver: zodResolver(analysisSchema),
    defaultValues: {
      name: selectedNode.data.name || "",
      type: selectedNode.data.type || "pca",
      dimensions: selectedNode.data.dimensions || 2,
      description: selectedNode.data.description || "",
    },
  })

  const onSubmit = useCallback(
    (values: any) => {
      setNodes((nds: any[]) =>
        nds.map((node) => {
          if (node.id === selectedNode.id) {
            return {
              ...node,
              data: {
                ...node.data,
                ...values,
              },
            }
          }
          return node
        }),
      )
    },
    [selectedNode.id, setNodes],
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle>Analysis Properties</CardTitle>
        <CardDescription>Configure your analysis method</CardDescription>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form className="space-y-4">
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Name</FormLabel>
                  <FormControl>
                    <Input placeholder="e.g., Principal Component Analysis" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="type"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Analysis Type</FormLabel>
                  <Select
                    onValueChange={(value) => {
                      field.onChange(value)
                      setTimeout(() => form.handleSubmit(onSubmit)(), 0)
                    }}
                    defaultValue={field.value}
                  >
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select analysis type" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value="pca">Principal Component Analysis (PCA)</SelectItem>
                      <SelectItem value="tsne">t-distributed Stochastic Neighbor Embedding (t-SNE)</SelectItem>
                      <SelectItem value="umap">Uniform Manifold Approximation and Projection (UMAP)</SelectItem>
                    </SelectContent>
                  </Select>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="dimensions"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Dimensions</FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      min={2}
                      max={3}
                      {...field}
                      onChange={(e) => field.onChange(Number.parseInt(e.target.value))}
                      value={field.value || 2}
                    />
                  </FormControl>
                  <FormDescription>Number of dimensions to reduce to</FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="description"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Description (Optional)</FormLabel>
                  <FormControl>
                    <Textarea placeholder="Brief description of this analysis step" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <Button type="button" onClick={form.handleSubmit(onSubmit)} className="w-full mt-2">
              Apply Changes
            </Button>
          </form>
        </Form>
      </CardContent>
    </Card>
  )
}

function OutputPanel({ selectedNode, setNodes }: any) {
  const [outputData, setOutputData] = useState([
    { x: 0, y: 0 },
    { x: 0.2, y: 0.3 },
    { x: 0.4, y: 0.5 },
    { x: 0.6, y: 0.8 },
    { x: 0.8, y: 0.6 },
    { x: 1, y: 1 },
  ])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Output Visualization</CardTitle>
        <CardDescription>Visualize the output of your workflow</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="bg-white rounded-md p-2 h-64 border">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={outputData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="y" stroke="#8884d8" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}

// Default export for the component
export default NodePanel
