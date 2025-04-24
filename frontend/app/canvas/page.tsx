"use client"

import type React from "react"

import { useState, useCallback, useRef, useEffect } from "react"
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  Panel,
  type Connection,
  type Edge,
  type NodeTypes,
} from "reactflow"
import "reactflow/dist/style.css"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ParameterNode } from "@/components/canvas/parameter-node"
import { ObjectiveNode } from "@/components/canvas/objective-node"
import { ConstraintNode } from "@/components/canvas/constraint-node"
import { StrategyNode } from "@/components/canvas/strategy-node"
import { OutputNode } from "@/components/canvas/output-node"
import { DataProcessingNode } from "@/components/canvas/data-processing-node"
import { AnalysisNode } from "@/components/canvas/analysis-node"
import { NodePanel } from "@/components/canvas/node-panel"
import { ModuleLibrary } from "@/components/canvas/module-library"
import { StrategyConfig } from "@/components/canvas/strategy-config"
import { StrategyComposer } from "@/components/canvas/strategy-composer"
import { useCanvasStore } from "@/lib/stores/canvas-store"
import { Download, Play, Save } from "lucide-react"
import { MarkerType } from "reactflow"

// Define custom node types
const nodeTypes: NodeTypes = {
  parameterNode: ParameterNode,
  objectiveNode: ObjectiveNode,
  constraintNode: ConstraintNode,
  strategyNode: StrategyNode,
  outputNode: OutputNode,
  dataProcessingNode: DataProcessingNode,
  analysisNode: AnalysisNode,
}

export default function CanvasPage() {
  // React Flow state
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [selectedNode, setSelectedNode] = useState<any>(null)
  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null)
  const { updateCanvasState } = useCanvasStore()
  const [isRunning, setIsRunning] = useState(false)
  const [activeTab, setActiveTab] = useState("canvas")
  const [strategyConfig, setStrategyConfig] = useState({
    explorationWeight: 0.5,
    kernelType: "rbf",
    acquisitionFunction: "ei",
    batchSize: 5,
    iterations: 10,
  })
  const [strategyStages, setStrategyStages] = useState([])
  const [strategyConfigTab, setStrategyConfigTab] = useState("basic")

  // Prevent infinite updates by using a ref to track if we need to update canvas state
  const shouldUpdateCanvasState = useRef(false)

  // Update canvas state when nodes or edges change, but only when needed
  useEffect(() => {
    if (shouldUpdateCanvasState.current) {
      updateCanvasState({ nodes, edges })
      shouldUpdateCanvasState.current = false
    }
  }, [nodes, edges, updateCanvasState])

  // Handle connections between nodes
  const onConnect = useCallback(
    (params: Connection | Edge) => {
      // Add animated and styled edges
      const edge = {
        ...params,
        animated: true,
        style: { stroke: "#3b82f6", strokeWidth: 2 },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: "#3b82f6",
        },
      }
      setEdges((eds) => addEdge(edge, eds))
      shouldUpdateCanvasState.current = true
    },
    [setEdges],
  )

  // Handle node selection for property panel
  const onNodeClick = useCallback((_: React.MouseEvent, node: any) => {
    setSelectedNode(node)
  }, [])

  // Handle drag and drop from library to canvas
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = "move"
  }, [])

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()

      const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect()
      const type = event.dataTransfer.getData("application/reactflow")

      // Check if the dropped element is valid
      if (typeof type === "undefined" || !type || !reactFlowInstance || !reactFlowBounds) {
        return
      }

      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      })

      // Create a new node based on the type
      const newNode = {
        id: `${type}-${Date.now()}`,
        type: `${type}Node`,
        position,
        data: { label: `New ${type.charAt(0).toUpperCase() + type.slice(1)}` },
      }

      // Add specific data based on node type
      switch (type) {
        case "parameter":
          newNode.data = {
            ...newNode.data,
            name: "Parameter",
            type: "continuous",
            min: 0,
            max: 1,
          }
          break
        case "objective":
          newNode.data = {
            ...newNode.data,
            name: "Objective",
            type: "maximize",
          }
          break
        case "constraint":
          newNode.data = {
            ...newNode.data,
            expression: "x1 + x2",
            type: "sum_equals",
            value: 1,
          }
          break
        case "strategy":
          newNode.data = {
            ...newNode.data,
            name: "Bayesian Optimization",
            type: "bo",
            batchSize: 5,
          }
          break
        case "output":
          newNode.data = {
            ...newNode.data,
            name: "Output",
            type: "json",
          }
          break
        case "dataProcessing":
          newNode.data = {
            ...newNode.data,
            name: "Data Processing",
            method: "min-max",
            type: "normalization",
          }
          break
        case "analysis":
          newNode.data = {
            ...newNode.data,
            name: "Result Analysis",
            type: "pareto",
            dimensions: 2,
          }
          break
      }

      setNodes((nds) => nds.concat(newNode))
      shouldUpdateCanvasState.current = true
    },
    [reactFlowInstance, setNodes],
  )

  // Generate configuration JSON from canvas state
  const generateConfig = useCallback(() => {
    const parameters = nodes
      .filter((node) => node.type === "parameterNode")
      .map((node) => ({
        id: node.id,
        name: node.data.name,
        type: node.data.type,
        min: node.data.min,
        max: node.data.max,
        step: node.data.step,
        values: node.data.values,
        dimensions: node.data.dimensions,
      }))

    const objectives = nodes
      .filter((node) => node.type === "objectiveNode")
      .map((node) => ({
        id: node.id,
        name: node.data.name,
        type: node.data.type,
        expression: node.data.expression,
        targetMin: node.data.targetMin,
        targetMax: node.data.targetMax,
      }))

    const constraints = nodes
      .filter((node) => node.type === "constraintNode")
      .map((node) => ({
        id: node.id,
        expression: node.data.expression,
        type: node.data.type,
        value: node.data.value,
      }))

    const strategy = nodes.find((node) => node.type === "strategyNode")?.data || {
      type: "bo",
      batchSize: 5,
    }

    // Add the strategy configuration
    const enhancedStrategy = {
      ...strategy,
      ...strategyConfig,
      stages: strategyStages,
    }

    return {
      parameters,
      objectives,
      constraints,
      strategy: enhancedStrategy,
    }
  }, [nodes, strategyConfig, strategyStages])

  // Update strategy configuration
  const handleStrategyConfigUpdate = useCallback((config: any) => {
    setStrategyConfig(config)
  }, [])

  // Update strategy stages
  const handleStrategyStagesUpdate = useCallback((stages: any[]) => {
    setStrategyStages(stages)
  }, [])

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Optimization Workflow Builder</h1>
        <div className="flex gap-2">
          <Button variant="outline">
            <Save className="mr-2 h-4 w-4" />
            Save
          </Button>
          <Button variant="outline">
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
          <Button>
            <Play className="mr-2 h-4 w-4" />
            Run Optimization
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-6">
          <TabsTrigger value="canvas">Canvas</TabsTrigger>
          <TabsTrigger value="strategy">Strategy Configuration</TabsTrigger>
          <TabsTrigger value="preview">Preview</TabsTrigger>
        </TabsList>

        <TabsContent value="canvas">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            <div className="lg:col-span-3">
              <Card className="h-[700px] overflow-hidden">
                <div className="h-full w-full" ref={reactFlowWrapper}>
                  <ReactFlowProvider>
                    <ReactFlow
                      nodes={nodes}
                      edges={edges}
                      onNodesChange={onNodesChange}
                      onEdgesChange={onEdgesChange}
                      onConnect={onConnect}
                      onInit={setReactFlowInstance}
                      onNodeClick={onNodeClick}
                      onDrop={onDrop}
                      onDragOver={onDragOver}
                      nodeTypes={nodeTypes}
                      fitView
                      deleteKeyCode="Delete"
                    >
                      <Controls />
                      <Background gap={12} size={1} />
                      <Panel position="top-left">
                        <div className="bg-background p-4 rounded-md shadow-md">
                          <h3 className="text-sm font-medium mb-4">Module Library</h3>
                          <ModuleLibrary />
                        </div>
                      </Panel>
                    </ReactFlow>
                  </ReactFlowProvider>
                </div>
              </Card>

              <Card className="mt-6 p-4">
                <Tabs defaultValue="json">
                  <TabsList className="mb-4">
                    <TabsTrigger value="json">JSON Configuration</TabsTrigger>
                    <TabsTrigger value="preview">Preview</TabsTrigger>
                  </TabsList>
                  <TabsContent value="json">
                    <pre className="bg-muted p-4 rounded-md overflow-auto text-xs h-[200px]">
                      {JSON.stringify(generateConfig(), null, 2)}
                    </pre>
                  </TabsContent>
                  <TabsContent value="preview">
                    <div className="h-[200px] flex items-center justify-center text-muted-foreground">
                      Run optimization to see results preview
                    </div>
                  </TabsContent>
                </Tabs>
              </Card>
            </div>

            <div className="lg:col-span-1">
              <NodePanel selectedNode={selectedNode} setNodes={setNodes} />
            </div>
          </div>
        </TabsContent>

        <TabsContent value="strategy">
          <Tabs value={strategyConfigTab} onValueChange={setStrategyConfigTab}>
            <TabsList className="mb-6">
              <TabsTrigger value="basic">Basic Configuration</TabsTrigger>
              <TabsTrigger value="multi">Multi-Strategy Composition</TabsTrigger>
            </TabsList>

            <TabsContent value="basic">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <StrategyConfig strategy={strategyConfig} onUpdate={handleStrategyConfigUpdate} />

                <Card>
                  <CardHeader>
                    <CardTitle>Strategy JSON Configuration</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <pre className="bg-muted p-4 rounded-md overflow-auto text-xs h-[400px]">
                      {JSON.stringify({ strategy: strategyConfig }, null, 2)}
                    </pre>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="multi">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <StrategyComposer onUpdate={handleStrategyStagesUpdate} />

                <Card>
                  <CardHeader>
                    <CardTitle>Multi-Strategy Configuration</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <pre className="bg-muted p-4 rounded-md overflow-auto text-xs h-[400px]">
                      {JSON.stringify({ stages: strategyStages }, null, 2)}
                    </pre>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </TabsContent>

        <TabsContent value="preview">
          <Card>
            <CardHeader>
              <CardTitle>Workflow Preview</CardTitle>
            </CardHeader>
            <CardContent className="h-[600px] flex items-center justify-center">
              <div className="text-center text-muted-foreground">
                <p>Run optimization to see workflow execution preview</p>
                <Button className="mt-4">
                  <Play className="mr-2 h-4 w-4" />
                  Run Optimization
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
