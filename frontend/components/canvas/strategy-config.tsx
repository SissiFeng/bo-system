"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { RefreshCwIcon as Refresh } from "lucide-react"

interface StrategyConfigProps {
  strategy: any
  onUpdate: (strategy: any) => void
}

export function StrategyConfig({ strategy, onUpdate }: StrategyConfigProps) {
  const [activeTab, setActiveTab] = useState("parameters")
  const [localStrategy, setLocalStrategy] = useState({
    explorationWeight: strategy.explorationWeight || 0.5,
    kernelType: strategy.kernelType || "rbf",
    acquisitionFunction: strategy.acquisitionFunction || "ei",
    batchSize: strategy.batchSize || 5,
    iterations: strategy.iterations || 10,
  })
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const updateTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const isInitialMount = useRef(true)

  // Initialize local state from props
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false
      return
    }
  }, [strategy])

  // Handle strategy updates with debounce
  const handleStrategyChange = useCallback(
    (updates: Partial<typeof localStrategy>) => {
      setLocalStrategy((prev) => {
        const newStrategy = { ...prev, ...updates }

        // Clear any existing timeout
        if (updateTimeoutRef.current) {
          clearTimeout(updateTimeoutRef.current)
        }

        // Set a new timeout to call onUpdate
        updateTimeoutRef.current = setTimeout(() => {
          onUpdate(newStrategy)
        }, 300)

        return newStrategy
      })
    },
    [onUpdate],
  )

  // Draw acquisition function visualization
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
    ctx.fillText("Parameter Space", width / 2, height - 10)

    ctx.save()
    ctx.translate(15, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = "center"
    ctx.fillText("Acquisition Value", 0, 0)
    ctx.restore()

    // Draw acquisition function
    ctx.beginPath()

    // Different curves based on acquisition function and exploration weight
    for (let x = padding; x <= width - padding; x++) {
      const normalizedX = (x - padding) / (width - 2 * padding)
      let y = 0

      if (localStrategy.acquisitionFunction === "ei") {
        // Expected Improvement - bell curve with skew based on exploration weight
        const peak = 0.3 + localStrategy.explorationWeight * 0.4 // Peak position shifts with exploration weight
        const variance = 0.05 + localStrategy.explorationWeight * 0.1 // Wider curve with more exploration
        y = height - padding - (height - 2 * padding) * Math.exp(-Math.pow((normalizedX - peak) / variance, 2) / 2)
      } else if (localStrategy.acquisitionFunction === "ucb") {
        // Upper Confidence Bound - more jagged with higher values on edges for exploration
        const base = Math.sin(normalizedX * Math.PI * 4) * (0.2 + localStrategy.explorationWeight * 0.3)
        const trend =
          normalizedX * (1 - localStrategy.explorationWeight) + (1 - normalizedX) * localStrategy.explorationWeight
        y = height - padding - (height - 2 * padding) * (0.3 + base + trend * 0.4)
      } else {
        // Probability of Improvement - sharper peak
        const peak = 0.5
        const variance = 0.03 + (1 - localStrategy.explorationWeight) * 0.05
        y = height - padding - (height - 2 * padding) * Math.exp(-Math.pow((normalizedX - peak) / variance, 2) / 2)
      }

      if (x === padding) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    }

    ctx.strokeStyle = "#3b82f6"
    ctx.lineWidth = 2
    ctx.stroke()

    // Draw area under curve
    ctx.lineTo(width - padding, height - padding)
    ctx.lineTo(padding, height - padding)
    ctx.closePath()
    ctx.fillStyle = "rgba(59, 130, 246, 0.1)"
    ctx.fill()

    // Draw next sampling points
    const numPoints = localStrategy.batchSize
    const points = []

    if (localStrategy.acquisitionFunction === "ei") {
      // EI tends to sample near the peak and some exploration
      const peak = 0.3 + localStrategy.explorationWeight * 0.4
      for (let i = 0; i < numPoints; i++) {
        const noise = (Math.random() - 0.5) * (0.2 + localStrategy.explorationWeight * 0.4)
        const x = padding + (width - 2 * padding) * (peak + noise)
        const y = height - padding - (height - 2 * padding) * (0.7 + Math.random() * 0.2)
        points.push({ x, y })
      }
    } else if (localStrategy.acquisitionFunction === "ucb") {
      // UCB tends to be more exploratory
      for (let i = 0; i < numPoints; i++) {
        const x = padding + (width - 2 * padding) * Math.random()
        const y = height - padding - (height - 2 * padding) * (0.4 + Math.random() * 0.5)
        points.push({ x, y })
      }
    } else {
      // PI tends to be more exploitative
      const peak = 0.5
      for (let i = 0; i < numPoints; i++) {
        const noise = (Math.random() - 0.5) * (0.1 + (1 - localStrategy.explorationWeight) * 0.1)
        const x = padding + (width - 2 * padding) * (peak + noise)
        const y = height - padding - (height - 2 * padding) * (0.8 + Math.random() * 0.15)
        points.push({ x, y })
      }
    }

    // Draw points
    points.forEach((point) => {
      ctx.beginPath()
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)
      ctx.fillStyle = "#3b82f6"
      ctx.fill()
    })

    // Draw legend
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "left"
    ctx.fillText(`${localStrategy.acquisitionFunction.toUpperCase()} Acquisition Function`, padding + 10, padding + 20)
    ctx.fillText(`Exploration Weight: ${localStrategy.explorationWeight.toFixed(2)}`, padding + 10, padding + 40)
    ctx.fillText(`Batch Size: ${localStrategy.batchSize}`, padding + 10, padding + 60)
  }, [localStrategy])

  // Clean up timeout on unmount
  useEffect(() => {
    return () => {
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current)
      }
    }
  }, [])

  const handleApplyChanges = () => {
    onUpdate(localStrategy)
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Strategy Configuration</CardTitle>
        <CardDescription>Fine-tune your optimization strategy parameters</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid grid-cols-3 mb-4">
            <TabsTrigger value="parameters">Parameters</TabsTrigger>
            <TabsTrigger value="visualization">Visualization</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>

          <TabsContent value="parameters" className="space-y-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Exploration vs. Exploitation</Label>
                <div className="flex justify-between text-sm text-muted-foreground mb-2">
                  <span>Exploitation</span>
                  <span>Balanced</span>
                  <span>Exploration</span>
                </div>
                <Slider
                  value={[localStrategy.explorationWeight * 100]}
                  min={0}
                  max={100}
                  step={5}
                  onValueChange={(value) => handleStrategyChange({ explorationWeight: value[0] / 100 })}
                />
              </div>

              <div className="space-y-2">
                <Label>Acquisition Function</Label>
                <RadioGroup
                  value={localStrategy.acquisitionFunction}
                  onValueChange={(value) => handleStrategyChange({ acquisitionFunction: value })}
                  className="flex flex-col space-y-2 mt-2"
                >
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="ei" id="ei" />
                    <Label htmlFor="ei">Expected Improvement (balanced)</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="ucb" id="ucb" />
                    <Label htmlFor="ucb">Upper Confidence Bound (exploration-focused)</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="pi" id="pi" />
                    <Label htmlFor="pi">Probability of Improvement (exploitation-focused)</Label>
                  </div>
                </RadioGroup>
              </div>

              <div className="space-y-2">
                <Label>Batch Size</Label>
                <div className="flex justify-between text-sm text-muted-foreground mb-2">
                  <span>Small (3)</span>
                  <span>Medium (6)</span>
                  <span>Large (10)</span>
                </div>
                <Slider
                  value={[localStrategy.batchSize]}
                  min={1}
                  max={10}
                  step={1}
                  onValueChange={(value) => handleStrategyChange({ batchSize: value[0] })}
                />
              </div>

              <div className="space-y-2">
                <Label>Number of Iterations</Label>
                <Input
                  type="number"
                  value={localStrategy.iterations}
                  onChange={(e) => handleStrategyChange({ iterations: Number(e.target.value) })}
                  min={1}
                  max={100}
                />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="visualization">
            <div className="space-y-4">
              <div className="border rounded-md p-2 bg-white">
                <canvas ref={canvasRef} width={500} height={300} className="w-full h-auto" />
              </div>

              <div className="flex justify-end">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    // Force redraw by creating a new object with the same values
                    setLocalStrategy({ ...localStrategy })
                  }}
                >
                  <Refresh className="mr-2 h-4 w-4" />
                  Refresh Visualization
                </Button>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="advanced" className="space-y-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Kernel Type</Label>
                <Select
                  value={localStrategy.kernelType}
                  onValueChange={(value) => handleStrategyChange({ kernelType: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select kernel type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="rbf">RBF (Radial Basis Function)</SelectItem>
                    <SelectItem value="matern">Matérn</SelectItem>
                    <SelectItem value="linear">Linear</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Kernel Length Scale</Label>
                <Input type="number" defaultValue="1.0" min="0.1" step="0.1" />
              </div>

              <div className="space-y-2">
                <Label>Random Seed</Label>
                <Input type="number" defaultValue="42" />
              </div>
            </div>
          </TabsContent>
        </Tabs>

        <div className="mt-6 pt-4 border-t">
          <Button onClick={handleApplyChanges} className="w-full">
            Apply Changes
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
