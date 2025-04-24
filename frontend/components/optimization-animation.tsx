"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Play, Pause, SkipBack, SkipForward, Clock } from "lucide-react"

interface OptimizationAnimationProps {
  data?: any[]
  width?: number
  height?: number
}

export function OptimizationAnimation({ data = [], width = 600, height = 400 }: OptimizationAnimationProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [speed, setSpeed] = useState(1)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | null>(null)
  const lastTimeRef = useRef<number>(0)

  // Generate sample data if none provided
  const animationData = data.length > 0 ? data : generateSampleData()

  function generateSampleData() {
    const steps = 50
    const sampleData = []

    // Generate initial random points
    const initialPoints = Array(10)
      .fill(0)
      .map(() => ({
        x: Math.random() * 0.8 + 0.1,
        y: Math.random() * 0.8 + 0.1,
        value: Math.random(),
        uncertainty: 0.2,
      }))

    sampleData.push({
      iteration: 0,
      points: initialPoints,
      best: initialPoints.reduce((best, point) => (point.value > best.value ? point : best), initialPoints[0]),
      acquisitionFunction: null,
      mean: null,
    })

    // Generate a simple objective function (for demonstration)
    const objectiveFunction = (x: number, y: number) => {
      const term1 = Math.sin(Math.PI * x) * Math.sin(Math.PI * y)
      const term2 = -((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.2
      return term1 + Math.exp(term2)
    }

    // Generate subsequent steps with BO-like behavior
    for (let i = 1; i < steps; i++) {
      // Create a grid for the mean and acquisition function
      const gridSize = 20
      const mean = Array(gridSize)
        .fill(0)
        .map(() => Array(gridSize).fill(0))
      const acquisition = Array(gridSize)
        .fill(0)
        .map(() => Array(gridSize).fill(0))

      // Fill the grid with values
      for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
          const xVal = x / (gridSize - 1)
          const yVal = y / (gridSize - 1)
          mean[x][y] = objectiveFunction(xVal, yVal)

          // Simple acquisition function (higher near existing points but not exactly at them)
          let acq = 0
          const prevPoints = sampleData[i - 1].points
          for (const point of prevPoints) {
            const dist = Math.sqrt((xVal - point.x) ** 2 + (yVal - point.y) ** 2)
            acq += point.uncertainty * Math.exp(-dist * 10) * (dist > 0.05 ? 1 : 0)
          }
          acquisition[x][y] = acq
        }
      }

      // Find the maximum acquisition point
      let maxAcq = Number.NEGATIVE_INFINITY
      let maxX = 0
      let maxY = 0
      for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
          if (acquisition[x][y] > maxAcq) {
            maxAcq = acquisition[x][y]
            maxX = x / (gridSize - 1)
            maxY = y / (gridSize - 1)
          }
        }
      }

      // Add a new point based on acquisition function
      const newPoint = {
        x: maxX,
        y: maxY,
        value: objectiveFunction(maxX, maxY) + (Math.random() - 0.5) * 0.1, // Add some noise
        uncertainty: 0.2 * Math.exp(-i / 10), // Uncertainty decreases over time
      }

      // Copy previous points and add the new one
      const points = [...sampleData[i - 1].points, newPoint]

      // Find the best point
      const best = points.reduce((best, point) => (point.value > best.value ? point : best), points[0])

      sampleData.push({
        iteration: i,
        points,
        best,
        acquisitionFunction: acquisition,
        mean,
      })
    }

    return sampleData
  }

  // Animation loop
  const animate = (time: number) => {
    if (!lastTimeRef.current) {
      lastTimeRef.current = time
    }

    const deltaTime = time - lastTimeRef.current
    if (deltaTime > 1000 / (speed * 2)) {
      // Update step based on speed
      if (currentStep < animationData.length - 1) {
        setCurrentStep((prev) => prev + 1)
      } else {
        setIsPlaying(false)
        return
      }
      lastTimeRef.current = time
    }

    if (isPlaying) {
      animationRef.current = requestAnimationFrame(animate)
    }
  }

  // Start/stop animation
  useEffect(() => {
    if (isPlaying) {
      animationRef.current = requestAnimationFrame(animate)
    } else if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isPlaying, speed])

  // Draw the current step
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const currentData = animationData[currentStep]
    if (!currentData) return

    // Draw background grid
    ctx.strokeStyle = "#ddd"
    ctx.lineWidth = 0.5
    for (let i = 0; i <= 10; i++) {
      const pos = (i / 10) * canvas.width
      ctx.beginPath()
      ctx.moveTo(pos, 0)
      ctx.lineTo(pos, canvas.height)
      ctx.stroke()

      ctx.beginPath()
      ctx.moveTo(0, pos)
      ctx.lineTo(canvas.width, pos)
      ctx.stroke()
    }

    // Draw mean function if available
    if (currentData.mean) {
      const gridSize = currentData.mean.length
      const cellWidth = canvas.width / gridSize
      const cellHeight = canvas.height / gridSize

      for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
          const value = currentData.mean[x][y]
          const normalizedValue = (value + 1) / 2 // Normalize to [0, 1]
          const intensity = Math.min(255, Math.max(0, Math.floor(normalizedValue * 255)))
          ctx.fillStyle = `rgba(0, 0, 255, ${normalizedValue * 0.3})`
          ctx.fillRect(x * cellWidth, (gridSize - y - 1) * cellHeight, cellWidth, cellHeight)
        }
      }
    }

    // Draw acquisition function if available
    if (currentData.acquisitionFunction && currentStep > 0) {
      const gridSize = currentData.acquisitionFunction.length
      const cellWidth = canvas.width / gridSize
      const cellHeight = canvas.height / gridSize

      // Find max value for normalization
      let maxVal = Number.NEGATIVE_INFINITY
      for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
          maxVal = Math.max(maxVal, currentData.acquisitionFunction[x][y])
        }
      }

      for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
          const value = currentData.acquisitionFunction[x][y] / maxVal
          ctx.fillStyle = `rgba(255, 165, 0, ${value * 0.5})`
          ctx.fillRect(x * cellWidth, (gridSize - y - 1) * cellHeight, cellWidth, cellHeight)
        }
      }
    }

    // Draw all points
    currentData.points.forEach((point: any, index: number) => {
      const x = point.x * canvas.width
      const y = (1 - point.y) * canvas.height // Flip y-axis
      const radius = 5 + point.uncertainty * 20

      // Draw uncertainty circle
      ctx.beginPath()
      ctx.arc(x, y, radius, 0, Math.PI * 2)
      ctx.fillStyle = "rgba(200, 200, 200, 0.3)"
      ctx.fill()

      // Draw point
      ctx.beginPath()
      ctx.arc(x, y, 5, 0, Math.PI * 2)
      ctx.fillStyle = index === currentData.points.length - 1 && currentStep > 0 ? "#ff0000" : "#3b82f6"
      ctx.fill()
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Label the point with its index
      ctx.fillStyle = "#000"
      ctx.font = "10px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText(index.toString(), x, y)
    })

    // Highlight the best point
    if (currentData.best) {
      const x = currentData.best.x * canvas.width
      const y = (1 - currentData.best.y) * canvas.height
      ctx.beginPath()
      ctx.arc(x, y, 8, 0, Math.PI * 2)
      ctx.strokeStyle = "#22c55e"
      ctx.lineWidth = 2
      ctx.stroke()
    }

    // Draw axes labels
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Parameter 1", canvas.width / 2, canvas.height - 5)

    ctx.save()
    ctx.translate(10, canvas.height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = "center"
    ctx.fillText("Parameter 2", 0, 0)
    ctx.restore()
  }, [currentStep, animationData])

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  const restart = () => {
    setCurrentStep(0)
    setIsPlaying(false)
  }

  const skipToEnd = () => {
    setCurrentStep(animationData.length - 1)
    setIsPlaying(false)
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex justify-between items-center">
          <div>
            <CardTitle>Optimization Process Animation</CardTitle>
            <CardDescription>Watch how the algorithm explores the parameter space</CardDescription>
          </div>
          <Badge variant="outline" className="font-mono">
            <Clock className="mr-1 h-3 w-3" />
            Iteration {currentStep} / {animationData.length - 1}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="border rounded-md p-2 bg-white">
            <canvas ref={canvasRef} width={width} height={height} className="w-full h-auto" />
          </div>

          <div className="flex items-center justify-between">
            <div className="flex space-x-2">
              <Button variant="outline" size="icon" onClick={restart}>
                <SkipBack className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="icon" onClick={togglePlayPause}>
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button variant="outline" size="icon" onClick={skipToEnd}>
                <SkipForward className="h-4 w-4" />
              </Button>
            </div>

            <div className="flex items-center space-x-2 w-1/3">
              <span className="text-sm">Speed:</span>
              <Slider value={[speed]} min={0.5} max={5} step={0.5} onValueChange={(value) => setSpeed(value[0])} />
              <span className="text-sm font-mono w-8">{speed}x</span>
            </div>

            <div className="w-1/3">
              <Slider
                value={[currentStep]}
                min={0}
                max={animationData.length - 1}
                step={1}
                onValueChange={(value) => {
                  setCurrentStep(value[0])
                  setIsPlaying(false)
                }}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="font-medium">Current Strategy:</p>
              <p className="text-muted-foreground">
                {currentStep < 10
                  ? "Latin Hypercube Sampling (Initial Exploration)"
                  : "Bayesian Optimization (Exploitation)"}
              </p>
            </div>
            <div>
              <p className="font-medium">Best Value Found:</p>
              <p className="text-muted-foreground">
                {animationData[currentStep]?.best?.value.toFixed(4) || "N/A"} at (
                {animationData[currentStep]?.best?.x.toFixed(2) || "N/A"},{" "}
                {animationData[currentStep]?.best?.y.toFixed(2) || "N/A"})
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
