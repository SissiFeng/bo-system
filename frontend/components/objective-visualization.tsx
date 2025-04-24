"use client"

import { useEffect, useRef } from "react"
import { type Objective, OptimizationType } from "@/lib/types"

interface ObjectiveVisualizationProps {
  objective: Objective
}

export function ObjectiveVisualization({ objective }: ObjectiveVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

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
    const padding = 20

    // Draw based on objective type
    if (objective.type === OptimizationType.MAXIMIZE) {
      drawMaximizeObjective(ctx, objective, width, height, padding)
    } else if (objective.type === OptimizationType.MINIMIZE) {
      drawMinimizeObjective(ctx, objective, width, height, padding)
    } else if (objective.type === OptimizationType.TARGET_RANGE) {
      drawTargetRangeObjective(ctx, objective, width, height, padding)
    }
  }, [objective])

  const drawMaximizeObjective = (
    ctx: CanvasRenderingContext2D,
    objective: Objective,
    width: number,
    height: number,
    padding: number,
  ) => {
    // Draw axes
    drawAxes(ctx, width, height, padding)

    // Draw increasing utility function
    ctx.beginPath()
    ctx.moveTo(padding, height - padding)

    // Draw a curve that increases from bottom-left to top-right
    for (let x = padding; x <= width - padding; x++) {
      const normalizedX = (x - padding) / (width - 2 * padding)
      // Sigmoid-like function
      const y = height - padding - (height - 2 * padding) / (1 + Math.exp(-10 * (normalizedX - 0.5)))
      ctx.lineTo(x, y)
    }

    ctx.strokeStyle = "#22c55e" // Green
    ctx.lineWidth = 2
    ctx.stroke()

    // Fill area under curve
    ctx.lineTo(width - padding, height - padding)
    ctx.lineTo(padding, height - padding)
    ctx.closePath()
    ctx.fillStyle = "rgba(34, 197, 94, 0.2)" // Light green
    ctx.fill()

    // Draw arrow indicating "higher is better"
    drawArrow(ctx, width - padding - 40, height - padding - 40, width - padding - 20, height - padding - 60, "#22c55e")

    // Draw labels
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(`${objective.name} (Maximize)`, width / 2, padding - 5)
    ctx.fillText("Higher is better", width - padding - 60, height - padding - 70)

    // Draw x-axis label
    ctx.fillText("Parameter Value", width / 2, height - 5)

    // Draw y-axis label
    ctx.save()
    ctx.translate(15, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = "center"
    ctx.fillText("Objective Value", 0, 0)
    ctx.restore()
  }

  const drawMinimizeObjective = (
    ctx: CanvasRenderingContext2D,
    objective: Objective,
    width: number,
    height: number,
    padding: number,
  ) => {
    // Draw axes
    drawAxes(ctx, width, height, padding)

    // Draw decreasing utility function
    ctx.beginPath()
    ctx.moveTo(padding, padding)

    // Draw a curve that decreases from top-left to bottom-right
    for (let x = padding; x <= width - padding; x++) {
      const normalizedX = (x - padding) / (width - 2 * padding)
      // Inverse sigmoid-like function
      const y = padding + (height - 2 * padding) / (1 + Math.exp(-10 * (normalizedX - 0.5)))
      ctx.lineTo(x, y)
    }

    ctx.strokeStyle = "#ef4444" // Red
    ctx.lineWidth = 2
    ctx.stroke()

    // Fill area under curve
    ctx.lineTo(width - padding, height - padding)
    ctx.lineTo(padding, height - padding)
    ctx.lineTo(padding, padding)
    ctx.closePath()
    ctx.fillStyle = "rgba(239, 68, 68, 0.2)" // Light red
    ctx.fill()

    // Draw arrow indicating "lower is better"
    drawArrow(ctx, width - padding - 40, padding + 60, width - padding - 20, padding + 40, "#ef4444")

    // Draw labels
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(`${objective.name} (Minimize)`, width / 2, padding - 5)
    ctx.fillText("Lower is better", width - padding - 60, padding + 30)

    // Draw x-axis label
    ctx.fillText("Parameter Value", width / 2, height - 5)

    // Draw y-axis label
    ctx.save()
    ctx.translate(15, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = "center"
    ctx.fillText("Objective Value", 0, 0)
    ctx.restore()
  }

  const drawTargetRangeObjective = (
    ctx: CanvasRenderingContext2D,
    objective: Objective,
    width: number,
    height: number,
    padding: number,
  ) => {
    // Draw axes
    drawAxes(ctx, width, height, padding)

    const targetMin = objective.targetMin ?? 0
    const targetMax = objective.targetMax ?? 1

    // Calculate positions for target range
    const rangeStart = padding + (width - 2 * padding) * 0.3
    const rangeEnd = padding + (width - 2 * padding) * 0.7

    // Draw bell-shaped utility function
    ctx.beginPath()

    for (let x = padding; x <= width - padding; x++) {
      const normalizedX = (x - padding) / (width - 2 * padding)
      let y

      if (x >= rangeStart && x <= rangeEnd) {
        // Flat top in the target range
        y = padding + 20
      } else {
        // Bell curve tails
        const distToRange = x < rangeStart ? rangeStart - x : x - rangeEnd
        const normalizedDist = distToRange / (width - 2 * padding)
        y = padding + 20 + (height - 2 * padding - 20) * (1 - Math.exp(-10 * normalizedDist))
      }

      if (x === padding) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    }

    ctx.strokeStyle = "#8b5cf6" // Purple
    ctx.lineWidth = 2
    ctx.stroke()

    // Fill area under curve
    ctx.lineTo(width - padding, height - padding)
    ctx.lineTo(padding, height - padding)
    ctx.closePath()
    ctx.fillStyle = "rgba(139, 92, 246, 0.2)" // Light purple
    ctx.fill()

    // Draw target range rectangle
    ctx.fillStyle = "rgba(139, 92, 246, 0.3)" // Semi-transparent purple
    ctx.fillRect(rangeStart, padding, rangeEnd - rangeStart, height - 2 * padding)

    // Draw target range labels
    ctx.fillStyle = "#000"
    ctx.font = "10px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(targetMin.toString(), rangeStart, height - padding + 15)
    ctx.fillText(targetMax.toString(), rangeEnd, height - padding + 15)

    // Draw labels
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(`${objective.name} (Target Range)`, width / 2, padding - 5)
    ctx.fillText("Target Zone", (rangeStart + rangeEnd) / 2, padding + 15)

    // Draw x-axis label
    ctx.fillText("Parameter Value", width / 2, height - 5)

    // Draw y-axis label
    ctx.save()
    ctx.translate(15, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = "center"
    ctx.fillText("Objective Value", 0, 0)
    ctx.restore()
  }

  const drawAxes = (ctx: CanvasRenderingContext2D, width: number, height: number, padding: number) => {
    // Draw x-axis
    ctx.beginPath()
    ctx.moveTo(padding, height - padding)
    ctx.lineTo(width - padding, height - padding)
    ctx.strokeStyle = "#888"
    ctx.stroke()

    // Draw y-axis
    ctx.beginPath()
    ctx.moveTo(padding, height - padding)
    ctx.lineTo(padding, padding)
    ctx.strokeStyle = "#888"
    ctx.stroke()

    // Draw ticks on x-axis
    for (let i = 0; i <= 5; i++) {
      const x = padding + ((width - 2 * padding) * i) / 5

      // Draw tick
      ctx.beginPath()
      ctx.moveTo(x, height - padding)
      ctx.lineTo(x, height - padding + 5)
      ctx.strokeStyle = "#888"
      ctx.stroke()
    }

    // Draw ticks on y-axis
    for (let i = 0; i <= 5; i++) {
      const y = height - padding - ((height - 2 * padding) * i) / 5

      // Draw tick
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(padding - 5, y)
      ctx.strokeStyle = "#888"
      ctx.stroke()
    }
  }

  const drawArrow = (
    ctx: CanvasRenderingContext2D,
    fromX: number,
    fromY: number,
    toX: number,
    toY: number,
    color: string,
  ) => {
    const headLength = 10
    const angle = Math.atan2(toY - fromY, toX - fromX)

    ctx.beginPath()
    ctx.moveTo(fromX, fromY)
    ctx.lineTo(toX, toY)
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.stroke()

    // Draw arrowhead
    ctx.beginPath()
    ctx.moveTo(toX, toY)
    ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI / 6), toY - headLength * Math.sin(angle - Math.PI / 6))
    ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI / 6), toY - headLength * Math.sin(angle + Math.PI / 6))
    ctx.closePath()
    ctx.fillStyle = color
    ctx.fill()
  }

  return (
    <div className="border rounded-md p-2 bg-white">
      <canvas ref={canvasRef} width={300} height={150} className="w-full h-auto" />
    </div>
  )
}
