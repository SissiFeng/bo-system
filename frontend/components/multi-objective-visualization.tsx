"use client"

import { useEffect, useRef } from "react"
import { type Objective, OptimizationType } from "@/lib/types"

interface MultiObjectiveVisualizationProps {
  objectives: Objective[]
}

export function MultiObjectiveVisualization({ objectives }: MultiObjectiveVisualizationProps) {
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
    const padding = 40

    // If we have less than 2 objectives, show a message
    if (objectives.length < 2) {
      ctx.fillStyle = "#888"
      ctx.font = "14px sans-serif"
      ctx.textAlign = "center"
      ctx.fillText("Add at least 2 objectives to visualize trade-offs", width / 2, height / 2)
      return
    }

    // Draw Pareto front visualization
    ctx.fillStyle = "#000"
    ctx.font = "16px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Multi-Objective Optimization Space", width / 2, padding - 10)

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

    // Get the first two objectives
    const obj1 = objectives[0]
    const obj2 = objectives[1]

    // Draw axis labels
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(
      `${obj1.name} (${obj1.type === OptimizationType.MAXIMIZE ? "↑" : obj1.type === OptimizationType.MINIMIZE ? "↓" : "→"})`,
      width / 2,
      height - 10,
    )

    ctx.save()
    ctx.translate(15, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = "center"
    ctx.fillText(
      `${obj2.name} (${obj2.type === OptimizationType.MAXIMIZE ? "↑" : obj2.type === OptimizationType.MINIMIZE ? "↓" : "→"})`,
      0,
      0,
    )
    ctx.restore()

    // Draw Pareto front
    ctx.beginPath()

    // Determine curve direction based on optimization types
    const isObj1Maximize = obj1.type === OptimizationType.MAXIMIZE
    const isObj2Maximize = obj2.type === OptimizationType.MAXIMIZE

    if ((isObj1Maximize && isObj2Maximize) || (!isObj1Maximize && !isObj2Maximize)) {
      // Both maximize or both minimize - convex Pareto front
      ctx.moveTo(padding, height - padding - (height - 2 * padding) * 0.8)
      ctx.bezierCurveTo(
        padding + (width - 2 * padding) * 0.3,
        height - padding - (height - 2 * padding) * 0.9,
        padding + (width - 2 * padding) * 0.7,
        height - padding - (height - 2 * padding) * 0.3,
        padding + (width - 2 * padding) * 0.9,
        height - padding,
      )
    } else {
      // One maximize, one minimize - concave Pareto front
      ctx.moveTo(padding, height - padding)
      ctx.bezierCurveTo(
        padding + (width - 2 * padding) * 0.3,
        height - padding - (height - 2 * padding) * 0.3,
        padding + (width - 2 * padding) * 0.7,
        height - padding - (height - 2 * padding) * 0.9,
        padding + (width - 2 * padding),
        height - padding - (height - 2 * padding) * 0.8,
      )
    }

    ctx.strokeStyle = "#3b82f6" // Blue
    ctx.lineWidth = 2
    ctx.stroke()
    ctx.lineWidth = 1

    // Draw Pareto front label
    ctx.fillStyle = "#3b82f6"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Pareto Front", padding + (width - 2 * padding) * 0.5, height - padding - (height - 2 * padding) * 0.5)

    // Draw sample points
    const pointCount = 20
    for (let i = 0; i < pointCount; i++) {
      const x = padding + Math.random() * (width - 2 * padding)
      const y = height - padding - Math.random() * (height - 2 * padding)

      // Determine if point is on/near Pareto front
      let isOnPareto = false
      if ((isObj1Maximize && isObj2Maximize) || (!isObj1Maximize && !isObj2Maximize)) {
        // Convex front
        const idealY =
          height - padding - (height - 2 * padding) * 0.8 * Math.pow(1 - (x - padding) / (width - 2 * padding), 2)
        isOnPareto = Math.abs(y - idealY) < 15
      } else {
        // Concave front
        const idealY =
          height - padding - (height - 2 * padding) * 0.8 * Math.pow((x - padding) / (width - 2 * padding), 2)
        isOnPareto = Math.abs(y - idealY) < 15
      }

      ctx.beginPath()
      ctx.arc(x, y, isOnPareto ? 5 : 3, 0, Math.PI * 2)
      ctx.fillStyle = isOnPareto ? "#3b82f6" : "#888"
      ctx.fill()
    }

    // If we have more than 2 objectives, add a note
    if (objectives.length > 2) {
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "left"
      ctx.fillText(`+ ${objectives.length - 2} more objectives`, padding, padding - 10)
    }
  }, [objectives])

  return (
    <div className="mt-4 border rounded-md p-2 bg-white">
      <canvas ref={canvasRef} width={500} height={400} className="w-full h-auto" />
    </div>
  )
}
