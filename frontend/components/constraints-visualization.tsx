"use client"

import { useEffect, useRef } from "react"
import { type Constraint, ConstraintType } from "@/lib/types"

interface ConstraintsVisualizationProps {
  constraints: Constraint[]
}

export function ConstraintsVisualization({ constraints }: ConstraintsVisualizationProps) {
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

    // If we have no constraints, show a message
    if (constraints.length === 0) {
      ctx.fillStyle = "#888"
      ctx.font = "14px sans-serif"
      ctx.textAlign = "center"
      ctx.fillText("Add constraints to visualize the feasible region", width / 2, height / 2)
      return
    }

    // Draw a 2D parameter space with constraints
    ctx.fillStyle = "#000"
    ctx.font = "16px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Constraint Visualization", width / 2, padding - 10)

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
    ctx.fillText("Parameter 1", width / 2, height - 10)

    ctx.save()
    ctx.translate(15, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = "center"
    ctx.fillText("Parameter 2", 0, 0)
    ctx.restore()

    // Draw grid
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
    const colors = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6"]

    constraints.forEach((constraint, index) => {
      const color = colors[index % colors.length]

      // For simplicity, we'll visualize each constraint as a line
      // In a real implementation, you'd parse the expression and draw accordingly

      // Generate random line parameters for demonstration
      const startX = padding
      const startY = height - padding - Math.random() * (height - 2 * padding) * 0.8
      const endX = width - padding
      const endY = height - padding - Math.random() * (height - 2 * padding) * 0.8

      ctx.beginPath()
      ctx.moveTo(startX, startY)
      ctx.lineTo(endX, endY)

      if (constraint.type === ConstraintType.SUM_EQUALS) {
        ctx.strokeStyle = color
        ctx.setLineDash([])
      } else if (constraint.type === ConstraintType.SUM_LESS_THAN) {
        ctx.strokeStyle = color
        ctx.setLineDash([5, 5])

        // Fill area below line
        ctx.lineTo(endX, height - padding)
        ctx.lineTo(startX, height - padding)
        ctx.closePath()
        ctx.fillStyle = `${color}20` // 20% opacity
        ctx.fill()

        ctx.beginPath()
        ctx.moveTo(startX, startY)
        ctx.lineTo(endX, endY)
      } else if (constraint.type === ConstraintType.SUM_GREATER_THAN) {
        ctx.strokeStyle = color
        ctx.setLineDash([5, 5])

        // Fill area above line
        ctx.lineTo(endX, padding)
        ctx.lineTo(startX, padding)
        ctx.closePath()
        ctx.fillStyle = `${color}20` // 20% opacity
        ctx.fill()

        ctx.beginPath()
        ctx.moveTo(startX, startY)
        ctx.lineTo(endX, endY)
      }

      ctx.stroke()
      ctx.setLineDash([])

      // Add constraint label
      const labelX = padding + (width - 2 * padding) * 0.3 + index * 20
      const labelY = height - padding - (height - 2 * padding) * 0.8 + index * 20

      ctx.fillStyle = color
      ctx.font = "12px sans-serif"
      ctx.textAlign = "left"

      const constraintSymbol =
        constraint.type === ConstraintType.SUM_EQUALS
          ? "="
          : constraint.type === ConstraintType.SUM_LESS_THAN
            ? "≤"
            : "≥"

      ctx.fillText(`${constraint.expression} ${constraintSymbol} ${constraint.value}`, labelX, labelY)
    })

    // Draw feasible region (intersection of all constraints)
    // This is a simplified visualization - in reality you'd compute the actual intersection
    if (constraints.length > 1) {
      // Draw a polygon representing the feasible region
      ctx.beginPath()
      ctx.moveTo(padding + (width - 2 * padding) * 0.3, height - padding - (height - 2 * padding) * 0.3)
      ctx.lineTo(padding + (width - 2 * padding) * 0.7, height - padding - (height - 2 * padding) * 0.3)
      ctx.lineTo(padding + (width - 2 * padding) * 0.5, height - padding - (height - 2 * padding) * 0.7)
      ctx.closePath()

      ctx.fillStyle = "rgba(59, 130, 246, 0.2)" // Light blue
      ctx.fill()
      ctx.strokeStyle = "rgba(59, 130, 246, 0.8)" // Blue
      ctx.stroke()

      // Add feasible region label
      ctx.fillStyle = "#3b82f6"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.fillText(
        "Feasible Region",
        padding + (width - 2 * padding) * 0.5,
        height - padding - (height - 2 * padding) * 0.5,
      )
    }
  }, [constraints])

  return (
    <div className="mt-4 border rounded-md p-2 bg-white">
      <canvas ref={canvasRef} width={500} height={400} className="w-full h-auto" />
    </div>
  )
}
