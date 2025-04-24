"use client"

import { useEffect, useRef } from "react"
import { type Constraint, ConstraintType } from "@/lib/types"

interface ConstraintVisualizationProps {
  constraint: Constraint
}

export function ConstraintVisualization({ constraint }: ConstraintVisualizationProps) {
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

    // Draw 2D parameter space with constraint
    drawConstraint(ctx, constraint, width, height, padding)
  }, [constraint])

  const drawConstraint = (
    ctx: CanvasRenderingContext2D,
    constraint: Constraint,
    width: number,
    height: number,
    padding: number,
  ) => {
    // Draw axes
    drawAxes(ctx, width, height, padding)

    // Parse the expression to determine visualization
    const expression = constraint.expression
    const value = constraint.value

    // Draw constraint line
    const lineY = height / 2
    ctx.beginPath()
    ctx.moveTo(padding, lineY)
    ctx.lineTo(width - padding, lineY)
    ctx.strokeStyle = "#888"
    ctx.setLineDash([5, 5])
    ctx.stroke()
    ctx.setLineDash([])

    // Draw constraint value marker
    const markerX = padding + (width - 2 * padding) * 0.6 // Arbitrary position for visualization
    const markerRadius = 6

    ctx.beginPath()
    ctx.arc(markerX, lineY, markerRadius, 0, Math.PI * 2)
    ctx.fillStyle = "#3b82f6"
    ctx.fill()

    // Draw constraint type visualization
    switch (constraint.type) {
      case ConstraintType.SUM_EQUALS:
        // Draw equals sign
        ctx.beginPath()
        ctx.moveTo(markerX + 20, lineY - 5)
        ctx.lineTo(markerX + 40, lineY - 5)
        ctx.moveTo(markerX + 20, lineY + 5)
        ctx.lineTo(markerX + 40, lineY + 5)
        ctx.strokeStyle = "#000"
        ctx.lineWidth = 2
        ctx.stroke()
        ctx.lineWidth = 1
        break

      case ConstraintType.SUM_LESS_THAN:
        // Draw less than sign and fill lower region
        ctx.beginPath()
        ctx.moveTo(markerX + 40, lineY - 10)
        ctx.lineTo(markerX + 20, lineY)
        ctx.lineTo(markerX + 40, lineY + 10)
        ctx.strokeStyle = "#000"
        ctx.lineWidth = 2
        ctx.stroke()
        ctx.lineWidth = 1

        // Fill the "less than" region
        ctx.beginPath()
        ctx.rect(padding, lineY, width - 2 * padding, height - padding - lineY)
        ctx.fillStyle = "rgba(59, 130, 246, 0.1)" // Light blue
        ctx.fill()
        break

      case ConstraintType.SUM_GREATER_THAN:
        // Draw greater than sign and fill upper region
        ctx.beginPath()
        ctx.moveTo(markerX + 20, lineY - 10)
        ctx.lineTo(markerX + 40, lineY)
        ctx.lineTo(markerX + 20, lineY + 10)
        ctx.strokeStyle = "#000"
        ctx.lineWidth = 2
        ctx.stroke()
        ctx.lineWidth = 1

        // Fill the "greater than" region
        ctx.beginPath()
        ctx.rect(padding, padding, width - 2 * padding, lineY - padding)
        ctx.fillStyle = "rgba(59, 130, 246, 0.1)" // Light blue
        ctx.fill()
        break
    }

    // Draw expression and value
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(`${expression} ${getConstraintSymbol(constraint.type)} ${value}`, width / 2, padding - 5)

    // Draw explanation based on constraint type
    let explanation = ""
    switch (constraint.type) {
      case ConstraintType.SUM_EQUALS:
        explanation = "Parameters must sum exactly to the target value"
        break
      case ConstraintType.SUM_LESS_THAN:
        explanation = "Parameters must sum to less than the target value"
        break
      case ConstraintType.SUM_GREATER_THAN:
        explanation = "Parameters must sum to greater than the target value"
        break
    }

    ctx.fillStyle = "#666"
    ctx.font = "10px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(explanation, width / 2, height - 5)
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

    // Draw axis labels
    ctx.fillStyle = "#888"
    ctx.font = "10px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Parameter Space", width / 2, height - padding + 15)

    ctx.save()
    ctx.translate(padding - 15, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = "center"
    ctx.fillText("Constraint", 0, 0)
    ctx.restore()
  }

  const getConstraintSymbol = (type: ConstraintType) => {
    switch (type) {
      case ConstraintType.SUM_EQUALS:
        return "="
      case ConstraintType.SUM_LESS_THAN:
        return "≤"
      case ConstraintType.SUM_GREATER_THAN:
        return "≥"
      default:
        return "="
    }
  }

  return (
    <div className="border rounded-md p-2 bg-white">
      <canvas ref={canvasRef} width={300} height={150} className="w-full h-auto" />
    </div>
  )
}
