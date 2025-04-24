"use client"

import { useEffect, useRef } from "react"
import { type Parameter, ParameterType } from "@/lib/types"

interface ParameterSpaceVisualizationProps {
  parameters: Parameter[]
}

export function ParameterSpaceVisualization({ parameters }: ParameterSpaceVisualizationProps) {
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

    // Draw a sample feasible region (for illustration)
    ctx.beginPath()
    ctx.moveTo(padding + (width - 2 * padding) * 0.2, height - padding)
    ctx.bezierCurveTo(
      padding + (width - 2 * padding) * 0.2,
      height - padding - (height - 2 * padding) * 0.6,
      padding + (width - 2 * padding) * 0.8,
      height - padding - (height - 2 * padding) * 0.6,
      padding + (width - 2 * padding) * 0.8,
      height - padding,
    )
    ctx.lineTo(padding + (width - 2 * padding) * 0.2, height - padding)
    ctx.fillStyle = "rgba(59, 130, 246, 0.1)" // Light blue
    ctx.fill()
    ctx.strokeStyle = "rgba(59, 130, 246, 0.5)" // Blue
    ctx.stroke()

    // Add legend for feasible region
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Sample Feasible Region", width / 2, height - padding - (height - 2 * padding) * 0.3)
  }, [parameters])

  return (
    <div className="mt-4 border rounded-md p-2 bg-white">
      <canvas ref={canvasRef} width={500} height={400} className="w-full h-auto" />
    </div>
  )
}
