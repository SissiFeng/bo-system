"use client"

import { useEffect, useRef } from "react"
import { type Parameter, ParameterType } from "@/lib/types"

interface ParameterVisualizationProps {
  parameter: Parameter
}

export function ParameterVisualization({ parameter }: ParameterVisualizationProps) {
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

    // Draw based on parameter type
    if (parameter.type === ParameterType.CONTINUOUS || parameter.type === ParameterType.DISCRETE) {
      drawNumericalParameter(ctx, parameter, width, height, padding)
    } else if (parameter.type === ParameterType.CATEGORICAL) {
      drawCategoricalParameter(ctx, parameter, width, height, padding)
    }
  }, [parameter])

  const drawNumericalParameter = (
    ctx: CanvasRenderingContext2D,
    parameter: Parameter,
    width: number,
    height: number,
    padding: number,
  ) => {
    const min = parameter.min ?? 0
    const max = parameter.max ?? 1
    const range = max - min

    // Draw axis
    ctx.beginPath()
    ctx.moveTo(padding, height / 2)
    ctx.lineTo(width - padding, height / 2)
    ctx.strokeStyle = "#888"
    ctx.stroke()

    // Draw ticks and values
    const numTicks = parameter.type === ParameterType.DISCRETE ? Math.min(10, Math.floor(range) + 1) : 5
    for (let i = 0; i <= numTicks; i++) {
      const x = padding + ((width - 2 * padding) * i) / numTicks
      const value = min + (range * i) / numTicks

      // Draw tick
      ctx.beginPath()
      ctx.moveTo(x, height / 2 - 5)
      ctx.lineTo(x, height / 2 + 5)
      ctx.strokeStyle = "#888"
      ctx.stroke()

      // Draw value
      ctx.fillStyle = "#888"
      ctx.font = "10px sans-serif"
      ctx.textAlign = "center"
      ctx.fillText(
        parameter.type === ParameterType.DISCRETE ? Math.round(value).toString() : value.toFixed(1),
        x,
        height / 2 + 20,
      )
    }

    // Draw distribution
    ctx.beginPath()
    if (parameter.type === ParameterType.CONTINUOUS) {
      // Draw bell curve for continuous parameters
      for (let x = padding; x <= width - padding; x++) {
        const normalizedX = (x - padding) / (width - 2 * padding)
        const paramValue = min + normalizedX * range
        const midpoint = (min + max) / 2
        const y =
          height / 2 -
          40 * Math.exp(-Math.pow((paramValue - midpoint) / (range / 4), 2) / 2) -
          5 * Math.sin((x / width) * Math.PI * 4)
        if (x === padding) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      }
    } else {
      // Draw bars for discrete parameters
      const step = parameter.step ?? 1
      const numSteps = Math.floor(range / step) + 1
      const barWidth = (width - 2 * padding) / numSteps

      for (let i = 0; i < numSteps; i++) {
        const x = padding + i * barWidth
        const barHeight = 30 + Math.random() * 20
        ctx.rect(x, height / 2 - barHeight, barWidth * 0.8, barHeight)
      }
    }

    ctx.strokeStyle = "#3b82f6"
    ctx.stroke()
    ctx.fillStyle = "rgba(59, 130, 246, 0.2)"
    ctx.fill()

    // Draw parameter name and range
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(
      `${parameter.name} (${parameter.type === ParameterType.CONTINUOUS ? "Continuous" : "Discrete"})`,
      width / 2,
      padding,
    )
    ctx.fillText(`Range: ${min} to ${max}${parameter.unit ? ` ${parameter.unit}` : ""}`, width / 2, height - 5)
  }

  const drawCategoricalParameter = (
    ctx: CanvasRenderingContext2D,
    parameter: Parameter,
    width: number,
    height: number,
    padding: number,
  ) => {
    const values = parameter.values || []
    if (values.length === 0) {
      ctx.fillStyle = "#888"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.fillText("No categorical values defined", width / 2, height / 2)
      return
    }

    // Draw bars for each category
    const barWidth = (width - 2 * padding) / values.length
    const maxBarHeight = height - 2 * padding - 40 // Leave space for labels

    for (let i = 0; i < values.length; i++) {
      const x = padding + i * barWidth
      const barHeight = 20 + Math.random() * maxBarHeight
      const hue = (i * 360) / values.length

      // Draw bar
      ctx.fillStyle = `hsla(${hue}, 70%, 60%, 0.7)`
      ctx.fillRect(x, height - padding - barHeight, barWidth * 0.8, barHeight)

      // Draw category label
      ctx.fillStyle = "#000"
      ctx.font = "10px sans-serif"
      ctx.textAlign = "center"
      ctx.fillText(values[i].toString(), x + barWidth * 0.4, height - padding + 15)
    }

    // Draw parameter name
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(`${parameter.name} (Categorical)`, width / 2, padding)
  }

  return (
    <div className="border rounded-md p-2 bg-white">
      <canvas ref={canvasRef} width={300} height={150} className="w-full h-auto" />
    </div>
  )
}
