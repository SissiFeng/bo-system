"use client"

import { useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { AlertTriangle } from "lucide-react"

interface PredictionRealityVisualizationProps {
  targetValue: number
  predictedValue: number
  uncertainty: number
  historicalRange: [number, number]
  experimentalValues?: number[]
}

export function PredictionRealityVisualization({
  targetValue,
  predictedValue,
  uncertainty,
  historicalRange,
  experimentalValues,
}: PredictionRealityVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Calculate confidence level based on how close the historical range is to the prediction uncertainty
  const calculateConfidence = (): { level: "high" | "medium" | "low"; message: string } => {
    const predictionRange = uncertainty * 2 // ±uncertainty
    const historicalRangeSize = historicalRange[1] - historicalRange[0]
    const ratio = historicalRangeSize / predictionRange

    if (ratio <= 3) {
      return { level: "high", message: "High confidence in prediction" }
    } else if (ratio <= 6) {
      return { level: "medium", message: "Moderate confidence in prediction" }
    } else {
      return { level: "low", message: "Low confidence - high experimental variability" }
    }
  }

  const confidence = calculateConfidence()

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
    const padding = { top: 40, right: 30, bottom: 60, left: 50 }
    const plotWidth = width - padding.left - padding.right
    const plotHeight = height - padding.top - padding.bottom

    // Calculate scales
    const minValue = Math.min(historicalRange[0], targetValue - 3 * uncertainty)
    const maxValue = Math.max(historicalRange[1], targetValue + 3 * uncertainty)
    const valueRange = maxValue - minValue

    const valueToX = (value: number): number => {
      return padding.left + ((value - minValue) / valueRange) * plotWidth
    }

    // Draw axes
    ctx.beginPath()
    ctx.moveTo(padding.left, height - padding.bottom)
    ctx.lineTo(width - padding.right, height - padding.bottom)
    ctx.strokeStyle = "#888"
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw target line
    const targetX = valueToX(targetValue)
    ctx.beginPath()
    ctx.moveTo(targetX, padding.top)
    ctx.lineTo(targetX, height - padding.bottom)
    ctx.strokeStyle = "#10b981" // Green
    ctx.lineWidth = 2
    ctx.setLineDash([5, 3])
    ctx.stroke()
    ctx.setLineDash([])

    // Draw target label
    ctx.fillStyle = "#10b981"
    ctx.font = "bold 12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(`Target ${targetValue}`, targetX, padding.top - 10)

    // Draw normal distribution for prediction
    const predictedX = valueToX(predictedValue)
    const stdDevInPixels = (uncertainty / valueRange) * plotWidth

    // Draw prediction curve
    ctx.beginPath()
    for (let x = padding.left; x <= width - padding.right; x++) {
      const value = minValue + ((x - padding.left) / plotWidth) * valueRange
      const y =
        height -
        padding.bottom -
        plotHeight * 0.4 * Math.exp(-Math.pow(value - predictedValue, 2) / (2 * Math.pow(uncertainty, 2)))
      if (x === padding.left) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    }
    ctx.strokeStyle = "#6366f1" // Indigo
    ctx.lineWidth = 2
    ctx.stroke()

    // Fill area under prediction curve
    ctx.lineTo(width - padding.right, height - padding.bottom)
    ctx.lineTo(padding.left, height - padding.bottom)
    ctx.closePath()
    ctx.fillStyle = "rgba(99, 102, 241, 0.2)" // Light indigo
    ctx.fill()

    // Draw prediction uncertainty range
    const lowerBoundX = valueToX(predictedValue - uncertainty)
    const upperBoundX = valueToX(predictedValue + uncertainty)

    ctx.beginPath()
    ctx.moveTo(lowerBoundX, height - padding.bottom - plotHeight * 0.4)
    ctx.lineTo(lowerBoundX, height - padding.bottom)
    ctx.strokeStyle = "#6366f1" // Indigo
    ctx.lineWidth = 1
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(upperBoundX, height - padding.bottom - plotHeight * 0.4)
    ctx.lineTo(upperBoundX, height - padding.bottom)
    ctx.strokeStyle = "#6366f1" // Indigo
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw prediction labels
    ctx.fillStyle = "#6366f1"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(`${predictedValue - uncertainty}`, lowerBoundX, height - padding.bottom + 15)
    ctx.fillText(`${predictedValue + uncertainty}`, upperBoundX, height - padding.bottom + 15)
    ctx.fillText(`${predictedValue}`, predictedX, height - padding.bottom + 15)

    // Draw historical range
    const historicalLowerX = valueToX(historicalRange[0])
    const historicalUpperX = valueToX(historicalRange[1])

    // Draw historical range bar
    const histogramHeight = 20
    const histogramY = height - padding.bottom + 30

    ctx.fillStyle = "rgba(234, 88, 12, 0.2)" // Light orange
    ctx.fillRect(historicalLowerX, histogramY, historicalUpperX - historicalLowerX, histogramHeight)

    ctx.beginPath()
    ctx.rect(historicalLowerX, histogramY, historicalUpperX - historicalLowerX, histogramHeight)
    ctx.strokeStyle = "#ea580c" // Orange
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw historical range labels
    ctx.fillStyle = "#ea580c"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(`${historicalRange[0]}`, historicalLowerX, histogramY + histogramHeight + 15)
    ctx.fillText(`${historicalRange[1]}`, historicalUpperX, histogramY + histogramHeight + 15)
    ctx.fillText("Historical Range", (historicalLowerX + historicalUpperX) / 2, histogramY + histogramHeight / 2 + 4)

    // Draw experimental values if provided
    if (experimentalValues && experimentalValues.length > 0) {
      experimentalValues.forEach((value) => {
        const x = valueToX(value)

        // Draw circle for experimental value
        ctx.beginPath()
        ctx.arc(x, histogramY + histogramHeight / 2, 4, 0, 2 * Math.PI)
        ctx.fillStyle = "#000"
        ctx.fill()
      })
    }

    // Draw x-axis label
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Value", width / 2, height - 5)
  }, [targetValue, predictedValue, uncertainty, historicalRange, experimentalValues])

  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-start">
          <div>
            <CardTitle>Prediction vs. Reality</CardTitle>
            <CardDescription>Model prediction compared to experimental variability</CardDescription>
          </div>
          <Badge
            variant="outline"
            className={
              confidence.level === "high"
                ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300"
                : confidence.level === "medium"
                  ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300"
                  : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300"
            }
          >
            {confidence.level === "low" && <AlertTriangle className="h-3 w-3 mr-1 inline" />}
            {confidence.level.charAt(0).toUpperCase() + confidence.level.slice(1)} Confidence
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex flex-col items-center">
            <canvas ref={canvasRef} width={600} height={300} className="w-full h-auto" />
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="font-medium">Prediction:</p>
              <p>
                {predictedValue} ± {uncertainty} (σ)
              </p>
              <p className="text-muted-foreground">
                Confidence interval: [{predictedValue - uncertainty}, {predictedValue + uncertainty}]
              </p>
            </div>
            <div>
              <p className="font-medium">Historical Measurements:</p>
              <p>
                Range: [{historicalRange[0]}, {historicalRange[1]}]
              </p>
              <p className="text-muted-foreground">{confidence.message}</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
