"use client"

import { useEffect, useRef } from "react"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Info } from "lucide-react"

interface Solution {
  x: number
  y: number
  id: number
  uncertainty?: number
}

interface ObjectiveInfo {
  name: string
  type: "maximize" | "minimize"
}

interface ParetoFrontVisualizationProps {
  paretoFront: Solution[]
  nonDominatedSolutions?: Solution[]
  dominatedSolutions?: Solution[]
  objective1: ObjectiveInfo
  objective2: ObjectiveInfo
  selectedSolution?: number
  onSelectSolution?: (id: number) => void
}

export function ParetoFrontVisualization({
  paretoFront,
  nonDominatedSolutions = [],
  dominatedSolutions = [],
  objective1,
  objective2,
  selectedSolution,
  onSelectSolution,
}: ParetoFrontVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Adjust x and y values based on objective types
  const adjustValue = (value: number, type: "maximize" | "minimize") => {
    return type === "maximize" ? value : 1 - value
  }

  const adjustedParetoFront = paretoFront.map((solution) => ({
    ...solution,
    x: adjustValue(solution.x, objective1.type),
    y: adjustValue(solution.y, objective2.type),
  }))

  const adjustedNonDominated = nonDominatedSolutions.map((solution) => ({
    ...solution,
    x: adjustValue(solution.x, objective1.type),
    y: adjustValue(solution.y, objective2.type),
  }))

  const adjustedDominated = dominatedSolutions.map((solution) => ({
    ...solution,
    x: adjustValue(solution.x, objective1.type),
    y: adjustValue(solution.y, objective2.type),
  }))

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
    const padding = { top: 40, right: 40, bottom: 60, left: 60 }
    const plotWidth = width - padding.left - padding.right
    const plotHeight = height - padding.top - padding.bottom

    // Find min and max values
    const allPoints = [...adjustedParetoFront, ...adjustedNonDominated, ...adjustedDominated]
    const minX = Math.min(...allPoints.map((p) => p.x)) - 0.05
    const maxX = Math.max(...allPoints.map((p) => p.x)) + 0.05
    const minY = Math.min(...allPoints.map((p) => p.y)) - 0.05
    const maxY = Math.max(...allPoints.map((p) => p.y)) + 0.05

    // Scale functions
    const scaleX = (x: number) => padding.left + ((x - minX) / (maxX - minX)) * plotWidth
    const scaleY = (y: number) => height - padding.bottom - ((y - minY) / (maxY - minY)) * plotHeight

    // Draw axes
    ctx.beginPath()
    ctx.moveTo(padding.left, height - padding.bottom)
    ctx.lineTo(width - padding.right, height - padding.bottom)
    ctx.strokeStyle = "#888"
    ctx.lineWidth = 1
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(padding.left, height - padding.bottom)
    ctx.lineTo(padding.left, padding.top)
    ctx.strokeStyle = "#888"
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw grid lines
    ctx.setLineDash([2, 2])
    for (let i = 0; i <= 5; i++) {
      const x = padding.left + (i / 5) * plotWidth
      ctx.beginPath()
      ctx.moveTo(x, height - padding.bottom)
      ctx.lineTo(x, padding.top)
      ctx.strokeStyle = "#ddd"
      ctx.lineWidth = 1
      ctx.stroke()

      const y = height - padding.bottom - (i / 5) * plotHeight
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(width - padding.right, y)
      ctx.strokeStyle = "#ddd"
      ctx.lineWidth = 1
      ctx.stroke()
    }
    ctx.setLineDash([])

    // Draw axis labels
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.textBaseline = "middle"

    // X-axis label
    ctx.fillText(
      `${objective1.name} (${objective1.type === "maximize" ? "Higher is better" : "Lower is better"})`,
      padding.left + plotWidth / 2,
      height - padding.bottom / 2,
    )

    // Y-axis label
    ctx.save()
    ctx.translate(padding.left / 3, padding.top + plotHeight / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText(
      `${objective2.name} (${objective2.type === "maximize" ? "Higher is better" : "Lower is better"})`,
      0,
      0,
    )
    ctx.restore()

    // Draw axis ticks and values
    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    for (let i = 0; i <= 5; i++) {
      const x = padding.left + (i / 5) * plotWidth
      const value = minX + (i / 5) * (maxX - minX)
      ctx.fillText(value.toFixed(2), x, height - padding.bottom + 10)
    }

    ctx.textAlign = "right"
    ctx.textBaseline = "middle"
    for (let i = 0; i <= 5; i++) {
      const y = height - padding.bottom - (i / 5) * plotHeight
      const value = minY + (i / 5) * (maxY - minY)
      ctx.fillText(value.toFixed(2), padding.left - 10, y)
    }

    // Draw dominated solutions
    adjustedDominated.forEach((solution) => {
      const x = scaleX(solution.x)
      const y = scaleY(solution.y)

      ctx.beginPath()
      ctx.arc(x, y, 5, 0, 2 * Math.PI)
      ctx.fillStyle = "rgba(156, 163, 175, 0.5)" // Gray
      ctx.fill()
      ctx.strokeStyle = "#9ca3af"
      ctx.lineWidth = 1
      ctx.stroke()
    })

    // Draw non-dominated solutions
    adjustedNonDominated.forEach((solution) => {
      const x = scaleX(solution.x)
      const y = scaleY(solution.y)

      ctx.beginPath()
      ctx.arc(x, y, 5, 0, 2 * Math.PI)
      ctx.fillStyle = "rgba(59, 130, 246, 0.5)" // Blue
      ctx.fill()
      ctx.strokeStyle = "#3b82f6"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw uncertainty if available
      if (solution.uncertainty) {
        const radius = solution.uncertainty * 50 // Scale uncertainty for visualization
        ctx.beginPath()
        ctx.arc(x, y, radius, 0, 2 * Math.PI)
        ctx.strokeStyle = "rgba(59, 130, 246, 0.3)"
        ctx.lineWidth = 1
        ctx.stroke()
      }
    })

    // Draw Pareto front line
    if (adjustedParetoFront.length > 1) {
      // Sort points by x-coordinate
      const sortedFront = [...adjustedParetoFront].sort((a, b) => a.x - b.x)

      ctx.beginPath()
      const firstPoint = sortedFront[0]
      ctx.moveTo(scaleX(firstPoint.x), scaleY(firstPoint.y))

      for (let i = 1; i < sortedFront.length; i++) {
        const point = sortedFront[i]
        ctx.lineTo(scaleX(point.x), scaleY(point.y))
      }

      ctx.strokeStyle = "#10b981" // Green
      ctx.lineWidth = 2
      ctx.stroke()
    }

    // Draw Pareto front points
    adjustedParetoFront.forEach((solution) => {
      const x = scaleX(solution.x)
      const y = scaleY(solution.y)

      ctx.beginPath()
      ctx.arc(x, y, 6, 0, 2 * Math.PI)
      ctx.fillStyle = solution.id === selectedSolution ? "#f59e0b" : "#10b981" // Amber if selected, green otherwise
      ctx.fill()
      ctx.strokeStyle = "#047857"
      ctx.lineWidth = 1.5
      ctx.stroke()

      // Draw solution ID
      ctx.fillStyle = "#000"
      ctx.font = "10px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText(solution.id.toString(), x, y)

      // Draw uncertainty if available
      if (solution.uncertainty) {
        const radius = solution.uncertainty * 50 // Scale uncertainty for visualization
        ctx.beginPath()
        ctx.arc(x, y, radius, 0, 2 * Math.PI)
        ctx.strokeStyle = "rgba(16, 185, 129, 0.3)"
        ctx.lineWidth = 1
        ctx.stroke()
      }
    })

    // Add click handler for solution selection
    if (onSelectSolution) {
      canvas.onclick = (event) => {
        const rect = canvas.getBoundingClientRect()
        const x = event.clientX - rect.left
        const y = event.clientY - rect.top

        // Check if click is within any solution point
        const allSolutions = [...adjustedParetoFront, ...adjustedNonDominated]
        for (const solution of allSolutions) {
          const sx = scaleX(solution.x)
          const sy = scaleY(solution.y)
          const distance = Math.sqrt(Math.pow(x - sx, 2) + Math.pow(y - sy, 2))
          if (distance <= 8) {
            // If click is within 8px of a solution center
            onSelectSolution(solution.id)
            break
          }
        }
      }
    }

    // Draw ideal point and nadir point
    const idealX = objective1.type === "maximize" ? maxX : minX
    const idealY = objective2.type === "maximize" ? maxY : minY
    const nadirX = objective1.type === "maximize" ? minX : maxX
    const nadirY = objective2.type === "maximize" ? minY : maxY

    // Draw ideal point
    ctx.beginPath()
    ctx.arc(scaleX(idealX), scaleY(idealY), 5, 0, 2 * Math.PI)
    ctx.fillStyle = "#f59e0b" // Amber
    ctx.fill()
    ctx.strokeStyle = "#d97706"
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw "Ideal" label
    ctx.fillStyle = "#000"
    ctx.font = "10px sans-serif"
    ctx.textAlign = "center"
    ctx.textBaseline = "bottom"
    ctx.fillText("Ideal", scaleX(idealX), scaleY(idealY) - 8)

    // Draw nadir point
    ctx.beginPath()
    ctx.arc(scaleX(nadirX), scaleY(nadirY), 5, 0, 2 * Math.PI)
    ctx.fillStyle = "#ef4444" // Red
    ctx.fill()
    ctx.strokeStyle = "#b91c1c"
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw "Nadir" label
    ctx.fillStyle = "#000"
    ctx.font = "10px sans-serif"
    ctx.textAlign = "center"
    ctx.textBaseline = "bottom"
    ctx.fillText("Nadir", scaleX(nadirX), scaleY(nadirY) - 8)

    // Draw legend
    const legendX = width - padding.right + 10
    const legendY = padding.top
    const legendWidth = 120
    const legendHeight = 130
    const legendPadding = 10
    const legendItemHeight = 20

    ctx.fillStyle = "rgba(255, 255, 255, 0.9)"
    ctx.fillRect(legendX, legendY, legendWidth, legendHeight)
    ctx.strokeStyle = "#ddd"
    ctx.lineWidth = 1
    ctx.strokeRect(legendX, legendY, legendWidth, legendHeight)

    ctx.fillStyle = "#000"
    ctx.font = "bold 10px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "top"
    ctx.fillText("Legend", legendX + legendPadding, legendY + legendPadding)

    // Pareto front
    ctx.beginPath()
    ctx.arc(legendX + legendPadding + 5, legendY + legendPadding + legendItemHeight, 5, 0, 2 * Math.PI)
    ctx.fillStyle = "#10b981"
    ctx.fill()
    ctx.strokeStyle = "#047857"
    ctx.lineWidth = 1
    ctx.stroke()

    ctx.fillStyle = "#000"
    ctx.font = "10px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"
    ctx.fillText("Pareto Front", legendX + legendPadding + 15, legendY + legendPadding + legendItemHeight)

    // Non-dominated
    ctx.beginPath()
    ctx.arc(legendX + legendPadding + 5, legendY + legendPadding + legendItemHeight * 2, 5, 0, 2 * Math.PI)
    ctx.fillStyle = "rgba(59, 130, 246, 0.5)"
    ctx.fill()
    ctx.strokeStyle = "#3b82f6"
    ctx.lineWidth = 1
    ctx.stroke()

    ctx.fillStyle = "#000"
    ctx.font = "10px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"
    ctx.fillText("Non-dominated", legendX + legendPadding + 15, legendY + legendPadding + legendItemHeight * 2)

    // Dominated
    ctx.beginPath()
    ctx.arc(legendX + legendPadding + 5, legendY + legendPadding + legendItemHeight * 3, 5, 0, 2 * Math.PI)
    ctx.fillStyle = "rgba(156, 163, 175, 0.5)"
    ctx.fill()
    ctx.strokeStyle = "#9ca3af"
    ctx.lineWidth = 1
    ctx.stroke()

    ctx.fillStyle = "#000"
    ctx.font = "10px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"
    ctx.fillText("Dominated", legendX + legendPadding + 15, legendY + legendPadding + legendItemHeight * 3)

    // Ideal point
    ctx.beginPath()
    ctx.arc(legendX + legendPadding + 5, legendY + legendPadding + legendItemHeight * 4, 5, 0, 2 * Math.PI)
    ctx.fillStyle = "#f59e0b"
    ctx.fill()
    ctx.strokeStyle = "#d97706"
    ctx.lineWidth = 1
    ctx.stroke()

    ctx.fillStyle = "#000"
    ctx.font = "10px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"
    ctx.fillText("Ideal Point", legendX + legendPadding + 15, legendY + legendPadding + legendItemHeight * 4)

    // Nadir point
    ctx.beginPath()
    ctx.arc(legendX + legendPadding + 5, legendY + legendPadding + legendItemHeight * 5, 5, 0, 2 * Math.PI)
    ctx.fillStyle = "#ef4444"
    ctx.fill()
    ctx.strokeStyle = "#b91c1c"
    ctx.lineWidth = 1
    ctx.stroke()

    ctx.fillStyle = "#000"
    ctx.font = "10px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"
    ctx.fillText("Nadir Point", legendX + legendPadding + 15, legendY + legendPadding + legendItemHeight * 5)
  }, [adjustedParetoFront, adjustedNonDominated, adjustedDominated, objective1, objective2, selectedSolution])

  return (
    <div className="w-full h-full">
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="bg-green-100 text-green-800">
            Pareto Front
          </Badge>
          <Badge variant="outline" className="bg-blue-100 text-blue-800">
            {adjustedNonDominated.length} Non-dominated
          </Badge>
          <Badge variant="outline" className="bg-gray-100 text-gray-800">
            {adjustedDominated.length} Dominated
          </Badge>
        </div>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center text-sm text-muted-foreground cursor-help">
                <Info className="h-4 w-4 mr-1" />
                About Pareto Front
              </div>
            </TooltipTrigger>
            <TooltipContent side="left" className="max-w-sm">
              <p>
                The Pareto front represents optimal trade-offs between competing objectives. Points on this front cannot
                be improved in one objective without sacrificing performance in another objective.
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      <canvas ref={canvasRef} width={800} height={500} className="w-full h-full" />
    </div>
  )
}
