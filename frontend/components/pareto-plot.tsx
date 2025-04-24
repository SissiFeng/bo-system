"use client"

import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ZAxis,
} from "recharts"

interface Recommendation {
  id: number
  [key: string]: number | string
}

interface ParetoPlotProps {
  recommendations: Recommendation[]
}

export function ParetoPlot({ recommendations }: ParetoPlotProps) {
  // Transform data for the chart
  const chartData = recommendations.map((rec) => {
    return {
      name: `Rec ${rec.id}`,
      x: rec.y1,
      y: rec.y2,
      z: rec.uncertainty * 100, // Scale for visibility
    }
  })

  // Add some Pareto front points
  const paretoData = [
    { x: 0.85, y: 0.1, z: 5 },
    { x: 0.89, y: 0.12, z: 5 },
    { x: 0.92, y: 0.15, z: 5 },
    { x: 0.95, y: 0.18, z: 5 },
  ]

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
        <CartesianGrid />
        <XAxis
          type="number"
          dataKey="x"
          name="Objective 1"
          label={{ value: "Objective 1", position: "insideBottomRight", offset: -5 }}
        />
        <YAxis
          type="number"
          dataKey="y"
          name="Objective 2"
          label={{ value: "Objective 2", angle: -90, position: "insideLeft" }}
        />
        <ZAxis type="number" dataKey="z" range={[60, 200]} name="Uncertainty" />
        <Tooltip
          cursor={{ strokeDasharray: "3 3" }}
          formatter={(value, name) => [value, name === "z" ? "Uncertainty" : name]}
        />
        <Legend />
        <Scatter name="Recommendations" data={chartData} fill="#8884d8" shape="circle" />
        <Scatter name="Pareto Front" data={paretoData} fill="#ff7300" line shape="cross" lineType="fitting" />
      </ScatterChart>
    </ResponsiveContainer>
  )
}
