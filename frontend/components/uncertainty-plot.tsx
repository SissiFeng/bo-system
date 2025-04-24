"use client"
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ErrorBar,
  Legend,
  ReferenceLine,
} from "recharts"

interface Recommendation {
  id: number
  [key: string]: number | string
}

interface UncertaintyPlotProps {
  recommendations: Recommendation[]
}

export function UncertaintyPlot({ recommendations }: UncertaintyPlotProps) {
  // Transform data for the chart
  const chartData = recommendations.map((rec) => {
    return {
      name: `Rec ${rec.id}`,
      y1: rec.y1,
      y1Error: [rec.uncertainty, rec.uncertainty],
      y2: rec.y2,
      y2Error: [rec.uncertainty, rec.uncertainty],
    }
  })

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="y1" fill="#8884d8" name="Objective 1">
          <ErrorBar dataKey="y1Error" width={4} strokeWidth={2} stroke="#8884d8" />
        </Bar>
        <Bar dataKey="y2" fill="#82ca9d" name="Objective 2">
          <ErrorBar dataKey="y2Error" width={4} strokeWidth={2} stroke="#82ca9d" />
        </Bar>
        <ReferenceLine y={0} stroke="#000" />
      </BarChart>
    </ResponsiveContainer>
  )
}
