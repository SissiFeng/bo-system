"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts"

// Mock data for feature importance
const featureImportanceData = [
  { name: "Parameter 1", importance: 0.45 },
  { name: "Parameter 2", importance: 0.35 },
  { name: "Parameter 3", importance: 0.2 },
]

export function ModelPerformance() {
  // Mock R² value
  const r2Value = 0.87

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Model Accuracy (R²)</CardTitle>
            <CardDescription>How well the model fits the observed data</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">R² Value</span>
                <span className="text-sm font-medium">{r2Value.toFixed(2)}</span>
              </div>
              <Progress value={r2Value * 100} className="h-2" />
              <p className="text-sm text-muted-foreground">
                {r2Value > 0.8
                  ? "Good model fit. Predictions should be reliable."
                  : r2Value > 0.5
                    ? "Moderate model fit. Consider collecting more data."
                    : "Poor model fit. Predictions may be unreliable."}
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Cross-Validation</CardTitle>
            <CardDescription>Model performance across different data splits</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">CV Score</span>
                <span className="text-sm font-medium">0.82 ± 0.05</span>
              </div>
              <Progress value={82} className="h-2" />
              <p className="text-sm text-muted-foreground">
                Cross-validation shows consistent performance across different data splits.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Feature Importance</CardTitle>
          <CardDescription>Relative importance of each parameter in the model</CardDescription>
        </CardHeader>
        <CardContent className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={featureImportanceData}
              layout="vertical"
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 1]} />
              <YAxis type="category" dataKey="name" />
              <Tooltip formatter={(value) => [`${(Number(value) * 100).toFixed(1)}%`, "Importance"]} />
              <Legend />
              <Bar dataKey="importance" fill="#8884d8" name="Importance" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}
