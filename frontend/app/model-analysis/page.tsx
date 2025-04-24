"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { PredictionRealityVisualization } from "@/components/prediction-reality-visualization"
import { MultiObjectiveOptimization } from "@/components/multi-objective-optimization"

export default function ModelAnalysisPage() {
  const [activeTab, setActiveTab] = useState("prediction-reality")
  const [mooConfig, setMooConfig] = useState({})

  const handleMooConfigChange = (config: any) => {
    setMooConfig(config)
    console.log("MOO Config updated:", config)
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Model Analysis & Multi-Objective Optimization</h1>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid grid-cols-2 w-full">
          <TabsTrigger value="prediction-reality">Prediction vs. Reality</TabsTrigger>
          <TabsTrigger value="multi-objective">Multi-Objective Optimization</TabsTrigger>
        </TabsList>

        <TabsContent value="prediction-reality" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Model Prediction vs. Experimental Reality</CardTitle>
              <CardDescription>
                Visualize the relationship between model predictions and actual experimental measurements
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <PredictionRealityVisualization
                  targetValue={400}
                  predictedValue={410}
                  uncertainty={5}
                  historicalRange={[300, 700]}
                  experimentalValues={[350, 420, 480, 650]}
                />

                <Card>
                  <CardHeader>
                    <CardTitle>Understanding Prediction Uncertainty</CardTitle>
                    <CardDescription>How to interpret the prediction vs. reality visualization</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <h3 className="font-medium">Model Prediction</h3>
                      <p className="text-sm text-muted-foreground">
                        The bell curve shows the model's prediction (410) and its uncertainty (±5). This represents the
                        model's confidence in its prediction.
                      </p>
                    </div>

                    <div>
                      <h3 className="font-medium">Historical Range</h3>
                      <p className="text-sm text-muted-foreground">
                        The orange bar shows the range of actual experimental measurements (300-700) that have been
                        observed historically for similar designs. This represents the experimental variability.
                      </p>
                    </div>

                    <div>
                      <h3 className="font-medium">Confidence Assessment</h3>
                      <p className="text-sm text-muted-foreground">
                        When the historical range is much wider than the prediction uncertainty, it indicates that the
                        model may be overconfident or that there is high experimental variability that the model doesn't
                        account for.
                      </p>
                    </div>

                    <div>
                      <h3 className="font-medium">Target Value</h3>
                      <p className="text-sm text-muted-foreground">
                        The green line shows the target value (400) that you're aiming for. The model predicts a value
                        close to this target, but actual results may vary significantly.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <div className="mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Example Recommendations</CardTitle>
                    <CardDescription>
                      Predicted values and their uncertainty compared to experimental reality
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      <PredictionRealityVisualization
                        targetValue={400}
                        predictedValue={405}
                        uncertainty={3}
                        historicalRange={[390, 420]}
                      />
                      <PredictionRealityVisualization
                        targetValue={400}
                        predictedValue={410}
                        uncertainty={5}
                        historicalRange={[300, 700]}
                      />
                      <PredictionRealityVisualization
                        targetValue={400}
                        predictedValue={380}
                        uncertainty={10}
                        historicalRange={[350, 450]}
                      />
                    </div>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="multi-objective" className="space-y-4">
          <MultiObjectiveOptimization onConfigChange={handleMooConfigChange} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
