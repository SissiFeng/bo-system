"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { RecommendationsTable } from "@/components/recommendations-table"
import { ModelPerformance } from "@/components/model-performance"
import { ParetoPlot } from "@/components/pareto-plot"
import { UncertaintyPlot } from "@/components/uncertainty-plot"
import { HumanFeedbackPanel } from "@/components/human-feedback-panel"
import { NotificationSettings } from "@/components/notification-settings"
import { StrategyConversation } from "@/components/strategy-conversation"
import { ArrowLeft, Download, Play, Send } from "lucide-react"
import Link from "next/link"
import { useToast } from "@/components/ui/use-toast"

// Mock data for demonstration
const mockRecommendations = [
  {
    id: 1,
    parameters: { x1: 0.45, x2: 0.35, x3: 0.2 },
    objectives: { y1: 0.89, y2: 0.12 },
    uncertainty: 0.05,
    reason: "Low uncertainty region with high predicted performance. This is an exploitation-focused recommendation.",
  },
  {
    id: 2,
    parameters: { x1: 0.5, x2: 0.3, x3: 0.2 },
    objectives: { y1: 0.92, y2: 0.15 },
    uncertainty: 0.03,
    reason: "Best predicted performance with very low uncertainty. Highly confident in this recommendation.",
  },
  {
    id: 3,
    parameters: { x1: 0.4, x2: 0.4, x3: 0.2 },
    objectives: { y1: 0.85, y2: 0.1 },
    uncertainty: 0.07,
    reason: "Balanced trade-off between objectives y1 and y2. Good compromise solution.",
  },
  {
    id: 4,
    parameters: { x1: 0.6, x2: 0.2, x3: 0.2 },
    objectives: { y1: 0.95, y2: 0.18 },
    uncertainty: 0.25,
    reason: "High uncertainty region with potentially high rewards. This is an exploration-focused recommendation.",
  },
  {
    id: 5,
    parameters: { x1: 0.55, x2: 0.25, x3: 0.2 },
    objectives: { y1: 0.93, y2: 0.16 },
    uncertainty: 0.02,
    reason: "Very low uncertainty with excellent predicted performance. Strong exploitation recommendation.",
  },
]

export default function ResultsPage() {
  const [activeTab, setActiveTab] = useState("recommendations")
  const [showStrategyDialog, setShowStrategyDialog] = useState(false)
  const { toast } = useToast()

  const handleFeedbackSubmit = (selectedRecommendations: any[], customPoints: any[], notes: string) => {
    toast({
      title: "Feedback submitted",
      description: `Selected ${selectedRecommendations.length} recommendations and ${customPoints.length} custom points for the next round.`,
    })
    console.log({ selectedRecommendations, customPoints, notes })
    // Here you would typically send this data to your backend
  }

  const handleNotificationSave = (settings: any) => {
    console.log("Notification settings saved:", settings)
    // Here you would typically save these settings to your backend
  }

  const handleStrategySelect = (strategy: any) => {
    toast({
      title: "Strategy applied",
      description: `Applied "${strategy.name}" with ${strategy.batchSize} designs per batch.`,
    })
    setShowStrategyDialog(false)
    console.log("Selected strategy:", strategy)
    // Here you would typically update your optimization settings
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Results & Visualization</h1>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/config/constraints">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Link>
          </Button>
          <Button onClick={() => setShowStrategyDialog(true)}>
            <Play className="mr-2 h-4 w-4" />
            Run Optimization
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid grid-cols-5 w-full">
          <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          <TabsTrigger value="uncertainty">Uncertainty</TabsTrigger>
          <TabsTrigger value="pareto">Pareto Front</TabsTrigger>
          <TabsTrigger value="model">Model Performance</TabsTrigger>
          <TabsTrigger value="feedback">Human Feedback</TabsTrigger>
        </TabsList>

        <TabsContent value="recommendations" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recommended Parameter Sets</CardTitle>
              <CardDescription>Top recommendations based on optimization objectives</CardDescription>
            </CardHeader>
            <CardContent>
              <RecommendationsTable recommendations={mockRecommendations} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="uncertainty" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Prediction Uncertainty</CardTitle>
              <CardDescription>Visualization of prediction uncertainty for each recommendation</CardDescription>
            </CardHeader>
            <CardContent className="h-[500px]">
              <UncertaintyPlot recommendations={mockRecommendations} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="pareto" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Pareto Front</CardTitle>
              <CardDescription>Multi-objective optimization trade-off visualization</CardDescription>
            </CardHeader>
            <CardContent className="h-[500px]">
              <ParetoPlot recommendations={mockRecommendations} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="model" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Model Performance</CardTitle>
              <CardDescription>Metrics and diagnostics for the surrogate model</CardDescription>
            </CardHeader>
            <CardContent>
              <ModelPerformance />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="feedback" className="space-y-4">
          <HumanFeedbackPanel recommendations={mockRecommendations} onSubmitFeedback={handleFeedbackSubmit} />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <NotificationSettings onSave={handleNotificationSave} />

            <Card>
              <CardHeader>
                <CardTitle>Export Options</CardTitle>
                <CardDescription>Download your results in various formats</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button variant="outline" className="w-full">
                  <Download className="mr-2 h-4 w-4" />
                  Export as CSV
                </Button>
                <Button variant="outline" className="w-full">
                  <Download className="mr-2 h-4 w-4" />
                  Export as JSON
                </Button>
                <Button variant="outline" className="w-full">
                  <Download className="mr-2 h-4 w-4" />
                  Export as Python Script
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {showStrategyDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-background rounded-lg shadow-lg w-full max-w-3xl max-h-[90vh] overflow-auto">
            <StrategyConversation onSelectStrategy={handleStrategySelect} />
            <div className="p-4 flex justify-end border-t">
              <Button variant="outline" onClick={() => setShowStrategyDialog(false)}>
                Cancel
              </Button>
            </div>
          </div>
        </div>
      )}

      <div className="mt-6 flex justify-end gap-4">
        <Button variant="outline">
          <Download className="mr-2 h-4 w-4" />
          Export Configuration
        </Button>
        <Button variant="outline">
          <Send className="mr-2 h-4 w-4" />
          Send Summary
        </Button>
      </div>
    </div>
  )
}
