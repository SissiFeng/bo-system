"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Plus, Send } from "lucide-react"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Badge } from "@/components/ui/badge"

interface Recommendation {
  id: number
  parameters: Record<string, number | string>
  objectives: Record<string, number>
  uncertainty: number
  reason: string
}

interface HumanFeedbackPanelProps {
  recommendations: Recommendation[]
  onSubmitFeedback: (selectedRecommendations: Recommendation[], customPoints: any[], notes: string) => void
}

export function HumanFeedbackPanel({ recommendations, onSubmitFeedback }: HumanFeedbackPanelProps) {
  const [selectedRecommendations, setSelectedRecommendations] = useState<number[]>(recommendations.map((rec) => rec.id))
  const [customPoints, setCustomPoints] = useState<any[]>([])
  const [notes, setNotes] = useState("")
  const [newPoint, setNewPoint] = useState<Record<string, string>>({})
  const [showNewPointForm, setShowNewPointForm] = useState(false)

  const handleToggleRecommendation = (id: number) => {
    setSelectedRecommendations((prev) => (prev.includes(id) ? prev.filter((recId) => recId !== id) : [...prev, id]))
  }

  const handleAddCustomPoint = () => {
    // Convert string values to numbers where appropriate
    const processedPoint = Object.entries(newPoint).reduce(
      (acc, [key, value]) => {
        acc[key] = isNaN(Number(value)) ? value : Number(value)
        return acc
      },
      {} as Record<string, number | string>,
    )

    setCustomPoints([...customPoints, { id: Date.now(), parameters: processedPoint }])
    setNewPoint({})
    setShowNewPointForm(false)
  }

  const handleRemoveCustomPoint = (id: number) => {
    setCustomPoints(customPoints.filter((point) => point.id !== id))
  }

  const handleSubmit = () => {
    const filteredRecommendations = recommendations.filter((rec) => selectedRecommendations.includes(rec.id))
    onSubmitFeedback(filteredRecommendations, customPoints, notes)
  }

  // Extract parameter names from the first recommendation
  const parameterNames = recommendations.length > 0 ? Object.keys(recommendations[0].parameters) : []

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Human-in-the-Loop Feedback</CardTitle>
        <CardDescription>Review and select recommendations for the next optimization round</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <TooltipProvider>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-12">Select</TableHead>
                <TableHead>Design</TableHead>
                {parameterNames.map((param) => (
                  <TableHead key={param}>{param}</TableHead>
                ))}
                <TableHead>Predicted Performance</TableHead>
                <TableHead>Uncertainty</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {recommendations.map((rec) => (
                <TableRow key={rec.id}>
                  <TableCell>
                    <Checkbox
                      checked={selectedRecommendations.includes(rec.id)}
                      onCheckedChange={() => handleToggleRecommendation(rec.id)}
                    />
                  </TableCell>
                  <TableCell>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="font-medium cursor-help">Design #{rec.id}</span>
                      </TooltipTrigger>
                      <TooltipContent side="right" className="max-w-sm">
                        <p className="font-medium">Why this design?</p>
                        <p className="text-sm">{rec.reason}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TableCell>
                  {parameterNames.map((param) => (
                    <TableCell key={param}>{rec.parameters[param]}</TableCell>
                  ))}
                  <TableCell>
                    {Object.entries(rec.objectives).map(([key, value]) => (
                      <div key={key}>
                        {key}: {value.toFixed(2)}
                      </div>
                    ))}
                  </TableCell>
                  <TableCell>
                    <Badge variant={rec.uncertainty > 0.2 ? "secondary" : "outline"}>
                      σ = {rec.uncertainty.toFixed(2)}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}

              {customPoints.map((point) => (
                <TableRow key={point.id}>
                  <TableCell>
                    <Checkbox checked={true} disabled />
                  </TableCell>
                  <TableCell>
                    <span className="font-medium">Custom #{point.id}</span>
                  </TableCell>
                  {parameterNames.map((param) => (
                    <TableCell key={param}>{point.parameters[param] || "—"}</TableCell>
                  ))}
                  <TableCell colSpan={2}>
                    <Button variant="ghost" size="sm" onClick={() => handleRemoveCustomPoint(point.id)}>
                      Remove
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TooltipProvider>

        {showNewPointForm ? (
          <Card className="border border-dashed">
            <CardHeader className="py-3">
              <CardTitle className="text-sm">Add Custom Point</CardTitle>
            </CardHeader>
            <CardContent className="py-2">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {parameterNames.map((param) => (
                  <div key={param} className="space-y-1">
                    <label className="text-sm font-medium">{param}</label>
                    <Input
                      value={newPoint[param] || ""}
                      onChange={(e) => setNewPoint({ ...newPoint, [param]: e.target.value })}
                      placeholder="Value"
                    />
                  </div>
                ))}
              </div>
            </CardContent>
            <CardFooter className="flex justify-between py-3">
              <Button variant="ghost" onClick={() => setShowNewPointForm(false)}>
                Cancel
              </Button>
              <Button onClick={handleAddCustomPoint}>Add Point</Button>
            </CardFooter>
          </Card>
        ) : (
          <Button variant="outline" className="w-full" onClick={() => setShowNewPointForm(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Add Custom Point
          </Button>
        )}

        <div className="space-y-2">
          <label className="text-sm font-medium">Notes for Next Round</label>
          <Textarea
            placeholder="Add any notes or observations about these designs..."
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
        </div>
      </CardContent>
      <CardFooter className="flex justify-between">
        <div className="text-sm text-muted-foreground">{selectedRecommendations.length} designs selected</div>
        <Button onClick={handleSubmit}>
          <Send className="mr-2 h-4 w-4" />
          Submit for Next Round
        </Button>
      </CardFooter>
    </Card>
  )
}
