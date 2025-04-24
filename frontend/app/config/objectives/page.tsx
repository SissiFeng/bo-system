"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { ObjectiveForm } from "@/components/objective-form"
import { ObjectiveList } from "@/components/objective-list"
import { type Objective, OptimizationType } from "@/lib/types"
import Link from "next/link"
import { ArrowLeft, ArrowRight, Plus } from "lucide-react"
// Import the visualization component
import { MultiObjectiveVisualization } from "@/components/multi-objective-visualization"

export default function ObjectivesPage() {
  const [objectives, setObjectives] = useState<Objective[]>([])
  const [editingObjective, setEditingObjective] = useState<Objective | null>(null)

  const handleAddObjective = (objective: Objective) => {
    if (editingObjective) {
      setObjectives(objectives.map((o) => (o.id === objective.id ? objective : o)))
      setEditingObjective(null)
    } else {
      setObjectives([...objectives, { ...objective, id: Date.now().toString() }])
    }
  }

  const handleEditObjective = (objective: Objective) => {
    setEditingObjective(objective)
  }

  const handleDeleteObjective = (id: string) => {
    setObjectives(objectives.filter((o) => o.id !== id))
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Objective Functions</h1>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/config/parameters">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Link>
          </Button>
          <Button asChild>
            <Link href="/config/constraints">
              Next
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Objectives</CardTitle>
              <CardDescription>Define the KPIs and objectives for your optimization problem</CardDescription>
            </CardHeader>
            <CardContent>
              {objectives.length === 0 ? (
                <div className="flex flex-col items-center justify-center p-8 text-center border border-dashed rounded-lg">
                  <p className="mb-4 text-muted-foreground">No objectives defined yet</p>
                  <Button
                    variant="outline"
                    onClick={() => setEditingObjective({ id: "", name: "", type: OptimizationType.MAXIMIZE })}
                  >
                    <Plus className="mr-2 h-4 w-4" />
                    Add Objective
                  </Button>
                </div>
              ) : (
                <ObjectiveList objectives={objectives} onEdit={handleEditObjective} onDelete={handleDeleteObjective} />
              )}
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button
                variant="outline"
                onClick={() => setEditingObjective({ id: "", name: "", type: OptimizationType.MAXIMIZE })}
              >
                <Plus className="mr-2 h-4 w-4" />
                Add Objective
              </Button>
              <Button asChild>
                <Link href="/config/constraints">
                  Next: Define Constraints
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>
        </div>

        <div>
          <Card>
            <CardHeader>
              <CardTitle>{editingObjective ? "Edit Objective" : "Add Objective"}</CardTitle>
              <CardDescription>
                {editingObjective ? "Modify objective properties" : "Define a new KPI for optimization"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ObjectiveForm
                objective={editingObjective || { id: "", name: "", type: OptimizationType.MAXIMIZE }}
                onSubmit={handleAddObjective}
                onCancel={() => setEditingObjective(null)}
              />
            </CardContent>
          </Card>

          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Objectives Preview</CardTitle>
              <CardDescription>JSON representation of your objectives</CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-muted p-4 rounded-md overflow-auto text-xs">{JSON.stringify(objectives, null, 2)}</pre>
            </CardContent>
          </Card>

          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Multi-Objective Visualization</CardTitle>
              <CardDescription>Trade-offs between objectives and Pareto front</CardDescription>
            </CardHeader>
            <CardContent>
              <MultiObjectiveVisualization objectives={objectives} />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
