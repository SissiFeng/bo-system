"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { ConstraintForm } from "@/components/constraint-form"
import { ConstraintList } from "@/components/constraint-list"
import { type Constraint, ConstraintType } from "@/lib/types"
import Link from "next/link"
import { ArrowLeft, ArrowRight, Plus } from "lucide-react"
import { ConstraintsVisualization } from "@/components/constraints-visualization"

export default function ConstraintsPage() {
  const [constraints, setConstraints] = useState<Constraint[]>([])
  const [editingConstraint, setEditingConstraint] = useState<Constraint | null>(null)

  const handleAddConstraint = (constraint: Constraint) => {
    if (editingConstraint) {
      setConstraints(constraints.map((c) => (c.id === constraint.id ? constraint : c)))
      setEditingConstraint(null)
    } else {
      setConstraints([...constraints, { ...constraint, id: Date.now().toString() }])
    }
  }

  const handleEditConstraint = (constraint: Constraint) => {
    setEditingConstraint(constraint)
  }

  const handleDeleteConstraint = (id: string) => {
    setConstraints(constraints.filter((c) => c.id !== id))
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Constraints</h1>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/config/objectives">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Link>
          </Button>
          <Button asChild>
            <Link href="/results">
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
              <CardTitle>Constraints</CardTitle>
              <CardDescription>Define constraints for your parameter space</CardDescription>
            </CardHeader>
            <CardContent>
              {constraints.length === 0 ? (
                <div className="flex flex-col items-center justify-center p-8 text-center border border-dashed rounded-lg">
                  <p className="mb-4 text-muted-foreground">No constraints defined yet</p>
                  <Button
                    variant="outline"
                    onClick={() =>
                      setEditingConstraint({ id: "", expression: "", type: ConstraintType.SUM_EQUALS, value: 1 })
                    }
                  >
                    <Plus className="mr-2 h-4 w-4" />
                    Add Constraint
                  </Button>
                </div>
              ) : (
                <ConstraintList
                  constraints={constraints}
                  onEdit={handleEditConstraint}
                  onDelete={handleDeleteConstraint}
                />
              )}
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button
                variant="outline"
                onClick={() =>
                  setEditingConstraint({ id: "", expression: "", type: ConstraintType.SUM_EQUALS, value: 1 })
                }
              >
                <Plus className="mr-2 h-4 w-4" />
                Add Constraint
              </Button>
              <Button asChild>
                <Link href="/results">
                  Next: View Results
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>
        </div>

        <div>
          <Card>
            <CardHeader>
              <CardTitle>{editingConstraint ? "Edit Constraint" : "Add Constraint"}</CardTitle>
              <CardDescription>
                {editingConstraint ? "Modify constraint properties" : "Define a new constraint for your parameters"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ConstraintForm
                constraint={editingConstraint || { id: "", expression: "", type: ConstraintType.SUM_EQUALS, value: 1 }}
                onSubmit={handleAddConstraint}
                onCancel={() => setEditingConstraint(null)}
              />
            </CardContent>
          </Card>

          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Constraints Preview</CardTitle>
              <CardDescription>JSON representation of your constraints</CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-muted p-4 rounded-md overflow-auto text-xs">
                {JSON.stringify(constraints, null, 2)}
              </pre>
            </CardContent>
          </Card>

          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Constraints Visualization</CardTitle>
              <CardDescription>Constraints and feasible region</CardDescription>
            </CardHeader>
            <CardContent>
              <ConstraintsVisualization constraints={constraints} />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
