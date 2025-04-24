"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { ParameterForm } from "@/components/parameter-form"
import { ParameterList } from "@/components/parameter-list"
import { ParameterType, type Parameter } from "@/lib/types"
import Link from "next/link"
import { ArrowLeft, ArrowRight, Plus } from "lucide-react"
import { ModeToggleSwitch } from "@/components/mode-toggle-switch"
import { ParameterSpaceVisualization } from "@/components/parameter-space-visualization"

export default function ParametersPage() {
  const [parameters, setParameters] = useState<Parameter[]>([])
  const [editingParameter, setEditingParameter] = useState<Parameter | null>(null)

  const handleAddParameter = (parameter: Parameter) => {
    if (editingParameter && editingParameter.id) {
      setParameters(parameters.map((p) => (p.id === parameter.id ? parameter : p)))
      setEditingParameter(null)
    } else {
      setParameters([...parameters, { ...parameter, id: Date.now().toString() }])
    }
  }

  const handleEditParameter = (parameter: Parameter) => {
    setEditingParameter(parameter)
  }

  const handleDeleteParameter = (id: string) => {
    setParameters(parameters.filter((p) => p.id !== id))
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Parameter Space Configuration</h1>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Link>
          </Button>
          <Button asChild>
            <Link href="/config/objectives">
              Next
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>

      <div className="mb-6">
        <ModeToggleSwitch />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Parameters</CardTitle>
              <CardDescription>Define the parameters for your optimization problem</CardDescription>
            </CardHeader>
            <CardContent>
              {parameters.length === 0 ? (
                <div className="flex flex-col items-center justify-center p-8 text-center border border-dashed rounded-lg">
                  <p className="mb-4 text-muted-foreground">No parameters defined yet</p>
                  <Button
                    variant="outline"
                    onClick={() =>
                      setEditingParameter({ id: "", name: "", type: ParameterType.CONTINUOUS, min: 0, max: 1 })
                    }
                  >
                    <Plus className="mr-2 h-4 w-4" />
                    Add Parameter
                  </Button>
                </div>
              ) : (
                <ParameterList parameters={parameters} onEdit={handleEditParameter} onDelete={handleDeleteParameter} />
              )}
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button
                variant="outline"
                onClick={() =>
                  setEditingParameter({ id: "", name: "", type: ParameterType.CONTINUOUS, min: 0, max: 1 })
                }
              >
                <Plus className="mr-2 h-4 w-4" />
                Add Parameter
              </Button>
              <Button asChild>
                <Link href="/config/objectives">
                  Next: Define Objectives
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>
        </div>

        <div>
          <Card>
            <CardHeader>
              <CardTitle>{editingParameter ? "Edit Parameter" : "Add Parameter"}</CardTitle>
              <CardDescription>
                {editingParameter ? "Modify parameter properties" : "Define a new parameter for optimization"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ParameterForm
                parameter={
                  editingParameter || {
                    id: "",
                    name: "",
                    type: ParameterType.CONTINUOUS,
                    min: 0,
                    max: 1,
                  }
                }
                onSubmit={handleAddParameter}
                onCancel={() => setEditingParameter(null)}
                existingParameters={parameters}
              />
            </CardContent>
          </Card>

          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Parameter Space Preview</CardTitle>
              <CardDescription>JSON representation of your parameter space</CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-muted p-4 rounded-md overflow-auto text-xs">{JSON.stringify(parameters, null, 2)}</pre>
            </CardContent>
          </Card>

          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Parameter Space Visualization</CardTitle>
              <CardDescription>Parameter relationships and feasible region</CardDescription>
            </CardHeader>
            <CardContent>
              <ParameterSpaceVisualization parameters={parameters} />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
