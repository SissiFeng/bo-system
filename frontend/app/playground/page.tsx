"use client"

import { useState } from "react"
import { ParameterPlayground } from "@/components/parameter-playground"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { ArrowLeft, ArrowRight } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function PlaygroundPage() {
  const [parameters, setParameters] = useState([])
  const [constraints, setConstraints] = useState([])

  const handleUpdate = (updatedParameters: any[], updatedConstraints: any[]) => {
    setParameters(updatedParameters)
    setConstraints(updatedConstraints)
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <Button variant="outline" asChild className="mr-4">
            <Link href="/">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Home
            </Link>
          </Button>
          <h1 className="text-3xl font-bold">Parameter Space Playground</h1>
        </div>
        <Button asChild>
          <Link href="/canvas">
            Continue to Canvas
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ParameterPlayground onUpdate={handleUpdate} />
        </div>

        <div>
          <Card>
            <CardHeader>
              <CardTitle>Parameter Space Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="bg-muted p-4 rounded-md overflow-auto text-xs h-[400px]">
                {JSON.stringify({ parameters, constraints }, null, 2)}
              </pre>
            </CardContent>
          </Card>

          <div className="mt-6 flex justify-end">
            <Button asChild>
              <Link href="/canvas">
                Continue to Canvas
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
