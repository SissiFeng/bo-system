"use client"

import { useState } from "react"
import { OptimizationWizard } from "@/components/optimization-wizard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"

export default function WizardPage() {
  const [wizardComplete, setWizardComplete] = useState(false)
  const [generatedConfig, setGeneratedConfig] = useState<any>(null)

  const handleWizardComplete = (config: any) => {
    setGeneratedConfig(config)
    setWizardComplete(true)
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center mb-6">
        <Button variant="outline" asChild className="mr-4">
          <Link href="/">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Link>
        </Button>
        <h1 className="text-3xl font-bold">Optimization Wizard</h1>
      </div>

      {!wizardComplete ? (
        <OptimizationWizard onComplete={handleWizardComplete} />
      ) : (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Your Optimization Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="bg-muted p-4 rounded-md overflow-auto text-xs h-[400px]">
                {JSON.stringify(generatedConfig, null, 2)}
              </pre>
            </CardContent>
          </Card>

          <div className="flex justify-between">
            <Button variant="outline" onClick={() => setWizardComplete(false)}>
              Modify Configuration
            </Button>
            <Button asChild>
              <Link href="/canvas">Continue to Canvas Editor</Link>
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
