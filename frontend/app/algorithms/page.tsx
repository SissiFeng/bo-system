"use client"

import { useState } from "react"
import { AlgorithmSelector } from "@/components/algorithm-selector"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"

export default function AlgorithmsPage() {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<any>(null)

  const handleAlgorithmSelect = (algorithm: any) => {
    setSelectedAlgorithm(algorithm)
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
          <h1 className="text-3xl font-bold">Algorithm Selector</h1>
        </div>
        <Button asChild>
          <Link href="/canvas">Continue to Canvas</Link>
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <AlgorithmSelector onSelect={handleAlgorithmSelect} />
        </div>

        <div>
          <Card>
            <CardHeader>
              <CardTitle>Selected Algorithm</CardTitle>
            </CardHeader>
            <CardContent>
              {selectedAlgorithm ? (
                <pre className="bg-muted p-4 rounded-md overflow-auto text-xs h-[400px]">
                  {JSON.stringify(selectedAlgorithm, null, 2)}
                </pre>
              ) : (
                <div className="h-[400px] flex items-center justify-center text-muted-foreground">
                  <p>Select an algorithm to see its configuration</p>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="mt-6 flex justify-end">
            <Button asChild>
              <Link href="/canvas">Continue to Canvas</Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
