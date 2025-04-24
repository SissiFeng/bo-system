"use client"

import { OptimizationAnimation } from "@/components/optimization-animation"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"

export default function AnimationPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex items-center mb-6">
        <Button variant="outline" asChild className="mr-4">
          <Link href="/">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Link>
        </Button>
        <h1 className="text-3xl font-bold">Optimization Process Animation</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <OptimizationAnimation width={600} height={400} />
        </div>

        <div className="space-y-6">
          <div className="bg-muted rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">How to Read This Animation</h2>
            <ul className="space-y-2 text-sm">
              <li>
                <span className="font-medium">Blue background:</span> Represents the objective function landscape
                (darker = higher value)
              </li>
              <li>
                <span className="font-medium">Orange overlay:</span> Shows the acquisition function (where the algorithm
                wants to sample next)
              </li>
              <li>
                <span className="font-medium">Blue dots:</span> Previously sampled points
              </li>
              <li>
                <span className="font-medium">Red dot:</span> Most recently sampled point
              </li>
              <li>
                <span className="font-medium">Green circle:</span> Current best point found
              </li>
              <li>
                <span className="font-medium">Gray circles:</span> Uncertainty around each point (larger = more
                uncertain)
              </li>
            </ul>
          </div>

          <div className="bg-muted rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Optimization Phases</h2>
            <div className="space-y-4">
              <div>
                <h3 className="font-medium">Phase 1: Exploration (Iterations 0-10)</h3>
                <p className="text-sm text-muted-foreground">
                  Initial points are sampled using Latin Hypercube Sampling to explore the parameter space efficiently.
                </p>
              </div>
              <div>
                <h3 className="font-medium">Phase 2: Exploitation (Iterations 11+)</h3>
                <p className="text-sm text-muted-foreground">
                  Bayesian Optimization uses the acquired data to build a surrogate model and focuses sampling on
                  promising regions.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
