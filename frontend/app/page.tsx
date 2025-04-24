import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import Link from "next/link"
import { ArrowRight, Beaker, ChevronRight, LineChart, Pencil, Settings } from "lucide-react"

export default function Home() {
  return (
    <main className="container mx-auto px-4 py-8">
      <div className="flex flex-col items-center justify-center text-center mb-12">
        <h1 className="text-4xl font-bold tracking-tight mb-4">Bayesian Optimization Platform</h1>
        <p className="text-lg text-muted-foreground max-w-2xl">
          A no-code solution for scientists to optimize experimental parameters using Bayesian methods
        </p>
        <div className="flex gap-4 mt-8">
          <Button asChild size="lg">
            <Link href="/config/parameters">
              New Optimization
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
          <Button variant="outline" size="lg">
            Load Configuration
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Settings className="mr-2 h-5 w-5 text-primary" />
              Parameter Space
            </CardTitle>
            <CardDescription>Define your experimental variables</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="mb-4">Configure continuous, discrete, and categorical parameters with constraints.</p>
            <Button variant="secondary" asChild className="w-full">
              <Link href="/config/parameters">
                Configure Parameters
                <ChevronRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Pencil className="mr-2 h-5 w-5 text-primary" />
              Canvas Workflow
            </CardTitle>
            <CardDescription>Design your optimization visually</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="mb-4">
              Create optimization workflows by connecting parameter, objective, and strategy nodes.
            </p>
            <Button variant="secondary" asChild className="w-full">
              <Link href="/canvas">
                Open Canvas
                <ChevronRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <LineChart className="mr-2 h-5 w-5 text-primary" />
              Visualization
            </CardTitle>
            <CardDescription>Analyze optimization results</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="mb-4">Visualize Pareto fronts, uncertainty, and model performance metrics.</p>
            <Button variant="secondary" asChild className="w-full">
              <Link href="/results">
                View Results
                <ChevronRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardContent>
        </Card>
      </div>

      <div className="bg-muted rounded-lg p-6">
        <h2 className="text-2xl font-semibold mb-4">How It Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="flex flex-col items-center text-center">
            <div className="bg-primary/10 rounded-full p-3 mb-4">
              <Settings className="h-8 w-8 text-primary" />
            </div>
            <h3 className="text-lg font-medium mb-2">1. Define Parameters</h3>
            <p className="text-muted-foreground">Set up parameter space with ranges and constraints</p>
          </div>
          <div className="flex flex-col items-center text-center">
            <div className="bg-primary/10 rounded-full p-3 mb-4">
              <Beaker className="h-8 w-8 text-primary" />
            </div>
            <h3 className="text-lg font-medium mb-2">2. Set Objectives</h3>
            <p className="text-muted-foreground">Define KPIs and optimization goals</p>
          </div>
          <div className="flex flex-col items-center text-center">
            <div className="bg-primary/10 rounded-full p-3 mb-4">
              <Pencil className="h-8 w-8 text-primary" />
            </div>
            <h3 className="text-lg font-medium mb-2">3. Design Workflow</h3>
            <p className="text-muted-foreground">Connect nodes on canvas to create optimization flow</p>
          </div>
          <div className="flex flex-col items-center text-center">
            <div className="bg-primary/10 rounded-full p-3 mb-4">
              <LineChart className="h-8 w-8 text-primary" />
            </div>
            <h3 className="text-lg font-medium mb-2">4. Analyze Results</h3>
            <p className="text-muted-foreground">Visualize recommendations and model performance</p>
          </div>
        </div>
      </div>
    </main>
  )
}
