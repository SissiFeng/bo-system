"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Textarea } from "@/components/ui/textarea"
import { ArrowRight, Check, HelpCircle, Lightbulb } from "lucide-react"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface OptimizationWizardProps {
  onComplete: (config: any) => void
}

export function OptimizationWizard({ onComplete }: OptimizationWizardProps) {
  const [step, setStep] = useState(1)
  const [config, setConfig] = useState({
    objective: {
      type: "maximize",
      name: "",
      description: "",
    },
    parameters: {
      type: "continuous",
      count: 2,
      priorKnowledge: "some",
    },
    budget: {
      experiments: 30,
      parallelRuns: 1,
    },
    strategy: {
      type: "bo",
      explorationWeight: 0.5,
    },
  })

  const updateConfig = (section: string, updates: any) => {
    setConfig((prev) => ({
      ...prev,
      [section]: {
        ...prev[section],
        ...updates,
      },
    }))
  }

  const nextStep = () => {
    setStep((prev) => prev + 1)
  }

  const prevStep = () => {
    setStep((prev) => Math.max(1, prev - 1))
  }

  const completeWizard = () => {
    // Generate recommended strategy based on inputs
    const recommendedConfig = generateRecommendedConfig(config)
    onComplete(recommendedConfig)
  }

  const generateRecommendedConfig = (userConfig: any) => {
    // Logic to generate a recommended optimization strategy based on user inputs
    const { objective, parameters, budget, strategy } = userConfig

    // Default strategy
    const recommendedStrategy = {
      stages: [
        {
          id: "stage-1",
          type: "lhs",
          name: "Initial Exploration",
          iterations: Math.max(5, Math.floor(budget.experiments * 0.3)),
        },
        {
          id: "stage-2",
          type: "bo",
          name: "Bayesian Optimization",
          iterations: Math.floor(budget.experiments * 0.7),
          explorationWeight: strategy.explorationWeight,
          acquisitionFunction: "ei",
          batchSize: budget.parallelRuns,
        },
      ],
    }

    // Adjust based on parameter type
    if (parameters.type === "categorical" || parameters.priorKnowledge === "none") {
      // For categorical parameters or no prior knowledge, use more exploration
      recommendedStrategy.stages[0].iterations = Math.max(10, Math.floor(budget.experiments * 0.4))
      recommendedStrategy.stages[1].iterations = budget.experiments - recommendedStrategy.stages[0].iterations
      recommendedStrategy.stages[1].explorationWeight = 0.7
    } else if (parameters.priorKnowledge === "extensive") {
      // For extensive prior knowledge, use less exploration
      recommendedStrategy.stages[0].iterations = Math.max(3, Math.floor(budget.experiments * 0.2))
      recommendedStrategy.stages[1].iterations = budget.experiments - recommendedStrategy.stages[0].iterations
      recommendedStrategy.stages[1].explorationWeight = 0.3
    }

    // Adjust based on objective type
    if (objective.type === "target_range") {
      recommendedStrategy.stages[1].acquisitionFunction = "pi" // Probability of improvement for target range
    } else if (parameters.count > 5) {
      // For high-dimensional spaces, use UCB
      recommendedStrategy.stages[1].acquisitionFunction = "ucb"
    }

    return {
      objective: objective,
      parameters: parameters,
      budget: budget,
      strategy: recommendedStrategy,
    }
  }

  return (
    <Card className="w-full max-w-3xl mx-auto">
      <CardHeader>
        <CardTitle>Optimization Wizard</CardTitle>
        <CardDescription>Let's set up your optimization experiment step by step</CardDescription>
      </CardHeader>
      <CardContent>
        {step === 1 && (
          <div className="space-y-6">
            <div className="space-y-2">
              <Label>What are you trying to optimize?</Label>
              <Textarea
                placeholder="e.g., Reaction yield, catalyst performance, material strength..."
                value={config.objective.name}
                onChange={(e) => updateConfig("objective", { name: e.target.value })}
              />
            </div>

            <div className="space-y-2">
              <Label>Optimization Goal</Label>
              <RadioGroup
                value={config.objective.type}
                onValueChange={(value) => updateConfig("objective", { type: value })}
                className="flex flex-col space-y-2"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="maximize" id="maximize" />
                  <Label htmlFor="maximize" className="flex items-center">
                    Maximize
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <HelpCircle className="ml-1 h-4 w-4 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="w-80">
                            Choose this when you want to make a value as large as possible (e.g., yield, efficiency)
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="minimize" id="minimize" />
                  <Label htmlFor="minimize" className="flex items-center">
                    Minimize
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <HelpCircle className="ml-1 h-4 w-4 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="w-80">
                            Choose this when you want to make a value as small as possible (e.g., cost, error)
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="target_range" id="target_range" />
                  <Label htmlFor="target_range" className="flex items-center">
                    Target a specific range
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <HelpCircle className="ml-1 h-4 w-4 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="w-80">
                            Choose this when you want to hit a specific value or range (e.g., pH, temperature)
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </Label>
                </div>
              </RadioGroup>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="space-y-6">
            <div className="space-y-2">
              <Label>What type of parameters are you optimizing?</Label>
              <RadioGroup
                value={config.parameters.type}
                onValueChange={(value) => updateConfig("parameters", { type: value })}
                className="flex flex-col space-y-2"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="continuous" id="continuous" />
                  <Label htmlFor="continuous">Continuous (e.g., temperature, pressure)</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="discrete" id="discrete" />
                  <Label htmlFor="discrete">Discrete (e.g., number of cycles, count)</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="categorical" id="categorical" />
                  <Label htmlFor="categorical">Categorical (e.g., material type, catalyst)</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="mixed" id="mixed" />
                  <Label htmlFor="mixed">Mixed (combination of different types)</Label>
                </div>
              </RadioGroup>
            </div>

            <div className="space-y-2">
              <Label>How many parameters do you have? {config.parameters.count}</Label>
              <Slider
                value={[config.parameters.count]}
                min={1}
                max={10}
                step={1}
                onValueChange={(value) => updateConfig("parameters", { count: value[0] })}
              />
            </div>

            <div className="space-y-2">
              <Label>How much prior knowledge do you have about the parameter space?</Label>
              <Select
                value={config.parameters.priorKnowledge}
                onValueChange={(value) => updateConfig("parameters", { priorKnowledge: value })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select level of prior knowledge" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None (first time exploring)</SelectItem>
                  <SelectItem value="some">Some (have some intuition)</SelectItem>
                  <SelectItem value="extensive">Extensive (know the rough optimum region)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        )}

        {step === 3 && (
          <div className="space-y-6">
            <div className="space-y-2">
              <Label>How many experiments can you run in total? {config.budget.experiments}</Label>
              <Slider
                value={[config.budget.experiments]}
                min={10}
                max={100}
                step={5}
                onValueChange={(value) => updateConfig("budget", { experiments: value[0] })}
              />
              <p className="text-sm text-muted-foreground">
                This is the total number of experiments you can afford to run
              </p>
            </div>

            <div className="space-y-2">
              <Label>How many experiments can you run in parallel? {config.budget.parallelRuns}</Label>
              <Slider
                value={[config.budget.parallelRuns]}
                min={1}
                max={10}
                step={1}
                onValueChange={(value) => updateConfig("budget", { parallelRuns: value[0] })}
              />
              <p className="text-sm text-muted-foreground">This determines the batch size for optimization</p>
            </div>

            <div className="space-y-2">
              <Label>Exploration vs. Exploitation Balance</Label>
              <div className="flex justify-between text-sm text-muted-foreground mb-2">
                <span>Exploitation (Use what works)</span>
                <span>Exploration (Try new areas)</span>
              </div>
              <Slider
                value={[config.strategy.explorationWeight * 100]}
                min={0}
                max={100}
                step={5}
                onValueChange={(value) => updateConfig("strategy", { explorationWeight: value[0] / 100 })}
              />
              <div className="flex justify-between">
                <span className="text-sm">Focused search</span>
                <span className="text-sm">Broad search</span>
              </div>
            </div>
          </div>
        )}

        {step === 4 && (
          <div className="space-y-6">
            <div className="flex items-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <Lightbulb className="h-8 w-8 text-blue-500 mr-4" />
              <div>
                <h3 className="font-medium">Recommended Optimization Strategy</h3>
                <p className="text-sm text-muted-foreground">
                  Based on your inputs, we've created a customized optimization strategy
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-sm font-medium">Objective</p>
                  <p className="text-sm">{config.objective.name || "Not specified"}</p>
                  <p className="text-xs text-muted-foreground">
                    {config.objective.type === "maximize"
                      ? "Maximize"
                      : config.objective.type === "minimize"
                        ? "Minimize"
                        : "Target Range"}
                  </p>
                </div>

                <div className="space-y-1">
                  <p className="text-sm font-medium">Parameters</p>
                  <p className="text-sm">
                    {config.parameters.count} {config.parameters.type} parameters
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {config.parameters.priorKnowledge === "none"
                      ? "No prior knowledge"
                      : config.parameters.priorKnowledge === "some"
                        ? "Some prior knowledge"
                        : "Extensive prior knowledge"}
                  </p>
                </div>
              </div>

              <div className="space-y-1">
                <p className="text-sm font-medium">Recommended Strategy</p>
                <div className="bg-muted p-3 rounded-md">
                  <ol className="list-decimal list-inside space-y-2 text-sm">
                    <li>
                      <span className="font-medium">Initial Exploration:</span> Latin Hypercube Sampling with{" "}
                      {Math.max(5, Math.floor(config.budget.experiments * 0.3))} experiments
                    </li>
                    <li>
                      <span className="font-medium">Bayesian Optimization:</span>{" "}
                      {Math.floor(config.budget.experiments * 0.7)} experiments with{" "}
                      {config.parameters.priorKnowledge === "none"
                        ? "high"
                        : config.parameters.priorKnowledge === "extensive"
                          ? "low"
                          : "balanced"}{" "}
                      exploration ({config.strategy.explorationWeight.toFixed(2)})
                    </li>
                    <li>
                      <span className="font-medium">Acquisition Function:</span>{" "}
                      {config.objective.type === "target_range"
                        ? "Probability of Improvement"
                        : config.parameters.count > 5
                          ? "Upper Confidence Bound"
                          : "Expected Improvement"}
                    </li>
                  </ol>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
      <CardFooter className="flex justify-between">
        {step > 1 ? (
          <Button variant="outline" onClick={prevStep}>
            Back
          </Button>
        ) : (
          <div></div>
        )}

        {step < 4 ? (
          <Button onClick={nextStep}>
            Continue
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        ) : (
          <Button onClick={completeWizard}>
            <Check className="mr-2 h-4 w-4" />
            Apply Recommendations
          </Button>
        )}
      </CardFooter>
    </Card>
  )
}
