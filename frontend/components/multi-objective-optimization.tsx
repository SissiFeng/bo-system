"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { Info, HelpCircle, Settings } from "lucide-react"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { ParetoFrontVisualization } from "@/components/pareto-front-visualization"

interface MultiObjectiveOptimizationProps {
  onConfigChange: (config: any) => void
}

export function MultiObjectiveOptimization({ onConfigChange }: MultiObjectiveOptimizationProps) {
  const [activeTab, setActiveTab] = useState("config")
  const [mooEnabled, setMooEnabled] = useState(true)
  const [mooConfig, setMooConfig] = useState({
    algorithm: "ehvi",
    noisyMode: true,
    referencePoint: "auto",
    customReferencePoint: { x: 0, y: 0 },
    objectives: [
      { id: "obj1", name: "Objective 1", weight: 1.0, type: "maximize" },
      { id: "obj2", name: "Objective 2", weight: 1.0, type: "minimize" },
    ],
    paretoFrontSize: 10,
    explorationWeight: 0.5,
    normalizeObjectives: true,
    constraintHandling: "penalty",
  })

  // Mock data for Pareto front visualization
  const mockParetoFront = [
    { x: 0.85, y: 0.15, id: 1, uncertainty: 0.02 },
    { x: 0.88, y: 0.18, id: 2, uncertainty: 0.03 },
    { x: 0.9, y: 0.22, id: 3, uncertainty: 0.01 },
    { x: 0.92, y: 0.25, id: 4, uncertainty: 0.04 },
    { x: 0.94, y: 0.3, id: 5, uncertainty: 0.02 },
    { x: 0.96, y: 0.35, id: 6, uncertainty: 0.05 },
    { x: 0.97, y: 0.4, id: 7, uncertainty: 0.03 },
    { x: 0.98, y: 0.5, id: 8, uncertainty: 0.02 },
    { x: 0.99, y: 0.65, id: 9, uncertainty: 0.04 },
  ]

  // Mock data for non-dominated solutions
  const mockNonDominatedSolutions = [
    { x: 0.82, y: 0.12, id: 10, uncertainty: 0.06 },
    { x: 0.86, y: 0.16, id: 11, uncertainty: 0.05 },
    { x: 0.89, y: 0.2, id: 12, uncertainty: 0.07 },
    { x: 0.93, y: 0.28, id: 13, uncertainty: 0.04 },
    { x: 0.95, y: 0.33, id: 14, uncertainty: 0.06 },
    { x: 0.97, y: 0.45, id: 15, uncertainty: 0.05 },
  ]

  // Mock data for dominated solutions
  const mockDominatedSolutions = [
    { x: 0.75, y: 0.25, id: 16, uncertainty: 0.08 },
    { x: 0.8, y: 0.3, id: 17, uncertainty: 0.07 },
    { x: 0.85, y: 0.35, id: 18, uncertainty: 0.09 },
    { x: 0.9, y: 0.4, id: 19, uncertainty: 0.08 },
    { x: 0.7, y: 0.45, id: 20, uncertainty: 0.1 },
  ]

  const updateConfig = (key: string, value: any) => {
    const newConfig = { ...mooConfig, [key]: value }
    setMooConfig(newConfig)
    onConfigChange(newConfig)
  }

  const updateObjective = (index: number, key: string, value: any) => {
    const newObjectives = [...mooConfig.objectives]
    newObjectives[index] = { ...newObjectives[index], [key]: value }
    updateConfig("objectives", newObjectives)
  }

  const toggleMOO = (enabled: boolean) => {
    setMooEnabled(enabled)
    // Notify parent component about the change
    if (!enabled) {
      onConfigChange({ enabled: false })
    } else {
      onConfigChange({ enabled: true, ...mooConfig })
    }
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <div>
            <CardTitle>Multi-Objective Optimization</CardTitle>
            <CardDescription>Configure and visualize multi-objective optimization</CardDescription>
          </div>
          <div className="flex items-center space-x-2">
            <Label htmlFor="moo-toggle" className="text-sm">
              Enable MOO
            </Label>
            <Switch id="moo-toggle" checked={mooEnabled} onCheckedChange={toggleMOO} />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {mooEnabled ? (
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid grid-cols-3 mb-4">
              <TabsTrigger value="config">Configuration</TabsTrigger>
              <TabsTrigger value="pareto">Pareto Front</TabsTrigger>
              <TabsTrigger value="advanced">Advanced Settings</TabsTrigger>
            </TabsList>

            <TabsContent value="config">
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Algorithm</Label>
                      <Select value={mooConfig.algorithm} onValueChange={(value) => updateConfig("algorithm", value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select MOO algorithm" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="ehvi">Expected Hypervolume Improvement (EHVI)</SelectItem>
                          <SelectItem value="parego">ParEGO</SelectItem>
                          <SelectItem value="smsego">SMS-EGO</SelectItem>
                          <SelectItem value="nsga2">NSGA-II</SelectItem>
                          <SelectItem value="moead">MOEA/D</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        {mooConfig.algorithm === "ehvi"
                          ? "EHVI maximizes the expected increase in the hypervolume of the Pareto front"
                          : mooConfig.algorithm === "parego"
                            ? "ParEGO uses scalarization to convert multi-objective to single-objective problems"
                            : mooConfig.algorithm === "smsego"
                              ? "SMS-EGO uses the S-metric (hypervolume) for multi-objective optimization"
                              : mooConfig.algorithm === "nsga2"
                                ? "NSGA-II is an evolutionary algorithm for multi-objective optimization"
                                : "MOEA/D decomposes multi-objective problems into single-objective subproblems"}
                      </p>
                    </div>

                    {(mooConfig.algorithm === "ehvi" || mooConfig.algorithm === "smsego") && (
                      <div className="flex items-center space-x-2">
                        <Switch
                          id="noisy-mode"
                          checked={mooConfig.noisyMode}
                          onCheckedChange={(checked) => updateConfig("noisyMode", checked)}
                        />
                        <Label htmlFor="noisy-mode" className="flex items-center">
                          Noisy Mode
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <HelpCircle className="ml-1 h-4 w-4 text-muted-foreground" />
                              </TooltipTrigger>
                              <TooltipContent side="right">
                                <p className="w-80">
                                  Enable this when your experimental measurements have noise or uncertainty. The
                                  algorithm will account for measurement uncertainty in the optimization process.
                                </p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        </Label>
                      </div>
                    )}

                    <div className="space-y-2">
                      <Label>Reference Point</Label>
                      <Select
                        value={mooConfig.referencePoint}
                        onValueChange={(value) => updateConfig("referencePoint", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select reference point method" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto">Automatic</SelectItem>
                          <SelectItem value="custom">Custom</SelectItem>
                          <SelectItem value="adaptive">Adaptive</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        Reference point for hypervolume calculation in EHVI and other algorithms
                      </p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <Label>Objective 1</Label>
                        <Badge variant="outline" className="bg-blue-50 text-blue-700">
                          {mooConfig.objectives[0].type}
                        </Badge>
                      </div>
                      <Select
                        value={mooConfig.objectives[0].type}
                        onValueChange={(value) => updateObjective(0, "type", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select optimization type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="maximize">Maximize</SelectItem>
                          <SelectItem value="minimize">Minimize</SelectItem>
                        </SelectContent>
                      </Select>
                      <div className="pt-2">
                        <Label className="text-xs">Weight: {mooConfig.objectives[0].weight.toFixed(1)}</Label>
                        <Slider
                          value={[mooConfig.objectives[0].weight * 10]}
                          min={1}
                          max={10}
                          step={1}
                          onValueChange={(value) => updateObjective(0, "weight", value[0] / 10)}
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <Label>Objective 2</Label>
                        <Badge variant="outline" className="bg-green-50 text-green-700">
                          {mooConfig.objectives[1].type}
                        </Badge>
                      </div>
                      <Select
                        value={mooConfig.objectives[1].type}
                        onValueChange={(value) => updateObjective(1, "type", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select optimization type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="maximize">Maximize</SelectItem>
                          <SelectItem value="minimize">Minimize</SelectItem>
                        </SelectContent>
                      </Select>
                      <div className="pt-2">
                        <Label className="text-xs">Weight: {mooConfig.objectives[1].weight.toFixed(1)}</Label>
                        <Slider
                          value={[mooConfig.objectives[1].weight * 10]}
                          min={1}
                          max={10}
                          step={1}
                          onValueChange={(value) => updateObjective(1, "weight", value[0] / 10)}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-medium">Pareto Front Size: {mooConfig.paretoFrontSize}</h3>
                    <div className="w-1/2">
                      <Slider
                        value={[mooConfig.paretoFrontSize]}
                        min={5}
                        max={50}
                        step={5}
                        onValueChange={(value) => updateConfig("paretoFrontSize", value[0])}
                      />
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Number of Pareto-optimal solutions to generate</p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="pareto">
              <div className="space-y-4">
                <div className="h-[400px]">
                  <ParetoFrontVisualization
                    paretoFront={mockParetoFront}
                    nonDominatedSolutions={mockNonDominatedSolutions}
                    dominatedSolutions={mockDominatedSolutions}
                    objective1={{
                      name: mooConfig.objectives[0].name,
                      type: mooConfig.objectives[0].type,
                    }}
                    objective2={{
                      name: mooConfig.objectives[1].name,
                      type: mooConfig.objectives[1].type,
                    }}
                  />
                </div>

                <div className="bg-muted p-4 rounded-md">
                  <div className="flex items-start">
                    <Info className="h-5 w-5 text-blue-500 mr-2 mt-0.5" />
                    <div>
                      <h3 className="font-medium">About Pareto Optimization</h3>
                      <p className="text-sm text-muted-foreground mt-1">
                        The Pareto front represents the set of non-dominated solutions where improving one objective
                        would worsen another. Points on the Pareto front represent optimal trade-offs between competing
                        objectives. The algorithm will focus on finding and improving this front.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="advanced">
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label>Exploration Weight: {mooConfig.explorationWeight.toFixed(2)}</Label>
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <HelpCircle className="h-4 w-4 text-muted-foreground" />
                            </TooltipTrigger>
                            <TooltipContent side="top">
                              <p className="w-80">
                                Controls the exploration-exploitation trade-off. Higher values favor exploration of
                                uncertain regions, lower values favor exploitation of promising regions.
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </div>
                      <Slider
                        value={[mooConfig.explorationWeight * 100]}
                        min={0}
                        max={100}
                        step={5}
                        onValueChange={(value) => updateConfig("explorationWeight", value[0] / 100)}
                      />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Exploitation</span>
                        <span>Balanced</span>
                        <span>Exploration</span>
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Switch
                        id="normalize-objectives"
                        checked={mooConfig.normalizeObjectives}
                        onCheckedChange={(checked) => updateConfig("normalizeObjectives", checked)}
                      />
                      <Label htmlFor="normalize-objectives" className="flex items-center">
                        Normalize Objectives
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <HelpCircle className="ml-1 h-4 w-4 text-muted-foreground" />
                            </TooltipTrigger>
                            <TooltipContent side="right">
                              <p className="w-80">
                                Normalize objectives to have similar scales. Recommended when objectives have different
                                units or magnitudes.
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </Label>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Constraint Handling</Label>
                      <Select
                        value={mooConfig.constraintHandling}
                        onValueChange={(value) => updateConfig("constraintHandling", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select constraint handling method" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="penalty">Penalty Function</SelectItem>
                          <SelectItem value="feasibility">Feasibility Rules</SelectItem>
                          <SelectItem value="repair">Solution Repair</SelectItem>
                          <SelectItem value="constrained">Constrained Domination</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">Method for handling constraints in MOO</p>
                    </div>

                    {mooConfig.algorithm === "ehvi" && (
                      <div className="pt-4">
                        <Button variant="outline" size="sm" className="w-full">
                          <Settings className="mr-2 h-4 w-4" />
                          Configure EHVI Parameters
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        ) : (
          <div className="flex flex-col items-center justify-center py-8">
            <div className="text-center space-y-2">
              <p className="text-muted-foreground">Multi-objective optimization is currently disabled.</p>
              <p className="text-sm text-muted-foreground">Enable it to configure and visualize Pareto optimization.</p>
              <Button variant="outline" className="mt-4" onClick={() => toggleMOO(true)}>
                Enable Multi-Objective Optimization
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
