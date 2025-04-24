"use client"

import type React from "react"
import { Button } from "@/components/ui/button"
import { Beaker, Box, Calculator, LineChart, Settings, Database, BarChart, ChevronDown } from "lucide-react"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

export function ModuleLibrary() {
  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData("application/reactflow", nodeType)
    event.dataTransfer.effectAllowed = "move"
  }

  return (
    <div className="space-y-4">
      <TooltipProvider>
        <Accordion type="multiple" defaultValue={["input", "algorithm", "output"]}>
          <AccordionItem value="input">
            <AccordionTrigger className="text-sm font-medium">Input Modules</AccordionTrigger>
            <AccordionContent>
              <div className="flex flex-col gap-2 pt-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      className="justify-start"
                      draggable
                      onDragStart={(event) => onDragStart(event, "parameter")}
                    >
                      <Settings className="mr-2 h-4 w-4 text-blue-500" />
                      Parameter Space
                      <Badge variant="outline" className="ml-auto">
                        Input
                      </Badge>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    <p className="max-w-xs">Define your parameter space with ranges and constraints</p>
                  </TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      className="justify-start"
                      draggable
                      onDragStart={(event) => onDragStart(event, "objective")}
                    >
                      <LineChart className="mr-2 h-4 w-4 text-green-500" />
                      Objective Function
                      <Badge variant="outline" className="ml-auto">
                        Input
                      </Badge>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    <p className="max-w-xs">Define the function to optimize (maximize or minimize)</p>
                  </TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      className="justify-start"
                      draggable
                      onDragStart={(event) => onDragStart(event, "constraint")}
                    >
                      <Calculator className="mr-2 h-4 w-4 text-red-500" />
                      Constraint
                      <Badge variant="outline" className="ml-auto">
                        Input
                      </Badge>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    <p className="max-w-xs">Add constraints to your parameter space</p>
                  </TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      className="justify-start"
                      draggable
                      onDragStart={(event) => onDragStart(event, "dataProcessing")}
                    >
                      <Database className="mr-2 h-4 w-4 text-amber-500" />
                      Data Processing
                      <Badge variant="outline" className="ml-auto">
                        Input
                      </Badge>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    <p className="max-w-xs">Process and transform input data</p>
                  </TooltipContent>
                </Tooltip>
              </div>
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="algorithm">
            <AccordionTrigger className="text-sm font-medium">Algorithm Modules</AccordionTrigger>
            <AccordionContent>
              <div className="flex flex-col gap-2 pt-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      className="justify-start"
                      draggable
                      onDragStart={(event) => onDragStart(event, "strategy")}
                    >
                      <Beaker className="mr-2 h-4 w-4 text-purple-500" />
                      Bayesian Optimization
                      <Badge variant="outline" className="ml-auto">
                        Algorithm
                      </Badge>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    <p className="max-w-xs">Bayesian Optimization with customizable acquisition functions</p>
                  </TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" className="justify-start opacity-70" disabled>
                      <Beaker className="mr-2 h-4 w-4 text-indigo-500" />
                      Genetic Algorithm
                      <Badge variant="outline" className="ml-auto">
                        Coming Soon
                      </Badge>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    <p className="max-w-xs">Evolutionary optimization algorithm (coming soon)</p>
                  </TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" className="justify-start opacity-70" disabled>
                      <Beaker className="mr-2 h-4 w-4 text-sky-500" />
                      Grid Search
                      <Badge variant="outline" className="ml-auto">
                        Coming Soon
                      </Badge>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    <p className="max-w-xs">Exhaustive search through parameter space (coming soon)</p>
                  </TooltipContent>
                </Tooltip>
              </div>
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="output">
            <AccordionTrigger className="text-sm font-medium">Output Modules</AccordionTrigger>
            <AccordionContent>
              <div className="flex flex-col gap-2 pt-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      className="justify-start"
                      draggable
                      onDragStart={(event) => onDragStart(event, "analysis")}
                    >
                      <BarChart className="mr-2 h-4 w-4 text-cyan-500" />
                      Result Analysis
                      <Badge variant="outline" className="ml-auto">
                        Output
                      </Badge>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    <p className="max-w-xs">Analyze and visualize optimization results</p>
                  </TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      className="justify-start"
                      draggable
                      onDragStart={(event) => onDragStart(event, "output")}
                    >
                      <Box className="mr-2 h-4 w-4 text-slate-500" />
                      Output
                      <Badge variant="outline" className="ml-auto">
                        Output
                      </Badge>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    <p className="max-w-xs">Export optimization results in various formats</p>
                  </TooltipContent>
                </Tooltip>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </TooltipProvider>

      <div className="pt-4 border-t">
        <h3 className="text-sm font-medium mb-2">Quick Templates</h3>
        <div className="flex flex-col gap-2">
          <Button variant="secondary" size="sm" className="justify-start">
            <ChevronDown className="mr-2 h-4 w-4" />
            Basic BO Workflow
          </Button>
          <Button variant="secondary" size="sm" className="justify-start">
            <ChevronDown className="mr-2 h-4 w-4" />
            Multi-Objective Optimization
          </Button>
        </div>
      </div>
    </div>
  )
}
