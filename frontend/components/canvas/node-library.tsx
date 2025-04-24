"use client"

import type React from "react"

import { Button } from "@/components/ui/button"
import { Beaker, Box, Calculator, LineChart, Settings } from "lucide-react"

export function NodeLibrary() {
  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData("application/reactflow", nodeType)
    event.dataTransfer.effectAllowed = "move"
  }

  return (
    <div className="flex flex-col gap-2">
      <Button
        variant="outline"
        className="justify-start"
        draggable
        onDragStart={(event) => onDragStart(event, "parameter")}
      >
        <Settings className="mr-2 h-4 w-4" />
        Parameter
      </Button>
      <Button
        variant="outline"
        className="justify-start"
        draggable
        onDragStart={(event) => onDragStart(event, "objective")}
      >
        <LineChart className="mr-2 h-4 w-4" />
        Objective
      </Button>
      <Button
        variant="outline"
        className="justify-start"
        draggable
        onDragStart={(event) => onDragStart(event, "constraint")}
      >
        <Calculator className="mr-2 h-4 w-4" />
        Constraint
      </Button>
      <Button
        variant="outline"
        className="justify-start"
        draggable
        onDragStart={(event) => onDragStart(event, "strategy")}
      >
        <Beaker className="mr-2 h-4 w-4" />
        Strategy
      </Button>
      <Button
        variant="outline"
        className="justify-start"
        draggable
        onDragStart={(event) => onDragStart(event, "output")}
      >
        <Box className="mr-2 h-4 w-4" />
        Output
      </Button>
    </div>
  )
}
