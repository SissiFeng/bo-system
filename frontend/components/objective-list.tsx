"use client"

import { Button } from "@/components/ui/button"
import { type Objective, OptimizationType } from "@/lib/types"
import { Edit, Trash2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

interface ObjectiveListProps {
  objectives: Objective[]
  onEdit: (objective: Objective) => void
  onDelete: (id: string) => void
}

export function ObjectiveList({ objectives, onEdit, onDelete }: ObjectiveListProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Type</TableHead>
          <TableHead>Target</TableHead>
          <TableHead className="text-right">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {objectives.map((objective) => (
          <TableRow key={objective.id}>
            <TableCell className="font-medium">{objective.name}</TableCell>
            <TableCell>
              <Badge
                variant={
                  objective.type === OptimizationType.MAXIMIZE
                    ? "default"
                    : objective.type === OptimizationType.MINIMIZE
                      ? "destructive"
                      : "outline"
                }
              >
                {objective.type === OptimizationType.MAXIMIZE
                  ? "Maximize"
                  : objective.type === OptimizationType.MINIMIZE
                    ? "Minimize"
                    : "Target Range"}
              </Badge>
            </TableCell>
            <TableCell>
              {objective.type === OptimizationType.TARGET_RANGE ? (
                <span>
                  {objective.targetMin} to {objective.targetMax}
                </span>
              ) : (
                <span>{objective.type === OptimizationType.MAXIMIZE ? "Higher is better" : "Lower is better"}</span>
              )}
            </TableCell>
            <TableCell className="text-right">
              <div className="flex justify-end gap-2">
                <Button variant="ghost" size="icon" onClick={() => onEdit(objective)}>
                  <Edit className="h-4 w-4" />
                </Button>
                <Button variant="ghost" size="icon" onClick={() => onDelete(objective.id)}>
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}
