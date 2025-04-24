"use client"

import { Button } from "@/components/ui/button"
import { type Parameter, ParameterType } from "@/lib/types"
import { Edit, Trash2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

interface ParameterListProps {
  parameters: Parameter[]
  onEdit: (parameter: Parameter) => void
  onDelete: (id: string) => void
}

export function ParameterList({ parameters, onEdit, onDelete }: ParameterListProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Type</TableHead>
          <TableHead>Range / Values</TableHead>
          <TableHead className="text-right">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {parameters.map((parameter) => (
          <TableRow key={parameter.id}>
            <TableCell className="font-medium">{parameter.name}</TableCell>
            <TableCell>
              <Badge variant="outline">
                {parameter.type === ParameterType.CONTINUOUS
                  ? "Continuous"
                  : parameter.type === ParameterType.DISCRETE
                    ? "Discrete"
                    : "Categorical"}
              </Badge>
            </TableCell>
            <TableCell>
              {parameter.type === ParameterType.CATEGORICAL ? (
                <span>{parameter.values?.join(", ")}</span>
              ) : (
                <span>
                  {parameter.min} to {parameter.max}
                  {parameter.type === ParameterType.DISCRETE && parameter.step && ` (step: ${parameter.step})`}
                </span>
              )}
            </TableCell>
            <TableCell className="text-right">
              <div className="flex justify-end gap-2">
                <Button variant="ghost" size="icon" onClick={() => onEdit(parameter)}>
                  <Edit className="h-4 w-4" />
                </Button>
                <Button variant="ghost" size="icon" onClick={() => onDelete(parameter.id)}>
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
