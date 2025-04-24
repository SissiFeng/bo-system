"use client"

import { Button } from "@/components/ui/button"
import { type Constraint, ConstraintType } from "@/lib/types"
import { Edit, Trash2 } from "lucide-react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

interface ConstraintListProps {
  constraints: Constraint[]
  onEdit: (constraint: Constraint) => void
  onDelete: (id: string) => void
}

export function ConstraintList({ constraints, onEdit, onDelete }: ConstraintListProps) {
  const getConstraintSymbol = (type: ConstraintType) => {
    switch (type) {
      case ConstraintType.SUM_EQUALS:
        return "="
      case ConstraintType.SUM_LESS_THAN:
        return "≤"
      case ConstraintType.SUM_GREATER_THAN:
        return "≥"
      default:
        return "="
    }
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Expression</TableHead>
          <TableHead>Constraint</TableHead>
          <TableHead className="text-right">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {constraints.map((constraint) => (
          <TableRow key={constraint.id}>
            <TableCell className="font-medium">{constraint.expression}</TableCell>
            <TableCell>
              {constraint.expression} {getConstraintSymbol(constraint.type)} {constraint.value}
            </TableCell>
            <TableCell className="text-right">
              <div className="flex justify-end gap-2">
                <Button variant="ghost" size="icon" onClick={() => onEdit(constraint)}>
                  <Edit className="h-4 w-4" />
                </Button>
                <Button variant="ghost" size="icon" onClick={() => onDelete(constraint.id)}>
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
