"use client"

import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import * as z from "zod"
import { Button } from "@/components/ui/button"
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { type Constraint, ConstraintType } from "@/lib/types"
import { ConstraintVisualization } from "@/components/constraint-visualization"
import { useState, useEffect } from "react"

const constraintSchema = z.object({
  id: z.string().optional(),
  expression: z.string().min(1, "Expression is required"),
  type: z.nativeEnum(ConstraintType),
  value: z.number(),
})

interface ConstraintFormProps {
  constraint: Constraint
  onSubmit: (constraint: Constraint) => void
  onCancel: () => void
}

export function ConstraintForm({ constraint, onSubmit, onCancel }: ConstraintFormProps) {
  const form = useForm<Constraint>({
    resolver: zodResolver(constraintSchema),
    defaultValues: constraint,
  })

  // Add live update state
  const [liveConstraint, setLiveConstraint] = useState<Constraint>(constraint)

  // Watch form value changes and update visualization
  useEffect(() => {
    const subscription = form.watch((value) => {
      setLiveConstraint(value as Constraint)
    })
    return () => subscription.unsubscribe()
  }, [form])

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
        <FormField
          control={form.control}
          name="expression"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Expression</FormLabel>
              <FormControl>
                <Input placeholder="e.g., x1 + x2 + x3" {...field} />
              </FormControl>
              <FormDescription>Variables in your parameter space (e.g., x1, x2)</FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="type"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Constraint Type</FormLabel>
              <Select onValueChange={(value) => field.onChange(value as ConstraintType)} defaultValue={field.value}>
                <FormControl>
                  <SelectTrigger>
                    <SelectValue placeholder="Select constraint type" />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  <SelectItem value={ConstraintType.SUM_EQUALS}>Sum Equals (∑xᵢ = value)</SelectItem>
                  <SelectItem value={ConstraintType.SUM_LESS_THAN}>Sum Less Than (∑xᵢ ≤ value)</SelectItem>
                  <SelectItem value={ConstraintType.SUM_GREATER_THAN}>Sum Greater Than (∑xᵢ ≥ value)</SelectItem>
                </SelectContent>
              </Select>
              <FormDescription>The type of constraint to apply</FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="value"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Value</FormLabel>
              <FormControl>
                <Input
                  type="number"
                  placeholder="1"
                  {...field}
                  onChange={(e) => field.onChange(Number.parseFloat(e.target.value))}
                />
              </FormControl>
              <FormDescription>The target value for this constraint</FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        {liveConstraint.expression && (
          <div className="mt-6 border-t pt-4">
            <h4 className="text-sm font-medium mb-2">Constraint Visualization Preview</h4>
            <ConstraintVisualization constraint={liveConstraint} />
          </div>
        )}

        <div className="flex justify-end gap-2">
          <Button type="button" variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="submit">{constraint.id ? "Update" : "Add"} Constraint</Button>
        </div>
      </form>
    </Form>
  )
}
