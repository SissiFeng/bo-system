"use client"

import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import * as z from "zod"
import { Button } from "@/components/ui/button"
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { type Objective, OptimizationType } from "@/lib/types"
import { useEffect, useState } from "react"
import { useModeStore } from "@/lib/stores/mode-store"
import { Card, CardContent } from "@/components/ui/card"
import { Check, ChevronsUpDown } from "lucide-react"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { cn } from "@/lib/utils"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { objectiveTemplates } from "@/lib/templates"
import { ObjectiveVisualization } from "@/components/objective-visualization"

const objectiveSchema = z.object({
  id: z.string().optional(),
  name: z.string().min(1, "Objective name is required"),
  type: z.nativeEnum(OptimizationType),
  targetMin: z.number().optional(),
  targetMax: z.number().optional(),
  unit: z.string().optional(),
  description: z.string().optional(),
})

interface ObjectiveFormProps {
  objective: Objective
  onSubmit: (objective: Objective) => void
  onCancel: () => void
}

export function ObjectiveForm({ objective, onSubmit, onCancel }: ObjectiveFormProps) {
  const { isSimpleMode } = useModeStore()
  const [activeTab, setActiveTab] = useState("manual")
  const [templateOpen, setTemplateOpen] = useState(false)
  const [selectedTemplate, setSelectedTemplate] = useState("")

  const form = useForm<Objective>({
    resolver: zodResolver(objectiveSchema),
    defaultValues: objective,
  })

  // Add live update state
  const [liveObjective, setLiveObjective] = useState<Objective>(objective)

  // Watch form value changes and update visualization
  useEffect(() => {
    const subscription = form.watch((value) => {
      setLiveObjective(value as Objective)
    })
    return () => subscription.unsubscribe()
  }, [form])

  const objectiveType = form.watch("type")
  const objectiveName = form.watch("name")

  // Auto-detect optimization direction based on name
  useEffect(() => {
    if (isSimpleMode && objectiveName && !objective.id) {
      const name = objectiveName.toLowerCase()
      if (
        name.includes("efficiency") ||
        name.includes("yield") ||
        name.includes("performance") ||
        name.includes("activity") ||
        name.includes("selectivity")
      ) {
        form.setValue("type", OptimizationType.MAXIMIZE)
      } else if (
        name.includes("cost") ||
        name.includes("time") ||
        name.includes("error") ||
        name.includes("loss") ||
        name.includes("consumption") ||
        name.includes("waste") ||
        name.includes("overpotential")
      ) {
        form.setValue("type", OptimizationType.MINIMIZE)
      } else if (name.includes("ph") || name.includes("temperature") || name.includes("bandgap")) {
        form.setValue("type", OptimizationType.TARGET_RANGE)

        // Set reasonable defaults based on objective name
        if (name.includes("ph")) {
          form.setValue("targetMin", 6.5)
          form.setValue("targetMax", 7.5)
          form.setValue("unit", "")
        } else if (name.includes("temperature")) {
          form.setValue("targetMin", 25)
          form.setValue("targetMax", 35)
          form.setValue("unit", "°C")
        } else if (name.includes("bandgap")) {
          form.setValue("targetMin", 1.5)
          form.setValue("targetMax", 2.5)
          form.setValue("unit", "eV")
        }
      }
    }
  }, [objectiveName, form, isSimpleMode, objective.id])

  // Apply template when selected
  useEffect(() => {
    if (selectedTemplate) {
      const template = objectiveTemplates.find((t) => t.id === selectedTemplate)
      if (template) {
        form.reset({
          ...objective,
          name: template.name,
          type: template.type,
          targetMin: template.targetMin,
          targetMax: template.targetMax,
          unit: template.unit,
          description: template.description,
        })
      }
      setTemplateOpen(false)
    }
  }, [selectedTemplate, form, objective])

  // Create a dummy field for template selection to use FormField properly
  const handleTemplateSelect = (templateId: string) => {
    setSelectedTemplate(templateId === selectedTemplate ? "" : templateId)
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid grid-cols-2 mb-4">
            <TabsTrigger value="manual">Manual</TabsTrigger>
            <TabsTrigger value="template">Templates</TabsTrigger>
          </TabsList>

          <TabsContent value="manual">
            <div className="space-y-6">
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Objective Name</FormLabel>
                    <FormControl>
                      <Input placeholder="e.g., OER, current density" {...field} />
                    </FormControl>
                    <FormDescription>A descriptive name for this KPI</FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="type"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Optimization Type</FormLabel>
                    <Select
                      onValueChange={(value) => field.onChange(value as OptimizationType)}
                      defaultValue={field.value}
                    >
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select optimization type" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value={OptimizationType.MAXIMIZE}>Maximize</SelectItem>
                        <SelectItem value={OptimizationType.MINIMIZE}>Minimize</SelectItem>
                        <SelectItem value={OptimizationType.TARGET_RANGE}>Target Range</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormDescription>How this objective should be optimized</FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              {objectiveType === OptimizationType.TARGET_RANGE && (
                <div className="grid grid-cols-2 gap-4">
                  <FormField
                    control={form.control}
                    name="targetMin"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Target Minimum</FormLabel>
                        <FormControl>
                          <Input
                            type="number"
                            placeholder="0"
                            {...field}
                            onChange={(e) => field.onChange(Number.parseFloat(e.target.value))}
                            value={field.value ?? ""}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="targetMax"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Target Maximum</FormLabel>
                        <FormControl>
                          <Input
                            type="number"
                            placeholder="1"
                            {...field}
                            onChange={(e) => field.onChange(Number.parseFloat(e.target.value))}
                            value={field.value ?? ""}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
              )}

              <FormField
                control={form.control}
                name="unit"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Unit (Optional)</FormLabel>
                    <FormControl>
                      <Input placeholder="e.g., %, mA/cm²" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="description"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Description (Optional)</FormLabel>
                    <FormControl>
                      <Input placeholder="Brief description of this objective" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              {/* Add objective visualization */}
              {liveObjective.name && (
                <div className="mt-6 border-t pt-4">
                  <h4 className="text-sm font-medium mb-2">Objective Function Visualization Preview</h4>
                  <ObjectiveVisualization objective={liveObjective} />
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="template">
            <div className="space-y-4">
              {/* Use a hidden FormField to maintain form context */}
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem className="space-y-2">
                    <FormLabel>Select Template</FormLabel>
                    <FormControl>
                      <Popover open={templateOpen} onOpenChange={setTemplateOpen}>
                        <PopoverTrigger asChild>
                          <Button
                            variant="outline"
                            role="combobox"
                            aria-expanded={templateOpen}
                            className="w-full justify-between"
                          >
                            {selectedTemplate
                              ? objectiveTemplates.find((template) => template.id === selectedTemplate)?.name
                              : "Select an objective template..."}
                            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                          </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-full p-0">
                          <Command>
                            <CommandInput placeholder="Search templates..." />
                            <CommandList>
                              <CommandEmpty>No template found.</CommandEmpty>
                              <CommandGroup>
                                {objectiveTemplates.map((template) => (
                                  <CommandItem key={template.id} value={template.id} onSelect={handleTemplateSelect}>
                                    <Check
                                      className={cn(
                                        "mr-2 h-4 w-4",
                                        selectedTemplate === template.id ? "opacity-100" : "opacity-0",
                                      )}
                                    />
                                    <div className="flex flex-col">
                                      <span>{template.name}</span>
                                      <span className="text-xs text-muted-foreground">{template.description}</span>
                                    </div>
                                  </CommandItem>
                                ))}
                              </CommandGroup>
                            </CommandList>
                          </Command>
                        </PopoverContent>
                      </Popover>
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              {selectedTemplate && (
                <Card>
                  <CardContent className="pt-6">
                    <div className="space-y-2">
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-sm font-medium">Type:</p>
                          <p className="text-sm">
                            {objectiveTemplates.find((t) => t.id === selectedTemplate)?.type ===
                            OptimizationType.MAXIMIZE
                              ? "Maximize"
                              : objectiveTemplates.find((t) => t.id === selectedTemplate)?.type ===
                                  OptimizationType.MINIMIZE
                                ? "Minimize"
                                : "Target Range"}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm font-medium">Unit:</p>
                          <p className="text-sm">
                            {objectiveTemplates.find((t) => t.id === selectedTemplate)?.unit || "N/A"}
                          </p>
                        </div>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Description:</p>
                        <p className="text-sm">
                          {objectiveTemplates.find((t) => t.id === selectedTemplate)?.description}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>
        </Tabs>

        <div className="flex justify-end gap-2">
          <Button type="button" variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="submit">{objective.id ? "Update" : "Add"} Objective</Button>
        </div>
      </form>
    </Form>
  )
}
