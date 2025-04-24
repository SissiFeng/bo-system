"use client"

import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import * as z from "zod"
import { Button } from "@/components/ui/button"
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { type Parameter, ParameterType } from "@/lib/types"
import { useEffect, useState } from "react"
import { useModeStore } from "@/lib/stores/mode-store"
import { Card, CardContent } from "@/components/ui/card"
import { Check, ChevronsUpDown, FileUp, Lightbulb } from "lucide-react"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { cn } from "@/lib/utils"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { parameterTemplates } from "@/lib/templates"
// Import the visualization component
import { ParameterVisualization } from "@/components/parameter-visualization"

const parameterSchema = z.object({
  id: z.string().optional(),
  name: z.string().min(1, "Parameter name is required"),
  type: z.nativeEnum(ParameterType),
  min: z.number().optional(),
  max: z.number().optional(),
  values: z.array(z.string()).optional(),
  step: z.number().optional(),
  unit: z.string().optional(),
  description: z.string().optional(),
})

interface ParameterFormProps {
  parameter: Parameter
  onSubmit: (parameter: Parameter) => void
  onCancel: () => void
  existingParameters?: Parameter[]
}

export function ParameterForm({ parameter, onSubmit, onCancel, existingParameters = [] }: ParameterFormProps) {
  const { isSimpleMode } = useModeStore()
  const [activeTab, setActiveTab] = useState("manual")
  const [templateOpen, setTemplateOpen] = useState(false)
  const [selectedTemplate, setSelectedTemplate] = useState("")
  const [batchInput, setBatchInput] = useState("")

  const form = useForm<Parameter>({
    resolver: zodResolver(parameterSchema),
    defaultValues: parameter,
  })

  // Add live update state
  const [liveParameter, setLiveParameter] = useState<Parameter>(parameter)

  // Watch form value changes and update visualization
  useEffect(() => {
    const subscription = form.watch((value) => {
      setLiveParameter(value as Parameter)
    })
    return () => subscription.unsubscribe()
  }, [form])

  const parameterType = form.watch("type")
  const parameterName = form.watch("name")

  // Auto-suggest parameter type based on name
  useEffect(() => {
    if (isSimpleMode && parameterName && !parameter.id) {
      const name = parameterName.toLowerCase()
      if (name.includes("temp") || name.includes("temperature") || name.includes("pressure") || name.includes("time")) {
        form.setValue("type", ParameterType.CONTINUOUS)

        // Set reasonable defaults based on parameter name
        if (name.includes("temp") || name.includes("temperature")) {
          form.setValue("min", 25)
          form.setValue("max", 100)
          form.setValue("unit", "°C")
        } else if (name.includes("pressure")) {
          form.setValue("min", 1)
          form.setValue("max", 10)
          form.setValue("unit", "bar")
        } else if (name.includes("time")) {
          form.setValue("min", 1)
          form.setValue("max", 60)
          form.setValue("unit", "min")
        }
      } else if (name.includes("count") || name.includes("number") || name.includes("quantity")) {
        form.setValue("type", ParameterType.DISCRETE)
        form.setValue("min", 1)
        form.setValue("max", 10)
        form.setValue("step", 1)
      } else if (name.includes("type") || name.includes("category") || name.includes("material")) {
        form.setValue("type", ParameterType.CATEGORICAL)
        form.setValue("values", ["Option 1", "Option 2", "Option 3"])
      }
    }
  }, [parameterName, form, isSimpleMode, parameter.id])

  // Apply template when selected
  useEffect(() => {
    if (selectedTemplate) {
      const template = parameterTemplates.find((t) => t.id === selectedTemplate)
      if (template) {
        // Use form.reset to set all values at once instead of multiple setValue calls
        form.reset({
          ...parameter,
          name: template.name,
          type: template.type,
          min: template.min,
          max: template.max,
          step: template.step,
          values: template.values,
          unit: template.unit,
          description: template.description,
        })
      }
      setTemplateOpen(false)
    }
  }, [selectedTemplate, form, parameter])

  const handleTemplateSelect = (templateId: string) => {
    setSelectedTemplate(templateId === selectedTemplate ? "" : templateId)
  }

  const handleBatchImport = () => {
    try {
      // Simple CSV parsing
      const lines = batchInput.trim().split("\n")
      const headers = lines[0].split(",").map((h) => h.trim())

      // Extract parameter names from the first line
      const paramNames = headers.filter((h) => h !== "")

      // Create a sample parameter from the first name
      if (paramNames.length > 0) {
        const newParameter = { ...parameter }
        newParameter.name = paramNames[0]

        // Try to guess if it's numeric or categorical
        const secondLine = lines.length > 1 ? lines[1].split(",") : []
        if (secondLine.length > 0) {
          const firstValue = secondLine[0].trim()
          const isNumeric = !isNaN(Number(firstValue))

          if (isNumeric) {
            newParameter.type = ParameterType.CONTINUOUS

            // Try to determine min/max from data
            const values = lines.slice(1).map((line) => Number(line.split(",")[0].trim()))
            const validValues = values.filter((v) => !isNaN(v))

            if (validValues.length > 0) {
              newParameter.min = Math.min(...validValues)
              newParameter.max = Math.max(...validValues)
            }
          } else {
            newParameter.type = ParameterType.CATEGORICAL

            // Extract unique values for categorical
            const uniqueValues = new Set<string>()
            lines.slice(1).forEach((line) => {
              const value = line.split(",")[0].trim()
              if (value) uniqueValues.add(value)
            })

            newParameter.values = Array.from(uniqueValues)
          }
        }

        // Reset the form with the new values
        form.reset(newParameter)
        setActiveTab("manual")
      }
    } catch (error) {
      console.error("Error parsing batch input:", error)
    }
  }

  // Create a single form that wraps all tabs
  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid grid-cols-3 mb-4">
            <TabsTrigger value="manual">Manual</TabsTrigger>
            <TabsTrigger value="template">Templates</TabsTrigger>
            <TabsTrigger value="batch">Batch Import</TabsTrigger>
          </TabsList>

          <TabsContent value="manual">
            <div className="space-y-6">
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Parameter Name</FormLabel>
                    <FormControl>
                      <Input placeholder="e.g., temperature" {...field} />
                    </FormControl>
                    <FormDescription>A unique identifier for this parameter</FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="type"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Parameter Type</FormLabel>
                    <Select
                      onValueChange={(value) => field.onChange(value as ParameterType)}
                      defaultValue={field.value}
                    >
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select parameter type" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value={ParameterType.CONTINUOUS}>Continuous (e.g., temperature)</SelectItem>
                        <SelectItem value={ParameterType.DISCRETE}>Discrete (e.g., count of items)</SelectItem>
                        <SelectItem value={ParameterType.CATEGORICAL}>Categorical (e.g., material type)</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormDescription>The type determines how this parameter is sampled</FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              {parameterType === ParameterType.CONTINUOUS && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <FormField
                      control={form.control}
                      name="min"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Minimum Value</FormLabel>
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
                      name="max"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Maximum Value</FormLabel>
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
                  <FormField
                    control={form.control}
                    name="unit"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Unit (Optional)</FormLabel>
                        <FormControl>
                          <Input placeholder="e.g., °C, bar, %" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </>
              )}

              {parameterType === ParameterType.DISCRETE && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <FormField
                      control={form.control}
                      name="min"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Minimum Value</FormLabel>
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
                      name="max"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Maximum Value</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              placeholder="10"
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
                  <FormField
                    control={form.control}
                    name="step"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Step Size</FormLabel>
                        <FormControl>
                          <Input
                            type="number"
                            placeholder="1"
                            {...field}
                            onChange={(e) => field.onChange(Number.parseFloat(e.target.value))}
                            value={field.value ?? ""}
                          />
                        </FormControl>
                        <FormDescription>The increment between values (e.g., 1 for integers)</FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </>
              )}

              {parameterType === ParameterType.CATEGORICAL && (
                <FormField
                  control={form.control}
                  name="values"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Possible Values</FormLabel>
                      <FormControl>
                        <Input
                          placeholder="value1, value2, value3"
                          value={field.value?.join(", ") || ""}
                          onChange={(e) => field.onChange(e.target.value.split(",").map((v) => v.trim()))}
                        />
                      </FormControl>
                      <FormDescription>Comma-separated list of possible values</FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              )}

              <FormField
                control={form.control}
                name="description"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Description (Optional)</FormLabel>
                    <FormControl>
                      <Input placeholder="Brief description of this parameter" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              {/* Add parameter visualization */}
              {liveParameter.name && (
                <div className="mt-6 border-t pt-4">
                  <h4 className="text-sm font-medium mb-2">Parameter Visualization Preview</h4>
                  <ParameterVisualization parameter={liveParameter} />
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
                              ? parameterTemplates.find((template) => template.id === selectedTemplate)?.name
                              : "Select a parameter template..."}
                            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                          </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-full p-0">
                          <Command>
                            <CommandInput placeholder="Search templates..." />
                            <CommandList>
                              <CommandEmpty>No template found.</CommandEmpty>
                              <CommandGroup>
                                {parameterTemplates.map((template) => (
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
                            {parameterTemplates.find((t) => t.id === selectedTemplate)?.type ===
                            ParameterType.CONTINUOUS
                              ? "Continuous"
                              : parameterTemplates.find((t) => t.id === selectedTemplate)?.type ===
                                  ParameterType.DISCRETE
                                ? "Discrete"
                                : "Categorical"}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm font-medium">Range:</p>
                          <p className="text-sm">
                            {parameterTemplates.find((t) => t.id === selectedTemplate)?.type ===
                            ParameterType.CATEGORICAL
                              ? parameterTemplates.find((t) => t.id === selectedTemplate)?.values?.join(", ")
                              : `${parameterTemplates.find((t) => t.id === selectedTemplate)?.min} to ${
                                  parameterTemplates.find((t) => t.id === selectedTemplate)?.max
                                } ${parameterTemplates.find((t) => t.id === selectedTemplate)?.unit || ""}`}
                          </p>
                        </div>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Description:</p>
                        <p className="text-sm">
                          {parameterTemplates.find((t) => t.id === selectedTemplate)?.description}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          <TabsContent value="batch">
            <div className="space-y-4">
              {/* Use a hidden FormField to maintain form context */}
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem className="space-y-2">
                    <FormLabel>Paste CSV or Excel Data</FormLabel>
                    <FormControl>
                      <Textarea
                        placeholder="parameter1,parameter2,parameter3
10,20,30
15,25,35"
                        className="min-h-[150px] font-mono"
                        value={batchInput}
                        onChange={(e) => setBatchInput(e.target.value)}
                      />
                    </FormControl>
                    <FormDescription>
                      First row should contain parameter names. We'll extract the first parameter for now.
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <div className="flex items-center gap-2 text-sm">
                <Lightbulb className="h-4 w-4 text-yellow-500" />
                <span className="text-muted-foreground">
                  Tip: You can import multiple parameters at once by adding them one by one after preview.
                </span>
              </div>

              <Button type="button" onClick={handleBatchImport} disabled={!batchInput.trim()}>
                <FileUp className="mr-2 h-4 w-4" />
                Preview Parameter
              </Button>
            </div>
          </TabsContent>
        </Tabs>

        <div className="flex justify-end gap-2">
          <Button type="button" variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="submit">{parameter.id ? "Update" : "Add"} Parameter</Button>
        </div>
      </form>
    </Form>
  )
}
