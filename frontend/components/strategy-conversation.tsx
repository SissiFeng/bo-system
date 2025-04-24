"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Check, MessageCircle } from "lucide-react"

interface StrategyOption {
  id: string
  name: string
  description: string
  explorationWeight: number
  batchSize: number
  acquisitionFunction: string
}

interface StrategyConversationProps {
  onSelectStrategy: (strategy: StrategyOption) => void
}

export function StrategyConversation({ onSelectStrategy }: StrategyConversationProps) {
  const [activeTab, setActiveTab] = useState("options")
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null)
  const [customStrategy, setCustomStrategy] = useState<StrategyOption>({
    id: "custom",
    name: "Custom Strategy",
    description: "Your custom optimization strategy",
    explorationWeight: 0.5,
    batchSize: 5,
    acquisitionFunction: "ei",
  })

  const strategyOptions: StrategyOption[] = [
    {
      id: "exploit",
      name: "Exploit-Heavy Batch",
      description:
        "Focus on areas with high predicted performance. Good when you're confident in your model and want to refine the best solutions.",
      explorationWeight: 0.2,
      batchSize: 5,
      acquisitionFunction: "ei",
    },
    {
      id: "explore",
      name: "Exploration Batch",
      description:
        "Focus on areas with high uncertainty (σ > 40%). Good for early stages or when you want to discover new regions.",
      explorationWeight: 0.8,
      batchSize: 8,
      acquisitionFunction: "ucb",
    },
    {
      id: "balanced",
      name: "Balanced Approach",
      description: "Equal mix of exploration and exploitation. A good default strategy for most optimization problems.",
      explorationWeight: 0.5,
      batchSize: 6,
      acquisitionFunction: "ei",
    },
  ]

  const handleSelectPreset = (strategyId: string) => {
    setSelectedStrategy(strategyId)
  }

  const handleApplyStrategy = () => {
    if (selectedStrategy) {
      const strategy = strategyOptions.find((s) => s.id === selectedStrategy) || customStrategy
      onSelectStrategy(strategy)
    } else {
      onSelectStrategy(customStrategy)
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center">
          <MessageCircle className="mr-2 h-5 w-5" />
          Optimization Strategy Discussion
        </CardTitle>
        <CardDescription>Let's discuss the best approach for your next optimization round</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid grid-cols-2 mb-4">
            <TabsTrigger value="options">Strategy Options</TabsTrigger>
            <TabsTrigger value="custom">Custom Strategy</TabsTrigger>
          </TabsList>

          <TabsContent value="options" className="space-y-4">
            <div className="text-sm text-muted-foreground mb-4">
              Select a strategy that best fits your current optimization stage:
            </div>

            <div className="space-y-4">
              {strategyOptions.map((strategy) => (
                <Card
                  key={strategy.id}
                  className={`cursor-pointer transition-all ${
                    selectedStrategy === strategy.id ? "border-primary ring-1 ring-primary" : ""
                  }`}
                  onClick={() => handleSelectPreset(strategy.id)}
                >
                  <CardContent className="p-4">
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-medium flex items-center">
                          {strategy.name}
                          {selectedStrategy === strategy.id && <Check className="ml-2 h-4 w-4 text-primary" />}
                        </h3>
                        <p className="text-sm text-muted-foreground mt-1">{strategy.description}</p>
                      </div>
                      <Badge
                        variant={
                          strategy.explorationWeight > 0.6
                            ? "secondary"
                            : strategy.explorationWeight < 0.4
                              ? "default"
                              : "outline"
                        }
                      >
                        {strategy.explorationWeight > 0.6
                          ? "Explore"
                          : strategy.explorationWeight < 0.4
                            ? "Exploit"
                            : "Balanced"}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-2 gap-4 mt-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Batch Size:</span> {strategy.batchSize}
                      </div>
                      <div>
                        <span className="text-muted-foreground">Acquisition:</span>{" "}
                        {strategy.acquisitionFunction === "ei"
                          ? "Expected Improvement"
                          : strategy.acquisitionFunction === "ucb"
                            ? "Upper Confidence Bound"
                            : "Probability of Improvement"}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="custom" className="space-y-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Exploration vs. Exploitation</Label>
                <div className="flex justify-between text-sm text-muted-foreground mb-2">
                  <span>Exploitation</span>
                  <span>Balanced</span>
                  <span>Exploration</span>
                </div>
                <Slider
                  value={[customStrategy.explorationWeight * 100]}
                  min={0}
                  max={100}
                  step={5}
                  onValueChange={(value) =>
                    setCustomStrategy({
                      ...customStrategy,
                      explorationWeight: value[0] / 100,
                    })
                  }
                />
              </div>

              <div className="space-y-2">
                <Label>Batch Size</Label>
                <div className="flex justify-between text-sm text-muted-foreground mb-2">
                  <span>Small (3)</span>
                  <span>Medium (6)</span>
                  <span>Large (10)</span>
                </div>
                <Slider
                  value={[customStrategy.batchSize]}
                  min={3}
                  max={10}
                  step={1}
                  onValueChange={(value) =>
                    setCustomStrategy({
                      ...customStrategy,
                      batchSize: value[0],
                    })
                  }
                />
              </div>

              <div className="space-y-2">
                <Label>Acquisition Function</Label>
                <RadioGroup
                  value={customStrategy.acquisitionFunction}
                  onValueChange={(value) =>
                    setCustomStrategy({
                      ...customStrategy,
                      acquisitionFunction: value,
                    })
                  }
                  className="flex flex-col space-y-2 mt-2"
                >
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="ei" id="ei" />
                    <Label htmlFor="ei">Expected Improvement (balanced)</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="ucb" id="ucb" />
                    <Label htmlFor="ucb">Upper Confidence Bound (exploration-focused)</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="pi" id="pi" />
                    <Label htmlFor="pi">Probability of Improvement (exploitation-focused)</Label>
                  </div>
                </RadioGroup>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
      <CardFooter>
        <Button onClick={handleApplyStrategy}>Apply Strategy</Button>
      </CardFooter>
    </Card>
  )
}
