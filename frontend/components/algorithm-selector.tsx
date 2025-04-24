"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { HelpCircle, Info, Check, BarChart, Lightbulb } from "lucide-react"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface AlgorithmSelectorProps {
  onSelect: (algorithm: any) => void
}

export function AlgorithmSelector({ onSelect }: AlgorithmSelectorProps) {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState("bayesian")
  const [activeTab, setActiveTab] = useState("select")
  const [compareMode, setCompareMode] = useState(false)
  const [algorithmConfig, setAlgorithmConfig] = useState({
    // Bayesian Optimization settings
    bayesian: {
      multiObjective: false,
      acquisitionFunction: "ei",
      mooAcquisitionFunction: "ehvi",
      noisyMoo: false,
      referencePointMethod: "auto",
      kernel: "matern",
      explorationWeight: 0.5,
      noiseLevel: 0.1,
      lengthScale: 1.0,
      useARD: false,
      batchSize: 5,
      iterations: 20,
      initialSampling: "lhs",
      initialSamples: 10,
    },
    // Genetic Algorithm settings
    genetic: {
      populationSize: 50,
      generations: 100,
      crossoverRate: 0.8,
      mutationRate: 0.1,
      elitismCount: 2,
      selectionMethod: "tournament",
      tournamentSize: 3,
      crossoverMethod: "uniform",
      mutationMethod: "gaussian",
      mutationSigma: 0.1,
    },
    // Simulated Annealing settings
    annealing: {
      initialTemperature: 100,
      coolingRate: 0.95,
      iterations: 1000,
      neighborhoodSize: 0.1,
      reheatingSchedule: "none",
      minTemperature: 0.01,
      equilibriumSteps: 10,
    },
    // Particle Swarm Optimization settings
    pso: {
      swarmSize: 30,
      iterations: 100,
      inertiaWeight: 0.7,
      cognitiveCoefficient: 1.5,
      socialCoefficient: 1.5,
      velocityClamp: 2.0,
      topology: "global",
      neighborhoodSize: 3,
    },
    // NSGA-II settings
    nsga: {
      populationSize: 100,
      generations: 100,
      crossoverRate: 0.9,
      mutationRate: 0.1,
      tournamentSize: 2,
      crossoverDistribution: 20,
      mutationDistribution: 20,
      referencePoints: false,
      numReferencePoints: 10,
    },
    // Grid Search settings
    grid: {
      pointsPerDimension: 10,
      useLogScale: false,
      parallelEvaluation: true,
      maxPoints: 1000,
    },
    // Random Search settings
    random: {
      numSamples: 100,
      samplingMethod: "uniform",
      seed: 42,
      parallelEvaluation: true,
    },
    // Evolutionary Strategy settings
    evolution: {
      populationSize: 20,
      offspringSize: 100,
      generations: 100,
      adaptiveSigma: true,
      initialSigma: 0.1,
      recombinationMethod: "intermediate",
      selectionPressure: 0.5,
    },
  })

  const updateAlgorithmConfig = (algorithm: string, key: string, value: any) => {
    setAlgorithmConfig({
      ...algorithmConfig,
      [algorithm]: {
        ...algorithmConfig[algorithm as keyof typeof algorithmConfig],
        [key]: value,
      },
    })
  }

  const handleApply = () => {
    onSelect({
      type: selectedAlgorithm,
      config: algorithmConfig[selectedAlgorithm as keyof typeof algorithmConfig],
    })
  }

  const algorithmInfo = {
    bayesian: {
      name: "Bayesian Optimization",
      description:
        "Model-based optimization that builds a surrogate model of the objective function and uses an acquisition function to decide where to sample next.",
      pros: ["Sample-efficient", "Handles noisy observations", "Works well with expensive evaluations"],
      cons: ["Scales poorly with dimensions", "Complex hyperparameter tuning", "Computationally intensive"],
      bestFor: ["Expensive black-box functions", "Low to medium dimensions (1-20)", "Noisy evaluations"],
      complexity: "Medium",
      explorationExploitation: "Balanced",
      parallelization: "Medium",
    },
    genetic: {
      name: "Genetic Algorithm",
      description:
        "Evolutionary algorithm that mimics natural selection, using mutation, crossover, and selection to evolve a population of solutions.",
      pros: ["Handles discrete and mixed variables", "Good for combinatorial problems", "Parallelizable"],
      cons: ["Requires many function evaluations", "May converge to local optima", "Parameter tuning can be difficult"],
      bestFor: ["Combinatorial problems", "Mixed variable types", "Non-differentiable functions"],
      complexity: "Medium",
      explorationExploitation: "Exploration-heavy",
      parallelization: "High",
    },
    annealing: {
      name: "Simulated Annealing",
      description:
        "Probabilistic technique inspired by annealing in metallurgy, gradually decreasing the probability of accepting worse solutions.",
      pros: ["Can escape local optima", "Simple to implement", "Works well for discrete problems"],
      cons: ["Slow convergence", "Many function evaluations", "Sensitive to cooling schedule"],
      bestFor: ["Discrete optimization", "Problems with many local optima", "When gradient information is unavailable"],
      complexity: "Low",
      explorationExploitation: "Starts exploration-heavy, becomes exploitation-heavy",
      parallelization: "Low",
    },
    pso: {
      name: "Particle Swarm Optimization",
      description:
        "Population-based stochastic optimization technique inspired by social behavior of bird flocking or fish schooling.",
      pros: ["Simple implementation", "Few parameters", "Works well in continuous spaces"],
      cons: ["Can get trapped in local optima", "Sensitive to parameter settings", "Many function evaluations"],
      bestFor: ["Continuous optimization", "Non-differentiable functions", "Multi-modal landscapes"],
      complexity: "Low",
      explorationExploitation: "Balanced",
      parallelization: "High",
    },
    nsga: {
      name: "NSGA-II",
      description:
        "Multi-objective genetic algorithm that uses non-dominated sorting and crowding distance to find the Pareto front.",
      pros: ["Handles multiple objectives", "Finds diverse Pareto-optimal solutions", "Elitist approach"],
      cons: ["Computationally intensive", "Many function evaluations", "Complex implementation"],
      bestFor: ["Multi-objective optimization", "Finding Pareto fronts", "When trade-offs are important"],
      complexity: "High",
      explorationExploitation: "Balanced",
      parallelization: "High",
    },
    grid: {
      name: "Grid Search",
      description: "Exhaustive search through a manually specified subset of the parameter space.",
      pros: ["Simple to implement", "Deterministic", "Easy to parallelize"],
      cons: ["Curse of dimensionality", "Inefficient", "Fixed resolution"],
      bestFor: ["Low-dimensional spaces", "When exhaustive search is feasible", "Benchmark comparisons"],
      complexity: "Very Low",
      explorationExploitation: "Fixed exploration",
      parallelization: "Very High",
    },
    random: {
      name: "Random Search",
      description: "Randomly samples points from the parameter space.",
      pros: ["Simple to implement", "Often better than grid search", "Easy to parallelize"],
      cons: ["Inefficient", "No learning from previous evaluations", "Many function evaluations"],
      bestFor: ["Initial exploration", "Baseline comparison", "When prior knowledge is limited"],
      complexity: "Very Low",
      explorationExploitation: "Pure exploration",
      parallelization: "Very High",
    },
    evolution: {
      name: "Evolution Strategy",
      description:
        "Evolutionary algorithm that adapts mutation step sizes, commonly used in the form of CMA-ES (Covariance Matrix Adaptation Evolution Strategy).",
      pros: ["Adapts to the landscape", "Works well for non-convex problems", "No gradient needed"],
      cons: ["Many function evaluations", "Complex implementation", "Sensitive to initialization"],
      bestFor: ["Non-convex optimization", "When gradients are unavailable", "Black-box optimization"],
      complexity: "High",
      explorationExploitation: "Adaptive",
      parallelization: "Medium",
    },
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Algorithm Selector</CardTitle>
        <CardDescription>Choose and configure the optimization algorithm for your problem</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid grid-cols-3 mb-4">
            <TabsTrigger value="select">Select Algorithm</TabsTrigger>
            <TabsTrigger value="configure">Configure</TabsTrigger>
            <TabsTrigger value="compare">Compare</TabsTrigger>
          </TabsList>

          <TabsContent value="select">
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(algorithmInfo).map(([key, info]) => (
                  <Card
                    key={key}
                    className={`cursor-pointer transition-all ${
                      selectedAlgorithm === key ? "border-primary ring-1 ring-primary" : ""
                    }`}
                    onClick={() => setSelectedAlgorithm(key)}
                  >
                    <CardHeader className="p-4 pb-2">
                      <CardTitle className="text-base flex items-center justify-between">
                        {info.name}
                        {selectedAlgorithm === key && <Check className="h-4 w-4 text-primary" />}
                      </CardTitle>
                      <CardDescription className="line-clamp-2">{info.description}</CardDescription>
                    </CardHeader>
                    <CardContent className="p-4 pt-0">
                      <div className="flex flex-wrap gap-1 mt-2">
                        <Badge
                          variant="outline"
                          className="bg-blue-50 text-blue-700 dark:bg-blue-900 dark:text-blue-300"
                        >
                          {info.complexity} complexity
                        </Badge>
                        <Badge
                          variant="outline"
                          className="bg-green-50 text-green-700 dark:bg-green-900 dark:text-green-300"
                        >
                          {info.parallelization} parallelization
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>

              <div className="bg-muted p-4 rounded-md">
                <div className="flex items-start">
                  <Lightbulb className="h-5 w-5 text-amber-500 mr-2 mt-0.5" />
                  <div>
                    <h3 className="font-medium">Recommended for your problem</h3>
                    <p className="text-sm text-muted-foreground">
                      Based on your parameter space (5 parameters, 2 objectives), we recommend:
                    </p>
                    <ul className="text-sm mt-2 space-y-1">
                      <li className="flex items-center">
                        <Check className="h-4 w-4 text-green-500 mr-1" />
                        <span className="font-medium">Bayesian Optimization</span> - Efficient for expensive evaluations
                      </li>
                      <li className="flex items-center">
                        <Check className="h-4 w-4 text-green-500 mr-1" />
                        <span className="font-medium">NSGA-II</span> - Good for multi-objective problems
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="configure">
            {selectedAlgorithm === "bayesian" && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Acquisition Function</Label>
                      <Select
                        value={algorithmConfig.bayesian.acquisitionFunction}
                        onValueChange={(value) => updateAlgorithmConfig("bayesian", "acquisitionFunction", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select acquisition function" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="ei">Expected Improvement (EI)</SelectItem>
                          <SelectItem value="ucb">Upper Confidence Bound (UCB)</SelectItem>
                          <SelectItem value="pi">Probability of Improvement (PI)</SelectItem>
                          <SelectItem value="mes">Max-value Entropy Search (MES)</SelectItem>
                          <SelectItem value="thompson">Thompson Sampling</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        {algorithmConfig.bayesian.acquisitionFunction === "ei"
                          ? "Balances exploration and exploitation, good default choice"
                          : algorithmConfig.bayesian.acquisitionFunction === "ucb"
                            ? "More exploration-focused, good for unknown landscapes"
                            : algorithmConfig.bayesian.acquisitionFunction === "pi"
                              ? "More exploitation-focused, good when you have prior knowledge"
                              : algorithmConfig.bayesian.acquisitionFunction === "mes"
                                ? "Information-theoretic approach, good for complex problems"
                                : "Randomized approach, good for avoiding local optima"}
                      </p>
                    </div>

                    <div className="space-y-2 pt-4 border-t">
                      <div className="flex items-center space-x-2">
                        <Switch
                          id="multi-objective"
                          checked={algorithmConfig.bayesian.multiObjective}
                          onCheckedChange={(checked) => updateAlgorithmConfig("bayesian", "multiObjective", checked)}
                        />
                        <Label htmlFor="multi-objective" className="flex items-center">
                          Multi-Objective Optimization
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <HelpCircle className="ml-1 h-4 w-4 text-muted-foreground" />
                              </TooltipTrigger>
                              <TooltipContent side="right">
                                <p className="w-80">
                                  Enable this for problems with multiple competing objectives. The algorithm will find
                                  Pareto-optimal solutions.
                                </p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        </Label>
                      </div>

                      {algorithmConfig.bayesian.multiObjective && (
                        <div className="space-y-4 pl-6 mt-2">
                          <div className="space-y-2">
                            <Label>MOO Acquisition Function</Label>
                            <Select
                              value={algorithmConfig.bayesian.mooAcquisitionFunction}
                              onValueChange={(value) =>
                                updateAlgorithmConfig("bayesian", "mooAcquisitionFunction", value)
                              }
                            >
                              <SelectTrigger>
                                <SelectValue placeholder="Select MOO acquisition function" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="ehvi">Expected Hypervolume Improvement (EHVI)</SelectItem>
                                <SelectItem value="parego">ParEGO</SelectItem>
                                <SelectItem value="smsego">SMS-EGO</SelectItem>
                                <SelectItem value="mesmo">Max-value Entropy Search for MOO</SelectItem>
                              </SelectContent>
                            </Select>
                            <p className="text-xs text-muted-foreground">
                              {algorithmConfig.bayesian.mooAcquisitionFunction === "ehvi"
                                ? "EHVI maximizes the expected increase in the hypervolume of the Pareto front"
                                : algorithmConfig.bayesian.mooAcquisitionFunction === "parego"
                                  ? "ParEGO uses scalarization to convert multi-objective to single-objective problems"
                                  : algorithmConfig.bayesian.mooAcquisitionFunction === "smsego"
                                    ? "SMS-EGO uses the S-metric (hypervolume) for multi-objective optimization"
                                    : "MESMO is an information-theoretic approach for multi-objective optimization"}
                            </p>
                          </div>

                          <div className="flex items-center space-x-2">
                            <Switch
                              id="noisy-moo"
                              checked={algorithmConfig.bayesian.noisyMoo}
                              onCheckedChange={(checked) => updateAlgorithmConfig("bayesian", "noisyMoo", checked)}
                            />
                            <Label htmlFor="noisy-moo">Noisy MOO (for uncertain measurements)</Label>
                          </div>

                          <div className="space-y-2">
                            <Label>Reference Point Method</Label>
                            <Select
                              value={algorithmConfig.bayesian.referencePointMethod}
                              onValueChange={(value) =>
                                updateAlgorithmConfig("bayesian", "referencePointMethod", value)
                              }
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
                              Method for setting the reference point for hypervolume calculation
                            </p>
                          </div>
                        </div>
                      )}
                    </div>

                    <div className="space-y-2">
                      <Label>Kernel Function</Label>
                      <Select
                        value={algorithmConfig.bayesian.kernel}
                        onValueChange={(value) => updateAlgorithmConfig("bayesian", "kernel", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select kernel function" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="rbf">Radial Basis Function (RBF)</SelectItem>
                          <SelectItem value="matern">Matérn</SelectItem>
                          <SelectItem value="matern32">Matérn 3/2</SelectItem>
                          <SelectItem value="matern52">Matérn 5/2</SelectItem>
                          <SelectItem value="linear">Linear</SelectItem>
                          <SelectItem value="polynomial">Polynomial</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        {algorithmConfig.bayesian.kernel === "rbf"
                          ? "Smooth kernel, assumes infinitely differentiable function"
                          : algorithmConfig.bayesian.kernel === "matern"
                            ? "Less smooth than RBF, good default for real-world functions"
                            : algorithmConfig.bayesian.kernel === "matern32" ||
                                algorithmConfig.bayesian.kernel === "matern52"
                              ? "Specific smoothness Matérn kernel, more flexible"
                              : algorithmConfig.bayesian.kernel === "linear"
                                ? "Assumes linear relationship, very simple"
                                : "Higher-order relationships, good for specific problems"}
                      </p>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label>Exploration Weight: {algorithmConfig.bayesian.explorationWeight.toFixed(2)}</Label>
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
                        value={[algorithmConfig.bayesian.explorationWeight * 100]}
                        min={0}
                        max={100}
                        step={5}
                        onValueChange={(value) =>
                          updateAlgorithmConfig("bayesian", "explorationWeight", value[0] / 100)
                        }
                      />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Exploitation</span>
                        <span>Balanced</span>
                        <span>Exploration</span>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Initial Sampling Method</Label>
                      <Select
                        value={algorithmConfig.bayesian.initialSampling}
                        onValueChange={(value) => updateAlgorithmConfig("bayesian", "initialSampling", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select sampling method" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="lhs">Latin Hypercube Sampling</SelectItem>
                          <SelectItem value="sobol">Sobol Sequence</SelectItem>
                          <SelectItem value="halton">Halton Sequence</SelectItem>
                          <SelectItem value="random">Random Sampling</SelectItem>
                          <SelectItem value="grid">Grid Sampling</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">Method for initial design of experiments</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Initial Samples: {algorithmConfig.bayesian.initialSamples}</Label>
                      <Slider
                        value={[algorithmConfig.bayesian.initialSamples]}
                        min={5}
                        max={50}
                        step={5}
                        onValueChange={(value) => updateAlgorithmConfig("bayesian", "initialSamples", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">
                        Number of initial samples before starting Bayesian optimization
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label>Batch Size: {algorithmConfig.bayesian.batchSize}</Label>
                      <Slider
                        value={[algorithmConfig.bayesian.batchSize]}
                        min={1}
                        max={20}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("bayesian", "batchSize", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Number of points to evaluate in parallel</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Total Iterations: {algorithmConfig.bayesian.iterations}</Label>
                      <Slider
                        value={[algorithmConfig.bayesian.iterations]}
                        min={10}
                        max={100}
                        step={5}
                        onValueChange={(value) => updateAlgorithmConfig("bayesian", "iterations", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">
                        Total number of optimization iterations (after initial sampling)
                      </p>
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <h3 className="text-sm font-medium mb-2">Advanced Settings</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Noise Level: {algorithmConfig.bayesian.noiseLevel.toFixed(2)}</Label>
                      <Slider
                        value={[algorithmConfig.bayesian.noiseLevel * 100]}
                        min={0}
                        max={50}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("bayesian", "noiseLevel", value[0] / 100)}
                      />
                      <p className="text-xs text-muted-foreground">Assumed noise in observations (0 = deterministic)</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Length Scale: {algorithmConfig.bayesian.lengthScale.toFixed(2)}</Label>
                      <Slider
                        value={[algorithmConfig.bayesian.lengthScale * 10]}
                        min={1}
                        max={30}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("bayesian", "lengthScale", value[0] / 10)}
                      />
                      <p className="text-xs text-muted-foreground">
                        Kernel length scale (smaller = more wiggly function)
                      </p>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Switch
                        id="use-ard"
                        checked={algorithmConfig.bayesian.useARD}
                        onCheckedChange={(checked) => updateAlgorithmConfig("bayesian", "useARD", checked)}
                      />
                      <Label htmlFor="use-ard">Use ARD (Automatic Relevance Determination)</Label>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {selectedAlgorithm === "genetic" && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Population Size: {algorithmConfig.genetic.populationSize}</Label>
                      <Slider
                        value={[algorithmConfig.genetic.populationSize]}
                        min={10}
                        max={200}
                        step={10}
                        onValueChange={(value) => updateAlgorithmConfig("genetic", "populationSize", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Number of individuals in the population</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Generations: {algorithmConfig.genetic.generations}</Label>
                      <Slider
                        value={[algorithmConfig.genetic.generations]}
                        min={10}
                        max={500}
                        step={10}
                        onValueChange={(value) => updateAlgorithmConfig("genetic", "generations", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Number of evolutionary generations</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Crossover Rate: {algorithmConfig.genetic.crossoverRate.toFixed(2)}</Label>
                      <Slider
                        value={[algorithmConfig.genetic.crossoverRate * 100]}
                        min={0}
                        max={100}
                        step={5}
                        onValueChange={(value) => updateAlgorithmConfig("genetic", "crossoverRate", value[0] / 100)}
                      />
                      <p className="text-xs text-muted-foreground">Probability of crossover between parents</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Mutation Rate: {algorithmConfig.genetic.mutationRate.toFixed(2)}</Label>
                      <Slider
                        value={[algorithmConfig.genetic.mutationRate * 100]}
                        min={0}
                        max={50}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("genetic", "mutationRate", value[0] / 100)}
                      />
                      <p className="text-xs text-muted-foreground">Probability of mutation for each gene</p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Selection Method</Label>
                      <Select
                        value={algorithmConfig.genetic.selectionMethod}
                        onValueChange={(value) => updateAlgorithmConfig("genetic", "selectionMethod", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select selection method" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="tournament">Tournament Selection</SelectItem>
                          <SelectItem value="roulette">Roulette Wheel Selection</SelectItem>
                          <SelectItem value="rank">Rank Selection</SelectItem>
                          <SelectItem value="sus">Stochastic Universal Sampling</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">Method for selecting parents for reproduction</p>
                    </div>

                    {algorithmConfig.genetic.selectionMethod === "tournament" && (
                      <div className="space-y-2">
                        <Label>Tournament Size: {algorithmConfig.genetic.tournamentSize}</Label>
                        <Slider
                          value={[algorithmConfig.genetic.tournamentSize]}
                          min={2}
                          max={10}
                          step={1}
                          onValueChange={(value) => updateAlgorithmConfig("genetic", "tournamentSize", value[0])}
                        />
                        <p className="text-xs text-muted-foreground">Number of individuals in each tournament</p>
                      </div>
                    )}

                    <div className="space-y-2">
                      <Label>Crossover Method</Label>
                      <Select
                        value={algorithmConfig.genetic.crossoverMethod}
                        onValueChange={(value) => updateAlgorithmConfig("genetic", "crossoverMethod", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select crossover method" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="uniform">Uniform Crossover</SelectItem>
                          <SelectItem value="onepoint">One-Point Crossover</SelectItem>
                          <SelectItem value="twopoint">Two-Point Crossover</SelectItem>
                          <SelectItem value="blx">Blend Crossover (BLX-α)</SelectItem>
                          <SelectItem value="sbx">Simulated Binary Crossover (SBX)</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">Method for combining parent genes</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Mutation Method</Label>
                      <Select
                        value={algorithmConfig.genetic.mutationMethod}
                        onValueChange={(value) => updateAlgorithmConfig("genetic", "mutationMethod", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select mutation method" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="gaussian">Gaussian Mutation</SelectItem>
                          <SelectItem value="uniform">Uniform Mutation</SelectItem>
                          <SelectItem value="polynomial">Polynomial Mutation</SelectItem>
                          <SelectItem value="swap">Swap Mutation</SelectItem>
                          <SelectItem value="inversion">Inversion Mutation</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">Method for mutating genes</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Elitism Count: {algorithmConfig.genetic.elitismCount}</Label>
                      <Slider
                        value={[algorithmConfig.genetic.elitismCount]}
                        min={0}
                        max={10}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("genetic", "elitismCount", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Number of best individuals to preserve unchanged</p>
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <h3 className="text-sm font-medium mb-2">Advanced Settings</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Mutation Sigma: {algorithmConfig.genetic.mutationSigma.toFixed(2)}</Label>
                      <Slider
                        value={[algorithmConfig.genetic.mutationSigma * 100]}
                        min={1}
                        max={50}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("genetic", "mutationSigma", value[0] / 100)}
                      />
                      <p className="text-xs text-muted-foreground">
                        Standard deviation for Gaussian mutation (higher = more exploration)
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {selectedAlgorithm === "annealing" && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Initial Temperature: {algorithmConfig.annealing.initialTemperature}</Label>
                      <Slider
                        value={[algorithmConfig.annealing.initialTemperature]}
                        min={10}
                        max={1000}
                        step={10}
                        onValueChange={(value) => updateAlgorithmConfig("annealing", "initialTemperature", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Starting temperature (higher = more exploration)</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Cooling Rate: {algorithmConfig.annealing.coolingRate.toFixed(3)}</Label>
                      <Slider
                        value={[algorithmConfig.annealing.coolingRate * 1000]}
                        min={800}
                        max={999}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("annealing", "coolingRate", value[0] / 1000)}
                      />
                      <p className="text-xs text-muted-foreground">
                        Rate at which temperature decreases (closer to 1 = slower cooling)
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label>Iterations: {algorithmConfig.annealing.iterations}</Label>
                      <Slider
                        value={[algorithmConfig.annealing.iterations]}
                        min={100}
                        max={10000}
                        step={100}
                        onValueChange={(value) => updateAlgorithmConfig("annealing", "iterations", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Total number of iterations</p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Neighborhood Size: {algorithmConfig.annealing.neighborhoodSize.toFixed(2)}</Label>
                      <Slider
                        value={[algorithmConfig.annealing.neighborhoodSize * 100]}
                        min={1}
                        max={50}
                        step={1}
                        onValueChange={(value) =>
                          updateAlgorithmConfig("annealing", "neighborhoodSize", value[0] / 100)
                        }
                      />
                      <p className="text-xs text-muted-foreground">
                        Size of neighborhood for generating new solutions (relative to parameter range)
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label>Minimum Temperature: {algorithmConfig.annealing.minTemperature.toFixed(3)}</Label>
                      <Slider
                        value={[algorithmConfig.annealing.minTemperature * 1000]}
                        min={1}
                        max={100}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("annealing", "minTemperature", value[0] / 1000)}
                      />
                      <p className="text-xs text-muted-foreground">Stopping temperature</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Reheating Schedule</Label>
                      <Select
                        value={algorithmConfig.annealing.reheatingSchedule}
                        onValueChange={(value) => updateAlgorithmConfig("annealing", "reheatingSchedule", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select reheating schedule" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">No Reheating</SelectItem>
                          <SelectItem value="periodic">Periodic Reheating</SelectItem>
                          <SelectItem value="adaptive">Adaptive Reheating</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        Strategy for occasionally increasing temperature to escape local optima
                      </p>
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <h3 className="text-sm font-medium mb-2">Advanced Settings</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Equilibrium Steps: {algorithmConfig.annealing.equilibriumSteps}</Label>
                      <Slider
                        value={[algorithmConfig.annealing.equilibriumSteps]}
                        min={1}
                        max={50}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("annealing", "equilibriumSteps", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Number of iterations at each temperature level</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {selectedAlgorithm === "pso" && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Swarm Size: {algorithmConfig.pso.swarmSize}</Label>
                      <Slider
                        value={[algorithmConfig.pso.swarmSize]}
                        min={10}
                        max={100}
                        step={5}
                        onValueChange={(value) => updateAlgorithmConfig("pso", "swarmSize", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Number of particles in the swarm</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Iterations: {algorithmConfig.pso.iterations}</Label>
                      <Slider
                        value={[algorithmConfig.pso.iterations]}
                        min={10}
                        max={500}
                        step={10}
                        onValueChange={(value) => updateAlgorithmConfig("pso", "iterations", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Number of iterations</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Inertia Weight: {algorithmConfig.pso.inertiaWeight.toFixed(2)}</Label>
                      <Slider
                        value={[algorithmConfig.pso.inertiaWeight * 100]}
                        min={0}
                        max={100}
                        step={5}
                        onValueChange={(value) => updateAlgorithmConfig("pso", "inertiaWeight", value[0] / 100)}
                      />
                      <p className="text-xs text-muted-foreground">
                        Weight of previous velocity (higher = more exploration)
                      </p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Cognitive Coefficient: {algorithmConfig.pso.cognitiveCoefficient.toFixed(2)}</Label>
                      <Slider
                        value={[algorithmConfig.pso.cognitiveCoefficient * 100]}
                        min={0}
                        max={400}
                        step={10}
                        onValueChange={(value) => updateAlgorithmConfig("pso", "cognitiveCoefficient", value[0] / 100)}
                      />
                      <p className="text-xs text-muted-foreground">
                        Weight of particle's own best position (personal memory)
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label>Social Coefficient: {algorithmConfig.pso.socialCoefficient.toFixed(2)}</Label>
                      <Slider
                        value={[algorithmConfig.pso.socialCoefficient * 100]}
                        min={0}
                        max={400}
                        step={10}
                        onValueChange={(value) => updateAlgorithmConfig("pso", "socialCoefficient", value[0] / 100)}
                      />
                      <p className="text-xs text-muted-foreground">
                        Weight of swarm's best position (social influence)
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label>Velocity Clamp: {algorithmConfig.pso.velocityClamp.toFixed(1)}</Label>
                      <Slider
                        value={[algorithmConfig.pso.velocityClamp * 10]}
                        min={1}
                        max={50}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("pso", "velocityClamp", value[0] / 10)}
                      />
                      <p className="text-xs text-muted-foreground">Maximum velocity as a fraction of parameter range</p>
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <h3 className="text-sm font-medium mb-2">Advanced Settings</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Topology</Label>
                      <Select
                        value={algorithmConfig.pso.topology}
                        onValueChange={(value) => updateAlgorithmConfig("pso", "topology", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select swarm topology" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="global">Global Best</SelectItem>
                          <SelectItem value="ring">Ring Topology</SelectItem>
                          <SelectItem value="vonneumann">Von Neumann Topology</SelectItem>
                          <SelectItem value="random">Random Topology</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        Communication structure between particles in the swarm
                      </p>
                    </div>

                    {algorithmConfig.pso.topology !== "global" && (
                      <div className="space-y-2">
                        <Label>Neighborhood Size: {algorithmConfig.pso.neighborhoodSize}</Label>
                        <Slider
                          value={[algorithmConfig.pso.neighborhoodSize]}
                          min={2}
                          max={10}
                          step={1}
                          onValueChange={(value) => updateAlgorithmConfig("pso", "neighborhoodSize", value[0])}
                        />
                        <p className="text-xs text-muted-foreground">
                          Number of neighbors in local topologies (ring, von Neumann)
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {selectedAlgorithm === "nsga" && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Population Size: {algorithmConfig.nsga.populationSize}</Label>
                      <Slider
                        value={[algorithmConfig.nsga.populationSize]}
                        min={20}
                        max={500}
                        step={10}
                        onValueChange={(value) => updateAlgorithmConfig("nsga", "populationSize", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Number of individuals in the population</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Generations: {algorithmConfig.nsga.generations}</Label>
                      <Slider
                        value={[algorithmConfig.nsga.generations]}
                        min={10}
                        max={500}
                        step={10}
                        onValueChange={(value) => updateAlgorithmConfig("nsga", "generations", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Number of evolutionary generations</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Crossover Rate: {algorithmConfig.nsga.crossoverRate.toFixed(2)}</Label>
                      <Slider
                        value={[algorithmConfig.nsga.crossoverRate * 100]}
                        min={0}
                        max={100}
                        step={5}
                        onValueChange={(value) => updateAlgorithmConfig("nsga", "crossoverRate", value[0] / 100)}
                      />
                      <p className="text-xs text-muted-foreground">Probability of crossover between parents</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Mutation Rate: {algorithmConfig.nsga.mutationRate.toFixed(2)}</Label>
                      <Slider
                        value={[algorithmConfig.nsga.mutationRate * 100]}
                        min={0}
                        max={50}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("nsga", "mutationRate", value[0] / 100)}
                      />
                      <p className="text-xs text-muted-foreground">Probability of mutation for each gene</p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Tournament Size: {algorithmConfig.nsga.tournamentSize}</Label>
                      <Slider
                        value={[algorithmConfig.nsga.tournamentSize]}
                        min={2}
                        max={10}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("nsga", "tournamentSize", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Number of individuals in each tournament</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Crossover Distribution: {algorithmConfig.nsga.crossoverDistribution}</Label>
                      <Slider
                        value={[algorithmConfig.nsga.crossoverDistribution]}
                        min={1}
                        max={100}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("nsga", "crossoverDistribution", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">
                        Distribution index for SBX crossover (higher = closer to parents)
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label>Mutation Distribution: {algorithmConfig.nsga.mutationDistribution}</Label>
                      <Slider
                        value={[algorithmConfig.nsga.mutationDistribution]}
                        min={1}
                        max={100}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("nsga", "mutationDistribution", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">
                        Distribution index for polynomial mutation (higher = closer to parent)
                      </p>
                    </div>
                  </div>
                </div>

                <div className="pt-4 border-t">
                  <h3 className="text-sm font-medium mb-2">Advanced Settings</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="reference-points"
                        checked={algorithmConfig.nsga.referencePoints}
                        onCheckedChange={(checked) => updateAlgorithmConfig("nsga", "referencePoints", checked)}
                      />
                      <Label htmlFor="reference-points">Use Reference Points (NSGA-III)</Label>
                    </div>

                    {algorithmConfig.nsga.referencePoints && (
                      <div className="space-y-2">
                        <Label>Number of Reference Points: {algorithmConfig.nsga.numReferencePoints}</Label>
                        <Slider
                          value={[algorithmConfig.nsga.numReferencePoints]}
                          min={5}
                          max={50}
                          step={5}
                          onValueChange={(value) => updateAlgorithmConfig("nsga", "numReferencePoints", value[0])}
                        />
                        <p className="text-xs text-muted-foreground">
                          Number of reference points for NSGA-III (more = better diversity)
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {selectedAlgorithm === "grid" && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Points Per Dimension: {algorithmConfig.grid.pointsPerDimension}</Label>
                      <Slider
                        value={[algorithmConfig.grid.pointsPerDimension]}
                        min={2}
                        max={50}
                        step={1}
                        onValueChange={(value) => updateAlgorithmConfig("grid", "pointsPerDimension", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">
                        Number of grid points along each parameter dimension
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label>Maximum Total Points: {algorithmConfig.grid.maxPoints}</Label>
                      <Slider
                        value={[algorithmConfig.grid.maxPoints]}
                        min={100}
                        max={10000}
                        step={100}
                        onValueChange={(value) => updateAlgorithmConfig("grid", "maxPoints", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Maximum number of total grid points to evaluate</p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="use-log-scale"
                        checked={algorithmConfig.grid.useLogScale}
                        onCheckedChange={(checked) => updateAlgorithmConfig("grid", "useLogScale", checked)}
                      />
                      <Label htmlFor="use-log-scale">Use Logarithmic Scale</Label>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Switch
                        id="parallel-evaluation"
                        checked={algorithmConfig.grid.parallelEvaluation}
                        onCheckedChange={(checked) => updateAlgorithmConfig("grid", "parallelEvaluation", checked)}
                      />
                      <Label htmlFor="parallel-evaluation">Parallel Evaluation</Label>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {selectedAlgorithm === "random" && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Number of Samples: {algorithmConfig.random.numSamples}</Label>
                      <Slider
                        value={[algorithmConfig.random.numSamples]}
                        min={10}
                        max={1000}
                        step={10}
                        onValueChange={(value) => updateAlgorithmConfig("random", "numSamples", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Total number of random samples to evaluate</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Sampling Method</Label>
                      <Select
                        value={algorithmConfig.random.samplingMethod}
                        onValueChange={(value) => updateAlgorithmConfig("random", "samplingMethod", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select sampling method" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="uniform">Uniform Random</SelectItem>
                          <SelectItem value="lhs">Latin Hypercube Sampling</SelectItem>
                          <SelectItem value="sobol">Sobol Sequence</SelectItem>
                          <SelectItem value="halton">Halton Sequence</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">Method for generating random samples</p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Random Seed: {algorithmConfig.random.seed}</Label>
                      <Input
                        type="number"
                        value={algorithmConfig.random.seed}
                        onChange={(e) => updateAlgorithmConfig("random", "seed", Number.parseInt(e.target.value))}
                        min={0}
                      />
                      <p className="text-xs text-muted-foreground">
                        Seed for random number generator (for reproducibility)
                      </p>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Switch
                        id="parallel-evaluation-random"
                        checked={algorithmConfig.random.parallelEvaluation}
                        onCheckedChange={(checked) => updateAlgorithmConfig("random", "parallelEvaluation", checked)}
                      />
                      <Label htmlFor="parallel-evaluation-random">Parallel Evaluation</Label>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {selectedAlgorithm === "evolution" && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Population Size: {algorithmConfig.evolution.populationSize}</Label>
                      <Slider
                        value={[algorithmConfig.evolution.populationSize]}
                        min={5}
                        max={100}
                        step={5}
                        onValueChange={(value) => updateAlgorithmConfig("evolution", "populationSize", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Number of parents in each generation</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Offspring Size: {algorithmConfig.evolution.offspringSize}</Label>
                      <Slider
                        value={[algorithmConfig.evolution.offspringSize]}
                        min={10}
                        max={500}
                        step={10}
                        onValueChange={(value) => updateAlgorithmConfig("evolution", "offspringSize", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Number of offspring in each generation</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Generations: {algorithmConfig.evolution.generations}</Label>
                      <Slider
                        value={[algorithmConfig.evolution.generations]}
                        min={10}
                        max={500}
                        step={10}
                        onValueChange={(value) => updateAlgorithmConfig("evolution", "generations", value[0])}
                      />
                      <p className="text-xs text-muted-foreground">Number of evolutionary generations</p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="adaptive-sigma"
                        checked={algorithmConfig.evolution.adaptiveSigma}
                        onCheckedChange={(checked) => updateAlgorithmConfig("evolution", "adaptiveSigma", checked)}
                      />
                      <Label htmlFor="adaptive-sigma">Adaptive Step Size (CMA-ES)</Label>
                    </div>

                    {!algorithmConfig.evolution.adaptiveSigma && (
                      <div className="space-y-2">
                        <Label>Initial Sigma: {algorithmConfig.evolution.initialSigma.toFixed(2)}</Label>
                        <Slider
                          value={[algorithmConfig.evolution.initialSigma * 100]}
                          min={1}
                          max={50}
                          step={1}
                          onValueChange={(value) => updateAlgorithmConfig("evolution", "initialSigma", value[0] / 100)}
                        />
                        <p className="text-xs text-muted-foreground">
                          Initial mutation step size (relative to parameter range)
                        </p>
                      </div>
                    )}

                    <div className="space-y-2">
                      <Label>Recombination Method</Label>
                      <Select
                        value={algorithmConfig.evolution.recombinationMethod}
                        onValueChange={(value) => updateAlgorithmConfig("evolution", "recombinationMethod", value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select recombination method" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="intermediate">Intermediate Recombination</SelectItem>
                          <SelectItem value="weighted">Weighted Recombination</SelectItem>
                          <SelectItem value="dominant">Dominant Recombination</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">Method for combining parent solutions</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Selection Pressure: {algorithmConfig.evolution.selectionPressure.toFixed(2)}</Label>
                      <Slider
                        value={[algorithmConfig.evolution.selectionPressure * 100]}
                        min={10}
                        max={100}
                        step={5}
                        onValueChange={(value) =>
                          updateAlgorithmConfig("evolution", "selectionPressure", value[0] / 100)
                        }
                      />
                      <p className="text-xs text-muted-foreground">
                        Fraction of offspring selected as parents for next generation
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="compare">
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-medium">Algorithm Comparison</h3>
                <div className="flex items-center space-x-2">
                  <Label htmlFor="compare-mode" className="text-sm">
                    Detailed Comparison
                  </Label>
                  <Switch id="compare-mode" checked={compareMode} onCheckedChange={setCompareMode} />
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-4">Algorithm</th>
                      <th className="text-left py-2 px-4">Best For</th>
                      <th className="text-left py-2 px-4">Sample Efficiency</th>
                      <th className="text-left py-2 px-4">Parallelization</th>
                      {compareMode && (
                        <>
                          <th className="text-left py-2 px-4">Exploration/Exploitation</th>
                          <th className="text-left py-2 px-4">Implementation Complexity</th>
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(algorithmInfo).map(([key, info]) => (
                      <tr
                        key={key}
                        className={`border-b hover:bg-muted/50 ${selectedAlgorithm === key ? "bg-primary/5" : ""}`}
                        onClick={() => setSelectedAlgorithm(key)}
                      >
                        <td className="py-2 px-4 font-medium">{info.name}</td>
                        <td className="py-2 px-4">
                          <ul className="list-disc list-inside text-sm">
                            {info.bestFor.map((item, i) => (
                              <li key={i}>{item}</li>
                            ))}
                          </ul>
                        </td>
                        <td className="py-2 px-4">
                          <div className="flex items-center">
                            {key === "bayesian" ? (
                              <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300">
                                High
                              </Badge>
                            ) : key === "grid" || key === "random" ? (
                              <Badge className="bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300">Low</Badge>
                            ) : (
                              <Badge className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300">
                                Medium
                              </Badge>
                            )}
                          </div>
                        </td>
                        <td className="py-2 px-4">
                          <div className="flex items-center">
                            {info.parallelization === "Very High" || info.parallelization === "High" ? (
                              <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300">
                                {info.parallelization}
                              </Badge>
                            ) : info.parallelization === "Low" || info.parallelization === "Very Low" ? (
                              <Badge className="bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300">
                                {info.parallelization}
                              </Badge>
                            ) : (
                              <Badge className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300">
                                {info.parallelization}
                              </Badge>
                            )}
                          </div>
                        </td>
                        {compareMode && (
                          <>
                            <td className="py-2 px-4">{info.explorationExploitation}</td>
                            <td className="py-2 px-4">{info.complexity}</td>
                          </>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {compareMode && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                  <Card>
                    <CardHeader className="py-3">
                      <CardTitle className="text-sm">Pros & Cons: {algorithmInfo[selectedAlgorithm].name}</CardTitle>
                    </CardHeader>
                    <CardContent className="py-2">
                      <div className="space-y-4">
                        <div>
                          <h4 className="text-sm font-medium flex items-center">
                            <Check className="h-4 w-4 text-green-500 mr-1" />
                            Pros
                          </h4>
                          <ul className="list-disc list-inside text-sm pl-4 space-y-1 mt-1">
                            {algorithmInfo[selectedAlgorithm].pros.map((pro, i) => (
                              <li key={i}>{pro}</li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <h4 className="text-sm font-medium flex items-center">
                            <Info className="h-4 w-4 text-red-500 mr-1" />
                            Cons
                          </h4>
                          <ul className="list-disc list-inside text-sm pl-4 space-y-1 mt-1">
                            {algorithmInfo[selectedAlgorithm].cons.map((con, i) => (
                              <li key={i}>{con}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="py-3">
                      <CardTitle className="text-sm">Performance Comparison</CardTitle>
                    </CardHeader>
                    <CardContent className="py-2 h-[200px] flex items-center justify-center">
                      <div className="flex items-center">
                        <BarChart className="h-6 w-6 text-muted-foreground mr-2" />
                        <span className="text-sm text-muted-foreground">
                          Performance visualization will appear here
                        </span>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline" onClick={() => setActiveTab("select")}>
          Back
        </Button>
        <Button onClick={handleApply}>Apply Algorithm</Button>
      </CardFooter>
    </Card>
  )
}
