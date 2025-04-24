import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Download, Eye, HelpCircle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface Recommendation {
  id: number
  parameters?: Record<string, number | string>
  objectives?: Record<string, number>
  uncertainty?: number
  reason?: string
  [key: string]: any
}

interface RecommendationsTableProps {
  recommendations: Recommendation[]
}

export function RecommendationsTable({ recommendations }: RecommendationsTableProps) {
  if (!recommendations.length) return <div>No recommendations available</div>

  // Get all keys except 'id' and 'reason'
  const keys = Object.keys(recommendations[0]).filter((key) => key !== "id" && key !== "reason")

  // Separate input parameters (x) from outputs (y)
  const parameterKeys = recommendations[0].parameters ? Object.keys(recommendations[0].parameters) : []
  const objectiveKeys = recommendations[0].objectives ? Object.keys(recommendations[0].objectives) : []
  const otherKeys = keys.filter((key) => key !== "parameters" && key !== "objectives" && key !== "uncertainty")

  return (
    <div>
      <TooltipProvider>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Rank</TableHead>
              {parameterKeys.map((key) => (
                <TableHead key={key}>Parameter {key}</TableHead>
              ))}
              {objectiveKeys.map((key) => (
                <TableHead key={key}>Objective {key}</TableHead>
              ))}
              <TableHead>Uncertainty</TableHead>
              <TableHead>Explanation</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {recommendations.map((rec, index) => (
              <TableRow key={rec.id}>
                <TableCell className="font-medium">{index + 1}</TableCell>
                {parameterKeys.map((key) => (
                  <TableCell key={key}>{rec.parameters?.[key]}</TableCell>
                ))}
                {objectiveKeys.map((key) => (
                  <TableCell key={key}>{rec.objectives?.[key]?.toFixed(2)}</TableCell>
                ))}
                <TableCell>
                  <Badge variant={rec.uncertainty && rec.uncertainty > 0.2 ? "secondary" : "outline"}>
                    σ = {rec.uncertainty?.toFixed(2)}
                  </Badge>
                </TableCell>
                <TableCell>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="ghost" size="icon">
                        <HelpCircle className="h-4 w-4" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent side="left" className="max-w-xs">
                      <p>{rec.reason || "No explanation available"}</p>
                    </TooltipContent>
                  </Tooltip>
                </TableCell>
                <TableCell className="text-right">
                  <div className="flex justify-end gap-2">
                    <Button variant="ghost" size="icon">
                      <Eye className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="icon">
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TooltipProvider>
    </div>
  )
}
