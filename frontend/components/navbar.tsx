import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ModeToggle } from "@/components/mode-toggle"
import { Home, Settings, LineChart, Pencil, Wand2, Boxes, Play, Cpu } from "lucide-react"

export function Navbar() {
  return (
    <header className="border-b">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2 font-semibold">
            <span className="text-primary font-bold text-xl">BO-Optimizer</span>
          </Link>
          <nav className="hidden md:flex gap-6">
            <Link href="/" className="flex items-center text-sm font-medium">
              <Home className="mr-2 h-4 w-4" />
              Home
            </Link>
            <Link href="/config/parameters" className="flex items-center text-sm font-medium">
              <Settings className="mr-2 h-4 w-4" />
              Form Config
            </Link>
            <Link href="/playground" className="flex items-center text-sm font-medium">
              <Boxes className="mr-2 h-4 w-4" />
              Playground
            </Link>
            <Link href="/algorithms" className="flex items-center text-sm font-medium">
              <Cpu className="mr-2 h-4 w-4" />
              Algorithms
            </Link>
            <Link href="/canvas" className="flex items-center text-sm font-medium">
              <Pencil className="mr-2 h-4 w-4" />
              Canvas
            </Link>
            <Link href="/wizard" className="flex items-center text-sm font-medium">
              <Wand2 className="mr-2 h-4 w-4" />
              Wizard
            </Link>
            <Link href="/animation" className="flex items-center text-sm font-medium">
              <Play className="mr-2 h-4 w-4" />
              Animation
            </Link>
            <Link href="/results" className="flex items-center text-sm font-medium">
              <LineChart className="mr-2 h-4 w-4" />
              Results
            </Link>
            <Link href="/model-analysis" className="flex items-center text-sm font-medium">
              <LineChart className="mr-2 h-4 w-4" />
              Model Analysis
            </Link>
          </nav>
        </div>
        <div className="flex items-center gap-4">
          <ModeToggle />
          <Button variant="outline" size="sm">
            Export
          </Button>
          <Button size="sm">Run Optimization</Button>
        </div>
      </div>
    </header>
  )
}
