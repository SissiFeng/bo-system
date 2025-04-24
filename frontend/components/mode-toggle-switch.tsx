"use client"
import { Switch } from "@/components/ui/switch"
import { useToast } from "@/components/ui/use-toast"
import { useModeStore } from "@/lib/stores/mode-store"

export function ModeToggleSwitch() {
  const { isSimpleMode, toggleMode } = useModeStore()
  const { toast } = useToast()

  const handleToggle = () => {
    toggleMode()
    toast({
      title: isSimpleMode ? "Switched to Professional Mode" : "Switched to Simple Mode",
      description: isSimpleMode
        ? "You now have full control over all configuration options."
        : "We'll help guide you through the optimization process with smart defaults.",
      duration: 3000,
    })
  }

  return (
    <div className="flex items-center space-x-4 rounded-lg border p-4">
      <div className="flex-1 space-y-1">
        <div className="flex items-center">
          <p className="text-sm font-medium leading-none">{isSimpleMode ? "Simple Mode" : "Professional Mode"}</p>
        </div>
        <p className="text-sm text-muted-foreground">
          {isSimpleMode
            ? "We'll help guide you with smart defaults and suggestions"
            : "Full control over all configuration options"}
        </p>
      </div>
      <Switch checked={!isSimpleMode} onCheckedChange={handleToggle} />
    </div>
  )
}
