"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Switch } from "@/components/ui/switch"
import { Bell, Mail, MessageSquare } from "lucide-react"
import { toast } from "@/components/ui/use-toast"

interface NotificationSettingsProps {
  onSave: (settings: NotificationSettings) => void
}

export interface NotificationSettings {
  enabled: boolean
  method: "email" | "slack" | "huggingface"
  destination: string
  frequency: "immediate" | "daily" | "weekly"
  includeDetails: boolean
}

export function NotificationSettings({ onSave }: NotificationSettingsProps) {
  const [settings, setSettings] = useState<NotificationSettings>({
    enabled: false,
    method: "email",
    destination: "",
    frequency: "immediate",
    includeDetails: true,
  })

  const handleSave = () => {
    if (settings.enabled && !settings.destination) {
      toast({
        title: "Destination required",
        description: "Please enter an email address, Slack webhook, or Hugging Face space URL.",
        variant: "destructive",
      })
      return
    }

    onSave(settings)
    toast({
      title: "Notification settings saved",
      description: settings.enabled
        ? `You'll receive ${settings.frequency} notifications via ${settings.method}.`
        : "Notifications are disabled.",
    })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Notification Settings</CardTitle>
        <CardDescription>Configure how you want to receive optimization updates</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="notifications">Enable Notifications</Label>
            <p className="text-sm text-muted-foreground">
              Receive updates about optimization progress and recommendations
            </p>
          </div>
          <Switch
            id="notifications"
            checked={settings.enabled}
            onCheckedChange={(checked) => setSettings({ ...settings, enabled: checked })}
          />
        </div>

        {settings.enabled && (
          <>
            <div className="space-y-4">
              <Label>Notification Method</Label>
              <RadioGroup
                value={settings.method}
                onValueChange={(value) => setSettings({ ...settings, method: value as any })}
                className="flex flex-col space-y-2"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="email" id="email" />
                  <Label htmlFor="email" className="flex items-center">
                    <Mail className="mr-2 h-4 w-4" />
                    Email
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="slack" id="slack" />
                  <Label htmlFor="slack" className="flex items-center">
                    <MessageSquare className="mr-2 h-4 w-4" />
                    Slack
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="huggingface" id="huggingface" />
                  <Label htmlFor="huggingface" className="flex items-center">
                    <Bell className="mr-2 h-4 w-4" />
                    Hugging Face Space
                  </Label>
                </div>
              </RadioGroup>
            </div>

            <div className="space-y-2">
              <Label htmlFor="destination">
                {settings.method === "email"
                  ? "Email Address"
                  : settings.method === "slack"
                    ? "Slack Webhook URL"
                    : "Hugging Face Space URL"}
              </Label>
              <Input
                id="destination"
                placeholder={
                  settings.method === "email"
                    ? "you@example.com"
                    : settings.method === "slack"
                      ? "https://hooks.slack.com/..."
                      : "https://huggingface.co/spaces/..."
                }
                value={settings.destination}
                onChange={(e) => setSettings({ ...settings, destination: e.target.value })}
              />
            </div>

            <div className="space-y-4">
              <Label>Notification Frequency</Label>
              <RadioGroup
                value={settings.frequency}
                onValueChange={(value) => setSettings({ ...settings, frequency: value as any })}
                className="flex flex-col space-y-2"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="immediate" id="immediate" />
                  <Label htmlFor="immediate">Immediate (after each optimization round)</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="daily" id="daily" />
                  <Label htmlFor="daily">Daily summary</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="weekly" id="weekly" />
                  <Label htmlFor="weekly">Weekly summary</Label>
                </div>
              </RadioGroup>
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="includeDetails"
                checked={settings.includeDetails}
                onCheckedChange={(checked) => setSettings({ ...settings, includeDetails: checked })}
              />
              <Label htmlFor="includeDetails">Include detailed parameter values and predictions</Label>
            </div>
          </>
        )}
      </CardContent>
      <CardFooter>
        <Button onClick={handleSave}>Save Notification Settings</Button>
      </CardFooter>
    </Card>
  )
}
