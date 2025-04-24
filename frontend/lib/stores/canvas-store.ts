import { create } from "zustand"
import { persist } from "zustand/middleware"

interface CanvasState {
  nodes: any[]
  edges: any[]
  updateCanvasState: (state: { nodes: any[]; edges: any[] }) => void
}

export const useCanvasStore = create<CanvasState>()(
  persist(
    (set) => ({
      nodes: [],
      edges: [],
      updateCanvasState: (state) => set(state),
    }),
    {
      name: "canvas-storage",
    },
  ),
)
