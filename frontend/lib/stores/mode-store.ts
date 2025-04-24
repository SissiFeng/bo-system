import { create } from "zustand"
import { persist } from "zustand/middleware"

interface ModeState {
  isSimpleMode: boolean
  toggleMode: () => void
}

export const useModeStore = create<ModeState>()(
  persist(
    (set) => ({
      isSimpleMode: true,
      toggleMode: () => set((state) => ({ isSimpleMode: !state.isSimpleMode })),
    }),
    {
      name: "mode-storage",
    },
  ),
)
