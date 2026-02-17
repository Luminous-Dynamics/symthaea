import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Theme = 'light' | 'dark' | 'system';

interface ThemeState {
  theme: Theme;
  isDark: boolean;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

const getSystemTheme = (): boolean => {
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
};

const applyTheme = (theme: Theme) => {
  const isDark = theme === 'dark' || (theme === 'system' && getSystemTheme());

  if (isDark) {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }

  return isDark;
};

export const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      theme: 'system',
      isDark: false,

      setTheme: (theme: Theme) => {
        const isDark = applyTheme(theme);
        set({ theme, isDark });
      },

      toggleTheme: () => {
        const currentTheme = get().theme;
        const newTheme: Theme = currentTheme === 'dark' ? 'light' : 'dark';
        const isDark = applyTheme(newTheme);
        set({ theme: newTheme, isDark });
      },
    }),
    {
      name: 'mycelix-theme',
      onRehydrateStorage: () => (state) => {
        if (state) {
          // Apply theme on initial load
          applyTheme(state.theme);
        }
      },
    }
  )
);

// Listen for system theme changes
if (typeof window !== 'undefined') {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    const state = useThemeStore.getState();
    if (state.theme === 'system') {
      state.setTheme('system');
    }
  });
}
