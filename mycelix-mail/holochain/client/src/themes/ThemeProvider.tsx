// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Theme Provider - Comprehensive Theme System
 *
 * Features:
 * - Light/Dark/System modes
 * - Custom color schemes
 * - CSS variables integration
 * - Theme marketplace support
 * - Per-component theming
 */

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  ReactNode,
} from 'react';

export interface ColorPalette {
  50: string;
  100: string;
  200: string;
  300: string;
  400: string;
  500: string;
  600: string;
  700: string;
  800: string;
  900: string;
}

export interface Theme {
  id: string;
  name: string;
  author?: string;
  description?: string;
  mode: 'light' | 'dark';
  colors: {
    primary: ColorPalette;
    secondary: ColorPalette;
    accent: ColorPalette;
    neutral: ColorPalette;
    success: ColorPalette;
    warning: ColorPalette;
    error: ColorPalette;
    info: ColorPalette;
  };
  semantic: {
    background: string;
    foreground: string;
    card: string;
    cardForeground: string;
    popover: string;
    popoverForeground: string;
    muted: string;
    mutedForeground: string;
    border: string;
    input: string;
    ring: string;
  };
  typography: {
    fontFamily: string;
    fontFamilyMono: string;
    fontSize: {
      xs: string;
      sm: string;
      base: string;
      lg: string;
      xl: string;
      '2xl': string;
      '3xl': string;
    };
  };
  spacing: {
    unit: number;
  };
  borderRadius: {
    none: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
    full: string;
  };
  shadows: {
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
}

// Default themes
const lightTheme: Theme = {
  id: 'light',
  name: 'Light',
  mode: 'light',
  colors: {
    primary: {
      50: '#eff6ff', 100: '#dbeafe', 200: '#bfdbfe', 300: '#93c5fd',
      400: '#60a5fa', 500: '#3b82f6', 600: '#2563eb', 700: '#1d4ed8',
      800: '#1e40af', 900: '#1e3a8a',
    },
    secondary: {
      50: '#f8fafc', 100: '#f1f5f9', 200: '#e2e8f0', 300: '#cbd5e1',
      400: '#94a3b8', 500: '#64748b', 600: '#475569', 700: '#334155',
      800: '#1e293b', 900: '#0f172a',
    },
    accent: {
      50: '#fdf4ff', 100: '#fae8ff', 200: '#f5d0fe', 300: '#f0abfc',
      400: '#e879f9', 500: '#d946ef', 600: '#c026d3', 700: '#a21caf',
      800: '#86198f', 900: '#701a75',
    },
    neutral: {
      50: '#fafafa', 100: '#f4f4f5', 200: '#e4e4e7', 300: '#d4d4d8',
      400: '#a1a1aa', 500: '#71717a', 600: '#52525b', 700: '#3f3f46',
      800: '#27272a', 900: '#18181b',
    },
    success: {
      50: '#f0fdf4', 100: '#dcfce7', 200: '#bbf7d0', 300: '#86efac',
      400: '#4ade80', 500: '#22c55e', 600: '#16a34a', 700: '#15803d',
      800: '#166534', 900: '#14532d',
    },
    warning: {
      50: '#fffbeb', 100: '#fef3c7', 200: '#fde68a', 300: '#fcd34d',
      400: '#fbbf24', 500: '#f59e0b', 600: '#d97706', 700: '#b45309',
      800: '#92400e', 900: '#78350f',
    },
    error: {
      50: '#fef2f2', 100: '#fee2e2', 200: '#fecaca', 300: '#fca5a5',
      400: '#f87171', 500: '#ef4444', 600: '#dc2626', 700: '#b91c1c',
      800: '#991b1b', 900: '#7f1d1d',
    },
    info: {
      50: '#ecfeff', 100: '#cffafe', 200: '#a5f3fc', 300: '#67e8f9',
      400: '#22d3ee', 500: '#06b6d4', 600: '#0891b2', 700: '#0e7490',
      800: '#155e75', 900: '#164e63',
    },
  },
  semantic: {
    background: '#ffffff',
    foreground: '#0f172a',
    card: '#ffffff',
    cardForeground: '#0f172a',
    popover: '#ffffff',
    popoverForeground: '#0f172a',
    muted: '#f1f5f9',
    mutedForeground: '#64748b',
    border: '#e2e8f0',
    input: '#e2e8f0',
    ring: '#3b82f6',
  },
  typography: {
    fontFamily: 'Inter, system-ui, sans-serif',
    fontFamilyMono: 'JetBrains Mono, monospace',
    fontSize: {
      xs: '0.75rem', sm: '0.875rem', base: '1rem', lg: '1.125rem',
      xl: '1.25rem', '2xl': '1.5rem', '3xl': '1.875rem',
    },
  },
  spacing: { unit: 4 },
  borderRadius: {
    none: '0', sm: '0.125rem', md: '0.375rem', lg: '0.5rem', xl: '0.75rem', full: '9999px',
  },
  shadows: {
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
    md: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
    xl: '0 20px 25px -5px rgb(0 0 0 / 0.1)',
  },
};

const darkTheme: Theme = {
  ...lightTheme,
  id: 'dark',
  name: 'Dark',
  mode: 'dark',
  semantic: {
    background: '#0f172a',
    foreground: '#f8fafc',
    card: '#1e293b',
    cardForeground: '#f8fafc',
    popover: '#1e293b',
    popoverForeground: '#f8fafc',
    muted: '#334155',
    mutedForeground: '#94a3b8',
    border: '#334155',
    input: '#334155',
    ring: '#60a5fa',
  },
};

// Mycelix branded themes
const mycelixTheme: Theme = {
  ...lightTheme,
  id: 'mycelix',
  name: 'Mycelix',
  author: 'Mycelix Team',
  description: 'The official Mycelix Mail theme',
  colors: {
    ...lightTheme.colors,
    primary: {
      50: '#f0f9ff', 100: '#e0f2fe', 200: '#bae6fd', 300: '#7dd3fc',
      400: '#38bdf8', 500: '#0ea5e9', 600: '#0284c7', 700: '#0369a1',
      800: '#075985', 900: '#0c4a6e',
    },
    accent: {
      50: '#fdf2f8', 100: '#fce7f3', 200: '#fbcfe8', 300: '#f9a8d4',
      400: '#f472b6', 500: '#ec4899', 600: '#db2777', 700: '#be185d',
      800: '#9d174d', 900: '#831843',
    },
  },
};

export interface ThemeContextValue {
  theme: Theme;
  mode: 'light' | 'dark' | 'system';
  setMode: (mode: 'light' | 'dark' | 'system') => void;
  setTheme: (theme: Theme) => void;
  availableThemes: Theme[];
  addTheme: (theme: Theme) => void;
  removeTheme: (themeId: string) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function useTheme(): ThemeContextValue {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
}

interface ThemeProviderProps {
  children: ReactNode;
  defaultTheme?: Theme;
  defaultMode?: 'light' | 'dark' | 'system';
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({
  children,
  defaultTheme,
  defaultMode = 'system',
}) => {
  const [themes, setThemes] = useState<Theme[]>([lightTheme, darkTheme, mycelixTheme]);
  const [mode, setModeState] = useState<'light' | 'dark' | 'system'>(() => {
    const saved = localStorage.getItem('mycelix-theme-mode');
    return (saved as 'light' | 'dark' | 'system') || defaultMode;
  });
  const [customTheme, setCustomTheme] = useState<Theme | null>(() => {
    const saved = localStorage.getItem('mycelix-custom-theme');
    return saved ? JSON.parse(saved) : defaultTheme || null;
  });

  const getSystemMode = useCallback((): 'light' | 'dark' => {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }, []);

  const resolvedMode = mode === 'system' ? getSystemMode() : mode;

  const theme = customTheme || (resolvedMode === 'dark' ? darkTheme : lightTheme);

  // Apply theme to CSS variables
  useEffect(() => {
    const root = document.documentElement;

    // Set color mode
    root.classList.remove('light', 'dark');
    root.classList.add(resolvedMode);

    // Set CSS variables
    Object.entries(theme.colors).forEach(([name, palette]) => {
      Object.entries(palette).forEach(([shade, color]) => {
        root.style.setProperty(`--color-${name}-${shade}`, color);
      });
    });

    Object.entries(theme.semantic).forEach(([name, value]) => {
      root.style.setProperty(`--${name}`, value);
    });

    Object.entries(theme.typography.fontSize).forEach(([name, value]) => {
      root.style.setProperty(`--font-size-${name}`, value);
    });

    root.style.setProperty('--font-family', theme.typography.fontFamily);
    root.style.setProperty('--font-family-mono', theme.typography.fontFamilyMono);

    Object.entries(theme.borderRadius).forEach(([name, value]) => {
      root.style.setProperty(`--radius-${name}`, value);
    });

    Object.entries(theme.shadows).forEach(([name, value]) => {
      root.style.setProperty(`--shadow-${name}`, value);
    });
  }, [theme, resolvedMode]);

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    const handleChange = () => {
      if (mode === 'system') {
        // Force re-render
        setModeState((m) => m);
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [mode]);

  const setMode = useCallback((newMode: 'light' | 'dark' | 'system') => {
    setModeState(newMode);
    localStorage.setItem('mycelix-theme-mode', newMode);
  }, []);

  const setTheme = useCallback((newTheme: Theme) => {
    setCustomTheme(newTheme);
    localStorage.setItem('mycelix-custom-theme', JSON.stringify(newTheme));
  }, []);

  const addTheme = useCallback((newTheme: Theme) => {
    setThemes((prev) => {
      if (prev.some((t) => t.id === newTheme.id)) {
        return prev.map((t) => (t.id === newTheme.id ? newTheme : t));
      }
      return [...prev, newTheme];
    });
  }, []);

  const removeTheme = useCallback((themeId: string) => {
    setThemes((prev) => prev.filter((t) => t.id !== themeId));
    if (customTheme?.id === themeId) {
      setCustomTheme(null);
      localStorage.removeItem('mycelix-custom-theme');
    }
  }, [customTheme]);

  const value: ThemeContextValue = {
    theme,
    mode,
    setMode,
    setTheme,
    availableThemes: themes,
    addTheme,
    removeTheme,
  };

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
};

// Theme-aware styled component helper
export function themed<T extends keyof Theme['colors']>(
  colorName: T,
  shade: keyof ColorPalette = 500
): string {
  return `var(--color-${colorName}-${shade})`;
}

// CSS variable helper
export function cssVar(name: string): string {
  return `var(--${name})`;
}

export default ThemeProvider;
