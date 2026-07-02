// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Theme System
 *
 * Comprehensive theming with CSS variables, dark/light modes,
 * and custom theme creation support.
 */

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';

// ==================== Types ====================

export interface ThemeColors {
  // Primary palette
  primary: string;
  primaryHover: string;
  primaryActive: string;
  primaryMuted: string;

  // Secondary palette
  secondary: string;
  secondaryHover: string;

  // Accent colors
  accent: string;
  accentHover: string;

  // Semantic colors
  success: string;
  warning: string;
  error: string;
  info: string;

  // Background layers
  background: string;
  backgroundElevated: string;
  backgroundSunken: string;
  backgroundOverlay: string;

  // Surface colors
  surface: string;
  surfaceHover: string;
  surfaceActive: string;
  surfaceBorder: string;

  // Text colors
  text: string;
  textMuted: string;
  textSubtle: string;
  textInverse: string;

  // Audio-specific colors
  waveformPrimary: string;
  waveformSecondary: string;
  waveformProgress: string;
  meterLow: string;
  meterMid: string;
  meterHigh: string;
  meterClip: string;

  // Track colors (for mixer/timeline)
  trackColors: string[];
}

export interface ThemeTypography {
  fontFamily: string;
  fontFamilyMono: string;
  fontSizeXs: string;
  fontSizeSm: string;
  fontSizeMd: string;
  fontSizeLg: string;
  fontSizeXl: string;
  fontSizeXxl: string;
  fontWeightNormal: number;
  fontWeightMedium: number;
  fontWeightBold: number;
  lineHeightTight: number;
  lineHeightNormal: number;
  lineHeightRelaxed: number;
}

export interface ThemeSpacing {
  xs: string;
  sm: string;
  md: string;
  lg: string;
  xl: string;
  xxl: string;
}

export interface ThemeRadii {
  none: string;
  sm: string;
  md: string;
  lg: string;
  xl: string;
  full: string;
}

export interface ThemeShadows {
  sm: string;
  md: string;
  lg: string;
  xl: string;
  inner: string;
  glow: string;
}

export interface ThemeTransitions {
  fast: string;
  normal: string;
  slow: string;
  spring: string;
}

export interface Theme {
  name: string;
  mode: 'light' | 'dark';
  colors: ThemeColors;
  typography: ThemeTypography;
  spacing: ThemeSpacing;
  radii: ThemeRadii;
  shadows: ThemeShadows;
  transitions: ThemeTransitions;
}

// ==================== Default Themes ====================

const baseTypography: ThemeTypography = {
  fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  fontFamilyMono: "'JetBrains Mono', 'Fira Code', 'Monaco', monospace",
  fontSizeXs: '0.75rem',
  fontSizeSm: '0.875rem',
  fontSizeMd: '1rem',
  fontSizeLg: '1.125rem',
  fontSizeXl: '1.25rem',
  fontSizeXxl: '1.5rem',
  fontWeightNormal: 400,
  fontWeightMedium: 500,
  fontWeightBold: 600,
  lineHeightTight: 1.25,
  lineHeightNormal: 1.5,
  lineHeightRelaxed: 1.75,
};

const baseSpacing: ThemeSpacing = {
  xs: '0.25rem',
  sm: '0.5rem',
  md: '1rem',
  lg: '1.5rem',
  xl: '2rem',
  xxl: '3rem',
};

const baseRadii: ThemeRadii = {
  none: '0',
  sm: '0.25rem',
  md: '0.5rem',
  lg: '0.75rem',
  xl: '1rem',
  full: '9999px',
};

const baseTransitions: ThemeTransitions = {
  fast: '150ms ease',
  normal: '250ms ease',
  slow: '400ms ease',
  spring: '500ms cubic-bezier(0.34, 1.56, 0.64, 1)',
};

const trackColors = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
  '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
  '#BB8FCE', '#85C1E9', '#F8B500', '#00CED1',
  '#FF69B4', '#32CD32', '#FFD700', '#9370DB',
];

export const darkTheme: Theme = {
  name: 'Dark',
  mode: 'dark',
  colors: {
    primary: '#8B5CF6',
    primaryHover: '#A78BFA',
    primaryActive: '#7C3AED',
    primaryMuted: 'rgba(139, 92, 246, 0.2)',

    secondary: '#06B6D4',
    secondaryHover: '#22D3EE',

    accent: '#F59E0B',
    accentHover: '#FBBF24',

    success: '#10B981',
    warning: '#F59E0B',
    error: '#EF4444',
    info: '#3B82F6',

    background: '#0F0F0F',
    backgroundElevated: '#1A1A1A',
    backgroundSunken: '#080808',
    backgroundOverlay: 'rgba(0, 0, 0, 0.8)',

    surface: '#1F1F1F',
    surfaceHover: '#2A2A2A',
    surfaceActive: '#333333',
    surfaceBorder: '#333333',

    text: '#FAFAFA',
    textMuted: '#A1A1A1',
    textSubtle: '#6B6B6B',
    textInverse: '#0F0F0F',

    waveformPrimary: '#8B5CF6',
    waveformSecondary: '#4C1D95',
    waveformProgress: '#A78BFA',
    meterLow: '#10B981',
    meterMid: '#F59E0B',
    meterHigh: '#EF4444',
    meterClip: '#FF0000',

    trackColors,
  },
  typography: baseTypography,
  spacing: baseSpacing,
  radii: baseRadii,
  shadows: {
    sm: '0 1px 2px rgba(0, 0, 0, 0.5)',
    md: '0 4px 6px rgba(0, 0, 0, 0.5)',
    lg: '0 10px 15px rgba(0, 0, 0, 0.5)',
    xl: '0 20px 25px rgba(0, 0, 0, 0.5)',
    inner: 'inset 0 2px 4px rgba(0, 0, 0, 0.5)',
    glow: '0 0 20px rgba(139, 92, 246, 0.4)',
  },
  transitions: baseTransitions,
};

export const lightTheme: Theme = {
  name: 'Light',
  mode: 'light',
  colors: {
    primary: '#7C3AED',
    primaryHover: '#6D28D9',
    primaryActive: '#5B21B6',
    primaryMuted: 'rgba(124, 58, 237, 0.1)',

    secondary: '#0891B2',
    secondaryHover: '#0E7490',

    accent: '#D97706',
    accentHover: '#B45309',

    success: '#059669',
    warning: '#D97706',
    error: '#DC2626',
    info: '#2563EB',

    background: '#FAFAFA',
    backgroundElevated: '#FFFFFF',
    backgroundSunken: '#F0F0F0',
    backgroundOverlay: 'rgba(255, 255, 255, 0.9)',

    surface: '#FFFFFF',
    surfaceHover: '#F5F5F5',
    surfaceActive: '#EBEBEB',
    surfaceBorder: '#E5E5E5',

    text: '#171717',
    textMuted: '#525252',
    textSubtle: '#A3A3A3',
    textInverse: '#FAFAFA',

    waveformPrimary: '#7C3AED',
    waveformSecondary: '#C4B5FD',
    waveformProgress: '#5B21B6',
    meterLow: '#059669',
    meterMid: '#D97706',
    meterHigh: '#DC2626',
    meterClip: '#FF0000',

    trackColors,
  },
  typography: baseTypography,
  spacing: baseSpacing,
  radii: baseRadii,
  shadows: {
    sm: '0 1px 2px rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 15px rgba(0, 0, 0, 0.1)',
    xl: '0 20px 25px rgba(0, 0, 0, 0.15)',
    inner: 'inset 0 2px 4px rgba(0, 0, 0, 0.05)',
    glow: '0 0 20px rgba(124, 58, 237, 0.2)',
  },
  transitions: baseTransitions,
};

export const studioTheme: Theme = {
  ...darkTheme,
  name: 'Studio',
  colors: {
    ...darkTheme.colors,
    primary: '#00D9FF',
    primaryHover: '#33E1FF',
    primaryActive: '#00B8D9',
    primaryMuted: 'rgba(0, 217, 255, 0.2)',

    background: '#0A0A0A',
    backgroundElevated: '#141414',
    surface: '#1A1A1A',
    surfaceHover: '#242424',

    waveformPrimary: '#00D9FF',
    waveformSecondary: '#006680',
    waveformProgress: '#33E1FF',
  },
  shadows: {
    ...darkTheme.shadows,
    glow: '0 0 20px rgba(0, 217, 255, 0.4)',
  },
};

export const synthwaveTheme: Theme = {
  ...darkTheme,
  name: 'Synthwave',
  colors: {
    ...darkTheme.colors,
    primary: '#FF00FF',
    primaryHover: '#FF33FF',
    primaryActive: '#CC00CC',
    primaryMuted: 'rgba(255, 0, 255, 0.2)',

    secondary: '#00FFFF',
    secondaryHover: '#33FFFF',

    accent: '#FF6600',
    accentHover: '#FF8533',

    background: '#0D0221',
    backgroundElevated: '#150533',
    surface: '#1A0A3E',
    surfaceHover: '#2A1A5E',

    waveformPrimary: '#FF00FF',
    waveformSecondary: '#660066',
    waveformProgress: '#00FFFF',
  },
  shadows: {
    ...darkTheme.shadows,
    glow: '0 0 30px rgba(255, 0, 255, 0.5)',
  },
};

export const themes: Record<string, Theme> = {
  dark: darkTheme,
  light: lightTheme,
  studio: studioTheme,
  synthwave: synthwaveTheme,
};

// ==================== Context ====================

interface ThemeContextValue {
  theme: Theme;
  themeName: string;
  setTheme: (name: string) => void;
  toggleMode: () => void;
  createCustomTheme: (name: string, overrides: Partial<Theme>) => Theme;
  registerTheme: (name: string, theme: Theme) => void;
  availableThemes: string[];
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

// ==================== CSS Variables ====================

function themeToCSSVariables(theme: Theme): Record<string, string> {
  const vars: Record<string, string> = {};

  // Colors
  Object.entries(theme.colors).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      value.forEach((color, i) => {
        vars[`--color-${kebabCase(key)}-${i}`] = color;
      });
    } else {
      vars[`--color-${kebabCase(key)}`] = value;
    }
  });

  // Typography
  Object.entries(theme.typography).forEach(([key, value]) => {
    vars[`--${kebabCase(key)}`] = String(value);
  });

  // Spacing
  Object.entries(theme.spacing).forEach(([key, value]) => {
    vars[`--spacing-${key}`] = value;
  });

  // Radii
  Object.entries(theme.radii).forEach(([key, value]) => {
    vars[`--radius-${key}`] = value;
  });

  // Shadows
  Object.entries(theme.shadows).forEach(([key, value]) => {
    vars[`--shadow-${key}`] = value;
  });

  // Transitions
  Object.entries(theme.transitions).forEach(([key, value]) => {
    vars[`--transition-${key}`] = value;
  });

  return vars;
}

function kebabCase(str: string): string {
  return str.replace(/([a-z])([A-Z])/g, '$1-$2').toLowerCase();
}

// ==================== Provider ====================

interface ThemeProviderProps {
  children: ReactNode;
  defaultTheme?: string;
  storageKey?: string;
}

export function ThemeProvider({
  children,
  defaultTheme = 'dark',
  storageKey = 'mycelix-theme',
}: ThemeProviderProps) {
  const [customThemes, setCustomThemes] = useState<Record<string, Theme>>({});
  const [themeName, setThemeName] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem(storageKey) || defaultTheme;
    }
    return defaultTheme;
  });

  const allThemes = { ...themes, ...customThemes };
  const theme = allThemes[themeName] || darkTheme;

  // Apply CSS variables
  useEffect(() => {
    const root = document.documentElement;
    const vars = themeToCSSVariables(theme);

    Object.entries(vars).forEach(([key, value]) => {
      root.style.setProperty(key, value);
    });

    // Set color scheme for native elements
    root.style.colorScheme = theme.mode;

    // Add theme class
    root.dataset.theme = themeName;
  }, [theme, themeName]);

  // Persist theme choice
  useEffect(() => {
    localStorage.setItem(storageKey, themeName);
  }, [themeName, storageKey]);

  const setTheme = useCallback((name: string) => {
    if (allThemes[name]) {
      setThemeName(name);
    }
  }, [allThemes]);

  const toggleMode = useCallback(() => {
    if (theme.mode === 'dark') {
      setThemeName('light');
    } else {
      setThemeName('dark');
    }
  }, [theme.mode]);

  const createCustomTheme = useCallback((name: string, overrides: Partial<Theme>): Theme => {
    const baseTheme = overrides.mode === 'light' ? lightTheme : darkTheme;
    return {
      ...baseTheme,
      ...overrides,
      name,
      colors: { ...baseTheme.colors, ...overrides.colors },
      typography: { ...baseTheme.typography, ...overrides.typography },
      spacing: { ...baseTheme.spacing, ...overrides.spacing },
      radii: { ...baseTheme.radii, ...overrides.radii },
      shadows: { ...baseTheme.shadows, ...overrides.shadows },
      transitions: { ...baseTheme.transitions, ...overrides.transitions },
    } as Theme;
  }, []);

  const registerTheme = useCallback((name: string, newTheme: Theme) => {
    setCustomThemes(prev => ({ ...prev, [name]: newTheme }));
  }, []);

  const value: ThemeContextValue = {
    theme,
    themeName,
    setTheme,
    toggleMode,
    createCustomTheme,
    registerTheme,
    availableThemes: Object.keys(allThemes),
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

// ==================== Hook ====================

export function useTheme(): ThemeContextValue {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

// ==================== Styled Utilities ====================

export function cssVar(name: string): string {
  return `var(--${name})`;
}

export function color(name: keyof ThemeColors): string {
  return `var(--color-${kebabCase(name)})`;
}

export function spacing(size: keyof ThemeSpacing): string {
  return `var(--spacing-${size})`;
}

export function radius(size: keyof ThemeRadii): string {
  return `var(--radius-${size})`;
}

export function shadow(size: keyof ThemeShadows): string {
  return `var(--shadow-${size})`;
}

export function transition(speed: keyof ThemeTransitions): string {
  return `var(--transition-${speed})`;
}

// ==================== Theme Switcher Component ====================

interface ThemeSwitcherProps {
  className?: string;
}

export function ThemeSwitcher({ className }: ThemeSwitcherProps) {
  const { themeName, setTheme, availableThemes } = useTheme();

  return (
    <select
      value={themeName}
      onChange={(e) => setTheme(e.target.value)}
      className={className}
      style={{
        background: 'var(--color-surface)',
        color: 'var(--color-text)',
        border: '1px solid var(--color-surface-border)',
        borderRadius: 'var(--radius-md)',
        padding: 'var(--spacing-sm) var(--spacing-md)',
        fontSize: 'var(--font-size-sm)',
        cursor: 'pointer',
      }}
    >
      {availableThemes.map(name => (
        <option key={name} value={name}>
          {themes[name]?.name || name}
        </option>
      ))}
    </select>
  );
}
