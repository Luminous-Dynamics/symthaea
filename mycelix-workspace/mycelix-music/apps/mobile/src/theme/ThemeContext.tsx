// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Theme Context for Mycelix Mobile
 *
 * Provides consistent theming across the app.
 */

import { createContext, useContext, ReactNode } from 'react';

export interface ThemeColors {
  primary: string;
  secondary: string;
  accent: string;
  background: string;
  surface: string;
  text: string;
  textMuted: string;
  border: string;
  error: string;
  success: string;
  warning: string;
}

export interface ThemeSpacing {
  xs: number;
  sm: number;
  md: number;
  lg: number;
  xl: number;
  xxl: number;
}

export interface Theme {
  colors: ThemeColors;
  spacing: ThemeSpacing;
  borderRadius: {
    sm: number;
    md: number;
    lg: number;
    xl: number;
    full: number;
  };
}

const darkTheme: Theme = {
  colors: {
    primary: '#7DD3FC',      // Sky blue - mycelium glow
    secondary: '#A78BFA',    // Purple - connection
    accent: '#34D399',       // Green - growth
    background: '#0a0a0a',   // Deep black
    surface: '#1a1a1a',      // Card surface
    text: '#FAFAFA',         // White text
    textMuted: '#A1A1AA',    // Gray text
    border: '#27272A',       // Subtle borders
    error: '#EF4444',        // Red
    success: '#22C55E',      // Green
    warning: '#F59E0B',      // Amber
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
  },
  borderRadius: {
    sm: 4,
    md: 8,
    lg: 12,
    xl: 16,
    full: 9999,
  },
};

interface ThemeContextValue extends Theme {
  isDark: boolean;
}

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  // For now, only dark theme (can add light theme toggle later)
  const value: ThemeContextValue = {
    ...darkTheme,
    isDark: true,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextValue {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
