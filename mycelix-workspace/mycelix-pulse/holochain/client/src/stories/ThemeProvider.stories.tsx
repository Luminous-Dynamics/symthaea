// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * ThemeProvider Component Stories
 */

import React, { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react';
import { ThemeProvider, useTheme, Theme } from '../themes/ThemeProvider';

const meta: Meta<typeof ThemeProvider> = {
  title: 'Theme/ThemeProvider',
  component: ThemeProvider,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'Comprehensive theme system with light/dark modes, custom color schemes, and CSS variables.',
      },
    },
  },
  tags: ['autodocs'],
};

export default meta;
type Story = StoryObj<typeof ThemeProvider>;

// Demo component to show theme values
const ThemeDemo: React.FC = () => {
  const { theme, mode, setMode, availableThemes, setTheme } = useTheme();

  return (
    <div
      style={{
        minHeight: '100vh',
        padding: 24,
        backgroundColor: theme.semantic.background,
        color: theme.semantic.foreground,
        fontFamily: theme.typography.fontFamily,
      }}
    >
      <h1 style={{ fontSize: theme.typography.fontSize['2xl'], marginBottom: 16 }}>
        Theme System Demo
      </h1>

      {/* Mode selector */}
      <div style={{ marginBottom: 24 }}>
        <label style={{ display: 'block', marginBottom: 8, fontWeight: 500 }}>
          Color Mode
        </label>
        <div style={{ display: 'flex', gap: 8 }}>
          {(['light', 'dark', 'system'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              style={{
                padding: '8px 16px',
                borderRadius: theme.borderRadius.md,
                border: mode === m ? `2px solid ${theme.colors.primary[500]}` : `1px solid ${theme.semantic.border}`,
                backgroundColor: mode === m ? theme.colors.primary[50] : theme.semantic.card,
                color: theme.semantic.cardForeground,
                cursor: 'pointer',
                textTransform: 'capitalize',
              }}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Theme selector */}
      <div style={{ marginBottom: 24 }}>
        <label style={{ display: 'block', marginBottom: 8, fontWeight: 500 }}>
          Theme
        </label>
        <div style={{ display: 'flex', gap: 8 }}>
          {availableThemes.map((t) => (
            <button
              key={t.id}
              onClick={() => setTheme(t)}
              style={{
                padding: '8px 16px',
                borderRadius: theme.borderRadius.md,
                border: theme.id === t.id ? `2px solid ${theme.colors.primary[500]}` : `1px solid ${theme.semantic.border}`,
                backgroundColor: theme.id === t.id ? theme.colors.primary[50] : theme.semantic.card,
                color: theme.semantic.cardForeground,
                cursor: 'pointer',
              }}
            >
              {t.name}
            </button>
          ))}
        </div>
      </div>

      {/* Color palette display */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: theme.typography.fontSize.xl, marginBottom: 12 }}>
          Color Palette
        </h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {Object.entries(theme.colors).map(([name, palette]) => (
            <div key={name}>
              <div style={{ marginBottom: 4, fontWeight: 500, textTransform: 'capitalize' }}>
                {name}
              </div>
              <div style={{ display: 'flex', gap: 2 }}>
                {Object.entries(palette).map(([shade, color]) => (
                  <div
                    key={shade}
                    style={{
                      width: 48,
                      height: 48,
                      backgroundColor: color,
                      borderRadius: theme.borderRadius.sm,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 10,
                      color: parseInt(shade) >= 500 ? '#fff' : '#000',
                    }}
                  >
                    {shade}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Typography */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: theme.typography.fontSize.xl, marginBottom: 12 }}>
          Typography
        </h2>
        <div
          style={{
            padding: 16,
            backgroundColor: theme.semantic.card,
            borderRadius: theme.borderRadius.lg,
            border: `1px solid ${theme.semantic.border}`,
          }}
        >
          {Object.entries(theme.typography.fontSize).map(([size, value]) => (
            <div key={size} style={{ fontSize: value, marginBottom: 8 }}>
              {size}: The quick brown fox jumps over the lazy dog
            </div>
          ))}
        </div>
      </div>

      {/* Components preview */}
      <div>
        <h2 style={{ fontSize: theme.typography.fontSize.xl, marginBottom: 12 }}>
          Sample Components
        </h2>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <button
            style={{
              padding: '10px 20px',
              backgroundColor: theme.colors.primary[500],
              color: '#fff',
              border: 'none',
              borderRadius: theme.borderRadius.md,
              cursor: 'pointer',
              boxShadow: theme.shadows.md,
            }}
          >
            Primary Button
          </button>
          <button
            style={{
              padding: '10px 20px',
              backgroundColor: theme.semantic.muted,
              color: theme.semantic.mutedForeground,
              border: 'none',
              borderRadius: theme.borderRadius.md,
              cursor: 'pointer',
            }}
          >
            Secondary Button
          </button>
          <button
            style={{
              padding: '10px 20px',
              backgroundColor: 'transparent',
              color: theme.colors.primary[500],
              border: `1px solid ${theme.colors.primary[500]}`,
              borderRadius: theme.borderRadius.md,
              cursor: 'pointer',
            }}
          >
            Outline Button
          </button>
        </div>

        <div style={{ marginTop: 16, display: 'flex', gap: 16 }}>
          <input
            type="text"
            placeholder="Input field"
            style={{
              padding: '10px 12px',
              backgroundColor: theme.semantic.background,
              border: `1px solid ${theme.semantic.input}`,
              borderRadius: theme.borderRadius.md,
              color: theme.semantic.foreground,
              fontSize: theme.typography.fontSize.base,
            }}
          />
          <div
            style={{
              padding: 16,
              backgroundColor: theme.semantic.card,
              border: `1px solid ${theme.semantic.border}`,
              borderRadius: theme.borderRadius.lg,
              boxShadow: theme.shadows.sm,
            }}
          >
            Card component
          </div>
        </div>
      </div>
    </div>
  );
};

export const Default: Story = {
  render: () => (
    <ThemeProvider>
      <ThemeDemo />
    </ThemeProvider>
  ),
};

export const DarkMode: Story = {
  render: () => (
    <ThemeProvider defaultMode="dark">
      <ThemeDemo />
    </ThemeProvider>
  ),
};

export const SystemMode: Story = {
  render: () => (
    <ThemeProvider defaultMode="system">
      <ThemeDemo />
    </ThemeProvider>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Automatically follows system color scheme preference.',
      },
    },
  },
};

// Color scheme comparison
export const ColorSchemeComparison: Story = {
  render: () => (
    <div style={{ display: 'flex' }}>
      <div style={{ flex: 1 }}>
        <ThemeProvider defaultMode="light">
          <div style={{ padding: 24, backgroundColor: '#fff', minHeight: '50vh' }}>
            <h2>Light Mode</h2>
            <ThemeDemo />
          </div>
        </ThemeProvider>
      </div>
      <div style={{ flex: 1 }}>
        <ThemeProvider defaultMode="dark">
          <div style={{ padding: 24, backgroundColor: '#0f172a', minHeight: '50vh' }}>
            <h2 style={{ color: '#fff' }}>Dark Mode</h2>
            <ThemeDemo />
          </div>
        </ThemeProvider>
      </div>
    </div>
  ),
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        story: 'Side-by-side comparison of light and dark modes.',
      },
    },
  },
};
