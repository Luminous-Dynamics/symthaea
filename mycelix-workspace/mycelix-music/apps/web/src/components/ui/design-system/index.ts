// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Design System
 *
 * Comprehensive UI primitives:
 * - Tokens (colors, spacing, typography)
 * - Base components
 * - Animation utilities
 * - Responsive helpers
 */

// ==================== Design Tokens ====================

export const colors = {
  // Brand
  primary: {
    50: '#f0f5ff',
    100: '#e0eaff',
    200: '#c7d7fe',
    300: '#a4bcfd',
    400: '#8098f9',
    500: '#6366f1',
    600: '#5145e5',
    700: '#4338ca',
    800: '#3730a3',
    900: '#312e81',
    950: '#1e1b4b',
  },
  // Secondary - Teal for music/audio
  secondary: {
    50: '#f0fdfa',
    100: '#ccfbf1',
    200: '#99f6e4',
    300: '#5eead4',
    400: '#2dd4bf',
    500: '#14b8a6',
    600: '#0d9488',
    700: '#0f766e',
    800: '#115e59',
    900: '#134e4a',
    950: '#042f2e',
  },
  // Accent - Amber for highlights
  accent: {
    50: '#fffbeb',
    100: '#fef3c7',
    200: '#fde68a',
    300: '#fcd34d',
    400: '#fbbf24',
    500: '#f59e0b',
    600: '#d97706',
    700: '#b45309',
    800: '#92400e',
    900: '#78350f',
    950: '#451a03',
  },
  // Semantic
  success: {
    light: '#86efac',
    DEFAULT: '#22c55e',
    dark: '#15803d',
  },
  warning: {
    light: '#fde047',
    DEFAULT: '#eab308',
    dark: '#a16207',
  },
  error: {
    light: '#fca5a5',
    DEFAULT: '#ef4444',
    dark: '#b91c1c',
  },
  info: {
    light: '#93c5fd',
    DEFAULT: '#3b82f6',
    dark: '#1d4ed8',
  },
  // Neutral
  gray: {
    50: '#fafafa',
    100: '#f4f4f5',
    200: '#e4e4e7',
    300: '#d4d4d8',
    400: '#a1a1aa',
    500: '#71717a',
    600: '#52525b',
    700: '#3f3f46',
    800: '#27272a',
    900: '#18181b',
    950: '#09090b',
  },
} as const;

export const spacing = {
  px: '1px',
  0: '0',
  0.5: '0.125rem',
  1: '0.25rem',
  1.5: '0.375rem',
  2: '0.5rem',
  2.5: '0.625rem',
  3: '0.75rem',
  3.5: '0.875rem',
  4: '1rem',
  5: '1.25rem',
  6: '1.5rem',
  7: '1.75rem',
  8: '2rem',
  9: '2.25rem',
  10: '2.5rem',
  11: '2.75rem',
  12: '3rem',
  14: '3.5rem',
  16: '4rem',
  20: '5rem',
  24: '6rem',
  28: '7rem',
  32: '8rem',
  36: '9rem',
  40: '10rem',
  44: '11rem',
  48: '12rem',
  52: '13rem',
  56: '14rem',
  60: '15rem',
  64: '16rem',
  72: '18rem',
  80: '20rem',
  96: '24rem',
} as const;

export const typography = {
  fontFamily: {
    sans: ['Inter', 'system-ui', 'sans-serif'],
    mono: ['JetBrains Mono', 'monospace'],
    display: ['Cabinet Grotesk', 'Inter', 'sans-serif'],
  },
  fontSize: {
    xs: ['0.75rem', { lineHeight: '1rem' }],
    sm: ['0.875rem', { lineHeight: '1.25rem' }],
    base: ['1rem', { lineHeight: '1.5rem' }],
    lg: ['1.125rem', { lineHeight: '1.75rem' }],
    xl: ['1.25rem', { lineHeight: '1.75rem' }],
    '2xl': ['1.5rem', { lineHeight: '2rem' }],
    '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
    '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
    '5xl': ['3rem', { lineHeight: '1' }],
    '6xl': ['3.75rem', { lineHeight: '1' }],
    '7xl': ['4.5rem', { lineHeight: '1' }],
    '8xl': ['6rem', { lineHeight: '1' }],
    '9xl': ['8rem', { lineHeight: '1' }],
  },
  fontWeight: {
    thin: '100',
    extralight: '200',
    light: '300',
    normal: '400',
    medium: '500',
    semibold: '600',
    bold: '700',
    extrabold: '800',
    black: '900',
  },
} as const;

export const shadows = {
  sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
  DEFAULT: '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
  md: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
  lg: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
  xl: '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
  '2xl': '0 25px 50px -12px rgb(0 0 0 / 0.25)',
  inner: 'inset 0 2px 4px 0 rgb(0 0 0 / 0.05)',
  glow: '0 0 20px rgb(99 102 241 / 0.3)',
  'glow-lg': '0 0 40px rgb(99 102 241 / 0.4)',
} as const;

export const animation = {
  duration: {
    instant: '0ms',
    fast: '150ms',
    normal: '300ms',
    slow: '500ms',
    slower: '700ms',
    slowest: '1000ms',
  },
  easing: {
    linear: 'linear',
    ease: 'ease',
    easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
    easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
    spring: 'cubic-bezier(0.175, 0.885, 0.32, 1.275)',
  },
  keyframes: {
    fadeIn: {
      from: { opacity: '0' },
      to: { opacity: '1' },
    },
    fadeOut: {
      from: { opacity: '1' },
      to: { opacity: '0' },
    },
    slideInUp: {
      from: { transform: 'translateY(10px)', opacity: '0' },
      to: { transform: 'translateY(0)', opacity: '1' },
    },
    slideInDown: {
      from: { transform: 'translateY(-10px)', opacity: '0' },
      to: { transform: 'translateY(0)', opacity: '1' },
    },
    slideInLeft: {
      from: { transform: 'translateX(-10px)', opacity: '0' },
      to: { transform: 'translateX(0)', opacity: '1' },
    },
    slideInRight: {
      from: { transform: 'translateX(10px)', opacity: '0' },
      to: { transform: 'translateX(0)', opacity: '1' },
    },
    scaleIn: {
      from: { transform: 'scale(0.95)', opacity: '0' },
      to: { transform: 'scale(1)', opacity: '1' },
    },
    pulse: {
      '0%, 100%': { opacity: '1' },
      '50%': { opacity: '0.5' },
    },
    spin: {
      from: { transform: 'rotate(0deg)' },
      to: { transform: 'rotate(360deg)' },
    },
    bounce: {
      '0%, 100%': { transform: 'translateY(-5%)', animationTimingFunction: 'cubic-bezier(0.8, 0, 1, 1)' },
      '50%': { transform: 'translateY(0)', animationTimingFunction: 'cubic-bezier(0, 0, 0.2, 1)' },
    },
    shimmer: {
      '0%': { backgroundPosition: '-200% 0' },
      '100%': { backgroundPosition: '200% 0' },
    },
    waveform: {
      '0%, 100%': { transform: 'scaleY(0.5)' },
      '50%': { transform: 'scaleY(1)' },
    },
  },
} as const;

export const borderRadius = {
  none: '0',
  sm: '0.125rem',
  DEFAULT: '0.25rem',
  md: '0.375rem',
  lg: '0.5rem',
  xl: '0.75rem',
  '2xl': '1rem',
  '3xl': '1.5rem',
  full: '9999px',
} as const;

export const breakpoints = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
} as const;

export const zIndex = {
  hide: -1,
  auto: 'auto',
  base: 0,
  docked: 10,
  dropdown: 1000,
  sticky: 1100,
  banner: 1200,
  overlay: 1300,
  modal: 1400,
  popover: 1500,
  skipLink: 1600,
  toast: 1700,
  tooltip: 1800,
} as const;

// ==================== CSS Custom Properties ====================

export function generateCSSVariables(): string {
  const lines: string[] = [':root {'];

  // Colors
  for (const [colorName, shades] of Object.entries(colors)) {
    if (typeof shades === 'object' && 'DEFAULT' in shades) {
      lines.push(`  --color-${colorName}: ${shades.DEFAULT};`);
      lines.push(`  --color-${colorName}-light: ${shades.light};`);
      lines.push(`  --color-${colorName}-dark: ${shades.dark};`);
    } else if (typeof shades === 'object') {
      for (const [shade, value] of Object.entries(shades)) {
        lines.push(`  --color-${colorName}-${shade}: ${value};`);
      }
    }
  }

  // Spacing
  for (const [key, value] of Object.entries(spacing)) {
    lines.push(`  --spacing-${key.replace('.', '_')}: ${value};`);
  }

  // Shadows
  for (const [key, value] of Object.entries(shadows)) {
    const name = key === 'DEFAULT' ? 'base' : key;
    lines.push(`  --shadow-${name}: ${value};`);
  }

  // Animation
  for (const [key, value] of Object.entries(animation.duration)) {
    lines.push(`  --duration-${key}: ${value};`);
  }

  for (const [key, value] of Object.entries(animation.easing)) {
    lines.push(`  --easing-${key}: ${value};`);
  }

  // Border radius
  for (const [key, value] of Object.entries(borderRadius)) {
    const name = key === 'DEFAULT' ? 'base' : key;
    lines.push(`  --radius-${name}: ${value};`);
  }

  // Z-index
  for (const [key, value] of Object.entries(zIndex)) {
    lines.push(`  --z-${key}: ${value};`);
  }

  lines.push('}');
  return lines.join('\n');
}

// ==================== Utility Classes ====================

export const utilityClasses = {
  // Flex utilities
  'flex-center': 'flex items-center justify-center',
  'flex-between': 'flex items-center justify-between',
  'flex-start': 'flex items-center justify-start',
  'flex-end': 'flex items-center justify-end',
  'flex-col-center': 'flex flex-col items-center justify-center',

  // Text utilities
  'text-balance': 'text-wrap: balance',
  'text-gradient': 'bg-clip-text text-transparent bg-gradient-to-r',
  'truncate-2': 'overflow-hidden display: -webkit-box -webkit-line-clamp: 2 -webkit-box-orient: vertical',
  'truncate-3': 'overflow-hidden display: -webkit-box -webkit-line-clamp: 3 -webkit-box-orient: vertical',

  // Interactive
  'focus-ring': 'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
  'focus-visible-ring': 'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2',
  'hover-lift': 'transition-transform hover:-translate-y-0.5',
  'hover-scale': 'transition-transform hover:scale-105',
  'press-scale': 'active:scale-95',

  // Layout
  'container-narrow': 'max-w-2xl mx-auto px-4',
  'container-wide': 'max-w-7xl mx-auto px-4 sm:px-6 lg:px-8',
  'full-bleed': 'w-screen relative left-1/2 right-1/2 -mx-[50vw]',

  // Glass effect
  'glass': 'backdrop-blur-lg bg-white/10 border border-white/20',
  'glass-dark': 'backdrop-blur-lg bg-black/20 border border-white/10',
} as const;

// ==================== Component Variants ====================

export const buttonVariants = {
  base: 'inline-flex items-center justify-center font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
  variants: {
    variant: {
      primary: 'bg-primary-600 text-white hover:bg-primary-700 focus-visible:ring-primary-500',
      secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200 focus-visible:ring-gray-500 dark:bg-gray-800 dark:text-gray-100 dark:hover:bg-gray-700',
      outline: 'border-2 border-primary-600 text-primary-600 hover:bg-primary-50 focus-visible:ring-primary-500 dark:hover:bg-primary-950',
      ghost: 'text-gray-600 hover:bg-gray-100 hover:text-gray-900 focus-visible:ring-gray-500 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-100',
      danger: 'bg-error-DEFAULT text-white hover:bg-error-dark focus-visible:ring-error-DEFAULT',
      success: 'bg-success-DEFAULT text-white hover:bg-success-dark focus-visible:ring-success-DEFAULT',
    },
    size: {
      xs: 'h-7 px-2.5 text-xs rounded',
      sm: 'h-8 px-3 text-sm rounded-md',
      md: 'h-10 px-4 text-sm rounded-lg',
      lg: 'h-12 px-6 text-base rounded-lg',
      xl: 'h-14 px-8 text-lg rounded-xl',
      icon: 'h-10 w-10 rounded-lg',
      'icon-sm': 'h-8 w-8 rounded-md',
      'icon-lg': 'h-12 w-12 rounded-xl',
    },
  },
  defaultVariants: {
    variant: 'primary',
    size: 'md',
  },
} as const;

export const inputVariants = {
  base: 'flex w-full rounded-lg border bg-transparent px-3 py-2 text-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-gray-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
  variants: {
    variant: {
      default: 'border-gray-300 focus-visible:ring-primary-500 dark:border-gray-700',
      error: 'border-error-DEFAULT focus-visible:ring-error-DEFAULT',
      success: 'border-success-DEFAULT focus-visible:ring-success-DEFAULT',
    },
    size: {
      sm: 'h-8 text-xs',
      md: 'h-10',
      lg: 'h-12 text-base',
    },
  },
  defaultVariants: {
    variant: 'default',
    size: 'md',
  },
} as const;

export const badgeVariants = {
  base: 'inline-flex items-center rounded-full font-medium transition-colors',
  variants: {
    variant: {
      default: 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300',
      secondary: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300',
      success: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
      warning: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300',
      error: 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300',
      outline: 'border border-current',
    },
    size: {
      sm: 'px-2 py-0.5 text-xs',
      md: 'px-2.5 py-0.5 text-sm',
      lg: 'px-3 py-1 text-base',
    },
  },
  defaultVariants: {
    variant: 'default',
    size: 'md',
  },
} as const;

// ==================== Exports ====================

export default {
  colors,
  spacing,
  typography,
  shadows,
  animation,
  borderRadius,
  breakpoints,
  zIndex,
  generateCSSVariables,
  utilityClasses,
  buttonVariants,
  inputVariants,
  badgeVariants,
};
