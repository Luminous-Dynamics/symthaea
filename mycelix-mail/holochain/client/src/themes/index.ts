// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Theme System Module
 *
 * Comprehensive theming with light/dark modes and CSS variables
 */

export {
  ThemeProvider,
  useTheme,
  themed,
  cssVar,
  type Theme,
  type ColorPalette,
  type ThemeContextValue,
} from './ThemeProvider';

export default ThemeProvider;
