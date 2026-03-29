// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail - Vue Integration
 *
 * Complete Vue 3 support with composables for reactive state management.
 */

// Composables
export * from './composables';

// Re-export client for convenience
export { getMycelixClient, createMycelixClient } from '../bootstrap';
