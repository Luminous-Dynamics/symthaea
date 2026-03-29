// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Framework Integrations for Mycelix SDK
 *
 * This module provides framework-specific bindings for popular JavaScript frameworks.
 * Each framework integration is available as a separate export path to enable
 * tree-shaking and avoid bundling unnecessary dependencies.
 *
 * @module frameworks
 * @packageDocumentation
 *
 * ## Available Frameworks
 *
 * ### React
 * ```ts
 * import { MycelixProvider, useMycelix, useIdentity, useProposals } from '@mycelix/sdk/react';
 * ```
 *
 * ### Svelte
 * ```ts
 * import { mycelix, identity, proposals, initMycelix } from '@mycelix/sdk/svelte';
 * ```
 *
 * ### Vue
 * ```ts
 * import { MycelixPlugin, useMycelix, useIdentity, useGovernance } from '@mycelix/sdk/vue';
 * ```
 *
 * ## Installation
 *
 * The framework integrations are included in the main @mycelix/sdk package.
 * You do NOT need to install separate packages.
 *
 * ```bash
 * npm install @mycelix/sdk
 * ```
 *
 * However, you need the respective framework as a peer dependency:
 *
 * - React: `react >= 18.0.0`
 * - Svelte: `svelte >= 4.0.0`
 * - Vue: `vue >= 3.3.0`
 *
 * ## Framework-Specific Documentation
 *
 * See the individual module documentation for detailed usage:
 *
 * - {@link module:frameworks/react | React Integration}
 * - {@link module:frameworks/svelte | Svelte Integration}
 * - {@link module:frameworks/vue | Vue Integration}
 */

// =============================================================================
// Framework Re-exports
// =============================================================================

// Note: We re-export as namespaces to allow importing all framework
// integrations from a single entry point if needed.

export * as react from './react/index.js';
export * as svelte from './svelte/index.js';
export * as vue from './vue/index.js';
