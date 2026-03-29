// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix-DAO TypeScript SDK
 *
 * Client library for interacting with the Mycelix-DAO governance system.
 *
 * @packageDocumentation
 */

export * from './types';
export * from './governance';
export * from './epistemic';
export * from './identity';
export * from './voting';
export * from './delegation';
export * from './conviction';
export * from './simulation';

// Re-export main client
export { DAOClient } from './client';

// Re-export simulation factory
export { createSimulator, GovernanceSimulator } from './simulation';
