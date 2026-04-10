// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix SDK Core
 *
 * Core client architecture providing unified access to all Civilizational OS hApps.
 *
 * This module provides:
 * - **Mycelix**: Unified client factory for connecting to Holochain
 * - **ZomeClient**: Base class for building domain-specific clients
 * - **Retry utilities**: Exponential backoff retry patterns
 * - **Error handling**: Standardized error types and codes
 *
 * @module @mycelix/sdk/core
 *
 * @example
 * ```typescript
 * import { Mycelix, SdkError, SdkErrorCode, withRetry } from '@mycelix/sdk/core';
 *
 * // Connect to Holochain
 * const mycelix = await Mycelix.connect({
 *   conductorUrl: 'ws://localhost:8888',
 *   appId: 'mycelix-civilizational-os',
 * });
 *
 * // Access domain clients
 * const identity = mycelix.identity;
 * const governance = mycelix.governance;
 * const finance = mycelix.finance;
 *
 * // Use retry utilities
 * const result = await withRetry(
 *   async () => await someUnreliableOperation(),
 *   { maxRetries: 3, baseDelay: 1000 }
 * );
 *
 * // Handle errors
 * try {
 *   await mycelix.identity.did.createDid();
 * } catch (error) {
 *   if (error instanceof SdkError) {
 *     console.error(`[${error.code}] ${error.message}`);
 *   }
 * }
 * ```
 */

// =============================================================================
// Unified Client
// =============================================================================

export {
  Mycelix,
  type MycelixConfig,
  type MycelixClientOptions,
  type ConnectionState,
  type SignalHandler,
  type ConnectionStateListener,
} from './client.js';

// =============================================================================
// Error Handling
// =============================================================================

export {
  // Error codes
  SdkErrorCode,
  isRetryableCode,

  // Error types
  SdkError,
  NetworkError,
  ValidationError,
  NotFoundError,
  TimeoutError,
  UnauthorizedError,
  ZomeCallError,

  // Utilities
  wrapError,

  // Types
  type SdkErrorDetails,
} from './errors.js';

// =============================================================================
// Retry Utilities
// =============================================================================

export {
  // Main functions
  withRetry,
  tryWithRetry,
  withTimeout,
  withRetryAndTimeout,
  createRetryWrapper,

  // Policies
  RetryPolicies,
  DEFAULT_RETRY_CONFIG,

  // Types
  type RetryConfig,
  type RetryResult,
} from './retry.js';

// =============================================================================
// Consciousness Gate Middleware
// =============================================================================

export {
  // Error types
  ConsciousnessGateError,

  // Pre-flight check
  canPerform,
  combinedScore,
  tierFromScore,

  // Retry middleware
  withGateRetry,

  // Governance audit query
  queryGovernanceAudit,

  // Types
  type ConsciousnessTier,
  type ConsciousnessProfile,
  type ConsciousnessCredential,
  type ConsciousnessGateRejection,
  type GateEligibility,
  type GateAuditEntry,
  type GovernanceAuditFilter,
  type GovernanceAuditResult,
  type ZomeCallable,
} from './consciousness-gate.js';

// =============================================================================
// 8D Sovereign Profile Gate (replaces consciousness gate)
// =============================================================================

export {
  CivicGateError,
  combinedScore as sovereignCombinedScore,
  tierFromScore as civicTierFromScore,
  meetsRequirement,
  decayScore,
  daysUntilThreshold,
  legacyCombinedScore,
  CIVIC_TIERS,
  TIER_THRESHOLDS,
  TIER_VOTE_WEIGHT_BP,
  WEIGHTS_GOVERNANCE,
  DIMENSION_LABELS,
  LAMBDA_MIN,
  LAMBDA_MAX,
  type SovereignProfile,
  type SovereignDimension,
  type CivicTier,
  type DimensionWeights,
  type CivicRequirement,
  type SovereignCredential,
  type CivicEligibility,
  type CivicGateRejection,
} from './sovereign-gate.js';

// =============================================================================
// Zome Client Base
// =============================================================================

export {
  ZomeClient,
  createZomeClient,

  // Types
  type ZomeClientConfig,
} from './zome-client.js';

// =============================================================================
// Re-exports from domain clients
// =============================================================================

// These are re-exported for convenience when using the core module
export type { MycelixIdentityClient } from '../integrations/identity/client.js';
export type { MycelixGovernanceClient } from '../integrations/governance/client.js';
