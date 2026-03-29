// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Safe Belief Propagation Module
 *
 * Manages how belief updates flow through the knowledge graph with
 * circuit breakers to prevent cascade failures.
 *
 * @packageDocumentation
 * @module propagation
 */

// Types
export {
  type CircuitBreakerConfig,
  type PropagationState,
  type BlockReason,
  type PropagationRequest,
  type PropagationResult,
  type AffectedClaim,
  type PropagationTrace,
  type PropagationStep,
  type CircuitBreakerCheck,
  type QuarantinedPropagation,
  type SuspiciousPattern,
  type QuarantineDecision,
  type ClaimRateLimitState,
  type GlobalRateLimitState,
  type HumanReviewRequest,
  type HumanReviewDecision,
  type PropagationEvent,
  type PropagationConfig,
  DEFAULT_PROPAGATION_CONFIG,
  DEFAULT_CIRCUIT_BREAKER_CONFIG,
} from './types.js';

// Engine
export {
  SafeBeliefPropagator,
  type PropagationStats,
  getSafeBeliefPropagator,
  resetSafeBeliefPropagator,
} from './engine.js';
