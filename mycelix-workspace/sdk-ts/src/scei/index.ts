// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Self-Correcting Epistemic Infrastructure (SCEI)
 *
 * Core infrastructure for maintaining epistemic integrity across the knowledge graph.
 *
 * Modules:
 * - **Validation**: Runtime input validation with clear error messages
 * - **Event Bus**: Cross-module communication for feedback loops
 * - **Metrics**: Prometheus-style observability
 * - **Persistence**: Pluggable storage backends
 *
 * @packageDocumentation
 * @module scei
 */

// Validation utilities
export {
  // Types
  type ValidationResult,
  type ValidationError,
  type ValidationWarning,
  // Validators
  validateConfidence,
  validateDegradationFactor,
  validatePositiveInteger,
  validateTimestamp,
  validateNonEmptyString,
  validateId,
  validateEnum,
  validateArray,
  // Helpers
  combineValidations,
  assertValid,
  SCEIValidationError,
  // Numerical stability
  safeLog,
  clampConfidence,
  safeDivide,
  safeWeightedAverage,
} from './validation.js';

// Event bus
export {
  // Types
  type SCEIEventType,
  type SCEIEvent,
  type SCEIModule,
  type SCEIEventListener,
  type SCEIEventFilter,
  type Subscription,
  type SCEIEventBusMetrics,
  type SCEIEventBusOptions,
  type SCEIEventPayloadMap,
  // Payload types
  type CalibrationPredictionRecordedPayload,
  type CalibrationPredictionResolvedPayload,
  type MetabolismClaimBornPayload,
  type MetabolismClaimDiedPayload,
  type MetabolismHealthUpdatedPayload,
  type PropagationCompletedPayload,
  type DiscoveryGapDetectedPayload,
  type DiscoveryGapResolvedPayload,
  // Classes
  SCEIEventBus,
  SCEIEventBuilder,
  ListenerTimeoutError,
  // Singleton
  getSCEIEventBus,
  resetSCEIEventBus,
  // Helpers
  createSCEIEvent,
  createCorrelationId,
  validateEventPayload,
} from './event-bus.js';

// Metrics
export {
  // Types
  type MetricType,
  type MetricLabels,
  type MetricDefinition,
  type CounterValue,
  type GaugeValue,
  type HistogramValue,
  type SummaryValue,
  type SCEIMetricsSnapshot,
  // Definitions
  SCEI_METRICS,
  // Class
  SCEIMetricsCollector,
  // Singleton
  getSCEIMetrics,
  resetSCEIMetrics,
} from './metrics.js';

// Persistence
export {
  // Types
  type StorageKey,
  type SCEINamespace,
  type StorageOptions,
  type StorageMetadata,
  type StoredItem,
  type QueryOptions,
  type StorageStats,
  type SCEIPersistence,
  // Implementations
  InMemorySCEIStorage,
  SCEIPersistenceManager,
  // Singleton
  getSCEIPersistence,
  setSCEIPersistence,
  resetSCEIPersistence,
} from './persistence.js';

// Unified Impact Measurement
export {
  // Service and factory
  UnifiedImpactService,
  getImpactService,
  resetImpactService,
  DEFAULT_IMPACT_CONFIG,
  // Types
  type SCEIDimension,
  type ImpactPolarity,
  type ImpactTimeframe,
  type ImpactMagnitude,
  type ImpactMetric,
  type SocialMetrics,
  type CivilizationalMetrics,
  type EcologicalMetrics,
  type IndividualMetrics,
  type ImpactAssessment,
  type ImpactSubject,
  type DomainImpact,
  type StakeholderImpact,
  type ImpactVerification,
  type ImpactAttestation,
  type ImpactRecommendation,
  type ImpactTrend,
  type CrossDomainCorrelation,
  type ImpactServiceConfig,
  type ImpactReport,
} from './unified-impact.js';

// ============================================================================
// CONVENIENCE INITIALIZER
// ============================================================================

import {
  getSCEIEventBus,
  resetSCEIEventBus,
  type SCEIEventBus,
} from './event-bus.js';
import { getSCEIMetrics, resetSCEIMetrics, type SCEIMetricsCollector } from './metrics.js';
import {
  getSCEIPersistence,
  resetSCEIPersistence,
  type SCEIPersistenceManager,
} from './persistence.js';

/**
 * SCEI Infrastructure Bundle
 *
 * Convenient access to all SCEI infrastructure components.
 */
export interface SCEIInfrastructure {
  eventBus: SCEIEventBus;
  metrics: SCEIMetricsCollector;
  persistence: SCEIPersistenceManager;
}

/**
 * Get all SCEI infrastructure components
 *
 * @example
 * ```typescript
 * const { eventBus, metrics, persistence } = getSCEIInfrastructure();
 *
 * // Subscribe to all events
 * eventBus.subscribe((event) => {
 *   console.log('SCEI Event:', event.type);
 *   metrics.incrementCounter('calibration_predictions_total', { domain: 'science' });
 * });
 * ```
 */
export function getSCEIInfrastructure(): SCEIInfrastructure {
  return {
    eventBus: getSCEIEventBus(),
    metrics: getSCEIMetrics(),
    persistence: getSCEIPersistence(),
  };
}

/**
 * Reset all SCEI infrastructure (for testing)
 *
 * This is the recommended way to reset all SCEI singletons between tests.
 * Resets: event bus, metrics collector, and persistence manager.
 *
 * @example
 * ```typescript
 * beforeEach(() => {
 *   resetSCEIInfrastructure();
 * });
 * ```
 */
export function resetSCEIInfrastructure(): void {
  resetSCEIEventBus();
  resetSCEIMetrics();
  resetSCEIPersistence();
}
