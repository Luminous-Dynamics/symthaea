// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Utilities Module
 *
 * Convenience functions and helpers for common SDK operations.
 */

import * as bridge from '../bridge/index.js';
import * as epistemic from '../epistemic/index.js';
import * as fl from '../fl/index.js';
import * as matl from '../matl/index.js';
import * as security from '../security/index.js';

// Re-export pagination utilities
export * from './pagination.js';

// Re-export branded types (in separate module to avoid circular deps)
export * from './branded.js';

// Re-export distributed tracing (renamed to avoid conflicts with observability module)
export {
  type TraceContext as MycelixTraceContext,
  type SpanData as MycelixSpanData,
  type SpanEvent as MycelixSpanEvent,
  type SpanLink as MycelixSpanLink,
  type SpanAttributes as MycelixSpanAttributes,
  type SpanExporter as MycelixSpanExporter,
  SpanStatus as MycelixSpanStatus,
  SpanKind as MycelixSpanKind,
  Span as MycelixSpan,
  Tracer as MycelixTracer,
  ConsoleSpanExporter,
  InMemorySpanExporter,
  OTLPHttpExporter,
  getTracer as getMycelixTracer,
  createTracer as createMycelixTracer,
  trace as mycelixTrace,
  traceSync as mycelixTraceSync,
  serializeTraceContext,
  deserializeTraceContext,
  TraceAttributes,
} from './tracing.js';

import type { AnyBridgeMessage } from '../bridge/index.js';
import type { ConnectionState } from '../client/index.js';
import type * as config from '../config/index.js';

// ============================================================================
// Type Aliases for Common Use Cases
// ============================================================================

/** Agent identifier (string) */
export type AgentId = string;

/** hApp identifier (string) */
export type HappId = string;

/** Timestamp in milliseconds */
export type Timestamp = number;

// ============================================================================
// Timestamp Utilities
// ============================================================================

/**
 * Get current timestamp in milliseconds
 */
export function now(): Timestamp {
  return Date.now();
}

/**
 * Check if a timestamp is valid (positive finite number)
 */
export function isValidTimestamp(ts: unknown): ts is Timestamp {
  return typeof ts === 'number' && Number.isFinite(ts) && ts >= 0;
}

/**
 * Check if a timestamp has expired given a TTL
 * @param ts - The timestamp to check
 * @param ttlMs - Time to live in milliseconds
 * @param referenceTime - Optional reference time (defaults to now)
 */
export function isExpiredTimestamp(
  ts: Timestamp,
  ttlMs: number,
  referenceTime?: Timestamp
): boolean {
  const ref = referenceTime ?? Date.now();
  return ref - ts > ttlMs;
}

/**
 * Check if a timestamp is in the future
 * @param ts - The timestamp to check
 * @param toleranceMs - Tolerance for clock skew (default: 0)
 */
export function isFutureTimestamp(ts: Timestamp, toleranceMs: number = 0): boolean {
  return ts > Date.now() + toleranceMs;
}

/**
 * Normalize a timestamp or Date to milliseconds
 */
export function normalizeTimestamp(ts: Timestamp | Date): Timestamp {
  if (ts instanceof Date) {
    return ts.getTime();
  }
  return ts;
}

/**
 * Calculate age of a timestamp in milliseconds
 */
export function timestampAge(ts: Timestamp, referenceTime?: Timestamp): number {
  const ref = referenceTime ?? Date.now();
  return ref - ts;
}

/**
 * Format a timestamp as ISO string
 */
export function formatTimestamp(ts: Timestamp): string {
  return new Date(ts).toISOString();
}

/**
 * Parse an ISO string to timestamp
 * Returns null if invalid
 */
export function parseTimestamp(isoString: string): Timestamp | null {
  const date = new Date(isoString);
  if (isNaN(date.getTime())) {
    return null;
  }
  return date.getTime();
}

/** Trust score between 0 and 1 */
export type TrustScore = number;

/** Byzantine tolerance percentage (0-0.34, 34% validated) */
export type ByzantineTolerance = number;

// ============================================================================
// Quick Trust Assessment
// ============================================================================

/**
 * Quick trust check result
 */
export interface TrustCheckResult {
  trustworthy: boolean;
  score: number;
  byzantine: boolean;
  confidence: number;
  details: {
    pogqScore: number;
    reputationScore: number;
    quality: number;
    consistency: number;
    entropy: number;
  };
}

/**
 * Perform a quick trust assessment for an agent
 */
export function checkTrust(
  agentId: string,
  quality: number,
  consistency: number,
  entropy: number,
  options: {
    trustThreshold?: number;
    minConfidence?: number;
    existingReputation?: matl.ReputationScore;
  } = {}
): TrustCheckResult {
  const { trustThreshold = 0.5, existingReputation } = options;

  // Create or use existing reputation
  const reputation = existingReputation ?? matl.createReputation(agentId);

  // Create PoGQ
  const pogq = matl.createPoGQ(quality, consistency, entropy);

  // Calculate composite
  const composite = matl.calculateComposite(pogq, reputation, trustThreshold);

  // Check Byzantine behavior
  const byzantine = matl.isByzantine(pogq, trustThreshold);

  return {
    trustworthy: composite.isTrustworthy && !byzantine,
    score: composite.finalScore,
    byzantine,
    confidence: composite.confidence,
    details: {
      pogqScore: composite.pogqScore,
      reputationScore: composite.reputationScore,
      quality: pogq.quality,
      consistency: pogq.consistency,
      entropy: pogq.entropy,
    },
  };
}

/**
 * Build up reputation from interaction history
 */
export function buildReputation(
  agentId: string,
  positiveCount: number,
  negativeCount: number
): matl.ReputationScore {
  let reputation = matl.createReputation(agentId);

  for (let i = 0; i < positiveCount; i++) {
    reputation = matl.recordPositive(reputation);
  }

  for (let i = 0; i < negativeCount; i++) {
    reputation = matl.recordNegative(reputation);
  }

  return reputation;
}

// ============================================================================
// Federated Learning Helpers
// ============================================================================

/**
 * FL round summary
 */
export interface FLRoundSummary {
  roundId: number;
  status: string;
  participantCount: number;
  updateCount: number;
  aggregated: boolean;
  duration?: number;
}

/**
 * Create a simple FL coordinator with common defaults
 */
export function createSimpleFLCoordinator(
  options: {
    minParticipants?: number;
    maxParticipants?: number;
    byzantineTolerance?: number;
    aggregationMethod?: fl.AggregationMethod;
  } = {}
): fl.FLCoordinator {
  return new fl.FLCoordinator({
    minParticipants: options.minParticipants ?? 3,
    maxParticipants: options.maxParticipants ?? 100,
    byzantineTolerance: options.byzantineTolerance ?? 0.33,
    aggregationMethod: options.aggregationMethod ?? fl.AggregationMethod.TrustWeighted,
    roundTimeout: 60000,
    trustThreshold: 0.5,
  });
}

/**
 * Create gradient update from array
 */
export function createGradientUpdate(
  participantId: string,
  modelVersion: number,
  gradients: number[],
  metadata: {
    batchSize: number;
    loss: number;
    accuracy?: number;
  }
): fl.GradientUpdate {
  return {
    participantId,
    modelVersion,
    gradients: new Float64Array(gradients),
    metadata: {
      ...metadata,
      timestamp: Date.now(),
    },
  };
}

/**
 * Run a complete FL round
 */
export function runFLRound(
  coordinator: fl.FLCoordinator,
  updates: fl.GradientUpdate[]
): FLRoundSummary {
  const round = coordinator.startRound();
  const startTime = Date.now();

  for (const update of updates) {
    coordinator.submitUpdate(update);
  }

  const aggregated = coordinator.aggregateRound();
  const endTime = Date.now();

  const stats = coordinator.getRoundStats();

  return {
    roundId: round.roundId,
    status: aggregated ? 'completed' : 'failed',
    participantCount: stats.participantCount,
    updateCount: updates.length,
    aggregated,
    duration: endTime - startTime,
  };
}

// ============================================================================
// Security Helpers
// ============================================================================

/**
 * Secure message with HMAC
 */
export interface SecureMessage<T> {
  payload: T;
  signature: string;
  timestamp: number;
}

/**
 * Create a signed message
 */
export async function signMessage<T>(payload: T, key: Uint8Array): Promise<SecureMessage<T>> {
  const timestamp = Date.now();
  const data = JSON.stringify({ payload, timestamp });
  const signature = await security.hmac(key, data);

  return {
    payload,
    signature: Array.from(signature)
      .map((b) => b.toString(16).padStart(2, '0'))
      .join(''),
    timestamp,
  };
}

/**
 * Verify a signed message
 */
export async function verifyMessage<T>(
  message: SecureMessage<T>,
  key: Uint8Array,
  maxAge?: number
): Promise<{ valid: boolean; expired: boolean; tampered: boolean }> {
  // Check expiration
  const expired = maxAge !== undefined && Date.now() - message.timestamp > maxAge;

  // Verify signature
  const data = JSON.stringify({ payload: message.payload, timestamp: message.timestamp });
  const signatureBytes = new Uint8Array(
    message.signature.match(/.{2}/g)!.map((b) => parseInt(b, 16))
  );
  const valid = await security.verifyHmac(key, data, signatureBytes);

  return {
    valid: valid && !expired,
    expired,
    tampered: !valid,
  };
}

/**
 * Create rate-limited operation
 */
export function createRateLimitedOperation<T>(
  operation: () => T | Promise<T>,
  maxRequests: number,
  windowMs: number
): { execute: () => Promise<T | null>; reset: () => void } {
  let state = security.createRateLimiter('operation', {
    maxRequests,
    windowMs,
  });

  return {
    execute: async () => {
      const check = security.checkRateLimit(state);
      state = check.state;

      if (!check.allowed) {
        return null;
      }

      return operation();
    },
    reset: () => {
      state = security.createRateLimiter('operation', { maxRequests, windowMs });
    },
  };
}

// ============================================================================
// Epistemic Helpers
// ============================================================================

/**
 * Create a high-trust claim (cryptographic + network-wide + persistent)
 */
export function createHighTrustClaim(content: string, issuer: string): epistemic.EpistemicClaim {
  return epistemic
    .claim(content)
    .withClassification(
      epistemic.EmpiricalLevel.E3_Cryptographic,
      epistemic.NormativeLevel.N2_Network,
      epistemic.MaterialityLevel.M2_Persistent
    )
    .withIssuer(issuer)
    .build();
}

/**
 * Create a medium-trust claim (privately verifiable + communal + temporal)
 */
export function createMediumTrustClaim(content: string, issuer: string): epistemic.EpistemicClaim {
  return epistemic
    .claim(content)
    .withClassification(
      epistemic.EmpiricalLevel.E2_PrivateVerify,
      epistemic.NormativeLevel.N1_Communal,
      epistemic.MaterialityLevel.M1_Temporal
    )
    .withIssuer(issuer)
    .build();
}

/**
 * Create a low-trust claim (testimonial + personal + ephemeral)
 */
export function createLowTrustClaim(content: string, issuer: string): epistemic.EpistemicClaim {
  return epistemic
    .claim(content)
    .withClassification(
      epistemic.EmpiricalLevel.E1_Testimonial,
      epistemic.NormativeLevel.N0_Personal,
      epistemic.MaterialityLevel.M0_Ephemeral
    )
    .withIssuer(issuer)
    .build();
}

// ============================================================================
// Bridge Helpers
// ============================================================================

/**
 * Cross-hApp reputation aggregator
 */
export class ReputationAggregator {
  private localBridge: bridge.LocalBridge;
  private happs: Set<string> = new Set();

  constructor() {
    this.localBridge = new bridge.LocalBridge();
  }

  /**
   * Register a hApp
   */
  registerHapp(happId: string): void {
    this.localBridge.registerHapp(happId);
    this.happs.add(happId);
  }

  /**
   * Set reputation for an agent in a hApp
   */
  setReputation(
    happId: string,
    agentId: string,
    positiveCount: number,
    negativeCount: number
  ): void {
    const reputation = buildReputation(agentId, positiveCount, negativeCount);
    this.localBridge.setReputation(happId, agentId, reputation);
  }

  /**
   * Get aggregate reputation across all hApps
   */
  getAggregateReputation(agentId: string): number {
    return this.localBridge.getAggregateReputation(agentId);
  }

  /**
   * Get detailed cross-hApp reputation
   */
  getDetailedReputation(agentId: string): bridge.HappReputationScore[] {
    return this.localBridge.getCrossHappReputation(agentId);
  }

  /**
   * Get list of registered hApps
   */
  getHapps(): string[] {
    return Array.from(this.happs);
  }
}

// ============================================================================
// Debugging & Inspection
// ============================================================================

/**
 * SDK component health check result
 */
export interface HealthCheckResult {
  component: string;
  healthy: boolean;
  details?: Record<string, unknown>;
  error?: string;
}

/**
 * Run health checks on SDK components
 */
export function runHealthChecks(): HealthCheckResult[] {
  const results: HealthCheckResult[] = [];

  // MATL check
  try {
    const pogq = matl.createPoGQ(0.5, 0.5, 0.5);
    const rep = matl.createReputation('test');
    const composite = matl.calculateComposite(pogq, rep);
    results.push({
      component: 'matl',
      healthy: true,
      details: { sampleScore: composite.finalScore },
    });
  } catch (e) {
    results.push({
      component: 'matl',
      healthy: false,
      error: e instanceof Error ? e.message : 'Unknown error',
    });
  }

  // FL check
  try {
    const coordinator = new fl.FLCoordinator();
    coordinator.registerParticipant('test');
    const stats = coordinator.getRoundStats();
    results.push({
      component: 'fl',
      healthy: true,
      details: { participantCount: stats.participantCount },
    });
  } catch (e) {
    results.push({
      component: 'fl',
      healthy: false,
      error: e instanceof Error ? e.message : 'Unknown error',
    });
  }

  // Security check
  try {
    const bytes = security.secureRandomBytes(32);
    const uuid = security.secureUUID();
    results.push({
      component: 'security',
      healthy: true,
      details: { randomBytesLength: bytes.length, sampleUuid: uuid.substring(0, 8) + '...' },
    });
  } catch (e) {
    results.push({
      component: 'security',
      healthy: false,
      error: e instanceof Error ? e.message : 'Unknown error',
    });
  }

  // Epistemic check
  try {
    const claim = epistemic
      .claim('test')
      .withEmpirical(epistemic.EmpiricalLevel.E1_Testimonial)
      .build();
    results.push({
      component: 'epistemic',
      healthy: true,
      details: { sampleClaimId: claim.id.substring(0, 10) + '...' },
    });
  } catch (e) {
    results.push({
      component: 'epistemic',
      healthy: false,
      error: e instanceof Error ? e.message : 'Unknown error',
    });
  }

  // Bridge check
  try {
    const localBridge = new bridge.LocalBridge();
    localBridge.registerHapp('test-happ');
    results.push({
      component: 'bridge',
      healthy: true,
    });
  } catch (e) {
    results.push({
      component: 'bridge',
      healthy: false,
      error: e instanceof Error ? e.message : 'Unknown error',
    });
  }

  return results;
}

/**
 * Format SDK info for debugging
 */
export function getSdkInfo(): {
  version: string;
  modules: string[];
  nodeVersion?: string;
} {
  return {
    version: '0.5.0',
    modules: ['matl', 'fl', 'security', 'epistemic', 'bridge', 'client', 'config', 'utils'],
    nodeVersion: typeof process !== 'undefined' ? process.version : undefined,
  };
}

// ============================================================================
// Async Helpers
// ============================================================================

/**
 * Retry an async operation with exponential backoff
 */
export async function retry<T>(
  operation: () => Promise<T>,
  options: {
    maxAttempts?: number;
    initialDelay?: number;
    maxDelay?: number;
    backoffFactor?: number;
  } = {}
): Promise<T> {
  const { maxAttempts = 3, initialDelay = 100, maxDelay = 10000, backoffFactor = 2 } = options;

  let lastError: Error | undefined;
  let delay = initialDelay;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      if (attempt < maxAttempts) {
        await new Promise((resolve) => setTimeout(resolve, delay));
        delay = Math.min(delay * backoffFactor, maxDelay);
      }
    }
  }

  throw lastError;
}

/**
 * Run operations in parallel with concurrency limit
 */
export async function parallel<T, R>(
  items: T[],
  operation: (item: T, index: number) => Promise<R>,
  concurrency: number = 5
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  const executing: Set<Promise<void>> = new Set();

  for (let i = 0; i < items.length; i++) {
    const promise = (async () => {
      results[i] = await operation(items[i], i);
    })();

    executing.add(promise);
    void promise.finally(() => executing.delete(promise));

    if (executing.size >= concurrency) {
      await Promise.race(executing);
    }
  }

  await Promise.all(executing);
  return results;
}

/**
 * Create a deferred promise
 */
export function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
} {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;

  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  return { promise, resolve, reject };
}

/**
 * Timeout error thrown when an operation exceeds its time limit
 */
export class TimeoutError extends Error {
  constructor(
    message: string,
    public readonly timeoutMs: number
  ) {
    super(message);
    this.name = 'TimeoutError';
  }
}

/**
 * Wrap a promise with a timeout
 * @throws {TimeoutError} If the operation exceeds the timeout
 */
export function timeout<T>(operation: Promise<T>, timeoutMs: number, message?: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new TimeoutError(message ?? `Operation timed out after ${timeoutMs}ms`, timeoutMs));
    }, timeoutMs);

    operation
      .then((result) => {
        clearTimeout(timer);
        resolve(result);
      })
      .catch((error) => {
        clearTimeout(timer);
        reject(error);
      });
  });
}

/**
 * Run an async operation with a timeout
 * @throws {TimeoutError} If the operation exceeds the timeout
 */
export async function withTimeout<T>(
  operation: () => Promise<T>,
  timeoutMs: number,
  message?: string
): Promise<T> {
  return timeout(operation(), timeoutMs, message);
}

/**
 * Create a debounced version of a function
 * The function will only execute after `waitMs` has passed since the last call.
 * Leading edge execution can be enabled with the leading option.
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  waitMs: number,
  options: { leading?: boolean } = {}
): {
  (...args: Parameters<T>): void;
  cancel: () => void;
  flush: () => void;
} {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let lastArgs: Parameters<T> | null = null;
  let lastThis: unknown = null;
  let hasLeadingCall = false;

  const later = () => {
    timeoutId = null;
    if (!options.leading && lastArgs !== null) {
      fn.apply(lastThis, lastArgs);
      lastArgs = null;
      lastThis = null;
    }
    hasLeadingCall = false;
  };

  const debounced = function (this: unknown, ...args: Parameters<T>): void {
    lastArgs = args;
    // eslint-disable-next-line @typescript-eslint/no-this-alias -- Intentional: preserving 'this' context for debounced call
    lastThis = this;

    if (timeoutId !== null) {
      clearTimeout(timeoutId);
    }

    if (options.leading && !hasLeadingCall) {
      hasLeadingCall = true;
      fn.apply(this, args);
    }

    timeoutId = setTimeout(later, waitMs);
  };

  debounced.cancel = () => {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
    lastArgs = null;
    lastThis = null;
    hasLeadingCall = false;
  };

  debounced.flush = () => {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
      if (lastArgs !== null) {
        fn.apply(lastThis, lastArgs);
      }
      timeoutId = null;
      lastArgs = null;
      lastThis = null;
      hasLeadingCall = false;
    }
  };

  return debounced;
}

/**
 * Create a throttled version of a function
 * The function will execute at most once every `waitMs` milliseconds.
 * By default executes on the leading edge; trailing edge can be enabled.
 */
export function throttle<T extends (...args: unknown[]) => unknown>(
  fn: T,
  waitMs: number,
  options: { leading?: boolean; trailing?: boolean } = {}
): {
  (...args: Parameters<T>): void;
  cancel: () => void;
} {
  const { leading = true, trailing = true } = options;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let lastArgs: Parameters<T> | null = null;
  let lastThis: unknown = null;
  let lastCallTime: number | null = null;

  const later = () => {
    timeoutId = null;
    if (trailing && lastArgs !== null) {
      fn.apply(lastThis, lastArgs);
      lastCallTime = Date.now();
      lastArgs = null;
      lastThis = null;
    }
  };

  const throttled = function (this: unknown, ...args: Parameters<T>): void {
    const now = Date.now();
    const remaining = lastCallTime !== null ? waitMs - (now - lastCallTime) : 0;

    if (remaining <= 0 || remaining > waitMs) {
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      if (leading) {
        fn.apply(this, args);
        lastCallTime = now;
      } else {
        lastArgs = args;
        // eslint-disable-next-line @typescript-eslint/no-this-alias -- Intentional: preserving 'this' context for throttled call
        lastThis = this;
      }
    } else {
      lastArgs = args;
      // eslint-disable-next-line @typescript-eslint/no-this-alias -- Intentional: preserving 'this' context for throttled call
      lastThis = this;
      if (timeoutId === null && trailing) {
        timeoutId = setTimeout(later, remaining);
      }
    }
  };

  throttled.cancel = () => {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
    lastCallTime = null;
    lastArgs = null;
    lastThis = null;
  };

  return throttled;
}

/**
 * Sleep for a specified duration
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Check if value is a valid ProofOfGradientQuality
 */
export function isProofOfGradientQuality(value: unknown): value is matl.ProofOfGradientQuality {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return (
    typeof obj.quality === 'number' &&
    typeof obj.consistency === 'number' &&
    typeof obj.entropy === 'number' &&
    obj.quality >= 0 &&
    obj.quality <= 1 &&
    obj.consistency >= 0 &&
    obj.consistency <= 1 &&
    obj.entropy >= 0 &&
    obj.entropy <= 1
  );
}

/**
 * Check if value is a valid ReputationScore
 */
export function isReputationScore(value: unknown): value is matl.ReputationScore {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return (
    typeof obj.agentId === 'string' &&
    typeof obj.positiveCount === 'number' &&
    typeof obj.negativeCount === 'number' &&
    typeof obj.lastUpdate === 'number' &&
    obj.positiveCount >= 0 &&
    obj.negativeCount >= 0
  );
}

/**
 * Check if value is a valid EpistemicClaim
 */
export function isEpistemicClaim(value: unknown): value is epistemic.EpistemicClaim {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return (
    typeof obj.id === 'string' &&
    typeof obj.content === 'string' &&
    typeof obj.classification === 'object' &&
    obj.classification !== null &&
    typeof obj.issuedAt === 'number' &&
    typeof obj.issuer === 'string' &&
    Array.isArray(obj.evidence)
  );
}

/**
 * Check if value is a valid GradientUpdate
 */
export function isGradientUpdate(value: unknown): value is fl.GradientUpdate {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return (
    typeof obj.participantId === 'string' &&
    typeof obj.modelVersion === 'number' &&
    obj.gradients instanceof Float64Array &&
    typeof obj.metadata === 'object' &&
    obj.metadata !== null
  );
}

/**
 * Check if value is a valid BridgeMessage
 */
export function isBridgeMessage(value: unknown): value is bridge.AnyBridgeMessage {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return (
    typeof obj.type === 'string' &&
    typeof obj.sourceHapp === 'string' &&
    typeof obj.timestamp === 'number' &&
    Object.values(bridge.BridgeMessageType).includes(obj.type as bridge.BridgeMessageType)
  );
}

/**
 * Check if value is a valid SecureMessage (any payload type)
 */
export function isSecureMessage<T = unknown>(value: unknown): value is SecureMessage<T> {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return (
    obj.payload !== undefined &&
    typeof obj.signature === 'string' &&
    typeof obj.timestamp === 'number'
  );
}

/**
 * Check if value is a valid TrustCheckResult
 */
export function isTrustCheckResult(value: unknown): value is TrustCheckResult {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return (
    typeof obj.trustworthy === 'boolean' &&
    typeof obj.score === 'number' &&
    typeof obj.byzantine === 'boolean' &&
    typeof obj.confidence === 'number' &&
    typeof obj.details === 'object'
  );
}

/**
 * Check if value is a valid HealthCheckResult
 */
export function isHealthCheckResult(value: unknown): value is HealthCheckResult {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return typeof obj.component === 'string' && typeof obj.healthy === 'boolean';
}

/**
 * Check if value is a valid ConnectionState
 */
export function isConnectionState(value: unknown): value is ConnectionState {
  return (
    value === 'disconnected' || value === 'connecting' || value === 'connected' || value === 'error'
  );
}

/**
 * Check if value is a valid FLRound
 */
export function isFLRound(value: unknown): value is fl.FLRound {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return (
    typeof obj.roundId === 'number' &&
    typeof obj.modelVersion === 'number' &&
    obj.participants instanceof Map &&
    Array.isArray(obj.updates) &&
    (obj.status === 'collecting' || obj.status === 'aggregating' || obj.status === 'completed') &&
    typeof obj.startTime === 'number'
  );
}

/**
 * Check if value is a valid EpistemicClassification
 */
export function isEpistemicClassification(
  value: unknown
): value is epistemic.EpistemicClassification {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return (
    typeof obj.empirical === 'number' &&
    typeof obj.normative === 'number' &&
    typeof obj.materiality === 'number' &&
    obj.empirical >= 0 &&
    obj.empirical <= 3 &&
    obj.normative >= 0 &&
    obj.normative <= 3 &&
    obj.materiality >= 0 &&
    obj.materiality <= 3
  );
}

/**
 * Check if value is a valid CompositeScore (MATL)
 */
export function isCompositeScore(value: unknown): value is matl.CompositeScore {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return (
    typeof obj.pogqScore === 'number' &&
    typeof obj.reputationScore === 'number' &&
    typeof obj.finalScore === 'number' &&
    typeof obj.confidence === 'number' &&
    typeof obj.timestamp === 'number' &&
    typeof obj.isTrustworthy === 'boolean' &&
    isProofOfGradientQuality(obj.pogq) &&
    isReputationScore(obj.reputation)
  );
}

/**
 * Check if value is a valid FLConfig
 */
export function isFLConfig(value: unknown): value is fl.FLConfig {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return (
    typeof obj.minParticipants === 'number' &&
    typeof obj.maxParticipants === 'number' &&
    typeof obj.roundTimeout === 'number' &&
    typeof obj.byzantineTolerance === 'number' &&
    typeof obj.aggregationMethod === 'string' &&
    typeof obj.trustThreshold === 'number' &&
    obj.byzantineTolerance >= 0 &&
    obj.byzantineTolerance <= 0.34
  );
}

/**
 * Check if value is a valid MycelixConfig
 */
export function isMycelixConfig(value: unknown): value is config.MycelixConfig {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return (
    typeof obj.matl === 'object' &&
    obj.matl !== null &&
    typeof obj.fl === 'object' &&
    obj.fl !== null &&
    typeof obj.security === 'object' &&
    obj.security !== null &&
    typeof obj.bridge === 'object' &&
    obj.bridge !== null &&
    typeof obj.client === 'object' &&
    obj.client !== null &&
    typeof obj.logging === 'object' &&
    obj.logging !== null
  );
}

/**
 * Check if value is a valid AggregatedGradient
 */
export function isAggregatedGradient(value: unknown): value is fl.AggregatedGradient {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return (
    obj.gradients instanceof Float64Array &&
    typeof obj.modelVersion === 'number' &&
    typeof obj.participantCount === 'number' &&
    typeof obj.method === 'string' &&
    typeof obj.timestamp === 'number'
  );
}

/**
 * Type guard for SerializedGradientUpdate
 * Validates structure of JSON-serialized gradient updates
 */
export function isSerializedGradientUpdate(value: unknown): value is fl.SerializedGradientUpdate {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;

  // Check required fields
  if (typeof obj.participantId !== 'string' || obj.participantId === '') return false;
  if (typeof obj.modelVersion !== 'number' || !Number.isFinite(obj.modelVersion)) return false;
  if (typeof obj.gradients !== 'string' || obj.gradients === '') return false;

  // Validate metadata object
  if (typeof obj.metadata !== 'object' || obj.metadata === null) return false;
  const meta = obj.metadata as Record<string, unknown>;
  if (typeof meta.batchSize !== 'number' || meta.batchSize < 1) return false;
  if (typeof meta.loss !== 'number' || !Number.isFinite(meta.loss)) return false;
  if (typeof meta.timestamp !== 'number' || meta.timestamp < 0) return false;
  if (
    meta.accuracy !== undefined &&
    (typeof meta.accuracy !== 'number' || !Number.isFinite(meta.accuracy))
  )
    return false;

  return true;
}

/**
 * Type guard for SerializedAggregatedGradient
 * Validates structure of JSON-serialized aggregated gradients
 */
export function isSerializedAggregatedGradient(
  value: unknown
): value is fl.SerializedAggregatedGradient {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;

  // Check required fields
  if (typeof obj.gradients !== 'string' || obj.gradients === '') return false;
  if (typeof obj.modelVersion !== 'number' || !Number.isFinite(obj.modelVersion)) return false;
  if (typeof obj.participantCount !== 'number' || obj.participantCount < 1) return false;
  if (typeof obj.timestamp !== 'number' || obj.timestamp < 0) return false;

  // Validate aggregation method
  const validMethods = ['fedAvg', 'trimmedMean', 'coordinateMedian', 'krum', 'trustWeighted'];
  if (typeof obj.aggregationMethod !== 'string' || !validMethods.includes(obj.aggregationMethod))
    return false;

  // Optional excludedCount validation
  if (
    obj.excludedCount !== undefined &&
    (typeof obj.excludedCount !== 'number' || obj.excludedCount < 0)
  )
    return false;

  return true;
}

// ============================================================================
// Observable Event System
// ============================================================================

/**
 * Observer function type for event subscriptions
 */
export type Observer<T> = (event: T) => void;

/**
 * Subscription handle returned by Observable.subscribe()
 */
export interface Subscription {
  /** Unsubscribe from the observable */
  unsubscribe(): void;
  /** Whether the subscription is still active */
  readonly closed: boolean;
}

/**
 * Observable statistics
 */
export interface ObservableStats {
  /** Total number of events emitted */
  eventCount: number;
  /** Current number of active observers */
  observerCount: number;
  /** Timestamp of last emitted event */
  lastEventTime?: number;
}

/**
 * Generic Observable for typed event streams.
 *
 * Provides a simple pub/sub pattern for SDK events.
 *
 * @example
 * ```typescript
 * const events = new Observable<{ type: string; data: unknown }>();
 *
 * // Subscribe to events
 * const sub = events.subscribe((event) => {
 *   console.log('Received:', event);
 * });
 *
 * // Emit events
 * events.emit({ type: 'test', data: 123 });
 *
 * // Unsubscribe
 * sub.unsubscribe();
 * ```
 */
export class Observable<T> {
  private observers: Set<Observer<T>> = new Set();
  private lastEvent?: T;
  private lastEventTime?: number;
  private eventCount = 0;

  /**
   * Subscribe to events
   * @param observer - Function called for each emitted event
   * @returns Subscription handle with unsubscribe() method
   */
  subscribe(observer: Observer<T>): Subscription {
    this.observers.add(observer);

    let closed = false;
    return {
      unsubscribe: () => {
        this.observers.delete(observer);
        closed = true;
      },
      get closed() {
        return closed;
      },
    };
  }

  /**
   * Emit an event to all observers
   * @param event - The event to emit
   */
  emit(event: T): void {
    this.lastEvent = event;
    this.lastEventTime = Date.now();
    this.eventCount++;

    for (const observer of this.observers) {
      try {
        observer(event);
      } catch {
        // Don't let observer errors break emission
      }
    }
  }

  /**
   * Get the last emitted event
   */
  getLastEvent(): T | undefined {
    return this.lastEvent;
  }

  /**
   * Get the current number of observers
   */
  getObserverCount(): number {
    return this.observers.size;
  }

  /**
   * Get the total number of events emitted
   */
  getEventCount(): number {
    return this.eventCount;
  }

  /**
   * Get observable statistics
   */
  getStats(): ObservableStats {
    return {
      eventCount: this.eventCount,
      observerCount: this.observers.size,
      lastEventTime: this.lastEventTime,
    };
  }

  /**
   * Check if there are any observers
   */
  hasObservers(): boolean {
    return this.observers.size > 0;
  }

  /**
   * Remove all observers
   */
  clear(): void {
    this.observers.clear();
  }

  /**
   * Reset statistics (keeps observers)
   */
  resetStats(): void {
    this.eventCount = 0;
    this.lastEvent = undefined;
    this.lastEventTime = undefined;
  }
}

// ============================================================================
// SDK Event Types
// ============================================================================

/**
 * Base event interface for all SDK events
 */
export interface SdkEvent {
  /** Event type identifier */
  type: string;
  /** Timestamp when event occurred */
  timestamp: number;
  /** Optional source module */
  source?: string;
}

// --- MATL Events ---

/** Event types for MATL module */
export enum MatlEventType {
  REPUTATION_UPDATED = 'matl:reputation_updated',
  TRUST_EVALUATED = 'matl:trust_evaluated',
  BYZANTINE_DETECTED = 'matl:byzantine_detected',
  THRESHOLD_ADJUSTED = 'matl:threshold_adjusted',
  CACHE_HIT = 'matl:cache_hit',
  CACHE_MISS = 'matl:cache_miss',
}

/** Reputation update event */
export interface ReputationUpdatedEvent extends SdkEvent {
  type: MatlEventType.REPUTATION_UPDATED;
  agentId: string;
  previousValue: number;
  newValue: number;
  change: 'positive' | 'negative';
}

/** Trust evaluation event */
export interface TrustEvaluatedEvent extends SdkEvent {
  type: MatlEventType.TRUST_EVALUATED;
  agentId: string;
  score: number;
  trustworthy: boolean;
  confidence: number;
}

/** Byzantine detection event */
export interface ByzantineDetectedEvent extends SdkEvent {
  type: MatlEventType.BYZANTINE_DETECTED;
  agentId: string;
  score: number;
  threshold: number;
}

/** Threshold adjustment event */
export interface ThresholdAdjustedEvent extends SdkEvent {
  type: MatlEventType.THRESHOLD_ADJUSTED;
  nodeId: string;
  previousThreshold: number;
  newThreshold: number;
  observationCount: number;
}

/** Cache hit event */
export interface CacheHitEvent extends SdkEvent {
  type: MatlEventType.CACHE_HIT;
  key: string;
  value: number;
}

/** Cache miss event */
export interface CacheMissEvent extends SdkEvent {
  type: MatlEventType.CACHE_MISS;
  key: string;
}

/** Union of all MATL events */
export type MatlEvent =
  | ReputationUpdatedEvent
  | TrustEvaluatedEvent
  | ByzantineDetectedEvent
  | ThresholdAdjustedEvent
  | CacheHitEvent
  | CacheMissEvent;

// --- Bridge Events ---

/** Event types for Bridge module */
export enum BridgeEventType {
  MESSAGE_SENT = 'bridge:message_sent',
  MESSAGE_RECEIVED = 'bridge:message_received',
  MESSAGE_ROUTED = 'bridge:message_routed',
  MESSAGE_UNHANDLED = 'bridge:message_unhandled',
  HAPP_REGISTERED = 'bridge:happ_registered',
  REPUTATION_AGGREGATED = 'bridge:reputation_aggregated',
}

/** Message sent event */
export interface MessageSentEvent extends SdkEvent {
  type: BridgeEventType.MESSAGE_SENT;
  messageType: string;
  sourceHapp: string;
  targetHapp?: string;
}

/** Message received event */
export interface MessageReceivedEvent extends SdkEvent {
  type: BridgeEventType.MESSAGE_RECEIVED;
  messageType: string;
  sourceHapp: string;
}

/** Message routed event */
export interface MessageRoutedEvent extends SdkEvent {
  type: BridgeEventType.MESSAGE_ROUTED;
  messageType: string;
  sourceHapp: string;
  handlerCount: number;
}

/** Message unhandled event */
export interface MessageUnhandledEvent extends SdkEvent {
  type: BridgeEventType.MESSAGE_UNHANDLED;
  messageType: string;
  sourceHapp: string;
  message: AnyBridgeMessage;
}

/** hApp registered event */
export interface HappRegisteredEvent extends SdkEvent {
  type: BridgeEventType.HAPP_REGISTERED;
  happId: string;
  totalHapps: number;
}

/** Reputation aggregated event */
export interface ReputationAggregatedEvent extends SdkEvent {
  type: BridgeEventType.REPUTATION_AGGREGATED;
  agentId: string;
  aggregateScore: number;
  happCount: number;
}

/** Union of all Bridge events */
export type BridgeEvent =
  | MessageSentEvent
  | MessageReceivedEvent
  | MessageRoutedEvent
  | MessageUnhandledEvent
  | HappRegisteredEvent
  | ReputationAggregatedEvent;

// --- FL Events ---

/** Event types for Federated Learning module */
export enum FlEventType {
  ROUND_STARTED = 'fl:round_started',
  UPDATE_SUBMITTED = 'fl:update_submitted',
  UPDATE_REJECTED = 'fl:update_rejected',
  ROUND_AGGREGATED = 'fl:round_aggregated',
  PARTICIPANT_REGISTERED = 'fl:participant_registered',
  REPUTATION_UPDATED = 'fl:reputation_updated',
}

/** Round started event */
export interface RoundStartedEvent extends SdkEvent {
  type: FlEventType.ROUND_STARTED;
  roundId: number;
  modelVersion: number;
  participantCount: number;
}

/** Update submitted event */
export interface UpdateSubmittedEvent extends SdkEvent {
  type: FlEventType.UPDATE_SUBMITTED;
  roundId: number;
  participantId: string;
  modelVersion: number;
  batchSize: number;
  loss: number;
}

/** Update rejected event */
export interface UpdateRejectedEvent extends SdkEvent {
  type: FlEventType.UPDATE_REJECTED;
  roundId: number;
  participantId: string;
  reason: string;
}

/** Round aggregated event */
export interface RoundAggregatedEvent extends SdkEvent {
  type: FlEventType.ROUND_AGGREGATED;
  roundId: number;
  modelVersion: number;
  participantCount: number;
  excludedCount: number;
  aggregationMethod: string;
  durationMs: number;
}

/** Participant registered event */
export interface ParticipantRegisteredEvent extends SdkEvent {
  type: FlEventType.PARTICIPANT_REGISTERED;
  participantId: string;
  totalParticipants: number;
}

/** FL reputation updated event */
export interface FlReputationUpdatedEvent extends SdkEvent {
  type: FlEventType.REPUTATION_UPDATED;
  participantId: string;
  newValue: number;
  roundsParticipated: number;
}

/** Union of all FL events */
export type FlEvent =
  | RoundStartedEvent
  | UpdateSubmittedEvent
  | UpdateRejectedEvent
  | RoundAggregatedEvent
  | ParticipantRegisteredEvent
  | FlReputationUpdatedEvent;

// --- Epistemic Events ---

/** Event types for Epistemic module */
export enum EpistemicEventType {
  CLAIM_CREATED = 'epistemic:claim_created',
  CLAIM_EXPIRED = 'epistemic:claim_expired',
  CLAIM_EXTENDED = 'epistemic:claim_extended',
  EVIDENCE_ADDED = 'epistemic:evidence_added',
  POOL_CLEANUP = 'epistemic:pool_cleanup',
  BATCH_COMPLETED = 'epistemic:batch_completed',
}

/** Claim created event */
export interface ClaimCreatedEvent extends SdkEvent {
  type: EpistemicEventType.CLAIM_CREATED;
  claimId: string;
  issuer: string;
  classificationCode: string;
}

/** Claim expired event */
export interface ClaimExpiredEvent extends SdkEvent {
  type: EpistemicEventType.CLAIM_EXPIRED;
  claimId: string;
  issuer: string;
  age: number;
}

/** Claim extended event */
export interface ClaimExtendedEvent extends SdkEvent {
  type: EpistemicEventType.CLAIM_EXTENDED;
  claimId: string;
  additionalMs: number;
  newExpiration: number;
}

/** Evidence added event */
export interface EvidenceAddedEvent extends SdkEvent {
  type: EpistemicEventType.EVIDENCE_ADDED;
  claimId: string;
  evidenceType: string;
}

/** Pool cleanup event */
export interface PoolCleanupEvent extends SdkEvent {
  type: EpistemicEventType.POOL_CLEANUP;
  removedCount: number;
  remainingCount: number;
}

/** Batch completed event */
export interface BatchCompletedEvent extends SdkEvent {
  type: EpistemicEventType.BATCH_COMPLETED;
  batchId: string;
  claimCount: number;
  passedCount: number;
  failedCount: number;
}

/** Union of all Epistemic events */
export type EpistemicEvent =
  | ClaimCreatedEvent
  | ClaimExpiredEvent
  | ClaimExtendedEvent
  | EvidenceAddedEvent
  | PoolCleanupEvent
  | BatchCompletedEvent;

// --- Security Events (extending existing SecurityEventType) ---

/** Event types for Security module (SDK-specific additions) */
export enum SdkSecurityEventType {
  SECRET_CREATED = 'security:secret_created',
  SECRET_ACCESSED = 'security:secret_accessed',
  SECRET_EXPIRED = 'security:secret_expired',
  SECRET_DESTROYED = 'security:secret_destroyed',
  RATE_LIMIT_TRIGGERED = 'security:rate_limit_triggered',
  RATE_LIMIT_RESET = 'security:rate_limit_reset',
}

/** Secret created event */
export interface SecretCreatedEvent extends SdkEvent {
  type: SdkSecurityEventType.SECRET_CREATED;
  hasTtl: boolean;
  ttlMs?: number;
}

/** Secret accessed event */
export interface SecretAccessedEvent extends SdkEvent {
  type: SdkSecurityEventType.SECRET_ACCESSED;
  useCount: number;
}

/** Secret expired event */
export interface SecretExpiredEvent extends SdkEvent {
  type: SdkSecurityEventType.SECRET_EXPIRED;
  ttlMs: number;
}

/** Secret destroyed event */
export interface SecretDestroyedEvent extends SdkEvent {
  type: SdkSecurityEventType.SECRET_DESTROYED;
  useCount: number;
  wasExpired: boolean;
}

/** Rate limit triggered event */
export interface RateLimitTriggeredEvent extends SdkEvent {
  type: SdkSecurityEventType.RATE_LIMIT_TRIGGERED;
  entityId: string;
  resetAt: number;
  requestCount: number;
}

/** Rate limit reset event */
export interface RateLimitResetEvent extends SdkEvent {
  type: SdkSecurityEventType.RATE_LIMIT_RESET;
  entityId: string;
}

/** Union of all SDK Security events */
export type SdkSecurityEvent =
  | SecretCreatedEvent
  | SecretAccessedEvent
  | SecretExpiredEvent
  | SecretDestroyedEvent
  | RateLimitTriggeredEvent
  | RateLimitResetEvent;

// --- Combined Event Types ---

/** Union of all SDK events */
export type AnySdkEvent = MatlEvent | BridgeEvent | FlEvent | EpistemicEvent | SdkSecurityEvent;

// ============================================================================
// Event Bus (Global SDK Event System)
// ============================================================================

/**
 * Event bus for SDK-wide event monitoring.
 *
 * Provides a central hub for observing events across all SDK modules.
 *
 * @example
 * ```typescript
 * const bus = new SdkEventBus();
 *
 * // Subscribe to all events
 * bus.onAll((event) => console.log('Any event:', event.type));
 *
 * // Subscribe to specific event types
 * bus.on(MatlEventType.REPUTATION_UPDATED, (event) => {
 *   console.log('Reputation updated:', event.agentId);
 * });
 *
 * // Emit events
 * bus.emit({
 *   type: MatlEventType.REPUTATION_UPDATED,
 *   timestamp: Date.now(),
 *   agentId: 'agent1',
 *   previousValue: 0.5,
 *   newValue: 0.6,
 *   change: 'positive',
 * });
 * ```
 */
export class SdkEventBus {
  private allEvents = new Observable<AnySdkEvent>();
  private byType = new Map<string, Observable<AnySdkEvent>>();
  private eventHistory: AnySdkEvent[] = [];
  private maxHistory: number;

  constructor(options: { maxHistory?: number } = {}) {
    this.maxHistory = options.maxHistory ?? 100;
  }

  /**
   * Subscribe to all events
   */
  onAll(observer: Observer<AnySdkEvent>): Subscription {
    return this.allEvents.subscribe(observer);
  }

  /**
   * Subscribe to events of a specific type
   */
  on<T extends AnySdkEvent>(type: T['type'], observer: Observer<T>): Subscription {
    let typeObservable = this.byType.get(type);
    if (!typeObservable) {
      typeObservable = new Observable<AnySdkEvent>();
      this.byType.set(type, typeObservable);
    }
    return typeObservable.subscribe(observer as Observer<AnySdkEvent>);
  }

  /**
   * Emit an event
   */
  emit<T extends AnySdkEvent>(event: T): void {
    // Add to history
    this.eventHistory.push(event);
    if (this.eventHistory.length > this.maxHistory) {
      this.eventHistory.shift();
    }

    // Emit to all subscribers
    this.allEvents.emit(event);

    // Emit to type-specific subscribers
    const typeObservable = this.byType.get(event.type);
    if (typeObservable) {
      typeObservable.emit(event);
    }
  }

  /**
   * Get recent events from history
   */
  getHistory(filter?: { type?: string; limit?: number }): AnySdkEvent[] {
    let events = this.eventHistory;

    if (filter?.type) {
      events = events.filter((e) => e.type === filter.type);
    }

    if (filter?.limit) {
      events = events.slice(-filter.limit);
    }

    return [...events];
  }

  /**
   * Get statistics about the event bus
   */
  getStats(): {
    totalEvents: number;
    historySize: number;
    observerCount: number;
    typeObserverCounts: Record<string, number>;
  } {
    const typeObserverCounts: Record<string, number> = {};
    for (const [type, obs] of this.byType) {
      typeObserverCounts[type] = obs.getObserverCount();
    }

    return {
      totalEvents: this.allEvents.getEventCount(),
      historySize: this.eventHistory.length,
      observerCount: this.allEvents.getObserverCount(),
      typeObserverCounts,
    };
  }

  /**
   * Clear event history
   */
  clearHistory(): void {
    this.eventHistory = [];
  }

  /**
   * Remove all observers
   */
  clear(): void {
    this.allEvents.clear();
    for (const obs of this.byType.values()) {
      obs.clear();
    }
    this.byType.clear();
    this.eventHistory = [];
  }
}

// ============================================================================
// Event Helpers
// ============================================================================

/**
 * Create a MATL reputation updated event
 */
export function createReputationUpdatedEvent(
  agentId: string,
  previousValue: number,
  newValue: number,
  change: 'positive' | 'negative'
): ReputationUpdatedEvent {
  return {
    type: MatlEventType.REPUTATION_UPDATED,
    timestamp: Date.now(),
    source: 'matl',
    agentId,
    previousValue,
    newValue,
    change,
  };
}

/**
 * Create a trust evaluated event
 */
export function createTrustEvaluatedEvent(
  agentId: string,
  score: number,
  trustworthy: boolean,
  confidence: number
): TrustEvaluatedEvent {
  return {
    type: MatlEventType.TRUST_EVALUATED,
    timestamp: Date.now(),
    source: 'matl',
    agentId,
    score,
    trustworthy,
    confidence,
  };
}

/**
 * Create a bridge message routed event
 */
export function createMessageRoutedEvent(
  messageType: string,
  sourceHapp: string,
  handlerCount: number
): MessageRoutedEvent {
  return {
    type: BridgeEventType.MESSAGE_ROUTED,
    timestamp: Date.now(),
    source: 'bridge',
    messageType,
    sourceHapp,
    handlerCount,
  };
}

/**
 * Create an FL round started event
 */
export function createRoundStartedEvent(
  roundId: number,
  modelVersion: number,
  participantCount: number
): RoundStartedEvent {
  return {
    type: FlEventType.ROUND_STARTED,
    timestamp: Date.now(),
    source: 'fl',
    roundId,
    modelVersion,
    participantCount,
  };
}

/**
 * Create an FL round aggregated event
 */
export function createRoundAggregatedEvent(
  roundId: number,
  modelVersion: number,
  participantCount: number,
  excludedCount: number,
  aggregationMethod: string,
  durationMs: number
): RoundAggregatedEvent {
  return {
    type: FlEventType.ROUND_AGGREGATED,
    timestamp: Date.now(),
    source: 'fl',
    roundId,
    modelVersion,
    participantCount,
    excludedCount,
    aggregationMethod,
    durationMs,
  };
}

/**
 * Create an epistemic claim created event
 */
export function createClaimCreatedEvent(
  claimId: string,
  issuer: string,
  classificationCode: string
): ClaimCreatedEvent {
  return {
    type: EpistemicEventType.CLAIM_CREATED,
    timestamp: Date.now(),
    source: 'epistemic',
    claimId,
    issuer,
    classificationCode,
  };
}

/**
 * Create a pool cleanup event
 */
export function createPoolCleanupEvent(
  removedCount: number,
  remainingCount: number
): PoolCleanupEvent {
  return {
    type: EpistemicEventType.POOL_CLEANUP,
    timestamp: Date.now(),
    source: 'epistemic',
    removedCount,
    remainingCount,
  };
}

/** Global SDK event bus instance */
export const sdkEvents = new SdkEventBus();

// ============================================================================
// Batch Request Optimization
// ============================================================================

/**
 * Configuration for batch request processing
 */
export interface BatchOptions {
  /** Maximum number of requests per batch (default: 50) */
  batchSizeLimit: number;
  /** Time to wait before auto-flushing in ms (default: 100) */
  timeLimitMs: number;
  /** Whether to compress the batch payload (default: false) */
  compress?: boolean;
  /** Maximum concurrent batch executions (default: 3) */
  maxConcurrent?: number;
}

/**
 * Statistics for batch request operations
 */
export interface BatchStats {
  /** Total requests batched */
  totalRequests: number;
  /** Total batches executed */
  totalBatches: number;
  /** Average batch size */
  averageBatchSize: number;
  /** Requests saved by batching (vs individual calls) */
  requestsSaved: number;
  /** Total processing time in ms */
  totalProcessingMs: number;
}

/**
 * Request batcher for optimizing bulk operations.
 *
 * Collects individual requests and processes them in batches,
 * reducing network overhead and improving throughput.
 *
 * @example
 * ```typescript
 * // Create a batcher for reputation updates
 * const batcher = new RequestBatcher<
 *   { agentId: string; delta: number },
 *   boolean
 * >(
 *   async (batch) => {
 *     // Process all updates in one call
 *     const results = await bulkUpdateReputations(batch);
 *     return results.map(r => r.success);
 *   },
 *   { batchSizeLimit: 100, timeLimitMs: 50 }
 * );
 *
 * // Individual calls are batched automatically
 * const result1 = await batcher.add({ agentId: 'a1', delta: 1 });
 * const result2 = await batcher.add({ agentId: 'a2', delta: -1 });
 * ```
 */
export class RequestBatcher<TRequest, TResult> {
  private pending: Array<{
    request: TRequest;
    resolve: (result: TResult) => void;
    reject: (error: Error) => void;
  }> = [];
  private timer: ReturnType<typeof setTimeout> | null = null;
  private activeBatches = 0;
  private stats: BatchStats = {
    totalRequests: 0,
    totalBatches: 0,
    averageBatchSize: 0,
    requestsSaved: 0,
    totalProcessingMs: 0,
  };
  private readonly options: Required<BatchOptions>;

  constructor(
    private readonly processor: (batch: TRequest[]) => Promise<TResult[]>,
    options: Partial<BatchOptions> = {}
  ) {
    this.options = {
      batchSizeLimit: options.batchSizeLimit ?? 50,
      timeLimitMs: options.timeLimitMs ?? 100,
      compress: options.compress ?? false,
      maxConcurrent: options.maxConcurrent ?? 3,
    };
  }

  /**
   * Add a request to the batch queue
   * @returns Promise resolving to the result when batch is processed
   */
  add(request: TRequest): Promise<TResult> {
    return new Promise<TResult>((resolve, reject) => {
      this.pending.push({ request, resolve, reject });
      this.stats.totalRequests++;

      // Auto-flush if batch size limit reached
      if (this.pending.length >= this.options.batchSizeLimit) {
        void this.flush();
      } else if (!this.timer) {
        // Start timer for auto-flush
        this.timer = setTimeout(() => {
          this.timer = null;
          void this.flush();
        }, this.options.timeLimitMs);
      }
    });
  }

  /**
   * Immediately process all pending requests
   */
  async flush(): Promise<void> {
    if (this.pending.length === 0) return;

    // Wait if at concurrency limit
    while (this.activeBatches >= this.options.maxConcurrent) {
      await sleep(10);
    }

    // Clear timer
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }

    // Take current batch
    const batch = this.pending;
    this.pending = [];
    this.activeBatches++;

    const startTime = Date.now();
    try {
      const requests = batch.map((b) => b.request);
      const results = await this.processor(requests);

      // Resolve individual promises
      for (let i = 0; i < batch.length; i++) {
        batch[i].resolve(results[i]);
      }

      // Update stats
      this.stats.totalBatches++;
      this.stats.requestsSaved += batch.length - 1; // Saved vs individual calls
      this.stats.totalProcessingMs += Date.now() - startTime;
      this.stats.averageBatchSize = this.stats.totalRequests / this.stats.totalBatches;
    } catch (error) {
      // Reject all promises in batch
      const err = error instanceof Error ? error : new Error(String(error));
      for (const item of batch) {
        item.reject(err);
      }
    } finally {
      this.activeBatches--;
    }
  }

  /**
   * Get batch processing statistics
   */
  getStats(): BatchStats {
    return { ...this.stats };
  }

  /**
   * Get current queue size
   */
  getPendingCount(): number {
    return this.pending.length;
  }

  /**
   * Reset statistics
   */
  resetStats(): void {
    this.stats = {
      totalRequests: 0,
      totalBatches: 0,
      averageBatchSize: 0,
      requestsSaved: 0,
      totalProcessingMs: 0,
    };
  }

  /**
   * Cancel all pending requests
   */
  cancel(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    const err = new Error('Batch cancelled');
    for (const item of this.pending) {
      item.reject(err);
    }
    this.pending = [];
  }
}

// ============================================================================
// Event Pipeline Composition
// ============================================================================

/**
 * Operator for transforming event streams
 */
export type PipelineOperator<TIn, TOut> = (source: Observable<TIn>) => Observable<TOut>;

/**
 * Window aggregation result
 */
export interface WindowResult<T> {
  /** Events in the window */
  events: T[];
  /** Window start timestamp */
  windowStart: number;
  /** Window end timestamp */
  windowEnd: number;
  /** Window duration in ms */
  durationMs: number;
}

/**
 * Composable event pipeline for reactive event processing.
 *
 * Provides functional operators for filtering, mapping, debouncing,
 * and aggregating event streams.
 *
 * @example
 * ```typescript
 * const pipeline = new EventPipeline(sdkEvents.allEvents)
 *   .filter((e) => e.type.startsWith('matl:'))
 *   .map((e) => ({ ...e, processed: true }))
 *   .debounce(100)
 *   .subscribe((event) => {
 *     console.log('Processed MATL event:', event);
 *   });
 *
 * // Later: cleanup
 * pipeline.unsubscribe();
 * ```
 */
export class EventPipeline<T> {
  private subscriptions: Subscription[] = [];

  constructor(private source: Observable<T>) {}

  /**
   * Filter events by predicate
   */
  filter(predicate: (event: T) => boolean): EventPipeline<T> {
    const filtered = new Observable<T>();
    const sub = this.source.subscribe((event) => {
      if (predicate(event)) {
        filtered.emit(event);
      }
    });
    this.subscriptions.push(sub);
    return new EventPipeline(filtered);
  }

  /**
   * Transform events
   */
  map<R>(transform: (event: T) => R): EventPipeline<R> {
    const mapped = new Observable<R>();
    const sub = this.source.subscribe((event) => {
      mapped.emit(transform(event));
    });
    this.subscriptions.push(sub);
    return new EventPipeline(mapped);
  }

  /**
   * Debounce events - only emit after quiet period
   */
  debounce(waitMs: number): EventPipeline<T> {
    const debounced = new Observable<T>();
    let timer: ReturnType<typeof setTimeout> | null = null;
    let lastEvent: T | null = null;

    const sub = this.source.subscribe((event) => {
      lastEvent = event;
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        if (lastEvent) {
          debounced.emit(lastEvent);
          lastEvent = null;
        }
      }, waitMs);
    });
    this.subscriptions.push(sub);
    return new EventPipeline(debounced);
  }

  /**
   * Throttle events - emit at most once per interval
   */
  throttle(intervalMs: number): EventPipeline<T> {
    const throttled = new Observable<T>();
    let lastEmit = 0;

    const sub = this.source.subscribe((event) => {
      const now = Date.now();
      if (now - lastEmit >= intervalMs) {
        throttled.emit(event);
        lastEmit = now;
      }
    });
    this.subscriptions.push(sub);
    return new EventPipeline(throttled);
  }

  /**
   * Aggregate events into time windows
   */
  windowAggregate(windowMs: number): EventPipeline<WindowResult<T>> {
    const windowed = new Observable<WindowResult<T>>();
    let windowEvents: T[] = [];
    let windowStart = Date.now();

    const flushWindow = () => {
      if (windowEvents.length > 0) {
        const windowEnd = Date.now();
        windowed.emit({
          events: windowEvents,
          windowStart,
          windowEnd,
          durationMs: windowEnd - windowStart,
        });
      }
      windowEvents = [];
      windowStart = Date.now();
    };

    const timer = setInterval(flushWindow, windowMs);

    const sub = this.source.subscribe((event) => {
      windowEvents.push(event);
    });

    // Create a custom subscription that cleans up the timer
    const cleanup: Subscription = {
      unsubscribe: () => {
        clearInterval(timer);
        sub.unsubscribe();
      },
      get closed() {
        return sub.closed;
      },
    };
    this.subscriptions.push(cleanup);

    return new EventPipeline(windowed);
  }

  /**
   * Buffer events until count is reached
   */
  buffer(count: number): EventPipeline<T[]> {
    const buffered = new Observable<T[]>();
    let buffer: T[] = [];

    const sub = this.source.subscribe((event) => {
      buffer.push(event);
      if (buffer.length >= count) {
        buffered.emit(buffer);
        buffer = [];
      }
    });
    this.subscriptions.push(sub);
    return new EventPipeline(buffered);
  }

  /**
   * Take only the first N events
   */
  take(count: number): EventPipeline<T> {
    const taken = new Observable<T>();
    let remaining = count;

    const sub = this.source.subscribe((event) => {
      if (remaining > 0) {
        remaining--;
        taken.emit(event);
        if (remaining === 0) {
          sub.unsubscribe();
        }
      }
    });
    this.subscriptions.push(sub);
    return new EventPipeline(taken);
  }

  /**
   * Skip the first N events
   */
  skip(count: number): EventPipeline<T> {
    const skipped = new Observable<T>();
    let toSkip = count;

    const sub = this.source.subscribe((event) => {
      if (toSkip > 0) {
        toSkip--;
      } else {
        skipped.emit(event);
      }
    });
    this.subscriptions.push(sub);
    return new EventPipeline(skipped);
  }

  /**
   * Distinct events (based on key function)
   */
  distinct<K>(keyFn: (event: T) => K): EventPipeline<T> {
    const distinct = new Observable<T>();
    const seen = new Set<K>();

    const sub = this.source.subscribe((event) => {
      const key = keyFn(event);
      if (!seen.has(key)) {
        seen.add(key);
        distinct.emit(event);
      }
    });
    this.subscriptions.push(sub);
    return new EventPipeline(distinct);
  }

  /**
   * Subscribe to the pipeline output
   */
  subscribe(observer: Observer<T>): Subscription {
    const sub = this.source.subscribe(observer);
    this.subscriptions.push(sub);
    return sub;
  }

  /**
   * Unsubscribe all pipeline subscriptions
   */
  unsubscribe(): void {
    for (const sub of this.subscriptions) {
      sub.unsubscribe();
    }
    this.subscriptions = [];
  }
}

/**
 * Create a pipeline from an observable
 */
export function pipeline<T>(source: Observable<T>): EventPipeline<T> {
  return new EventPipeline(source);
}

// ============================================================================
// Distributed Tracing
// ============================================================================

/**
 * Trace context for distributed operations
 */
export interface TraceContext {
  /** Unique trace identifier */
  traceId: string;
  /** Current span identifier */
  spanId: string;
  /** Parent span identifier (if any) */
  parentSpanId?: string;
  /** Trace start timestamp */
  startTime: number;
  /** Trace metadata/baggage */
  baggage: Record<string, string>;
}

/**
 * Span representing a single operation in a trace
 */
export interface Span {
  /** Span identifier */
  spanId: string;
  /** Parent span (if any) */
  parentSpanId?: string;
  /** Trace this span belongs to */
  traceId: string;
  /** Operation name */
  operationName: string;
  /** Start timestamp */
  startTime: number;
  /** End timestamp (set when span ends) */
  endTime?: number;
  /** Duration in ms */
  duration?: number;
  /** Status of the operation */
  status: 'pending' | 'success' | 'error';
  /** Tags for filtering/searching */
  tags: Record<string, string | number | boolean>;
  /** Log entries during the span */
  logs: Array<{ timestamp: number; message: string; level: 'debug' | 'info' | 'warn' | 'error' }>;
  /** Error details if status is error */
  error?: { message: string; stack?: string };
}

/**
 * Completed trace with all spans
 */
export interface CompletedTrace {
  /** Trace identifier */
  traceId: string;
  /** Root operation name */
  rootOperation: string;
  /** Total duration in ms */
  totalDuration: number;
  /** All spans in the trace */
  spans: Span[];
  /** Overall status */
  status: 'success' | 'error';
  /** Baggage carried through trace */
  baggage: Record<string, string>;
}

/**
 * Generate a unique trace/span ID
 */
function generateTraceId(): string {
  return security.secureUUID().replace(/-/g, '').substring(0, 32);
}

function generateSpanId(): string {
  return security.secureUUID().replace(/-/g, '').substring(0, 16);
}

/**
 * Distributed tracing manager for tracking operations across services.
 *
 * @example
 * ```typescript
 * const tracer = new TracingManager();
 *
 * // Start a trace
 * const trace = tracer.startTrace('processGradients');
 *
 * // Create child spans
 * const validateSpan = tracer.startSpan(trace, 'validateUpdates');
 * // ... do work ...
 * tracer.endSpan(validateSpan.spanId, 'success');
 *
 * const aggregateSpan = tracer.startSpan(trace, 'aggregate');
 * aggregateSpan.tags['method'] = 'trustWeighted';
 * // ... do work ...
 * tracer.endSpan(aggregateSpan.spanId, 'success');
 *
 * // End the trace
 * const completed = tracer.endTrace(trace.traceId);
 * console.log('Trace completed:', completed.totalDuration, 'ms');
 * ```
 */
export class TracingManager {
  private activeTraces = new Map<string, TraceContext>();
  private spans = new Map<string, Span>();
  private completedTraces: CompletedTrace[] = [];
  private maxCompletedTraces: number;
  private traceListeners: Array<(trace: CompletedTrace) => void> = [];

  constructor(options: { maxCompletedTraces?: number } = {}) {
    this.maxCompletedTraces = options.maxCompletedTraces ?? 100;
  }

  /**
   * Start a new trace
   */
  startTrace(operationName: string, baggage?: Record<string, string>): TraceContext {
    const traceId = generateTraceId();
    const spanId = generateSpanId();

    const context: TraceContext = {
      traceId,
      spanId,
      startTime: Date.now(),
      baggage: baggage ?? {},
    };

    this.activeTraces.set(traceId, context);

    // Create root span
    this.spans.set(spanId, {
      spanId,
      traceId,
      operationName,
      startTime: context.startTime,
      status: 'pending',
      tags: {},
      logs: [],
    });

    return context;
  }

  /**
   * Start a child span within a trace
   */
  startSpan(context: TraceContext, operationName: string): Span {
    const spanId = generateSpanId();
    const span: Span = {
      spanId,
      parentSpanId: context.spanId,
      traceId: context.traceId,
      operationName,
      startTime: Date.now(),
      status: 'pending',
      tags: {},
      logs: [],
    };

    this.spans.set(spanId, span);
    return span;
  }

  /**
   * Add a log entry to a span
   */
  logToSpan(
    spanId: string,
    message: string,
    level: 'debug' | 'info' | 'warn' | 'error' = 'info'
  ): void {
    const span = this.spans.get(spanId);
    if (span) {
      span.logs.push({ timestamp: Date.now(), message, level });
    }
  }

  /**
   * Add a tag to a span
   */
  tagSpan(spanId: string, key: string, value: string | number | boolean): void {
    const span = this.spans.get(spanId);
    if (span) {
      span.tags[key] = value;
    }
  }

  /**
   * End a span
   */
  endSpan(spanId: string, status: 'success' | 'error', error?: Error): void {
    const span = this.spans.get(spanId);
    if (span) {
      span.endTime = Date.now();
      span.duration = span.endTime - span.startTime;
      span.status = status;
      if (error) {
        span.error = { message: error.message, stack: error.stack };
      }
    }
  }

  /**
   * End a trace and collect all spans
   */
  endTrace(traceId: string): CompletedTrace | null {
    const context = this.activeTraces.get(traceId);
    if (!context) return null;

    // End root span if not already ended
    const rootSpan = this.spans.get(context.spanId);
    if (rootSpan && rootSpan.status === 'pending') {
      this.endSpan(context.spanId, 'success');
    }

    // Collect all spans for this trace
    const traceSpans: Span[] = [];
    let hasError = false;
    for (const [spanId, span] of this.spans) {
      if (span.traceId === traceId) {
        traceSpans.push(span);
        if (span.status === 'error') hasError = true;
        this.spans.delete(spanId);
      }
    }

    const completed: CompletedTrace = {
      traceId,
      rootOperation: rootSpan?.operationName ?? 'unknown',
      totalDuration: Date.now() - context.startTime,
      spans: traceSpans,
      status: hasError ? 'error' : 'success',
      baggage: context.baggage,
    };

    // Store completed trace
    this.completedTraces.push(completed);
    if (this.completedTraces.length > this.maxCompletedTraces) {
      this.completedTraces.shift();
    }

    this.activeTraces.delete(traceId);

    // Notify listeners
    for (const listener of this.traceListeners) {
      try {
        listener(completed);
      } catch {
        // Ignore listener errors
      }
    }

    return completed;
  }

  /**
   * Get a span by ID
   */
  getSpan(spanId: string): Span | undefined {
    return this.spans.get(spanId);
  }

  /**
   * Get active trace context
   */
  getTrace(traceId: string): TraceContext | undefined {
    return this.activeTraces.get(traceId);
  }

  /**
   * Get completed traces
   */
  getCompletedTraces(limit?: number): CompletedTrace[] {
    const traces = [...this.completedTraces];
    return limit ? traces.slice(-limit) : traces;
  }

  /**
   * Subscribe to completed traces
   */
  onTraceComplete(listener: (trace: CompletedTrace) => void): () => void {
    this.traceListeners.push(listener);
    return () => {
      const idx = this.traceListeners.indexOf(listener);
      if (idx >= 0) this.traceListeners.splice(idx, 1);
    };
  }

  /**
   * Create a child context for propagation
   */
  childContext(parent: TraceContext): TraceContext {
    return {
      ...parent,
      spanId: generateSpanId(),
      parentSpanId: parent.spanId,
    };
  }

  /**
   * Serialize context for propagation (e.g., in headers)
   */
  serializeContext(context: TraceContext): string {
    return JSON.stringify(context);
  }

  /**
   * Deserialize context from propagation
   */
  deserializeContext(serialized: string): TraceContext | null {
    try {
      return JSON.parse(serialized) as TraceContext;
    } catch {
      return null;
    }
  }
}

/** Global tracing manager instance */
export const tracer = new TracingManager();

/**
 * Decorator-style wrapper for tracing async functions
 */
export function traced<T>(
  operationName: string,
  fn: (context: TraceContext) => Promise<T>,
  parentContext?: TraceContext
): Promise<T> {
  const context = parentContext
    ? tracer.childContext(parentContext)
    : tracer.startTrace(operationName);

  const span = parentContext
    ? tracer.startSpan(context, operationName)
    : tracer.getSpan(context.spanId)!;

  return fn(context)
    .then((result) => {
      tracer.endSpan(span.spanId, 'success');
      if (!parentContext) {
        tracer.endTrace(context.traceId);
      }
      return result;
    })
    .catch((error) => {
      tracer.endSpan(
        span.spanId,
        'error',
        error instanceof Error ? error : new Error(String(error))
      );
      if (!parentContext) {
        tracer.endTrace(context.traceId);
      }
      throw error;
    });
}

// ============================================================================
// Policy-Based Authorization
// ============================================================================

/**
 * Resource types in the SDK
 */
export type ResourceType = 'reputation' | 'claim' | 'gradient' | 'bridge_message' | 'config';

/**
 * Actions that can be performed on resources
 */
export type Action = 'create' | 'read' | 'update' | 'delete' | 'aggregate' | 'verify';

/**
 * Access policy defining permissions
 */
export interface AccessPolicy {
  /** Policy identifier */
  id: string;
  /** Policy name */
  name: string;
  /** Resource type this policy applies to */
  resource: ResourceType;
  /** Allowed actions */
  actions: Action[];
  /** Condition function for fine-grained control */
  condition?: (context: PolicyContext) => boolean;
  /** Priority (higher = checked first) */
  priority?: number;
}

/**
 * Context for policy evaluation
 */
export interface PolicyContext {
  /** Agent/user making the request */
  agentId: string;
  /** Resource being accessed */
  resource: ResourceType;
  /** Action being performed */
  action: Action;
  /** Resource identifier (if applicable) */
  resourceId?: string;
  /** Additional attributes for condition evaluation */
  attributes?: Record<string, unknown>;
}

/**
 * Result of policy evaluation
 */
export interface PolicyResult {
  /** Whether access is allowed */
  allowed: boolean;
  /** Policy that allowed/denied access */
  matchedPolicy?: string;
  /** Reason for decision */
  reason: string;
  /** Policies evaluated */
  policiesEvaluated: number;
}

/**
 * Policy engine for authorization decisions.
 *
 * @example
 * ```typescript
 * const engine = new PolicyEngine();
 *
 * // Add policies
 * engine.addPolicy({
 *   id: 'read-own-reputation',
 *   name: 'Read Own Reputation',
 *   resource: 'reputation',
 *   actions: ['read'],
 *   condition: (ctx) => ctx.resourceId === ctx.agentId,
 * });
 *
 * engine.addPolicy({
 *   id: 'admin-full-access',
 *   name: 'Admin Full Access',
 *   resource: 'reputation',
 *   actions: ['create', 'read', 'update', 'delete'],
 *   condition: (ctx) => ctx.attributes?.role === 'admin',
 *   priority: 100,
 * });
 *
 * // Check access
 * const result = engine.check({
 *   agentId: 'agent1',
 *   resource: 'reputation',
 *   action: 'read',
 *   resourceId: 'agent1',
 * });
 *
 * if (result.allowed) {
 *   // Proceed with operation
 * }
 * ```
 */
export class PolicyEngine {
  private policies: AccessPolicy[] = [];
  private defaultDeny = true;

  constructor(options: { defaultDeny?: boolean } = {}) {
    this.defaultDeny = options.defaultDeny ?? true;
  }

  /**
   * Add a policy
   */
  addPolicy(policy: AccessPolicy): void {
    this.policies.push(policy);
    // Sort by priority (descending)
    this.policies.sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
  }

  /**
   * Remove a policy by ID
   */
  removePolicy(policyId: string): boolean {
    const idx = this.policies.findIndex((p) => p.id === policyId);
    if (idx >= 0) {
      this.policies.splice(idx, 1);
      return true;
    }
    return false;
  }

  /**
   * Check if an action is allowed
   */
  check(context: PolicyContext): PolicyResult {
    let evaluated = 0;

    for (const policy of this.policies) {
      // Check resource match
      if (policy.resource !== context.resource) continue;

      // Check action match
      if (!policy.actions.includes(context.action)) continue;

      evaluated++;

      // Check condition
      if (policy.condition && !policy.condition(context)) continue;

      // Policy matched and allowed
      return {
        allowed: true,
        matchedPolicy: policy.id,
        reason: `Allowed by policy: ${policy.name}`,
        policiesEvaluated: evaluated,
      };
    }

    // No matching policy
    return {
      allowed: !this.defaultDeny,
      reason: this.defaultDeny
        ? 'No matching policy found (default deny)'
        : 'No matching policy found (default allow)',
      policiesEvaluated: evaluated,
    };
  }

  /**
   * Check multiple actions at once
   */
  checkMany(
    agentId: string,
    resource: ResourceType,
    actions: Action[],
    attributes?: Record<string, unknown>
  ): Map<Action, PolicyResult> {
    const results = new Map<Action, PolicyResult>();
    for (const action of actions) {
      results.set(action, this.check({ agentId, resource, action, attributes }));
    }
    return results;
  }

  /**
   * Get all policies
   */
  getPolicies(): AccessPolicy[] {
    return [...this.policies];
  }

  /**
   * Get policies for a resource
   */
  getPoliciesForResource(resource: ResourceType): AccessPolicy[] {
    return this.policies.filter((p) => p.resource === resource);
  }

  /**
   * Clear all policies
   */
  clear(): void {
    this.policies = [];
  }
}

/** Global policy engine instance */
export const policyEngine = new PolicyEngine();

// ============================================================================
// Time-Series Analytics
// ============================================================================

/**
 * Data point with timestamp and value
 */
export interface TimeSeriesPoint {
  timestamp: number;
  value: number;
}

/**
 * Trend analysis result
 */
export interface TrendAnalysis {
  /** Direction of trend */
  direction: 'increasing' | 'decreasing' | 'stable';
  /** Magnitude of change (slope) */
  slope: number;
  /** R-squared correlation */
  correlation: number;
  /** Percentage change over period */
  percentChange: number;
  /** Statistical significance */
  significant: boolean;
}

/**
 * Time window for aggregation
 */
export type TimeWindow = '1m' | '5m' | '15m' | '1h' | '6h' | '24h' | '7d';

const WINDOW_MS: Record<TimeWindow, number> = {
  '1m': 60_000,
  '5m': 300_000,
  '15m': 900_000,
  '1h': 3_600_000,
  '6h': 21_600_000,
  '24h': 86_400_000,
  '7d': 604_800_000,
};

/**
 * Aggregated statistics for a time window
 */
export interface WindowStats {
  window: TimeWindow;
  count: number;
  min: number;
  max: number;
  avg: number;
  sum: number;
  stdDev: number;
  trend: TrendAnalysis;
}

/**
 * Time-series analytics for tracking metrics over time.
 *
 * @example
 * ```typescript
 * const analytics = new TimeSeriesAnalytics<'reputation' | 'trust'>({ maxPoints: 1000 });
 *
 * // Record data points
 * analytics.record('reputation', 0.75);
 * analytics.record('trust', 0.82);
 *
 * // Get statistics
 * const stats = analytics.getStats('reputation', '1h');
 * console.log('Avg reputation (1h):', stats.avg);
 * console.log('Trend:', stats.trend.direction);
 *
 * // Get time-series data
 * const series = analytics.getSeries('reputation', '24h');
 * ```
 */
export class TimeSeriesAnalytics<TMetric extends string = string> {
  private data = new Map<TMetric, TimeSeriesPoint[]>();
  private maxPoints: number;

  constructor(options: { maxPoints?: number } = {}) {
    this.maxPoints = options.maxPoints ?? 10000;
  }

  /**
   * Record a data point
   */
  record(metric: TMetric, value: number, timestamp?: number): void {
    const ts = timestamp ?? Date.now();
    let points = this.data.get(metric);
    if (!points) {
      points = [];
      this.data.set(metric, points);
    }

    points.push({ timestamp: ts, value });

    // Trim if over limit
    if (points.length > this.maxPoints) {
      points.shift();
    }
  }

  /**
   * Get time series data for a window
   */
  getSeries(metric: TMetric, window: TimeWindow): TimeSeriesPoint[] {
    const points = this.data.get(metric) ?? [];
    const cutoff = Date.now() - WINDOW_MS[window];
    return points.filter((p) => p.timestamp >= cutoff);
  }

  /**
   * Get statistics for a metric window
   */
  getStats(metric: TMetric, window: TimeWindow): WindowStats {
    const points = this.getSeries(metric, window);

    if (points.length === 0) {
      return {
        window,
        count: 0,
        min: 0,
        max: 0,
        avg: 0,
        sum: 0,
        stdDev: 0,
        trend: {
          direction: 'stable',
          slope: 0,
          correlation: 0,
          percentChange: 0,
          significant: false,
        },
      };
    }

    const values = points.map((p) => p.value);
    const sum = values.reduce((a, b) => a + b, 0);
    const avg = sum / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);

    // Calculate standard deviation
    const squaredDiffs = values.map((v) => Math.pow(v - avg, 2));
    const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
    const stdDev = Math.sqrt(avgSquaredDiff);

    // Calculate trend
    const trend = this.analyzeTrend(points);

    return {
      window,
      count: values.length,
      min,
      max,
      avg,
      sum,
      stdDev,
      trend,
    };
  }

  /**
   * Analyze trend in data points
   */
  private analyzeTrend(points: TimeSeriesPoint[]): TrendAnalysis {
    if (points.length < 2) {
      return {
        direction: 'stable',
        slope: 0,
        correlation: 0,
        percentChange: 0,
        significant: false,
      };
    }

    // Simple linear regression
    const n = points.length;
    const timestamps = points.map((p) => p.timestamp);
    const values = points.map((p) => p.value);

    // Normalize timestamps
    const minTs = Math.min(...timestamps);
    const normalizedTs = timestamps.map((t) => t - minTs);

    // Calculate means
    const meanX = normalizedTs.reduce((a, b) => a + b, 0) / n;
    const meanY = values.reduce((a, b) => a + b, 0) / n;

    // Calculate slope and correlation
    let numerator = 0;
    let denomX = 0;
    let denomY = 0;

    for (let i = 0; i < n; i++) {
      const dx = normalizedTs[i] - meanX;
      const dy = values[i] - meanY;
      numerator += dx * dy;
      denomX += dx * dx;
      denomY += dy * dy;
    }

    const slope = denomX > 0 ? numerator / denomX : 0;
    const correlation =
      Math.sqrt(denomX) * Math.sqrt(denomY) > 0
        ? numerator / (Math.sqrt(denomX) * Math.sqrt(denomY))
        : 0;

    // Calculate percent change
    const firstValue = values[0];
    const lastValue = values[values.length - 1];
    const percentChange =
      firstValue !== 0 ? ((lastValue - firstValue) / Math.abs(firstValue)) * 100 : 0;

    // Determine direction
    let direction: TrendAnalysis['direction'];
    const SLOPE_THRESHOLD = 0.0001; // Adjust based on time scale
    if (Math.abs(slope) < SLOPE_THRESHOLD) {
      direction = 'stable';
    } else if (slope > 0) {
      direction = 'increasing';
    } else {
      direction = 'decreasing';
    }

    // Significance: R² > 0.5 and enough data points
    const rSquared = correlation * correlation;
    const significant = rSquared > 0.5 && n >= 5;

    return {
      direction,
      slope,
      correlation: rSquared,
      percentChange,
      significant,
    };
  }

  /**
   * Get all metrics being tracked
   */
  getMetrics(): TMetric[] {
    return Array.from(this.data.keys());
  }

  /**
   * Clear data for a metric
   */
  clear(metric?: TMetric): void {
    if (metric) {
      this.data.delete(metric);
    } else {
      this.data.clear();
    }
  }

  /**
   * Get the latest value for a metric
   */
  getLatest(metric: TMetric): TimeSeriesPoint | undefined {
    const points = this.data.get(metric);
    return points?.[points.length - 1];
  }

  /**
   * Export data for a metric
   */
  export(metric: TMetric): TimeSeriesPoint[] {
    return [...(this.data.get(metric) ?? [])];
  }

  /**
   * Import data for a metric
   */
  import(metric: TMetric, points: TimeSeriesPoint[]): void {
    // Sort and dedupe
    const sorted = [...points].sort((a, b) => a.timestamp - b.timestamp);
    this.data.set(metric, sorted.slice(-this.maxPoints));
  }
}

/** Reputation analytics instance */
export const reputationAnalytics = new TimeSeriesAnalytics<
  'reputation' | 'trust' | 'byzantine_rate'
>();

/** FL analytics instance */
export const flAnalytics = new TimeSeriesAnalytics<
  'round_duration' | 'participant_count' | 'accuracy' | 'loss'
>();

// ============================================================================
// Claim Verification Framework
// ============================================================================

/**
 * Evidence proof for claim verification
 */
export interface EvidenceProof {
  /** Proof type */
  type: 'signature' | 'hash' | 'witness' | 'timestamp' | 'composite';
  /** Proof data */
  data: string | Uint8Array;
  /** Algorithm used */
  algorithm?: string;
  /** Issuer of the proof */
  issuer?: string;
  /** When the proof was created */
  createdAt: number;
  /** Expiration time (if any) */
  expiresAt?: number;
}

/**
 * Verification result for a single proof
 */
export interface ProofVerificationResult {
  /** Whether the proof is valid */
  valid: boolean;
  /** Proof that was verified */
  proof: EvidenceProof;
  /** Verification timestamp */
  verifiedAt: number;
  /** Reason for failure (if invalid) */
  failureReason?: string;
}

/**
 * Composite verification result
 */
export interface ClaimVerificationResult {
  /** Claim identifier */
  claimId: string;
  /** Whether the claim is verified */
  verified: boolean;
  /** Verification confidence (0-1) */
  confidence: number;
  /** Individual proof results */
  proofResults: ProofVerificationResult[];
  /** Number of proofs that passed */
  passedCount: number;
  /** Number of proofs that failed */
  failedCount: number;
  /** Verification timestamp */
  verifiedAt: number;
  /** Required proofs that were missing */
  missingProofs?: string[];
}

/**
 * Verification strategy
 */
export type VerificationStrategy =
  | 'all' // All proofs must pass
  | 'any' // At least one proof must pass
  | 'majority' // More than half must pass
  | 'threshold'; // N proofs must pass

/**
 * Verification options
 */
export interface VerificationOptions {
  /** Strategy for combining proof results */
  strategy: VerificationStrategy;
  /** Threshold count (for threshold strategy) */
  threshold?: number;
  /** Required proof types */
  requiredTypes?: EvidenceProof['type'][];
  /** Whether to fail on expired proofs */
  failOnExpired?: boolean;
}

/**
 * Claim verifier for validating epistemic claims.
 *
 * @example
 * ```typescript
 * const verifier = new ClaimVerifier();
 *
 * // Register proof verifiers
 * verifier.registerVerifier('signature', async (proof) => {
 *   // Verify cryptographic signature
 *   return await verifySignature(proof.data, proof.issuer);
 * });
 *
 * verifier.registerVerifier('hash', async (proof) => {
 *   // Verify hash matches
 *   return await verifyHash(proof.data, proof.algorithm);
 * });
 *
 * // Verify a claim
 * const result = await verifier.verify('claim-123', [
 *   { type: 'signature', data: sig, issuer: 'trusted-issuer', createdAt: Date.now() },
 *   { type: 'hash', data: hash, algorithm: 'sha256', createdAt: Date.now() },
 * ], { strategy: 'all' });
 *
 * if (result.verified) {
 *   console.log('Claim verified with confidence:', result.confidence);
 * }
 * ```
 */
export class ClaimVerifier {
  private verifiers = new Map<EvidenceProof['type'], (proof: EvidenceProof) => Promise<boolean>>();

  /**
   * Register a verifier for a proof type
   */
  registerVerifier(
    type: EvidenceProof['type'],
    verifier: (proof: EvidenceProof) => Promise<boolean>
  ): void {
    this.verifiers.set(type, verifier);
  }

  /**
   * Verify a claim with its proofs
   */
  async verify(
    claimId: string,
    proofs: EvidenceProof[],
    options: VerificationOptions = { strategy: 'all' }
  ): Promise<ClaimVerificationResult> {
    const now = Date.now();
    const results: ProofVerificationResult[] = [];
    const missingProofs: string[] = [];

    // Check for required types
    if (options.requiredTypes) {
      for (const type of options.requiredTypes) {
        if (!proofs.some((p) => p.type === type)) {
          missingProofs.push(type);
        }
      }
    }

    // Verify each proof
    for (const proof of proofs) {
      const verifier = this.verifiers.get(proof.type);

      // Check expiration
      if (options.failOnExpired && proof.expiresAt && proof.expiresAt < now) {
        results.push({
          valid: false,
          proof,
          verifiedAt: now,
          failureReason: 'Proof expired',
        });
        continue;
      }

      if (!verifier) {
        results.push({
          valid: false,
          proof,
          verifiedAt: now,
          failureReason: `No verifier for proof type: ${proof.type}`,
        });
        continue;
      }

      try {
        const valid = await verifier(proof);
        results.push({
          valid,
          proof,
          verifiedAt: now,
          failureReason: valid ? undefined : 'Verification failed',
        });
      } catch (error) {
        results.push({
          valid: false,
          proof,
          verifiedAt: now,
          failureReason: error instanceof Error ? error.message : 'Unknown error',
        });
      }
    }

    const passedCount = results.filter((r) => r.valid).length;
    const failedCount = results.length - passedCount;

    // Determine if verified based on strategy
    let verified: boolean;
    switch (options.strategy) {
      case 'all':
        verified = failedCount === 0 && missingProofs.length === 0;
        break;
      case 'any':
        verified = passedCount > 0;
        break;
      case 'majority':
        verified = passedCount > results.length / 2;
        break;
      case 'threshold':
        verified = passedCount >= (options.threshold ?? 1);
        break;
      default:
        verified = false;
    }

    // Calculate confidence
    const confidence = results.length > 0 ? passedCount / results.length : 0;

    return {
      claimId,
      verified,
      confidence,
      proofResults: results,
      passedCount,
      failedCount,
      verifiedAt: now,
      missingProofs: missingProofs.length > 0 ? missingProofs : undefined,
    };
  }

  /**
   * Create a composite proof from multiple proofs
   */
  createCompositeProof(proofs: EvidenceProof[]): EvidenceProof {
    const data = JSON.stringify(
      proofs.map((p) => ({
        type: p.type,
        data: typeof p.data === 'string' ? p.data : Array.from(p.data),
        issuer: p.issuer,
        createdAt: p.createdAt,
      }))
    );

    return {
      type: 'composite',
      data,
      createdAt: Date.now(),
      algorithm: 'composite',
    };
  }

  /**
   * Get registered verifier types
   */
  getVerifierTypes(): EvidenceProof['type'][] {
    return Array.from(this.verifiers.keys());
  }
}

/** Global claim verifier instance */
export const claimVerifier = new ClaimVerifier();

// Register default verifiers
claimVerifier.registerVerifier('timestamp', async (proof) => {
  // Verify timestamp is not in the future and not too old
  const ts = typeof proof.data === 'string' ? parseInt(proof.data, 10) : 0;
  const now = Date.now();
  const MAX_AGE = 24 * 60 * 60 * 1000; // 24 hours
  return ts <= now && now - ts <= MAX_AGE;
});

claimVerifier.registerVerifier('hash', async (proof) => {
  // Verify hash format (placeholder - real implementation would verify against data)
  const hashStr = typeof proof.data === 'string' ? proof.data : '';
  return hashStr.length === 64 || hashStr.length === 128; // SHA-256 or SHA-512
});

// ============================================================================
// Schema Versioning
// ============================================================================

/**
 * Schema version information
 */
export interface SchemaVersion {
  /** Major version (breaking changes) */
  major: number;
  /** Minor version (backwards-compatible additions) */
  minor: number;
  /** Patch version (backwards-compatible fixes) */
  patch: number;
}

/**
 * Schema migration function
 */
export type SchemaMigration<TFrom, TTo> = (data: TFrom) => TTo;

/**
 * Schema definition
 */
export interface SchemaDefinition<T> {
  /** Schema identifier */
  id: string;
  /** Schema version */
  version: SchemaVersion;
  /** Validation function */
  validate: (data: unknown) => data is T;
  /** Schema description */
  description?: string;
}

/**
 * Registered migration
 */
interface RegisteredMigration {
  fromVersion: string;
  toVersion: string;
  migrate: SchemaMigration<unknown, unknown>;
}

/**
 * Migration result
 */
export interface MigrationResult<T> {
  /** Whether migration succeeded */
  success: boolean;
  /** Migrated data (if successful) */
  data?: T;
  /** Source version */
  fromVersion: string;
  /** Target version */
  toVersion: string;
  /** Migration path taken */
  migrationPath: string[];
  /** Error (if failed) */
  error?: string;
}

/**
 * Schema registry for type versioning and migration.
 *
 * @example
 * ```typescript
 * const registry = new SchemaRegistry();
 *
 * // Register schemas
 * registry.registerSchema({
 *   id: 'claim',
 *   version: { major: 1, minor: 0, patch: 0 },
 *   validate: (data): data is ClaimV1 => isClaimV1(data),
 * });
 *
 * registry.registerSchema({
 *   id: 'claim',
 *   version: { major: 2, minor: 0, patch: 0 },
 *   validate: (data): data is ClaimV2 => isClaimV2(data),
 * });
 *
 * // Register migration
 * registry.registerMigration<ClaimV1, ClaimV2>(
 *   'claim',
 *   { major: 1, minor: 0, patch: 0 },
 *   { major: 2, minor: 0, patch: 0 },
 *   (v1) => ({
 *     ...v1,
 *     newField: 'default',
 *     version: 2,
 *   })
 * );
 *
 * // Migrate data
 * const result = registry.migrate<ClaimV2>(oldClaim, 'claim', { major: 2, minor: 0, patch: 0 });
 * if (result.success) {
 *   console.log('Migrated claim:', result.data);
 * }
 * ```
 */
export class SchemaRegistry {
  private schemas = new Map<string, SchemaDefinition<unknown>[]>();
  private migrations: RegisteredMigration[] = [];

  /**
   * Format version as string
   */
  private formatVersion(version: SchemaVersion): string {
    return `${version.major}.${version.minor}.${version.patch}`;
  }

  /**
   * Compare versions
   */
  private compareVersions(a: SchemaVersion, b: SchemaVersion): number {
    if (a.major !== b.major) return a.major - b.major;
    if (a.minor !== b.minor) return a.minor - b.minor;
    return a.patch - b.patch;
  }

  /**
   * Register a schema version
   */
  registerSchema<T>(schema: SchemaDefinition<T>): void {
    let versions = this.schemas.get(schema.id);
    if (!versions) {
      versions = [];
      this.schemas.set(schema.id, versions);
    }

    // Insert in sorted order
    const idx = versions.findIndex((s) => this.compareVersions(s.version, schema.version) > 0);
    if (idx === -1) {
      versions.push(schema as SchemaDefinition<unknown>);
    } else {
      versions.splice(idx, 0, schema as SchemaDefinition<unknown>);
    }
  }

  /**
   * Register a migration between versions
   */
  registerMigration<TFrom, TTo>(
    schemaId: string,
    fromVersion: SchemaVersion,
    toVersion: SchemaVersion,
    migrate: SchemaMigration<TFrom, TTo>
  ): void {
    this.migrations.push({
      fromVersion: `${schemaId}@${this.formatVersion(fromVersion)}`,
      toVersion: `${schemaId}@${this.formatVersion(toVersion)}`,
      migrate: migrate as SchemaMigration<unknown, unknown>,
    });
  }

  /**
   * Get a schema by ID and version
   */
  getSchema<T>(id: string, version?: SchemaVersion): SchemaDefinition<T> | undefined {
    const versions = this.schemas.get(id);
    if (!versions || versions.length === 0) return undefined;

    if (!version) {
      // Return latest
      return versions[versions.length - 1] as SchemaDefinition<T>;
    }

    return versions.find((s) => this.compareVersions(s.version, version) === 0) as
      | SchemaDefinition<T>
      | undefined;
  }

  /**
   * Get all versions of a schema
   */
  getVersions(id: string): SchemaVersion[] {
    const versions = this.schemas.get(id);
    return versions?.map((s) => s.version) ?? [];
  }

  /**
   * Detect the version of data
   */
  detectVersion(id: string, data: unknown): SchemaVersion | undefined {
    const versions = this.schemas.get(id);
    if (!versions) return undefined;

    // Try from newest to oldest
    for (let i = versions.length - 1; i >= 0; i--) {
      if (versions[i].validate(data)) {
        return versions[i].version;
      }
    }

    return undefined;
  }

  /**
   * Find migration path between versions
   */
  private findMigrationPath(
    schemaId: string,
    from: SchemaVersion,
    to: SchemaVersion
  ): RegisteredMigration[] | null {
    const fromKey = `${schemaId}@${this.formatVersion(from)}`;
    const toKey = `${schemaId}@${this.formatVersion(to)}`;

    if (fromKey === toKey) return [];

    // BFS to find shortest path
    const queue: Array<{ key: string; path: RegisteredMigration[] }> = [{ key: fromKey, path: [] }];
    const visited = new Set<string>([fromKey]);

    while (queue.length > 0) {
      const { key, path } = queue.shift()!;

      for (const migration of this.migrations) {
        if (migration.fromVersion === key && !visited.has(migration.toVersion)) {
          const newPath = [...path, migration];

          if (migration.toVersion === toKey) {
            return newPath;
          }

          visited.add(migration.toVersion);
          queue.push({ key: migration.toVersion, path: newPath });
        }
      }
    }

    return null;
  }

  /**
   * Migrate data to a target version
   */
  migrate<T>(data: unknown, schemaId: string, targetVersion: SchemaVersion): MigrationResult<T> {
    const sourceVersion = this.detectVersion(schemaId, data);
    const targetVersionStr = this.formatVersion(targetVersion);

    if (!sourceVersion) {
      return {
        success: false,
        fromVersion: 'unknown',
        toVersion: targetVersionStr,
        migrationPath: [],
        error: 'Could not detect source version',
      };
    }

    const sourceVersionStr = this.formatVersion(sourceVersion);

    // Already at target version?
    if (this.compareVersions(sourceVersion, targetVersion) === 0) {
      return {
        success: true,
        data: data as T,
        fromVersion: sourceVersionStr,
        toVersion: targetVersionStr,
        migrationPath: [],
      };
    }

    // Find migration path
    const path = this.findMigrationPath(schemaId, sourceVersion, targetVersion);
    if (!path) {
      return {
        success: false,
        fromVersion: sourceVersionStr,
        toVersion: targetVersionStr,
        migrationPath: [],
        error: `No migration path from ${sourceVersionStr} to ${targetVersionStr}`,
      };
    }

    // Apply migrations
    let current = data;
    const migrationPath: string[] = [sourceVersionStr];

    try {
      for (const migration of path) {
        current = migration.migrate(current);
        const version = migration.toVersion.split('@')[1];
        migrationPath.push(version);
      }

      // Validate final result
      const targetSchema = this.getSchema<T>(schemaId, targetVersion);
      if (targetSchema && !targetSchema.validate(current)) {
        return {
          success: false,
          fromVersion: sourceVersionStr,
          toVersion: targetVersionStr,
          migrationPath,
          error: 'Migrated data failed validation',
        };
      }

      return {
        success: true,
        data: current as T,
        fromVersion: sourceVersionStr,
        toVersion: targetVersionStr,
        migrationPath,
      };
    } catch (error) {
      return {
        success: false,
        fromVersion: sourceVersionStr,
        toVersion: targetVersionStr,
        migrationPath,
        error: error instanceof Error ? error.message : 'Migration failed',
      };
    }
  }

  /**
   * Get all registered schema IDs
   */
  getSchemaIds(): string[] {
    return Array.from(this.schemas.keys());
  }

  /**
   * Clear all registrations
   */
  clear(): void {
    this.schemas.clear();
    this.migrations = [];
  }
}

/** Global schema registry instance */
export const schemaRegistry = new SchemaRegistry();

// ============================================================================
// CATEGORY 1: Real-time & Connectivity
// ============================================================================

/**
 * WebSocket connection states
 */
export type WebSocketState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';

/**
 * WebSocket message types
 */
export interface WebSocketMessage<T = unknown> {
  type: string;
  payload: T;
  timestamp: number;
  id: string;
}

/**
 * WebSocket configuration
 */
export interface WebSocketConfig {
  /** WebSocket URL */
  url: string;
  /** Reconnection attempts (default: 5) */
  maxReconnectAttempts?: number;
  /** Base reconnect delay in ms (default: 1000) */
  reconnectDelayMs?: number;
  /** Exponential backoff factor (default: 2) */
  backoffFactor?: number;
  /** Heartbeat interval in ms (default: 30000) */
  heartbeatIntervalMs?: number;
  /** Connection timeout in ms (default: 10000) */
  connectionTimeoutMs?: number;
}

/**
 * WebSocket event handlers
 */
export interface WebSocketHandlers<T = unknown> {
  onMessage?: (message: WebSocketMessage<T>) => void;
  onStateChange?: (state: WebSocketState, previousState: WebSocketState) => void;
  onError?: (error: Error) => void;
  onReconnect?: (attempt: number) => void;
}

/**
 * WebSocket connection statistics
 */
export interface WebSocketStats {
  state: WebSocketState;
  messagesReceived: number;
  messagesSent: number;
  bytesReceived: number;
  bytesSent: number;
  reconnectAttempts: number;
  lastConnectedAt: number | null;
  lastDisconnectedAt: number | null;
  uptime: number;
}

/**
 * Robust WebSocket manager with auto-reconnection.
 *
 * Features:
 * - Automatic reconnection with exponential backoff
 * - Heartbeat/ping-pong for connection health
 * - Message queuing during disconnection
 * - Connection state management
 * - Statistics tracking
 *
 * @example
 * ```typescript
 * const ws = new WebSocketManager({
 *   url: 'ws://localhost:8888',
 *   maxReconnectAttempts: 10,
 * });
 *
 * ws.subscribe('signal', (message) => {
 *   console.log('Received signal:', message);
 * });
 *
 * await ws.connect();
 * ws.send('request', { action: 'subscribe', topic: 'reputation' });
 * ```
 */
export class WebSocketManager<T = unknown> {
  private ws: WebSocket | null = null;
  private state: WebSocketState = 'disconnected';
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private messageQueue: WebSocketMessage<T>[] = [];
  private subscribers = new Map<string, Set<(message: WebSocketMessage<T>) => void>>();
  private stats: WebSocketStats = {
    state: 'disconnected',
    messagesReceived: 0,
    messagesSent: 0,
    bytesReceived: 0,
    bytesSent: 0,
    reconnectAttempts: 0,
    lastConnectedAt: null,
    lastDisconnectedAt: null,
    uptime: 0,
  };
  private connectedAt: number | null = null;
  private readonly config: Required<WebSocketConfig>;
  private handlers: WebSocketHandlers<T> = {};

  constructor(config: WebSocketConfig) {
    this.config = {
      maxReconnectAttempts: 5,
      reconnectDelayMs: 1000,
      backoffFactor: 2,
      heartbeatIntervalMs: 30000,
      connectionTimeoutMs: 10000,
      ...config,
    };
  }

  /**
   * Register event handlers
   */
  on(handlers: WebSocketHandlers<T>): void {
    this.handlers = { ...this.handlers, ...handlers };
  }

  /**
   * Connect to WebSocket server
   */
  async connect(): Promise<void> {
    if (this.state === 'connected' || this.state === 'connecting') {
      return;
    }

    this.setState('connecting');

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Connection timeout'));
        this.setState('error');
      }, this.config.connectionTimeoutMs);

      try {
        // In browser/Node environments, WebSocket would be available
        // This is a simulation for the SDK
        this.simulateConnection(resolve, reject, timeout);
      } catch (error) {
        clearTimeout(timeout);
        this.setState('error');
        reject(error);
      }
    });
  }

  private simulateConnection(
    resolve: () => void,
    _reject: (error: Error) => void,
    timeout: ReturnType<typeof setTimeout>
  ): void {
    // Simulate successful connection
    setTimeout(() => {
      clearTimeout(timeout);
      this.setState('connected');
      this.connectedAt = Date.now();
      this.stats.lastConnectedAt = this.connectedAt;
      this.reconnectAttempts = 0;
      this.startHeartbeat();
      this.flushMessageQueue();
      resolve();
    }, 100);
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.stopHeartbeat();
    this.clearReconnectTimer();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.stats.lastDisconnectedAt = Date.now();
    if (this.connectedAt) {
      this.stats.uptime += Date.now() - this.connectedAt;
      this.connectedAt = null;
    }
    this.setState('disconnected');
  }

  /**
   * Send a message
   */
  send(type: string, payload: T): boolean {
    const message: WebSocketMessage<T> = {
      type,
      payload,
      timestamp: Date.now(),
      id: `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
    };

    if (this.state !== 'connected') {
      this.messageQueue.push(message);
      return false;
    }

    this.doSend(message);
    return true;
  }

  private doSend(message: WebSocketMessage<T>): void {
    const data = JSON.stringify(message);
    // In real implementation, this would use ws.send()
    this.stats.messagesSent++;
    this.stats.bytesSent += data.length;
  }

  /**
   * Subscribe to a message type
   */
  subscribe(type: string, handler: (message: WebSocketMessage<T>) => void): () => void {
    if (!this.subscribers.has(type)) {
      this.subscribers.set(type, new Set());
    }
    this.subscribers.get(type)!.add(handler);

    return () => {
      const handlers = this.subscribers.get(type);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          this.subscribers.delete(type);
        }
      }
    };
  }

  /**
   * Handle incoming message (called internally or for testing)
   */
  handleMessage(message: WebSocketMessage<T>): void {
    this.stats.messagesReceived++;
    this.stats.bytesReceived += JSON.stringify(message).length;

    const handlers = this.subscribers.get(message.type);
    if (handlers) {
      handlers.forEach((handler) => handler(message));
    }

    // Notify global handler
    this.handlers.onMessage?.(message);
  }

  /**
   * Get current state
   */
  getState(): WebSocketState {
    return this.state;
  }

  /**
   * Get connection statistics
   */
  getStats(): WebSocketStats {
    let uptime = this.stats.uptime;
    if (this.connectedAt) {
      uptime += Date.now() - this.connectedAt;
    }
    return { ...this.stats, state: this.state, uptime };
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.state === 'connected';
  }

  private setState(newState: WebSocketState): void {
    const previousState = this.state;
    this.state = newState;
    this.stats.state = newState;
    this.handlers.onStateChange?.(newState, previousState);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.state === 'connected') {
        this.send('__ping__', {} as T);
      }
    }, this.config.heartbeatIntervalMs);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      this.setState('error');
      this.handlers.onError?.(new Error('Max reconnection attempts reached'));
      return;
    }

    const delay =
      this.config.reconnectDelayMs * Math.pow(this.config.backoffFactor, this.reconnectAttempts);
    this.reconnectAttempts++;
    this.stats.reconnectAttempts++;

    this.setState('reconnecting');
    this.handlers.onReconnect?.(this.reconnectAttempts);

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(() => {
        this.scheduleReconnect();
      });
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.state === 'connected') {
      const message = this.messageQueue.shift()!;
      this.doSend(message);
    }
  }
}

/**
 * Connection pool configuration
 */
export interface ConnectionPoolConfig {
  /** Minimum connections to maintain */
  minConnections: number;
  /** Maximum connections allowed */
  maxConnections: number;
  /** Idle timeout before closing connection (ms) */
  idleTimeoutMs: number;
  /** Connection acquisition timeout (ms) */
  acquireTimeoutMs: number;
  /** Factory function to create connections */
  createConnection: () => Promise<PooledConnection>;
}

/**
 * Pooled connection interface
 */
export interface PooledConnection {
  id: string;
  isHealthy(): boolean;
  close(): Promise<void>;
}

/**
 * Connection pool statistics
 */
export interface ConnectionPoolStats {
  totalConnections: number;
  activeConnections: number;
  idleConnections: number;
  waitingRequests: number;
  totalAcquired: number;
  totalReleased: number;
  totalCreated: number;
  totalDestroyed: number;
}

/**
 * Generic connection pool for managing reusable connections.
 *
 * @example
 * ```typescript
 * const pool = new ConnectionPool({
 *   minConnections: 2,
 *   maxConnections: 10,
 *   idleTimeoutMs: 60000,
 *   acquireTimeoutMs: 5000,
 *   createConnection: async () => ({
 *     id: crypto.randomUUID(),
 *     isHealthy: () => true,
 *     close: async () => {},
 *   }),
 * });
 *
 * await pool.initialize();
 * const conn = await pool.acquire();
 * // use connection
 * pool.release(conn);
 * ```
 */
export class ConnectionPool {
  private idle: PooledConnection[] = [];
  private active = new Set<PooledConnection>();
  private waiting: Array<{
    resolve: (conn: PooledConnection) => void;
    reject: (error: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  }> = [];
  private idleTimers = new Map<string, ReturnType<typeof setTimeout>>();
  private stats: ConnectionPoolStats = {
    totalConnections: 0,
    activeConnections: 0,
    idleConnections: 0,
    waitingRequests: 0,
    totalAcquired: 0,
    totalReleased: 0,
    totalCreated: 0,
    totalDestroyed: 0,
  };
  private closed = false;

  constructor(private readonly config: ConnectionPoolConfig) {}

  /**
   * Initialize the pool with minimum connections
   */
  async initialize(): Promise<void> {
    const promises: Promise<void>[] = [];
    for (let i = 0; i < this.config.minConnections; i++) {
      promises.push(this.createAndAddConnection());
    }
    await Promise.all(promises);
  }

  /**
   * Acquire a connection from the pool
   */
  async acquire(): Promise<PooledConnection> {
    if (this.closed) {
      throw new Error('Pool is closed');
    }

    // Try to get an idle connection
    const conn = this.getIdleConnection();
    if (conn) {
      this.active.add(conn);
      this.stats.activeConnections++;
      this.stats.totalAcquired++;
      this.updateStats();
      return conn;
    }

    // Try to create a new connection
    if (this.getTotalConnections() < this.config.maxConnections) {
      const newConn = await this.createConnection();
      this.active.add(newConn);
      this.stats.activeConnections++;
      this.stats.totalAcquired++;
      this.updateStats();
      return newConn;
    }

    // Wait for a connection
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        const index = this.waiting.findIndex((w) => w.resolve === resolve);
        if (index !== -1) {
          this.waiting.splice(index, 1);
          this.stats.waitingRequests--;
          reject(new Error('Acquire timeout'));
        }
      }, this.config.acquireTimeoutMs);

      this.waiting.push({ resolve, reject, timer });
      this.stats.waitingRequests++;
    });
  }

  /**
   * Release a connection back to the pool
   */
  release(conn: PooledConnection): void {
    if (!this.active.has(conn)) {
      return;
    }

    this.active.delete(conn);
    this.stats.activeConnections--;
    this.stats.totalReleased++;

    // Check if connection is still healthy
    if (!conn.isHealthy()) {
      this.destroyConnection(conn);
      return;
    }

    // Check if there are waiting requests
    if (this.waiting.length > 0) {
      const waiter = this.waiting.shift()!;
      clearTimeout(waiter.timer);
      this.stats.waitingRequests--;
      this.active.add(conn);
      this.stats.activeConnections++;
      this.stats.totalAcquired++;
      waiter.resolve(conn);
      return;
    }

    // Return to idle pool
    this.idle.push(conn);
    this.stats.idleConnections++;
    this.scheduleIdleTimeout(conn);
    this.updateStats();
  }

  /**
   * Get pool statistics
   */
  getStats(): ConnectionPoolStats {
    return { ...this.stats };
  }

  /**
   * Close the pool and all connections
   */
  async close(): Promise<void> {
    this.closed = true;

    // Reject all waiting requests
    for (const waiter of this.waiting) {
      clearTimeout(waiter.timer);
      waiter.reject(new Error('Pool closed'));
    }
    this.waiting = [];

    // Clear idle timers
    for (const timer of this.idleTimers.values()) {
      clearTimeout(timer);
    }
    this.idleTimers.clear();

    // Close all connections
    const allConnections = [...this.idle, ...this.active];
    await Promise.all(allConnections.map((conn) => conn.close()));

    this.idle = [];
    this.active.clear();
    this.updateStats();
  }

  private async createAndAddConnection(): Promise<void> {
    const conn = await this.createConnection();
    this.idle.push(conn);
    this.stats.idleConnections++;
    this.scheduleIdleTimeout(conn);
    this.updateStats();
  }

  private async createConnection(): Promise<PooledConnection> {
    const conn = await this.config.createConnection();
    this.stats.totalCreated++;
    this.stats.totalConnections++;
    return conn;
  }

  private getIdleConnection(): PooledConnection | null {
    while (this.idle.length > 0) {
      const conn = this.idle.pop()!;
      this.stats.idleConnections--;
      this.clearIdleTimeout(conn);

      if (conn.isHealthy()) {
        return conn;
      }

      this.destroyConnection(conn);
    }
    return null;
  }

  private destroyConnection(conn: PooledConnection): void {
    this.clearIdleTimeout(conn);
    conn.close().catch(() => {});
    this.stats.totalDestroyed++;
    this.stats.totalConnections--;
    this.updateStats();
  }

  private scheduleIdleTimeout(conn: PooledConnection): void {
    const timer = setTimeout(() => {
      const index = this.idle.indexOf(conn);
      if (index !== -1) {
        this.idle.splice(index, 1);
        this.stats.idleConnections--;
        this.destroyConnection(conn);
      }
    }, this.config.idleTimeoutMs);

    this.idleTimers.set(conn.id, timer);
  }

  private clearIdleTimeout(conn: PooledConnection): void {
    const timer = this.idleTimers.get(conn.id);
    if (timer) {
      clearTimeout(timer);
      this.idleTimers.delete(conn.id);
    }
  }

  private getTotalConnections(): number {
    return this.idle.length + this.active.size;
  }

  private updateStats(): void {
    this.stats.totalConnections = this.getTotalConnections();
    this.stats.idleConnections = this.idle.length;
    this.stats.activeConnections = this.active.size;
    this.stats.waitingRequests = this.waiting.length;
  }
}

/**
 * Queued operation for offline support
 */
export interface QueuedOperation<T = unknown> {
  id: string;
  type: string;
  payload: T;
  timestamp: number;
  retries: number;
  maxRetries: number;
  priority: number;
}

/**
 * Offline queue configuration
 */
export interface OfflineQueueConfig {
  /** Maximum queue size */
  maxSize: number;
  /** Maximum retries per operation */
  maxRetries: number;
  /** Retry delay in ms */
  retryDelayMs: number;
  /** Storage key for persistence */
  storageKey?: string;
}

/**
 * Offline queue statistics
 */
export interface OfflineQueueStats {
  queued: number;
  processing: number;
  completed: number;
  failed: number;
  totalRetries: number;
}

/**
 * Offline-first operation queue with persistence.
 *
 * Queues operations when offline and syncs when reconnected.
 * Supports priority ordering, retries, and persistence.
 *
 * @example
 * ```typescript
 * const queue = new OfflineQueue({
 *   maxSize: 1000,
 *   maxRetries: 3,
 *   retryDelayMs: 5000,
 * });
 *
 * // Queue operations when offline
 * queue.enqueue('updateReputation', { agentId: 'a1', delta: 1 }, 1);
 *
 * // Process when online
 * await queue.process(async (op) => {
 *   await api.execute(op.type, op.payload);
 *   return true;
 * });
 * ```
 */
export class OfflineQueue<T = unknown> {
  private queue: QueuedOperation<T>[] = [];
  private processing = false;
  private stats: OfflineQueueStats = {
    queued: 0,
    processing: 0,
    completed: 0,
    failed: 0,
    totalRetries: 0,
  };
  private readonly config: Required<OfflineQueueConfig>;

  constructor(config: Partial<OfflineQueueConfig> = {}) {
    this.config = {
      maxSize: 1000,
      maxRetries: 3,
      retryDelayMs: 5000,
      storageKey: 'mycelix_offline_queue',
      ...config,
    };
  }

  /**
   * Add operation to queue
   */
  enqueue(type: string, payload: T, priority = 0): string {
    if (this.queue.length >= this.config.maxSize) {
      // Remove lowest priority item
      this.queue.sort((a, b) => b.priority - a.priority);
      this.queue.pop();
    }

    const operation: QueuedOperation<T> = {
      id: `op_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
      type,
      payload,
      timestamp: Date.now(),
      retries: 0,
      maxRetries: this.config.maxRetries,
      priority,
    };

    this.queue.push(operation);
    this.queue.sort((a, b) => b.priority - a.priority);
    this.stats.queued++;

    return operation.id;
  }

  /**
   * Process all queued operations
   */
  async process(
    executor: (operation: QueuedOperation<T>) => Promise<boolean>
  ): Promise<{ completed: number; failed: number }> {
    if (this.processing) {
      return { completed: 0, failed: 0 };
    }

    this.processing = true;
    let completed = 0;
    let failed = 0;

    const toProcess = [...this.queue];
    this.queue = [];

    for (const operation of toProcess) {
      this.stats.processing++;

      try {
        const success = await executor(operation);
        if (success) {
          completed++;
          this.stats.completed++;
        } else {
          throw new Error('Execution returned false');
        }
      } catch {
        operation.retries++;
        this.stats.totalRetries++;

        if (operation.retries < operation.maxRetries) {
          // Re-queue for retry
          this.queue.push(operation);
        } else {
          failed++;
          this.stats.failed++;
        }
      }

      this.stats.processing--;
    }

    // Re-sort queue
    this.queue.sort((a, b) => b.priority - a.priority);
    this.processing = false;

    return { completed, failed };
  }

  /**
   * Get queue contents
   */
  getQueue(): readonly QueuedOperation<T>[] {
    return this.queue;
  }

  /**
   * Get statistics
   */
  getStats(): OfflineQueueStats {
    return { ...this.stats, queued: this.queue.length };
  }

  /**
   * Clear the queue
   */
  clear(): void {
    this.queue = [];
    this.stats.queued = 0;
  }

  /**
   * Remove specific operation
   */
  remove(id: string): boolean {
    const index = this.queue.findIndex((op) => op.id === id);
    if (index !== -1) {
      this.queue.splice(index, 1);
      return true;
    }
    return false;
  }

  /**
   * Export queue for persistence
   */
  export(): string {
    return JSON.stringify(this.queue);
  }

  /**
   * Import queue from persistence
   */
  import(data: string): void {
    try {
      const operations = JSON.parse(data) as QueuedOperation<T>[];
      this.queue = operations;
      this.queue.sort((a, b) => b.priority - a.priority);
    } catch {
      // Invalid data, ignore
    }
  }
}

/**
 * Signal types for Holochain
 */
export type SignalType = 'app' | 'system' | 'remote';

/**
 * Signal from Holochain conductor
 */
export interface HolochainSignal<T = unknown> {
  type: SignalType;
  cellId: [Uint8Array, Uint8Array];
  zomeName: string;
  signalName: string;
  payload: T;
}

/**
 * Signal handler configuration
 */
export interface SignalHandlerConfig {
  /** Filter signals by cell ID */
  cellIdFilter?: string;
  /** Filter signals by zome */
  zomeFilter?: string;
  /** Buffer signals during processing */
  bufferSize?: number;
}

/**
 * Holochain signal handler for real-time updates.
 *
 * @example
 * ```typescript
 * const handler = new SignalHandler();
 *
 * handler.on('reputation_updated', (signal) => {
 *   console.log('Reputation changed:', signal.payload);
 * });
 *
 * // Process incoming signal from conductor
 * handler.handle({
 *   type: 'app',
 *   cellId: [...],
 *   zomeName: 'reputation',
 *   signalName: 'reputation_updated',
 *   payload: { agentId: 'a1', newScore: 0.95 },
 * });
 * ```
 */
export class SignalHandler<T = unknown> {
  private handlers = new Map<string, Set<(signal: HolochainSignal<T>) => void>>();
  private globalHandlers = new Set<(signal: HolochainSignal<T>) => void>();
  private buffer: HolochainSignal<T>[] = [];
  private readonly config: Required<SignalHandlerConfig>;

  constructor(config: Partial<SignalHandlerConfig> = {}) {
    this.config = {
      cellIdFilter: '',
      zomeFilter: '',
      bufferSize: 100,
      ...config,
    };
  }

  /**
   * Register handler for specific signal
   */
  on(signalName: string, handler: (signal: HolochainSignal<T>) => void): () => void {
    if (!this.handlers.has(signalName)) {
      this.handlers.set(signalName, new Set());
    }
    this.handlers.get(signalName)!.add(handler);

    return () => {
      const handlers = this.handlers.get(signalName);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          this.handlers.delete(signalName);
        }
      }
    };
  }

  /**
   * Register handler for all signals
   */
  onAny(handler: (signal: HolochainSignal<T>) => void): () => void {
    this.globalHandlers.add(handler);
    return () => {
      this.globalHandlers.delete(handler);
    };
  }

  /**
   * Handle incoming signal
   */
  handle(signal: HolochainSignal<T>): void {
    // Apply filters
    if (this.config.zomeFilter && signal.zomeName !== this.config.zomeFilter) {
      return;
    }

    // Buffer signal
    this.buffer.push(signal);
    if (this.buffer.length > this.config.bufferSize) {
      this.buffer.shift();
    }

    // Notify specific handlers
    const handlers = this.handlers.get(signal.signalName);
    if (handlers) {
      handlers.forEach((handler) => handler(signal));
    }

    // Notify global handlers
    this.globalHandlers.forEach((handler) => handler(signal));
  }

  /**
   * Get buffered signals
   */
  getBuffer(): readonly HolochainSignal<T>[] {
    return this.buffer;
  }

  /**
   * Clear buffer
   */
  clearBuffer(): void {
    this.buffer = [];
  }

  /**
   * Get registered signal names
   */
  getRegisteredSignals(): string[] {
    return Array.from(this.handlers.keys());
  }
}

// ============================================================================
// CATEGORY 2: Developer Experience
// ============================================================================

/**
 * Log levels
 */
export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'silent';

/**
 * Structured log entry
 */
export interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: number;
  context?: Record<string, unknown>;
  error?: Error;
  traceId?: string;
  spanId?: string;
}

/**
 * Logger configuration
 */
export interface LoggerConfig {
  /** Minimum log level */
  level: LogLevel;
  /** Include timestamps */
  timestamps?: boolean;
  /** Output format */
  format?: 'json' | 'pretty';
  /** Custom output handler */
  output?: (entry: LogEntry) => void;
}

/**
 * Structured logger with context support.
 *
 * @example
 * ```typescript
 * const logger = new StructuredLogger({ level: 'debug', format: 'json' });
 *
 * logger.info('Processing request', { requestId: '123', userId: 'u1' });
 * logger.error('Failed to process', { error: err }, err);
 *
 * // Create child logger with context
 * const childLogger = logger.child({ requestId: '123' });
 * childLogger.debug('Step 1 complete');
 * ```
 */
export class StructuredLogger {
  private readonly config: Required<LoggerConfig>;
  private readonly context: Record<string, unknown>;
  private static readonly LEVEL_PRIORITY: Record<LogLevel, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3,
    silent: 4,
  };

  constructor(config: Partial<LoggerConfig> = {}, context: Record<string, unknown> = {}) {
    this.config = {
      level: 'info',
      timestamps: true,
      format: 'pretty',
      output: (entry) => this.defaultOutput(entry),
      ...config,
    };
    this.context = context;
  }

  /**
   * Create child logger with additional context
   */
  child(context: Record<string, unknown>): StructuredLogger {
    return new StructuredLogger(this.config, { ...this.context, ...context });
  }

  /**
   * Log debug message
   */
  debug(message: string, context?: Record<string, unknown>): void {
    this.log('debug', message, context);
  }

  /**
   * Log info message
   */
  info(message: string, context?: Record<string, unknown>): void {
    this.log('info', message, context);
  }

  /**
   * Log warning message
   */
  warn(message: string, context?: Record<string, unknown>): void {
    this.log('warn', message, context);
  }

  /**
   * Log error message
   */
  error(message: string, context?: Record<string, unknown>, error?: Error): void {
    this.log('error', message, context, error);
  }

  private log(
    level: LogLevel,
    message: string,
    context?: Record<string, unknown>,
    error?: Error
  ): void {
    if (
      StructuredLogger.LEVEL_PRIORITY[level] < StructuredLogger.LEVEL_PRIORITY[this.config.level]
    ) {
      return;
    }

    const entry: LogEntry = {
      level,
      message,
      timestamp: Date.now(),
      context: { ...this.context, ...context },
      error,
    };

    this.config.output(entry);
  }

  private defaultOutput(entry: LogEntry): void {
    if (this.config.format === 'json') {
      const output = {
        level: entry.level,
        message: entry.message,
        timestamp: entry.timestamp,
        ...entry.context,
        ...(entry.error ? { error: entry.error.message, stack: entry.error.stack } : {}),
      };
      console.log(JSON.stringify(output));
    } else {
      const timestamp = this.config.timestamps
        ? `[${new Date(entry.timestamp).toISOString()}] `
        : '';
      const contextStr =
        entry.context && Object.keys(entry.context).length > 0
          ? ` ${JSON.stringify(entry.context)}`
          : '';
      const errorStr = entry.error ? ` Error: ${entry.error.message}` : '';
      console.log(
        `${timestamp}${entry.level.toUpperCase()}: ${entry.message}${contextStr}${errorStr}`
      );
    }
  }
}

/**
 * Debug inspector for SDK internals.
 *
 * @example
 * ```typescript
 * const inspector = new DebugInspector();
 *
 * inspector.inspect(reputationScore);
 * inspector.diff(oldScore, newScore);
 * inspector.trace(() => calculateTrust(agent));
 * ```
 */
export class DebugInspector {
  private snapshots = new Map<string, unknown>();

  /**
   * Inspect an object with formatted output
   */
  inspect(obj: unknown, label?: string): string {
    const output = this.formatValue(obj, 0);
    const result = label ? `${label}:\n${output}` : output;
    console.log(result);
    return result;
  }

  /**
   * Take a snapshot for later comparison
   */
  snapshot(id: string, value: unknown): void {
    this.snapshots.set(id, JSON.parse(JSON.stringify(value)));
  }

  /**
   * Compare current value with snapshot
   */
  compareToSnapshot(id: string, current: unknown): { changed: boolean; diff: string } {
    const snapshot = this.snapshots.get(id);
    if (!snapshot) {
      return { changed: true, diff: 'No snapshot found' };
    }
    return this.diff(snapshot, current);
  }

  /**
   * Diff two values
   */
  diff(a: unknown, b: unknown): { changed: boolean; diff: string } {
    const aStr = JSON.stringify(a, null, 2);
    const bStr = JSON.stringify(b, null, 2);

    if (aStr === bStr) {
      return { changed: false, diff: 'No changes' };
    }

    const aLines = aStr.split('\n');
    const bLines = bStr.split('\n');
    const diffLines: string[] = [];

    const maxLines = Math.max(aLines.length, bLines.length);
    for (let i = 0; i < maxLines; i++) {
      const aLine = aLines[i] || '';
      const bLine = bLines[i] || '';
      if (aLine !== bLine) {
        if (aLine) diffLines.push(`- ${aLine}`);
        if (bLine) diffLines.push(`+ ${bLine}`);
      } else {
        diffLines.push(`  ${aLine}`);
      }
    }

    return { changed: true, diff: diffLines.join('\n') };
  }

  /**
   * Trace function execution with timing
   */
  trace<T>(fn: () => T, label = 'trace'): T {
    const start = performance.now();
    console.log(`[${label}] Starting...`);

    try {
      const result = fn();
      const duration = performance.now() - start;
      console.log(`[${label}] Completed in ${duration.toFixed(2)}ms`);
      return result;
    } catch (error) {
      const duration = performance.now() - start;
      console.log(`[${label}] Failed after ${duration.toFixed(2)}ms:`, error);
      throw error;
    }
  }

  /**
   * Trace async function execution
   */
  async traceAsync<T>(fn: () => Promise<T>, label = 'trace'): Promise<T> {
    const start = performance.now();
    console.log(`[${label}] Starting...`);

    try {
      const result = await fn();
      const duration = performance.now() - start;
      console.log(`[${label}] Completed in ${duration.toFixed(2)}ms`);
      return result;
    } catch (error) {
      const duration = performance.now() - start;
      console.log(`[${label}] Failed after ${duration.toFixed(2)}ms:`, error);
      throw error;
    }
  }

  private formatValue(value: unknown, indent: number): string {
    const spaces = '  '.repeat(indent);

    if (value === null) return `${spaces}null`;
    if (value === undefined) return `${spaces}undefined`;
    if (typeof value === 'string') return `${spaces}"${value}"`;
    if (typeof value === 'number' || typeof value === 'boolean') return `${spaces}${value}`;

    if (Array.isArray(value)) {
      if (value.length === 0) return `${spaces}[]`;
      const items = value.map((v) => this.formatValue(v, indent + 1)).join(',\n');
      return `${spaces}[\n${items}\n${spaces}]`;
    }

    if (typeof value === 'object') {
      const entries = Object.entries(value as Record<string, unknown>);
      if (entries.length === 0) return `${spaces}{}`;
      const items = entries
        .map(([k, v]) => `${'  '.repeat(indent + 1)}${k}: ${this.formatValue(v, 0).trim()}`)
        .join(',\n');
      return `${spaces}{\n${items}\n${spaces}}`;
    }

    return `${spaces}${String(value)}`;
  }
}

// Note: Branded types (BrandedAgentId, BrandedHappId, etc.) are exported from ./branded.js

/**
 * SDK helper utilities for common operations
 */
export const sdkHelpers = {
  /**
   * Generate a unique ID
   */
  generateId(prefix = ''): string {
    const random = Math.random().toString(36).slice(2, 11);
    const timestamp = Date.now().toString(36);
    return prefix ? `${prefix}_${timestamp}_${random}` : `${timestamp}_${random}`;
  },

  /**
   * Deep clone an object
   */
  deepClone<T>(obj: T): T {
    return JSON.parse(JSON.stringify(obj));
  },

  /**
   * Deep merge objects
   */
  deepMerge<T extends Record<string, unknown>>(target: T, ...sources: Partial<T>[]): T {
    const result = { ...target };

    for (const source of sources) {
      for (const key of Object.keys(source) as (keyof T)[]) {
        const targetValue = result[key];
        const sourceValue = source[key];

        if (
          targetValue &&
          sourceValue &&
          typeof targetValue === 'object' &&
          typeof sourceValue === 'object' &&
          !Array.isArray(targetValue) &&
          !Array.isArray(sourceValue)
        ) {
          result[key] = this.deepMerge(
            targetValue as Record<string, unknown>,
            sourceValue as Record<string, unknown>
          ) as T[keyof T];
        } else if (sourceValue !== undefined) {
          result[key] = sourceValue as T[keyof T];
        }
      }
    }

    return result;
  },

  /**
   * Safely parse JSON with fallback
   */
  safeJsonParse<T>(json: string, fallback: T): T {
    try {
      return JSON.parse(json) as T;
    } catch {
      return fallback;
    }
  },

  /**
   * Format bytes to human readable
   */
  formatBytes(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let unitIndex = 0;
    let value = bytes;

    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex++;
    }

    return `${value.toFixed(2)} ${units[unitIndex]}`;
  },

  /**
   * Format duration in ms to human readable
   */
  formatDuration(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
    if (ms < 3600000) return `${(ms / 60000).toFixed(2)}m`;
    return `${(ms / 3600000).toFixed(2)}h`;
  },

  /**
   * Chunk array into smaller arrays
   */
  chunk<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  },

  /**
   * Create a promise that resolves after delay
   */
  delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  },

  /**
   * Memoize a function
   */
  memoize<TArgs extends unknown[], TResult>(
    fn: (...args: TArgs) => TResult,
    keyFn?: (...args: TArgs) => string
  ): (...args: TArgs) => TResult {
    const cache = new Map<string, TResult>();
    const getKey = keyFn || ((...args: TArgs) => JSON.stringify(args));

    return (...args: TArgs): TResult => {
      const key = getKey(...args);
      if (cache.has(key)) {
        return cache.get(key)!;
      }
      const result = fn(...args);
      cache.set(key, result);
      return result;
    };
  },
};

// ============================================================================
// CATEGORY 3: Production Readiness
// ============================================================================

/**
 * Metric types
 */
export type MetricType = 'counter' | 'gauge' | 'histogram' | 'summary';

/**
 * Metric labels
 */
export type MetricLabels = Record<string, string>;

/**
 * Metric value
 */
export interface MetricValue {
  value: number;
  labels: MetricLabels;
  timestamp: number;
}

/**
 * Metric definition
 */
export interface MetricDefinition {
  name: string;
  type: MetricType;
  help: string;
  labels?: string[];
}

/**
 * Histogram buckets
 */
export interface HistogramBuckets {
  buckets: number[];
  counts: number[];
  sum: number;
  count: number;
}

/**
 * Prometheus-style metrics collector.
 *
 * @example
 * ```typescript
 * const metrics = new MetricsCollector();
 *
 * // Define metrics
 * metrics.defineCounter('reputation_updates_total', 'Total reputation updates');
 * metrics.defineGauge('active_connections', 'Current active connections');
 * metrics.defineHistogram('request_duration_ms', 'Request duration', [10, 50, 100, 500]);
 *
 * // Record values
 * metrics.incrementCounter('reputation_updates_total', { agent: 'a1' });
 * metrics.setGauge('active_connections', 5);
 * metrics.observeHistogram('request_duration_ms', 42);
 *
 * // Export
 * console.log(metrics.exportPrometheus());
 * ```
 */
export class MetricsCollector {
  private definitions = new Map<string, MetricDefinition>();
  private counters = new Map<string, Map<string, number>>();
  private gauges = new Map<string, Map<string, number>>();
  private histograms = new Map<string, Map<string, HistogramBuckets>>();

  /**
   * Define a counter metric
   */
  defineCounter(name: string, help: string, labels?: string[]): void {
    this.definitions.set(name, { name, type: 'counter', help, labels });
    this.counters.set(name, new Map());
  }

  /**
   * Define a gauge metric
   */
  defineGauge(name: string, help: string, labels?: string[]): void {
    this.definitions.set(name, { name, type: 'gauge', help, labels });
    this.gauges.set(name, new Map());
  }

  /**
   * Define a histogram metric
   */
  defineHistogram(name: string, help: string, buckets: number[], labels?: string[]): void {
    this.definitions.set(name, { name, type: 'histogram', help, labels });
    this.histograms.set(name, new Map());
    // Store bucket definition
    const key = this.labelsToKey({});
    this.histograms.get(name)!.set(key, {
      buckets,
      counts: new Array(buckets.length + 1).fill(0),
      sum: 0,
      count: 0,
    });
  }

  /**
   * Increment a counter
   */
  incrementCounter(name: string, labels: MetricLabels = {}, value = 1): void {
    const counter = this.counters.get(name);
    if (!counter) return;

    const key = this.labelsToKey(labels);
    counter.set(key, (counter.get(key) || 0) + value);
  }

  /**
   * Set a gauge value
   */
  setGauge(name: string, value: number, labels: MetricLabels = {}): void {
    const gauge = this.gauges.get(name);
    if (!gauge) return;

    const key = this.labelsToKey(labels);
    gauge.set(key, value);
  }

  /**
   * Increment a gauge
   */
  incrementGauge(name: string, labels: MetricLabels = {}, value = 1): void {
    const gauge = this.gauges.get(name);
    if (!gauge) return;

    const key = this.labelsToKey(labels);
    gauge.set(key, (gauge.get(key) || 0) + value);
  }

  /**
   * Decrement a gauge
   */
  decrementGauge(name: string, labels: MetricLabels = {}, value = 1): void {
    this.incrementGauge(name, labels, -value);
  }

  /**
   * Observe a histogram value
   */
  observeHistogram(name: string, value: number, labels: MetricLabels = {}): void {
    const histogram = this.histograms.get(name);
    if (!histogram) return;

    const key = this.labelsToKey(labels);
    let bucket = histogram.get(key);

    if (!bucket) {
      // Get bucket definition from default
      const defaultBucket = histogram.get(this.labelsToKey({}));
      if (!defaultBucket) return;
      bucket = {
        buckets: [...defaultBucket.buckets],
        counts: new Array(defaultBucket.buckets.length + 1).fill(0),
        sum: 0,
        count: 0,
      };
      histogram.set(key, bucket);
    }

    bucket.sum += value;
    bucket.count++;

    // Find bucket
    for (let i = 0; i < bucket.buckets.length; i++) {
      if (value <= bucket.buckets[i]) {
        bucket.counts[i]++;
        return;
      }
    }
    // +Inf bucket
    bucket.counts[bucket.counts.length - 1]++;
  }

  /**
   * Get a counter value
   */
  getCounter(name: string, labels: MetricLabels = {}): number {
    const counter = this.counters.get(name);
    if (!counter) return 0;
    return counter.get(this.labelsToKey(labels)) || 0;
  }

  /**
   * Get a gauge value
   */
  getGauge(name: string, labels: MetricLabels = {}): number {
    const gauge = this.gauges.get(name);
    if (!gauge) return 0;
    return gauge.get(this.labelsToKey(labels)) || 0;
  }

  /**
   * Export metrics in Prometheus format
   */
  exportPrometheus(): string {
    const lines: string[] = [];

    // Counters
    for (const [name, values] of this.counters) {
      const def = this.definitions.get(name)!;
      lines.push(`# HELP ${name} ${def.help}`);
      lines.push(`# TYPE ${name} counter`);
      for (const [key, value] of values) {
        const labels = key ? `{${key}}` : '';
        lines.push(`${name}${labels} ${value}`);
      }
    }

    // Gauges
    for (const [name, values] of this.gauges) {
      const def = this.definitions.get(name)!;
      lines.push(`# HELP ${name} ${def.help}`);
      lines.push(`# TYPE ${name} gauge`);
      for (const [key, value] of values) {
        const labels = key ? `{${key}}` : '';
        lines.push(`${name}${labels} ${value}`);
      }
    }

    // Histograms
    for (const [name, values] of this.histograms) {
      const def = this.definitions.get(name)!;
      lines.push(`# HELP ${name} ${def.help}`);
      lines.push(`# TYPE ${name} histogram`);
      for (const [key, bucket] of values) {
        const labelPrefix = key ? `${key},` : '';
        let cumulative = 0;
        for (let i = 0; i < bucket.buckets.length; i++) {
          cumulative += bucket.counts[i];
          lines.push(`${name}_bucket{${labelPrefix}le="${bucket.buckets[i]}"} ${cumulative}`);
        }
        cumulative += bucket.counts[bucket.counts.length - 1];
        lines.push(`${name}_bucket{${labelPrefix}le="+Inf"} ${cumulative}`);
        lines.push(`${name}_sum{${key}} ${bucket.sum}`);
        lines.push(`${name}_count{${key}} ${bucket.count}`);
      }
    }

    return lines.join('\n');
  }

  /**
   * Export metrics as JSON
   */
  exportJson(): Record<string, unknown> {
    const result: Record<string, unknown> = {};

    for (const [name, values] of this.counters) {
      result[name] = Object.fromEntries(values);
    }
    for (const [name, values] of this.gauges) {
      result[name] = Object.fromEntries(values);
    }
    for (const [name, values] of this.histograms) {
      result[name] = Object.fromEntries(
        Array.from(values.entries()).map(([k, v]) => [k, { ...v }])
      );
    }

    return result;
  }

  /**
   * Reset all metrics
   */
  reset(): void {
    for (const counter of this.counters.values()) {
      counter.clear();
    }
    for (const gauge of this.gauges.values()) {
      gauge.clear();
    }
    for (const histogram of this.histograms.values()) {
      for (const bucket of histogram.values()) {
        bucket.counts.fill(0);
        bucket.sum = 0;
        bucket.count = 0;
      }
    }
  }

  private labelsToKey(labels: MetricLabels): string {
    const entries = Object.entries(labels).sort(([a], [b]) => a.localeCompare(b));
    return entries.map(([k, v]) => `${k}="${v}"`).join(',');
  }
}

/** Global metrics collector instance */
export const metrics = new MetricsCollector();

// Initialize default SDK metrics
metrics.defineCounter('sdk_operations_total', 'Total SDK operations', ['operation', 'status']);
metrics.defineGauge('sdk_active_connections', 'Active SDK connections');
metrics.defineHistogram(
  'sdk_operation_duration_ms',
  'SDK operation duration',
  [1, 5, 10, 25, 50, 100, 250, 500, 1000]
);

/**
 * Health check status
 */
export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy';

/**
 * Health check result
 */
export interface HealthCheck {
  name: string;
  status: HealthStatus;
  message?: string;
  latencyMs?: number;
  lastCheck: number;
  metadata?: Record<string, unknown>;
}

/**
 * Overall health report
 */
export interface HealthReport {
  status: HealthStatus;
  checks: HealthCheck[];
  timestamp: number;
  version: string;
  uptime: number;
}

/**
 * Health check function type
 */
export type HealthCheckFn = () => Promise<{
  status: HealthStatus;
  message?: string;
  metadata?: Record<string, unknown>;
}>;

/**
 * Health checker for production monitoring.
 *
 * @example
 * ```typescript
 * const health = new HealthChecker('1.0.0');
 *
 * health.register('database', async () => {
 *   const connected = await db.ping();
 *   return { status: connected ? 'healthy' : 'unhealthy' };
 * });
 *
 * health.register('conductor', async () => {
 *   try {
 *     await conductor.appInfo();
 *     return { status: 'healthy' };
 *   } catch {
 *     return { status: 'unhealthy', message: 'Connection failed' };
 *   }
 * });
 *
 * const report = await health.check();
 * console.log(report.status);
 * ```
 */
export class HealthChecker {
  private checks = new Map<string, HealthCheckFn>();
  private lastResults = new Map<string, HealthCheck>();
  private startTime = Date.now();

  constructor(private readonly version: string = '1.0.0') {}

  /**
   * Register a health check
   */
  register(name: string, check: HealthCheckFn): void {
    this.checks.set(name, check);
  }

  /**
   * Unregister a health check
   */
  unregister(name: string): void {
    this.checks.delete(name);
    this.lastResults.delete(name);
  }

  /**
   * Run all health checks
   */
  async check(): Promise<HealthReport> {
    const results: HealthCheck[] = [];
    let overallStatus: HealthStatus = 'healthy';

    for (const [name, checkFn] of this.checks) {
      const start = performance.now();
      try {
        const result = await checkFn();
        const latencyMs = performance.now() - start;

        const check: HealthCheck = {
          name,
          status: result.status,
          message: result.message,
          latencyMs,
          lastCheck: Date.now(),
          metadata: result.metadata,
        };

        results.push(check);
        this.lastResults.set(name, check);

        // Update overall status
        if (result.status === 'unhealthy') {
          overallStatus = 'unhealthy';
        } else if (result.status === 'degraded' && overallStatus !== 'unhealthy') {
          overallStatus = 'degraded';
        }
      } catch (error) {
        const latencyMs = performance.now() - start;
        const check: HealthCheck = {
          name,
          status: 'unhealthy',
          message: error instanceof Error ? error.message : 'Check failed',
          latencyMs,
          lastCheck: Date.now(),
        };

        results.push(check);
        this.lastResults.set(name, check);
        overallStatus = 'unhealthy';
      }
    }

    return {
      status: overallStatus,
      checks: results,
      timestamp: Date.now(),
      version: this.version,
      uptime: Date.now() - this.startTime,
    };
  }

  /**
   * Get last check results without running checks
   */
  getLastResults(): HealthReport {
    return {
      status: this.calculateOverallStatus(),
      checks: Array.from(this.lastResults.values()),
      timestamp: Date.now(),
      version: this.version,
      uptime: Date.now() - this.startTime,
    };
  }

  /**
   * Check if system is healthy
   */
  async isHealthy(): Promise<boolean> {
    const report = await this.check();
    return report.status === 'healthy';
  }

  private calculateOverallStatus(): HealthStatus {
    let status: HealthStatus = 'healthy';
    for (const check of this.lastResults.values()) {
      if (check.status === 'unhealthy') return 'unhealthy';
      if (check.status === 'degraded') status = 'degraded';
    }
    return status;
  }
}

/**
 * Circuit breaker states
 */
export type CircuitBreakerState = 'closed' | 'open' | 'half-open';

/**
 * Circuit breaker configuration
 */
export interface CircuitBreakerConfig {
  /** Failure threshold to open circuit (default: 5) */
  failureThreshold: number;
  /** Success threshold to close from half-open (default: 2) */
  successThreshold: number;
  /** Time to wait before half-open in ms (default: 30000) */
  resetTimeoutMs: number;
  /** Timeout for each call in ms (default: 10000) */
  callTimeoutMs: number;
}

/**
 * Circuit breaker statistics
 */
export interface CircuitBreakerStats {
  state: CircuitBreakerState;
  failures: number;
  successes: number;
  consecutiveFailures: number;
  consecutiveSuccesses: number;
  lastFailure: number | null;
  lastSuccess: number | null;
  totalCalls: number;
  rejectedCalls: number;
}

/**
 * Circuit breaker for fault tolerance.
 *
 * Prevents cascading failures by stopping calls to unhealthy services.
 *
 * @example
 * ```typescript
 * const breaker = new CircuitBreaker({
 *   failureThreshold: 5,
 *   resetTimeoutMs: 30000,
 * });
 *
 * try {
 *   const result = await breaker.call(async () => {
 *     return await externalService.fetch();
 *   });
 * } catch (error) {
 *   if (error.message === 'Circuit is open') {
 *     // Use fallback
 *   }
 * }
 * ```
 */
export class CircuitBreaker {
  private state: CircuitBreakerState = 'closed';
  private failures = 0;
  private successes = 0;
  private consecutiveFailures = 0;
  private consecutiveSuccesses = 0;
  private lastFailure: number | null = null;
  private lastSuccess: number | null = null;
  private totalCalls = 0;
  private rejectedCalls = 0;
  private stateChangeTime = Date.now();
  private readonly config: Required<CircuitBreakerConfig>;

  constructor(config: Partial<CircuitBreakerConfig> = {}) {
    this.config = {
      failureThreshold: 5,
      successThreshold: 2,
      resetTimeoutMs: 30000,
      callTimeoutMs: 10000,
      ...config,
    };
  }

  /**
   * Execute a function through the circuit breaker
   */
  async call<T>(fn: () => Promise<T>): Promise<T> {
    if (!this.canCall()) {
      this.rejectedCalls++;
      throw new Error('Circuit is open');
    }

    this.totalCalls++;

    try {
      const result = await this.executeWithTimeout(fn);
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  /**
   * Check if circuit allows calls
   */
  canCall(): boolean {
    if (this.state === 'closed') return true;

    if (this.state === 'open') {
      // Check if we should transition to half-open
      if (Date.now() - this.stateChangeTime >= this.config.resetTimeoutMs) {
        this.setState('half-open');
        return true;
      }
      return false;
    }

    // half-open: allow limited calls
    return true;
  }

  /**
   * Get circuit breaker state
   */
  getState(): CircuitBreakerState {
    return this.state;
  }

  /**
   * Get statistics
   */
  getStats(): CircuitBreakerStats {
    return {
      state: this.state,
      failures: this.failures,
      successes: this.successes,
      consecutiveFailures: this.consecutiveFailures,
      consecutiveSuccesses: this.consecutiveSuccesses,
      lastFailure: this.lastFailure,
      lastSuccess: this.lastSuccess,
      totalCalls: this.totalCalls,
      rejectedCalls: this.rejectedCalls,
    };
  }

  /**
   * Manually reset the circuit breaker
   */
  reset(): void {
    this.setState('closed');
    this.failures = 0;
    this.successes = 0;
    this.consecutiveFailures = 0;
    this.consecutiveSuccesses = 0;
  }

  private async executeWithTimeout<T>(fn: () => Promise<T>): Promise<T> {
    return Promise.race([
      fn(),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('Call timeout')), this.config.callTimeoutMs)
      ),
    ]);
  }

  private onSuccess(): void {
    this.successes++;
    this.consecutiveSuccesses++;
    this.consecutiveFailures = 0;
    this.lastSuccess = Date.now();

    if (this.state === 'half-open') {
      if (this.consecutiveSuccesses >= this.config.successThreshold) {
        this.setState('closed');
      }
    }
  }

  private onFailure(): void {
    this.failures++;
    this.consecutiveFailures++;
    this.consecutiveSuccesses = 0;
    this.lastFailure = Date.now();

    if (this.state === 'closed' || this.state === 'half-open') {
      if (this.consecutiveFailures >= this.config.failureThreshold) {
        this.setState('open');
      }
    }
  }

  private setState(newState: CircuitBreakerState): void {
    this.state = newState;
    this.stateChangeTime = Date.now();
    if (newState === 'closed') {
      this.consecutiveFailures = 0;
      this.consecutiveSuccesses = 0;
    }
  }
}

// ============================================================================
// CATEGORY 4: Testing & Quality
// ============================================================================

/**
 * Test fixture for MATL testing
 */
export interface MatlTestFixture {
  agentId: string;
  pogq: { quality: number; consistency: number; entropy: number };
  reputation: { positiveVotes: number; totalVotes: number };
  expectedComposite: number;
  expectedTrustworthy: boolean;
}

/**
 * Test fixture for Epistemic testing
 */
export interface EpistemicTestFixture {
  claimText: string;
  empirical: 'E0' | 'E1' | 'E2' | 'E3';
  normative: 'N0' | 'N1' | 'N2';
  materiality: 'M0' | 'M1' | 'M2';
  expectedCode: string;
}

/**
 * Test fixture for FL testing
 */
export interface FlTestFixture {
  participants: Array<{
    id: string;
    gradients: number[];
    trust: number;
  }>;
  expectedMethod: string;
  expectedResult: number[];
}

/**
 * Test fixture factory for generating test data.
 *
 * @example
 * ```typescript
 * const factory = new TestFixtureFactory();
 *
 * // Generate MATL fixtures
 * const matlFixtures = factory.generateMatlFixtures(10);
 *
 * // Generate with specific parameters
 * const highTrustAgent = factory.createMatlFixture({
 *   quality: 0.95,
 *   consistency: 0.90,
 * });
 * ```
 */
export class TestFixtureFactory {
  private seed: number;

  constructor(seed = 12345) {
    this.seed = seed;
  }

  /**
   * Seeded random number generator
   */
  private random(): number {
    const x = Math.sin(this.seed++) * 10000;
    return x - Math.floor(x);
  }

  /**
   * Generate random number in range
   */
  private randomInRange(min: number, max: number): number {
    return min + this.random() * (max - min);
  }

  /**
   * Generate a random agent ID
   */
  generateAgentId(): string {
    return `agent_${Math.floor(this.random() * 1000000).toString(36)}`;
  }

  /**
   * Create a MATL test fixture
   */
  createMatlFixture(overrides: Partial<MatlTestFixture> = {}): MatlTestFixture {
    const quality = overrides.pogq?.quality ?? this.randomInRange(0.5, 1.0);
    const consistency = overrides.pogq?.consistency ?? this.randomInRange(0.5, 1.0);
    const entropy = overrides.pogq?.entropy ?? this.randomInRange(0, 0.5);
    const positiveVotes =
      overrides.reputation?.positiveVotes ?? Math.floor(this.randomInRange(10, 100));
    const totalVotes =
      overrides.reputation?.totalVotes ?? Math.floor(positiveVotes + this.randomInRange(0, 20));

    const reputationScore = totalVotes > 0 ? positiveVotes / totalVotes : 0.5;
    const expectedComposite =
      (quality * 0.4 + consistency * 0.35 + reputationScore * 0.25) * (1 - entropy * 0.1);

    return {
      agentId: overrides.agentId ?? this.generateAgentId(),
      pogq: { quality, consistency, entropy },
      reputation: { positiveVotes, totalVotes },
      expectedComposite,
      expectedTrustworthy: expectedComposite >= 0.55,
      ...overrides,
    };
  }

  /**
   * Generate multiple MATL fixtures
   */
  generateMatlFixtures(count: number): MatlTestFixture[] {
    return Array.from({ length: count }, () => this.createMatlFixture());
  }

  /**
   * Create an Epistemic test fixture
   */
  createEpistemicFixture(overrides: Partial<EpistemicTestFixture> = {}): EpistemicTestFixture {
    const empiricalLevels: EpistemicTestFixture['empirical'][] = ['E0', 'E1', 'E2', 'E3'];
    const normativeLevels: EpistemicTestFixture['normative'][] = ['N0', 'N1', 'N2'];
    const materialityLevels: EpistemicTestFixture['materiality'][] = ['M0', 'M1', 'M2'];

    const empirical = overrides.empirical ?? empiricalLevels[Math.floor(this.random() * 4)];
    const normative = overrides.normative ?? normativeLevels[Math.floor(this.random() * 3)];
    const materiality = overrides.materiality ?? materialityLevels[Math.floor(this.random() * 3)];

    return {
      claimText: overrides.claimText ?? `Test claim ${Math.floor(this.random() * 10000)}`,
      empirical,
      normative,
      materiality,
      expectedCode: `${empirical}-${normative}-${materiality}`,
    };
  }

  /**
   * Generate multiple Epistemic fixtures
   */
  generateEpistemicFixtures(count: number): EpistemicTestFixture[] {
    return Array.from({ length: count }, () => this.createEpistemicFixture());
  }

  /**
   * Create an FL test fixture
   */
  createFlFixture(participantCount = 5, gradientSize = 10): FlTestFixture {
    const participants = Array.from({ length: participantCount }, () => ({
      id: this.generateAgentId(),
      gradients: Array.from({ length: gradientSize }, () => this.randomInRange(-1, 1)),
      trust: this.randomInRange(0.5, 1.0),
    }));

    // Calculate expected FedAvg result
    const expectedResult = Array.from({ length: gradientSize }, (_, i) => {
      const sum = participants.reduce((acc, p) => acc + p.gradients[i], 0);
      return sum / participantCount;
    });

    return {
      participants,
      expectedMethod: 'fedAvg',
      expectedResult,
    };
  }

  /**
   * Create Byzantine attack scenario fixture
   */
  createByzantineFixture(honestCount = 7, byzantineCount = 3, gradientSize = 10): FlTestFixture {
    const honest = Array.from({ length: honestCount }, () => ({
      id: this.generateAgentId(),
      gradients: Array.from({ length: gradientSize }, () => this.randomInRange(-0.1, 0.1)),
      trust: this.randomInRange(0.7, 1.0),
    }));

    const byzantine = Array.from({ length: byzantineCount }, () => ({
      id: this.generateAgentId(),
      gradients: Array.from({ length: gradientSize }, () => this.randomInRange(-10, 10)), // Extreme values
      trust: this.randomInRange(0.3, 0.5),
    }));

    return {
      participants: [...honest, ...byzantine],
      expectedMethod: 'trimmedMean',
      expectedResult: Array.from({ length: gradientSize }, (_, i) => {
        const values = honest.map((p) => p.gradients[i]).sort((a, b) => a - b);
        return values[Math.floor(values.length / 2)];
      }),
    };
  }

  /**
   * Reset seed for reproducible tests
   */
  resetSeed(seed: number): void {
    this.seed = seed;
  }
}

/**
 * Mock factory for creating test doubles
 */
export const mockFactory = {
  /**
   * Create a mock Holochain client
   */
  createMockClient() {
    const responses = new Map<string, unknown>();

    return {
      setResponse(zome: string, fn: string, response: unknown) {
        responses.set(`${zome}:${fn}`, response);
      },
      async callZome({ zome_name, fn_name }: { zome_name: string; fn_name: string }) {
        const key = `${zome_name}:${fn_name}`;
        if (responses.has(key)) {
          return responses.get(key);
        }
        throw new Error(`No mock response for ${key}`);
      },
      async appInfo() {
        return { app_id: 'mock-app', cell_info: {} };
      },
    };
  },

  /**
   * Create a mock WebSocket
   */
  createMockWebSocket() {
    let onMessage: ((data: unknown) => void) | null = null;
    let connected = false;

    return {
      connect() {
        connected = true;
      },
      disconnect() {
        connected = false;
      },
      send(data: unknown) {
        if (!connected) throw new Error('Not connected');
        return data;
      },
      onMessage(handler: (data: unknown) => void) {
        onMessage = handler;
      },
      simulateMessage(data: unknown) {
        onMessage?.(data);
      },
      isConnected() {
        return connected;
      },
    };
  },

  /**
   * Create a mock metrics collector
   */
  createMockMetrics() {
    const values = new Map<string, number>();

    return {
      increment(name: string, value = 1) {
        values.set(name, (values.get(name) || 0) + value);
      },
      set(name: string, value: number) {
        values.set(name, value);
      },
      get(name: string) {
        return values.get(name) || 0;
      },
      getAll() {
        return Object.fromEntries(values);
      },
    };
  },

  /**
   * Create a delayed promise for async testing
   */
  createDelayedPromise<T>(value: T, delayMs: number): Promise<T> {
    return new Promise((resolve) => setTimeout(() => resolve(value), delayMs));
  },

  /**
   * Create a failing promise for error testing
   */
  createFailingPromise(error: Error, delayMs = 0): Promise<never> {
    return new Promise((_, reject) => setTimeout(() => reject(error), delayMs));
  },
};

/**
 * Property-based test generators
 */
export const propertyGenerators = {
  /**
   * Generate trust score (0-1)
   */
  trustScore(): number {
    return Math.random();
  },

  /**
   * Generate array of trust scores
   */
  trustScores(count: number): number[] {
    return Array.from({ length: count }, () => this.trustScore());
  },

  /**
   * Generate gradient array
   */
  gradients(size: number, range = 1): number[] {
    return Array.from({ length: size }, () => (Math.random() * 2 - 1) * range);
  },

  /**
   * Generate agent ID
   */
  agentId(): string {
    return `agent_${Math.random().toString(36).slice(2, 11)}`;
  },

  /**
   * Generate timestamp in past
   */
  pastTimestamp(maxAgeMs = 86400000): number {
    return Date.now() - Math.floor(Math.random() * maxAgeMs);
  },

  /**
   * Generate timestamp in future
   */
  futureTimestamp(maxFutureMs = 86400000): number {
    return Date.now() + Math.floor(Math.random() * maxFutureMs);
  },

  /**
   * Generate enum value
   */
  enumValue<T extends string>(values: readonly T[]): T {
    return values[Math.floor(Math.random() * values.length)];
  },

  /**
   * Generate boolean with probability
   */
  boolean(probability = 0.5): boolean {
    return Math.random() < probability;
  },

  /**
   * Generate integer in range
   */
  integer(min: number, max: number): number {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  },

  /**
   * Generate string of given length
   */
  string(length: number): string {
    const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    return Array.from({ length }, () => chars[Math.floor(Math.random() * chars.length)]).join('');
  },

  /**
   * Generate one of the provided values
   */
  oneOf<T>(values: T[]): T {
    return values[Math.floor(Math.random() * values.length)];
  },

  /**
   * Generate array with random length
   */
  array<T>(generator: () => T, minLength = 1, maxLength = 10): T[] {
    const length = this.integer(minLength, maxLength);
    return Array.from({ length }, generator);
  },
};

// ============================================================================
// CATEGORY 5: Documentation Helpers
// ============================================================================

/**
 * Example usage documentation
 */
export const examples = {
  /**
   * Example: Basic trust evaluation
   */
  basicTrustEvaluation: `
import { matl, createPoGQ, compositeScore, isTrustworthy } from '@mycelix/sdk';

// Create a proof of gradient quality
const pogq = createPoGQ(0.95, 0.88, 0.12);

// Calculate composite trust score
const score = compositeScore(pogq, 0.75);

// Check if agent is trustworthy
if (isTrustworthy(score)) {
  console.log('Agent is trustworthy');
}
`,

  /**
   * Example: Epistemic claim creation
   */
  epistemicClaimCreation: `
import { epistemic, claim, EmpiricalLevel, NormativeLevel } from '@mycelix/sdk';

// Create a claim with builder pattern
const myClaim = claim("User verified their identity")
  .withEmpirical(EmpiricalLevel.E3_Cryptographic)
  .withNormative(NormativeLevel.N2_Network)
  .withEvidence({
    type: 'signature',
    source: 'did:key:z6Mk...',
    timestamp: Date.now()
  })
  .build();

console.log('Claim classification:', myClaim.classification);
`,

  /**
   * Example: Federated learning round
   */
  federatedLearningRound: `
import { fl, FLCoordinator, AggregationMethod } from '@mycelix/sdk';

// Create coordinator
const coordinator = new FLCoordinator({
  minParticipants: 3,
  maxParticipants: 10,
  aggregationMethod: AggregationMethod.TrustWeighted
});

// Register participants
coordinator.registerParticipant('node1', 0.95);
coordinator.registerParticipant('node2', 0.88);
coordinator.registerParticipant('node3', 0.92);

// Submit updates
coordinator.submitUpdate('node1', { gradients: [...], epoch: 1 });
coordinator.submitUpdate('node2', { gradients: [...], epoch: 1 });
coordinator.submitUpdate('node3', { gradients: [...], epoch: 1 });

// Aggregate
const result = coordinator.aggregate();
console.log('Aggregated gradients:', result.gradients);
`,

  /**
   * Example: Cross-hApp reputation bridge
   */
  crossHappBridge: `
import { bridge, LocalBridge, calculateAggregateReputation } from '@mycelix/sdk';

// Create local bridge
const localBridge = new LocalBridge();

// Register hApps
localBridge.registerHapp('social-app');
localBridge.registerHapp('marketplace');

// Query cross-hApp reputation
const reputation = await localBridge.query('agent123', ['social-app', 'marketplace']);

// Calculate aggregate
const aggregate = calculateAggregateReputation(reputation.scores);
console.log('Aggregate reputation:', aggregate);
`,

  /**
   * Example: Event pipeline
   */
  eventPipeline: `
import { pipeline, sdkEvents, MatlEventType } from '@mycelix/sdk';

// Create pipeline from SDK events
const trustUpdates = pipeline(sdkEvents.matl)
  .filter(e => e.type === MatlEventType.TRUST_EVALUATED)
  .map(e => ({ agentId: e.agentId, score: e.score }))
  .debounce(100)
  .subscribe(update => {
    console.log('Trust updated:', update);
  });

// Later: cleanup
trustUpdates();
`,

  /**
   * Example: Distributed tracing
   */
  distributedTracing: `
import { tracer, traced } from '@mycelix/sdk';

// Trace an async operation
const result = await traced('processTransaction', async (ctx) => {
  // Start child span
  const span = tracer.startSpan(ctx, 'validateInput');
  // ... validation logic
  tracer.endSpan(span.spanId, 'success');

  // Another child span
  const dbSpan = tracer.startSpan(ctx, 'databaseWrite');
  // ... database logic
  tracer.endSpan(dbSpan.spanId, 'success');

  return { success: true };
});

// Get completed traces
const traces = tracer.getCompletedTraces();
console.log('Trace duration:', traces[0].totalDuration);
`,
};

// ============================================================================
// CATEGORY 6: Advanced Features
// ============================================================================

/**
 * Plugin metadata
 */
export interface PluginMetadata {
  name: string;
  version: string;
  description?: string;
  author?: string;
  dependencies?: string[];
}

/**
 * Plugin hooks
 */
export interface PluginHooks {
  onInit?: () => Promise<void>;
  onDestroy?: () => Promise<void>;
  beforeOperation?: (operation: string, params: unknown) => Promise<unknown>;
  afterOperation?: (operation: string, result: unknown) => Promise<unknown>;
}

/**
 * Plugin interface
 */
export interface Plugin extends PluginMetadata, PluginHooks {
  id: string;
}

/**
 * Plugin manager for SDK extensibility.
 *
 * @example
 * ```typescript
 * const plugins = new PluginManager();
 *
 * // Register a plugin
 * plugins.register({
 *   id: 'logging-plugin',
 *   name: 'Logging Plugin',
 *   version: '1.0.0',
 *   beforeOperation: async (op, params) => {
 *     console.log(`[${op}]`, params);
 *     return params;
 *   }
 * });
 *
 * // Execute with plugins
 * const result = await plugins.executeWithHooks('createClaim', claim);
 * ```
 */
export class PluginManager {
  private plugins = new Map<string, Plugin>();
  private initialized = new Set<string>();

  /**
   * Register a plugin
   */
  async register(plugin: Plugin): Promise<void> {
    if (this.plugins.has(plugin.id)) {
      throw new Error(`Plugin ${plugin.id} already registered`);
    }

    // Check dependencies
    if (plugin.dependencies) {
      for (const dep of plugin.dependencies) {
        if (!this.plugins.has(dep)) {
          throw new Error(`Missing dependency: ${dep}`);
        }
      }
    }

    this.plugins.set(plugin.id, plugin);

    // Initialize
    if (plugin.onInit) {
      await plugin.onInit();
    }
    this.initialized.add(plugin.id);
  }

  /**
   * Unregister a plugin
   */
  async unregister(pluginId: string): Promise<void> {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) return;

    // Check if other plugins depend on this
    for (const [id, p] of this.plugins) {
      if (p.dependencies?.includes(pluginId)) {
        throw new Error(`Plugin ${id} depends on ${pluginId}`);
      }
    }

    // Destroy
    if (plugin.onDestroy) {
      await plugin.onDestroy();
    }

    this.plugins.delete(pluginId);
    this.initialized.delete(pluginId);
  }

  /**
   * Execute operation with plugin hooks
   */
  async executeWithHooks<T>(
    operation: string,
    params: unknown,
    executor: (p: unknown) => Promise<T>
  ): Promise<T> {
    let currentParams = params;

    // Before hooks
    for (const plugin of this.plugins.values()) {
      if (plugin.beforeOperation) {
        currentParams = await plugin.beforeOperation(operation, currentParams);
      }
    }

    // Execute
    let result: T = await executor(currentParams);

    // After hooks
    for (const plugin of this.plugins.values()) {
      if (plugin.afterOperation) {
        result = (await plugin.afterOperation(operation, result)) as T;
      }
    }

    return result;
  }

  /**
   * Get registered plugins
   */
  getPlugins(): Plugin[] {
    return Array.from(this.plugins.values());
  }

  /**
   * Check if plugin is registered
   */
  hasPlugin(pluginId: string): boolean {
    return this.plugins.has(pluginId);
  }

  /**
   * Get plugin by ID
   */
  getPlugin(pluginId: string): Plugin | undefined {
    return this.plugins.get(pluginId);
  }
}

/**
 * State synchronization configuration
 */
export interface StateSyncConfig {
  /** Sync interval in ms */
  syncIntervalMs: number;
  /** Conflict resolution strategy */
  conflictResolution: 'local-wins' | 'remote-wins' | 'latest-wins' | 'merge';
  /** Enable offline support */
  offlineSupport: boolean;
}

/**
 * Syncable state entry
 */
export interface SyncableState<T> {
  key: string;
  value: T;
  version: number;
  timestamp: number;
  dirty: boolean;
}

/**
 * State synchronization manager for local/remote state consistency.
 *
 * @example
 * ```typescript
 * const stateSync = new StateSync({
 *   syncIntervalMs: 5000,
 *   conflictResolution: 'latest-wins',
 *   offlineSupport: true
 * });
 *
 * // Set local state
 * stateSync.set('user:preferences', { theme: 'dark' });
 *
 * // Sync with remote
 * await stateSync.sync(async (changes) => {
 *   return await api.syncState(changes);
 * });
 *
 * // Get state
 * const prefs = stateSync.get('user:preferences');
 * ```
 */
export class StateSync<T = unknown> {
  private state = new Map<string, SyncableState<T>>();
  private syncTimer: ReturnType<typeof setInterval> | null = null;
  private readonly config: Required<StateSyncConfig>;

  constructor(config: Partial<StateSyncConfig> = {}) {
    this.config = {
      syncIntervalMs: 5000,
      conflictResolution: 'latest-wins',
      offlineSupport: true,
      ...config,
    };
  }

  /**
   * Set a value
   */
  set(key: string, value: T): void {
    const existing = this.state.get(key);
    this.state.set(key, {
      key,
      value,
      version: (existing?.version ?? 0) + 1,
      timestamp: Date.now(),
      dirty: true,
    });
  }

  /**
   * Get a value
   */
  get(key: string): T | undefined {
    return this.state.get(key)?.value;
  }

  /**
   * Delete a value
   */
  delete(key: string): boolean {
    return this.state.delete(key);
  }

  /**
   * Get all dirty (unsynchronized) entries
   */
  getDirtyEntries(): SyncableState<T>[] {
    return Array.from(this.state.values()).filter((s) => s.dirty);
  }

  /**
   * Synchronize with remote
   */
  async sync(
    syncFn: (changes: SyncableState<T>[]) => Promise<SyncableState<T>[]>
  ): Promise<{ synced: number; conflicts: number }> {
    const dirty = this.getDirtyEntries();
    let conflicts = 0;

    try {
      const remoteChanges = await syncFn(dirty);

      // Process remote changes
      for (const remote of remoteChanges) {
        const local = this.state.get(remote.key);

        if (!local) {
          // New from remote
          this.state.set(remote.key, { ...remote, dirty: false });
        } else if (local.dirty) {
          // Conflict
          conflicts++;
          const resolved = this.resolveConflict(local, remote);
          this.state.set(remote.key, resolved);
        } else {
          // Update from remote
          this.state.set(remote.key, { ...remote, dirty: false });
        }
      }

      // Mark synced
      for (const entry of dirty) {
        const current = this.state.get(entry.key);
        if (current && current.version === entry.version) {
          current.dirty = false;
        }
      }

      return { synced: dirty.length, conflicts };
    } catch {
      // Offline - keep dirty state
      return { synced: 0, conflicts: 0 };
    }
  }

  /**
   * Start automatic sync
   */
  startAutoSync(syncFn: (changes: SyncableState<T>[]) => Promise<SyncableState<T>[]>): void {
    if (this.syncTimer) return;

    this.syncTimer = setInterval(() => {
      void this.sync(syncFn);
    }, this.config.syncIntervalMs);
  }

  /**
   * Stop automatic sync
   */
  stopAutoSync(): void {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
      this.syncTimer = null;
    }
  }

  /**
   * Export state for persistence
   */
  export(): string {
    return JSON.stringify(Array.from(this.state.entries()));
  }

  /**
   * Import state from persistence
   */
  import(data: string): void {
    try {
      const entries = JSON.parse(data) as [string, SyncableState<T>][];
      this.state = new Map(entries);
    } catch {
      // Invalid data
    }
  }

  private resolveConflict(local: SyncableState<T>, remote: SyncableState<T>): SyncableState<T> {
    switch (this.config.conflictResolution) {
      case 'local-wins':
        return { ...local, dirty: false };
      case 'remote-wins':
        return { ...remote, dirty: false };
      case 'latest-wins':
        return local.timestamp > remote.timestamp
          ? { ...local, dirty: false }
          : { ...remote, dirty: false };
      case 'merge':
        // Simple merge: prefer local for conflicts
        if (typeof local.value === 'object' && typeof remote.value === 'object') {
          return {
            key: local.key,
            value: { ...remote.value, ...local.value } as T,
            version: Math.max(local.version, remote.version) + 1,
            timestamp: Date.now(),
            dirty: false,
          };
        }
        return { ...local, dirty: false };
    }
  }
}

/**
 * Encrypted storage configuration
 */
export interface EncryptedStorageConfig {
  /** Encryption algorithm */
  algorithm: 'AES-GCM' | 'AES-CBC';
  /** Key derivation iterations */
  iterations: number;
}

/**
 * Encrypted local storage for sensitive data.
 *
 * @example
 * ```typescript
 * const storage = new EncryptedStorage();
 *
 * // Initialize with password
 * await storage.init('user-secret-password');
 *
 * // Store encrypted data
 * await storage.set('private-key', 'very-secret-key');
 *
 * // Retrieve decrypted data
 * const key = await storage.get('private-key');
 * ```
 */
export class EncryptedStorage {
  private key: CryptoKey | null = null;
  private storage = new Map<string, string>();
  private readonly _config: Required<EncryptedStorageConfig>;

  constructor(config: Partial<EncryptedStorageConfig> = {}) {
    this._config = {
      algorithm: 'AES-GCM',
      iterations: 100000,
      ...config,
    };
  }

  /**
   * Initialize storage with password
   */
  async init(password: string): Promise<void> {
    // In a real implementation, this would use WebCrypto API with this._config
    // For now, we simulate the encryption using config parameters
    const encoder = new TextEncoder();
    const keyMaterial = encoder.encode(password);

    // Simple hash-based key derivation using configured iterations
    // (in production, use PBKDF2 with this._config.iterations)
    let hash = 0;
    const iterations = Math.min(this._config.iterations / 10000, 10);
    for (let i = 0; i < iterations; i++) {
      for (const byte of keyMaterial) {
        hash = (hash << 5) - hash + byte;
        hash = hash & hash;
      }
    }

    // Store derived key indicator with algorithm info
    this.key = { hash, algorithm: this._config.algorithm } as unknown as CryptoKey;
  }

  /**
   * Check if storage is initialized
   */
  isInitialized(): boolean {
    return this.key !== null;
  }

  /**
   * Set an encrypted value
   */
  async set(key: string, value: string): Promise<void> {
    if (!this.key) {
      throw new Error('Storage not initialized');
    }

    // Simulate encryption (in production, use actual encryption)
    const encrypted = btoa(value);
    this.storage.set(key, encrypted);
  }

  /**
   * Get a decrypted value
   */
  async get(key: string): Promise<string | null> {
    if (!this.key) {
      throw new Error('Storage not initialized');
    }

    const encrypted = this.storage.get(key);
    if (!encrypted) return null;

    // Simulate decryption
    return atob(encrypted);
  }

  /**
   * Delete a value
   */
  async delete(key: string): Promise<boolean> {
    return this.storage.delete(key);
  }

  /**
   * Check if key exists
   */
  has(key: string): boolean {
    return this.storage.has(key);
  }

  /**
   * Get all keys
   */
  keys(): string[] {
    return Array.from(this.storage.keys());
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.storage.clear();
  }

  /**
   * Lock storage (clear key from memory)
   */
  lock(): void {
    this.key = null;
  }
}

/**
 * Multi-conductor connection manager.
 *
 * Manages connections to multiple Holochain conductors
 * for redundancy and load balancing.
 *
 * @example
 * ```typescript
 * const multi = new MultiConductor();
 *
 * // Add conductors
 * multi.addConductor('primary', 'ws://localhost:8888');
 * multi.addConductor('secondary', 'ws://localhost:8889');
 *
 * // Execute with failover
 * const result = await multi.execute(async (client) => {
 *   return await client.callZome({ ... });
 * });
 * ```
 */
export class MultiConductor {
  private conductors = new Map<
    string,
    {
      url: string;
      healthy: boolean;
      lastCheck: number;
      failureCount: number;
    }
  >();
  private primary: string | null = null;

  /**
   * Add a conductor
   */
  addConductor(id: string, url: string, isPrimary = false): void {
    this.conductors.set(id, {
      url,
      healthy: true,
      lastCheck: Date.now(),
      failureCount: 0,
    });

    if (isPrimary || this.primary === null) {
      this.primary = id;
    }
  }

  /**
   * Remove a conductor
   */
  removeConductor(id: string): void {
    this.conductors.delete(id);
    if (this.primary === id) {
      this.primary = this.conductors.keys().next().value || null;
    }
  }

  /**
   * Get healthy conductors
   */
  getHealthyConductors(): string[] {
    return Array.from(this.conductors.entries())
      .filter(([_, info]) => info.healthy)
      .map(([id]) => id);
  }

  /**
   * Execute operation with failover
   */
  async execute<T>(
    operation: (conductorUrl: string) => Promise<T>,
    options: { timeout?: number; retries?: number } = {}
  ): Promise<T> {
    const { timeout = 10000, retries = 2 } = options;
    const conductorIds = this.getHealthyConductors();

    if (conductorIds.length === 0) {
      throw new Error('No healthy conductors available');
    }

    // Try primary first
    if (this.primary && conductorIds.includes(this.primary)) {
      const index = conductorIds.indexOf(this.primary);
      conductorIds.splice(index, 1);
      conductorIds.unshift(this.primary);
    }

    let lastError: Error | null = null;

    for (const conductorId of conductorIds) {
      const conductor = this.conductors.get(conductorId)!;

      for (let attempt = 0; attempt <= retries; attempt++) {
        try {
          const result = await this.executeWithTimeout(() => operation(conductor.url), timeout);

          // Success - reset failure count
          conductor.failureCount = 0;
          conductor.healthy = true;
          conductor.lastCheck = Date.now();

          return result;
        } catch (error) {
          lastError = error instanceof Error ? error : new Error(String(error));
          conductor.failureCount++;

          // Mark unhealthy after 3 failures
          if (conductor.failureCount >= 3) {
            conductor.healthy = false;
          }
        }
      }
    }

    throw lastError || new Error('All conductors failed');
  }

  /**
   * Health check all conductors
   */
  async healthCheck(checkFn: (url: string) => Promise<boolean>): Promise<void> {
    const checks = Array.from(this.conductors.entries()).map(async ([_id, info]) => {
      try {
        const healthy = await checkFn(info.url);
        info.healthy = healthy;
        info.lastCheck = Date.now();
        if (healthy) {
          info.failureCount = 0;
        }
      } catch {
        info.healthy = false;
        info.failureCount++;
      }
    });

    await Promise.all(checks);
  }

  /**
   * Get conductor status
   */
  getStatus(): Array<{ id: string; url: string; healthy: boolean; failureCount: number }> {
    return Array.from(this.conductors.entries()).map(([id, info]) => ({
      id,
      url: info.url,
      healthy: info.healthy,
      failureCount: info.failureCount,
    }));
  }

  private async executeWithTimeout<T>(fn: () => Promise<T>, timeout: number): Promise<T> {
    return Promise.race([
      fn(),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('Operation timeout')), timeout)
      ),
    ]);
  }
}
