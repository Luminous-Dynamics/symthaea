// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix SDK Security Module
 *
 * Provides cryptographic utilities, rate limiting, input sanitization,
 * and secure handling of sensitive data for Byzantine-resistant systems.
 */

import { MycelixError, validate, type ErrorCode } from '../errors.js';

// ============================================================================
// Security Error Types
// ============================================================================

export enum SecurityErrorCode {
  RATE_LIMITED = 7001,
  INVALID_SIGNATURE = 7002,
  HASH_MISMATCH = 7003,
  ENTROPY_INSUFFICIENT = 7004,
  SECRET_EXPIRED = 7005,
  INPUT_SANITIZATION_FAILED = 7006,
  TIMING_ATTACK_DETECTED = 7007,
}

export class SecurityError extends MycelixError {
  constructor(
    message: string,
    code: SecurityErrorCode,
    context: Record<string, unknown> = {}
  ) {
    super(message, code as unknown as ErrorCode, context);
    this.name = 'SecurityError';
  }
}

// ============================================================================
// Secure Random Number Generation
// ============================================================================

/**
 * Cryptographically secure random bytes using Web Crypto API
 * Falls back to Node.js crypto if available
 */
export function secureRandomBytes(length: number): Uint8Array {
  validate().positive('length', length).throwIfInvalid();

  // Use Web Crypto API (works in both browser and Node.js 20+)
  if (typeof globalThis.crypto !== 'undefined' && globalThis.crypto.getRandomValues) {
    const bytes = new Uint8Array(length);
    globalThis.crypto.getRandomValues(bytes);
    return bytes;
  }

  // Fallback for older Node.js
  try {
    // Dynamic import to avoid bundling issues
    const crypto = require('crypto');
    return new Uint8Array(crypto.randomBytes(length));
  } catch {
    throw new SecurityError(
      'No secure random source available',
      SecurityErrorCode.ENTROPY_INSUFFICIENT
    );
  }
}

/**
 * Generate a secure random number in range [0, 1)
 */
export function secureRandomFloat(): number {
  const bytes = secureRandomBytes(8);
  const view = new DataView(bytes.buffer);
  // Use all 64 bits for maximum precision
  const high = view.getUint32(0) >>> 0;
  const low = view.getUint32(4) >>> 0;
  // Combine to get a number in [0, 1)
  return (high * 0x100000000 + low) / 0x10000000000000000;
}

/**
 * Generate a secure random integer in range [min, max]
 */
export function secureRandomInt(min: number, max: number): number {
  validate()
    .custom('min', min, (v) => Number.isInteger(v), 'must be an integer')
    .custom('max', max, (v) => Number.isInteger(v), 'must be an integer')
    .custom('range', max, (v) => (v as number) >= min, 'max must be >= min')
    .throwIfInvalid();

  const range = max - min + 1;
  const bytesNeeded = Math.ceil(Math.log2(range) / 8) || 1;
  const maxValid = Math.pow(256, bytesNeeded);
  const limit = maxValid - (maxValid % range);

  let result: number;
  do {
    const bytes = secureRandomBytes(bytesNeeded);
    result = bytes.reduce((acc, byte, i) => acc + byte * Math.pow(256, i), 0);
  } while (result >= limit);

  return min + (result % range);
}

/**
 * Generate a cryptographically secure UUID v4
 */
export function secureUUID(): string {
  const bytes = secureRandomBytes(16);
  // Set version (4) and variant (RFC 4122)
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;

  const hex = Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');

  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

/**
 * Shuffle an array using Fisher-Yates with secure randomness
 */
export function secureShuffle<T>(array: T[]): T[] {
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = secureRandomInt(0, i);
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

/**
 * @deprecated Use `secureShuffle` instead (typo fix)
 */
export const secureShufffle = secureShuffle;

// ============================================================================
// Cryptographic Hashing (using Web Crypto API)
// ============================================================================

export type HashAlgorithm = 'SHA-256' | 'SHA-384' | 'SHA-512';

/**
 * Compute a cryptographic hash of data
 */
export async function hash(
  data: string | Uint8Array,
  algorithm: HashAlgorithm = 'SHA-256'
): Promise<Uint8Array> {
  const encoder = new TextEncoder();
  const dataBytes = typeof data === 'string' ? encoder.encode(data) : new Uint8Array(data);

  if (typeof globalThis.crypto?.subtle !== 'undefined') {
    const hashBuffer = await globalThis.crypto.subtle.digest(algorithm, dataBytes.buffer.slice(dataBytes.byteOffset, dataBytes.byteOffset + dataBytes.byteLength));
    return new Uint8Array(hashBuffer);
  }

  // Fallback to Node.js crypto
  try {
    const crypto = require('crypto');
    const nodeAlg = algorithm.toLowerCase().replace('-', '');
    const hashObj = crypto.createHash(nodeAlg);
    hashObj.update(dataBytes);
    return new Uint8Array(hashObj.digest());
  } catch {
    throw new SecurityError(
      'No cryptographic hash implementation available',
      SecurityErrorCode.ENTROPY_INSUFFICIENT
    );
  }
}

/**
 * Compute a hash and return as hex string
 */
export async function hashHex(
  data: string | Uint8Array,
  algorithm: HashAlgorithm = 'SHA-256'
): Promise<string> {
  const bytes = await hash(data, algorithm);
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Compute HMAC using Web Crypto API
 */
export async function hmac(
  key: Uint8Array,
  data: string | Uint8Array,
  algorithm: HashAlgorithm = 'SHA-256'
): Promise<Uint8Array> {
  const encoder = new TextEncoder();
  const dataBytes = typeof data === 'string' ? encoder.encode(data) : new Uint8Array(data);
  const keyBytes = new Uint8Array(key);

  if (typeof globalThis.crypto?.subtle !== 'undefined') {
    const keyBuffer = keyBytes.buffer.slice(keyBytes.byteOffset, keyBytes.byteOffset + keyBytes.byteLength);
    const dataBuffer = dataBytes.buffer.slice(dataBytes.byteOffset, dataBytes.byteOffset + dataBytes.byteLength);

    const cryptoKey = await globalThis.crypto.subtle.importKey(
      'raw',
      keyBuffer,
      { name: 'HMAC', hash: algorithm },
      false,
      ['sign']
    );
    const signature = await globalThis.crypto.subtle.sign('HMAC', cryptoKey, dataBuffer);
    return new Uint8Array(signature);
  }

  // Fallback to Node.js crypto
  try {
    const crypto = require('crypto');
    const nodeAlg = algorithm.toLowerCase().replace('-', '');
    const hmacObj = crypto.createHmac(nodeAlg, key);
    hmacObj.update(dataBytes);
    return new Uint8Array(hmacObj.digest());
  } catch {
    throw new SecurityError(
      'No HMAC implementation available',
      SecurityErrorCode.ENTROPY_INSUFFICIENT
    );
  }
}

/**
 * Constant-time comparison to prevent timing attacks
 */
export function constantTimeEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) {
    return false;
  }

  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a[i] ^ b[i];
  }
  return result === 0;
}

/**
 * Verify HMAC with constant-time comparison
 */
export async function verifyHmac(
  key: Uint8Array,
  data: string | Uint8Array,
  expectedMac: Uint8Array,
  algorithm: HashAlgorithm = 'SHA-256'
): Promise<boolean> {
  const computedMac = await hmac(key, data, algorithm);
  return constantTimeEqual(computedMac, expectedMac);
}

// ============================================================================
// Rate Limiting
// ============================================================================

export interface RateLimitConfig {
  /** Maximum number of requests in the window */
  maxRequests: number;
  /** Time window in milliseconds */
  windowMs: number;
  /** Whether to use sliding window (more accurate but more memory) */
  slidingWindow?: boolean;
}

export interface RateLimitState {
  id: string;
  config: RateLimitConfig;
  timestamps: number[];
  blocked: boolean;
  blockedUntil?: number;
}

/**
 * Create a new rate limiter for an entity
 */
export function createRateLimiter(
  id: string,
  config: RateLimitConfig
): RateLimitState {
  validate()
    .notEmpty('id', id)
    .positive('maxRequests', config.maxRequests)
    .positive('windowMs', config.windowMs)
    .throwIfInvalid();

  return {
    id,
    config,
    timestamps: [],
    blocked: false,
  };
}

/**
 * Check if a request should be allowed
 */
export function checkRateLimit(state: RateLimitState): {
  allowed: boolean;
  remaining: number;
  resetAt: number;
  state: RateLimitState;
} {
  const now = Date.now();
  const windowStart = now - state.config.windowMs;

  // Check if currently blocked
  if (state.blocked && state.blockedUntil && now < state.blockedUntil) {
    return {
      allowed: false,
      remaining: 0,
      resetAt: state.blockedUntil,
      state,
    };
  }

  // Filter timestamps within the window
  const validTimestamps = state.config.slidingWindow
    ? state.timestamps.filter((t) => t > windowStart)
    : state.timestamps.filter((t) => t > windowStart);

  const requestsInWindow = validTimestamps.length;
  const allowed = requestsInWindow < state.config.maxRequests;

  if (allowed) {
    // Add current timestamp
    const newTimestamps = [...validTimestamps, now];
    return {
      allowed: true,
      remaining: state.config.maxRequests - requestsInWindow - 1,
      resetAt: now + state.config.windowMs,
      state: { ...state, timestamps: newTimestamps, blocked: false },
    };
  }

  // Rate limited - block until window expires
  const blockedUntil = validTimestamps[0] + state.config.windowMs;
  return {
    allowed: false,
    remaining: 0,
    resetAt: blockedUntil,
    state: { ...state, timestamps: validTimestamps, blocked: true, blockedUntil },
  };
}

/**
 * Rate limiter with automatic cleanup for multiple entities
 */
/**
 * Metrics for a single rate limiter
 */
export interface RateLimitMetrics {
  /** Entity identifier */
  id: string;
  /** Total requests attempted */
  totalRequests: number;
  /** Requests that were rejected */
  rejectedRequests: number;
  /** Peak requests in any window */
  peakRequestsInWindow: number;
  /** Number of window resets */
  windowResets: number;
  /** Whether currently rate limited */
  isRateLimited: boolean;
  /** First request timestamp */
  firstRequestAt: number;
  /** Last request timestamp */
  lastRequestAt: number;
}

interface MetricsState {
  totalRequests: number;
  rejectedRequests: number;
  peakRequestsInWindow: number;
  windowResets: number;
  firstRequestAt: number;
  lastRequestAt: number;
  lastWindowStart: number;
}

export class RateLimiterRegistry {
  private limiters = new Map<string, RateLimitState>();
  private metricsMap = new Map<string, MetricsState>();
  private defaultConfig: RateLimitConfig;

  constructor(defaultConfig: RateLimitConfig) {
    this.defaultConfig = defaultConfig;
  }

  /**
   * Initialize or get metrics for an entity
   */
  private getOrCreateMetrics(id: string): MetricsState {
    let metrics = this.metricsMap.get(id);
    if (!metrics) {
      const now = Date.now();
      metrics = {
        totalRequests: 0,
        rejectedRequests: 0,
        peakRequestsInWindow: 0,
        windowResets: 0,
        firstRequestAt: now,
        lastRequestAt: now,
        lastWindowStart: now,
      };
      this.metricsMap.set(id, metrics);
    }
    return metrics;
  }

  /**
   * Check rate limit for an entity, creating limiter if needed
   */
  check(id: string, config?: RateLimitConfig): {
    allowed: boolean;
    remaining: number;
    resetAt: number;
  } {
    let state = this.limiters.get(id);
    if (!state) {
      state = createRateLimiter(id, config || this.defaultConfig);
    }

    // Update metrics
    const metrics = this.getOrCreateMetrics(id);
    const now = Date.now();
    metrics.totalRequests++;
    metrics.lastRequestAt = now;

    // Check for window reset
    const windowMs = state.config.windowMs;
    if (now - metrics.lastWindowStart >= windowMs) {
      metrics.windowResets++;
      metrics.lastWindowStart = now;
    }

    const result = checkRateLimit(state);
    this.limiters.set(id, result.state);

    // Track rejections and peak
    if (!result.allowed) {
      metrics.rejectedRequests++;
    }

    // Update peak requests in current window
    const currentWindowRequests = result.state.timestamps.length;
    if (currentWindowRequests > metrics.peakRequestsInWindow) {
      metrics.peakRequestsInWindow = currentWindowRequests;
    }

    return {
      allowed: result.allowed,
      remaining: result.remaining,
      resetAt: result.resetAt,
    };
  }

  /**
   * Reset rate limit for an entity
   */
  reset(id: string): void {
    this.limiters.delete(id);
  }

  /**
   * Reset metrics for an entity
   */
  resetMetrics(id: string): void {
    this.metricsMap.delete(id);
  }

  /**
   * Clean up expired limiters to prevent memory leaks
   */
  cleanup(): number {
    const now = Date.now();
    let cleaned = 0;

    for (const [id, state] of this.limiters) {
      const oldestValid = now - state.config.windowMs;
      const hasValidTimestamps = state.timestamps.some((t) => t > oldestValid);

      if (!hasValidTimestamps && !state.blocked) {
        this.limiters.delete(id);
        cleaned++;
      }
    }

    return cleaned;
  }

  /**
   * Get metrics for a specific entity
   */
  getMetrics(id: string): RateLimitMetrics | null {
    const metrics = this.metricsMap.get(id);
    const state = this.limiters.get(id);

    if (!metrics) {
      return null;
    }

    return {
      id,
      totalRequests: metrics.totalRequests,
      rejectedRequests: metrics.rejectedRequests,
      peakRequestsInWindow: metrics.peakRequestsInWindow,
      windowResets: metrics.windowResets,
      isRateLimited: state?.blocked ?? false,
      firstRequestAt: metrics.firstRequestAt,
      lastRequestAt: metrics.lastRequestAt,
    };
  }

  /**
   * Get metrics for all tracked entities
   */
  getAllMetrics(): Map<string, RateLimitMetrics> {
    const result = new Map<string, RateLimitMetrics>();

    for (const id of this.metricsMap.keys()) {
      const metrics = this.getMetrics(id);
      if (metrics) {
        result.set(id, metrics);
      }
    }

    return result;
  }

  /**
   * Get aggregate summary metrics across all limiters
   */
  getSummary(): {
    totalEntities: number;
    totalRequests: number;
    totalRejected: number;
    rejectionRate: number;
    currentlyRateLimited: number;
  } {
    let totalRequests = 0;
    let totalRejected = 0;
    let currentlyRateLimited = 0;

    for (const [id, metrics] of this.metricsMap) {
      totalRequests += metrics.totalRequests;
      totalRejected += metrics.rejectedRequests;

      const state = this.limiters.get(id);
      if (state?.blocked) {
        currentlyRateLimited++;
      }
    }

    return {
      totalEntities: this.metricsMap.size,
      totalRequests,
      totalRejected,
      rejectionRate: totalRequests > 0 ? totalRejected / totalRequests : 0,
      currentlyRateLimited,
    };
  }

  /**
   * Export all metrics as JSON string
   */
  exportMetrics(): string {
    const allMetrics: Record<string, RateLimitMetrics> = {};
    for (const [id, metrics] of this.getAllMetrics()) {
      allMetrics[id] = metrics;
    }
    return JSON.stringify({
      exportedAt: Date.now(),
      summary: this.getSummary(),
      entities: allMetrics,
    }, null, 2);
  }

  /**
   * Clear all metrics (but keep rate limiters)
   */
  clearMetrics(): void {
    this.metricsMap.clear();
  }
}

// ============================================================================
// Input Sanitization
// ============================================================================

/**
 * Sanitize a string to prevent XSS and injection attacks
 */
export function sanitizeString(input: string, options: {
  maxLength?: number;
  allowedPattern?: RegExp;
  stripHtml?: boolean;
  stripControl?: boolean;
} = {}): string {
  const {
    maxLength = 10000,
    stripHtml = true,
    stripControl = true,
  } = options;

  let result = input;

  // Enforce max length
  if (result.length > maxLength) {
    result = result.slice(0, maxLength);
  }

  // Strip HTML tags if requested
  if (stripHtml) {
    result = result.replace(/<[^>]*>/g, '');
  }

  // Strip control characters (except newlines and tabs)
  if (stripControl) {
    result = result.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
  }

  // Check against allowed pattern if provided
  if (options.allowedPattern && !options.allowedPattern.test(result)) {
    throw new SecurityError(
      'Input does not match allowed pattern',
      SecurityErrorCode.INPUT_SANITIZATION_FAILED,
      { input: result.slice(0, 100) }
    );
  }

  return result;
}

/**
 * Sanitize an identifier (alphanumeric + underscore/hyphen only)
 */
export function sanitizeId(input: string): string {
  const sanitized = input.replace(/[^a-zA-Z0-9_-]/g, '');
  if (sanitized.length === 0) {
    throw new SecurityError(
      'Invalid identifier after sanitization',
      SecurityErrorCode.INPUT_SANITIZATION_FAILED,
      { original: input.slice(0, 50) }
    );
  }
  return sanitized;
}

/**
 * Validate and sanitize a JSON string
 */
export function sanitizeJson<T>(
  input: string,
  maxDepth: number = 10,
  maxSize: number = 1000000
): T {
  if (input.length > maxSize) {
    throw new SecurityError(
      `JSON exceeds maximum size of ${maxSize} bytes`,
      SecurityErrorCode.INPUT_SANITIZATION_FAILED
    );
  }

  // Parse with depth check
  const checkDepth = (obj: unknown, depth: number): void => {
    if (depth > maxDepth) {
      throw new SecurityError(
        `JSON exceeds maximum depth of ${maxDepth}`,
        SecurityErrorCode.INPUT_SANITIZATION_FAILED
      );
    }
    if (Array.isArray(obj)) {
      obj.forEach((item) => checkDepth(item, depth + 1));
    } else if (obj !== null && typeof obj === 'object') {
      Object.values(obj).forEach((val) => checkDepth(val, depth + 1));
    }
  };

  try {
    const parsed = JSON.parse(input);
    checkDepth(parsed, 0);
    return parsed;
  } catch (error) {
    if (error instanceof SecurityError) {
      throw error;
    }
    throw new SecurityError(
      'Invalid JSON',
      SecurityErrorCode.INPUT_SANITIZATION_FAILED,
      { error: String(error) }
    );
  }
}

// ============================================================================
// Secure Secret Storage
// ============================================================================

/**
 * A secret value that clears itself after use or timeout
 */
export class SecureSecret {
  private value: Uint8Array;
  private cleared = false;
  private readonly expiresAt?: number;
  private clearTimer?: ReturnType<typeof setTimeout>;
  private _useCount = 0;

  constructor(value: string | Uint8Array, ttlMs?: number) {
    const encoder = new TextEncoder();
    this.value = typeof value === 'string' ? encoder.encode(value) : new Uint8Array(value);

    if (ttlMs) {
      this.expiresAt = Date.now() + ttlMs;
      this.clearTimer = setTimeout(() => this.clear(), ttlMs);
    }
  }

  /**
   * Number of times the secret has been accessed
   */
  get useCount(): number {
    return this._useCount;
  }

  /**
   * Whether the secret has been destroyed
   */
  get isDestroyed(): boolean {
    return this.cleared;
  }

  /**
   * Get the secret value (use immediately, don't store)
   */
  expose(): Uint8Array {
    if (this.cleared) {
      throw new SecurityError(
        'Secret has been cleared',
        SecurityErrorCode.SECRET_EXPIRED
      );
    }
    if (this.expiresAt && Date.now() > this.expiresAt) {
      this.clear();
      throw new SecurityError(
        'Secret has expired',
        SecurityErrorCode.SECRET_EXPIRED
      );
    }
    this._useCount++;
    return this.value;
  }

  /**
   * Get the secret as a string
   */
  exposeString(): string {
    const decoder = new TextDecoder();
    return decoder.decode(this.expose());
  }

  /**
   * Use the secret in a callback and auto-clear afterward (safe single-use pattern)
   */
  async use<T>(fn: (secret: Uint8Array) => T | Promise<T>): Promise<T> {
    try {
      const result = await fn(this.expose());
      return result;
    } finally {
      this.clear();
    }
  }

  /**
   * Use the secret without auto-clearing (for multiple uses)
   * Remember to call clear() when done!
   */
  peek<T>(fn: (secret: Uint8Array) => T): T {
    return fn(this.expose());
  }

  /**
   * Use the secret in an async callback without auto-clearing
   * Remember to call clear() when done!
   */
  async peekAsync<T>(fn: (secret: Uint8Array) => Promise<T>): Promise<T> {
    return await fn(this.expose());
  }

  /**
   * Clear the secret from memory
   */
  clear(): void {
    if (!this.cleared) {
      // Overwrite with random data before zeroing
      globalThis.crypto?.getRandomValues?.(this.value);
      this.value.fill(0);
      this.cleared = true;
      if (this.clearTimer) {
        clearTimeout(this.clearTimer);
      }
    }
  }

  /**
   * Destroy the secret (alias for clear)
   */
  destroy(): void {
    this.clear();
  }

  /**
   * Check if the secret is still valid
   */
  isValid(): boolean {
    return !this.cleared && (!this.expiresAt || Date.now() <= this.expiresAt);
  }
}

// ============================================================================
// Byzantine Resistance Utilities
// ============================================================================

/**
 * Threshold signature verification (simplified conceptual implementation)
 * In production, use a proper threshold cryptography library
 */
export interface ThresholdConfig {
  /** Total number of signers */
  n: number;
  /** Minimum required signers (threshold) */
  t: number;
}

/**
 * Validate Byzantine fault tolerance parameters
 */
export function validateBftParams(
  n: number,
  f: number
): { valid: boolean; message?: string } {
  // Standard BFT requirement: n >= 3f + 1
  if (n < 3 * f + 1) {
    return {
      valid: false,
      message: `Need at least ${3 * f + 1} nodes to tolerate ${f} Byzantine failures, but only have ${n}`,
    };
  }
  return { valid: true };
}

/**
 * Calculate maximum tolerable Byzantine failures for a network size
 */
export function maxByzantineFailures(n: number): number {
  return Math.floor((n - 1) / 3);
}

/**
 * Check if enough honest nodes remain after failures
 */
export function hasQuorum(
  totalNodes: number,
  byzantineNodes: number,
  requiredHonest: number
): boolean {
  const honestNodes = totalNodes - byzantineNodes;
  return honestNodes >= requiredHonest;
}

// ============================================================================
// Timing Attack Protection
// ============================================================================

/**
 * Add random delay to prevent timing attacks
 */
export async function randomDelay(
  minMs: number = 0,
  maxMs: number = 100
): Promise<void> {
  const delay = secureRandomInt(minMs, maxMs);
  await new Promise((resolve) => setTimeout(resolve, delay));
}

/**
 * Execute a function with a minimum duration to prevent timing attacks
 */
export async function constantTime<T>(
  fn: () => T | Promise<T>,
  minDurationMs: number = 50
): Promise<T> {
  const start = Date.now();
  const result = await fn();
  const elapsed = Date.now() - start;

  if (elapsed < minDurationMs) {
    await new Promise((resolve) => setTimeout(resolve, minDurationMs - elapsed));
  }

  return result;
}

// ============================================================================
// Security Audit Logging
// ============================================================================

export enum SecurityEventType {
  RATE_LIMIT_EXCEEDED = 'RATE_LIMIT_EXCEEDED',
  INVALID_INPUT = 'INVALID_INPUT',
  AUTHENTICATION_FAILED = 'AUTHENTICATION_FAILED',
  AUTH_SUCCESS = 'AUTH_SUCCESS',
  AUTHORIZATION_FAILED = 'AUTHORIZATION_FAILED',
  SUSPICIOUS_ACTIVITY = 'SUSPICIOUS_ACTIVITY',
  SECRET_ACCESSED = 'SECRET_ACCESSED',
  HASH_VERIFICATION_FAILED = 'HASH_VERIFICATION_FAILED',
  CRYPTO_OPERATION = 'CRYPTO_OPERATION',
  BFT_VIOLATION = 'BFT_VIOLATION',
}

export interface SecurityEvent {
  type: SecurityEventType;
  timestamp: number;
  entityId?: string;
  details: Record<string, unknown>;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export type SecurityEventHandler = (event: SecurityEvent) => void;

/**
 * Input for logging a security event
 */
export interface SecurityEventInput {
  type: SecurityEventType;
  message: string;
  severity?: 'low' | 'medium' | 'high' | 'critical';
  entityId?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Security event emitter for audit logging
 */
export class SecurityAuditLog {
  private handlers: SecurityEventHandler[] = [];
  private events: SecurityEvent[] = [];
  private maxEvents: number;

  constructor(maxEvents: number = 1000) {
    this.maxEvents = maxEvents;
  }

  /**
   * Register an event handler
   */
  onEvent(handler: SecurityEventHandler): void {
    this.handlers.push(handler);
  }

  /**
   * Log a security event (flexible signature)
   */
  log(input: SecurityEventInput): void;
  log(
    type: SecurityEventType,
    details: Record<string, unknown>,
    severity?: SecurityEvent['severity'],
    entityId?: string
  ): void;
  log(
    typeOrInput: SecurityEventType | SecurityEventInput,
    details?: Record<string, unknown>,
    severity: SecurityEvent['severity'] = 'medium',
    entityId?: string
  ): void {
    let event: SecurityEvent;

    if (typeof typeOrInput === 'object') {
      // New interface style
      event = {
        type: typeOrInput.type,
        timestamp: Date.now(),
        entityId: typeOrInput.entityId,
        details: {
          message: typeOrInput.message,
          ...typeOrInput.metadata,
        },
        severity: typeOrInput.severity || 'medium',
      };
    } else {
      // Legacy interface style
      event = {
        type: typeOrInput,
        timestamp: Date.now(),
        entityId,
        details: details || {},
        severity,
      };
    }

    this.events.push(event);
    if (this.events.length > this.maxEvents) {
      this.events.shift();
    }

    for (const handler of this.handlers) {
      try {
        handler(event);
      } catch {
        // Don't let handler errors break logging
      }
    }
  }

  /**
   * Get recent events
   */
  getEvents(
    filter?: Partial<Pick<SecurityEvent, 'type' | 'severity' | 'entityId'>>
  ): SecurityEvent[] {
    if (!filter) {
      return [...this.events];
    }

    return this.events.filter((e) => {
      if (filter.type && e.type !== filter.type) return false;
      if (filter.severity && e.severity !== filter.severity) return false;
      if (filter.entityId && e.entityId !== filter.entityId) return false;
      return true;
    });
  }

  /**
   * Get events by type
   */
  getEventsByType(type: SecurityEventType): SecurityEvent[] {
    return this.events.filter((e) => e.type === type);
  }

  /**
   * Clear all events
   */
  clear(): void {
    this.events = [];
  }

  /**
   * Get current event count
   */
  get eventCount(): number {
    return this.events.length;
  }
}

// Global audit log instance
export const securityAudit = new SecurityAuditLog();

// ============================================================================
// Exports
// ============================================================================

export const security = {
  // Random
  randomBytes: secureRandomBytes,
  secureRandomBytes,
  randomFloat: secureRandomFloat,
  randomInt: secureRandomInt,
  uuid: secureUUID,
  secureUUID,
  shuffle: secureShuffle,
  secureShuffle,
  /** @deprecated Use `secureShuffle` instead */
  secureShufffle,

  // Hashing
  hash,
  hashHex,
  hmac,
  verifyHmac,
  constantTimeEqual,

  // Rate limiting
  createRateLimiter,
  checkRateLimit,
  RateLimiterRegistry,

  // Sanitization
  sanitizeString,
  sanitizeId,
  sanitizeJson,

  // Secrets
  SecureSecret,

  // BFT utilities
  validateBftParams,
  maxByzantineFailures,
  hasQuorum,

  // Timing protection
  randomDelay,
  constantTime,

  // Audit
  SecurityAuditLog,
  securityAudit,
};
