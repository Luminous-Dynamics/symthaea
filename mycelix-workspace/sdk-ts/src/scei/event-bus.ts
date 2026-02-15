/**
 * @mycelix/sdk SCEI Cross-Module Event Bus
 *
 * Central event bus for communication between SCEI modules.
 * Enables feedback loops like:
 * - Calibration learns from failed claims (Metabolism → Calibration)
 * - Gap detector creates placeholder claims (Discovery → Metabolism)
 * - Propagation results feed back to calibration
 *
 * @packageDocumentation
 * @module scei/event-bus
 */

// ============================================================================
// EVENT TYPES
// ============================================================================

/**
 * All possible SCEI event types
 */
export type SCEIEventType =
  // Calibration events
  | 'calibration:prediction_recorded'
  | 'calibration:prediction_resolved'
  | 'calibration:report_generated'
  | 'calibration:calibration_drift_detected'
  // Metabolism events
  | 'metabolism:claim_born'
  | 'metabolism:claim_matured'
  | 'metabolism:claim_died'
  | 'metabolism:health_updated'
  | 'metabolism:tombstone_created'
  | 'metabolism:evidence_added'
  // Propagation events
  | 'propagation:started'
  | 'propagation:completed'
  | 'propagation:blocked'
  | 'propagation:quarantined'
  | 'propagation:human_review_requested'
  // Discovery events
  | 'discovery:gap_detected'
  | 'discovery:gap_resolved'
  | 'discovery:gap_dismissed'
  | 'discovery:scan_completed'
  // Algebra events
  | 'algebra:chain_degraded'
  | 'algebra:claims_corroborated'
  | 'algebra:claims_conjoined'
  // Cross-module coordination
  | 'scei:module_initialized'
  | 'scei:module_error'
  | 'scei:integrity_check_failed';

/**
 * Base event structure
 */
export interface SCEIEvent<T extends SCEIEventType = SCEIEventType, P = unknown> {
  type: T;
  timestamp: number;
  source: SCEIModule;
  correlationId?: string;
  payload: P;
}

/**
 * SCEI module identifiers
 */
export type SCEIModule =
  | 'calibration'
  | 'metabolism'
  | 'propagation'
  | 'discovery'
  | 'algebra'
  | 'event-bus';

// ============================================================================
// SPECIFIC EVENT PAYLOADS
// ============================================================================

export interface CalibrationPredictionRecordedPayload {
  predictionId: string;
  confidence: number;
  domain: string;
  source: string;
}

export interface CalibrationPredictionResolvedPayload {
  predictionId: string;
  confidence: number;
  outcome: boolean;
  domain: string;
  brierContribution: number;
}

export interface MetabolismClaimBornPayload {
  claimId: string;
  content: string;
  domain: string;
  authorId: string;
  initialConfidence: number;
}

export interface MetabolismClaimDiedPayload {
  claimId: string;
  reason: string;
  finalConfidence: number;
  lifespan: number;
  tombstoneId: string;
}

export interface MetabolismHealthUpdatedPayload {
  claimId: string;
  previousHealth: number;
  newHealth: number;
  phase: string;
}

export interface PropagationCompletedPayload {
  requestId: string;
  sourceClaimId: string;
  affectedClaimCount: number;
  maxDepth: number;
  wasBlocked: boolean;
  blockReason?: string;
}

export interface DiscoveryGapDetectedPayload {
  gapId: string;
  type: string;
  domain: string;
  severity: number;
  description: string;
  suggestedActions: string[];
}

export interface DiscoveryGapResolvedPayload {
  gapId: string;
  resolvedBy: string;
  method: 'addressed' | 'dismissed' | 'auto_resolved';
}

// ============================================================================
// PAYLOAD TYPE MAP (for type-safe validation)
// ============================================================================

/**
 * Maps event types to their payload types for compile-time safety
 */
export interface SCEIEventPayloadMap {
  'calibration:prediction_recorded': CalibrationPredictionRecordedPayload;
  'calibration:prediction_resolved': CalibrationPredictionResolvedPayload;
  'calibration:report_generated': { scope: string; brierScore: number };
  'calibration:calibration_drift_detected': { domain: string; drift: number };
  'metabolism:claim_born': MetabolismClaimBornPayload;
  'metabolism:claim_matured': { claimId: string; newPhase: string };
  'metabolism:claim_died': MetabolismClaimDiedPayload;
  'metabolism:health_updated': MetabolismHealthUpdatedPayload;
  'metabolism:tombstone_created': { claimId: string; tombstoneId: string };
  'metabolism:evidence_added': { claimId: string; evidenceId: string; weight: number };
  'propagation:started': { requestId: string; sourceClaimId: string };
  'propagation:completed': PropagationCompletedPayload;
  'propagation:blocked': { requestId: string; reason: string };
  'propagation:quarantined': { requestId: string; reason: string };
  'propagation:human_review_requested': { requestId: string; claimIds: string[] };
  'discovery:gap_detected': DiscoveryGapDetectedPayload;
  'discovery:gap_resolved': DiscoveryGapResolvedPayload;
  'discovery:gap_dismissed': { gapId: string; reason: string };
  'discovery:scan_completed': { domain: string; gapsFound: number };
  'algebra:chain_degraded': { chainId: string; degradation: number };
  'algebra:claims_corroborated': { claimIds: string[]; newConfidence: number };
  'algebra:claims_conjoined': { claimIds: string[]; resultId: string };
  'scei:module_initialized': { module: SCEIModule };
  'scei:module_error': { module: SCEIModule; error: string };
  'scei:integrity_check_failed': { module: SCEIModule; check: string };
}

// ============================================================================
// PAYLOAD VALIDATORS
// ============================================================================

type PayloadValidator<T> = (payload: unknown) => payload is T;

const payloadValidators: Partial<Record<SCEIEventType, PayloadValidator<unknown>>> = {
  'calibration:prediction_recorded': (p): p is CalibrationPredictionRecordedPayload => {
    const x = p as CalibrationPredictionRecordedPayload;
    return (
      typeof x === 'object' &&
      x !== null &&
      typeof x.predictionId === 'string' &&
      typeof x.confidence === 'number' &&
      typeof x.domain === 'string' &&
      typeof x.source === 'string'
    );
  },
  'calibration:prediction_resolved': (p): p is CalibrationPredictionResolvedPayload => {
    const x = p as CalibrationPredictionResolvedPayload;
    return (
      typeof x === 'object' &&
      x !== null &&
      typeof x.predictionId === 'string' &&
      typeof x.confidence === 'number' &&
      typeof x.outcome === 'boolean' &&
      typeof x.domain === 'string' &&
      typeof x.brierContribution === 'number'
    );
  },
  'metabolism:claim_born': (p): p is MetabolismClaimBornPayload => {
    const x = p as MetabolismClaimBornPayload;
    return (
      typeof x === 'object' &&
      x !== null &&
      typeof x.claimId === 'string' &&
      typeof x.content === 'string' &&
      typeof x.domain === 'string' &&
      typeof x.authorId === 'string' &&
      typeof x.initialConfidence === 'number'
    );
  },
  'metabolism:claim_died': (p): p is MetabolismClaimDiedPayload => {
    const x = p as MetabolismClaimDiedPayload;
    return (
      typeof x === 'object' &&
      x !== null &&
      typeof x.claimId === 'string' &&
      typeof x.reason === 'string' &&
      typeof x.finalConfidence === 'number' &&
      typeof x.lifespan === 'number' &&
      typeof x.tombstoneId === 'string'
    );
  },
  'metabolism:health_updated': (p): p is MetabolismHealthUpdatedPayload => {
    const x = p as MetabolismHealthUpdatedPayload;
    return (
      typeof x === 'object' &&
      x !== null &&
      typeof x.claimId === 'string' &&
      typeof x.previousHealth === 'number' &&
      typeof x.newHealth === 'number' &&
      typeof x.phase === 'string'
    );
  },
  'propagation:completed': (p): p is PropagationCompletedPayload => {
    const x = p as PropagationCompletedPayload;
    return (
      typeof x === 'object' &&
      x !== null &&
      typeof x.requestId === 'string' &&
      typeof x.sourceClaimId === 'string' &&
      typeof x.affectedClaimCount === 'number' &&
      typeof x.maxDepth === 'number' &&
      typeof x.wasBlocked === 'boolean'
    );
  },
  'discovery:gap_detected': (p): p is DiscoveryGapDetectedPayload => {
    const x = p as DiscoveryGapDetectedPayload;
    return (
      typeof x === 'object' &&
      x !== null &&
      typeof x.gapId === 'string' &&
      typeof x.type === 'string' &&
      typeof x.domain === 'string' &&
      typeof x.severity === 'number' &&
      typeof x.description === 'string' &&
      Array.isArray(x.suggestedActions)
    );
  },
  'discovery:gap_resolved': (p): p is DiscoveryGapResolvedPayload => {
    const x = p as DiscoveryGapResolvedPayload;
    return (
      typeof x === 'object' &&
      x !== null &&
      typeof x.gapId === 'string' &&
      typeof x.resolvedBy === 'string' &&
      ['addressed', 'dismissed', 'auto_resolved'].includes(x.method)
    );
  },
};

/**
 * Validate event payload against its schema
 */
export function validateEventPayload<T extends SCEIEventType>(
  type: T,
  payload: unknown
): { valid: boolean; error?: string } {
  const validator = payloadValidators[type];
  if (!validator) {
    // No validator defined - accept any payload
    return { valid: true };
  }

  if (validator(payload)) {
    return { valid: true };
  }

  return {
    valid: false,
    error: `Invalid payload for event type "${type}"`,
  };
}

// ============================================================================
// EVENT LISTENER TYPES
// ============================================================================

export type SCEIEventListener<T extends SCEIEventType = SCEIEventType> = (
  event: SCEIEvent<T>
) => void | Promise<void>;

export type SCEIEventFilter = {
  types?: SCEIEventType[];
  sources?: SCEIModule[];
  correlationId?: string;
};

export interface Subscription {
  id: string;
  unsubscribe: () => void;
}

/**
 * Listener timeout error
 */
export class ListenerTimeoutError extends Error {
  constructor(
    public readonly listenerId: string,
    public readonly timeoutMs: number
  ) {
    super(`Listener ${listenerId} timed out after ${timeoutMs}ms`);
    this.name = 'ListenerTimeoutError';
  }
}

// ============================================================================
// EVENT BUS IMPLEMENTATION
// ============================================================================

/**
 * SCEI Event Bus
 *
 * Central event hub for cross-module communication. Supports:
 * - Type-safe event emission and subscription
 * - Event filtering by type, source, or correlation ID
 * - Async event handlers
 * - Event history for debugging
 * - Metrics collection
 *
 * @example
 * ```typescript
 * const bus = getSCEIEventBus();
 *
 * // Subscribe to all metabolism events
 * bus.subscribe(
 *   (event) => console.log('Metabolism event:', event),
 *   { sources: ['metabolism'] }
 * );
 *
 * // Subscribe to specific event type
 * bus.on('metabolism:claim_died', async (event) => {
 *   // Feed back to calibration
 *   await calibrationEngine.recordNegativeOutcome(event.payload.claimId);
 * });
 *
 * // Emit an event
 * bus.emit({
 *   type: 'metabolism:claim_born',
 *   source: 'metabolism',
 *   timestamp: Date.now(),
 *   payload: { claimId: 'claim-123', ... }
 * });
 * ```
 */
/**
 * Event bus configuration options
 */
export interface SCEIEventBusOptions {
  /** Maximum number of events to keep in history (default: 1000) */
  maxHistorySize?: number;
  /** Whether to keep event history (default: true) */
  enableHistory?: boolean;
  /** Whether to collect metrics (default: true) */
  enableMetrics?: boolean;
  /** Timeout for async listeners in ms (default: 5000) */
  listenerTimeoutMs?: number;
  /** Whether to validate payloads on emit (default: true in development) */
  validatePayloads?: boolean;
}

export class SCEIEventBus {
  private listeners: Map<string, { listener: SCEIEventListener; filter?: SCEIEventFilter }> =
    new Map();
  private typeListeners: Map<SCEIEventType, Set<string>> = new Map();
  private sourceListeners: Map<SCEIModule, Set<string>> = new Map();
  private globalListeners: Set<string> = new Set(); // Listeners with no filter
  private eventHistory: SCEIEvent[] = [];
  private nextSubscriptionId = 1;

  // Configuration
  private maxHistorySize: number;
  private enableHistory: boolean;
  private enableMetrics: boolean;
  private listenerTimeoutMs: number;
  private validatePayloads: boolean;

  // Metrics
  private metrics = {
    totalEventsEmitted: 0,
    eventsByType: new Map<SCEIEventType, number>(),
    eventsBySource: new Map<SCEIModule, number>(),
    listenerErrors: 0,
    listenerTimeouts: 0,
    validationFailures: 0,
  };

  constructor(options: SCEIEventBusOptions = {}) {
    this.maxHistorySize = options.maxHistorySize ?? 1000;
    this.enableHistory = options.enableHistory ?? true;
    this.enableMetrics = options.enableMetrics ?? true;
    this.listenerTimeoutMs = options.listenerTimeoutMs ?? 5000;
    this.validatePayloads = options.validatePayloads ?? true;
  }

  /**
   * Subscribe to events with optional filter
   */
  subscribe(listener: SCEIEventListener, filter?: SCEIEventFilter): Subscription {
    const id = `sub_${this.nextSubscriptionId++}`;
    this.listeners.set(id, { listener, filter });

    // Index by type for O(1) lookup
    if (filter?.types) {
      for (const type of filter.types) {
        if (!this.typeListeners.has(type)) {
          this.typeListeners.set(type, new Set());
        }
        this.typeListeners.get(type)!.add(id);
      }
    }

    // Index by source for O(1) lookup
    if (filter?.sources) {
      for (const source of filter.sources) {
        if (!this.sourceListeners.has(source)) {
          this.sourceListeners.set(source, new Set());
        }
        this.sourceListeners.get(source)!.add(id);
      }
    }

    // Track global listeners (no type/source filter)
    if (!filter?.types && !filter?.sources) {
      this.globalListeners.add(id);
    }

    return {
      id,
      unsubscribe: () => this.unsubscribe(id),
    };
  }

  /**
   * Subscribe to a specific event type
   */
  on<T extends SCEIEventType>(type: T, listener: SCEIEventListener<T>): Subscription {
    return this.subscribe(listener as SCEIEventListener, { types: [type] });
  }

  /**
   * Subscribe to events from a specific module
   */
  onModule(module: SCEIModule, listener: SCEIEventListener): Subscription {
    return this.subscribe(listener, { sources: [module] });
  }

  /**
   * Unsubscribe a listener
   */
  unsubscribe(subscriptionId: string): boolean {
    const sub = this.listeners.get(subscriptionId);
    if (!sub) return false;

    this.listeners.delete(subscriptionId);

    // Remove from type index
    for (const typeSet of this.typeListeners.values()) {
      typeSet.delete(subscriptionId);
    }

    // Remove from source index
    for (const sourceSet of this.sourceListeners.values()) {
      sourceSet.delete(subscriptionId);
    }

    // Remove from global listeners
    this.globalListeners.delete(subscriptionId);

    return true;
  }

  /**
   * Get candidate listener IDs efficiently using indexes
   * Returns only listeners that might match the event
   */
  private getCandidateListenerIds(event: SCEIEvent): Set<string> {
    const candidates = new Set<string>();

    // Add global listeners (always candidates)
    for (const id of this.globalListeners) {
      candidates.add(id);
    }

    // Add type-filtered listeners
    const typeListenerIds = this.typeListeners.get(event.type);
    if (typeListenerIds) {
      for (const id of typeListenerIds) {
        candidates.add(id);
      }
    }

    // Add source-filtered listeners
    const sourceListenerIds = this.sourceListeners.get(event.source);
    if (sourceListenerIds) {
      for (const id of sourceListenerIds) {
        candidates.add(id);
      }
    }

    return candidates;
  }

  /**
   * Execute a listener with timeout protection
   */
  private async executeWithTimeout(
    id: string,
    listener: SCEIEventListener,
    event: SCEIEvent
  ): Promise<void> {
    return new Promise((resolve) => {
      let completed = false;

      const timeoutId = setTimeout(() => {
        if (!completed) {
          completed = true;
          this.metrics.listenerTimeouts++;
          console.error(
            `SCEI Event Bus: Listener ${id} timed out after ${this.listenerTimeoutMs}ms`,
            { eventType: event.type, source: event.source }
          );
          resolve();
        }
      }, this.listenerTimeoutMs);

      Promise.resolve()
        .then(() => listener(event))
        .then(() => {
          if (!completed) {
            completed = true;
            clearTimeout(timeoutId);
            resolve();
          }
        })
        .catch((error) => {
          if (!completed) {
            completed = true;
            clearTimeout(timeoutId);
            this.metrics.listenerErrors++;
            console.error(`SCEI Event Bus: Listener ${id} error:`, error);
            resolve();
          }
        });
    });
  }

  /**
   * Emit an event to all matching listeners
   *
   * Features:
   * - O(1) lookup for type/source-filtered listeners
   * - Timeout protection for slow listeners
   * - Optional payload validation
   */
  async emit<T extends SCEIEventType>(event: SCEIEvent<T>): Promise<void> {
    // Validate payload if enabled
    if (this.validatePayloads) {
      const validation = validateEventPayload(event.type, event.payload);
      if (!validation.valid) {
        this.metrics.validationFailures++;
        console.warn(`SCEI Event Bus: ${validation.error}`, {
          type: event.type,
          payload: event.payload,
        });
        // Continue emitting - validation is informational, not blocking
      }
    }

    // Update metrics
    if (this.enableMetrics) {
      this.metrics.totalEventsEmitted++;
      this.metrics.eventsByType.set(
        event.type,
        (this.metrics.eventsByType.get(event.type) ?? 0) + 1
      );
      this.metrics.eventsBySource.set(
        event.source,
        (this.metrics.eventsBySource.get(event.source) ?? 0) + 1
      );
    }

    // Store in history
    if (this.enableHistory) {
      this.eventHistory.push(event);
      if (this.eventHistory.length > this.maxHistorySize) {
        this.eventHistory.shift();
      }
    }

    // Get candidate listeners efficiently using indexes
    const candidateIds = this.getCandidateListenerIds(event);

    // Notify matching listeners with timeout protection
    const promises: Promise<void>[] = [];

    for (const id of candidateIds) {
      const sub = this.listeners.get(id);
      if (sub && this.matchesFilter(event, sub.filter)) {
        promises.push(this.executeWithTimeout(id, sub.listener, event));
      }
    }

    await Promise.all(promises);
  }

  /**
   * Emit an event synchronously (fire and forget)
   */
  emitSync<T extends SCEIEventType>(event: SCEIEvent<T>): void {
    this.emit(event).catch((error) => {
      console.error('SCEI Event Bus: Async emit error:', error);
    });
  }

  /**
   * Check if event matches filter
   */
  private matchesFilter(event: SCEIEvent, filter?: SCEIEventFilter): boolean {
    if (!filter) return true;

    if (filter.types && !filter.types.includes(event.type)) {
      return false;
    }

    if (filter.sources && !filter.sources.includes(event.source)) {
      return false;
    }

    if (filter.correlationId && event.correlationId !== filter.correlationId) {
      return false;
    }

    return true;
  }

  /**
   * Get recent events (for debugging)
   */
  getHistory(filter?: SCEIEventFilter, limit: number = 100): SCEIEvent[] {
    let events = this.eventHistory;

    if (filter) {
      events = events.filter((e) => this.matchesFilter(e, filter));
    }

    return events.slice(-limit);
  }

  /**
   * Get metrics
   */
  getMetrics(): SCEIEventBusMetrics {
    return {
      totalEventsEmitted: this.metrics.totalEventsEmitted,
      activeListeners: this.listeners.size,
      listenerErrors: this.metrics.listenerErrors,
      listenerTimeouts: this.metrics.listenerTimeouts,
      validationFailures: this.metrics.validationFailures,
      eventsByType: Object.fromEntries(this.metrics.eventsByType),
      eventsBySource: Object.fromEntries(this.metrics.eventsBySource),
      historySize: this.eventHistory.length,
    };
  }

  /**
   * Clear all listeners and history
   */
  reset(): void {
    this.listeners.clear();
    this.typeListeners.clear();
    this.sourceListeners.clear();
    this.globalListeners.clear();
    this.eventHistory = [];
    this.metrics = {
      totalEventsEmitted: 0,
      eventsByType: new Map(),
      eventsBySource: new Map(),
      listenerErrors: 0,
      listenerTimeouts: 0,
      validationFailures: 0,
    };
  }

  /**
   * Set listener timeout (for testing)
   */
  setListenerTimeout(timeoutMs: number): void {
    this.listenerTimeoutMs = timeoutMs;
  }

  /**
   * Enable/disable payload validation
   */
  setPayloadValidation(enabled: boolean): void {
    this.validatePayloads = enabled;
  }

  /**
   * Wait for a specific event (useful for testing)
   */
  waitFor<T extends SCEIEventType>(
    type: T,
    timeout: number = 5000
  ): Promise<SCEIEvent<T>> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        sub.unsubscribe();
        reject(new Error(`Timeout waiting for event: ${type}`));
      }, timeout);

      const sub = this.on(type, (event) => {
        clearTimeout(timer);
        sub.unsubscribe();
        resolve(event);
      });
    });
  }
}

export interface SCEIEventBusMetrics {
  totalEventsEmitted: number;
  activeListeners: number;
  listenerErrors: number;
  listenerTimeouts: number;
  validationFailures: number;
  eventsByType: Record<string, number>;
  eventsBySource: Record<string, number>;
  historySize: number;
}

// ============================================================================
// EVENT BUILDER (FLUENT API)
// ============================================================================

/**
 * Fluent event builder for type-safe event creation
 *
 * @example
 * ```typescript
 * const event = SCEIEventBuilder
 *   .create('calibration:prediction_recorded')
 *   .from('calibration')
 *   .withPayload({
 *     predictionId: 'pred-123',
 *     confidence: 0.8,
 *     domain: 'science',
 *     source: 'agent-1'
 *   })
 *   .withCorrelationId('corr-456')
 *   .build();
 * ```
 */
export class SCEIEventBuilder<T extends SCEIEventType> {
  private type: T;
  private source?: SCEIModule;
  private payload?: SCEIEventPayloadMap[T];
  private correlationId?: string;
  private timestamp?: number;

  private constructor(type: T) {
    this.type = type;
  }

  /**
   * Create a new event builder for the specified type
   */
  static create<T extends SCEIEventType>(type: T): SCEIEventBuilder<T> {
    return new SCEIEventBuilder(type);
  }

  /**
   * Set the source module
   */
  from(source: SCEIModule): this {
    this.source = source;
    return this;
  }

  /**
   * Set the event payload (type-checked against event type)
   */
  withPayload(payload: SCEIEventPayloadMap[T]): this {
    this.payload = payload;
    return this;
  }

  /**
   * Set a correlation ID for tracking related events
   */
  withCorrelationId(correlationId: string): this {
    this.correlationId = correlationId;
    return this;
  }

  /**
   * Set a custom timestamp (default: Date.now())
   */
  withTimestamp(timestamp: number): this {
    this.timestamp = timestamp;
    return this;
  }

  /**
   * Build the event, validating all required fields
   */
  build(): SCEIEvent<T, SCEIEventPayloadMap[T]> {
    if (!this.source) {
      throw new Error(`Event builder: source is required for event type "${this.type}"`);
    }
    if (this.payload === undefined) {
      throw new Error(`Event builder: payload is required for event type "${this.type}"`);
    }

    // Validate payload
    const validation = validateEventPayload(this.type, this.payload);
    if (!validation.valid) {
      throw new Error(`Event builder: ${validation.error}`);
    }

    return {
      type: this.type,
      source: this.source,
      timestamp: this.timestamp ?? Date.now(),
      payload: this.payload,
      correlationId: this.correlationId,
    };
  }

  /**
   * Build and emit the event directly
   */
  async emit(eventBus?: SCEIEventBus): Promise<void> {
    const bus = eventBus ?? getSCEIEventBus();
    const event = this.build();
    await bus.emit(event);
  }
}

// ============================================================================
// SINGLETON
// ============================================================================

let instance: SCEIEventBus | null = null;

/**
 * Get the global SCEI event bus instance
 */
export function getSCEIEventBus(): SCEIEventBus {
  if (!instance) {
    instance = new SCEIEventBus();
  }
  return instance;
}

/**
 * Reset the global event bus (for testing)
 */
export function resetSCEIEventBus(): void {
  if (instance) {
    instance.reset();
  }
  instance = null;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Create an event with proper structure
 */
export function createSCEIEvent<T extends SCEIEventType, P>(
  type: T,
  source: SCEIModule,
  payload: P,
  correlationId?: string
): SCEIEvent<T, P> {
  return {
    type,
    source,
    timestamp: Date.now(),
    payload,
    correlationId,
  };
}

/**
 * Create a correlation ID for tracking related events
 */
export function createCorrelationId(): string {
  return `corr_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
}
