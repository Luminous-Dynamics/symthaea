/**
 * @mycelix/sdk Cross-hApp Event Bus
 *
 * Pub/sub event system for decentralized cross-hApp communication.
 * Enables loose coupling between happs while maintaining type safety.
 *
 * @packageDocumentation
 * @module bridge/event-bus
 */

import { type BrandedHappId, type BrandedAgentId, type BrandedTimestampMs, brandTimestamp } from '../utils/branded.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Event priority levels
 */
export enum EventPriority {
  LOW = 0,
  NORMAL = 1,
  HIGH = 2,
  CRITICAL = 3,
}

/**
 * Event delivery guarantee levels
 */
export enum DeliveryGuarantee {
  /** Fire and forget - no delivery confirmation */
  BEST_EFFORT = 'best_effort',
  /** At least once delivery - may duplicate */
  AT_LEAST_ONCE = 'at_least_once',
  /** Exactly once delivery - with deduplication */
  EXACTLY_ONCE = 'exactly_once',
}

/**
 * Cross-hApp event structure
 */
export interface CrossHappEvent<T = unknown> {
  /** Unique event identifier */
  id: string;
  /** Event type/topic */
  type: string;
  /** Source hApp that emitted the event */
  sourceHapp: BrandedHappId | string;
  /** Source agent */
  sourceAgent?: BrandedAgentId | string;
  /** Event payload */
  payload: T;
  /** When the event was created */
  timestamp: BrandedTimestampMs | number;
  /** Event priority */
  priority: EventPriority;
  /** Correlation ID for request/response patterns */
  correlationId?: string;
  /** Metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Event subscription options
 */
export interface SubscriptionOptions {
  /** Filter by event types (glob patterns supported) */
  types?: string[];
  /** Filter by source happs */
  sourceHapps?: string[];
  /** Minimum priority to receive */
  minPriority?: EventPriority;
  /** Delivery guarantee level */
  deliveryGuarantee?: DeliveryGuarantee;
  /** Maximum events to buffer if handler is slow */
  bufferSize?: number;
  /** Process events in batches */
  batchSize?: number;
  /** Batch timeout in ms */
  batchTimeoutMs?: number;
}

/**
 * Event handler function
 */
export type EventHandler<T = unknown> = (event: CrossHappEvent<T>) => void | Promise<void>;

/**
 * Batch event handler function
 */
export type BatchEventHandler<T = unknown> = (events: CrossHappEvent<T>[]) => void | Promise<void>;

/**
 * Subscription handle for unsubscribing
 */
export interface Subscription {
  /** Subscription ID */
  id: string;
  /** Unsubscribe from events */
  unsubscribe(): void;
  /** Pause receiving events */
  pause(): void;
  /** Resume receiving events */
  resume(): void;
  /** Check if active */
  isActive(): boolean;
}

/**
 * Event bus statistics
 */
export interface EventBusStats {
  /** Total events published */
  totalPublished: number;
  /** Total events delivered */
  totalDelivered: number;
  /** Total events dropped (buffer overflow) */
  totalDropped: number;
  /** Active subscriptions */
  activeSubscriptions: number;
  /** Events per second (1 minute rolling average) */
  eventsPerSecond: number;
  /** Average delivery latency in ms */
  avgDeliveryLatencyMs: number;
}

// ============================================================================
// Implementation
// ============================================================================

/**
 * Internal subscription state
 */
interface InternalSubscription<T = unknown> {
  id: string;
  handler: EventHandler<T>;
  batchHandler?: BatchEventHandler<T>;
  options: SubscriptionOptions;
  active: boolean;
  buffer: CrossHappEvent<T>[];
  lastDelivery: number;
  deliveredCount: number;
  processedIds: Set<string>; // For exactly-once deduplication
}

/**
 * Cross-hApp Event Bus
 *
 * Provides pub/sub messaging across Mycelix happs with:
 * - Topic-based routing with glob patterns
 * - Priority levels
 * - Delivery guarantees
 * - Batching support
 * - Statistics
 */
export class EventBus {
  private subscriptions: Map<string, InternalSubscription> = new Map();
  private stats: EventBusStats = {
    totalPublished: 0,
    totalDelivered: 0,
    totalDropped: 0,
    activeSubscriptions: 0,
    eventsPerSecond: 0,
    avgDeliveryLatencyMs: 0,
  };
  private recentDeliveries: { timestamp: number; latencyMs: number }[] = [];
  private subscriptionCounter = 0;
  private batchTimers: Map<string, NodeJS.Timeout> = new Map();

  /**
   * Publish an event to all matching subscribers
   */
  async publish<T = unknown>(event: Omit<CrossHappEvent<T>, 'id' | 'timestamp'>): Promise<string> {
    const fullEvent: CrossHappEvent<T> = {
      ...event,
      id: this.generateEventId(),
      timestamp: brandTimestamp(Date.now()),
      priority: event.priority ?? EventPriority.NORMAL,
    };

    this.stats.totalPublished++;
    const startTime = Date.now();

    // Find matching subscriptions
    const matchingSubscriptions = this.findMatchingSubscriptions(fullEvent);

    // Deliver to each subscriber
    const deliveryPromises = matchingSubscriptions.map(async (sub) => {
      try {
        await this.deliverToSubscription(sub, fullEvent);
      } catch (error) {
        console.error(`Event delivery failed for subscription ${sub.id}:`, error);
      }
    });

    await Promise.all(deliveryPromises);

    // Update stats
    const latencyMs = Date.now() - startTime;
    this.recordDeliveryLatency(latencyMs);

    return fullEvent.id;
  }

  /**
   * Subscribe to events
   */
  subscribe<T = unknown>(
    handler: EventHandler<T>,
    options: SubscriptionOptions = {}
  ): Subscription {
    const id = `sub_${++this.subscriptionCounter}_${Date.now()}`;

    const internalSub: InternalSubscription<T> = {
      id,
      handler: handler as EventHandler,
      options: {
        bufferSize: 1000,
        deliveryGuarantee: DeliveryGuarantee.BEST_EFFORT,
        ...options,
      },
      active: true,
      buffer: [],
      lastDelivery: Date.now(),
      deliveredCount: 0,
      processedIds: new Set(),
    };

    this.subscriptions.set(id, internalSub as InternalSubscription);
    this.stats.activeSubscriptions++;

    // Set up batch timer if batching is enabled
    if (options.batchSize && options.batchTimeoutMs) {
      this.setupBatchTimer(id, options.batchTimeoutMs);
    }

    return this.createSubscriptionHandle(id);
  }

  /**
   * Subscribe with batch processing
   */
  subscribeBatch<T = unknown>(
    handler: BatchEventHandler<T>,
    options: SubscriptionOptions & { batchSize: number; batchTimeoutMs: number }
  ): Subscription {
    const id = `sub_${++this.subscriptionCounter}_${Date.now()}`;

    const internalSub: InternalSubscription<T> = {
      id,
      handler: () => {}, // Unused for batch
      batchHandler: handler as BatchEventHandler,
      options: {
        bufferSize: 1000,
        deliveryGuarantee: DeliveryGuarantee.BEST_EFFORT,
        ...options,
      },
      active: true,
      buffer: [],
      lastDelivery: Date.now(),
      deliveredCount: 0,
      processedIds: new Set(),
    };

    this.subscriptions.set(id, internalSub as InternalSubscription);
    this.stats.activeSubscriptions++;

    this.setupBatchTimer(id, options.batchTimeoutMs);

    return this.createSubscriptionHandle(id);
  }

  /**
   * Get current statistics
   */
  getStats(): EventBusStats {
    // Calculate events per second from recent deliveries
    const oneMinuteAgo = Date.now() - 60000;
    const recentCount = this.recentDeliveries.filter((d) => d.timestamp > oneMinuteAgo).length;
    this.stats.eventsPerSecond = recentCount / 60;

    return { ...this.stats };
  }

  /**
   * Clear all subscriptions (for testing)
   */
  clear(): void {
    for (const timer of this.batchTimers.values()) {
      clearInterval(timer);
    }
    this.batchTimers.clear();
    this.subscriptions.clear();
    this.stats.activeSubscriptions = 0;
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private generateEventId(): string {
    return `evt_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }

  private findMatchingSubscriptions<T>(event: CrossHappEvent<T>): InternalSubscription[] {
    const matching: InternalSubscription[] = [];

    for (const sub of this.subscriptions.values()) {
      if (!sub.active) continue;

      // Check priority filter
      if (
        sub.options.minPriority !== undefined &&
        event.priority < sub.options.minPriority
      ) {
        continue;
      }

      // Check type filter
      if (sub.options.types && sub.options.types.length > 0) {
        const matchesType = sub.options.types.some((pattern) =>
          this.matchGlob(pattern, event.type)
        );
        if (!matchesType) continue;
      }

      // Check source happ filter
      if (sub.options.sourceHapps && sub.options.sourceHapps.length > 0) {
        const sourceHapp = String(event.sourceHapp);
        if (!sub.options.sourceHapps.includes(sourceHapp)) continue;
      }

      matching.push(sub);
    }

    return matching;
  }

  private async deliverToSubscription<T>(
    sub: InternalSubscription<T>,
    event: CrossHappEvent<T>
  ): Promise<void> {
    // Exactly-once deduplication
    if (sub.options.deliveryGuarantee === DeliveryGuarantee.EXACTLY_ONCE) {
      if (sub.processedIds.has(event.id)) {
        return; // Already processed
      }
      sub.processedIds.add(event.id);

      // Clean up old IDs (keep last 10000)
      if (sub.processedIds.size > 10000) {
        const idsArray = Array.from(sub.processedIds);
        sub.processedIds = new Set(idsArray.slice(-5000));
      }
    }

    // Batch mode
    if (sub.batchHandler && sub.options.batchSize) {
      sub.buffer.push(event);

      // Check buffer overflow
      const bufferSize = sub.options.bufferSize ?? 1000;
      if (sub.buffer.length > bufferSize) {
        this.stats.totalDropped++;
        sub.buffer.shift(); // Drop oldest
      }

      // Flush if batch size reached
      if (sub.buffer.length >= sub.options.batchSize) {
        await this.flushBatch(sub);
      }
      return;
    }

    // Immediate delivery
    try {
      await sub.handler(event);
      sub.deliveredCount++;
      this.stats.totalDelivered++;
    } catch (error) {
      // For at_least_once, we'd retry here
      if (sub.options.deliveryGuarantee === DeliveryGuarantee.AT_LEAST_ONCE) {
        // Simple retry (in production, use exponential backoff)
        try {
          await sub.handler(event);
          sub.deliveredCount++;
          this.stats.totalDelivered++;
        } catch {
          console.error('Event delivery failed after retry:', error);
        }
      }
    }
  }

  private async flushBatch<T>(sub: InternalSubscription<T>): Promise<void> {
    if (sub.buffer.length === 0 || !sub.batchHandler) return;

    const batch = sub.buffer.splice(0, sub.options.batchSize);

    try {
      await sub.batchHandler(batch);
      sub.deliveredCount += batch.length;
      this.stats.totalDelivered += batch.length;
      sub.lastDelivery = Date.now();
    } catch (error) {
      console.error('Batch delivery failed:', error);
      // For at_least_once, re-add to buffer for retry
      if (sub.options.deliveryGuarantee === DeliveryGuarantee.AT_LEAST_ONCE) {
        sub.buffer.unshift(...batch);
      }
    }
  }

  private setupBatchTimer(subscriptionId: string, timeoutMs: number): void {
    const timer = setInterval(() => {
      void (async () => {
        const sub = this.subscriptions.get(subscriptionId);
        if (sub && sub.active && sub.buffer.length > 0) {
          await this.flushBatch(sub);
        }
      })();
    }, timeoutMs);

    this.batchTimers.set(subscriptionId, timer);
  }

  private createSubscriptionHandle(id: string): Subscription {
    return {
      id,
      unsubscribe: () => {
        this.subscriptions.delete(id);
        const timer = this.batchTimers.get(id);
        if (timer) {
          clearInterval(timer);
          this.batchTimers.delete(id);
        }
        this.stats.activeSubscriptions--;
      },
      pause: () => {
        const sub = this.subscriptions.get(id);
        if (sub) sub.active = false;
      },
      resume: () => {
        const sub = this.subscriptions.get(id);
        if (sub) sub.active = true;
      },
      isActive: () => {
        const sub = this.subscriptions.get(id);
        return sub?.active ?? false;
      },
    };
  }

  private matchGlob(pattern: string, value: string): boolean {
    // Simple glob matching: * matches any sequence
    const regexPattern = pattern
      .replace(/[.+^${}()|[\]\\]/g, '\\$&')
      .replace(/\*/g, '.*')
      .replace(/\?/g, '.');
    return new RegExp(`^${regexPattern}$`).test(value);
  }

  private recordDeliveryLatency(latencyMs: number): void {
    const now = Date.now();
    this.recentDeliveries.push({ timestamp: now, latencyMs });

    // Keep only last minute of data
    const oneMinuteAgo = now - 60000;
    this.recentDeliveries = this.recentDeliveries.filter((d) => d.timestamp > oneMinuteAgo);

    // Update average
    if (this.recentDeliveries.length > 0) {
      const totalLatency = this.recentDeliveries.reduce((sum, d) => sum + d.latencyMs, 0);
      this.stats.avgDeliveryLatencyMs = totalLatency / this.recentDeliveries.length;
    }
  }
}

// ============================================================================
// Singleton Instance & Helpers
// ============================================================================

/** Global event bus instance */
let globalEventBus: EventBus | null = null;

/**
 * Get the global event bus instance
 */
export function getEventBus(): EventBus {
  if (!globalEventBus) {
    globalEventBus = new EventBus();
  }
  return globalEventBus;
}

/**
 * Create a new isolated event bus (for testing)
 */
export function createEventBus(): EventBus {
  return new EventBus();
}

// ============================================================================
// Typed Event Helpers
// ============================================================================

/**
 * Create a typed event publisher for a specific event type
 */
export function createPublisher<T>(
  eventBus: EventBus,
  eventType: string,
  sourceHapp: string
): (payload: T, options?: Partial<CrossHappEvent<T>>) => Promise<string> {
  return (payload, options = {}) =>
    eventBus.publish({
      type: eventType,
      sourceHapp,
      payload,
      priority: EventPriority.NORMAL,
      ...options,
    });
}

/**
 * Create a typed event subscriber for a specific event type
 */
export function createSubscriber<T>(
  eventBus: EventBus,
  eventType: string,
  handler: (payload: T, event: CrossHappEvent<T>) => void | Promise<void>,
  options?: SubscriptionOptions
): Subscription {
  return eventBus.subscribe<T>((event) => handler(event.payload, event), {
    types: [eventType],
    ...options,
  });
}

// ============================================================================
// Standard Event Types
// ============================================================================

/**
 * Standard cross-hApp event types
 */
export const StandardEventTypes = {
  // Identity events
  IDENTITY_CREATED: 'mycelix.identity.created',
  IDENTITY_UPDATED: 'mycelix.identity.updated',
  CREDENTIAL_ISSUED: 'mycelix.identity.credential.issued',
  CREDENTIAL_REVOKED: 'mycelix.identity.credential.revoked',

  // Governance events
  PROPOSAL_CREATED: 'mycelix.governance.proposal.created',
  PROPOSAL_PASSED: 'mycelix.governance.proposal.passed',
  PROPOSAL_REJECTED: 'mycelix.governance.proposal.rejected',
  VOTE_CAST: 'mycelix.governance.vote.cast',

  // Knowledge events
  CLAIM_CREATED: 'mycelix.knowledge.claim.created',
  CLAIM_ENDORSED: 'mycelix.knowledge.claim.endorsed',
  CLAIM_REFUTED: 'mycelix.knowledge.claim.refuted',

  // Reputation events
  REPUTATION_UPDATED: 'mycelix.matl.reputation.updated',
  TRUST_EVALUATED: 'mycelix.matl.trust.evaluated',
  BYZANTINE_DETECTED: 'mycelix.matl.byzantine.detected',

  // Bridge events
  HAPP_REGISTERED: 'mycelix.bridge.happ.registered',
  MESSAGE_ROUTED: 'mycelix.bridge.message.routed',

  // FL events
  FL_ROUND_STARTED: 'mycelix.fl.round.started',
  FL_ROUND_COMPLETED: 'mycelix.fl.round.completed',
  FL_GRADIENT_SUBMITTED: 'mycelix.fl.gradient.submitted',
} as const;

export type StandardEventType = (typeof StandardEventTypes)[keyof typeof StandardEventTypes];
