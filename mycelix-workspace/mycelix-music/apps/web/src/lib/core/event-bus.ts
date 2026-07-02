// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Event Bus
 *
 * Cross-module communication system:
 * - Type-safe event emission and subscription
 * - Wildcard subscriptions
 * - Event history and replay
 * - Middleware support
 * - Async event handling
 */

// ==================== Event Types ====================

export interface PlatformEvents {
  // Audio Events
  'audio:play': { trackId: string; position?: number };
  'audio:pause': { trackId: string };
  'audio:stop': {};
  'audio:seek': { position: number };
  'audio:volume': { volume: number };
  'audio:ended': { trackId: string };
  'audio:error': { trackId: string; error: string };
  'audio:buffer': { trackId: string; progress: number };
  'audio:analysis': { trackId: string; data: Float32Array };

  // Queue Events
  'queue:add': { trackId: string; position?: number };
  'queue:remove': { trackId: string };
  'queue:clear': {};
  'queue:shuffle': {};
  'queue:repeat': { mode: 'none' | 'one' | 'all' };
  'queue:reorder': { from: number; to: number };

  // Collaboration Events
  'collab:join': { sessionId: string; userId: string };
  'collab:leave': { sessionId: string; userId: string };
  'collab:sync': { sessionId: string; operations: any[] };
  'collab:cursor': { userId: string; position: any };
  'collab:presence': { users: any[] };

  // Social Events
  'social:follow': { userId: string };
  'social:unfollow': { userId: string };
  'social:like': { trackId: string };
  'social:unlike': { trackId: string };
  'social:comment': { trackId: string; commentId: string };
  'social:share': { trackId: string; platform: string };
  'social:notification': { type: string; data: any };

  // Creator Events
  'creator:upload': { trackId: string; progress: number };
  'creator:publish': { trackId: string };
  'creator:earning': { amount: number; source: string };
  'creator:subscriber': { userId: string; tier: string };

  // AI Events
  'ai:generate:start': { type: string; params: any };
  'ai:generate:progress': { type: string; progress: number };
  'ai:generate:complete': { type: string; result: any };
  'ai:generate:error': { type: string; error: string };

  // Streaming Events
  'stream:start': { streamId: string };
  'stream:end': { streamId: string };
  'stream:viewer:join': { streamId: string; count: number };
  'stream:viewer:leave': { streamId: string; count: number };
  'stream:chat': { streamId: string; message: any };
  'stream:reaction': { streamId: string; type: string };
  'stream:tip': { streamId: string; amount: number; userId: string };

  // XR Events
  'xr:session:start': { mode: string };
  'xr:session:end': {};
  'xr:gesture': { hand: string; type: string };
  'xr:controller': { hand: string; action: string };

  // Plugin Events
  'plugin:install': { pluginId: string };
  'plugin:uninstall': { pluginId: string };
  'plugin:enable': { pluginId: string };
  'plugin:disable': { pluginId: string };
  'plugin:error': { pluginId: string; error: string };

  // System Events
  'system:online': {};
  'system:offline': {};
  'system:error': { error: string; context?: any };
  'system:warning': { message: string; context?: any };
  'system:performance': { metric: string; value: number };

  // User Events
  'user:login': { userId: string };
  'user:logout': {};
  'user:preferences': { key: string; value: any };
}

export type EventName = keyof PlatformEvents;
export type EventPayload<E extends EventName> = PlatformEvents[E];
export type EventHandler<E extends EventName> = (payload: EventPayload<E>) => void | Promise<void>;

export interface EventMeta {
  timestamp: number;
  source?: string;
  correlationId?: string;
}

export interface EventRecord<E extends EventName = EventName> {
  name: E;
  payload: EventPayload<E>;
  meta: EventMeta;
}

export type EventMiddleware = (
  event: EventRecord,
  next: () => Promise<void>
) => Promise<void>;

// ==================== Event Bus Implementation ====================

export class EventBus {
  private handlers: Map<string, Set<EventHandler<any>>> = new Map();
  private wildcardHandlers: Set<(event: EventRecord) => void> = new Set();
  private middleware: EventMiddleware[] = [];
  private history: EventRecord[] = [];
  private maxHistorySize = 1000;
  private isEnabled = true;

  /**
   * Subscribe to a specific event
   */
  on<E extends EventName>(event: E, handler: EventHandler<E>): () => void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler);

    // Return unsubscribe function
    return () => this.off(event, handler);
  }

  /**
   * Subscribe to an event once
   */
  once<E extends EventName>(event: E, handler: EventHandler<E>): () => void {
    const wrappedHandler: EventHandler<E> = (payload) => {
      this.off(event, wrappedHandler);
      return handler(payload);
    };
    return this.on(event, wrappedHandler);
  }

  /**
   * Unsubscribe from an event
   */
  off<E extends EventName>(event: E, handler: EventHandler<E>): void {
    const handlers = this.handlers.get(event);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  /**
   * Subscribe to all events (wildcard)
   */
  onAny(handler: (event: EventRecord) => void): () => void {
    this.wildcardHandlers.add(handler);
    return () => this.wildcardHandlers.delete(handler);
  }

  /**
   * Emit an event
   */
  async emit<E extends EventName>(
    event: E,
    payload: EventPayload<E>,
    meta?: Partial<EventMeta>
  ): Promise<void> {
    if (!this.isEnabled) return;

    const record: EventRecord<E> = {
      name: event,
      payload,
      meta: {
        timestamp: Date.now(),
        correlationId: crypto.randomUUID(),
        ...meta,
      },
    };

    // Add to history
    this.addToHistory(record);

    // Run middleware chain
    await this.runMiddleware(record, async () => {
      // Notify wildcard handlers
      for (const handler of this.wildcardHandlers) {
        try {
          handler(record);
        } catch (error) {
          console.error(`Wildcard handler error for ${event}:`, error);
        }
      }

      // Notify specific handlers
      const handlers = this.handlers.get(event);
      if (handlers) {
        const promises = Array.from(handlers).map(async (handler) => {
          try {
            await handler(payload);
          } catch (error) {
            console.error(`Event handler error for ${event}:`, error);
          }
        });
        await Promise.all(promises);
      }
    });
  }

  /**
   * Emit synchronously (fire and forget)
   */
  emitSync<E extends EventName>(
    event: E,
    payload: EventPayload<E>,
    meta?: Partial<EventMeta>
  ): void {
    this.emit(event, payload, meta).catch(console.error);
  }

  /**
   * Add middleware to the event pipeline
   */
  use(middleware: EventMiddleware): () => void {
    this.middleware.push(middleware);
    return () => {
      const index = this.middleware.indexOf(middleware);
      if (index !== -1) {
        this.middleware.splice(index, 1);
      }
    };
  }

  /**
   * Get event history
   */
  getHistory(filter?: {
    event?: EventName;
    since?: number;
    limit?: number;
  }): EventRecord[] {
    let results = [...this.history];

    if (filter?.event) {
      results = results.filter(r => r.name === filter.event);
    }

    if (filter?.since) {
      results = results.filter(r => r.meta.timestamp >= filter.since);
    }

    if (filter?.limit) {
      results = results.slice(-filter.limit);
    }

    return results;
  }

  /**
   * Replay events from history
   */
  async replay(events: EventRecord[]): Promise<void> {
    for (const event of events) {
      await this.emit(event.name, event.payload, event.meta);
    }
  }

  /**
   * Wait for a specific event
   */
  waitFor<E extends EventName>(
    event: E,
    timeout?: number
  ): Promise<EventPayload<E>> {
    return new Promise((resolve, reject) => {
      const timeoutId = timeout
        ? setTimeout(() => {
            unsubscribe();
            reject(new Error(`Timeout waiting for ${event}`));
          }, timeout)
        : null;

      const unsubscribe = this.once(event, (payload) => {
        if (timeoutId) clearTimeout(timeoutId);
        resolve(payload);
      });
    });
  }

  /**
   * Create a filtered event stream
   */
  filter<E extends EventName>(
    event: E,
    predicate: (payload: EventPayload<E>) => boolean
  ): {
    on: (handler: EventHandler<E>) => () => void;
  } {
    return {
      on: (handler: EventHandler<E>) => {
        return this.on(event, (payload) => {
          if (predicate(payload)) {
            handler(payload);
          }
        });
      },
    };
  }

  /**
   * Map events to a different format
   */
  map<E extends EventName, T>(
    event: E,
    mapper: (payload: EventPayload<E>) => T
  ): {
    on: (handler: (mapped: T) => void) => () => void;
  } {
    return {
      on: (handler: (mapped: T) => void) => {
        return this.on(event, (payload) => {
          handler(mapper(payload));
        });
      },
    };
  }

  /**
   * Debounce event emissions
   */
  debounce<E extends EventName>(event: E, delay: number): {
    emit: (payload: EventPayload<E>) => void;
  } {
    let timeoutId: NodeJS.Timeout | null = null;

    return {
      emit: (payload: EventPayload<E>) => {
        if (timeoutId) clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
          this.emitSync(event, payload);
        }, delay);
      },
    };
  }

  /**
   * Throttle event emissions
   */
  throttle<E extends EventName>(event: E, interval: number): {
    emit: (payload: EventPayload<E>) => void;
  } {
    let lastEmit = 0;
    let pendingPayload: EventPayload<E> | null = null;
    let timeoutId: NodeJS.Timeout | null = null;

    return {
      emit: (payload: EventPayload<E>) => {
        const now = Date.now();
        const timeSinceLastEmit = now - lastEmit;

        if (timeSinceLastEmit >= interval) {
          lastEmit = now;
          this.emitSync(event, payload);
        } else {
          pendingPayload = payload;
          if (!timeoutId) {
            timeoutId = setTimeout(() => {
              if (pendingPayload) {
                lastEmit = Date.now();
                this.emitSync(event, pendingPayload);
                pendingPayload = null;
              }
              timeoutId = null;
            }, interval - timeSinceLastEmit);
          }
        }
      },
    };
  }

  /**
   * Enable/disable the event bus
   */
  setEnabled(enabled: boolean): void {
    this.isEnabled = enabled;
  }

  /**
   * Clear all handlers
   */
  clear(): void {
    this.handlers.clear();
    this.wildcardHandlers.clear();
    this.middleware = [];
  }

  /**
   * Clear event history
   */
  clearHistory(): void {
    this.history = [];
  }

  private addToHistory(record: EventRecord): void {
    this.history.push(record);
    if (this.history.length > this.maxHistorySize) {
      this.history.shift();
    }
  }

  private async runMiddleware(
    event: EventRecord,
    finalHandler: () => Promise<void>
  ): Promise<void> {
    let index = 0;

    const next = async (): Promise<void> => {
      if (index < this.middleware.length) {
        const middleware = this.middleware[index++];
        await middleware(event, next);
      } else {
        await finalHandler();
      }
    };

    await next();
  }
}

// ==================== Built-in Middleware ====================

/**
 * Logging middleware
 */
export function createLoggingMiddleware(
  options: { events?: EventName[]; verbose?: boolean } = {}
): EventMiddleware {
  return async (event, next) => {
    const shouldLog = !options.events || options.events.includes(event.name);

    if (shouldLog) {
      if (options.verbose) {
        console.log(`[Event] ${event.name}`, event.payload, event.meta);
      } else {
        console.log(`[Event] ${event.name}`);
      }
    }

    await next();
  };
}

/**
 * Analytics middleware
 */
export function createAnalyticsMiddleware(
  trackEvent: (name: string, properties: any) => void
): EventMiddleware {
  const trackedEvents: EventName[] = [
    'audio:play',
    'social:like',
    'social:share',
    'creator:upload',
    'stream:start',
    'plugin:install',
  ];

  return async (event, next) => {
    if (trackedEvents.includes(event.name)) {
      trackEvent(event.name, event.payload);
    }
    await next();
  };
}

/**
 * Persistence middleware (for offline support)
 */
export function createPersistenceMiddleware(
  storage: Storage,
  events: EventName[]
): EventMiddleware {
  return async (event, next) => {
    if (events.includes(event.name)) {
      const key = `event:${event.name}:${event.meta.timestamp}`;
      storage.setItem(key, JSON.stringify(event));
    }
    await next();
  };
}

/**
 * Rate limiting middleware
 */
export function createRateLimitMiddleware(
  limits: Partial<Record<EventName, { max: number; window: number }>>
): EventMiddleware {
  const counters = new Map<string, { count: number; resetAt: number }>();

  return async (event, next) => {
    const limit = limits[event.name];
    if (!limit) {
      await next();
      return;
    }

    const now = Date.now();
    let counter = counters.get(event.name);

    if (!counter || counter.resetAt <= now) {
      counter = { count: 0, resetAt: now + limit.window };
      counters.set(event.name, counter);
    }

    if (counter.count >= limit.max) {
      console.warn(`Rate limit exceeded for ${event.name}`);
      return;
    }

    counter.count++;
    await next();
  };
}

// ==================== Singleton ====================

let eventBus: EventBus | null = null;

export function getEventBus(): EventBus {
  if (!eventBus) {
    eventBus = new EventBus();
  }
  return eventBus;
}

export default {
  EventBus,
  getEventBus,
  createLoggingMiddleware,
  createAnalyticsMiddleware,
  createPersistenceMiddleware,
  createRateLimitMiddleware,
};
