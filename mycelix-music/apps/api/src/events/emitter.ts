// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Type-Safe Event Emitter
 *
 * Provides a strongly-typed event system for decoupling components.
 * Events are used for:
 * - Song registration notifications
 * - Play recording
 * - Analytics updates
 * - WebSocket broadcasts
 */

import { EventEmitter } from 'events';

/**
 * Application event types
 */
export interface AppEvents {
  // Song events
  'song:created': { songId: string; title: string; artistAddress: string };
  'song:updated': { songId: string; changes: Record<string, unknown> };
  'song:deleted': { songId: string };
  'song:claimed': { songId: string; claimStreamId: string };

  // Play events
  'play:recorded': {
    playId: string;
    songId: string;
    listenerAddress: string;
    amount: string;
  };
  'play:confirmed': {
    playId: string;
    songId: string;
    transactionHash: string;
  };
  'play:failed': {
    playId: string;
    songId: string;
    reason: string;
  };

  // Analytics events
  'analytics:play-milestone': {
    songId: string;
    milestone: number;
  };
  'analytics:earnings-milestone': {
    artistAddress: string;
    milestone: string;
  };

  // System events
  'system:startup': { timestamp: Date };
  'system:shutdown': { timestamp: Date; reason: string };
  'system:error': { error: Error; context?: Record<string, unknown> };

  // Cache events
  'cache:invalidated': { keys: string[]; reason: string };
  'cache:warmed': { keys: string[] };
}

/**
 * Event handler type
 */
export type EventHandler<T> = (payload: T) => void | Promise<void>;

/**
 * Typed event emitter
 */
export class TypedEventEmitter {
  private emitter = new EventEmitter();
  private asyncHandlers = new Map<string, Set<EventHandler<unknown>>>();

  constructor() {
    // Increase max listeners for production
    this.emitter.setMaxListeners(50);
  }

  /**
   * Subscribe to an event
   */
  on<K extends keyof AppEvents>(
    event: K,
    handler: EventHandler<AppEvents[K]>
  ): () => void {
    this.emitter.on(event, handler);

    // Return unsubscribe function
    return () => this.off(event, handler);
  }

  /**
   * Subscribe to an event once
   */
  once<K extends keyof AppEvents>(
    event: K,
    handler: EventHandler<AppEvents[K]>
  ): void {
    this.emitter.once(event, handler);
  }

  /**
   * Unsubscribe from an event
   */
  off<K extends keyof AppEvents>(
    event: K,
    handler: EventHandler<AppEvents[K]>
  ): void {
    this.emitter.off(event, handler);
  }

  /**
   * Emit an event synchronously
   */
  emit<K extends keyof AppEvents>(event: K, payload: AppEvents[K]): boolean {
    return this.emitter.emit(event, payload);
  }

  /**
   * Emit an event and wait for all async handlers
   */
  async emitAsync<K extends keyof AppEvents>(
    event: K,
    payload: AppEvents[K]
  ): Promise<void> {
    const handlers = this.asyncHandlers.get(event);
    if (!handlers || handlers.size === 0) {
      this.emit(event, payload);
      return;
    }

    const promises = Array.from(handlers).map((handler) =>
      Promise.resolve(handler(payload)).catch((error) => {
        console.error(`[Events] Async handler error for ${event}:`, error);
      })
    );

    await Promise.all(promises);
    this.emit(event, payload);
  }

  /**
   * Register an async handler
   */
  onAsync<K extends keyof AppEvents>(
    event: K,
    handler: EventHandler<AppEvents[K]>
  ): () => void {
    if (!this.asyncHandlers.has(event)) {
      this.asyncHandlers.set(event, new Set());
    }
    this.asyncHandlers.get(event)!.add(handler as EventHandler<unknown>);

    return () => {
      this.asyncHandlers.get(event)?.delete(handler as EventHandler<unknown>);
    };
  }

  /**
   * Get listener count for an event
   */
  listenerCount<K extends keyof AppEvents>(event: K): number {
    return this.emitter.listenerCount(event);
  }

  /**
   * Remove all listeners for an event
   */
  removeAllListeners<K extends keyof AppEvents>(event?: K): void {
    if (event) {
      this.emitter.removeAllListeners(event);
      this.asyncHandlers.delete(event);
    } else {
      this.emitter.removeAllListeners();
      this.asyncHandlers.clear();
    }
  }
}

// Singleton instance
let _events: TypedEventEmitter | null = null;

/**
 * Get the global event emitter
 */
export function getEventEmitter(): TypedEventEmitter {
  if (!_events) {
    _events = new TypedEventEmitter();
  }
  return _events;
}

/**
 * Reset the event emitter (for testing)
 */
export function resetEventEmitter(): void {
  if (_events) {
    _events.removeAllListeners();
  }
  _events = null;
}

export default TypedEventEmitter;
