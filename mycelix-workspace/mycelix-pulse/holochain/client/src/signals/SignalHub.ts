// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Signal Hub - Unified signal processing for all Holochain zomes
 * Aggregates, filters, and dispatches real-time events
 */

import type { AppClient, AppSignal, AgentPubKey } from '@holochain/client';
import type {
  MycelixSignal,
  MessageSignal,
  TrustSignal,
  SyncSignal,
  FederationSignal,
} from '../types';

export type SignalHandler<T> = (signal: T) => void | Promise<void>;
export type SignalFilter = (signal: MycelixSignal) => boolean;

interface SignalSubscription {
  id: string;
  handler: SignalHandler<MycelixSignal>;
  filter?: SignalFilter;
  zomeFilter?: string[];
}

export class SignalHub {
  private subscriptions: Map<string, SignalSubscription> = new Map();
  private messageHandlers: Set<SignalHandler<MessageSignal>> = new Set();
  private trustHandlers: Set<SignalHandler<TrustSignal>> = new Set();
  private syncHandlers: Set<SignalHandler<SyncSignal>> = new Set();
  private federationHandlers: Set<SignalHandler<FederationSignal>> = new Set();
  private errorHandlers: Set<(error: Error, signal: unknown) => void> = new Set();

  private isConnected = false;
  private signalBuffer: MycelixSignal[] = [];
  private bufferSize = 100;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  constructor(private client: AppClient) {
    this.setupSignalHandler();
  }

  private setupSignalHandler(): void {
    this.client.on('signal', (signal: AppSignal) => {
      try {
        const parsed = this.parseSignal(signal);
        if (parsed) {
          this.dispatch(parsed);
        }
      } catch (error) {
        this.handleError(error as Error, signal);
      }
    });
    this.isConnected = true;
  }

  private parseSignal(appSignal: AppSignal): MycelixSignal | null {
    const payload = appSignal.payload as { type?: string; [key: string]: unknown };

    if (!payload || typeof payload.type !== 'string') {
      return null;
    }

    // Determine signal category based on type
    const type = payload.type;

    // Message signals
    if (['EmailReceived', 'DeliveryConfirmed', 'ReadReceiptReceived',
         'EmailStateChanged', 'ThreadActivity', 'TypingIndicator'].includes(type)) {
      return payload as MessageSignal;
    }

    // Trust signals
    if (['AttestationCreated', 'AttestationRevoked', 'TrustScoreChanged',
         'ByzantineFlagRaised'].includes(type)) {
      return payload as TrustSignal;
    }

    // Sync signals
    if (['SyncStateChanged', 'OperationReceived', 'ConflictDetected',
         'ConflictResolved', 'Online', 'Offline', 'SyncProgress'].includes(type)) {
      return payload as SyncSignal;
    }

    // Federation signals
    if (['NetworkDiscovered', 'EnvelopeReceived', 'EnvelopeDelivered',
         'DeliveryFailed', 'BridgeStatusChanged'].includes(type)) {
      return payload as FederationSignal;
    }

    return null;
  }

  private dispatch(signal: MycelixSignal): void {
    // Add to buffer for replay
    this.signalBuffer.push(signal);
    if (this.signalBuffer.length > this.bufferSize) {
      this.signalBuffer.shift();
    }

    // Dispatch to category-specific handlers
    this.dispatchToCategory(signal);

    // Dispatch to general subscribers
    for (const sub of this.subscriptions.values()) {
      if (this.matchesSubscription(signal, sub)) {
        try {
          sub.handler(signal);
        } catch (error) {
          this.handleError(error as Error, signal);
        }
      }
    }
  }

  private dispatchToCategory(signal: MycelixSignal): void {
    const type = signal.type;

    // Message signals
    if (['EmailReceived', 'DeliveryConfirmed', 'ReadReceiptReceived',
         'EmailStateChanged', 'ThreadActivity', 'TypingIndicator'].includes(type)) {
      this.messageHandlers.forEach(handler => {
        try { handler(signal as MessageSignal); }
        catch (e) { this.handleError(e as Error, signal); }
      });
    }

    // Trust signals
    if (['AttestationCreated', 'AttestationRevoked', 'TrustScoreChanged',
         'ByzantineFlagRaised'].includes(type)) {
      this.trustHandlers.forEach(handler => {
        try { handler(signal as TrustSignal); }
        catch (e) { this.handleError(e as Error, signal); }
      });
    }

    // Sync signals
    if (['SyncStateChanged', 'OperationReceived', 'ConflictDetected',
         'ConflictResolved', 'Online', 'Offline', 'SyncProgress'].includes(type)) {
      this.syncHandlers.forEach(handler => {
        try { handler(signal as SyncSignal); }
        catch (e) { this.handleError(e as Error, signal); }
      });
    }

    // Federation signals
    if (['NetworkDiscovered', 'EnvelopeReceived', 'EnvelopeDelivered',
         'DeliveryFailed', 'BridgeStatusChanged'].includes(type)) {
      this.federationHandlers.forEach(handler => {
        try { handler(signal as FederationSignal); }
        catch (e) { this.handleError(e as Error, signal); }
      });
    }
  }

  private matchesSubscription(signal: MycelixSignal, sub: SignalSubscription): boolean {
    if (sub.filter && !sub.filter(signal)) {
      return false;
    }
    return true;
  }

  private handleError(error: Error, signal: unknown): void {
    console.error('Signal processing error:', error, signal);
    this.errorHandlers.forEach(handler => handler(error, signal));
  }

  // ==================== PUBLIC API ====================

  /**
   * Subscribe to all signals with optional filter
   */
  subscribe(handler: SignalHandler<MycelixSignal>, filter?: SignalFilter): () => void {
    const id = crypto.randomUUID();
    this.subscriptions.set(id, { id, handler, filter });
    return () => this.subscriptions.delete(id);
  }

  /**
   * Subscribe to message signals only
   */
  onMessage(handler: SignalHandler<MessageSignal>): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  /**
   * Subscribe to trust signals only
   */
  onTrust(handler: SignalHandler<TrustSignal>): () => void {
    this.trustHandlers.add(handler);
    return () => this.trustHandlers.delete(handler);
  }

  /**
   * Subscribe to sync signals only
   */
  onSync(handler: SignalHandler<SyncSignal>): () => void {
    this.syncHandlers.add(handler);
    return () => this.syncHandlers.delete(handler);
  }

  /**
   * Subscribe to federation signals only
   */
  onFederation(handler: SignalHandler<FederationSignal>): () => void {
    this.federationHandlers.add(handler);
    return () => this.federationHandlers.delete(handler);
  }

  /**
   * Subscribe to errors
   */
  onError(handler: (error: Error, signal: unknown) => void): () => void {
    this.errorHandlers.add(handler);
    return () => this.errorHandlers.delete(handler);
  }

  /**
   * Subscribe to specific signal type
   */
  on<T extends MycelixSignal['type']>(
    type: T,
    handler: SignalHandler<Extract<MycelixSignal, { type: T }>>
  ): () => void {
    return this.subscribe(
      handler as SignalHandler<MycelixSignal>,
      (signal) => signal.type === type
    );
  }

  /**
   * Wait for a specific signal (one-time)
   */
  waitFor<T extends MycelixSignal['type']>(
    type: T,
    timeout?: number
  ): Promise<Extract<MycelixSignal, { type: T }>> {
    return new Promise((resolve, reject) => {
      const timer = timeout ? setTimeout(() => {
        unsubscribe();
        reject(new Error(`Timeout waiting for signal: ${type}`));
      }, timeout) : null;

      const unsubscribe = this.on(type, (signal) => {
        if (timer) clearTimeout(timer);
        unsubscribe();
        resolve(signal as Extract<MycelixSignal, { type: T }>);
      });
    });
  }

  /**
   * Get recent signals from buffer
   */
  getRecentSignals(count?: number): MycelixSignal[] {
    const n = count ?? this.bufferSize;
    return this.signalBuffer.slice(-n);
  }

  /**
   * Get signals by type from buffer
   */
  getSignalsByType<T extends MycelixSignal['type']>(
    type: T
  ): Array<Extract<MycelixSignal, { type: T }>> {
    return this.signalBuffer.filter(
      (s) => s.type === type
    ) as Array<Extract<MycelixSignal, { type: T }>>;
  }

  /**
   * Clear signal buffer
   */
  clearBuffer(): void {
    this.signalBuffer = [];
  }

  /**
   * Check if connected
   */
  get connected(): boolean {
    return this.isConnected;
  }

  /**
   * Unsubscribe all handlers
   */
  clear(): void {
    this.subscriptions.clear();
    this.messageHandlers.clear();
    this.trustHandlers.clear();
    this.syncHandlers.clear();
    this.federationHandlers.clear();
    this.errorHandlers.clear();
  }
}

/**
 * Create typed event emitter for UI frameworks
 */
export function createSignalEmitter(hub: SignalHub) {
  const listeners = new Map<string, Set<Function>>();

  hub.subscribe((signal) => {
    const handlers = listeners.get(signal.type);
    if (handlers) {
      handlers.forEach(fn => fn(signal));
    }
  });

  return {
    on(type: string, handler: Function) {
      if (!listeners.has(type)) {
        listeners.set(type, new Set());
      }
      listeners.get(type)!.add(handler);
      return () => listeners.get(type)?.delete(handler);
    },
    off(type: string, handler: Function) {
      listeners.get(type)?.delete(handler);
    },
    once(type: string, handler: Function) {
      const wrapper = (signal: unknown) => {
        handler(signal);
        listeners.get(type)?.delete(wrapper);
      };
      if (!listeners.has(type)) {
        listeners.set(type, new Set());
      }
      listeners.get(type)!.add(wrapper);
    },
  };
}
