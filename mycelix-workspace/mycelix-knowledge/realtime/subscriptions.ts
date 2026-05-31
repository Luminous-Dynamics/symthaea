// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Knowledge - Real-time Subscriptions System
 *
 * WebSocket-based real-time updates for claims, credibility, and markets
 */

import { EventEmitter } from 'events';
import type { KnowledgeClient } from '../client/src';

// ============================================================================
// Types
// ============================================================================

export type SubscriptionTopic =
  | 'claim:created'
  | 'claim:updated'
  | 'claim:deleted'
  | 'credibility:changed'
  | 'relationship:created'
  | 'relationship:deleted'
  | 'market:updated'
  | 'market:resolved'
  | 'factcheck:completed'
  | 'graph:propagation';

export interface SubscriptionMessage<T = unknown> {
  topic: SubscriptionTopic;
  timestamp: string;
  payload: T;
  metadata?: {
    affectedIds?: string[];
    source?: string;
    version?: number;
  };
}

export interface ClaimCreatedPayload {
  id: string;
  content: string;
  author: string;
  classification: {
    empirical: number;
    normative: number;
    mythic: number;
  };
}

export interface CredibilityChangedPayload {
  claimId: string;
  previousScore: number;
  newScore: number;
  factors: {
    sourceDiversity: number;
    authorReputation: number;
    temporalConsistency: number;
    crossValidation: number;
  };
  confidence: number;
}

export interface MarketUpdatedPayload {
  marketId: string;
  claimId: string;
  currentProbability: number;
  volume: number;
  status: 'OPEN' | 'CLOSED' | 'RESOLVED';
}

export interface GraphPropagationPayload {
  sourceClaimId: string;
  affectedClaims: string[];
  iterations: number;
  converged: boolean;
}

// ============================================================================
// Subscription Filters
// ============================================================================

export interface SubscriptionFilter {
  claimIds?: string[];
  authorDids?: string[];
  tags?: string[];
  minCredibilityChange?: number;
  epistemicType?: 'EMPIRICAL' | 'NORMATIVE' | 'MYTHIC';
}

function matchesFilter<T>(message: SubscriptionMessage<T>, filter: SubscriptionFilter): boolean {
  if (!filter) return true;

  const payload = message.payload as Record<string, unknown>;

  if (filter.claimIds?.length) {
    const claimId = payload.claimId || payload.id;
    if (!claimId || !filter.claimIds.includes(claimId as string)) {
      return false;
    }
  }

  if (filter.authorDids?.length) {
    const author = payload.author;
    if (!author || !filter.authorDids.includes(author as string)) {
      return false;
    }
  }

  if (filter.minCredibilityChange !== undefined) {
    const credPayload = payload as unknown as CredibilityChangedPayload;
    if (credPayload.previousScore !== undefined && credPayload.newScore !== undefined) {
      const change = Math.abs(credPayload.newScore - credPayload.previousScore);
      if (change < filter.minCredibilityChange) {
        return false;
      }
    }
  }

  return true;
}

// ============================================================================
// Subscription Manager
// ============================================================================

export class SubscriptionManager extends EventEmitter {
  private subscriptions: Map<string, Set<SubscriptionTopic>> = new Map();
  private filters: Map<string, SubscriptionFilter> = new Map();
  private client?: KnowledgeClient;
  private pollingIntervals: Map<SubscriptionTopic, NodeJS.Timeout> = new Map();
  private lastKnownState: Map<string, unknown> = new Map();

  constructor(client?: KnowledgeClient) {
    super();
    this.client = client;
    this.setMaxListeners(100);
  }

  /**
   * Subscribe to a topic with optional filter
   */
  subscribe(
    subscriberId: string,
    topics: SubscriptionTopic | SubscriptionTopic[],
    filter?: SubscriptionFilter
  ): () => void {
    const topicArray = Array.isArray(topics) ? topics : [topics];

    // Register subscriptions
    if (!this.subscriptions.has(subscriberId)) {
      this.subscriptions.set(subscriberId, new Set());
    }

    const subscriberTopics = this.subscriptions.get(subscriberId)!;
    topicArray.forEach((topic) => subscriberTopics.add(topic));

    // Store filter
    if (filter) {
      this.filters.set(subscriberId, filter);
    }

    // Start polling for each topic if not already
    topicArray.forEach((topic) => this.startPolling(topic));

    // Return unsubscribe function
    return () => this.unsubscribe(subscriberId, topicArray);
  }

  /**
   * Unsubscribe from topics
   */
  unsubscribe(subscriberId: string, topics?: SubscriptionTopic[]): void {
    const subscriberTopics = this.subscriptions.get(subscriberId);
    if (!subscriberTopics) return;

    if (topics) {
      topics.forEach((topic) => subscriberTopics.delete(topic));
      if (subscriberTopics.size === 0) {
        this.subscriptions.delete(subscriberId);
        this.filters.delete(subscriberId);
      }
    } else {
      this.subscriptions.delete(subscriberId);
      this.filters.delete(subscriberId);
    }

    // Check if we can stop polling for any topics
    this.cleanupPolling();
  }

  /**
   * Publish a message to subscribers
   */
  publish<T>(topic: SubscriptionTopic, payload: T, metadata?: SubscriptionMessage<T>['metadata']): void {
    const message: SubscriptionMessage<T> = {
      topic,
      timestamp: new Date().toISOString(),
      payload,
      metadata,
    };

    // Emit to all matching subscribers
    this.subscriptions.forEach((topics, subscriberId) => {
      if (topics.has(topic)) {
        const filter = this.filters.get(subscriberId);
        if (matchesFilter(message, filter || {})) {
          this.emit(`message:${subscriberId}`, message);
          this.emit('message', message, subscriberId);
        }
      }
    });

    // Emit topic-specific event
    this.emit(topic, message);
  }

  /**
   * Get an async iterator for a subscriber's messages
   */
  async *iterate(subscriberId: string): AsyncGenerator<SubscriptionMessage> {
    const queue: SubscriptionMessage[] = [];
    let resolveNext: ((value: SubscriptionMessage) => void) | null = null;

    const handler = (message: SubscriptionMessage) => {
      if (resolveNext) {
        resolveNext(message);
        resolveNext = null;
      } else {
        queue.push(message);
      }
    };

    this.on(`message:${subscriberId}`, handler);

    try {
      while (true) {
        if (queue.length > 0) {
          yield queue.shift()!;
        } else {
          yield await new Promise<SubscriptionMessage>((resolve) => {
            resolveNext = resolve;
          });
        }
      }
    } finally {
      this.off(`message:${subscriberId}`, handler);
    }
  }

  /**
   * Start polling for changes (fallback when Holochain signals not available)
   */
  private startPolling(topic: SubscriptionTopic): void {
    if (this.pollingIntervals.has(topic)) return;
    if (!this.client) return;

    const interval = setInterval(async () => {
      try {
        await this.checkForChanges(topic);
      } catch (error) {
        console.error(`Polling error for ${topic}:`, error);
      }
    }, 2000); // Poll every 2 seconds

    this.pollingIntervals.set(topic, interval);
  }

  /**
   * Check for changes and publish if found
   */
  private async checkForChanges(topic: SubscriptionTopic): Promise<void> {
    if (!this.client) return;

    switch (topic) {
      case 'claim:created':
      case 'claim:updated': {
        // Check for new/updated claims
        const claims = await this.client.query.listClaims({ limit: 10 });
        claims.items.forEach((claim) => {
          const key = `claim:${claim.id}`;
          const cached = this.lastKnownState.get(key) as Record<string, unknown> | undefined;
          if (!cached) {
            this.lastKnownState.set(key, claim);
            this.publish('claim:created', claim);
          } else if (JSON.stringify(cached) !== JSON.stringify(claim)) {
            this.lastKnownState.set(key, claim);
            this.publish('claim:updated', claim);
          }
        });
        break;
      }

      case 'credibility:changed': {
        // Check credibility changes for cached claims
        for (const [key, value] of this.lastKnownState.entries()) {
          if (key.startsWith('claim:')) {
            const claimId = key.replace('claim:', '');
            try {
              const newCred = await this.client.inference.calculateEnhancedCredibility(
                claimId,
                'Claim'
              );
              const cachedCred = (value as Record<string, unknown>).credibility as number | undefined;
              if (cachedCred !== undefined && Math.abs(newCred.score - cachedCred) > 0.01) {
                this.publish<CredibilityChangedPayload>('credibility:changed', {
                  claimId,
                  previousScore: cachedCred,
                  newScore: newCred.score,
                  factors: newCred.factors,
                  confidence: newCred.confidence,
                });
              }
            } catch {
              // Ignore errors
            }
          }
        }
        break;
      }
    }
  }

  /**
   * Stop polling for topics no longer needed
   */
  private cleanupPolling(): void {
    const activeTopics = new Set<SubscriptionTopic>();
    this.subscriptions.forEach((topics) => {
      topics.forEach((topic) => activeTopics.add(topic));
    });

    this.pollingIntervals.forEach((interval, topic) => {
      if (!activeTopics.has(topic)) {
        clearInterval(interval);
        this.pollingIntervals.delete(topic);
      }
    });
  }

  /**
   * Clean up all subscriptions and polling
   */
  destroy(): void {
    this.pollingIntervals.forEach((interval) => clearInterval(interval));
    this.pollingIntervals.clear();
    this.subscriptions.clear();
    this.filters.clear();
    this.lastKnownState.clear();
    this.removeAllListeners();
  }
}

// ============================================================================
// React Hook Integration
// ============================================================================

export interface UseSubscriptionOptions<T> {
  topics: SubscriptionTopic | SubscriptionTopic[];
  filter?: SubscriptionFilter;
  onMessage?: (message: SubscriptionMessage<T>) => void;
}

/**
 * Create subscription hook factory for React
 */
export function createSubscriptionHook(manager: SubscriptionManager) {
  return function useSubscription<T>(options: UseSubscriptionOptions<T>) {
    // This would be implemented in the React hooks package
    // Returns the latest message and subscription status
    return {
      subscribe: (callback: (msg: SubscriptionMessage<T>) => void) => {
        const id = Math.random().toString(36).substring(7);
        const unsubscribe = manager.subscribe(id, options.topics, options.filter);

        const handler = (msg: SubscriptionMessage) => {
          callback(msg as SubscriptionMessage<T>);
        };

        manager.on(`message:${id}`, handler);

        return () => {
          unsubscribe();
          manager.off(`message:${id}`, handler);
        };
      },
    };
  };
}

// ============================================================================
// WebSocket Server Integration
// ============================================================================

export interface WebSocketMessage {
  type: 'subscribe' | 'unsubscribe' | 'message';
  subscriberId?: string;
  topics?: SubscriptionTopic[];
  filter?: SubscriptionFilter;
  payload?: SubscriptionMessage;
}

/**
 * Create WebSocket handler for real-time subscriptions
 */
export function createWebSocketHandler(manager: SubscriptionManager) {
  return {
    onConnection: (ws: WebSocket, subscriberId: string) => {
      const subscriptions: (() => void)[] = [];

      const messageHandler = (message: SubscriptionMessage) => {
        ws.send(JSON.stringify({
          type: 'message',
          payload: message,
        }));
      };

      manager.on(`message:${subscriberId}`, messageHandler);

      return {
        onMessage: (data: string) => {
          try {
            const msg: WebSocketMessage = JSON.parse(data);

            switch (msg.type) {
              case 'subscribe':
                if (msg.topics) {
                  const unsub = manager.subscribe(subscriberId, msg.topics, msg.filter);
                  subscriptions.push(unsub);
                }
                break;

              case 'unsubscribe':
                manager.unsubscribe(subscriberId, msg.topics);
                break;
            }
          } catch (error) {
            console.error('WebSocket message error:', error);
          }
        },

        onClose: () => {
          subscriptions.forEach((unsub) => unsub());
          manager.off(`message:${subscriberId}`, messageHandler);
          manager.unsubscribe(subscriberId);
        },
      };
    },
  };
}

// ============================================================================
// Export singleton for convenience
// ============================================================================

export const subscriptionManager = new SubscriptionManager();

export default SubscriptionManager;
