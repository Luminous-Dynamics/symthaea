// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Cross-hApp Bridge Implementation
 *
 * Provides inter-hApp communication for the Mycelix ecosystem.
 * Enables coordinated workflows across governance, finance, identity,
 * knowledge, property, energy, media, and justice hApps.
 *
 * @packageDocumentation
 * @module bridge/cross-happ
 */

import { type ReputationScore, reputationValue } from '../matl/index.js';

// ============================================================================
// Bridge Types
// ============================================================================

/** Available hApp types in the ecosystem */
export type HappId =
  | 'governance'
  | 'finance'
  | 'identity'
  | 'knowledge'
  | 'property'
  | 'energy'
  | 'media'
  | 'justice'
  | 'mail'
  | 'marketplace'
  | 'praxis'
  | 'supplychain';

/** Message types for cross-hApp communication */
export type BridgeMessageType =
  | 'reputation_query'
  | 'reputation_update'
  | 'enforcement_request'
  | 'verification_request'
  | 'transfer_notification'
  | 'decision_broadcast'
  | 'subscription_event'
  | 'capability_response';

/** Bridge message structure */
export interface BridgeMessage<T = unknown> {
  id: string;
  type: BridgeMessageType;
  sourceHapp: HappId;
  targetHapp: HappId | 'broadcast';
  payload: T;
  timestamp: number;
  correlationId?: string;
  replyTo?: string;
}

/** Reputation query payload */
export interface ReputationQueryPayload {
  subjectDid: string;
  contextHapps: HappId[];
  includeHistory?: boolean;
}

/** Reputation query response */
export interface ReputationQueryResponse {
  subjectDid: string;
  aggregatedScore: number;
  scores: Partial<Record<HappId, number>>;
  confidence: number;
}

/** Enforcement request payload */
export interface EnforcementRequestPayload {
  decisionId: string;
  caseId: string;
  targetDid: string;
  remedyType: 'compensation' | 'action' | 'reputation_adjustment' | 'ban';
  details: Record<string, unknown>;
}

/** Verification request payload */
export interface VerificationRequestPayload {
  subjectDid: string;
  verificationType: 'identity' | 'ownership' | 'credential' | 'reputation';
  resource?: string;
}

/** Verification response */
export interface VerificationResponse {
  verified: boolean;
  level: number;
  details?: Record<string, unknown>;
}

/** Bridge event handler */
export type BridgeEventHandler<T = unknown> = (message: BridgeMessage<T>) => Promise<void>;

/** Subscription record */
interface Subscription {
  id: string;
  happId: HappId;
  messageType: BridgeMessageType | '*';
  handler: BridgeEventHandler;
}

// ============================================================================
// Cross-hApp Bridge
// ============================================================================

/**
 * Cross-hApp Bridge for coordinated ecosystem operations
 *
 * @remarks
 * Implements pub/sub messaging, reputation aggregation, and enforcement coordination.
 * In production, this would connect to the Holochain signal system.
 */
export class CrossHappBridge {
  private subscriptions: Subscription[] = [];
  private messageQueue: BridgeMessage[] = [];
  private reputationCache = new Map<string, Partial<Record<HappId, ReputationScore>>>();
  private registeredHapps = new Set<HappId>();

  constructor() {
    // Auto-register core hApps
    this.registeredHapps.add('governance');
    this.registeredHapps.add('finance');
    this.registeredHapps.add('identity');
    this.registeredHapps.add('knowledge');
    this.registeredHapps.add('property');
    this.registeredHapps.add('energy');
    this.registeredHapps.add('media');
    this.registeredHapps.add('justice');
  }

  /**
   * Register a hApp with the bridge
   */
  registerHapp(happId: HappId): void {
    this.registeredHapps.add(happId);
  }

  /**
   * Subscribe to bridge messages
   */
  subscribe(
    happId: HappId,
    messageType: BridgeMessageType | '*',
    handler: BridgeEventHandler
  ): string {
    const id = `sub-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    this.subscriptions.push({ id, happId, messageType, handler });
    return id;
  }

  /**
   * Unsubscribe from bridge messages
   */
  unsubscribe(subscriptionId: string): void {
    this.subscriptions = this.subscriptions.filter((s) => s.id !== subscriptionId);
  }

  /**
   * Send a message through the bridge
   */
  async send<T>(message: Omit<BridgeMessage<T>, 'id' | 'timestamp'>): Promise<void> {
    const fullMessage: BridgeMessage<T> = {
      ...message,
      id: `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      timestamp: Date.now(),
    };

    this.messageQueue.push(fullMessage as BridgeMessage);

    // Deliver to subscribers
    const targetSubs =
      message.targetHapp === 'broadcast'
        ? this.subscriptions
        : this.subscriptions.filter(
            (s) =>
              s.happId === message.targetHapp &&
              (s.messageType === '*' || s.messageType === message.type)
          );

    for (const sub of targetSubs) {
      try {
        await sub.handler(fullMessage as BridgeMessage);
      } catch (error) {
        console.error(`Bridge delivery error to ${sub.happId}:`, error);
      }
    }
  }

  /**
   * Query reputation across hApps
   */
  async queryReputation(
    subjectDid: string,
    contextHapps: HappId[]
  ): Promise<ReputationQueryResponse> {
    const scores: Partial<Record<HappId, number>> = {};
    let totalScore = 0;
    let count = 0;

    // Check cache first
    const cached = this.reputationCache.get(subjectDid);

    for (const happId of contextHapps) {
      if (cached?.[happId]) {
        const value = reputationValue(cached[happId]);
        scores[happId] = value;
        totalScore += value;
        count++;
      } else {
        // In production, would query the actual hApp
        // For now, use default neutral score
        scores[happId] = 0.5;
        totalScore += 0.5;
        count++;
      }
    }

    const aggregatedScore = count > 0 ? totalScore / count : 0.5;

    return {
      subjectDid,
      aggregatedScore,
      scores,
      confidence: count / contextHapps.length,
    };
  }

  /**
   * Update reputation in cache
   */
  updateReputation(did: string, happId: HappId, reputation: ReputationScore): void {
    const existing = this.reputationCache.get(did) || {};
    existing[happId] = reputation;
    this.reputationCache.set(did, existing);

    // Broadcast update
    this.send({
      type: 'reputation_update',
      sourceHapp: happId,
      targetHapp: 'broadcast',
      payload: { did, happId, score: reputationValue(reputation) },
    });
  }

  /**
   * Request enforcement action
   */
  async requestEnforcement(
    sourceHapp: HappId,
    targetHapp: HappId,
    payload: EnforcementRequestPayload
  ): Promise<{ acknowledged: boolean; estimatedCompletion?: number }> {
    await this.send({
      type: 'enforcement_request',
      sourceHapp,
      targetHapp,
      payload,
    });

    return {
      acknowledged: true,
      estimatedCompletion: Date.now() + 24 * 60 * 60 * 1000, // 24 hours
    };
  }

  /**
   * Request verification from another hApp
   */
  async requestVerification(
    sourceHapp: HappId,
    targetHapp: HappId,
    payload: VerificationRequestPayload
  ): Promise<VerificationResponse> {
    await this.send({
      type: 'verification_request',
      sourceHapp,
      targetHapp,
      payload,
    });

    // In production, would await response
    // For now, return positive verification for registered happs
    return {
      verified: this.registeredHapps.has(targetHapp),
      level: 1,
    };
  }

  /**
   * Broadcast a decision to all hApps
   */
  async broadcastDecision(
    sourceHapp: HappId,
    decisionId: string,
    outcome: string,
    affectedDids: string[]
  ): Promise<void> {
    await this.send({
      type: 'decision_broadcast',
      sourceHapp,
      targetHapp: 'broadcast',
      payload: { decisionId, outcome, affectedDids, broadcastAt: Date.now() },
    });
  }

  /**
   * Get message history
   */
  getMessageHistory(limit = 100): BridgeMessage[] {
    return this.messageQueue.slice(-limit);
  }

  /**
   * Clear message queue (for testing)
   */
  clearMessages(): void {
    this.messageQueue = [];
  }
}

// ============================================================================
// Bridge Factory
// ============================================================================

/** Shared bridge instance */
let sharedBridge: CrossHappBridge | null = null;

/**
 * Get the shared bridge instance
 */
export function getCrossHappBridge(): CrossHappBridge {
  if (!sharedBridge) {
    sharedBridge = new CrossHappBridge();
  }
  return sharedBridge;
}

/**
 * Reset the bridge (for testing)
 */
export function resetCrossHappBridge(): void {
  sharedBridge = null;
}

// ============================================================================
// Bridge Utilities
// ============================================================================

/**
 * Create a cross-hApp reputation query for specific hApps
 * Note: Named differently to avoid collision with bridge/index.ts createReputationQuery
 */
export function createCrossHappReputationQuery(
  subjectDid: string,
  contextHapps: HappId[],
  includeHistory = false
): ReputationQueryPayload {
  return { subjectDid, contextHapps, includeHistory };
}

/**
 * Create an enforcement request
 */
export function createEnforcementRequest(
  decisionId: string,
  caseId: string,
  targetDid: string,
  remedyType: EnforcementRequestPayload['remedyType'],
  details: Record<string, unknown> = {}
): EnforcementRequestPayload {
  return { decisionId, caseId, targetDid, remedyType, details };
}

/**
 * Aggregate reputation scores with weights
 */
export function aggregateReputationWithWeights(
  scores: Partial<Record<HappId, number>>,
  weights?: Partial<Record<HappId, number>>
): number {
  const defaultWeights: Partial<Record<HappId, number>> = {
    identity: 1.5, // Identity verification weighs more
    finance: 1.2,
    governance: 1.1,
    justice: 1.3, // Justice history weighs more
    knowledge: 0.8,
    property: 1.0,
    energy: 0.9,
    media: 0.7,
  };

  const effectiveWeights = { ...defaultWeights, ...weights };
  let totalScore = 0;
  let totalWeight = 0;

  for (const [happId, score] of Object.entries(scores)) {
    const weight = effectiveWeights[happId as HappId] || 1.0;
    totalScore += score * weight;
    totalWeight += weight;
  }

  return totalWeight > 0 ? totalScore / totalWeight : 0.5;
}
