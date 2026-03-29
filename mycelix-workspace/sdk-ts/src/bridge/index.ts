// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Bridge Protocol
 *
 * Inter-hApp communication for cross-hApp reputation and credentials.
 */

import { BridgeError, ErrorCode, validate, assertDefined } from '../errors.js';
import { type ReputationScore, reputationValue } from '../matl/index.js';

/**
 * Reputation score from a specific hApp
 */
export interface HappReputationScore {
  happ: string;
  score: number;
  weight: number;
  lastUpdate: number;
}

/**
 * Message types for inter-hApp communication
 */
export enum BridgeMessageType {
  ReputationQuery = 'reputation_query',
  CrossHappReputation = 'cross_happ_reputation',
  CredentialVerification = 'credential_verification',
  VerificationResult = 'verification_result',
  BroadcastEvent = 'broadcast_event',
  HappRegistration = 'happ_registration',
}

/**
 * Base bridge message
 */
export interface BridgeMessage {
  type: BridgeMessageType;
  timestamp: number;
  sourceHapp: string;
}

/**
 * Reputation query message
 */
export interface ReputationQueryMessage extends BridgeMessage {
  type: BridgeMessageType.ReputationQuery;
  agent: string;
}

/**
 * Cross-hApp reputation response
 */
export interface CrossHappReputationMessage extends BridgeMessage {
  type: BridgeMessageType.CrossHappReputation;
  agent: string;
  scores: HappReputationScore[];
  aggregate: number;
}

/**
 * Credential verification request
 */
export interface CredentialVerificationMessage extends BridgeMessage {
  type: BridgeMessageType.CredentialVerification;
  credentialHash: string;
  issuerHapp: string;
}

/**
 * Verification result
 */
export interface VerificationResultMessage extends BridgeMessage {
  type: BridgeMessageType.VerificationResult;
  credentialHash: string;
  valid: boolean;
  issuer: string;
  claims: string[];
}

/**
 * Broadcast event
 */
export interface BroadcastEventMessage extends BridgeMessage {
  type: BridgeMessageType.BroadcastEvent;
  eventType: string;
  payload: Uint8Array;
}

/**
 * hApp registration
 */
export interface HappRegistrationMessage extends BridgeMessage {
  type: BridgeMessageType.HappRegistration;
  happId: string;
  happName: string;
  capabilities: string[];
}

/**
 * Union of all bridge messages
 */
export type AnyBridgeMessage =
  | ReputationQueryMessage
  | CrossHappReputationMessage
  | CredentialVerificationMessage
  | VerificationResultMessage
  | BroadcastEventMessage
  | HappRegistrationMessage;

/**
 * Validate scores have valid weights
 */
function validateScores(scores: HappReputationScore[]): void {
  for (const score of scores) {
    if (score.weight < 0) {
      throw new BridgeError(
        `Invalid weight: ${score.weight} for hApp ${score.happ}`,
        ErrorCode.BRIDGE_INVALID_WEIGHT,
        { happ: score.happ, weight: score.weight }
      );
    }
    if (score.score < 0 || score.score > 1) {
      throw new BridgeError(
        `Invalid score: ${score.score} for hApp ${score.happ}`,
        ErrorCode.INVALID_ARGUMENT,
        { happ: score.happ, score: score.score }
      );
    }
  }
}

/**
 * Calculate weighted aggregate reputation across hApps
 * @throws {BridgeError} If any score has invalid weight
 */
export function calculateAggregateReputation(scores: HappReputationScore[]): number {
  assertDefined(scores, 'scores');
  if (scores.length === 0) return 0.5;

  validateScores(scores);

  const totalWeight = scores.reduce((sum, s) => sum + s.weight, 0);
  if (totalWeight === 0) return 0.5;

  const weightedSum = scores.reduce((sum, s) => sum + s.score * s.weight, 0);
  return weightedSum / totalWeight;
}

/**
 * Create a reputation query message
 * @throws {BridgeError} If sourceHapp or agent is empty
 */
export function createReputationQuery(sourceHapp: string, agent: string): ReputationQueryMessage {
  validate().notEmpty('sourceHapp', sourceHapp).notEmpty('agent', agent).throwIfInvalid();

  return {
    type: BridgeMessageType.ReputationQuery,
    timestamp: Date.now(),
    sourceHapp,
    agent,
  };
}

/**
 * Create a cross-hApp reputation response
 * @throws {BridgeError} If sourceHapp or agent is empty
 */
export function createCrossHappReputation(
  sourceHapp: string,
  agent: string,
  scores: HappReputationScore[]
): CrossHappReputationMessage {
  validate().notEmpty('sourceHapp', sourceHapp).notEmpty('agent', agent).throwIfInvalid();
  assertDefined(scores, 'scores');

  return {
    type: BridgeMessageType.CrossHappReputation,
    timestamp: Date.now(),
    sourceHapp,
    agent,
    scores,
    aggregate: calculateAggregateReputation(scores),
  };
}

/**
 * Create a credential verification request
 * @throws {BridgeError} If any string parameter is empty
 */
export function createCredentialVerification(
  sourceHapp: string,
  credentialHash: string,
  issuerHapp: string
): CredentialVerificationMessage {
  validate()
    .notEmpty('sourceHapp', sourceHapp)
    .notEmpty('credentialHash', credentialHash)
    .notEmpty('issuerHapp', issuerHapp)
    .throwIfInvalid();

  return {
    type: BridgeMessageType.CredentialVerification,
    timestamp: Date.now(),
    sourceHapp,
    credentialHash,
    issuerHapp,
  };
}

/**
 * Create a verification result
 * @throws {BridgeError} If sourceHapp, credentialHash, or issuer is empty
 */
export function createVerificationResult(
  sourceHapp: string,
  credentialHash: string,
  valid: boolean,
  issuer: string,
  claims: string[]
): VerificationResultMessage {
  validate()
    .notEmpty('sourceHapp', sourceHapp)
    .notEmpty('credentialHash', credentialHash)
    .notEmpty('issuer', issuer)
    .throwIfInvalid();
  assertDefined(claims, 'claims');

  return {
    type: BridgeMessageType.VerificationResult,
    timestamp: Date.now(),
    sourceHapp,
    credentialHash,
    valid,
    issuer,
    claims,
  };
}

/**
 * Create a broadcast event
 * @throws {BridgeError} If sourceHapp or eventType is empty
 */
export function createBroadcastEvent(
  sourceHapp: string,
  eventType: string,
  payload: Uint8Array
): BroadcastEventMessage {
  validate().notEmpty('sourceHapp', sourceHapp).notEmpty('eventType', eventType).throwIfInvalid();
  assertDefined(payload, 'payload');

  return {
    type: BridgeMessageType.BroadcastEvent,
    timestamp: Date.now(),
    sourceHapp,
    eventType,
    payload,
  };
}

/**
 * Create a hApp registration message
 * @throws {BridgeError} If any string parameter is empty
 */
export function createHappRegistration(
  sourceHapp: string,
  happId: string,
  happName: string,
  capabilities: string[]
): HappRegistrationMessage {
  validate()
    .notEmpty('sourceHapp', sourceHapp)
    .notEmpty('happId', happId)
    .notEmpty('happName', happName)
    .throwIfInvalid();
  assertDefined(capabilities, 'capabilities');

  return {
    type: BridgeMessageType.HappRegistration,
    timestamp: Date.now(),
    sourceHapp,
    happId,
    happName,
    capabilities,
  };
}

/**
 * Local bridge for testing (simulates inter-hApp communication)
 */
export class LocalBridge {
  private handlers: Map<string, Map<BridgeMessageType, (msg: AnyBridgeMessage) => void>> =
    new Map();
  private reputations: Map<string, Map<string, ReputationScore>> = new Map();

  /**
   * Register a hApp with the bridge
   * @throws {BridgeError} If happId is empty
   */
  registerHapp(happId: string): void {
    validate().notEmpty('happId', happId).throwIfInvalid();

    if (!this.handlers.has(happId)) {
      this.handlers.set(happId, new Map());
    }
    if (!this.reputations.has(happId)) {
      this.reputations.set(happId, new Map());
    }
  }

  /**
   * Check if a hApp is registered
   */
  isRegistered(happId: string): boolean {
    return this.handlers.has(happId);
  }

  /**
   * Get list of registered hApps
   */
  getRegisteredHapps(): string[] {
    return Array.from(this.handlers.keys());
  }

  /**
   * Register a message handler for a hApp
   * @throws {BridgeError} If happId is empty or handler is not defined
   */
  on(happId: string, type: BridgeMessageType, handler: (msg: AnyBridgeMessage) => void): void {
    validate().notEmpty('happId', happId).throwIfInvalid();
    assertDefined(handler, 'handler');

    this.registerHapp(happId);
    this.handlers.get(happId)!.set(type, handler);
  }

  /**
   * Send a message to a specific hApp
   * @throws {BridgeError} If targetHapp is empty or message is not defined
   */
  send(targetHapp: string, message: AnyBridgeMessage): void {
    validate().notEmpty('targetHapp', targetHapp).throwIfInvalid();
    assertDefined(message, 'message');

    const handlers = this.handlers.get(targetHapp);
    if (handlers && handlers.has(message.type)) {
      handlers.get(message.type)!(message);
    }
  }

  /**
   * Broadcast a message to all registered hApps
   * @throws {BridgeError} If message is not defined
   */
  broadcast(message: AnyBridgeMessage): void {
    assertDefined(message, 'message');

    for (const [happId, handlers] of this.handlers) {
      if (happId !== message.sourceHapp && handlers.has(message.type)) {
        handlers.get(message.type)!(message);
      }
    }
  }

  /**
   * Store reputation for an agent in a hApp
   * @throws {BridgeError} If happId or agentId is empty
   */
  setReputation(happId: string, agentId: string, reputation: ReputationScore): void {
    validate().notEmpty('happId', happId).notEmpty('agentId', agentId).throwIfInvalid();
    assertDefined(reputation, 'reputation');

    this.registerHapp(happId);
    this.reputations.get(happId)!.set(agentId, reputation);
  }

  /**
   * Get cross-hApp reputation for an agent
   * @throws {BridgeError} If agentId is empty
   */
  getCrossHappReputation(agentId: string): HappReputationScore[] {
    validate().notEmpty('agentId', agentId).throwIfInvalid();

    const scores: HappReputationScore[] = [];

    for (const [happId, reputations] of this.reputations) {
      const reputation = reputations.get(agentId);
      if (reputation) {
        scores.push({
          happ: happId,
          score: reputationValue(reputation),
          weight: 1.0,
          lastUpdate: reputation.lastUpdate,
        });
      }
    }

    return scores;
  }

  /**
   * Get aggregate reputation for an agent
   * @throws {BridgeError} If agentId is empty
   */
  getAggregateReputation(agentId: string): number {
    return calculateAggregateReputation(this.getCrossHappReputation(agentId));
  }

  /**
   * Unregister a hApp and remove all its data
   * @throws {BridgeError} If happId is empty
   */
  unregisterHapp(happId: string): void {
    validate().notEmpty('happId', happId).throwIfInvalid();
    this.handlers.delete(happId);
    this.reputations.delete(happId);
  }

  /**
   * Clear all registrations and data
   */
  clear(): void {
    this.handlers.clear();
    this.reputations.clear();
  }

  /**
   * Remove reputation for an agent from a specific hApp
   * @throws {BridgeError} If happId or agentId is empty
   */
  removeReputation(happId: string, agentId: string): void {
    validate().notEmpty('happId', happId).notEmpty('agentId', agentId).throwIfInvalid();

    this.reputations.get(happId)?.delete(agentId);
  }

  /**
   * Remove a message handler for a hApp
   * @throws {BridgeError} If happId is empty
   */
  off(happId: string, type: BridgeMessageType): void {
    validate().notEmpty('happId', happId).throwIfInvalid();
    this.handlers.get(happId)?.delete(type);
  }
}

// ============================================================================
// Bridge Router
// ============================================================================

/**
 * Handler function for a specific message type
 */
export type MessageHandler<T extends AnyBridgeMessage> = (message: T) => void | Promise<void>;

/**
 * Async handler function for a specific message type
 */
export type AsyncMessageHandler<T extends AnyBridgeMessage> = (message: T) => Promise<void>;

/**
 * Middleware function for processing messages
 */
export type BridgeMiddleware = (
  message: AnyBridgeMessage,
  next: () => Promise<void>
) => Promise<void>;

/**
 * Handler map with type-safe message handlers
 */
export interface BridgeRouterHandlers {
  onReputationQuery?: MessageHandler<ReputationQueryMessage>;
  onCrossHappReputation?: MessageHandler<CrossHappReputationMessage>;
  onCredentialVerification?: MessageHandler<CredentialVerificationMessage>;
  onVerificationResult?: MessageHandler<VerificationResultMessage>;
  onBroadcastEvent?: MessageHandler<BroadcastEventMessage>;
  onHappRegistration?: MessageHandler<HappRegistrationMessage>;
  onUnhandled?: MessageHandler<AnyBridgeMessage>;
}

/**
 * Configuration for BridgeRouter
 */
export interface BridgeRouterConfig {
  /** Throw error if no handler is found (default: false) */
  throwOnUnhandled?: boolean;
  /** Log unhandled messages (default: false) */
  logUnhandled?: boolean;
  /** Custom logger */
  logger?: (message: string, data: Record<string, unknown>) => void;
}

/**
 * Type-safe message router for bridge communications.
 * Provides pattern matching, middleware support, and exhaustiveness checking.
 *
 * @example
 * ```typescript
 * const router = new BridgeRouter({
 *   onReputationQuery: (msg) => {
 *     console.log(`Query for agent ${msg.agent}`);
 *   },
 *   onCredentialVerification: (msg) => {
 *     console.log(`Verify credential ${msg.credentialHash}`);
 *   },
 * });
 *
 * // Add middleware
 * router.use(async (msg, next) => {
 *   console.log(`Received ${msg.type}`);
 *   await next();
 * });
 *
 * // Route messages
 * await router.route(incomingMessage);
 * ```
 */
export class BridgeRouter {
  private handlers: BridgeRouterHandlers;
  private middleware: BridgeMiddleware[] = [];
  private config: BridgeRouterConfig;
  private stats = {
    messagesRouted: 0,
    messagesUnhandled: 0,
    byType: new Map<BridgeMessageType, number>(),
  };

  constructor(handlers: BridgeRouterHandlers = {}, config: BridgeRouterConfig = {}) {
    this.handlers = handlers;
    this.config = {
      throwOnUnhandled: false,
      logUnhandled: false,
      ...config,
    };
  }

  /**
   * Add middleware to the router.
   * Middleware is executed in order before the handler.
   *
   * @param middleware - Middleware function
   * @returns This router for chaining
   */
  use(middleware: BridgeMiddleware): this {
    this.middleware.push(middleware);
    return this;
  }

  /**
   * Register a handler for a specific message type.
   *
   * @param type - The message type to handle
   * @param handler - The handler function
   * @returns This router for chaining
   */
  on<T extends BridgeMessageType>(
    type: T,
    handler: MessageHandler<Extract<AnyBridgeMessage, { type: T }>>
  ): this {
    switch (type) {
      case BridgeMessageType.ReputationQuery:
        this.handlers.onReputationQuery = handler as MessageHandler<ReputationQueryMessage>;
        break;
      case BridgeMessageType.CrossHappReputation:
        this.handlers.onCrossHappReputation = handler as MessageHandler<CrossHappReputationMessage>;
        break;
      case BridgeMessageType.CredentialVerification:
        this.handlers.onCredentialVerification =
          handler as MessageHandler<CredentialVerificationMessage>;
        break;
      case BridgeMessageType.VerificationResult:
        this.handlers.onVerificationResult = handler as MessageHandler<VerificationResultMessage>;
        break;
      case BridgeMessageType.BroadcastEvent:
        this.handlers.onBroadcastEvent = handler as MessageHandler<BroadcastEventMessage>;
        break;
      case BridgeMessageType.HappRegistration:
        this.handlers.onHappRegistration = handler as MessageHandler<HappRegistrationMessage>;
        break;
    }
    return this;
  }

  /**
   * Route a message to the appropriate handler.
   *
   * @param message - The message to route
   * @throws {BridgeError} If throwOnUnhandled is true and no handler is found
   */
  async route(message: AnyBridgeMessage): Promise<void> {
    this.stats.messagesRouted++;
    this.stats.byType.set(message.type, (this.stats.byType.get(message.type) || 0) + 1);

    // Build middleware chain
    let index = 0;
    const executeNext = async (): Promise<void> => {
      if (index < this.middleware.length) {
        const mw = this.middleware[index];
        index++;
        await mw(message, executeNext);
      } else {
        await this.dispatch(message);
      }
    };

    await executeNext();
  }

  /**
   * Dispatch message to the appropriate handler.
   */
  private async dispatch(message: AnyBridgeMessage): Promise<void> {
    let handled = false;

    switch (message.type) {
      case BridgeMessageType.ReputationQuery:
        if (this.handlers.onReputationQuery) {
          await this.handlers.onReputationQuery(message);
          handled = true;
        }
        break;
      case BridgeMessageType.CrossHappReputation:
        if (this.handlers.onCrossHappReputation) {
          await this.handlers.onCrossHappReputation(message);
          handled = true;
        }
        break;
      case BridgeMessageType.CredentialVerification:
        if (this.handlers.onCredentialVerification) {
          await this.handlers.onCredentialVerification(message);
          handled = true;
        }
        break;
      case BridgeMessageType.VerificationResult:
        if (this.handlers.onVerificationResult) {
          await this.handlers.onVerificationResult(message);
          handled = true;
        }
        break;
      case BridgeMessageType.BroadcastEvent:
        if (this.handlers.onBroadcastEvent) {
          await this.handlers.onBroadcastEvent(message);
          handled = true;
        }
        break;
      case BridgeMessageType.HappRegistration:
        if (this.handlers.onHappRegistration) {
          await this.handlers.onHappRegistration(message);
          handled = true;
        }
        break;
      default: {
        // Exhaustiveness check - this should never happen
        const _exhaustive: never = message;
        throw new BridgeError(
          `Unknown message type: ${(_exhaustive as AnyBridgeMessage).type}`,
          ErrorCode.INVALID_ARGUMENT,
          { message: _exhaustive }
        );
      }
    }

    if (!handled) {
      this.stats.messagesUnhandled++;

      if (this.handlers.onUnhandled) {
        await this.handlers.onUnhandled(message);
      } else if (this.config.logUnhandled && this.config.logger) {
        this.config.logger('Unhandled message', {
          type: message.type,
          sourceHapp: message.sourceHapp,
        });
      }

      if (this.config.throwOnUnhandled) {
        throw new BridgeError(
          `No handler for message type: ${message.type}`,
          ErrorCode.BRIDGE_HAPP_NOT_FOUND,
          { type: message.type }
        );
      }
    }
  }

  /**
   * Route multiple messages in sequence.
   *
   * @param messages - Array of messages to route
   */
  async routeMany(messages: AnyBridgeMessage[]): Promise<void> {
    for (const message of messages) {
      await this.route(message);
    }
  }

  /**
   * Route multiple messages in parallel.
   *
   * @param messages - Array of messages to route
   */
  async routeParallel(messages: AnyBridgeMessage[]): Promise<void> {
    await Promise.all(messages.map((m) => this.route(m)));
  }

  /**
   * Check if a handler is registered for a message type.
   *
   * @param type - The message type
   * @returns True if a handler is registered
   */
  hasHandler(type: BridgeMessageType): boolean {
    switch (type) {
      case BridgeMessageType.ReputationQuery:
        return !!this.handlers.onReputationQuery;
      case BridgeMessageType.CrossHappReputation:
        return !!this.handlers.onCrossHappReputation;
      case BridgeMessageType.CredentialVerification:
        return !!this.handlers.onCredentialVerification;
      case BridgeMessageType.VerificationResult:
        return !!this.handlers.onVerificationResult;
      case BridgeMessageType.BroadcastEvent:
        return !!this.handlers.onBroadcastEvent;
      case BridgeMessageType.HappRegistration:
        return !!this.handlers.onHappRegistration;
      default:
        return false;
    }
  }

  /**
   * Get list of message types with handlers.
   */
  getHandledTypes(): BridgeMessageType[] {
    const types: BridgeMessageType[] = [];
    if (this.handlers.onReputationQuery) types.push(BridgeMessageType.ReputationQuery);
    if (this.handlers.onCrossHappReputation) types.push(BridgeMessageType.CrossHappReputation);
    if (this.handlers.onCredentialVerification)
      types.push(BridgeMessageType.CredentialVerification);
    if (this.handlers.onVerificationResult) types.push(BridgeMessageType.VerificationResult);
    if (this.handlers.onBroadcastEvent) types.push(BridgeMessageType.BroadcastEvent);
    if (this.handlers.onHappRegistration) types.push(BridgeMessageType.HappRegistration);
    return types;
  }

  /**
   * Get routing statistics.
   */
  getStats(): {
    messagesRouted: number;
    messagesUnhandled: number;
    byType: Record<string, number>;
  } {
    const byType: Record<string, number> = {};
    for (const [type, count] of this.stats.byType) {
      byType[type] = count;
    }
    return {
      messagesRouted: this.stats.messagesRouted,
      messagesUnhandled: this.stats.messagesUnhandled,
      byType,
    };
  }

  /**
   * Reset statistics.
   */
  resetStats(): void {
    this.stats.messagesRouted = 0;
    this.stats.messagesUnhandled = 0;
    this.stats.byType.clear();
  }

  /**
   * Remove all handlers.
   */
  clear(): void {
    this.handlers = {};
    this.middleware = [];
  }
}

/**
 * Create a simple message handler for specific types.
 * Useful for one-off message handling.
 *
 * @example
 * ```typescript
 * const handleQuery = createMessageHandler(BridgeMessageType.ReputationQuery, (msg) => {
 *   console.log(`Query for ${msg.agent}`);
 * });
 *
 * if (handleQuery.matches(message)) {
 *   handleQuery.handle(message);
 * }
 * ```
 */
export function createMessageHandler<T extends BridgeMessageType>(
  type: T,
  handler: MessageHandler<Extract<AnyBridgeMessage, { type: T }>>
): {
  type: T;
  matches: (message: AnyBridgeMessage) => message is Extract<AnyBridgeMessage, { type: T }>;
  handle: (message: Extract<AnyBridgeMessage, { type: T }>) => void | Promise<void>;
} {
  return {
    type,
    matches: (message: AnyBridgeMessage): message is Extract<AnyBridgeMessage, { type: T }> => {
      return message.type === type;
    },
    handle: handler,
  };
}

// Re-export cross-hApp bridge for ecosystem coordination
export * from './cross-happ.js';

// Re-export cross-hApp workflows for civilizational operations
export * from './workflows.js';

// Re-export epistemic evidence bridge for Knowledge ↔ Justice integration
export * from './epistemic-evidence.js';

// Re-export PQC signatures for secure cross-hApp messaging
export * from './pqc-signatures.js';

// Re-export capability-based access control
export * from './capability-access.js';

// Re-export cross-hApp event bus for pub/sub messaging
export * from './event-bus.js';

// Re-export unified identity verification for cross-hApp identity
export * from './unified-identity.js';

// Re-export cross-domain reputation service for multi-hApp trust aggregation
export * from './cross-domain-reputation.js';
