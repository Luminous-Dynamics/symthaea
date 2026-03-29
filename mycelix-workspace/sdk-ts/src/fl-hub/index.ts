// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Federated Learning Hub
 *
 * A comprehensive infrastructure for coordinating privacy-preserving
 * federated learning across the Mycelix ecosystem.
 *
 * Features:
 * - Session management for multi-round training
 * - Model registry for versioning and deployment
 * - Differential privacy with budget tracking
 * - Multiple aggregation strategies
 * - MATL trust integration
 *
 * @example Basic Usage
 * ```typescript
 * import { flHub } from '@mycelix/sdk';
 *
 * // Create the hub
 * const hub = flHub.createFLHub();
 *
 * // Register a model
 * const model = hub.getRegistry().registerModel({
 *   name: 'spam-classifier',
 *   architecture: 'mlp',
 *   inputShape: [256],
 *   outputShape: [2],
 *   parameterCount: 10000,
 *   createdBy: 'agent-1',
 * });
 *
 * // Create a training session
 * const session = hub.createSession({
 *   name: 'Round 1 Training',
 *   modelId: model.modelId,
 *   totalRounds: 10,
 *   minParticipants: 3,
 *   maxParticipants: 10,
 *   roundTimeout: 60000,
 *   aggregationMethod: AggregationMethod.FedAvg,
 * }, 'coordinator-agent');
 *
 * // Start recruiting
 * hub.startRecruiting(session.sessionId);
 *
 * // Participants join
 * hub.joinSession(session.sessionId, 'participant-1', 0.85);
 * hub.joinSession(session.sessionId, 'participant-2', 0.90);
 * hub.joinSession(session.sessionId, 'participant-3', 0.78);
 *
 * // Training starts automatically when min participants reached
 * ```
 *
 * @example With Differential Privacy
 * ```typescript
 * import { flHub } from '@mycelix/sdk';
 *
 * // Create with privacy budget
 * const session = hub.createSession({
 *   name: 'Private Training',
 *   modelId: model.modelId,
 *   totalRounds: 10,
 *   minParticipants: 5,
 *   maxParticipants: 20,
 *   roundTimeout: 120000,
 *   aggregationMethod: AggregationMethod.FedAvg,
 *   privacyBudget: {
 *     epsilon: 5,
 *     delta: 1e-5,
 *     accountingMethod: 'rdp',
 *   },
 * }, 'coordinator');
 * ```
 *
 * @example Model Deployment
 * ```typescript
 * import { flHub } from '@mycelix/sdk';
 *
 * const registry = hub.getRegistry();
 *
 * // After training completes, get the best checkpoint
 * const best = registry.getBestCheckpoint(model.modelId, 'loss', true);
 *
 * // Deploy it
 * const deployment = registry.deployModel({
 *   checkpointId: best.checkpointId,
 *   endpoint: 'https://api.mycelix.net/models/spam-classifier',
 * });
 *
 * console.log(`Model deployed: ${deployment.deploymentId}`);
 * ```
 *
 * @module fl-hub
 */

// Types
export {
  type SessionId,
  type ModelId,
  type ModelArchitecture,
  type SessionStatus,
  type ParticipantStatus,
  type ModelMetadata,
  type ModelCheckpoint,
  type ModelDeployment,
  type SessionConfig,
  type TrainingSession,
  type SessionParticipant,
  type ParticipantContribution,
  type ParticipantCriteria,
  type TrainingMetrics,
  type SessionMetrics,
  type RoundMetrics,
  type PrivacyBudget,
  type PrivacyState,
  type RoundPrivacyAllocation,
  type DifferentialPrivacyConfig,
  type ParticipantUpdate,
  type AggregatedUpdate,
  type SessionEvent,
  type SessionEventType,
  type FLHubConfig,
  DEFAULT_FL_HUB_CONFIG,
} from './types.js';

// Model Registry
export {
  ModelRegistry,
  createModelRegistry,
  type ModelRegistryConfig,
  type RegisterModelInput,
  type SaveCheckpointInput,
  type DeployModelInput,
  type ModelFilter,
  type RegistryStats,
} from './registry.js';

// Privacy Manager
export {
  PrivacyManager,
  PrivateAggregator,
  createPrivacyManager,
  createPrivateAggregator,
  createDevPrivacyBudget,
  createStrictPrivacyBudget,
  type PrivacyEstimate,
  type AggregationResult,
} from './privacy.js';

// Hub Coordinator
export {
  FLHubCoordinator,
  createFLHub,
  createFLHubWithRegistry,
  type FLHubCallbacks,
  type HubStats,
} from './coordinator.js';
