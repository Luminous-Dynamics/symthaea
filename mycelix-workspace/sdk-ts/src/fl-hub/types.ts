/**
 * Federated Learning Hub Types
 *
 * Type definitions for the FL Hub infrastructure that coordinates
 * federated learning across the Mycelix ecosystem.
 */

import type { AggregationMethod } from '../fl/index.js';
import type { AgentId, HappId, TrustScore } from '../utils/index.js';

// =============================================================================
// Core Types
// =============================================================================

/**
 * Unique identifier for a training session
 */
export type SessionId = string;

/**
 * Unique identifier for a model
 */
export type ModelId = string;

/**
 * Model architecture types
 */
export type ModelArchitecture =
  | 'linear'
  | 'mlp'
  | 'cnn'
  | 'rnn'
  | 'lstm'
  | 'transformer'
  | 'custom';

/**
 * Training session status
 */
export type SessionStatus =
  | 'created'
  | 'recruiting'
  | 'training'
  | 'aggregating'
  | 'completed'
  | 'failed'
  | 'cancelled';

/**
 * Participant status in a session
 */
export type ParticipantStatus =
  | 'invited'
  | 'accepted'
  | 'training'
  | 'submitted'
  | 'validated'
  | 'rejected';

// =============================================================================
// Model Registry Types
// =============================================================================

/**
 * Model metadata stored in the registry
 */
export interface ModelMetadata {
  modelId: ModelId;
  name: string;
  description: string;
  version: string;
  architecture: ModelArchitecture;
  inputShape: number[];
  outputShape: number[];
  parameterCount: number;
  createdAt: number;
  createdBy: AgentId;
  happId?: HappId;
  tags: string[];
}

/**
 * Trained model checkpoint
 */
export interface ModelCheckpoint {
  checkpointId: string;
  modelId: ModelId;
  sessionId: SessionId;
  round: number;
  parameters: Uint8Array;
  metrics: TrainingMetrics;
  timestamp: number;
  participantCount: number;
  aggregationMethod: AggregationMethod;
}

/**
 * Model deployment configuration
 */
export interface ModelDeployment {
  deploymentId: string;
  modelId: ModelId;
  checkpointId: string;
  deployedAt: number;
  endpoint?: string;
  status: 'active' | 'inactive' | 'deprecated';
}

// =============================================================================
// Training Session Types
// =============================================================================

/**
 * Configuration for a training session
 */
export interface SessionConfig {
  name: string;
  modelId: ModelId;
  totalRounds: number;
  minParticipants: number;
  maxParticipants: number;
  roundTimeout: number; // ms
  aggregationMethod: AggregationMethod;
  byzantineTolerance?: number; // 0-0.34 (34% validated max)
  privacyBudget?: PrivacyBudget;
  selectionCriteria?: ParticipantCriteria;
  validationSplit?: number;
}

/**
 * A federated learning training session
 */
export interface TrainingSession {
  sessionId: SessionId;
  config: SessionConfig;
  status: SessionStatus;
  currentRound: number;
  participants: SessionParticipant[];
  startedAt?: number;
  completedAt?: number;
  createdBy: AgentId;
  metrics: SessionMetrics;
}

/**
 * Participant in a training session
 */
export interface SessionParticipant {
  agentId: AgentId;
  status: ParticipantStatus;
  joinedAt: number;
  trustScore: TrustScore;
  roundsCompleted: number;
  lastUpdateAt?: number;
  dataSize?: number;
  contribution?: ParticipantContribution;
}

/**
 * Contribution metrics for a participant
 */
export interface ParticipantContribution {
  totalUpdates: number;
  acceptedUpdates: number;
  rejectedUpdates: number;
  averageQuality: number;
  privacyBudgetUsed: number;
}

/**
 * Criteria for selecting participants
 */
export interface ParticipantCriteria {
  minTrustScore: number;
  requiredHapps?: HappId[];
  minDataSize?: number;
  maxDataSize?: number;
  geographicRegions?: string[];
}

// =============================================================================
// Metrics Types
// =============================================================================

/**
 * Training metrics for a round or checkpoint
 */
export interface TrainingMetrics {
  loss: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1Score?: number;
  customMetrics?: Record<string, number>;
}

/**
 * Aggregated session metrics
 */
export interface SessionMetrics {
  totalRoundsCompleted: number;
  totalParticipants: number;
  activeParticipants: number;
  averageTrustScore: number;
  totalDataSize: number;
  latestMetrics?: TrainingMetrics;
  metricsHistory: RoundMetrics[];
  privacyBudgetRemaining?: number;
}

/**
 * Metrics for a single round
 */
export interface RoundMetrics {
  round: number;
  participantCount: number;
  metrics: TrainingMetrics;
  aggregationTime: number;
  timestamp: number;
}

// =============================================================================
// Privacy Types
// =============================================================================

/**
 * Privacy budget configuration
 */
export interface PrivacyBudget {
  epsilon: number; // Total epsilon budget
  delta: number; // Delta parameter
  accountingMethod: 'simple' | 'rdp' | 'gdp';
}

/**
 * Privacy budget state
 */
export interface PrivacyState {
  totalBudget: PrivacyBudget;
  usedEpsilon: number;
  remainingEpsilon: number;
  roundAllocations: RoundPrivacyAllocation[];
}

/**
 * Privacy allocation for a round
 */
export interface RoundPrivacyAllocation {
  round: number;
  epsilonUsed: number;
  noiseScale: number;
  clippingBound: number;
}

/**
 * Differential privacy configuration
 */
export interface DifferentialPrivacyConfig {
  enabled: boolean;
  mechanism: 'gaussian' | 'laplace';
  clippingBound: number;
  noiseMultiplier: number;
  targetEpsilon?: number;
  targetDelta?: number;
}

// =============================================================================
// Communication Types
// =============================================================================

/**
 * Update from a participant
 */
export interface ParticipantUpdate {
  sessionId: SessionId;
  participantId: AgentId;
  round: number;
  gradients: Uint8Array;
  metrics: TrainingMetrics;
  dataSize: number;
  timestamp: number;
  signature?: Uint8Array;
}

/**
 * Aggregated update to send back to participants
 */
export interface AggregatedUpdate {
  sessionId: SessionId;
  round: number;
  parameters: Uint8Array;
  metrics: TrainingMetrics;
  participantCount: number;
  timestamp: number;
  nextRoundDeadline?: number;
}

/**
 * Session event for real-time updates
 */
export interface SessionEvent {
  type: SessionEventType;
  sessionId: SessionId;
  timestamp: number;
  data: unknown;
}

export type SessionEventType =
  | 'session_created'
  | 'participant_joined'
  | 'participant_left'
  | 'round_started'
  | 'update_received'
  | 'round_completed'
  | 'session_completed'
  | 'session_failed';

// =============================================================================
// Hub Configuration
// =============================================================================

/**
 * FL Hub configuration
 */
export interface FLHubConfig {
  maxConcurrentSessions: number;
  defaultRoundTimeout: number;
  defaultMinParticipants: number;
  defaultByzantineTolerance: number; // 0-0.34 (34% validated max)
  enablePrivacy: boolean;
  defaultPrivacyBudget?: PrivacyBudget;
  modelStoragePath?: string;
  enableCompression: boolean;
  compressionLevel?: number;
}

/**
 * Default hub configuration
 */
export const DEFAULT_FL_HUB_CONFIG: FLHubConfig = {
  maxConcurrentSessions: 10,
  defaultRoundTimeout: 300000, // 5 minutes
  defaultMinParticipants: 3,
  defaultByzantineTolerance: 0.33, // 33% Byzantine tolerance (validated max: 34%)
  enablePrivacy: true,
  defaultPrivacyBudget: {
    epsilon: 10,
    delta: 1e-5,
    accountingMethod: 'rdp',
  },
  enableCompression: true,
  compressionLevel: 6,
};
