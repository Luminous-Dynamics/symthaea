/**
 * FL Hub Coordinator
 *
 * Coordinates federated learning sessions across the Mycelix ecosystem.
 * Manages participant recruitment, training rounds, and model aggregation.
 */

import { PrivateAggregator } from './privacy.js';
import { ModelRegistry } from './registry.js';
import {
  DEFAULT_FL_HUB_CONFIG,
  type SessionId,
  type SessionConfig,
  type TrainingSession,
  type SessionStatus,
  type SessionParticipant,
  type RoundMetrics,
  type TrainingMetrics,
  type ParticipantUpdate,
  type AggregatedUpdate,
  type SessionEvent,
  type SessionEventType,
  type FLHubConfig,
} from './types.js';
import {
  fedAvg,
  trimmedMean,
  coordinateMedian,
  krum,
  detectByzantine,
  AggregationMethod,
  type GradientUpdate,
} from '../fl/index.js';

import type { AgentId, TrustScore } from '../utils/index.js';

// =============================================================================
// FL Hub Coordinator
// =============================================================================

export interface FLHubCallbacks {
  onSessionEvent?: (event: SessionEvent) => void;
  onParticipantJoined?: (sessionId: SessionId, participant: SessionParticipant) => void;
  onRoundCompleted?: (sessionId: SessionId, round: number, metrics: TrainingMetrics) => void;
  onSessionCompleted?: (session: TrainingSession) => void;
}

/**
 * Central coordinator for federated learning sessions
 */
export class FLHubCoordinator {
  private config: FLHubConfig;
  private registry: ModelRegistry;
  private sessions: Map<SessionId, TrainingSession> = new Map();
  private pendingUpdates: Map<SessionId, Map<AgentId, ParticipantUpdate>> = new Map();
  private callbacks: FLHubCallbacks;
  private privacyManagers: Map<SessionId, PrivateAggregator> = new Map();

  constructor(
    config?: Partial<FLHubConfig>,
    registry?: ModelRegistry,
    callbacks?: FLHubCallbacks
  ) {
    this.config = { ...DEFAULT_FL_HUB_CONFIG, ...config };
    this.registry = registry ?? new ModelRegistry();
    this.callbacks = callbacks ?? {};
  }

  // ===========================================================================
  // Session Management
  // ===========================================================================

  /**
   * Create a new training session
   */
  createSession(config: SessionConfig, createdBy: AgentId): TrainingSession {
    // Check if model exists
    if (!this.registry.getModel(config.modelId)) {
      throw new Error(`Model ${config.modelId} not found in registry`);
    }

    // Check concurrent session limit
    const activeSessions = this.getActiveSessions();
    if (activeSessions.length >= this.config.maxConcurrentSessions) {
      throw new Error(
        `Maximum concurrent sessions (${this.config.maxConcurrentSessions}) reached`
      );
    }

    const sessionId = this.generateSessionId();
    // Validate byzantineTolerance
    const byzantineTolerance = config.byzantineTolerance ?? this.config.defaultByzantineTolerance;
    if (byzantineTolerance < 0 || byzantineTolerance > 0.34) {
      throw new Error(
        `byzantineTolerance must be between 0 and 0.34 (34% validated), got ${byzantineTolerance}`
      );
    }

    const session: TrainingSession = {
      sessionId,
      config: {
        ...config,
        roundTimeout: config.roundTimeout ?? this.config.defaultRoundTimeout,
        minParticipants: config.minParticipants ?? this.config.defaultMinParticipants,
        byzantineTolerance,
      },
      status: 'created',
      currentRound: 0,
      participants: [],
      createdBy,
      metrics: {
        totalRoundsCompleted: 0,
        totalParticipants: 0,
        activeParticipants: 0,
        averageTrustScore: 0,
        totalDataSize: 0,
        metricsHistory: [],
      },
    };

    this.sessions.set(sessionId, session);
    this.pendingUpdates.set(sessionId, new Map());

    // Initialize privacy manager if budget specified
    if (config.privacyBudget) {
      this.privacyManagers.set(
        sessionId,
        new PrivateAggregator(config.privacyBudget)
      );
    }

    this.emitEvent('session_created', sessionId, { config });
    return session;
  }

  /**
   * Get a session by ID
   */
  getSession(sessionId: SessionId): TrainingSession | null {
    return this.sessions.get(sessionId) ?? null;
  }

  /**
   * List all sessions
   */
  listSessions(status?: SessionStatus): TrainingSession[] {
    const sessions = Array.from(this.sessions.values());
    if (status) {
      return sessions.filter((s) => s.status === status);
    }
    return sessions;
  }

  /**
   * Get active (non-completed) sessions
   */
  getActiveSessions(): TrainingSession[] {
    return Array.from(this.sessions.values()).filter(
      (s) => !['completed', 'failed', 'cancelled'].includes(s.status)
    );
  }

  /**
   * Start recruiting participants
   */
  startRecruiting(sessionId: SessionId): boolean {
    const session = this.sessions.get(sessionId);
    if (!session || session.status !== 'created') {
      return false;
    }

    session.status = 'recruiting';
    return true;
  }

  /**
   * Cancel a session
   */
  cancelSession(sessionId: SessionId, reason?: string): boolean {
    const session = this.sessions.get(sessionId);
    if (!session) return false;

    if (['completed', 'cancelled'].includes(session.status)) {
      return false;
    }

    session.status = 'cancelled';
    this.emitEvent('session_failed', sessionId, { reason: reason ?? 'Cancelled by coordinator' });
    return true;
  }

  // ===========================================================================
  // Participant Management
  // ===========================================================================

  /**
   * Add a participant to a session
   */
  joinSession(
    sessionId: SessionId,
    agentId: AgentId,
    trustScore: TrustScore,
    dataSize?: number
  ): boolean {
    const session = this.sessions.get(sessionId);
    if (!session || session.status !== 'recruiting') {
      return false;
    }

    // Check if already joined
    if (session.participants.some((p) => p.agentId === agentId)) {
      return false;
    }

    // Check participant limit
    if (session.participants.length >= session.config.maxParticipants) {
      return false;
    }

    // Check selection criteria
    if (session.config.selectionCriteria) {
      if (trustScore < session.config.selectionCriteria.minTrustScore) {
        return false;
      }
    }

    const participant: SessionParticipant = {
      agentId,
      status: 'accepted',
      joinedAt: Date.now(),
      trustScore,
      roundsCompleted: 0,
      dataSize,
    };

    session.participants.push(participant);
    this.updateSessionMetrics(session);

    this.emitEvent('participant_joined', sessionId, { participant });
    this.callbacks.onParticipantJoined?.(sessionId, participant);

    // Auto-start if minimum participants reached
    if (
      session.participants.length >= session.config.minParticipants &&
      session.status === 'recruiting'
    ) {
      this.startTraining(sessionId);
    }

    return true;
  }

  /**
   * Remove a participant from a session
   */
  leaveSession(sessionId: SessionId, agentId: AgentId): boolean {
    const session = this.sessions.get(sessionId);
    if (!session) return false;

    const idx = session.participants.findIndex((p) => p.agentId === agentId);
    if (idx === -1) return false;

    session.participants.splice(idx, 1);
    this.updateSessionMetrics(session);

    this.emitEvent('participant_left', sessionId, { agentId });
    return true;
  }

  /**
   * Get participant info
   */
  getParticipant(
    sessionId: SessionId,
    agentId: AgentId
  ): SessionParticipant | null {
    const session = this.sessions.get(sessionId);
    if (!session) return null;

    return session.participants.find((p) => p.agentId === agentId) ?? null;
  }

  // ===========================================================================
  // Training Control
  // ===========================================================================

  /**
   * Start the training process
   */
  startTraining(sessionId: SessionId): boolean {
    const session = this.sessions.get(sessionId);
    if (!session) return false;

    if (session.participants.length < session.config.minParticipants) {
      return false;
    }

    session.status = 'training';
    session.startedAt = Date.now();
    session.currentRound = 1;

    // Mark participants as training
    for (const p of session.participants) {
      p.status = 'training';
    }

    this.emitEvent('round_started', sessionId, { round: 1 });

    // Set round timeout
    setTimeout(
      () => this.checkRoundTimeout(sessionId, 1),
      session.config.roundTimeout
    );

    return true;
  }

  /**
   * Submit a training update
   */
  async submitUpdate(update: ParticipantUpdate): Promise<boolean> {
    const session = this.sessions.get(update.sessionId);
    if (!session || session.status !== 'training') {
      return false;
    }

    if (update.round !== session.currentRound) {
      return false;
    }

    const participant = session.participants.find(
      (p) => p.agentId === update.participantId
    );
    if (!participant || participant.status !== 'training') {
      return false;
    }

    // Store the update
    const roundUpdates = this.pendingUpdates.get(update.sessionId);
    if (!roundUpdates) return false;

    roundUpdates.set(update.participantId, update);

    // Update participant status
    participant.status = 'submitted';
    participant.lastUpdateAt = update.timestamp;

    this.emitEvent('update_received', update.sessionId, {
      participantId: update.participantId,
      round: update.round,
    });

    // Check if all participants have submitted
    const submittedCount = session.participants.filter(
      (p) => p.status === 'submitted'
    ).length;

    if (submittedCount >= session.config.minParticipants) {
      await this.aggregateRound(update.sessionId);
    }

    return true;
  }

  // ===========================================================================
  // Aggregation
  // ===========================================================================

  /**
   * Aggregate updates for the current round
   */
  async aggregateRound(sessionId: SessionId): Promise<AggregatedUpdate | null> {
    const session = this.sessions.get(sessionId);
    if (!session) return null;

    session.status = 'aggregating';

    const roundUpdates = this.pendingUpdates.get(sessionId);
    if (!roundUpdates || roundUpdates.size === 0) {
      return null;
    }

    const updates = Array.from(roundUpdates.values());
    const round = session.currentRound;

    // Aggregate based on configured method
    let aggregatedGradients: number[];
    const gradientArrays = updates.map((u) => this.deserializeGradients(u.gradients));

    // Check for privacy aggregation
    const privacyAggregator = this.privacyManagers.get(sessionId);

    if (privacyAggregator) {
      const result = privacyAggregator.aggregateWithPrivacy(gradientArrays, round);
      aggregatedGradients = result.gradients;
    } else {
      aggregatedGradients = await this.performAggregation(
        gradientArrays,
        updates,
        session.config.aggregationMethod,
        session.config.byzantineTolerance ?? this.config.defaultByzantineTolerance
      );
    }

    // Calculate aggregated metrics
    const aggregatedMetrics = this.aggregateMetrics(updates.map((u) => u.metrics));

    // Save checkpoint
    this.registry.saveCheckpoint({
      modelId: session.config.modelId,
      sessionId,
      round,
      parameters: this.serializeGradients(aggregatedGradients),
      metrics: aggregatedMetrics,
      participantCount: updates.length,
      aggregationMethod: session.config.aggregationMethod,
    });

    // Update session metrics
    const roundMetrics: RoundMetrics = {
      round,
      participantCount: updates.length,
      metrics: aggregatedMetrics,
      aggregationTime: Date.now() - (session.startedAt ?? Date.now()),
      timestamp: Date.now(),
    };
    session.metrics.metricsHistory.push(roundMetrics);
    session.metrics.totalRoundsCompleted = round;
    session.metrics.latestMetrics = aggregatedMetrics;

    // Update participant stats
    for (const update of updates) {
      const participant = session.participants.find(
        (p) => p.agentId === update.participantId
      );
      if (participant) {
        participant.roundsCompleted++;
        participant.status = 'validated';
      }
    }

    // Emit events
    this.emitEvent('round_completed', sessionId, { round, metrics: aggregatedMetrics });
    this.callbacks.onRoundCompleted?.(sessionId, round, aggregatedMetrics);

    // Prepare for next round or complete
    if (round >= session.config.totalRounds) {
      this.completeSession(sessionId);
    } else {
      await this.startNextRound(sessionId);
    }

    // Clear pending updates
    roundUpdates.clear();

    return {
      sessionId,
      round,
      parameters: this.serializeGradients(aggregatedGradients),
      metrics: aggregatedMetrics,
      participantCount: updates.length,
      timestamp: Date.now(),
      nextRoundDeadline:
        round < session.config.totalRounds
          ? Date.now() + session.config.roundTimeout
          : undefined,
    };
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  private async performAggregation(
    gradients: number[][],
    updates: ParticipantUpdate[],
    method: AggregationMethod,
    byzantineTolerance: number = 0.33
  ): Promise<number[]> {
    // Convert to GradientUpdate format compatible with existing FL module
    const flUpdates: GradientUpdate[] = updates.map((u, i) => ({
      participantId: u.participantId,
      modelVersion: u.round,
      gradients: Float64Array.from(gradients[i]),
      metadata: {
        batchSize: u.dataSize,
        loss: u.metrics.loss,
        accuracy: u.metrics.accuracy,
        timestamp: u.timestamp,
      },
    }));

    // Byzantine detection and filtering
    let filtered = flUpdates;
    if (flUpdates.length >= 3) {
      const detection = detectByzantine(flUpdates);
      const byzantineFraction = detection.byzantineIndices.length / flUpdates.length;

      if (detection.shouldReject) {
        throw new Error('Too many Byzantine participants detected (>50%)');
      }
      if (byzantineFraction > byzantineTolerance) {
        throw new Error(
          `Byzantine fraction ${(byzantineFraction * 100).toFixed(1)}% exceeds tolerance ${(byzantineTolerance * 100).toFixed(1)}%`
        );
      }

      filtered = flUpdates.filter((_, i) => !detection.byzantineIndices.includes(i));
    }

    let result: Float64Array;
    switch (method) {
      case AggregationMethod.FedAvg:
        result = fedAvg(filtered);
        break;
      case AggregationMethod.TrimmedMean:
        result = trimmedMean(filtered, Math.max(0.1, byzantineTolerance));
        break;
      case AggregationMethod.Median:
        result = coordinateMedian(filtered);
        break;
      case AggregationMethod.Krum:
        result = krum(filtered, Math.floor(filtered.length / 3));
        break;
      default:
        result = fedAvg(filtered);
    }
    return Array.from(result);
  }

  private aggregateMetrics(metricsList: TrainingMetrics[]): TrainingMetrics {
    if (metricsList.length === 0) {
      return { loss: 0 };
    }

    const avgLoss = metricsList.reduce((sum, m) => sum + m.loss, 0) / metricsList.length;
    const avgAccuracy =
      metricsList.filter((m) => m.accuracy !== undefined).length > 0
        ? metricsList.reduce((sum, m) => sum + (m.accuracy ?? 0), 0) / metricsList.length
        : undefined;

    return {
      loss: avgLoss,
      accuracy: avgAccuracy,
    };
  }

  private async startNextRound(sessionId: SessionId): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    session.currentRound++;
    session.status = 'training';

    // Reset participant status
    for (const p of session.participants) {
      if (p.status === 'validated') {
        p.status = 'training';
      }
    }

    this.emitEvent('round_started', sessionId, { round: session.currentRound });

    // Set round timeout
    setTimeout(
      () => this.checkRoundTimeout(sessionId, session.currentRound),
      session.config.roundTimeout
    );
  }

  private completeSession(sessionId: SessionId): void {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    session.status = 'completed';
    session.completedAt = Date.now();

    this.emitEvent('session_completed', sessionId, { metrics: session.metrics });
    this.callbacks.onSessionCompleted?.(session);
  }

  private checkRoundTimeout(sessionId: SessionId, round: number): void {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    // Only timeout if still on the same round
    if (session.currentRound === round && session.status === 'training') {
      const submittedCount = session.participants.filter(
        (p) => p.status === 'submitted'
      ).length;

      if (submittedCount >= session.config.minParticipants) {
        // Enough participants, proceed with aggregation
        void this.aggregateRound(sessionId);
      } else {
        // Not enough participants, fail the session
        session.status = 'failed';
        this.emitEvent('session_failed', sessionId, {
          reason: `Round ${round} timeout: only ${submittedCount} of ${session.config.minParticipants} participants submitted`,
        });
      }
    }
  }

  private updateSessionMetrics(session: TrainingSession): void {
    const activeParticipants = session.participants.filter(
      (p) => p.status !== 'rejected'
    );

    session.metrics.totalParticipants = session.participants.length;
    session.metrics.activeParticipants = activeParticipants.length;
    session.metrics.averageTrustScore =
      activeParticipants.length > 0
        ? activeParticipants.reduce((sum, p) => sum + p.trustScore, 0) /
          activeParticipants.length
        : 0;
    session.metrics.totalDataSize = activeParticipants.reduce(
      (sum, p) => sum + (p.dataSize ?? 0),
      0
    );
  }

  private emitEvent(
    type: SessionEventType,
    sessionId: SessionId,
    data: unknown
  ): void {
    const event: SessionEvent = {
      type,
      sessionId,
      timestamp: Date.now(),
      data,
    };
    this.callbacks.onSessionEvent?.(event);
  }

  private generateSessionId(): SessionId {
    return `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  private serializeGradients(gradients: number[]): Uint8Array {
    const buffer = new ArrayBuffer(gradients.length * 4);
    const view = new Float32Array(buffer);
    view.set(gradients);
    return new Uint8Array(buffer);
  }

  private deserializeGradients(data: Uint8Array): number[] {
    const view = new Float32Array(data.buffer, data.byteOffset, data.length / 4);
    return Array.from(view);
  }

  // ===========================================================================
  // Getters
  // ===========================================================================

  /**
   * Get the model registry
   */
  getRegistry(): ModelRegistry {
    return this.registry;
  }

  /**
   * Get hub statistics
   */
  getStats(): HubStats {
    const sessions = Array.from(this.sessions.values());
    const completed = sessions.filter((s) => s.status === 'completed');
    const failed = sessions.filter((s) => s.status === 'failed');

    return {
      totalSessions: sessions.length,
      activeSessions: this.getActiveSessions().length,
      completedSessions: completed.length,
      failedSessions: failed.length,
      totalRoundsCompleted: sessions.reduce(
        (sum, s) => sum + s.metrics.totalRoundsCompleted,
        0
      ),
      totalParticipants: new Set(
        sessions.flatMap((s) => s.participants.map((p) => p.agentId))
      ).size,
      registryStats: this.registry.getStats(),
    };
  }
}

// =============================================================================
// Types
// =============================================================================

export interface HubStats {
  totalSessions: number;
  activeSessions: number;
  completedSessions: number;
  failedSessions: number;
  totalRoundsCompleted: number;
  totalParticipants: number;
  registryStats: ReturnType<ModelRegistry['getStats']>;
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create an FL Hub coordinator
 */
export function createFLHub(
  config?: Partial<FLHubConfig>,
  callbacks?: FLHubCallbacks
): FLHubCoordinator {
  return new FLHubCoordinator(config, undefined, callbacks);
}

/**
 * Create an FL Hub with a shared registry
 */
export function createFLHubWithRegistry(
  registry: ModelRegistry,
  config?: Partial<FLHubConfig>,
  callbacks?: FLHubCallbacks
): FLHubCoordinator {
  return new FLHubCoordinator(config, registry, callbacks);
}
