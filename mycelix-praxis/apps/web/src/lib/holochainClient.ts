// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Real Holochain client for connecting to a Holochain conductor
 *
 * This client connects to a running Holochain conductor via WebSocket and provides
 * typed wrapper functions for all zome calls across the four core zomes:
 * - Learning Zome
 * - FL (Federated Learning) Zome
 * - Credential Zome
 * - DAO Zome
 */

import { AppWebsocket } from '@holochain/client';
import type {
  ActionHash,
  AgentPubKey,
  AppInfo,
  InstalledAppId,
  RoleName,
  ZomeName,
} from '@holochain/client';

import type {
  // Learning Zome types
  Course,
  LearnerProgress,
  LearningActivity,
  LearningZomeFunctions,

  // FL Zome types
  FlRound,
  FlUpdate,
  FlZomeFunctions,

  // Credential Zome types
  VerifiableCredential,
  CredentialZomeFunctions,

  // DAO Zome types
  Proposal,
  Vote,
  ProposalCategory,
  DaoZomeFunctions,

  // SRS Zome types
  Deck,
  ReviewCard,
  RecallQuality,
  ReviewSession,
  SrsZomeFunctions,

  // Gamification Zome types
  LearnerXp,
  XpActivityType,
  LearnerStreak,
  BadgeDefinition,
  EarnedBadge,
  Leaderboard,
  LeaderboardType,
  LeaderboardPeriod,
  LeaderboardScope,
  GamificationZomeFunctions,

  // Adaptive Zome types
  LearnerProfile,
  SkillMastery,
  Recommendation,
  LearningGoal,
  SessionAnalytics,
  LearnerContext,
  SmartRecommendation,
  SmartRecsInput,
  FlowStateAnalysis,
  AnalyzeFlowInput,
  OptimalLearningWindow,
  PaginatedMasteriesInput,
  PaginatedMasteriesResult,
  PerformanceMetrics,
  CollectMetricsInput,
  BenchmarkSummary,
  RetentionPrediction,
  RetentionPredictionInput,
  BatchRetentionResult,
  BatchRetentionInput,
  ReviewScheduleResult,
  ReviewScheduleInput,
  AdaptiveZomeFunctions,

  // Integration Zome types
  IntegrationZomeFunctions,

  // Record type
  HolochainRecord,
} from '../types/zomes';

/**
 * Configuration for Holochain client connection
 */
export interface HolochainClientConfig {
  /** WebSocket URL for app interface (default: ws://localhost:8888) */
  appWsUrl?: string;

  /** Installed app ID (default: mycelix-praxis) */
  appId?: InstalledAppId;

  /** Role name within the app (default: praxis) */
  roleName?: RoleName;

  /** Timeout for connection attempts in ms (default: 10000) */
  connectionTimeout?: number;

  /** Enable verbose logging (default: false) */
  verbose?: boolean;

  /** Automatically reconnect on disconnect (default: true) */
  autoReconnect?: boolean;

  /** Reconnection delay in ms (default: 3000) */
  reconnectDelay?: number;
}

/**
 * Default configuration values
 */
const DEFAULT_CONFIG: Required<HolochainClientConfig> = {
  appWsUrl: 'ws://localhost:8888',
  appId: '9999', // Matches hc sandbox -a 9999
  roleName: 'praxis',
  connectionTimeout: 10000,
  verbose: false,
  autoReconnect: true,
  reconnectDelay: 3000,
};

/**
 * Connection status
 */
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

/**
 * Real Holochain Client
 *
 * Connects to a Holochain conductor and provides typed zome call wrappers.
 */
export class HolochainClient {
  private config: Required<HolochainClientConfig>;
  private client: AppWebsocket | null = null;
  private appInfo: AppInfo | null = null;
  private status: ConnectionStatus = 'disconnected';
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  // Connection callbacks
  private onStatusChangeCallbacks: Array<(status: ConnectionStatus) => void> = [];
  private onErrorCallbacks: Array<(error: Error) => void> = [];

  // Zome function implementations
  public learning: LearningZomeFunctions;
  public fl: FlZomeFunctions;
  public credential: CredentialZomeFunctions;
  public dao: DaoZomeFunctions;
  public srs: SrsZomeFunctions;
  public gamification: GamificationZomeFunctions;
  public adaptive: AdaptiveZomeFunctions;
  public integration: IntegrationZomeFunctions;

  constructor(config: HolochainClientConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    // Initialize zome function implementations
    this.learning = this.createLearningZomeFunctions();
    this.fl = this.createFlZomeFunctions();
    this.credential = this.createCredentialZomeFunctions();
    this.dao = this.createDaoZomeFunctions();
    this.srs = this.createSrsZomeFunctions();
    this.gamification = this.createGamificationZomeFunctions();
    this.adaptive = this.createAdaptiveZomeFunctions();
    this.integration = this.createIntegrationZomeFunctions();
  }

  // =============================================================================
  // Connection Management
  // =============================================================================

  /**
   * Connect to the Holochain conductor
   */
  async connect(): Promise<void> {
    if (this.status === 'connected') {
      this.log('Already connected');
      return;
    }

    this.setStatus('connecting');

    try {
      // Connect to conductor
      this.client = await AppWebsocket.connect({
        url: new URL(this.config.appWsUrl),
      });

      this.log('Connected to conductor at', this.config.appWsUrl);

      // Get app info
      this.appInfo = await this.client.appInfo();

      if (!this.appInfo) {
        throw new Error(`App ${this.config.appId} not found`);
      }

      this.log('App info retrieved:', this.appInfo);

      this.setStatus('connected');

      // Note: WebSocket close handling may vary by client version
      // We use try-catch to ensure compatibility
    } catch (error) {
      this.setStatus('error');
      const err = error instanceof Error ? error : new Error(String(error));
      this.notifyError(err);
      throw err;
    }
  }

  /**
   * Disconnect from the conductor
   */
  async disconnect(): Promise<void> {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.client) {
      // Close the WebSocket connection
      try {
        await (this.client as any).close?.();
      } catch {
        // Some versions may not have close
      }
      this.client = null;
    }

    this.appInfo = null;
    this.setStatus('disconnected');
    this.log('Disconnected from conductor');
  }

  /**
   * Get current connection status
   */
  getStatus(): ConnectionStatus {
    return this.status;
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.status === 'connected';
  }

  /**
   * Register a callback for status changes
   */
  onStatusChange(callback: (status: ConnectionStatus) => void): () => void {
    this.onStatusChangeCallbacks.push(callback);
    return () => {
      this.onStatusChangeCallbacks = this.onStatusChangeCallbacks.filter((cb) => cb !== callback);
    };
  }

  /**
   * Register a callback for errors
   */
  onError(callback: (error: Error) => void): () => void {
    this.onErrorCallbacks.push(callback);
    return () => {
      this.onErrorCallbacks = this.onErrorCallbacks.filter((cb) => cb !== callback);
    };
  }

  // =============================================================================
  // Private Helper Methods
  // =============================================================================

  /**
   * Set connection status and notify listeners
   */
  private setStatus(status: ConnectionStatus): void {
    this.status = status;
    this.onStatusChangeCallbacks.forEach((cb) => cb(status));
  }

  /**
   * Notify error listeners
   */
  private notifyError(error: Error): void {
    this.onErrorCallbacks.forEach((cb) => cb(error));
  }

  /**
   * Log message (if verbose mode enabled)
   */
  private log(...args: unknown[]): void {
    if (this.config.verbose) {
      console.log('[HolochainClient]', ...args);
    }
  }

  /**
   * Call a zome function
   */
  private async callZome<T = unknown>(
    zomeName: ZomeName,
    fnName: string,
    payload?: unknown
  ): Promise<T> {
    if (!this.client || !this.appInfo) {
      throw new Error('Not connected to conductor');
    }

    this.log(`Calling zome: ${zomeName}.${fnName}`, payload);

    try {
      const result = await this.client.callZome({
        role_name: this.config.roleName,
        zome_name: zomeName,
        fn_name: fnName,
        payload: payload ?? null,
      });

      this.log(`Zome call succeeded: ${zomeName}.${fnName}`, result);
      return result as T;
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      this.log(`Zome call failed: ${zomeName}.${fnName}`, err);
      throw err;
    }
  }

  /**
   * Decode entry from Holochain Record
   * Handles different @holochain/client versions
   */
  private decodeEntry<T>(record: HolochainRecord): T | null {
    if (!record || !record.entry) {
      return null;
    }
    // Handle different entry formats across versions
    const entry = record.entry as any;
    if (entry.Present !== undefined) {
      return entry.Present as T;
    }
    if (entry.Entry !== undefined) {
      return entry.Entry as T;
    }
    // If entry is the value directly
    if (typeof entry === 'object' && !('Hidden' in entry) && !('NotApplicable' in entry)) {
      return entry as T;
    }
    return null;
  }

  // =============================================================================
  // Learning Zome Functions
  // =============================================================================

  private createLearningZomeFunctions(): LearningZomeFunctions {
    return {
      create_course: async (course: Course): Promise<ActionHash> => {
        return this.callZome<ActionHash>('learning_coordinator', 'create_course', course);
      },

      get_course: async (course_hash: ActionHash): Promise<Course | null> => {
        const record = await this.callZome<HolochainRecord | null>('learning_coordinator', 'get_course', course_hash);
        if (!record) return null;
        return this.decodeEntry<Course>(record);
      },

      list_courses: async (): Promise<Course[]> => {
        const records = await this.callZome<HolochainRecord[]>('learning_coordinator', 'list_courses', null);
        return records.map(r => this.decodeEntry<Course>(r)).filter((c): c is Course => c !== null);
      },

      enroll: async (course_action_hash: ActionHash): Promise<void> => {
        await this.callZome<void>('learning_coordinator', 'enroll', course_action_hash);
      },

      update_progress: async (progress: LearnerProgress): Promise<ActionHash> => {
        return this.callZome<ActionHash>('learning_coordinator', 'update_progress', progress);
      },

      get_progress: async (action_hash: ActionHash): Promise<LearnerProgress | null> => {
        const record = await this.callZome<HolochainRecord | null>('learning_coordinator', 'get_progress', action_hash);
        if (!record) return null;
        return this.decodeEntry<LearnerProgress>(record);
      },

      get_enrolled_courses: async (): Promise<Course[]> => {
        const records = await this.callZome<HolochainRecord[]>('learning_coordinator', 'get_enrolled_courses', null);
        return records.map(r => this.decodeEntry<Course>(r)).filter((c): c is Course => c !== null);
      },

      record_activity: async (activity: LearningActivity): Promise<ActionHash> => {
        return this.callZome<ActionHash>('learning_coordinator', 'record_activity', activity);
      },

      get_course_enrollments: async (course_action_hash: ActionHash): Promise<AgentPubKey[]> => {
        return this.callZome<AgentPubKey[]>('learning_coordinator', 'get_course_enrollments', course_action_hash);
      },
    };
  }

  // =============================================================================
  // FL Zome Functions
  // =============================================================================

  private createFlZomeFunctions(): FlZomeFunctions {
    return {
      create_fl_round: async (round: FlRound): Promise<ActionHash> => {
        return this.callZome<ActionHash>('fl_coordinator', 'create_fl_round', round);
      },

      get_fl_round: async (round_hash: ActionHash): Promise<FlRound | null> => {
        const record = await this.callZome<HolochainRecord | null>('fl_coordinator', 'get_fl_round', round_hash);
        if (!record) return null;
        return this.decodeEntry<FlRound>(record);
      },

      get_active_rounds: async (model_id: string): Promise<FlRound[]> => {
        const records = await this.callZome<HolochainRecord[]>('fl_coordinator', 'get_active_rounds', { model_id });
        return records.map(r => this.decodeEntry<FlRound>(r)).filter((r): r is FlRound => r !== null);
      },

      submit_update: async (update: FlUpdate): Promise<ActionHash> => {
        return this.callZome<ActionHash>('fl_coordinator', 'submit_update', update);
      },

      get_round_updates: async (round_id: string): Promise<FlUpdate[]> => {
        const records = await this.callZome<HolochainRecord[]>('fl_coordinator', 'get_round_updates', { round_id });
        return records.map(r => this.decodeEntry<FlUpdate>(r)).filter((u): u is FlUpdate => u !== null);
      },

      aggregate_round: async (round_id: string): Promise<ActionHash> => {
        return this.callZome<ActionHash>('fl_coordinator', 'aggregate_round', { round_id });
      },

      // Type assertion to bridge SDK Record to our HolochainRecord interface
      get_aggregated_model: (async (round_id: string) => {
        const result = await this.callZome('fl_coordinator', 'get_aggregated_model', { round_id });
        return result as unknown as HolochainRecord | null;
      }) as (round_id: string) => Promise<HolochainRecord | null>,
    };
  }

  // =============================================================================
  // Credential Zome Functions
  // =============================================================================

  private createCredentialZomeFunctions(): CredentialZomeFunctions {
    return {
      issue_credential: async (credential: VerifiableCredential): Promise<ActionHash> => {
        return this.callZome<ActionHash>('credential_coordinator', 'issue_credential', credential);
      },

      get_credential: async (credential_hash: ActionHash): Promise<VerifiableCredential | null> => {
        const record = await this.callZome<HolochainRecord | null>(
          'credential_coordinator',
          'get_credential',
          credential_hash
        );
        if (!record) return null;
        return this.decodeEntry<VerifiableCredential>(record);
      },

      get_learner_credentials: async (learner_did: string): Promise<VerifiableCredential[]> => {
        const records = await this.callZome<HolochainRecord[]>('credential_coordinator', 'get_learner_credentials', {
          learner_did,
        });
        return records.map(r => this.decodeEntry<VerifiableCredential>(r)).filter((c): c is VerifiableCredential => c !== null);
      },

      get_course_credentials: async (course_id: string): Promise<VerifiableCredential[]> => {
        const records = await this.callZome<HolochainRecord[]>('credential_coordinator', 'get_course_credentials', {
          course_id,
        });
        return records.map(r => this.decodeEntry<VerifiableCredential>(r)).filter((c): c is VerifiableCredential => c !== null);
      },

      get_issuer_credentials: async (issuer_did: string): Promise<VerifiableCredential[]> => {
        const records = await this.callZome<HolochainRecord[]>('credential_coordinator', 'get_issuer_credentials', {
          issuer_did,
        });
        return records.map(r => this.decodeEntry<VerifiableCredential>(r)).filter((c): c is VerifiableCredential => c !== null);
      },

      verify_credential: async (credential_hash: ActionHash): Promise<boolean> => {
        return this.callZome<boolean>('credential_coordinator', 'verify_credential', credential_hash);
      },

      revoke_credential: async (credential_hash: ActionHash): Promise<ActionHash> => {
        return this.callZome<ActionHash>('credential_coordinator', 'revoke_credential', credential_hash);
      },
    };
  }

  // =============================================================================
  // DAO Zome Functions
  // =============================================================================

  private createDaoZomeFunctions(): DaoZomeFunctions {
    return {
      create_proposal: async (proposal: Proposal): Promise<ActionHash> => {
        return this.callZome<ActionHash>('dao_coordinator', 'create_proposal', proposal);
      },

      get_proposal: async (proposal_hash: ActionHash): Promise<Proposal | null> => {
        const record = await this.callZome<HolochainRecord | null>('dao_coordinator', 'get_proposal', proposal_hash);
        if (!record) return null;
        return this.decodeEntry<Proposal>(record);
      },

      get_proposals_by_category: async (category: ProposalCategory): Promise<Proposal[]> => {
        const records = await this.callZome<HolochainRecord[]>('dao_coordinator', 'get_proposals_by_category', { category });
        return records.map(r => this.decodeEntry<Proposal>(r)).filter((p): p is Proposal => p !== null);
      },

      get_agent_proposals: async (agent_did: string): Promise<Proposal[]> => {
        const records = await this.callZome<HolochainRecord[]>('dao_coordinator', 'get_agent_proposals', { agent_did });
        return records.map(r => this.decodeEntry<Proposal>(r)).filter((p): p is Proposal => p !== null);
      },

      list_proposals: async (): Promise<Proposal[]> => {
        const records = await this.callZome<HolochainRecord[]>('dao_coordinator', 'list_proposals', null);
        return records.map(r => this.decodeEntry<Proposal>(r)).filter((p): p is Proposal => p !== null);
      },

      cast_vote: async (vote: Vote): Promise<ActionHash> => {
        return this.callZome<ActionHash>('dao_coordinator', 'cast_vote', vote);
      },

      get_agent_votes: async (agent_did: string): Promise<Vote[]> => {
        const records = await this.callZome<HolochainRecord[]>('dao_coordinator', 'get_agent_votes', { agent_did });
        return records.map(r => this.decodeEntry<Vote>(r)).filter((v): v is Vote => v !== null);
      },
    };
  }

  // =============================================================================
  // SRS Zome Functions
  // =============================================================================

  private createSrsZomeFunctions(): SrsZomeFunctions {
    return {
      create_deck: async (input: { name: string; description: string }): Promise<ActionHash> => {
        return this.callZome<ActionHash>('srs_coordinator', 'create_deck', input);
      },

      get_deck: async (deck_hash: ActionHash): Promise<Deck | null> => {
        const record = await this.callZome<HolochainRecord | null>('srs_coordinator', 'get_deck', deck_hash);
        if (!record) return null;
        return this.decodeEntry<Deck>(record);
      },

      get_my_decks: async (): Promise<Deck[]> => {
        const records = await this.callZome<HolochainRecord[]>('srs_coordinator', 'get_my_decks', null);
        return records.map(r => this.decodeEntry<Deck>(r)).filter((d): d is Deck => d !== null);
      },

      create_card: async (input: { deck_hash: ActionHash; front: string; back: string; tags: string[] }): Promise<ActionHash> => {
        return this.callZome<ActionHash>('srs_coordinator', 'create_card', input);
      },

      get_card: async (card_hash: ActionHash): Promise<ReviewCard | null> => {
        const record = await this.callZome<HolochainRecord | null>('srs_coordinator', 'get_card', card_hash);
        if (!record) return null;
        return this.decodeEntry<ReviewCard>(record);
      },

      get_due_cards: async (input: { deck_hash?: ActionHash; limit: number }): Promise<ReviewCard[]> => {
        const records = await this.callZome<HolochainRecord[]>('srs_coordinator', 'get_due_cards', input.limit);
        return records.map(r => this.decodeEntry<ReviewCard>(r)).filter((c): c is ReviewCard => c !== null);
      },

      review_card: async (input: { card_hash: ActionHash; quality: RecallQuality }): Promise<ReviewCard> => {
        const record = await this.callZome<HolochainRecord>('srs_coordinator', 'submit_review', input);
        return this.decodeEntry<ReviewCard>(record)!;
      },

      start_session: async (deck_hash?: ActionHash): Promise<ActionHash> => {
        const record = await this.callZome<HolochainRecord>('srs_coordinator', 'start_session', deck_hash ?? null);
        return record.signed_action.hashed.hash;
      },

      end_session: async (session_hash: ActionHash): Promise<ReviewSession> => {
        const record = await this.callZome<HolochainRecord>('srs_coordinator', 'end_session', { session_hash });
        return this.decodeEntry<ReviewSession>(record)!;
      },

      get_stats: async (): Promise<{ cards_total: number; cards_due: number; reviews_today: number; accuracy_permille: number }> => {
        // Call the zome and get stats records
        const records = await this.callZome<HolochainRecord[]>('srs_coordinator', 'get_stats', { days: 1 });

        // Parse DailyStats from records
        // DailyStats structure: { learner, date, new_cards, reviews, relearns, study_time_seconds, retention_permille, streak_days }
        if (!records || records.length === 0) {
          return {
            cards_total: 0,
            cards_due: 0,
            reviews_today: 0,
            accuracy_permille: 0,
          };
        }

        // Decode the most recent day's stats
        interface DailyStats {
          learner: Uint8Array;
          date: number;
          new_cards: number;
          reviews: number;
          relearns: number;
          study_time_seconds: number;
          retention_permille: number;
          streak_days: number;
        }

        const dailyStats = this.decodeEntry<DailyStats>(records[0]);
        if (!dailyStats) {
          return {
            cards_total: 0,
            cards_due: 0,
            reviews_today: 0,
            accuracy_permille: 0,
          };
        }

        // Aggregate stats from records
        let totalCards = 0;
        let totalReviews = 0;
        let totalRetention = 0;
        let recordCount = 0;

        for (const record of records) {
          const stats = this.decodeEntry<DailyStats>(record);
          if (stats) {
            totalCards += stats.new_cards;
            totalReviews += stats.reviews;
            totalRetention += stats.retention_permille;
            recordCount++;
          }
        }

        // Calculate averages and return aggregated stats
        return {
          cards_total: totalCards,
          cards_due: dailyStats.new_cards + dailyStats.relearns, // Approximate due cards
          reviews_today: dailyStats.reviews,
          accuracy_permille: recordCount > 0 ? Math.round(totalRetention / recordCount) : 0,
        };
      },
    };
  }

  // =============================================================================
  // Gamification Zome Functions
  // =============================================================================

  private createGamificationZomeFunctions(): GamificationZomeFunctions {
    return {
      get_my_xp: async (): Promise<LearnerXp | null> => {
        const record = await this.callZome<HolochainRecord | null>('gamification_coordinator', 'get_or_create_xp', null);
        if (!record) return null;
        return this.decodeEntry<LearnerXp>(record);
      },

      award_xp: async (input: { amount: number; activity_type: XpActivityType; source_hash: ActionHash; description: string }): Promise<ActionHash> => {
        return this.callZome<ActionHash>('gamification_coordinator', 'award_xp', input);
      },

      get_my_streak: async (): Promise<LearnerStreak | null> => {
        const record = await this.callZome<HolochainRecord | null>('gamification_coordinator', 'get_or_create_streak', null);
        if (!record) return null;
        return this.decodeEntry<LearnerStreak>(record);
      },

      record_daily_activity: async (): Promise<LearnerStreak> => {
        const record = await this.callZome<HolochainRecord>('gamification_coordinator', 'record_activity', null);
        return this.decodeEntry<LearnerStreak>(record)!;
      },

      use_streak_freeze: async (): Promise<LearnerStreak> => {
        const record = await this.callZome<HolochainRecord>('gamification_coordinator', 'use_streak_freeze', null);
        return this.decodeEntry<LearnerStreak>(record)!;
      },

      get_badge_definitions: async (): Promise<BadgeDefinition[]> => {
        const records = await this.callZome<HolochainRecord[]>('gamification_coordinator', 'get_all_badges', null);
        return records.map(r => this.decodeEntry<BadgeDefinition>(r)).filter((b): b is BadgeDefinition => b !== null);
      },

      get_my_badges: async (): Promise<EarnedBadge[]> => {
        const records = await this.callZome<HolochainRecord[]>('gamification_coordinator', 'get_my_badges', null);
        return records.map(r => this.decodeEntry<EarnedBadge>(r)).filter((b): b is EarnedBadge => b !== null);
      },

      award_badge: async (input: { badge_id: string; source_hash?: ActionHash }): Promise<ActionHash> => {
        return this.callZome<ActionHash>('gamification_coordinator', 'award_badge', input);
      },

      get_leaderboard: async (input: { leaderboard_type: LeaderboardType; period: LeaderboardPeriod; scope: LeaderboardScope; limit: number }): Promise<Leaderboard> => {
        return this.callZome<Leaderboard>('gamification_coordinator', 'get_leaderboard', input);
      },
    };
  }

  // =============================================================================
  // Adaptive Zome Functions
  // =============================================================================

  private createAdaptiveZomeFunctions(): AdaptiveZomeFunctions {
    return {
      // Profile Management
      get_my_profile: async (): Promise<LearnerProfile | null> => {
        const record = await this.callZome<HolochainRecord | null>('adaptive_coordinator', 'get_or_create_profile', null);
        if (!record) return null;
        return this.decodeEntry<LearnerProfile>(record);
      },

      update_profile: async (profile: Partial<LearnerProfile>): Promise<ActionHash> => {
        return this.callZome<ActionHash>('adaptive_coordinator', 'update_profile', profile);
      },

      submit_style_assessment: async (scores: { visual: number; auditory: number; reading: number; kinesthetic: number }): Promise<LearnerProfile> => {
        const record = await this.callZome<HolochainRecord>('adaptive_coordinator', 'submit_style_assessment', scores);
        return this.decodeEntry<LearnerProfile>(record)!;
      },

      // Skill Mastery
      get_skill_mastery: async (skill_hash: ActionHash): Promise<SkillMastery | null> => {
        const record = await this.callZome<HolochainRecord | null>('adaptive_coordinator', 'get_skill_mastery', skill_hash);
        if (!record) return null;
        return this.decodeEntry<SkillMastery>(record);
      },

      get_my_skills: async (): Promise<SkillMastery[]> => {
        const records = await this.callZome<HolochainRecord[]>('adaptive_coordinator', 'get_my_skills', null);
        return records.map(r => this.decodeEntry<SkillMastery>(r)).filter((s): s is SkillMastery => s !== null);
      },

      update_skill_mastery: async (input: { skill_hash: ActionHash; correct: boolean }): Promise<SkillMastery> => {
        const record = await this.callZome<HolochainRecord>('adaptive_coordinator', 'record_attempt', input);
        return this.decodeEntry<SkillMastery>(record)!;
      },

      // Recommendations
      get_recommendations: async (limit: number): Promise<Recommendation[]> => {
        const records = await this.callZome<HolochainRecord[]>('adaptive_coordinator', 'generate_recommendations', { limit });
        return records.map(r => this.decodeEntry<Recommendation>(r)).filter((r): r is Recommendation => r !== null);
      },

      act_on_recommendation: async (recommendation_hash: ActionHash): Promise<Recommendation> => {
        const record = await this.callZome<HolochainRecord>('adaptive_coordinator', 'act_on_recommendation', recommendation_hash);
        return this.decodeEntry<Recommendation>(record)!;
      },

      generate_smart_recommendations_v2: async (input: SmartRecsInput): Promise<SmartRecommendation[]> => {
        return this.callZome<SmartRecommendation[]>('adaptive_coordinator', 'generate_smart_recommendations', input);
      },

      // Goals
      get_my_goals: async (): Promise<LearningGoal[]> => {
        const records = await this.callZome<HolochainRecord[]>('adaptive_coordinator', 'get_my_goals', null);
        return records.map(r => this.decodeEntry<LearningGoal>(r)).filter((g): g is LearningGoal => g !== null);
      },

      create_goal: async (goal: Omit<LearningGoal, 'learner' | 'current_value' | 'is_completed' | 'completed_at' | 'created_at'>): Promise<ActionHash> => {
        return this.callZome<ActionHash>('adaptive_coordinator', 'create_goal', goal);
      },

      update_goal_progress: async (goal_id: string, progress: number): Promise<LearningGoal> => {
        const record = await this.callZome<HolochainRecord>('adaptive_coordinator', 'update_goal_progress', { goal_id, progress });
        return this.decodeEntry<LearningGoal>(record)!;
      },

      // Session Management
      start_session: async (): Promise<ActionHash> => {
        const record = await this.callZome<HolochainRecord>('adaptive_coordinator', 'start_session', null);
        return record.signed_action.hashed.hash;
      },

      end_session: async (session_hash: ActionHash): Promise<SessionAnalytics> => {
        const record = await this.callZome<HolochainRecord>('adaptive_coordinator', 'end_session', session_hash);
        return this.decodeEntry<SessionAnalytics>(record)!;
      },

      get_session_analytics: async (session_hash: ActionHash): Promise<SessionAnalytics | null> => {
        const record = await this.callZome<HolochainRecord | null>('adaptive_coordinator', 'get_session_analytics', session_hash);
        if (!record) return null;
        return this.decodeEntry<SessionAnalytics>(record);
      },

      // Flow State Analysis
      analyze_flow_state: async (input: AnalyzeFlowInput): Promise<FlowStateAnalysis> => {
        return this.callZome<FlowStateAnalysis>('adaptive_coordinator', 'analyze_flow_state', input);
      },

      get_optimal_learning_window: async (): Promise<OptimalLearningWindow> => {
        return this.callZome<OptimalLearningWindow>('adaptive_coordinator', 'get_optimal_learning_window', null);
      },

      // Learner Context
      get_learner_context: async (): Promise<LearnerContext> => {
        return this.callZome<LearnerContext>('adaptive_coordinator', 'get_learner_context', null);
      },

      get_masteries_paginated: async (input: PaginatedMasteriesInput): Promise<PaginatedMasteriesResult> => {
        return this.callZome<PaginatedMasteriesResult>('adaptive_coordinator', 'get_masteries_paginated', input);
      },

      // Performance Metrics
      collect_performance_metrics: async (input?: CollectMetricsInput): Promise<PerformanceMetrics> => {
        return this.callZome<PerformanceMetrics>('adaptive_coordinator', 'collect_performance_metrics', input ?? {});
      },

      run_benchmarks: async (): Promise<BenchmarkSummary> => {
        return this.callZome<BenchmarkSummary>('adaptive_coordinator', 'run_benchmarks', null);
      },

      // Retention Prediction
      predict_skill_retention: async (input: RetentionPredictionInput): Promise<RetentionPrediction> => {
        return this.callZome<RetentionPrediction>('adaptive_coordinator', 'predict_skill_retention', input);
      },

      predict_retention_batch: async (input: BatchRetentionInput): Promise<BatchRetentionResult> => {
        return this.callZome<BatchRetentionResult>('adaptive_coordinator', 'predict_retention_batch', input);
      },

      get_optimal_review_schedule: async (input: ReviewScheduleInput): Promise<ReviewScheduleResult> => {
        return this.callZome<ReviewScheduleResult>('adaptive_coordinator', 'get_optimal_review_schedule', input);
      },

      // Metacognition - stub implementations
      calculate_calibration: async () => { throw new Error('Not implemented'); },
      record_confidence_judgment: async () => { throw new Error('Not implemented'); },
      get_calibration_history: async () => { throw new Error('Not implemented'); },
    } as unknown as AdaptiveZomeFunctions;
  }

  // =============================================================================
  // Integration Zome Functions
  // =============================================================================

  private createIntegrationZomeFunctions(): IntegrationZomeFunctions {
    return {
      // Unified Dashboard
      get_unified_dashboard: async () => {
        return this.callZome('integration_coordinator', 'get_unified_dashboard', null);
      },

      // Learning Events
      record_learning_event: async (input: unknown) => {
        return this.callZome<ActionHash>('integration_coordinator', 'record_learning_event', input);
      },

      // Smart Session Planning
      plan_smart_session: async (input: unknown) => {
        return this.callZome('integration_coordinator', 'plan_smart_session', input);
      },

      // Cross-zome orchestration
      complete_activity: async (input: unknown) => {
        return this.callZome('integration_coordinator', 'complete_activity', input);
      },

      get_learning_journey: async () => {
        return this.callZome('integration_coordinator', 'get_learning_journey', null);
      },
    } as unknown as IntegrationZomeFunctions;
  }
}

/**
 * Create and configure a Holochain client
 */
export function createHolochainClient(config?: HolochainClientConfig): HolochainClient {
  return new HolochainClient(config);
}

/**
 * Default export
 */
export default HolochainClient;
