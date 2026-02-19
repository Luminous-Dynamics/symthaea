/**
 * LUCID Bridge Zome Client
 *
 * Federation, reputation caching, coherence analysis storage,
 * and Phi streaming from Symthaea into the DHT.
 *
 * @module @mycelix/sdk/clients/lucid/bridge
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  RecordFederationInput,
  CacheReputationInput,
  StoreCoherenceInput,
  GetCoherenceCandidatesInput,
  CoherenceCandidate,
  CoherenceCheckResult,
  StreamPhiInput,
  PhiStreamEntry,
} from './types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface BridgeClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: BridgeClientConfig = {
  roleName: 'lucid',
};

/**
 * Client for the LUCID Bridge zome
 *
 * Handles Symthaea-DHT bridging: Phi streaming, coherence analysis,
 * federation records, and reputation caching.
 */
export class LucidBridgeClient extends ZomeClient {
  protected readonly zomeName = 'lucid_bridge';

  constructor(client: AppClient, config: BridgeClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Federation
  // ============================================================================

  /** Record federation of a thought to an external hApp */
  async recordFederation(input: RecordFederationInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('record_federation', input);
  }

  /** Get federation records for a thought */
  async getThoughtFederations(thoughtId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_thought_federations', thoughtId);
  }

  // ============================================================================
  // Reputation
  // ============================================================================

  /** Cache an external reputation score */
  async cacheReputation(input: CacheReputationInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('cache_reputation', input);
  }

  /** Get cached reputation for an agent */
  async getCachedReputation(agent: Uint8Array): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_cached_reputation', agent);
  }

  // ============================================================================
  // Coherence Analysis
  // ============================================================================

  /** Store a coherence analysis result (from Symthaea via Tauri) */
  async storeCoherenceAnalysis(input: StoreCoherenceInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('store_coherence_analysis', input);
  }

  /** Get coherence analyses involving a specific thought */
  async getThoughtCoherenceAnalyses(thoughtId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_thought_coherence_analyses', thoughtId);
  }

  /** Get a specific coherence analysis by ID */
  async getCoherenceAnalysis(analysisId: string): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_coherence_analysis', analysisId);
  }

  /** Get coherence candidates for Symthaea analysis */
  async getCoherenceCandidates(input: GetCoherenceCandidatesInput): Promise<CoherenceCandidate[]> {
    return this.callZome<CoherenceCandidate[]>('get_coherence_candidates', input);
  }

  /** Get the latest coherence status for a thought */
  async getThoughtCoherenceStatus(thoughtId: string): Promise<CoherenceCheckResult | null> {
    return this.callZomeOrNull<CoherenceCheckResult>('get_thought_coherence_status', thoughtId);
  }

  // ============================================================================
  // Phi Streaming (Symthaea -> DHT)
  // ============================================================================

  /** Stream a Phi score from Symthaea into the DHT */
  async streamPhiScore(input: StreamPhiInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('stream_phi_score', input);
  }

  /** Get the latest Phi score for an agent */
  async getLatestPhi(agent: Uint8Array): Promise<PhiStreamEntry | null> {
    return this.callZomeOrNull<PhiStreamEntry>('get_latest_phi', agent);
  }

  /** Get latest Phi scores for all agents (map of agent_b64 -> phi) */
  async getAllLatestPhiScores(): Promise<Record<string, number>> {
    return this.callZome<Record<string, number>>('get_all_latest_phi_scores', null);
  }

  /** Get Phi history for an agent (up to 100 entries, sorted by time) */
  async getPhiHistory(agent: Uint8Array): Promise<PhiStreamEntry[]> {
    return this.callZome<PhiStreamEntry[]>('get_phi_history', agent);
  }

  /** Get all thoughts with detected contradictions */
  async getContradictingThoughts(): Promise<CoherenceCheckResult[]> {
    return this.callZome<CoherenceCheckResult[]>('get_contradicting_thoughts', null);
  }
}
