/**
 * LUCID Reasoning Zome Client
 *
 * Contradiction detection, coherence reports, and logical inference.
 *
 * @module @mycelix/sdk/clients/lucid/reasoning
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  RecordContradictionInput,
  RecordContradictionAnalyzedInput,
  CreateReportInput,
  RecordInferenceInput,
} from './types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface ReasoningClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: ReasoningClientConfig = {
  roleName: 'lucid',
};

/**
 * Client for the LUCID Reasoning zome
 *
 * Handles contradiction recording, coherence reports, and inference chains.
 */
export class ReasoningClient extends ZomeClient {
  protected readonly zomeName = 'lucid_reasoning';

  constructor(client: AppClient, config: ReasoningClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Contradictions
  // ============================================================================

  /** Record a detected contradiction between two thoughts */
  async recordContradiction(input: RecordContradictionInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('record_contradiction', input);
  }

  /** Record a contradiction with Symthaea semantic similarity analysis */
  async recordContradictionAnalyzed(input: RecordContradictionAnalyzedInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('record_contradiction_analyzed', input);
  }

  /** Get contradictions involving a specific thought */
  async getThoughtContradictions(thoughtId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_thought_contradictions', thoughtId);
  }

  // ============================================================================
  // Coherence Reports
  // ============================================================================

  /** Create a coherence report analyzing a set of thoughts */
  async createCoherenceReport(input: CreateReportInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('create_coherence_report', input);
  }

  // ============================================================================
  // Inferences
  // ============================================================================

  /** Record a logical inference from premises to conclusion */
  async recordInference(input: RecordInferenceInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('record_inference', input);
  }

  /** Get all my recorded inferences */
  async getMyInferences(): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_my_inferences', null);
  }
}
