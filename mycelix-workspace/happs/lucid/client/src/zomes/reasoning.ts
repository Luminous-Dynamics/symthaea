// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Reasoning Zome Client
 *
 * Coherence checking, contradiction detection, and inference.
 */

import type { AppClient, Record as HolochainRecord } from '@holochain/client';
import type {
  Contradiction,
  RecordContradictionInput,
  CoherenceReport,
  CreateReportInput,
  Inference,
  RecordInferenceInput,
} from '../types';
import { decodeRecord, decodeRecords } from '../utils';

export class ReasoningZomeClient {
  constructor(
    private client: AppClient,
    private roleName: string = 'lucid',
    private zomeName: string = 'reasoning'
  ) {}

  private async callZome<T>(fnName: string, payload?: unknown): Promise<T> {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null,
    });
    return result as T;
  }

  // ============================================================================
  // CONTRADICTION OPERATIONS
  // ============================================================================

  /** Record a detected contradiction */
  async recordContradiction(input: RecordContradictionInput): Promise<Contradiction> {
    const record = await this.callZome<HolochainRecord>('record_contradiction', input);
    return decodeRecord<Contradiction>(record);
  }

  /** Get contradictions for a thought */
  async getThoughtContradictions(thoughtId: string): Promise<Contradiction[]> {
    const records = await this.callZome<HolochainRecord[]>('get_thought_contradictions', thoughtId);
    return decodeRecords<Contradiction>(records);
  }

  // ============================================================================
  // COHERENCE OPERATIONS
  // ============================================================================

  /** Create a coherence report */
  async createCoherenceReport(input: CreateReportInput): Promise<CoherenceReport> {
    const record = await this.callZome<HolochainRecord>('create_coherence_report', input);
    return decodeRecord<CoherenceReport>(record);
  }

  // ============================================================================
  // INFERENCE OPERATIONS
  // ============================================================================

  /** Record an inference */
  async recordInference(input: RecordInferenceInput): Promise<Inference> {
    const record = await this.callZome<HolochainRecord>('record_inference', input);
    return decodeRecord<Inference>(record);
  }

  /** Get all inferences */
  async getMyInferences(): Promise<Inference[]> {
    const records = await this.callZome<HolochainRecord[]>('get_my_inferences', null);
    return decodeRecords<Inference>(records);
  }
}
