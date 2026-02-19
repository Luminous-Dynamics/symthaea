/**
 * Diagnostics Zome Client
 *
 * Handles diagnostic runs, privacy preferences, and cognitive updates.
 *
 * @module @mycelix/sdk/clients/support/diagnostics
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  DiagnosticResultInput,
  PrivacyPreferenceInput,
  CognitiveUpdateInput,
  SupportCategory,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface DiagnosticsClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: DiagnosticsClientConfig = {
  roleName: 'commons',
};

/**
 * Client for Support Diagnostics operations
 */
export class DiagnosticsClient extends ZomeClient {
  protected readonly zomeName = 'support_diagnostics';

  constructor(client: AppClient, config: DiagnosticsClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Diagnostics
  // ============================================================================

  async runDiagnostic(input: DiagnosticResultInput): Promise<Uint8Array> {
    const record = await this.callZomeOnce<HolochainRecord>('run_diagnostic', {
      ticket_hash: input.ticketHash,
      diagnostic_type: input.diagnosticType,
      findings: input.findings,
      severity: input.severity,
      recommendations: input.recommendations,
      agent: input.agent,
      scrubbed: input.scrubbed,
      created_at: input.createdAt,
    });
    return this.getActionHash(record);
  }

  async getDiagnostic(diagnosticHash: ActionHash): Promise<any | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_diagnostic', diagnosticHash);
    if (!record) return null;
    return this.mapDiagnostic(record);
  }

  async listDiagnostics(ticketHash?: ActionHash): Promise<any[]> {
    const records = await this.callZome<HolochainRecord[]>('list_diagnostics', ticketHash ?? null);
    return records.map(r => this.mapDiagnostic(r));
  }

  // ============================================================================
  // Privacy
  // ============================================================================

  async setPrivacyPreference(input: PrivacyPreferenceInput): Promise<Uint8Array> {
    const record = await this.callZomeOnce<HolochainRecord>('set_privacy_preference', {
      agent: input.agent,
      sharing_tier: input.sharingTier,
      allowed_categories: input.allowedCategories,
      share_system_info: input.shareSystemInfo,
      share_resolution_patterns: input.shareResolutionPatterns,
      share_cognitive_updates: input.shareCognitiveUpdates,
      updated_at: input.updatedAt,
    });
    return this.getActionHash(record);
  }

  async getPrivacyPreference(agentPubKey: Uint8Array): Promise<any | null> {
    return this.callZomeOrNull<any>('get_privacy_preference', agentPubKey);
  }

  async getShareableDiagnostics(agentPubKey: Uint8Array): Promise<any[]> {
    return this.callZome<any[]>('get_shareable_diagnostics', agentPubKey);
  }

  // ============================================================================
  // Cognitive Updates
  // ============================================================================

  async publishCognitiveUpdate(input: CognitiveUpdateInput): Promise<Uint8Array> {
    const record = await this.callZomeOnce<HolochainRecord>('publish_cognitive_update', {
      category: input.category,
      encoding: input.encoding,
      phi: input.phi,
      resolution_pattern: input.resolutionPattern,
      source_agent: input.sourceAgent,
      created_at: input.createdAt,
    });
    return this.getActionHash(record);
  }

  async getCognitiveUpdatesByCategory(category: SupportCategory): Promise<any[]> {
    return this.callZome<any[]>('get_cognitive_updates_by_category', category);
  }

  async absorbCognitiveUpdate(updateHash: ActionHash): Promise<void> {
    await this.callZomeOnce<void>('absorb_cognitive_update', updateHash);
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapDiagnostic(record: HolochainRecord): any {
    const entry = this.extractEntry<any>(record);
    return {
      actionHash: record.signed_action.hashed.hash,
      ticketHash: entry.ticket_hash,
      diagnosticType: entry.diagnostic_type,
      findings: entry.findings,
      severity: entry.severity,
      recommendations: entry.recommendations,
      agent: entry.agent,
      scrubbed: entry.scrubbed,
      createdAt: entry.created_at,
    };
  }
}
