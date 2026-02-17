/**
 * Purity Zome Client
 *
 * Handles water quality readings, potability checks, and alerts.
 *
 * @module @mycelix/sdk/clients/water/purity
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  QualityReading,
  SubmitQualityReadingInput,
  WaterAlert,
  RaiseAlertInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface PurityClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: PurityClientConfig = {
  roleName: 'commons',
};

/**
 * Client for Water Purity operations
 */
export class PurityClient extends ZomeClient {
  protected readonly zomeName = 'water_purity';

  constructor(client: AppClient, config: PurityClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  async submitReading(input: SubmitQualityReadingInput): Promise<QualityReading> {
    const record = await this.callZomeOnce<HolochainRecord>('submit_reading', {
      source_id: input.sourceId,
      ph: input.ph,
      turbidity: input.turbidity,
      tds: input.tds,
      dissolved_oxygen: input.dissolvedOxygen,
      e_coli_count: input.eColiCount,
      temperature: input.temperature,
      contaminants: input.contaminants ?? {},
      notes: input.notes,
    });
    return this.mapReading(record);
  }

  async getReadingsForSource(sourceId: ActionHash, limit?: number): Promise<QualityReading[]> {
    const records = await this.callZome<HolochainRecord[]>('get_readings_for_source', {
      source_id: sourceId,
      limit: limit ?? 50,
    });
    return records.map(r => this.mapReading(r));
  }

  async checkPotability(sourceId: ActionHash): Promise<{ potable: boolean; issues: string[] }> {
    return this.callZome('check_potability', sourceId);
  }

  async raiseAlert(input: RaiseAlertInput): Promise<WaterAlert> {
    const record = await this.callZomeOnce<HolochainRecord>('raise_alert', {
      source_id: input.sourceId,
      severity: input.severity,
      description: input.description,
      contaminant: input.contaminant,
      reading_id: input.readingId,
    });
    return this.mapAlert(record);
  }

  async getActiveAlerts(sourceId?: ActionHash): Promise<WaterAlert[]> {
    const records = await this.callZome<HolochainRecord[]>('get_active_alerts', sourceId ?? null);
    return records.map(r => this.mapAlert(r));
  }

  async resolveAlert(alertId: ActionHash, resolution: string): Promise<WaterAlert> {
    const record = await this.callZomeOnce<HolochainRecord>('resolve_alert', {
      alert_id: alertId,
      resolution,
    });
    return this.mapAlert(record);
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapReading(record: HolochainRecord): QualityReading {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      sourceId: entry.source_id,
      testerDid: entry.tester_did,
      ph: entry.ph,
      turbidity: entry.turbidity,
      tds: entry.tds,
      dissolvedOxygen: entry.dissolved_oxygen,
      eColiCount: entry.e_coli_count,
      temperature: entry.temperature,
      contaminants: entry.contaminants,
      potable: entry.potable,
      notes: entry.notes,
      testedAt: entry.tested_at,
    };
  }

  private mapAlert(record: HolochainRecord): WaterAlert {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      sourceId: entry.source_id,
      raisedByDid: entry.raised_by_did,
      severity: entry.severity,
      description: entry.description,
      contaminant: entry.contaminant,
      readingId: entry.reading_id,
      status: entry.status,
      raisedAt: entry.raised_at,
      resolvedAt: entry.resolved_at,
    };
  }
}
