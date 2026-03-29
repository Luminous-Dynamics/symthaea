// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Prints Client
 *
 * Handles print job lifecycle with PoGF (Proof of Grounded Fabrication)
 * and Cincinnati Algorithm integration:
 * - Job creation and management
 * - PoGF scoring and MYCELIUM rewards
 * - Cincinnati quality monitoring
 * - Print records and statistics
 */

import type { AppClient, ActionHash, Record } from '@holochain/client';
import type {
  PrintResult,
  PrintIssue,
  AnomalyEvent,
  CreatePrintJobInput,
  RecordPrintInput,
} from '../types';

export interface PaginationInput {
  offset: number;
  limit: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}

export interface UpdateProgressInput {
  job_hash: ActionHash;
  progress_percent: number;
  current_layer?: number;
  material_used_grams?: number;
}

export interface CompletePrintInput {
  job_hash: ActionHash;
  result: PrintResult;
}

export interface CancelPrintInput {
  job_hash: ActionHash;
  reason: string;
}

export interface AddPhotosInput {
  record_hash: ActionHash;
  photos: string[];
}

export interface StartCincinnatiInput {
  job_hash: ActionHash;
  total_layers: number;
  sampling_rate_hz: number;
}

export interface ReportAnomalyInput {
  session_id: string;
  anomaly: AnomalyEvent;
}

export interface UpdateCincinnatiInput {
  session_hash: ActionHash;
  current_layer: number;
  health_score: number;
  anomaly_count: number;
}

export interface DesignPrintStats {
  design_hash: ActionHash;
  total_prints: number;
  successful_prints: number;
  failed_prints: number;
  average_quality: number;
  average_pog_score: number;
  total_mycelium_earned: number;
  common_issues: Array<[PrintIssue, number]>;
}

export class PrintsClient {
  constructor(
    private client: AppClient,
    private roleName: string,
    private zomeName: string = 'prints_coordinator'
  ) {}

  // =========================================================================
  // JOB MANAGEMENT
  // =========================================================================

  /**
   * Create a new print job
   */
  async createJob(input: CreatePrintJobInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'create_print_job',
      payload: input,
    });
  }

  /**
   * Accept a print job (as printer operator)
   */
  async acceptJob(jobHash: ActionHash): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'accept_print_job',
      payload: jobHash,
    });
  }

  /**
   * Start printing
   */
  async startPrint(jobHash: ActionHash): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'start_print',
      payload: jobHash,
    });
  }

  /**
   * Update print progress
   */
  async updateProgress(input: UpdateProgressInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'update_print_progress',
      payload: input,
    });
  }

  /**
   * Complete print with result
   */
  async completePrint(input: CompletePrintInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'complete_print',
      payload: input,
    });
  }

  /**
   * Cancel a print job
   */
  async cancelPrint(input: CancelPrintInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'cancel_print',
      payload: input,
    });
  }

  // =========================================================================
  // PRINT RECORDS (WITH PoGF)
  // =========================================================================

  /**
   * Record print result with PoGF scoring
   *
   * PoGF Score = (E_renewable * 0.3) + (M_circular * 0.3) + (Q_verified * 0.2) + (L_local * 0.2)
   */
  async recordResult(input: RecordPrintInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'record_print_result',
      payload: input,
    });
  }

  /**
   * Get print record
   */
  async getRecord(hash: ActionHash): Promise<Record | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_print_record',
      payload: hash,
    });
  }

  /**
   * Add photos to a print record
   */
  async addPhotos(input: AddPhotosInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'add_print_photos',
      payload: input,
    });
  }

  // =========================================================================
  // CINCINNATI ALGORITHM MONITORING
  // =========================================================================

  /**
   * Start Cincinnati monitoring session for a print job
   */
  async startCincinnatiMonitoring(input: StartCincinnatiInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'start_cincinnati_monitoring',
      payload: input,
    });
  }

  /**
   * Report a Cincinnati anomaly event
   */
  async reportCincinnatiAnomaly(input: ReportAnomalyInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'report_cincinnati_anomaly',
      payload: input,
    });
  }

  /**
   * Update Cincinnati monitoring session state
   */
  async updateCincinnatiSession(input: UpdateCincinnatiInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'update_cincinnati_session',
      payload: input,
    });
  }

  // =========================================================================
  // DISCOVERY & STATISTICS
  // =========================================================================

  /**
   * Get my print jobs
   */
  async getMyJobs(pagination?: PaginationInput): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_my_print_jobs',
      payload: { pagination: pagination ?? null },
    });
  }

  /**
   * Get jobs for a specific printer
   */
  async getPrinterJobs(
    printerHash: ActionHash,
    pagination?: PaginationInput
  ): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_printer_jobs',
      payload: { hash: printerHash, pagination: pagination ?? null },
    });
  }

  /**
   * Get all prints of a specific design
   */
  async getDesignPrints(
    designHash: ActionHash,
    pagination?: PaginationInput
  ): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design_prints',
      payload: { hash: designHash, pagination: pagination ?? null },
    });
  }

  /**
   * Get statistics for a design
   */
  async getStatistics(designHash: ActionHash): Promise<DesignPrintStats> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_print_statistics',
      payload: designHash,
    });
  }

  /**
   * Calculate PoGF score (client-side helper)
   *
   * @param energyRenewableFraction - Percentage of renewable energy used (0-1)
   * @param materialCircularity - Material circularity score (0-1)
   * @param qualityVerified - Quality verification score (0-1)
   * @param localParticipation - Local economy participation (0-1)
   * @returns PoGF score (0-1)
   */
  calculatePogScore(
    energyRenewableFraction: number,
    materialCircularity: number,
    qualityVerified: number,
    localParticipation: number
  ): number {
    const e = Math.max(0, Math.min(1, energyRenewableFraction));
    const m = Math.max(0, Math.min(1, materialCircularity));
    const q = Math.max(0, Math.min(1, qualityVerified));
    const l = Math.max(0, Math.min(1, localParticipation));

    return e * 0.3 + m * 0.3 + q * 0.2 + l * 0.2;
  }

  /**
   * Estimate MYCELIUM reward for a print (client-side helper)
   *
   * @param pogScore - PoGF score (0-1)
   * @param qualityScore - Quality score (0-1)
   * @returns Estimated MYCELIUM (CIV) reward
   */
  estimateMyceliumReward(pogScore: number, qualityScore: number): number {
    const MIN_POG_FOR_MYCELIUM = 0.6;
    const BASE_REWARD = 100;

    if (pogScore < MIN_POG_FOR_MYCELIUM) {
      return 0;
    }

    const multiplier = (pogScore + qualityScore) / 2;
    const bonus = (pogScore - MIN_POG_FOR_MYCELIUM) * 50;

    return Math.floor(BASE_REWARD * multiplier + bonus);
  }
}
