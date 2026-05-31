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
  PrintJob,
  PrintRecord,
  PrintSettings,
  PrintResult,
  PrintIssue,
  PrintStatistics,
  GroundingCertificate,
  CincinnatiSession,
  CincinnatiReport,
  CreatePrintJobInput,
  RecordPrintInput,
} from '../types';

export interface PrintProgressUpdate {
  jobHash: ActionHash;
  progress: number;
  currentLayer?: number;
  estimatedTimeRemaining?: number;
}

export class PrintsClient {
  constructor(
    private client: AppClient,
    private roleName: string,
    private zomeName: string = 'prints'
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
  async acceptJob(hash: ActionHash): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'accept_print_job',
      payload: hash,
    });
  }

  /**
   * Start printing
   */
  async startPrint(hash: ActionHash): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'start_print',
      payload: hash,
    });
  }

  /**
   * Update print progress
   */
  async updateProgress(hash: ActionHash, progress: number): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'update_print_progress',
      payload: { hash, progress },
    });
  }

  /**
   * Complete print with result
   */
  async completePrint(hash: ActionHash, result: PrintResult): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'complete_print',
      payload: { hash, result },
    });
  }

  /**
   * Cancel a print job
   */
  async cancelPrint(hash: ActionHash, reason: string): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'cancel_print',
      payload: { hash, reason },
    });
  }

  // =========================================================================
  // PRINT RECORDS (WITH PoGF)
  // =========================================================================

  /**
   * Record print result with PoGF scoring
   *
   * PoGF Score = (E_renewable × 0.3) + (M_circular × 0.3) + (Q_verified × 0.2) + (L_local × 0.2)
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
  async addPhotos(hash: ActionHash, photos: string[]): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'add_print_photos',
      payload: { hash, photos },
    });
  }

  // =========================================================================
  // CINCINNATI ALGORITHM MONITORING
  // =========================================================================

  /**
   * Start Cincinnati monitoring session
   */
  async startCincinnatiSession(
    jobHash: ActionHash,
    baselineSignature: number[]
  ): Promise<CincinnatiSession> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'start_cincinnati_session',
      payload: { job_hash: jobHash, baseline_signature: baselineSignature },
    });
  }

  /**
   * Record Cincinnati anomaly
   */
  async recordAnomaly(
    sessionId: string,
    anomalyType: string,
    severity: number,
    sensorData: object
  ): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'record_cincinnati_anomaly',
      payload: {
        session_id: sessionId,
        anomaly_type: anomalyType,
        severity,
        sensor_data: sensorData,
      },
    });
  }

  /**
   * Complete Cincinnati session and get report
   */
  async completeCincinnatiSession(sessionId: string): Promise<CincinnatiReport> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'complete_cincinnati_session',
      payload: sessionId,
    });
  }

  /**
   * Get Cincinnati report for a print job
   */
  async getCincinnatiReport(jobHash: ActionHash): Promise<CincinnatiReport | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_cincinnati_report',
      payload: jobHash,
    });
  }

  // =========================================================================
  // DISCOVERY & STATISTICS
  // =========================================================================

  /**
   * Get my print jobs
   */
  async getMyJobs(): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_my_print_jobs',
      payload: null,
    });
  }

  /**
   * Get jobs for a specific printer
   */
  async getPrinterJobs(printerHash: ActionHash): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_printer_jobs',
      payload: printerHash,
    });
  }

  /**
   * Get all prints of a specific design
   */
  async getDesignPrints(designHash: ActionHash): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design_prints',
      payload: designHash,
    });
  }

  /**
   * Get statistics for a design
   */
  async getStatistics(designHash: ActionHash): Promise<PrintStatistics> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_print_statistics',
      payload: designHash,
    });
  }

  /**
   * Calculate PoGF score
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
   * Estimate MYCELIUM reward for a print
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
