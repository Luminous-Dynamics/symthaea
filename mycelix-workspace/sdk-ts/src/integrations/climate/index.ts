/**
 * @mycelix/sdk Climate Integration
 *
 * hApp-specific adapter for Mycelix-Climate providing:
 * - Carbon credit issuance and retirement
 * - Renewable Energy Certificate (REC) tracking
 * - Climate project registration and verification
 * - Cross-hApp carbon offset settlement via Bridge
 * - MATL-based project reputation
 *
 * @packageDocumentation
 * @module integrations/climate
 * @see {@link ClimateService} - Main service class
 * @see {@link getClimateService} - Singleton accessor
 *
 * @example Basic carbon credit workflow
 * ```typescript
 * import { getClimateService } from '@mycelix/sdk/integrations/climate';
 *
 * const climate = getClimateService();
 *
 * // Register a climate project
 * const project = climate.registerProject({
 *   id: 'proj-001',
 *   name: 'Solar Farm Offset',
 *   type: 'renewable_energy',
 *   location: { lat: 32.7, lng: -96.8 },
 *   estimatedCreditsPerYear: 5000,
 *   verifier: 'verra',
 * });
 *
 * // Issue carbon credits
 * const credits = climate.issueCredits('proj-001', 1000, 'tCO2e');
 *
 * // Retire credits for offsetting
 * climate.retireCredits(credits.id, 'company-abc', 'Annual offset commitment');
 * ```
 */

import { LocalBridge } from '../../bridge/index.js';
import {
  createReputation,
  recordPositive,
  reputationValue,
  type ReputationScore,
} from '../../matl/index.js';

// ============================================================================
// Climate-Specific Types
// ============================================================================

/** Climate project type */
export type ClimateProjectType =
  | 'renewable_energy'
  | 'reforestation'
  | 'methane_capture'
  | 'direct_air_capture'
  | 'energy_efficiency'
  | 'blue_carbon'
  | 'soil_carbon';

/** Verification standard */
export type VerificationStandard = 'verra' | 'gold_standard' | 'acr' | 'car' | 'plan_vivo';

/** Credit unit */
export type CreditUnit = 'tCO2e' | 'kgCO2e' | 'MWh';

/** Geographic location */
export interface GeoLocation {
  lat: number;
  lng: number;
}

/** Climate project */
export interface ClimateProject {
  id: string;
  name: string;
  type: ClimateProjectType;
  location: GeoLocation;
  estimatedCreditsPerYear: number;
  verifier: VerificationStandard;
  verified: boolean;
  registeredAt: number;
  description?: string;
  vintage?: number; // year
}

/** Carbon credit batch */
export interface CarbonCreditBatch {
  id: string;
  projectId: string;
  amount: number;
  unit: CreditUnit;
  vintage: number;
  issuedAt: number;
  retiredAt?: number;
  retiredBy?: string;
  retirementReason?: string;
  serialStart: string;
  serialEnd: string;
  status: 'active' | 'retired' | 'cancelled';
}

/** Renewable Energy Certificate */
export interface RenewableEnergyCertificate {
  id: string;
  projectId: string;
  amountMwh: number;
  generationPeriodStart: number;
  generationPeriodEnd: number;
  source: 'solar' | 'wind' | 'hydro' | 'geothermal' | 'biomass';
  issuedAt: number;
  redeemedAt?: number;
  redeemedBy?: string;
  status: 'active' | 'redeemed' | 'expired';
}

/** Carbon offset settlement (cross-hApp) */
export interface CarbonOffsetSettlement {
  id: string;
  sourceHapp: string;
  buyerDid: string;
  creditBatchId: string;
  amountOffset: number;
  unit: CreditUnit;
  settledAt: number;
}

// ============================================================================
// Climate Service
// ============================================================================

/**
 * ClimateService - Carbon credit and REC management for climate action
 *
 * Integrates with the mycelix-climate hApp zomes:
 * - `carbon` - Carbon credit issuance and retirement
 * - `projects` - Climate project registration and verification
 * - `bridge` - Cross-hApp carbon offset settlement
 */
export class ClimateService {
  private projects: Map<string, ClimateProject> = new Map();
  private credits: Map<string, CarbonCreditBatch> = new Map();
  private recs: Map<string, RenewableEnergyCertificate> = new Map();
  private projectReputations: Map<string, ReputationScore> = new Map();
  private bridge: LocalBridge;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('climate');
  }

  /**
   * Register a new climate project
   */
  registerProject(input: Omit<ClimateProject, 'verified' | 'registeredAt'>): ClimateProject {
    const project: ClimateProject = {
      ...input,
      verified: false,
      registeredAt: Date.now(),
    };

    this.projects.set(project.id, project);
    this.projectReputations.set(project.id, createReputation(project.id));

    return project;
  }

  /**
   * Verify a climate project (updates reputation)
   */
  verifyProject(projectId: string): ClimateProject {
    const project = this.projects.get(projectId);
    if (!project) {
      throw new Error(`Project not found: ${projectId}`);
    }

    project.verified = true;

    const rep = this.projectReputations.get(projectId);
    if (rep) {
      this.projectReputations.set(projectId, recordPositive(rep));
    }

    return project;
  }

  /**
   * Issue a batch of carbon credits from a verified project
   */
  issueCredits(
    projectId: string,
    amount: number,
    unit: CreditUnit,
    vintage?: number
  ): CarbonCreditBatch {
    const project = this.projects.get(projectId);
    if (!project) {
      throw new Error(`Project not found: ${projectId}`);
    }
    if (!project.verified) {
      throw new Error(`Project not verified: ${projectId}`);
    }

    const batch: CarbonCreditBatch = {
      id: `credit-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      projectId,
      amount,
      unit,
      vintage: vintage || new Date().getFullYear(),
      issuedAt: Date.now(),
      serialStart: `${projectId}-${Date.now()}-0001`,
      serialEnd: `${projectId}-${Date.now()}-${String(amount).padStart(4, '0')}`,
      status: 'active',
    };

    this.credits.set(batch.id, batch);
    return batch;
  }

  /**
   * Retire carbon credits for offsetting
   */
  retireCredits(batchId: string, retiredBy: string, reason?: string): CarbonCreditBatch {
    const batch = this.credits.get(batchId);
    if (!batch) {
      throw new Error(`Credit batch not found: ${batchId}`);
    }
    if (batch.status !== 'active') {
      throw new Error(`Credit batch not active: ${batchId} (status: ${batch.status})`);
    }

    batch.status = 'retired';
    batch.retiredAt = Date.now();
    batch.retiredBy = retiredBy;
    batch.retirementReason = reason;

    return batch;
  }

  /**
   * Issue a Renewable Energy Certificate
   */
  issueREC(input: Omit<RenewableEnergyCertificate, 'id' | 'issuedAt' | 'status'>): RenewableEnergyCertificate {
    const rec: RenewableEnergyCertificate = {
      ...input,
      id: `rec-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      issuedAt: Date.now(),
      status: 'active',
    };

    this.recs.set(rec.id, rec);
    return rec;
  }

  /**
   * Redeem a REC
   */
  redeemREC(recId: string, redeemedBy: string): RenewableEnergyCertificate {
    const rec = this.recs.get(recId);
    if (!rec) {
      throw new Error(`REC not found: ${recId}`);
    }
    if (rec.status !== 'active') {
      throw new Error(`REC not active: ${recId} (status: ${rec.status})`);
    }

    rec.status = 'redeemed';
    rec.redeemedAt = Date.now();
    rec.redeemedBy = redeemedBy;

    return rec;
  }

  /**
   * Get project by ID
   */
  getProject(projectId: string): ClimateProject | undefined {
    return this.projects.get(projectId);
  }

  /**
   * Get all projects
   */
  getAllProjects(): ClimateProject[] {
    return Array.from(this.projects.values());
  }

  /**
   * Get project reputation
   */
  getProjectReputation(projectId: string): number {
    const rep = this.projectReputations.get(projectId);
    return rep ? reputationValue(rep) : 0;
  }

  /**
   * Get active credits for a project
   */
  getActiveCredits(projectId: string): CarbonCreditBatch[] {
    return Array.from(this.credits.values()).filter(
      (c) => c.projectId === projectId && c.status === 'active'
    );
  }

  /**
   * Get total retired credits (tCO2e)
   */
  getTotalRetired(): number {
    return Array.from(this.credits.values())
      .filter((c) => c.status === 'retired' && c.unit === 'tCO2e')
      .reduce((sum, c) => sum + c.amount, 0);
  }
}

// ============================================================================
// Singleton
// ============================================================================

let defaultService: ClimateService | null = null;

/**
 * Get the default ClimateService instance
 */
export function getClimateService(): ClimateService {
  if (!defaultService) {
    defaultService = new ClimateService();
  }
  return defaultService;
}

/**
 * Reset the default ClimateService instance (for testing)
 */
export function resetClimateService(): void {
  defaultService = null;
}
