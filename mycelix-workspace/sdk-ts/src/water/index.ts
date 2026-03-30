// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Water Commons Module
 *
 * Unified access to Mycelix modules configured for water sovereignty:
 * - FLOW: Water allocation, sharing economics, mutual credit (H2O currency)
 * - PURITY: Quality monitoring, treatment, contamination alerts
 * - CAPTURE: Rainwater harvesting, aquifer recharge, storage systems
 * - STEWARD: Watershed governance, water rights, commons management
 * - WISDOM: Traditional water knowledge, climate patterns, conservation
 *
 * This is NOT a new hApp—it's a configuration layer unifying existing modules.
 * Water Commons instantiates the Universal Commons Framework for water resources.
 *
 * @packageDocumentation
 * @module water
 */

import {
  UniversalCommonsService,
  type ResourceMetadata,
  type CommonsDefinition,
  type StewardProfile,
  type StewardRole,
  type GovernanceRule,
  type CommonsWisdom,
} from '../commons/index.js';
import {
  claim,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type EpistemicClaim,
} from '../epistemic/index.js';
import {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  type ReputationScore,
} from '../matl/index.js';

// ============================================================================
// FLOW - Water Allocation Economics
// ============================================================================

/** Water source types */
export type WaterSourceType =
  | 'municipal'
  | 'well'
  | 'spring'
  | 'rainwater'
  | 'river'
  | 'lake'
  | 'aquifer'
  | 'desalinated'
  | 'recycled';

/** Water classification by use */
export type WaterClassification =
  | 'potable'        // Safe for drinking
  | 'freshwater'     // Fresh but needs treatment
  | 'graywater'      // Lightly used (sinks, showers)
  | 'blackwater'     // Heavily contaminated (toilets)
  | 'irrigation'     // Agricultural use
  | 'industrial'     // Industrial processes
  | 'recreational';  // Swimming, fishing

/** Water allocation share */
export interface WaterShare {
  id: string;
  sourceId: string;
  sourceDid: string;
  holderId: string;
  holderDid: string;
  allocationType: 'fixed' | 'proportional' | 'priority' | 'emergency';
  volumePerPeriod: number;  // Liters
  periodType: 'daily' | 'weekly' | 'monthly' | 'seasonal';
  priority: number;         // 1 = highest (drinking), 5 = lowest (recreation)
  usageCategory: WaterClassification;
  startDate: number;
  endDate?: number;
  status: 'active' | 'suspended' | 'revoked';
  createdAt: number;
}

/** Water payment/credit record */
export interface WaterPayment {
  id: string;
  shareId: string;
  period: string;  // e.g., "2024-W01"
  volumeUsed: number;
  volumeAllocated: number;
  creditAmount: number;  // H2O credits
  status: 'pending' | 'confirmed' | 'disputed';
  paidAt?: number;
}

/** Water source profile */
export interface WaterSourceProfile {
  id: string;
  did: string;
  name: string;
  sourceType: WaterSourceType;
  location: string;
  coordinates?: { lat: number; lng: number };
  watershedId?: string;

  // Capacity
  maxCapacityLiters: number;
  currentLevelLiters: number;
  rechargeRateLitersPerDay: number;

  // Quality baseline
  baselineQuality: WaterQualityProfile;
  lastTestedAt: number;

  // Trust
  reputation: ReputationScore;
  trustScore: number;

  // Usage
  activeShares: number;
  totalUsers: number;

  // Infrastructure
  treatmentLevel: TreatmentLevel;
  distributionMethod: DistributionMethod;
}

/** Treatment levels */
export type TreatmentLevel =
  | 'none'           // Raw/natural
  | 'filtered'       // Physical filtration
  | 'uv_treated'     // UV disinfection
  | 'chlorinated'    // Chemical treatment
  | 'reverse_osmosis' // RO purification
  | 'distilled';     // Distillation

/** Distribution methods */
export type DistributionMethod =
  | 'gravity_fed'    // Gravity flow
  | 'pumped'         // Electric/manual pump
  | 'piped'          // Municipal pipes
  | 'trucked'        // Water truck delivery
  | 'containerized'  // Bottles/containers
  | 'canal';         // Open channel

// ============================================================================
// PURITY - Water Quality Monitoring
// ============================================================================

/** Water quality profile */
export interface WaterQualityProfile {
  id: string;
  sourceId: string;
  testedAt: number;
  testedBy: string;  // DID of tester

  // Physical
  temperature: number;        // Celsius
  turbidity: number;          // NTU (Nephelometric Turbidity Units)
  color: number;              // Pt-Co units
  odor: 'none' | 'mild' | 'strong';

  // Chemical
  ph: number;                 // 0-14 scale
  tds: number;                // Total Dissolved Solids (ppm)
  hardness: number;           // mg/L CaCO3
  chlorine: number;           // mg/L
  fluoride: number;           // mg/L
  nitrates: number;           // mg/L
  nitrites: number;           // mg/L
  arsenic: number;            // μg/L
  lead: number;               // μg/L

  // Biological
  coliformTotal: number;      // CFU/100mL
  coliformFecal: number;      // CFU/100mL
  eColi: boolean;             // Presence

  // Overall assessment
  potabilityScore: number;    // 0-1 (1 = fully potable)
  classification: WaterClassification;

  // Epistemic
  epistemicClaim?: EpistemicClaim;
  labCertified: boolean;
  certificationId?: string;
}

/** Water quality alert */
export interface WaterQualityAlert {
  id: string;
  sourceId: string;
  alertType: 'contamination' | 'low_level' | 'quality_decline' | 'infrastructure_failure';
  severity: 'advisory' | 'warning' | 'emergency';
  contaminant?: string;
  levelDetected?: number;
  safeLevel?: number;
  affectedShares: string[];
  issuedAt: number;
  resolvedAt?: number;
  recommendations: string[];
}

/** Contamination event record */
export interface ContaminationEvent {
  id: string;
  sourceId: string;
  contaminantType: string;
  contaminantSource?: string;
  detectedAt: number;
  levelDetected: number;
  unitOfMeasure: string;
  safeThreshold: number;
  affectedArea?: string;
  affectedPopulation?: number;
  remediationSteps: RemediationStep[];
  resolvedAt?: number;
  epistemicClaim?: EpistemicClaim;
}

/** Remediation step */
export interface RemediationStep {
  id: string;
  description: string;
  assignedTo?: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  startedAt?: number;
  completedAt?: number;
  verifiedBy?: string;
}

// ============================================================================
// CAPTURE - Water Harvesting & Storage
// ============================================================================

/** Rainwater harvesting system */
export interface RainwaterSystem {
  id: string;
  ownerDid: string;
  name: string;
  location: string;
  coordinates?: { lat: number; lng: number };

  // Collection
  catchmentAreaSqM: number;
  roofMaterial: string;
  gutterType: string;
  firstFlushDiverted: boolean;

  // Storage
  storageTanks: StorageTank[];
  totalCapacityLiters: number;
  currentLevelLiters: number;

  // Treatment
  treatmentLevel: TreatmentLevel;
  filterType?: string;

  // Usage
  intendedUse: WaterClassification[];
  connectedToCommons: boolean;
  commonsId?: string;

  // Verification
  lastInspectedAt?: number;
  inspectedBy?: string;
  certifications: string[];

  // Trust
  reputation: ReputationScore;
  trustScore: number;

  createdAt: number;
  updatedAt: number;
}

/** Storage tank */
export interface StorageTank {
  id: string;
  systemId: string;
  capacityLiters: number;
  currentLevelLiters: number;
  material: 'polyethylene' | 'concrete' | 'steel' | 'fiberglass' | 'ferrocement';
  aboveGround: boolean;
  hasLevelSensor: boolean;
  lastCleanedAt?: number;
}

/** Aquifer recharge project */
export interface AquiferRechargeProject {
  id: string;
  name: string;
  watershedId: string;
  location: string;
  coordinates?: { lat: number; lng: number };

  // Technical
  rechargeMethod: 'infiltration_basin' | 'injection_well' | 'spreading_basin' | 'recharge_trench';
  sourceWater: WaterSourceType;
  targetAquifer: string;

  // Capacity
  designCapacityLitersPerDay: number;
  actualRechargeLitersPerDay: number;
  totalRechargedLiters: number;

  // Quality
  pretreatmentRequired: boolean;
  qualityMonitoring: WaterQualityProfile[];

  // Governance
  stewards: StewardProfile[];
  fundingSource: string;

  // Verification
  permitId?: string;
  environmentalImpactAssessment?: string;

  createdAt: number;
  updatedAt: number;
}

// ============================================================================
// STEWARD - Watershed Governance
// ============================================================================

/** Watershed definition */
export interface Watershed {
  id: string;
  name: string;
  hucCode?: string;  // Hydrologic Unit Code (US)

  // Geography
  boundaryGeoJson?: unknown;  // GeoJSON polygon
  areaSqKm: number;
  mainRiver?: string;
  tributaries: string[];

  // Hydrology
  averageAnnualRainfallMm: number;
  averageAnnualDischargeLiters: number;
  groundwaterDepthM?: number;

  // Ecology
  ecosystemType: string;
  protectedAreas: string[];
  endangeredSpecies: string[];

  // Human use
  population: number;
  majorUses: WaterClassification[];
  industrialUsers: number;
  agriculturalAreaHa: number;

  // Governance
  governingBody?: string;
  waterRightsSystem: 'riparian' | 'prior_appropriation' | 'correlative' | 'public_trust' | 'commons';

  // Commons integration
  commonsId?: string;
  registeredSources: string[];

  createdAt: number;
  updatedAt: number;
}

/** Water right */
export interface WaterRight {
  id: string;
  holderId: string;
  holderDid: string;
  holderName: string;

  // Source
  sourceId: string;
  watershedId: string;

  // Allocation
  allocationType: 'riparian' | 'appropriative' | 'permit' | 'commons_share';
  annualVolumeLiters: number;
  maxInstantaneousFlowLitersPerSecond?: number;

  // Priority
  priorityDate?: number;  // For prior appropriation systems
  priorityNumber?: number;

  // Use
  beneficialUse: WaterClassification[];
  pointOfDiversion?: { lat: number; lng: number };
  placeOfUse?: string;

  // Status
  status: 'active' | 'suspended' | 'forfeited' | 'transferred';
  transferHistory: WaterRightTransfer[];

  // Governance
  governanceRules: GovernanceRule[];

  // Verification
  permitNumber?: string;
  issuedBy?: string;
  issuedAt?: number;
  expiresAt?: number;

  // Trust
  reputation: ReputationScore;
  complianceScore: number;  // 0-1

  createdAt: number;
  updatedAt: number;
}

/** Water right transfer */
export interface WaterRightTransfer {
  id: string;
  rightId: string;
  fromHolderId: string;
  toHolderId: string;
  volumeTransferred: number;
  transferType: 'permanent' | 'temporary' | 'lease';
  price?: number;
  currency?: string;
  approvedBy?: string;
  transferredAt: number;
}

/** Water dispute */
export interface WaterDispute {
  id: string;
  complainantId: string;
  respondentId: string;
  disputeType: 'allocation' | 'quality' | 'access' | 'contamination' | 'rights';
  description: string;
  evidenceIds: string[];

  // Resolution
  status: 'filed' | 'mediation' | 'arbitration' | 'resolved' | 'escalated';
  mediatorId?: string;
  resolution?: string;

  // Timeline
  filedAt: number;
  resolvedAt?: number;

  // Governance
  governingCommonsId?: string;
  appliedRules: string[];
}

// ============================================================================
// WISDOM - Traditional Water Knowledge
// ============================================================================

/** Traditional water practice */
export interface TraditionalWaterPractice {
  id: string;
  name: string;
  culture: string;
  region: string;

  // Knowledge
  description: string;
  practiceType: 'harvesting' | 'storage' | 'treatment' | 'distribution' | 'conservation' | 'ceremonial';

  // Technical
  technicalDetails?: string;
  materialsUsed?: string[];
  climateAdaptation?: string;

  // Validation
  scientificValidation?: string;
  modernApplications?: string[];

  // Provenance
  knowledgeHolders: string[];
  oralTradition: boolean;
  documentedSources?: string[];

  // Rights
  intellectualPropertyStatus: 'public_domain' | 'traditional_knowledge' | 'protected' | 'sacred';
  consentRequired: boolean;

  // Trust
  epistemicClaim?: EpistemicClaim;

  createdAt: number;
}

/** Water conservation practice */
export interface ConservationPractice {
  id: string;
  name: string;
  category: 'household' | 'agricultural' | 'industrial' | 'municipal' | 'landscape';

  // Impact
  averageWaterSavingPercent: number;
  implementationCost: 'low' | 'medium' | 'high';
  paybackPeriodMonths?: number;

  // Technical
  description: string;
  requirements: string[];
  contraindications?: string[];

  // Verification
  studies: string[];
  epistemicClaim?: EpistemicClaim;

  // Adoption
  adoptionRate?: number;
  successStories: string[];
}

/** Climate-water pattern */
export interface ClimateWaterPattern {
  id: string;
  watershedId: string;
  name: string;

  // Pattern
  patternType: 'seasonal' | 'drought_cycle' | 'flood_cycle' | 'long_term_trend';
  periodYears?: number;

  // Historical data
  historicalObservations: Array<{
    year: number;
    rainfallMm: number;
    temperatureC: number;
    waterAvailabilityPercent: number;
  }>;

  // Predictions
  projectedChange?: string;
  confidenceLevel?: number;

  // Traditional knowledge
  traditionalIndicators?: string[];
  indigenousNames?: Record<string, string>;

  // Adaptation
  recommendedAdaptations: string[];

  epistemicClaim?: EpistemicClaim;
  createdAt: number;
}

// ============================================================================
// Water Metadata (extends ResourceMetadata)
// ============================================================================

/** Water-specific metadata for CommonsResource */
export interface WaterMetadata extends ResourceMetadata {
  sourceType: WaterSourceType;
  classification: WaterClassification;
  treatmentLevel: TreatmentLevel;
  qualityProfile?: WaterQualityProfile;
  watershedId?: string;
  seasonalAvailability?: Record<string, number>;  // month -> availability %
  harvestMethod?: string;
  storageType?: string;
}

// ============================================================================
// Water Commons Service
// ============================================================================

/**
 * WaterCommonsService - Extends UniversalCommonsService for water-specific operations.
 *
 * Five Pillars:
 * - FLOW: Water allocation economics (H2O credits)
 * - PURITY: Quality monitoring and alerts
 * - CAPTURE: Harvesting and storage systems
 * - STEWARD: Watershed governance
 * - WISDOM: Traditional knowledge and conservation
 *
 * @example
 * ```typescript
 * const water = new WaterCommonsService();
 *
 * // Create a watershed commons
 * const commons = water.createWatershedCommons({
 *   name: 'Cedar Creek Watershed Commons',
 *   watershedId: 'HUC-12040101',
 *   // ...
 * });
 *
 * // Register a water source
 * const source = water.registerSource({
 *   name: 'Community Well #3',
 *   sourceType: 'well',
 *   // ...
 * });
 *
 * // Monitor quality
 * const quality = water.recordQualityTest({
 *   sourceId: source.id,
 *   ph: 7.2,
 *   turbidity: 0.5,
 *   // ...
 * });
 * ```
 */
export class WaterCommonsService extends UniversalCommonsService<WaterMetadata> {
  // Water-specific storage
  private waterSources: Map<string, WaterSourceProfile> = new Map();
  private watersheds: Map<string, Watershed> = new Map();
  private waterShares: Map<string, WaterShare> = new Map();
  private qualityProfiles: Map<string, WaterQualityProfile> = new Map();
  private qualityAlerts: Map<string, WaterQualityAlert> = new Map();
  private rainwaterSystems: Map<string, RainwaterSystem> = new Map();
  private rechargeProjects: Map<string, AquiferRechargeProject> = new Map();
  private waterRights: Map<string, WaterRight> = new Map();
  private disputes: Map<string, WaterDispute> = new Map();
  private traditionalPractices: Map<string, TraditionalWaterPractice> = new Map();
  private conservationPractices: Map<string, ConservationPractice> = new Map();
  private climatePatterns: Map<string, ClimateWaterPattern> = new Map();
  private contaminationEvents: Map<string, ContaminationEvent> = new Map();

  constructor() {
    super('water');
  }

  // ===========================================================================
  // FLOW - Water Allocation Economics
  // ===========================================================================

  /**
   * Create a watershed-based commons
   */
  createWatershedCommons(params: {
    name: string;
    description: string;
    watershed: Omit<Watershed, 'id' | 'commonsId' | 'registeredSources' | 'createdAt' | 'updatedAt'>;
    initialStewards: Array<{
      did: string;
      name: string;
      role: StewardRole;
      allocationRights: number;
    }>;
    governanceRules?: GovernanceRule[];
  }): { commons: CommonsDefinition; watershed: Watershed } {
    // Create the commons using base class
    const commons = this.createCommons({
      name: params.name,
      description: params.description,
      location: params.watershed.name,
      initialStewards: params.initialStewards,
      governanceRules: params.governanceRules,
    });

    // Create the watershed
    const watershed: Watershed = {
      ...params.watershed,
      id: `watershed-${Date.now()}`,
      commonsId: commons.id,
      registeredSources: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.watersheds.set(watershed.id, watershed);

    return { commons, watershed };
  }

  /**
   * Register a water source in the commons
   */
  registerSource(params: {
    name: string;
    sourceType: WaterSourceType;
    location: string;
    coordinates?: { lat: number; lng: number };
    watershedId?: string;
    maxCapacityLiters: number;
    currentLevelLiters: number;
    rechargeRateLitersPerDay: number;
    treatmentLevel: TreatmentLevel;
    distributionMethod: DistributionMethod;
    initialQuality?: Partial<WaterQualityProfile>;
    commonsId?: string;
  }): WaterSourceProfile {
    const source: WaterSourceProfile = {
      id: `source-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      did: `did:mycelix:water:source:${Date.now()}`,
      name: params.name,
      sourceType: params.sourceType,
      location: params.location,
      coordinates: params.coordinates,
      watershedId: params.watershedId,
      maxCapacityLiters: params.maxCapacityLiters,
      currentLevelLiters: params.currentLevelLiters,
      rechargeRateLitersPerDay: params.rechargeRateLitersPerDay,
      baselineQuality: this.createDefaultQualityProfile(params.initialQuality),
      lastTestedAt: Date.now(),
      reputation: createReputation(`water-source-${Date.now()}`),
      trustScore: 0.5,
      activeShares: 0,
      totalUsers: 0,
      treatmentLevel: params.treatmentLevel,
      distributionMethod: params.distributionMethod,
    };

    this.waterSources.set(source.id, source);

    // Register in watershed if specified
    if (params.watershedId) {
      const watershed = this.watersheds.get(params.watershedId);
      if (watershed) {
        watershed.registeredSources.push(source.id);
        watershed.updatedAt = Date.now();
      }
    }

    // Register as a commons resource
    this.registerResource({
      name: params.name,
      description: `${params.sourceType} water source`,
      quantity: params.currentLevelLiters,
      unit: 'liters',
      quality: source.baselineQuality.potabilityScore,
      commonsId: params.commonsId,
      metadata: {
        sourceType: params.sourceType,
        classification: source.baselineQuality.classification,
        treatmentLevel: params.treatmentLevel,
        qualityProfile: source.baselineQuality,
        watershedId: params.watershedId,
      },
    });

    return source;
  }

  /**
   * Create a water share allocation
   */
  createWaterShare(params: {
    sourceId: string;
    holderId: string;
    holderDid: string;
    allocationType: WaterShare['allocationType'];
    volumePerPeriod: number;
    periodType: WaterShare['periodType'];
    priority: number;
    usageCategory: WaterClassification;
    durationMonths?: number;
  }): WaterShare {
    const source = this.waterSources.get(params.sourceId);
    if (!source) throw new Error('Water source not found');

    const share: WaterShare = {
      id: `share-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      sourceId: params.sourceId,
      sourceDid: source.did,
      holderId: params.holderId,
      holderDid: params.holderDid,
      allocationType: params.allocationType,
      volumePerPeriod: params.volumePerPeriod,
      periodType: params.periodType,
      priority: params.priority,
      usageCategory: params.usageCategory,
      startDate: Date.now(),
      endDate: params.durationMonths
        ? Date.now() + params.durationMonths * 30 * 24 * 60 * 60 * 1000
        : undefined,
      status: 'active',
      createdAt: Date.now(),
    };

    this.waterShares.set(share.id, share);
    source.activeShares++;
    source.totalUsers++;

    // Issue H2O credits based on allocation
    this.issueCredits({
      earnedBy: params.holderId,
      earnedByDid: params.holderDid,
      amount: this.calculateInitialCredits(share),
      earnedThrough: 'stewardship',
    });

    return share;
  }

  /**
   * Record water usage
   */
  recordUsage(params: {
    shareId: string;
    volumeUsed: number;
    usedAt?: number;
  }): WaterPayment {
    const share = this.waterShares.get(params.shareId);
    if (!share) throw new Error('Water share not found');

    const period = this.getCurrentPeriod(share.periodType);
    const creditCost = this.calculateCreditCost(params.volumeUsed, share);

    const payment: WaterPayment = {
      id: `payment-${Date.now()}`,
      shareId: params.shareId,
      period,
      volumeUsed: params.volumeUsed,
      volumeAllocated: share.volumePerPeriod,
      creditAmount: creditCost,
      status: 'confirmed',
      paidAt: params.usedAt || Date.now(),
    };

    // Record consumption in base class
    this.recordConsumption({
      resourceId: share.sourceId,
      actorId: share.holderId,
      actorDid: share.holderDid,
      quantity: params.volumeUsed,
      destination: share.usageCategory,
    });

    return payment;
  }

  /**
   * Get water share holder's usage summary
   */
  getUsageSummary(holderId: string): {
    activeShares: WaterShare[];
    totalAllocated: number;
    totalUsed: number;
    efficiency: number;
    creditBalance: number;
  } {
    const shares = Array.from(this.waterShares.values())
      .filter(s => s.holderId === holderId && s.status === 'active');

    const totalAllocated = shares.reduce((sum, s) => sum + s.volumePerPeriod, 0);

    // Calculate usage from consumption records
    const consumptions = Array.from(this.flows.values())
      .filter(f => f.actorId === holderId && f.flowType === 'consumption');
    const totalUsed = consumptions.reduce((sum, c) => sum + c.quantity, 0);

    const efficiency = totalAllocated > 0 ? 1 - (totalUsed / totalAllocated) : 1;

    // Get credit balance
    const credits = Array.from(this.credits.values())
      .filter(c => c.earnedBy === holderId);
    const creditBalance = credits.reduce((sum, c) => sum + c.currentBalance, 0);

    return {
      activeShares: shares,
      totalAllocated,
      totalUsed,
      efficiency,
      creditBalance,
    };
  }

  // ===========================================================================
  // PURITY - Water Quality Monitoring
  // ===========================================================================

  /**
   * Record a water quality test
   */
  recordQualityTest(params: {
    sourceId: string;
    testedBy: string;
    physical?: {
      temperature?: number;
      turbidity?: number;
      color?: number;
      odor?: 'none' | 'mild' | 'strong';
    };
    chemical?: {
      ph?: number;
      tds?: number;
      hardness?: number;
      chlorine?: number;
      fluoride?: number;
      nitrates?: number;
      nitrites?: number;
      arsenic?: number;
      lead?: number;
    };
    biological?: {
      coliformTotal?: number;
      coliformFecal?: number;
      eColi?: boolean;
    };
    labCertified?: boolean;
    certificationId?: string;
  }): WaterQualityProfile {
    const source = this.waterSources.get(params.sourceId);
    if (!source) throw new Error('Water source not found');

    const profile: WaterQualityProfile = {
      id: `quality-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      sourceId: params.sourceId,
      testedAt: Date.now(),
      testedBy: params.testedBy,

      // Physical
      temperature: params.physical?.temperature ?? 20,
      turbidity: params.physical?.turbidity ?? 0,
      color: params.physical?.color ?? 0,
      odor: params.physical?.odor ?? 'none',

      // Chemical
      ph: params.chemical?.ph ?? 7,
      tds: params.chemical?.tds ?? 0,
      hardness: params.chemical?.hardness ?? 0,
      chlorine: params.chemical?.chlorine ?? 0,
      fluoride: params.chemical?.fluoride ?? 0,
      nitrates: params.chemical?.nitrates ?? 0,
      nitrites: params.chemical?.nitrites ?? 0,
      arsenic: params.chemical?.arsenic ?? 0,
      lead: params.chemical?.lead ?? 0,

      // Biological
      coliformTotal: params.biological?.coliformTotal ?? 0,
      coliformFecal: params.biological?.coliformFecal ?? 0,
      eColi: params.biological?.eColi ?? false,

      // Assessment
      potabilityScore: 0, // Calculated below
      classification: 'freshwater', // Determined below

      // Verification
      labCertified: params.labCertified ?? false,
      certificationId: params.certificationId,

      // Epistemic
      epistemicClaim: claim(`Water quality test for ${source.name}`)
        .withEmpirical(params.labCertified ? EmpiricalLevel.E3_Cryptographic : EmpiricalLevel.E2_PrivateVerify)
        .withNormative(NormativeLevel.N1_Communal)
        .withMateriality(MaterialityLevel.M2_Persistent)
        .build(),
    };

    // Calculate potability score
    profile.potabilityScore = this.calculatePotabilityScore(profile);
    profile.classification = this.determineClassification(profile);

    this.qualityProfiles.set(profile.id, profile);

    // Update source
    source.baselineQuality = profile;
    source.lastTestedAt = Date.now();

    // Check for alerts
    this.checkQualityThresholds(profile, source);

    // Update source reputation based on quality
    if (profile.potabilityScore > 0.8) {
      source.reputation = recordPositive(source.reputation);
    } else if (profile.potabilityScore < 0.5) {
      source.reputation = recordNegative(source.reputation);
    }
    source.trustScore = reputationValue(source.reputation);

    return profile;
  }

  /**
   * Get quality history for a source
   */
  getQualityHistory(sourceId: string, limit = 10): WaterQualityProfile[] {
    return Array.from(this.qualityProfiles.values())
      .filter(p => p.sourceId === sourceId)
      .sort((a, b) => b.testedAt - a.testedAt)
      .slice(0, limit);
  }

  /**
   * Create a contamination alert
   */
  createAlert(params: {
    sourceId: string;
    alertType: WaterQualityAlert['alertType'];
    severity: WaterQualityAlert['severity'];
    contaminant?: string;
    levelDetected?: number;
    safeLevel?: number;
    recommendations: string[];
  }): WaterQualityAlert {
    const source = this.waterSources.get(params.sourceId);
    if (!source) throw new Error('Water source not found');

    // Find affected shares
    const affectedShares = Array.from(this.waterShares.values())
      .filter(s => s.sourceId === params.sourceId && s.status === 'active')
      .map(s => s.id);

    const alert: WaterQualityAlert = {
      id: `alert-${Date.now()}`,
      sourceId: params.sourceId,
      alertType: params.alertType,
      severity: params.severity,
      contaminant: params.contaminant,
      levelDetected: params.levelDetected,
      safeLevel: params.safeLevel,
      affectedShares,
      issuedAt: Date.now(),
      recommendations: params.recommendations,
    };

    this.qualityAlerts.set(alert.id, alert);

    // If emergency, suspend all shares
    if (params.severity === 'emergency') {
      for (const shareId of affectedShares) {
        const share = this.waterShares.get(shareId);
        if (share) {
          share.status = 'suspended';
        }
      }
    }

    return alert;
  }

  /**
   * Get active alerts for a source or all sources
   */
  getActiveAlerts(sourceId?: string): WaterQualityAlert[] {
    const alerts = Array.from(this.qualityAlerts.values())
      .filter(a => !a.resolvedAt);

    if (sourceId) {
      return alerts.filter(a => a.sourceId === sourceId);
    }
    return alerts;
  }

  /**
   * Resolve an alert
   */
  resolveAlert(alertId: string, _resolution?: string): WaterQualityAlert {
    const alert = this.qualityAlerts.get(alertId);
    if (!alert) throw new Error('Alert not found');

    alert.resolvedAt = Date.now();

    // Reactivate suspended shares if this was an emergency
    if (alert.severity === 'emergency') {
      for (const shareId of alert.affectedShares) {
        const share = this.waterShares.get(shareId);
        if (share && share.status === 'suspended') {
          share.status = 'active';
        }
      }
    }

    return alert;
  }

  /**
   * Record a contamination event
   */
  recordContamination(params: {
    sourceId: string;
    contaminantType: string;
    contaminantSource?: string;
    levelDetected: number;
    unitOfMeasure: string;
    safeThreshold: number;
    affectedArea?: string;
    affectedPopulation?: number;
  }): ContaminationEvent {
    const event: ContaminationEvent = {
      id: `contamination-${Date.now()}`,
      sourceId: params.sourceId,
      contaminantType: params.contaminantType,
      contaminantSource: params.contaminantSource,
      detectedAt: Date.now(),
      levelDetected: params.levelDetected,
      unitOfMeasure: params.unitOfMeasure,
      safeThreshold: params.safeThreshold,
      affectedArea: params.affectedArea,
      affectedPopulation: params.affectedPopulation,
      remediationSteps: [],
      epistemicClaim: claim(`Contamination event: ${params.contaminantType}`)
        .withEmpirical(EmpiricalLevel.E2_PrivateVerify)
        .withNormative(NormativeLevel.N2_Network)
        .withMateriality(MaterialityLevel.M3_Immutable)
        .build(),
    };

    this.contaminationEvents.set(event.id, event);

    // Automatically create emergency alert
    this.createAlert({
      sourceId: params.sourceId,
      alertType: 'contamination',
      severity: params.levelDetected > params.safeThreshold * 2 ? 'emergency' : 'warning',
      contaminant: params.contaminantType,
      levelDetected: params.levelDetected,
      safeLevel: params.safeThreshold,
      recommendations: [
        'Do not use water for drinking or cooking',
        'Contact local health authorities',
        'Document any health symptoms',
      ],
    });

    return event;
  }

  // ===========================================================================
  // CAPTURE - Water Harvesting & Storage
  // ===========================================================================

  /**
   * Register a rainwater harvesting system
   */
  registerRainwaterSystem(params: {
    ownerDid: string;
    name: string;
    location: string;
    coordinates?: { lat: number; lng: number };
    catchmentAreaSqM: number;
    roofMaterial: string;
    gutterType: string;
    firstFlushDiverted: boolean;
    storageTanks: Array<{
      capacityLiters: number;
      material: StorageTank['material'];
      aboveGround: boolean;
      hasLevelSensor: boolean;
    }>;
    treatmentLevel: TreatmentLevel;
    filterType?: string;
    intendedUse: WaterClassification[];
    commonsId?: string;
  }): RainwaterSystem {
    const systemId = `rainwater-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    const tanks: StorageTank[] = params.storageTanks.map((t, i) => ({
      id: `tank-${systemId}-${i}`,
      systemId,
      capacityLiters: t.capacityLiters,
      currentLevelLiters: 0,
      material: t.material,
      aboveGround: t.aboveGround,
      hasLevelSensor: t.hasLevelSensor,
    }));

    const totalCapacity = tanks.reduce((sum, t) => sum + t.capacityLiters, 0);

    const system: RainwaterSystem = {
      id: systemId,
      ownerDid: params.ownerDid,
      name: params.name,
      location: params.location,
      coordinates: params.coordinates,
      catchmentAreaSqM: params.catchmentAreaSqM,
      roofMaterial: params.roofMaterial,
      gutterType: params.gutterType,
      firstFlushDiverted: params.firstFlushDiverted,
      storageTanks: tanks,
      totalCapacityLiters: totalCapacity,
      currentLevelLiters: 0,
      treatmentLevel: params.treatmentLevel,
      filterType: params.filterType,
      intendedUse: params.intendedUse,
      connectedToCommons: !!params.commonsId,
      commonsId: params.commonsId,
      certifications: [],
      reputation: createReputation(`rainwater-${systemId}`),
      trustScore: 0.5,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.rainwaterSystems.set(system.id, system);

    // If connected to commons, register as a source
    if (params.commonsId) {
      this.registerSource({
        name: params.name,
        sourceType: 'rainwater',
        location: params.location,
        coordinates: params.coordinates,
        maxCapacityLiters: totalCapacity,
        currentLevelLiters: 0,
        rechargeRateLitersPerDay: this.estimateRainwaterRecharge(params.catchmentAreaSqM),
        treatmentLevel: params.treatmentLevel,
        distributionMethod: 'gravity_fed',
        commonsId: params.commonsId,
      });
    }

    return system;
  }

  /**
   * Record rainwater harvest
   */
  recordHarvest(params: {
    systemId: string;
    volumeLiters: number;
    rainfallMm: number;
    harvestedAt?: number;
  }): RainwaterSystem {
    const system = this.rainwaterSystems.get(params.systemId);
    if (!system) throw new Error('Rainwater system not found');

    // Update current level
    const newLevel = Math.min(
      system.currentLevelLiters + params.volumeLiters,
      system.totalCapacityLiters
    );
    system.currentLevelLiters = newLevel;
    system.updatedAt = Date.now();

    // Distribute across tanks (simple proportional fill)
    const fillRatio = newLevel / system.totalCapacityLiters;
    for (const tank of system.storageTanks) {
      tank.currentLevelLiters = tank.capacityLiters * fillRatio;
    }

    // Record production if connected to commons
    if (system.commonsId) {
      this.recordProduction({
        resourceId: system.id,
        actorId: system.ownerDid,
        actorDid: system.ownerDid,
        quantity: params.volumeLiters,
        source: `rainwater_harvest:${params.rainfallMm}mm`,
      });

      // Issue H2O credits for harvesting
      this.issueCredits({
        earnedBy: system.ownerDid,
        earnedByDid: system.ownerDid,
        amount: Math.floor(params.volumeLiters / 100), // 1 credit per 100L
        earnedThrough: 'production',
      });
    }

    return system;
  }

  /**
   * Get rainwater systems in a region
   */
  getRainwaterSystems(params?: {
    commonsId?: string;
    minCapacityLiters?: number;
    treatmentLevel?: TreatmentLevel;
  }): RainwaterSystem[] {
    let systems = Array.from(this.rainwaterSystems.values());

    if (params?.commonsId) {
      systems = systems.filter(s => s.commonsId === params.commonsId);
    }
    if (params?.minCapacityLiters) {
      systems = systems.filter(s => s.totalCapacityLiters >= params.minCapacityLiters!);
    }
    if (params?.treatmentLevel) {
      systems = systems.filter(s => s.treatmentLevel === params.treatmentLevel);
    }

    return systems;
  }

  /**
   * Register an aquifer recharge project
   */
  registerRechargeProject(params: {
    name: string;
    watershedId: string;
    location: string;
    coordinates?: { lat: number; lng: number };
    rechargeMethod: AquiferRechargeProject['rechargeMethod'];
    sourceWater: WaterSourceType;
    targetAquifer: string;
    designCapacityLitersPerDay: number;
    pretreatmentRequired: boolean;
    initialStewards: Array<{
      did: string;
      name: string;
      role: StewardRole;
      allocationRights: number;
    }>;
    fundingSource: string;
    permitId?: string;
  }): AquiferRechargeProject {
    const project: AquiferRechargeProject = {
      id: `recharge-${Date.now()}`,
      name: params.name,
      watershedId: params.watershedId,
      location: params.location,
      coordinates: params.coordinates,
      rechargeMethod: params.rechargeMethod,
      sourceWater: params.sourceWater,
      targetAquifer: params.targetAquifer,
      designCapacityLitersPerDay: params.designCapacityLitersPerDay,
      actualRechargeLitersPerDay: 0,
      totalRechargedLiters: 0,
      pretreatmentRequired: params.pretreatmentRequired,
      qualityMonitoring: [],
      stewards: params.initialStewards.map((s, i) => ({
        id: `steward-${i}-${Date.now()}`,
        did: s.did,
        name: s.name,
        role: s.role,
        contributions: [],
        hoursContributed: 0,
        allocationRights: s.allocationRights,
        reputation: createReputation(s.did),
        trustScore: 0.5,
        joinedAt: Date.now(),
        lastActivity: Date.now(),
      })),
      fundingSource: params.fundingSource,
      permitId: params.permitId,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.rechargeProjects.set(project.id, project);

    // Update watershed
    const watershed = this.watersheds.get(params.watershedId);
    if (watershed) {
      watershed.updatedAt = Date.now();
    }

    return project;
  }

  /**
   * Record recharge activity
   */
  recordRecharge(params: {
    projectId: string;
    volumeLiters: number;
    rechargedAt?: number;
    qualityTest?: Partial<WaterQualityProfile>;
  }): AquiferRechargeProject {
    const project = this.rechargeProjects.get(params.projectId);
    if (!project) throw new Error('Recharge project not found');

    project.totalRechargedLiters += params.volumeLiters;
    project.actualRechargeLitersPerDay = params.volumeLiters; // Simplified
    project.updatedAt = Date.now();

    // Record quality if provided
    if (params.qualityTest) {
      const quality: WaterQualityProfile = {
        id: `quality-recharge-${Date.now()}`,
        sourceId: project.id,
        testedAt: params.rechargedAt || Date.now(),
        testedBy: 'recharge_monitoring',
        temperature: params.qualityTest.temperature ?? 20,
        turbidity: params.qualityTest.turbidity ?? 0,
        color: params.qualityTest.color ?? 0,
        odor: params.qualityTest.odor ?? 'none',
        ph: params.qualityTest.ph ?? 7,
        tds: params.qualityTest.tds ?? 0,
        hardness: params.qualityTest.hardness ?? 0,
        chlorine: params.qualityTest.chlorine ?? 0,
        fluoride: params.qualityTest.fluoride ?? 0,
        nitrates: params.qualityTest.nitrates ?? 0,
        nitrites: params.qualityTest.nitrites ?? 0,
        arsenic: params.qualityTest.arsenic ?? 0,
        lead: params.qualityTest.lead ?? 0,
        coliformTotal: params.qualityTest.coliformTotal ?? 0,
        coliformFecal: params.qualityTest.coliformFecal ?? 0,
        eColi: params.qualityTest.eColi ?? false,
        potabilityScore: 0.8,
        classification: 'freshwater',
        labCertified: false,
      };
      project.qualityMonitoring.push(quality);
    }

    return project;
  }

  // ===========================================================================
  // STEWARD - Watershed Governance
  // ===========================================================================

  /**
   * Register a water right
   */
  registerWaterRight(params: {
    holderId: string;
    holderDid: string;
    holderName: string;
    sourceId: string;
    watershedId: string;
    allocationType: WaterRight['allocationType'];
    annualVolumeLiters: number;
    maxInstantaneousFlowLitersPerSecond?: number;
    priorityDate?: number;
    beneficialUse: WaterClassification[];
    pointOfDiversion?: { lat: number; lng: number };
    placeOfUse?: string;
    permitNumber?: string;
    issuedBy?: string;
    expiresAt?: number;
  }): WaterRight {
    const right: WaterRight = {
      id: `right-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      holderId: params.holderId,
      holderDid: params.holderDid,
      holderName: params.holderName,
      sourceId: params.sourceId,
      watershedId: params.watershedId,
      allocationType: params.allocationType,
      annualVolumeLiters: params.annualVolumeLiters,
      maxInstantaneousFlowLitersPerSecond: params.maxInstantaneousFlowLitersPerSecond,
      priorityDate: params.priorityDate,
      beneficialUse: params.beneficialUse,
      pointOfDiversion: params.pointOfDiversion,
      placeOfUse: params.placeOfUse,
      status: 'active',
      transferHistory: [],
      governanceRules: [],
      permitNumber: params.permitNumber,
      issuedBy: params.issuedBy,
      issuedAt: Date.now(),
      expiresAt: params.expiresAt,
      reputation: createReputation(`water-right-${Date.now()}`),
      complianceScore: 1.0,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.waterRights.set(right.id, right);

    return right;
  }

  /**
   * Transfer a water right
   */
  transferWaterRight(params: {
    rightId: string;
    toHolderId: string;
    toHolderDid: string;
    volumeTransferred?: number;
    transferType: 'permanent' | 'temporary' | 'lease';
    price?: number;
    currency?: string;
    approvedBy?: string;
  }): WaterRightTransfer {
    const right = this.waterRights.get(params.rightId);
    if (!right) throw new Error('Water right not found');

    const transfer: WaterRightTransfer = {
      id: `transfer-${Date.now()}`,
      rightId: params.rightId,
      fromHolderId: right.holderId,
      toHolderId: params.toHolderId,
      volumeTransferred: params.volumeTransferred ?? right.annualVolumeLiters,
      transferType: params.transferType,
      price: params.price,
      currency: params.currency,
      approvedBy: params.approvedBy,
      transferredAt: Date.now(),
    };

    right.transferHistory.push(transfer);

    if (params.transferType === 'permanent') {
      right.holderId = params.toHolderId;
      right.holderDid = params.toHolderDid;
    }

    right.updatedAt = Date.now();

    return transfer;
  }

  /**
   * File a water dispute
   */
  fileDispute(params: {
    complainantId: string;
    respondentId: string;
    disputeType: WaterDispute['disputeType'];
    description: string;
    evidenceIds?: string[];
    governingCommonsId?: string;
  }): WaterDispute {
    const dispute: WaterDispute = {
      id: `dispute-${Date.now()}`,
      complainantId: params.complainantId,
      respondentId: params.respondentId,
      disputeType: params.disputeType,
      description: params.description,
      evidenceIds: params.evidenceIds || [],
      status: 'filed',
      filedAt: Date.now(),
      governingCommonsId: params.governingCommonsId,
      appliedRules: [],
    };

    this.disputes.set(dispute.id, dispute);

    return dispute;
  }

  /**
   * Get disputes for a party
   */
  getDisputes(partyId: string): WaterDispute[] {
    return Array.from(this.disputes.values())
      .filter(d => d.complainantId === partyId || d.respondentId === partyId);
  }

  /**
   * Resolve a dispute
   */
  resolveDispute(params: {
    disputeId: string;
    resolution: string;
    mediatorId?: string;
    appliedRules?: string[];
  }): WaterDispute {
    const dispute = this.disputes.get(params.disputeId);
    if (!dispute) throw new Error('Dispute not found');

    dispute.status = 'resolved';
    dispute.resolution = params.resolution;
    dispute.mediatorId = params.mediatorId;
    dispute.resolvedAt = Date.now();
    if (params.appliedRules) {
      dispute.appliedRules = params.appliedRules;
    }

    return dispute;
  }

  // ===========================================================================
  // WISDOM - Traditional Water Knowledge
  // ===========================================================================

  /**
   * Register traditional water practice
   */
  registerTraditionalPractice(params: {
    name: string;
    culture: string;
    region: string;
    description: string;
    practiceType: TraditionalWaterPractice['practiceType'];
    technicalDetails?: string;
    materialsUsed?: string[];
    climateAdaptation?: string;
    scientificValidation?: string;
    modernApplications?: string[];
    knowledgeHolders: string[];
    oralTradition: boolean;
    documentedSources?: string[];
    intellectualPropertyStatus: TraditionalWaterPractice['intellectualPropertyStatus'];
    consentRequired: boolean;
  }): TraditionalWaterPractice {
    const practice: TraditionalWaterPractice = {
      id: `practice-${Date.now()}`,
      name: params.name,
      culture: params.culture,
      region: params.region,
      description: params.description,
      practiceType: params.practiceType,
      technicalDetails: params.technicalDetails,
      materialsUsed: params.materialsUsed,
      climateAdaptation: params.climateAdaptation,
      scientificValidation: params.scientificValidation,
      modernApplications: params.modernApplications,
      knowledgeHolders: params.knowledgeHolders,
      oralTradition: params.oralTradition,
      documentedSources: params.documentedSources,
      intellectualPropertyStatus: params.intellectualPropertyStatus,
      consentRequired: params.consentRequired,
      epistemicClaim: claim(`Traditional water practice: ${params.name}`)
        .withEmpirical(EmpiricalLevel.E1_Testimonial)
        .withNormative(NormativeLevel.N1_Communal)
        .withMateriality(MaterialityLevel.M2_Persistent)
        .build(),
      createdAt: Date.now(),
    };

    this.traditionalPractices.set(practice.id, practice);

    return practice;
  }

  /**
   * Search traditional practices
   */
  searchTraditionalPractices(params?: {
    practiceType?: TraditionalWaterPractice['practiceType'];
    culture?: string;
    region?: string;
    hasScientificValidation?: boolean;
  }): TraditionalWaterPractice[] {
    let practices = Array.from(this.traditionalPractices.values());

    if (params?.practiceType) {
      practices = practices.filter(p => p.practiceType === params.practiceType);
    }
    if (params?.culture) {
      practices = practices.filter(p =>
        p.culture.toLowerCase().includes(params.culture!.toLowerCase())
      );
    }
    if (params?.region) {
      practices = practices.filter(p =>
        p.region.toLowerCase().includes(params.region!.toLowerCase())
      );
    }
    if (params?.hasScientificValidation !== undefined) {
      practices = practices.filter(p =>
        params.hasScientificValidation ? !!p.scientificValidation : !p.scientificValidation
      );
    }

    return practices;
  }

  /**
   * Register conservation practice
   */
  registerConservationPractice(params: {
    name: string;
    category: ConservationPractice['category'];
    averageWaterSavingPercent: number;
    implementationCost: 'low' | 'medium' | 'high';
    paybackPeriodMonths?: number;
    description: string;
    requirements: string[];
    contraindications?: string[];
    studies: string[];
  }): ConservationPractice {
    const practice: ConservationPractice = {
      id: `conservation-${Date.now()}`,
      name: params.name,
      category: params.category,
      averageWaterSavingPercent: params.averageWaterSavingPercent,
      implementationCost: params.implementationCost,
      paybackPeriodMonths: params.paybackPeriodMonths,
      description: params.description,
      requirements: params.requirements,
      contraindications: params.contraindications,
      studies: params.studies,
      epistemicClaim: claim(`Conservation practice: ${params.name}`)
        .withEmpirical(EmpiricalLevel.E2_PrivateVerify)
        .withNormative(NormativeLevel.N2_Network)
        .withMateriality(MaterialityLevel.M2_Persistent)
        .build(),
      adoptionRate: 0,
      successStories: [],
    };

    this.conservationPractices.set(practice.id, practice);

    return practice;
  }

  /**
   * Get conservation recommendations
   */
  getConservationRecommendations(params: {
    category?: ConservationPractice['category'];
    maxCost?: 'low' | 'medium' | 'high';
    minSavingPercent?: number;
  }): ConservationPractice[] {
    let practices = Array.from(this.conservationPractices.values());

    if (params.category) {
      practices = practices.filter(p => p.category === params.category);
    }

    if (params.maxCost) {
      const costOrder = { low: 1, medium: 2, high: 3 };
      practices = practices.filter(p => costOrder[p.implementationCost] <= costOrder[params.maxCost!]);
    }

    if (params.minSavingPercent) {
      practices = practices.filter(p => p.averageWaterSavingPercent >= params.minSavingPercent!);
    }

    // Sort by water saving potential
    return practices.sort((a, b) => b.averageWaterSavingPercent - a.averageWaterSavingPercent);
  }

  /**
   * Record climate-water pattern
   */
  recordClimatePattern(params: {
    watershedId: string;
    name: string;
    patternType: ClimateWaterPattern['patternType'];
    periodYears?: number;
    historicalObservations: ClimateWaterPattern['historicalObservations'];
    projectedChange?: string;
    confidenceLevel?: number;
    traditionalIndicators?: string[];
    indigenousNames?: Record<string, string>;
    recommendedAdaptations: string[];
  }): ClimateWaterPattern {
    const pattern: ClimateWaterPattern = {
      id: `climate-${Date.now()}`,
      watershedId: params.watershedId,
      name: params.name,
      patternType: params.patternType,
      periodYears: params.periodYears,
      historicalObservations: params.historicalObservations,
      projectedChange: params.projectedChange,
      confidenceLevel: params.confidenceLevel,
      traditionalIndicators: params.traditionalIndicators,
      indigenousNames: params.indigenousNames,
      recommendedAdaptations: params.recommendedAdaptations,
      epistemicClaim: claim(`Climate pattern: ${params.name}`)
        .withEmpirical(EmpiricalLevel.E2_PrivateVerify)
        .withNormative(NormativeLevel.N2_Network)
        .withMateriality(MaterialityLevel.M2_Persistent)
        .build(),
      createdAt: Date.now(),
    };

    this.climatePatterns.set(pattern.id, pattern);

    return pattern;
  }

  /**
   * Get water wisdom guidance
   */
  getWaterWisdom(params: {
    watershedId?: string;
    decision: string;
    context?: Record<string, unknown>;
  }): CommonsWisdom & {
    traditionalPractices: TraditionalWaterPractice[];
    conservationRecommendations: ConservationPractice[];
    climateConsiderations: ClimateWaterPattern[];
  } {
    // Get base wisdom from parent class
    const baseWisdom = this.getWisdomGuidance({
      resourceId: params.watershedId || 'general',
      decision: params.decision,
      context: params.context,
    });

    // Add water-specific wisdom
    const traditionalPractices = params.watershedId
      ? this.searchTraditionalPractices()
      : [];

    const conservationRecommendations = this.getConservationRecommendations({
      minSavingPercent: 10,
    });

    const climateConsiderations = params.watershedId
      ? Array.from(this.climatePatterns.values())
          .filter(p => p.watershedId === params.watershedId)
      : [];

    return {
      ...baseWisdom,
      traditionalPractices,
      conservationRecommendations,
      climateConsiderations,
    };
  }

  // ===========================================================================
  // Helpers
  // ===========================================================================

  private createDefaultQualityProfile(partial?: Partial<WaterQualityProfile>): WaterQualityProfile {
    return {
      id: `quality-default-${Date.now()}`,
      sourceId: '',
      testedAt: Date.now(),
      testedBy: 'system',
      temperature: partial?.temperature ?? 20,
      turbidity: partial?.turbidity ?? 1,
      color: partial?.color ?? 5,
      odor: partial?.odor ?? 'none',
      ph: partial?.ph ?? 7,
      tds: partial?.tds ?? 100,
      hardness: partial?.hardness ?? 50,
      chlorine: partial?.chlorine ?? 0,
      fluoride: partial?.fluoride ?? 0,
      nitrates: partial?.nitrates ?? 5,
      nitrites: partial?.nitrites ?? 0,
      arsenic: partial?.arsenic ?? 0,
      lead: partial?.lead ?? 0,
      coliformTotal: partial?.coliformTotal ?? 0,
      coliformFecal: partial?.coliformFecal ?? 0,
      eColi: partial?.eColi ?? false,
      potabilityScore: partial?.potabilityScore ?? 0.8,
      classification: partial?.classification ?? 'freshwater',
      labCertified: false,
    };
  }

  private calculatePotabilityScore(profile: WaterQualityProfile): number {
    let score = 1.0;

    // WHO/EPA thresholds
    if (profile.ph < 6.5 || profile.ph > 8.5) score -= 0.1;
    if (profile.turbidity > 1) score -= 0.1;
    if (profile.turbidity > 5) score -= 0.2;
    if (profile.tds > 500) score -= 0.1;
    if (profile.tds > 1000) score -= 0.2;
    if (profile.nitrates > 10) score -= 0.2;
    if (profile.nitrites > 1) score -= 0.2;
    if (profile.arsenic > 10) score -= 0.3;
    if (profile.lead > 10) score -= 0.3;
    if (profile.coliformTotal > 0) score -= 0.2;
    if (profile.coliformFecal > 0) score -= 0.3;
    if (profile.eColi) score -= 0.4;
    if (profile.odor !== 'none') score -= 0.1;

    return Math.max(0, score);
  }

  private determineClassification(profile: WaterQualityProfile): WaterClassification {
    if (profile.potabilityScore >= 0.9 && !profile.eColi && profile.coliformFecal === 0) {
      return 'potable';
    }
    if (profile.potabilityScore >= 0.7) {
      return 'freshwater';
    }
    if (profile.potabilityScore >= 0.5) {
      return 'irrigation';
    }
    if (profile.coliformFecal > 100 || profile.eColi) {
      return 'blackwater';
    }
    return 'graywater';
  }

  private checkQualityThresholds(profile: WaterQualityProfile, source: WaterSourceProfile): void {
    const alerts: Array<{ type: WaterQualityAlert['alertType']; severity: WaterQualityAlert['severity']; contaminant: string; level: number; safe: number }> = [];

    // Check critical thresholds
    if (profile.eColi) {
      alerts.push({ type: 'contamination', severity: 'emergency', contaminant: 'E. coli', level: 1, safe: 0 });
    }
    if (profile.lead > 15) {
      alerts.push({ type: 'contamination', severity: 'emergency', contaminant: 'Lead', level: profile.lead, safe: 15 });
    }
    if (profile.arsenic > 10) {
      alerts.push({ type: 'contamination', severity: 'warning', contaminant: 'Arsenic', level: profile.arsenic, safe: 10 });
    }
    if (profile.nitrates > 50) {
      alerts.push({ type: 'contamination', severity: 'warning', contaminant: 'Nitrates', level: profile.nitrates, safe: 50 });
    }

    // Create alerts
    for (const alert of alerts) {
      this.createAlert({
        sourceId: source.id,
        alertType: alert.type,
        severity: alert.severity,
        contaminant: alert.contaminant,
        levelDetected: alert.level,
        safeLevel: alert.safe,
        recommendations: this.getRemediationRecommendations(alert.contaminant),
      });
    }
  }

  private getRemediationRecommendations(contaminant: string): string[] {
    const recommendations: Record<string, string[]> = {
      'E. coli': [
        'Boil water before drinking',
        'Use certified water filters',
        'Investigate source of contamination',
        'Test regularly until resolved',
      ],
      'Lead': [
        'Do not use for drinking or cooking',
        'Flush pipes before use',
        'Replace lead service lines',
        'Use certified lead removal filters',
      ],
      'Arsenic': [
        'Install arsenic removal system',
        'Consider alternative water source',
        'Test regularly',
        'Consult with water treatment specialist',
      ],
      'Nitrates': [
        'Do not use for infant formula',
        'Install reverse osmosis system',
        'Investigate agricultural runoff',
        'Test well depth and casing',
      ],
    };

    return recommendations[contaminant] || [
      'Consult local water authority',
      'Consider alternative water source',
      'Install appropriate filtration',
    ];
  }

  private calculateInitialCredits(share: WaterShare): number {
    // Base credits based on allocation
    const baseCredits = Math.floor(share.volumePerPeriod / 1000); // 1 credit per 1000L

    // Priority bonus (higher priority = lower bonus, as critical uses shouldn't be gamified)
    const priorityMultiplier = 1 - (share.priority - 1) * 0.1;

    return Math.floor(baseCredits * priorityMultiplier);
  }

  private calculateCreditCost(volumeUsed: number, share: WaterShare): number {
    // Base cost
    let cost = Math.floor(volumeUsed / 100); // 1 credit per 100L

    // Over-allocation penalty
    if (volumeUsed > share.volumePerPeriod) {
      const overuse = volumeUsed - share.volumePerPeriod;
      cost += Math.floor(overuse / 50); // 2x penalty for overuse
    }

    return cost;
  }

  private getCurrentPeriod(periodType: WaterShare['periodType']): string {
    const now = new Date();
    switch (periodType) {
      case 'daily':
        return now.toISOString().split('T')[0];
      case 'weekly':
        const week = Math.ceil(now.getDate() / 7);
        return `${now.getFullYear()}-W${week.toString().padStart(2, '0')}`;
      case 'monthly':
        return `${now.getFullYear()}-${(now.getMonth() + 1).toString().padStart(2, '0')}`;
      case 'seasonal':
        const month = now.getMonth();
        const season = month < 3 ? 'winter' : month < 6 ? 'spring' : month < 9 ? 'summer' : 'fall';
        return `${now.getFullYear()}-${season}`;
      default:
        return now.toISOString();
    }
  }

  private estimateRainwaterRecharge(catchmentAreaSqM: number): number {
    // Rough estimate: assume 1000mm annual rainfall, 85% collection efficiency
    const annualRainfallMm = 1000;
    const collectionEfficiency = 0.85;
    const annualLiters = catchmentAreaSqM * annualRainfallMm * collectionEfficiency;
    return Math.floor(annualLiters / 365);
  }
}

// ============================================================================
// Re-export commons types for convenience
// ============================================================================
export type {
  CommonsResource,
  CommonsDefinition,
  StewardProfile,
  MetabolicFlow,
  CircularCredit,
  Allocation,
  Bioregion,
  GovernanceRule,
  ContributionRecord,
  CommonsWisdom,
} from '../commons/index.js';
