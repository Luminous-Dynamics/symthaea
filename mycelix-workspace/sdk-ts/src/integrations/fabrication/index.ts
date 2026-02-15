/**
 * @mycelix/sdk Fabrication Integration
 *
 * hApp-specific adapter for Mycelix-Fabrication providing:
 * - HDC-encoded parametric design management
 * - Proof of Grounded Fabrication (PoGF) scoring
 * - Anticipatory repair workflow automation
 * - Cincinnati Algorithm quality monitoring
 * - Cross-hApp bridge for metabolic economy integration
 *
 * @packageDocumentation
 * @module integrations/fabrication
 * @see {@link FabricationService} - Main service class
 * @see {@link getFabricationService} - Singleton accessor
 *
 * @example Basic design creation
 * ```typescript
 * import { getFabricationService } from '@mycelix/sdk/integrations/fabrication';
 *
 * const fab = getFabricationService();
 *
 * // Create a new design
 * const design = fab.createDesign({
 *   title: 'Replacement Battery Clip',
 *   description: 'Universal battery clip for power tools',
 *   category: 'Repair',
 *   safetyClass: 'Class1Functional',
 *   license: 'PublicDomain',
 * });
 *
 * console.log(`Design ID: ${design.id}`);
 * ```
 */

import { LocalBridge, createBroadcastEvent } from '../../bridge/index.js';
import { FLCoordinator, AggregationMethod } from '../../fl/index.js';
import {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  type ReputationScore,
} from '../../matl/index.js';

// ============================================================================
// Design Types
// ============================================================================

/**
 * Design category for classification and discovery
 */
export type DesignCategory =
  | 'Tools'
  | 'Parts'
  | 'Housewares'
  | 'Medical'
  | 'Accessibility'
  | 'Art'
  | 'Education'
  | 'Repair'
  | 'Custom';

/**
 * Safety classification for designs
 *
 * @remarks
 * Higher classes require more rigorous verification:
 * - Class0-1: Self-certification acceptable
 * - Class2-3: Community verification required
 * - Class4-5: Professional certification mandatory
 */
export type SafetyClass =
  | 'Class0Decorative'
  | 'Class1Functional'
  | 'Class2LoadBearing'
  | 'Class3BodyContact'
  | 'Class4Medical'
  | 'Class5Critical';

/**
 * License types for designs
 */
export type License =
  | 'PublicDomain'
  | 'CreativeCommons'
  | 'OpenHardware'
  | 'Proprietary'
  | { Custom: string };

/**
 * File format for design files
 */
export type FileFormat = 'STL' | 'STEP' | 'ThreeMF' | 'SCAD' | 'GCODE' | 'OBJ' | 'Other';

/**
 * Design file with IPFS storage
 */
export interface DesignFile {
  filename: string;
  format: FileFormat;
  ipfsCid: string;
  sizeBytes: number;
  checksumSha256: string;
}

/**
 * Parametric engine type
 */
export type ParametricEngine = 'OpenSCAD' | 'CadQuery' | 'FreeCAD' | 'Fusion360' | 'Other';

/**
 * Parameter type for parametric designs
 */
export type ParameterType =
  | 'Length'
  | 'Angle'
  | 'Count'
  | 'Boolean'
  | { Enum: string[] }
  | 'Material';

/**
 * Design parameter definition
 */
export interface DesignParameter {
  name: string;
  paramType: ParameterType;
  defaultValue: number | string | boolean;
  minValue?: number;
  maxValue?: number;
  unit?: string;
  hdcBinding?: string;
}

/**
 * Parametric schema for generative designs
 */
export interface ParametricSchema {
  engine: ParametricEngine;
  sourceTemplate: string;
  parameters: DesignParameter[];
  constraints: string[];
  autoGenerate: boolean;
}

/**
 * Repair manifest linking to parent products
 */
export interface RepairManifest {
  parentProductHash?: string;
  parentProductModel: string;
  partName: string;
  failureModes: FailureMode[];
  replacementIntervalHours?: number;
  repairDifficulty: 'Easy' | 'Moderate' | 'Difficult' | 'Expert';
}

/**
 * Failure mode for repair designs
 */
export type FailureMode =
  | 'MechanicalWear'
  | 'UvDegradation'
  | 'ThermalCycling'
  | 'ChemicalExposure'
  | 'ImpactDamage'
  | 'Fatigue';

/**
 * Epistemic classification for designs
 */
export interface DesignEpistemic {
  manufacturability: number; // 0.0-1.0
  safety: number; // 0.0-1.0
  usability: number; // 0.0-1.0
}

/**
 * Complete design entry
 */
export interface Design {
  id: string;
  title: string;
  description: string;
  category: DesignCategory;
  files: DesignFile[];
  parametricSchema?: ParametricSchema;
  repairManifest?: RepairManifest;
  circularityScore: number;
  embodiedEnergyKwh: number;
  license: License;
  safetyClass: SafetyClass;
  epistemic: DesignEpistemic;
  author: string;
  createdAt: number;
  updatedAt: number;
}

// ============================================================================
// Printer Types
// ============================================================================

/**
 * Printer technology type
 */
export type PrinterType = 'FDM' | 'SLA' | 'SLS' | 'DLP' | 'MJF' | { Other: string };

/**
 * Printer availability status
 */
export type AvailabilityStatus =
  | 'Available'
  | 'Busy'
  | 'Maintenance'
  | 'Offline'
  | 'ByAppointment';

/**
 * Build volume dimensions
 */
export interface BuildVolume {
  x: number;
  y: number;
  z: number;
}

/**
 * Printer capabilities
 */
export interface PrinterCapabilities {
  buildVolume: BuildVolume;
  layerHeights: number[];
  nozzleDiameters: number[];
  heatedBed: boolean;
  enclosure: boolean;
  multiMaterial?: number;
  maxTempHotend: number;
  maxTempBed: number;
  features: string[];
}

/**
 * Geographic location for printer discovery
 */
export interface GeoLocation {
  geohash: string;
  city?: string;
  region?: string;
  country: string;
}

/**
 * Printer rate structure
 */
export interface PrinterRates {
  baseFee: number;
  perHour: number;
  perGram: number;
}

/**
 * Complete printer registration
 */
export interface Printer {
  id: string;
  name: string;
  owner: string;
  location?: GeoLocation;
  printerType: PrinterType;
  capabilities: PrinterCapabilities;
  materialsAvailable: MaterialType[];
  availability: AvailabilityStatus;
  rates?: PrinterRates;
  createdAt: number;
  updatedAt: number;
}

/**
 * Printer match result for design compatibility
 */
export interface PrinterMatch {
  printer: Printer;
  compatibilityScore: number;
  estimatedTime: number;
  estimatedCost?: number;
  materialMatch: boolean;
  volumeMatch: boolean;
}

// ============================================================================
// Print Job Types
// ============================================================================

/**
 * Print job status
 */
export type PrintJobStatus =
  | 'Pending'
  | 'Accepted'
  | 'Queued'
  | 'Printing'
  | 'PostProcessing'
  | 'Completed'
  | 'Failed'
  | 'Cancelled';

/**
 * Print result type
 */
export type PrintResult = 'Success' | 'PartialSuccess' | { Failed: string };

/**
 * Print issues that can occur
 */
export type PrintIssue =
  | 'Warping'
  | 'LayerShift'
  | 'Stringing'
  | 'UnderExtrusion'
  | 'OverExtrusion'
  | 'BedAdhesion'
  | 'SupportFailure'
  | { Other: string };

/**
 * Temperature settings for print
 */
export interface TemperatureSettings {
  hotend: number;
  bed: number;
}

/**
 * Print job settings
 */
export interface PrintSettings {
  layerHeight: number;
  infillPercent: number;
  material: MaterialType;
  supports: boolean;
  raft: boolean;
  printSpeed?: number;
  temperatures: TemperatureSettings;
  customGcode?: string;
}

/**
 * Grounding request for PoGF compliance
 */
export interface GroundingRequest {
  requireRenewable: boolean;
  requireRecycledMaterial: boolean;
  targetPogScore: number;
}

/**
 * Energy source type
 */
export type EnergyType =
  | 'Solar'
  | 'Wind'
  | 'Hydro'
  | 'Geothermal'
  | 'Nuclear'
  | 'GridMix'
  | 'Unknown';

/**
 * Material origin for circularity tracking
 */
export type MaterialOrigin =
  | 'Virgin'
  | 'PostIndustrial'
  | 'PostConsumer'
  | 'Biobased'
  | 'UrbanMined';

/**
 * End of life strategy
 */
export type EndOfLifeStrategy =
  | 'MechanicalRecycling'
  | 'ChemicalRecycling'
  | 'Biodegradable'
  | 'IndustrialCompost'
  | 'Downcycle'
  | 'Landfill';

/**
 * Material passport for tracking
 */
export interface MaterialPassport {
  materialHash: string;
  batchId: string;
  origin: MaterialOrigin;
  recycledContentPercent: number;
  supplyChainHash?: string;
  certifications: string[];
  endOfLife: EndOfLifeStrategy;
}

/**
 * Grounding certificate for PoGF
 */
export interface GroundingCertificate {
  certificateId: string;
  terraAtlasEnergyHash?: string;
  energyType: EnergyType;
  gridCarbonIntensity: number;
  materialPassports: MaterialPassport[];
  hearthFundingHash?: string;
  issuedAt: number;
  issuerSignature: Uint8Array;
}

/**
 * Quality assessment for completed prints
 */
export interface QualityAssessment {
  dimensionalAccuracy: number;
  surfaceQuality: number;
  structuralIntegrity: number;
}

/**
 * Print job entry
 */
export interface PrintJob {
  id: string;
  designHash: string;
  printerHash: string;
  requester: string;
  settings: PrintSettings;
  status: PrintJobStatus;
  groundingRequest?: GroundingRequest;
  groundingCertificate?: GroundingCertificate;
  estimatedTimeMinutes?: number;
  actualTimeMinutes?: number;
  materialUsedGrams?: number;
  createdAt: number;
  startedAt?: number;
  completedAt?: number;
}

// ============================================================================
// Cincinnati Algorithm Types (Quality Monitoring)
// ============================================================================

/**
 * Anomaly type detected during print
 */
export type AnomalyType =
  | 'ExtrusionInconsistency'
  | 'TemperatureDeviation'
  | 'VibrationAnomaly'
  | 'LayerAdhesionFailure'
  | 'NozzleClog'
  | 'BedLevelDrift'
  | 'PowerFluctuation'
  | 'Unknown';

/**
 * Cincinnati action recommendation
 */
export type CincinnatiAction =
  | 'Continue'
  | { AdjustParameters: Record<string, number> }
  | 'PauseForInspection'
  | { AbortPrint: string }
  | { AlertOperator: string };

/**
 * Sensor snapshot during print
 */
export interface SensorSnapshot {
  hotendTemp: number;
  bedTemp: number;
  stepperCurrents: [number, number, number, number];
  vibrationRms: number;
  filamentTension?: number;
  ambientTemp?: number;
  humidity?: number;
}

/**
 * Anomaly event from Cincinnati monitoring
 */
export interface AnomalyEvent {
  timestampMs: number;
  layerNumber: number;
  anomalyType: AnomalyType;
  severity: number;
  sensorData: SensorSnapshot;
  actionTaken?: CincinnatiAction;
}

/**
 * Cincinnati monitoring report
 */
export interface CincinnatiReport {
  sessionId: string;
  totalSamples: number;
  anomaliesDetected: number;
  anomalyEvents: AnomalyEvent[];
  overallHealthScore: number;
  layerByLayerScores: number[];
  recommendedAction: CincinnatiAction;
}

/**
 * Print record with PoGF metrics
 */
export interface PrintRecord {
  jobHash: string;
  result: PrintResult;
  qualityScore?: number;
  pogScore: number;
  energyUsedKwh: number;
  carbonOffsetKg?: number;
  materialCircularity: number;
  myceliumEarned: number;
  cincinnatiReport?: CincinnatiReport;
  qualityAssessment?: QualityAssessment;
  photos: string[];
  notes: string;
  issues: PrintIssue[];
  recordedAt: number;
}

// ============================================================================
// Material Types
// ============================================================================

/**
 * Material type enumeration
 */
export type MaterialType =
  | 'PLA'
  | 'PETG'
  | 'ABS'
  | 'ASA'
  | 'TPU'
  | 'Nylon'
  | 'PC'
  | 'PEEK'
  | 'StandardResin'
  | 'ToughResin'
  | 'FlexibleResin'
  | 'CastableResin'
  | 'DentalResin'
  | 'NylonPowder'
  | 'MetalPowder'
  | { Custom: string };

/**
 * Material physical properties
 */
export interface MaterialProperties {
  printTempMin: number;
  printTempMax: number;
  bedTempMin?: number;
  bedTempMax?: number;
  densityGCm3: number;
  tensileStrengthMpa?: number;
  elongationPercent?: number;
  foodSafe: boolean;
  uvResistant: boolean;
  waterResistant: boolean;
  recyclable: boolean;
}

/**
 * Certification type for materials
 */
export type CertificationType =
  | 'FoodSafe'
  | 'Biocompatible'
  | 'FlameRetardant'
  | 'RoHSCompliant'
  | 'REACHCompliant'
  | 'ISO'
  | { Custom: string };

/**
 * Material certification
 */
export interface MaterialCertification {
  certType: CertificationType;
  issuer: string;
  validUntil?: number;
  documentCid?: string;
}

/**
 * Complete material entry
 */
export interface Material {
  id: string;
  name: string;
  materialType: MaterialType;
  properties: MaterialProperties;
  certifications: MaterialCertification[];
  suppliers: string[];
  safetyDataSheetCid?: string;
  createdAt: number;
}

// ============================================================================
// Verification Types
// ============================================================================

/**
 * Verification type
 */
export type VerificationType =
  | 'StructuralAnalysis'
  | 'MaterialCompatibility'
  | 'PrintabilityTest'
  | 'SafetyReview'
  | 'FoodSafeCertification'
  | 'MedicalCertification'
  | 'CommunityReview';

/**
 * Verification result
 */
export type VerificationResult =
  | { Passed: { confidence: number; notes: string } }
  | { Failed: { reasons: string[] } }
  | { ConditionalPass: { conditions: string[]; confidence: number } }
  | 'NeedsMoreEvidence';

/**
 * Design verification entry
 */
export interface DesignVerification {
  designHash: string;
  verificationType: VerificationType;
  result: VerificationResult;
  evidence: string[];
  verifier: string;
  verifierCredentials: string[];
  createdAt: number;
}

// ============================================================================
// Bridge Types (Anticipatory Repair)
// ============================================================================

/**
 * Repair action recommendation
 */
export type RepairAction =
  | 'OrderPart'
  | 'PrintReplacement'
  | 'CreateDesign'
  | 'ScheduleMaintenance'
  | 'Monitor';

/**
 * Repair workflow status
 */
export type RepairWorkflowStatus =
  | 'Predicted'
  | 'DesignFound'
  | 'PrinterMatched'
  | 'FundingApproved'
  | 'Printing'
  | 'ReadyForInstall'
  | 'Installed'
  | 'Cancelled';

/**
 * Repair prediction from Property hApp digital twin
 */
export interface RepairPrediction {
  propertyAssetHash: string;
  assetModel: string;
  predictedFailureComponent: string;
  failureProbability: number;
  estimatedFailureDate: number;
  confidenceIntervalDays: number;
  sensorDataSummary: string;
  recommendedAction: RepairAction;
  createdAt: number;
}

/**
 * Repair workflow tracking
 */
export interface RepairWorkflow {
  predictionHash: string;
  status: RepairWorkflowStatus;
  designHash?: string;
  printerHash?: string;
  hearthFundingHash?: string;
  printJobHash?: string;
  propertyInstallationHash?: string;
  createdAt: number;
  completedAt?: number;
}

/**
 * Fabrication event type
 */
export type FabricationEventType =
  | 'DesignPublished'
  | 'DesignVerified'
  | 'PrintCompleted'
  | 'PrinterRegistered'
  | 'MaterialShortage'
  | 'VerificationRequested';

/**
 * Fabrication event for bridge
 */
export interface FabricationEvent {
  eventType: FabricationEventType;
  designId?: string;
  payload: string;
  sourceHapp: string;
  timestamp: number;
}

/**
 * Listing type for marketplace integration
 */
export type ListingType = 'DesignSale' | 'PrintService' | 'PrintedProduct';

/**
 * Marketplace listing for designs
 */
export interface MarketplaceListing {
  designHash: string;
  marketplaceListingHash: string;
  price?: number;
  listingType: ListingType;
  createdAt: number;
}

// ============================================================================
// Fabrication Service
// ============================================================================

/**
 * FabricationService - Main service for fabrication operations
 *
 * @remarks
 * Provides:
 * - Design CRUD operations with PoGF scoring
 * - Printer registration and matching
 * - Print job lifecycle management
 * - Cincinnati quality monitoring integration
 * - Anticipatory repair workflow automation
 *
 * @example
 * ```typescript
 * const fab = new FabricationService();
 *
 * // Create a design
 * const design = fab.createDesign({
 *   title: 'Widget Mount',
 *   category: 'Parts',
 *   safetyClass: 'Class1Functional',
 * });
 *
 * // Match to printers
 * const matches = fab.matchDesignToPrinters(design.id);
 *
 * // Create print job
 * const job = fab.createPrintJob({
 *   designId: design.id,
 *   printerId: matches[0].printer.id,
 *   settings: {
 *     layerHeight: 0.2,
 *     infillPercent: 20,
 *     material: 'PLA',
 *   },
 * });
 * ```
 */
export class FabricationService {
  private designs: Map<string, Design> = new Map();
  private printers: Map<string, Printer> = new Map();
  private printJobs: Map<string, PrintJob> = new Map();
  private printerReputations: Map<string, ReputationScore> = new Map();
  private bridge: LocalBridge;
  private flCoordinator: FLCoordinator;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('fabrication');

    this.flCoordinator = new FLCoordinator({
      minParticipants: 5,
      aggregationMethod: AggregationMethod.TrustWeighted,
      byzantineTolerance: 0.34,
    });
  }

  // ============================================================================
  // Design Operations
  // ============================================================================

  /**
   * Create a new design
   */
  createDesign(input: {
    title: string;
    description?: string;
    category: DesignCategory;
    safetyClass: SafetyClass;
    license?: License;
    files?: DesignFile[];
    parametricSchema?: ParametricSchema;
    repairManifest?: RepairManifest;
  }): Design {
    const id = `design-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const now = Date.now();

    const design: Design = {
      id,
      title: input.title,
      description: input.description || '',
      category: input.category,
      files: input.files || [],
      parametricSchema: input.parametricSchema,
      repairManifest: input.repairManifest,
      circularityScore: 0.5, // Default, to be calculated
      embodiedEnergyKwh: 0,
      license: input.license || 'PublicDomain',
      safetyClass: input.safetyClass,
      epistemic: {
        manufacturability: 0.5,
        safety: 0.5,
        usability: 0.5,
      },
      author: 'local-agent', // Would be AgentPubKey in real implementation
      createdAt: now,
      updatedAt: now,
    };

    this.designs.set(id, design);
    return design;
  }

  /**
   * Get a design by ID
   */
  getDesign(id: string): Design | undefined {
    return this.designs.get(id);
  }

  /**
   * Search designs by category
   */
  getDesignsByCategory(category: DesignCategory): Design[] {
    return Array.from(this.designs.values()).filter((d) => d.category === category);
  }

  /**
   * Fork a design with modifications
   */
  forkDesign(
    parentId: string,
    modifications: Partial<Pick<Design, 'title' | 'description' | 'category'>>
  ): Design | undefined {
    const parent = this.designs.get(parentId);
    if (!parent) return undefined;

    return this.createDesign({
      title: modifications.title || `${parent.title} (Fork)`,
      description: modifications.description || parent.description,
      category: modifications.category || parent.category,
      safetyClass: parent.safetyClass,
      license: parent.license,
      files: parent.files,
      parametricSchema: parent.parametricSchema,
      repairManifest: parent.repairManifest,
    });
  }

  // ============================================================================
  // Printer Operations
  // ============================================================================

  /**
   * Register a printer
   */
  registerPrinter(input: {
    name: string;
    printerType: PrinterType;
    capabilities: PrinterCapabilities;
    materialsAvailable: MaterialType[];
    location?: GeoLocation;
    rates?: PrinterRates;
  }): Printer {
    const id = `printer-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const now = Date.now();

    const printer: Printer = {
      id,
      name: input.name,
      owner: 'local-agent',
      location: input.location,
      printerType: input.printerType,
      capabilities: input.capabilities,
      materialsAvailable: input.materialsAvailable,
      availability: 'Available',
      rates: input.rates,
      createdAt: now,
      updatedAt: now,
    };

    this.printers.set(id, printer);
    return printer;
  }

  /**
   * Find printers by capability requirements
   */
  findPrintersByCapability(requirements: {
    minBuildVolume?: BuildVolume;
    requiredMaterials?: MaterialType[];
    requireEnclosure?: boolean;
  }): Printer[] {
    return Array.from(this.printers.values()).filter((p) => {
      if (requirements.minBuildVolume) {
        const vol = p.capabilities.buildVolume;
        const req = requirements.minBuildVolume;
        if (vol.x < req.x || vol.y < req.y || vol.z < req.z) return false;
      }

      if (requirements.requiredMaterials) {
        const hasMaterials = requirements.requiredMaterials.every((m) =>
          p.materialsAvailable.includes(m)
        );
        if (!hasMaterials) return false;
      }

      if (requirements.requireEnclosure && !p.capabilities.enclosure) {
        return false;
      }

      return true;
    });
  }

  /**
   * Match a design to compatible printers
   */
  matchDesignToPrinters(designId: string): PrinterMatch[] {
    const design = this.designs.get(designId);
    if (!design) return [];

    return Array.from(this.printers.values())
      .filter((p) => p.availability === 'Available')
      .map((printer) => {
        let score = 0.5;
        const materialMatch = true; // Simplified
        const volumeMatch = true; // Would check against design dimensions

        if (materialMatch) score += 0.25;
        if (volumeMatch) score += 0.25;

        return {
          printer,
          compatibilityScore: score,
          estimatedTime: 120, // 2 hours placeholder
          estimatedCost: printer.rates ? printer.rates.baseFee + 120 * (printer.rates.perHour / 60) : undefined,
          materialMatch,
          volumeMatch,
        };
      })
      .sort((a, b) => b.compatibilityScore - a.compatibilityScore);
  }

  /**
   * Update printer availability
   */
  updatePrinterAvailability(printerId: string, status: AvailabilityStatus): void {
    const printer = this.printers.get(printerId);
    if (printer) {
      printer.availability = status;
      printer.updatedAt = Date.now();
    }
  }

  // ============================================================================
  // Print Job Operations
  // ============================================================================

  /**
   * Create a print job
   */
  createPrintJob(input: {
    designId: string;
    printerId: string;
    settings: Omit<PrintSettings, 'temperatures'> & { temperatures?: TemperatureSettings };
    groundingRequest?: GroundingRequest;
  }): PrintJob | undefined {
    const design = this.designs.get(input.designId);
    const printer = this.printers.get(input.printerId);
    if (!design || !printer) return undefined;

    const id = `job-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    const job: PrintJob = {
      id,
      designHash: input.designId,
      printerHash: input.printerId,
      requester: 'local-agent',
      settings: {
        ...input.settings,
        temperatures: input.settings.temperatures || { hotend: 200, bed: 60 },
      },
      status: 'Pending',
      groundingRequest: input.groundingRequest,
      createdAt: Date.now(),
    };

    this.printJobs.set(id, job);
    return job;
  }

  /**
   * Accept a print job
   */
  acceptPrintJob(jobId: string): boolean {
    const job = this.printJobs.get(jobId);
    if (!job || job.status !== 'Pending') return false;

    job.status = 'Accepted';
    return true;
  }

  /**
   * Start printing
   */
  startPrint(jobId: string): boolean {
    const job = this.printJobs.get(jobId);
    if (!job || (job.status !== 'Accepted' && job.status !== 'Queued')) return false;

    job.status = 'Printing';
    job.startedAt = Date.now();

    // Update printer availability
    this.updatePrinterAvailability(job.printerHash, 'Busy');

    return true;
  }

  /**
   * Complete a print with results
   */
  completePrint(
    jobId: string,
    result: PrintResult,
    assessment: QualityAssessment,
    materialUsedGrams: number
  ): PrintRecord | undefined {
    const job = this.printJobs.get(jobId);
    if (!job || job.status !== 'Printing') return undefined;

    job.status = 'Completed';
    job.completedAt = Date.now();
    job.materialUsedGrams = materialUsedGrams;
    job.actualTimeMinutes = job.startedAt
      ? Math.round((job.completedAt - job.startedAt) / 60000)
      : undefined;

    // Update printer availability
    this.updatePrinterAvailability(job.printerHash, 'Available');

    // Calculate PoGF score
    const pogScore = this.calculatePogScore(job, assessment);

    // Calculate mycelium (CIV) earned based on PoGF
    const myceliumEarned = Math.round(pogScore * 100);

    const record: PrintRecord = {
      jobHash: jobId,
      result,
      qualityScore: (assessment.dimensionalAccuracy + assessment.surfaceQuality + assessment.structuralIntegrity) / 3,
      pogScore,
      energyUsedKwh: 0.2, // Placeholder
      materialCircularity: 0.5, // Placeholder
      myceliumEarned,
      qualityAssessment: assessment,
      photos: [],
      notes: '',
      issues: [],
      recordedAt: Date.now(),
    };

    // Update reputation
    if (result === 'Success') {
      this.recordSuccessfulPrint(job.printerHash);
    } else {
      this.recordFailedPrint(job.printerHash);
    }

    return record;
  }

  /**
   * Calculate Proof of Grounded Fabrication score
   *
   * PoG_score = (E_renewable × 0.3) + (M_circular × 0.3) + (Q_verified × 0.2) + (L_local × 0.2)
   */
  private calculatePogScore(job: PrintJob, assessment: QualityAssessment): number {
    // Energy score (0-1)
    const energyScore = job.groundingCertificate
      ? job.groundingCertificate.energyType !== 'GridMix'
        ? 1.0
        : 0.3
      : 0.3;

    // Material circularity (0-1)
    const materialScore = job.groundingCertificate
      ? job.groundingCertificate.materialPassports.reduce(
          (acc, mp) => acc + mp.recycledContentPercent / 100,
          0
        ) / Math.max(1, job.groundingCertificate.materialPassports.length)
      : 0.2;

    // Quality score (0-1)
    const qualityScore =
      (assessment.dimensionalAccuracy + assessment.surfaceQuality + assessment.structuralIntegrity) / 3;

    // Local economy participation (0-1)
    const localScore = job.groundingCertificate?.hearthFundingHash ? 1.0 : 0.2;

    return energyScore * 0.3 + materialScore * 0.3 + qualityScore * 0.2 + localScore * 0.2;
  }

  // ============================================================================
  // Reputation Operations
  // ============================================================================

  private recordSuccessfulPrint(printerId: string): void {
    let rep = this.printerReputations.get(printerId) || createReputation(printerId);
    rep = recordPositive(rep);
    this.printerReputations.set(printerId, rep);
    this.bridge.setReputation('fabrication', printerId, rep);
  }

  private recordFailedPrint(printerId: string): void {
    let rep = this.printerReputations.get(printerId) || createReputation(printerId);
    rep = recordNegative(rep);
    this.printerReputations.set(printerId, rep);
    this.bridge.setReputation('fabrication', printerId, rep);
  }

  /**
   * Get printer trust score
   */
  getPrinterTrustScore(printerId: string): number {
    const rep = this.printerReputations.get(printerId);
    return rep ? reputationValue(rep) : 0.5;
  }

  /**
   * Check if printer is trustworthy
   */
  isPrinterTrustworthy(printerId: string, threshold = 0.7): boolean {
    const rep = this.printerReputations.get(printerId);
    return rep ? isTrustworthy(rep, threshold) : false;
  }

  // ============================================================================
  // Anticipatory Repair Operations
  // ============================================================================

  /**
   * Process a repair prediction from Property hApp
   */
  processRepairPrediction(prediction: RepairPrediction): RepairWorkflow {
    const workflow: RepairWorkflow = {
      predictionHash: `pred-${Date.now()}`,
      status: 'Predicted',
      createdAt: Date.now(),
    };

    // Auto-search for matching designs
    const repairDesigns = Array.from(this.designs.values()).filter(
      (d) =>
        d.category === 'Repair' &&
        d.repairManifest?.parentProductModel === prediction.assetModel &&
        d.repairManifest?.partName === prediction.predictedFailureComponent
    );

    if (repairDesigns.length > 0) {
      workflow.designHash = repairDesigns[0].id;
      workflow.status = 'DesignFound';

      // Auto-match printers
      const matches = this.matchDesignToPrinters(repairDesigns[0].id);
      if (matches.length > 0) {
        workflow.printerHash = matches[0].printer.id;
        workflow.status = 'PrinterMatched';
      }
    }

    return workflow;
  }

  // ============================================================================
  // Event Emission
  // ============================================================================

  /**
   * Emit fabrication event to bridge
   */
  emitEvent(eventType: FabricationEventType, designId?: string, payload?: Record<string, unknown>): void {
    const event: FabricationEvent = {
      eventType,
      designId,
      payload: JSON.stringify(payload || {}),
      sourceHapp: 'fabrication',
      timestamp: Date.now(),
    };

    // Encode event as Uint8Array for bridge protocol
    const eventData = new TextEncoder().encode(JSON.stringify(event));

    // Create properly typed broadcast messages
    const marketplaceMsg = createBroadcastEvent('fabrication', eventType, eventData);
    const supplychainMsg = createBroadcastEvent('fabrication', eventType, eventData);

    // Send to target hApps
    this.bridge.send('marketplace', marketplaceMsg);
    this.bridge.send('supplychain', supplychainMsg);
  }

  /**
   * Get FL coordinator for reputation aggregation
   */
  getFLCoordinator(): FLCoordinator {
    return this.flCoordinator;
  }
}

// ============================================================================
// Exports
// ============================================================================

export {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  AggregationMethod,
};

// Default service instance
let defaultService: FabricationService | null = null;

/**
 * Get the default fabrication service instance
 */
export function getFabricationService(): FabricationService {
  if (!defaultService) {
    defaultService = new FabricationService();
  }
  return defaultService;
}
