/**
 * Mycelix Fabrication SDK Types
 *
 * Complete type definitions for the Fabrication hApp including:
 * - HDC (Hyperdimensional Computing) types
 * - Proof of Grounded Fabrication (PoGF) types
 * - Cincinnati Algorithm types
 * - Cross-hApp integration types
 */

import type { ActionHash, AgentPubKey, EntryHash, Record } from '@holochain/client';

// =============================================================================
// HDC (HYPERDIMENSIONAL COMPUTING) TYPES
// =============================================================================

export interface HdcHypervector {
  dimensions: number;
  vector: number[];
  semanticBindings: SemanticBinding[];
  generationMethod: HdcMethod;
}

export interface SemanticBinding {
  concept: string;
  role: BindingRole;
  weight: number;
}

export type BindingRole = 'Base' | 'Modifier' | 'Dimensional' | 'Material' | 'Functional';

export type HdcMethod =
  | 'ManualEncoding'
  | 'SymthaeaGenerated'
  | 'LateralBinding'
  | 'EvolutionarySearch';

// =============================================================================
// DESIGN TYPES
// =============================================================================

export interface Design {
  id: string;
  title: string;
  description: string;
  category: DesignCategory;
  intentVector?: HdcHypervector;
  parametricSchema?: ParametricSchema;
  constraintGraph?: ConstraintGraph;
  materialCompatibility: MaterialBinding[];
  files: DesignFile[];
  circularityScore: number;
  embodiedEnergyKwh: number;
  repairManifest?: RepairManifest;
  license: License;
  safetyClass: SafetyClass;
  epistemic: DesignEpistemic;
  author: AgentPubKey;
  createdAt: number;
  updatedAt: number;
}

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

export interface ParametricSchema {
  engine: ParametricEngine;
  sourceTemplate: string;
  parameters: DesignParameter[];
  constraints: ParametricConstraint[];
  autoGenerate: boolean;
}

export type ParametricEngine = 'OpenSCAD' | 'CadQuery' | 'FreeCAD' | 'OnShape' | 'Custom';

export interface DesignParameter {
  name: string;
  paramType: ParameterType;
  defaultValue: string | number | boolean;
  minValue?: number;
  maxValue?: number;
  unit?: string;
  hdcBinding?: string;
}

export type ParameterType = 'Length' | 'Angle' | 'Count' | 'Boolean' | 'Enum' | 'Material';

export interface ParametricConstraint {
  constraintType: string;
  expression: string;
  parameters: string[];
}

export interface ConstraintGraph {
  nodes: ConstraintNode[];
  edges: ConstraintEdge[];
}

export interface ConstraintNode {
  id: string;
  nodeType: string;
  value?: string | number;
}

export interface ConstraintEdge {
  from: string;
  to: string;
  edgeType: string;
}

export interface MaterialBinding {
  materialType: MaterialType;
  suitability: number;
  notes?: string;
}

export interface DesignFile {
  filename: string;
  format: FileFormat;
  ipfsCid: string;
  sizeBytes: number;
  checksumSha256: string;
}

export type FileFormat = 'STL' | 'STEP' | 'ThreeMF' | 'SCAD' | 'GCODE' | 'OBJ' | 'AMF';

export interface RepairManifest {
  parentProductHash?: ActionHash;
  parentProductModel: string;
  partName: string;
  failureModes: FailureMode[];
  replacementInterval?: number;
  repairDifficulty: RepairDifficulty;
}

export type FailureMode =
  | 'MechanicalWear'
  | 'UvDegradation'
  | 'ThermalCycling'
  | 'ChemicalExposure'
  | 'ImpactDamage'
  | 'Fatigue';

export type RepairDifficulty = 'Easy' | 'Moderate' | 'Difficult' | 'Expert';

export type License =
  | { type: 'PublicDomain' }
  | { type: 'CreativeCommons'; variant: CCVariant }
  | { type: 'OpenHardware' }
  | { type: 'Proprietary' }
  | { type: 'Custom'; text: string };

export type CCVariant = 'CC0' | 'BY' | 'BY_SA' | 'BY_NC' | 'BY_NC_SA' | 'BY_ND' | 'BY_NC_ND';

export type SafetyClass =
  | 'Class0Decorative'
  | 'Class1Functional'
  | 'Class2LoadBearing'
  | 'Class3BodyContact'
  | 'Class4Medical'
  | 'Class5Critical';

export interface DesignEpistemic {
  manufacturability: number;
  safety: number;
  usability: number;
}

export type ParameterValue = string | number | boolean;

export interface GeneratedVariant {
  designHash: ActionHash;
  parameters: { [key: string]: ParameterValue };
  outputFile?: string;
  generationStatus: string;
}

// =============================================================================
// PRINTER TYPES
// =============================================================================

export interface Printer {
  id: string;
  name: string;
  owner: AgentPubKey;
  location?: GeoLocation;
  printerType: PrinterType;
  capabilities: PrinterCapabilities;
  materialsAvailable: MaterialType[];
  availability: AvailabilityStatus;
  rates?: PrinterRates;
  createdAt: number;
  updatedAt: number;
}

export type PrinterType = 'FDM' | 'SLA' | 'SLS' | 'DLP' | 'MJF' | { Other: string };

export interface PrinterCapabilities {
  buildVolume: BuildVolume;
  layerHeights: number[];
  nozzleDiameters: number[];
  heatedBed: boolean;
  enclosure: boolean;
  multiMaterial?: number;
  maxTempHotend: number;
  maxTempBed: number;
  features: PrinterFeature[];
}

export interface BuildVolume {
  x: number;
  y: number;
  z: number;
}

export type PrinterFeature =
  | 'AutoLeveling'
  | 'FilamentRunout'
  | 'PowerRecovery'
  | 'CameraMonitoring'
  | 'RemoteControl';

export interface GeoLocation {
  geohash: string;
  /** Latitude in decimal degrees (-90 to 90). Optional; used for Haversine distance. */
  lat?: number;
  /** Longitude in decimal degrees (-180 to 180). Optional; used for Haversine distance. */
  lon?: number;
  city?: string;
  region?: string;
  country: string;
}

export type AvailabilityStatus =
  | 'Available'
  | 'Busy'
  | 'Maintenance'
  | 'Offline'
  | 'ByAppointment';

export interface PrinterRates {
  setupFee?: number;
  perHour?: number;
  perGram?: number;
  currency: string;
}

export interface PrinterMatch {
  printerHash: ActionHash;
  compatibilityScore: number;
  matchReasons: string[];
  estimatedTime?: number;
  estimatedCost?: number;
}

// =============================================================================
// PRINT JOB & RECORD TYPES
// =============================================================================

export interface PrintJob {
  id: string;
  designHash: ActionHash;
  printerHash: ActionHash;
  requester: AgentPubKey;
  settings: PrintSettings;
  status: PrintJobStatus;
  groundingCertificate?: GroundingCertificate;
  energySource?: EnergySource;
  materialPassport?: MaterialPassport;
  cincinnatiSession?: CincinnatiSession;
  qualityPredictions: QualityPrediction[];
  estimatedTimeMinutes?: number;
  actualTimeMinutes?: number;
  materialUsedGrams?: number;
  createdAt: number;
  startedAt?: number;
  completedAt?: number;
}

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

export interface TemperatureSettings {
  hotend: number;
  bed?: number;
  chamber?: number;
}

export type PrintJobStatus =
  | 'Pending'
  | 'Accepted'
  | 'Queued'
  | 'Printing'
  | 'PostProcessing'
  | 'Completed'
  | 'Failed'
  | 'Cancelled';

export interface PrintRecord {
  jobHash: ActionHash;
  result: PrintResult;
  qualityScore?: number;
  pogScore: number;
  energyUsedKwh: number;
  carbonOffsetKg?: number;
  materialCircularity: number;
  myceliumEarned: number;
  cincinnatiReport?: CincinnatiReport;
  dimensionalAccuracy?: DimensionalAccuracy;
  photos: string[];
  notes: string;
  issues: PrintIssue[];
  verification?: ActionHash;
  recordedAt: number;
}

export type PrintResult =
  | { type: 'Success' }
  | { type: 'PartialSuccess' }
  | { type: 'Failed'; reason: FailureReason };

export type FailureReason =
  | 'NozzleClog'
  | 'BedAdhesion'
  | 'LayerShift'
  | 'PowerFailure'
  | 'FilamentRunout'
  | 'ThermalRunaway'
  | 'UserCancelled'
  | { Other: string };

export type PrintIssue =
  | 'Warping'
  | 'LayerShift'
  | 'Stringing'
  | 'UnderExtrusion'
  | 'OverExtrusion'
  | 'BedAdhesion'
  | 'SupportFailure'
  | { Other: string };

export interface DimensionalAccuracy {
  xDeviation: number;
  yDeviation: number;
  zDeviation: number;
  overallScore: number;
}

// =============================================================================
// PROOF OF GROUNDED FABRICATION (PoGF) TYPES
// =============================================================================

export interface GroundingCertificate {
  certificateId: string;
  terraAtlasEnergyHash?: ActionHash;
  energyType: EnergyType;
  gridCarbonIntensity: number;
  materialPassports: MaterialPassport[];
  hearthFundingHash?: ActionHash;
  issuedAt: number;
  issuerSignature: Uint8Array;
}

export type EnergyType =
  | 'Solar'
  | 'Wind'
  | 'Hydro'
  | 'Geothermal'
  | 'Nuclear'
  | 'GridMix'
  | 'Unknown';

export type EnergySource = EnergyType;

export interface MaterialPassport {
  materialHash: ActionHash;
  batchId: string;
  origin: MaterialOrigin;
  recycledContentPercent: number;
  supplyChainHash?: ActionHash;
  certifications: string[];
  endOfLife: EndOfLifeStrategy;
}

export type MaterialOrigin =
  | 'Virgin'
  | 'PostIndustrial'
  | 'PostConsumer'
  | 'Biobased'
  | 'UrbanMined';

export type EndOfLifeStrategy =
  | 'MechanicalRecycling'
  | 'ChemicalRecycling'
  | 'Biodegradable'
  | 'IndustrialCompost'
  | 'Downcycle'
  | 'Landfill';

// =============================================================================
// CINCINNATI ALGORITHM TYPES
// =============================================================================

export interface CincinnatiSession {
  sessionId: string;
  estimatorVersion: string;
  samplingRateHz: number;
  baselineSignature: number[];
  active: boolean;
  startedAt: number;
}

export interface CincinnatiReport {
  sessionId: string;
  totalSamples: number;
  anomaliesDetected: number;
  anomalyEvents: AnomalyEvent[];
  overallHealthScore: number;
  layerByLayerScores: number[];
  recommendedAction: CincinnatiAction;
}

export interface AnomalyEvent {
  timestampMs: number;
  layerNumber: number;
  anomalyType: AnomalyType;
  severity: number;
  sensorData: SensorSnapshot;
  actionTaken?: CincinnatiAction;
}

export type AnomalyType =
  | 'ExtrusionInconsistency'
  | 'TemperatureDeviation'
  | 'VibrationAnomaly'
  | 'LayerAdhesionFailure'
  | 'NozzleClog'
  | 'BedLevelDrift'
  | 'PowerFluctuation'
  | 'Unknown';

export interface SensorSnapshot {
  hotendTemp: number;
  bedTemp: number;
  stepperCurrents: [number, number, number, number];
  vibrationRms: number;
  filamentTension?: number;
  ambientTemp?: number;
  humidity?: number;
}

export type CincinnatiAction =
  | { type: 'Continue' }
  | { type: 'AdjustParameters'; adjustment: ParameterAdjustment }
  | { type: 'PauseForInspection' }
  | { type: 'AbortPrint'; reason: string }
  | { type: 'AlertOperator'; message: string };

export interface ParameterAdjustment {
  parameter: string;
  oldValue: number;
  newValue: number;
  reason: string;
}

export interface QualityPrediction {
  layerNumber: number;
  predictedQuality: number;
  confidence: number;
  riskFactors: string[];
}

// =============================================================================
// MATERIAL TYPES
// =============================================================================

export interface Material {
  id: string;
  name: string;
  materialType: MaterialType;
  properties: MaterialProperties;
  certifications: Certification[];
  suppliers: ActionHash[];
  safetyDataSheet?: string;
  createdAt: number;
}

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

export interface Certification {
  certType: CertificationType;
  issuer: string;
  validUntil?: number;
  documentCid?: string;
}

export type CertificationType =
  | 'FoodSafe'
  | 'Biocompatible'
  | 'FlameRetardant'
  | 'RoHSCompliant'
  | 'REACHCompliant'
  | 'ISO'
  | { Custom: string };

// =============================================================================
// VERIFICATION TYPES
// =============================================================================

export interface DesignVerification {
  designHash: ActionHash;
  verificationType: VerificationType;
  result: VerificationResult;
  evidence: ActionHash[];
  verifier: AgentPubKey;
  verifierCredentials: string[];
  createdAt: number;
}

export type VerificationType =
  | 'StructuralAnalysis'
  | 'MaterialCompatibility'
  | 'PrintabilityTest'
  | 'SafetyReview'
  | 'FoodSafeCertification'
  | 'MedicalCertification'
  | 'CommunityReview';

export type VerificationResult =
  | { type: 'Passed'; confidence: number; notes: string }
  | { type: 'Failed'; reasons: string[] }
  | { type: 'ConditionalPass'; conditions: string[]; confidence: number }
  | { type: 'NeedsMoreEvidence' };

export interface SafetyClaim {
  designHash: ActionHash;
  claimType: SafetyClaimType;
  claimText: string;
  epistemic: ClaimEpistemic;
  supportingEvidence: string[];
  knowledgeClaimHash?: ActionHash;
  author: AgentPubKey;
  createdAt: number;
}

export type SafetyClaimType =
  | { type: 'LoadCapacity'; value: string }
  | { type: 'MaterialSafety'; value: string }
  | { type: 'DimensionalAccuracy'; value: string }
  | { type: 'TemperatureRange'; value: string }
  | { type: 'ChemicalResistance'; value: string }
  | { type: 'Custom'; value: string };

export interface ClaimEpistemic {
  empirical: number;
  normative: number;
  mythic: number;
}

export interface EpistemicScore {
  empirical: number;
  normative: number;
  mythic: number;
  overallConfidence: number;
}

export interface VerificationSummary {
  designHash: ActionHash;
  totalVerifications: number;
  passed: number;
  failed: number;
  claimsCount: number;
  averageConfidence: number;
}

// =============================================================================
// BRIDGE / INTEGRATION TYPES
// =============================================================================

export interface RepairPrediction {
  propertyAssetHash: ActionHash;
  assetModel: string;
  predictedFailureComponent: string;
  failureProbability: number;
  estimatedFailureDate: number;
  confidenceIntervalDays: number;
  sensorDataSummary: string;
  recommendedAction: RepairAction;
  createdAt: number;
}

export type RepairAction =
  | 'OrderPart'
  | 'PrintReplacement'
  | 'CreateDesign'
  | 'ScheduleMaintenance'
  | 'Monitor';

export interface RepairWorkflow {
  predictionHash: ActionHash;
  status: RepairWorkflowStatus;
  designHash?: ActionHash;
  printerHash?: ActionHash;
  hearthFundingHash?: ActionHash;
  printJobHash?: ActionHash;
  propertyInstallationHash?: ActionHash;
  createdAt: number;
  completedAt?: number;
}

export type RepairWorkflowStatus =
  | 'Predicted'
  | 'DesignFound'
  | 'PrinterMatched'
  | 'FundingApproved'
  | 'Printing'
  | 'ReadyForInstall'
  | 'Installed'
  | 'Cancelled';

export interface MarketplaceListing {
  designHash: ActionHash;
  marketplaceListingHash?: ActionHash;
  price?: number;
  listingType: ListingType;
  createdAt: number;
}

export type ListingType = 'DesignSale' | 'PrintService' | 'PrintedProduct';

export interface SupplyChainLink {
  materialHash: ActionHash;
  supplyChainItemHash?: ActionHash;
  supplierDid: string;
  createdAt: number;
}

// =============================================================================
// SYMTHAEA / HDC OPERATION TYPES
// =============================================================================

export interface IntentResult {
  record: Record;
  bindings: SemanticBinding[];
  vectorHash: string;
}

export interface SearchResult {
  designHash: ActionHash;
  similarityScore: number;
  matchedBindings: string[];
}

export interface RepairPredictionResult {
  predictedComponent: string;
  failureProbability: number;
  estimatedRemainingHours: number;
  recommendedAction: string;
  matchingRepairDesigns: ActionHash[];
}

export interface SensorReading {
  timestamp: number;
  sensorType: string;
  value: number;
  unit: string;
}

// =============================================================================
// SDK INPUT TYPES
// =============================================================================

export interface CreateDesignInput {
  title: string;
  description: string;
  category: DesignCategory;
  files: DesignFile[];
  license: License;
  safetyClass: SafetyClass;
  parametricSchema?: ParametricSchema;
  repairManifest?: RepairManifest;
}

export interface UpdateDesignInput {
  designHash: ActionHash;
  title?: string;
  description?: string;
  files?: DesignFile[];
  safetyClass?: SafetyClass;
}

export interface RegisterPrinterInput {
  name: string;
  printerType: PrinterType;
  capabilities: PrinterCapabilities;
  materialsAvailable: MaterialType[];
  location?: GeoLocation;
  rates?: PrinterRates;
}

export interface CreatePrintJobInput {
  designHash: ActionHash;
  printerHash: ActionHash;
  settings: PrintSettings;
  groundingCertificate?: GroundingCertificate;
}

export interface RecordPrintInput {
  jobHash: ActionHash;
  result: PrintResult;
  qualityScore?: number;
  energyUsedKwh: number;
  photos?: string[];
  notes?: string;
  issues?: PrintIssue[];
}

export interface SubmitVerificationInput {
  designHash: ActionHash;
  verificationType: VerificationType;
  result: VerificationResult;
  evidence: ActionHash[];
  credentials: string[];
}

export interface SubmitClaimInput {
  designHash: ActionHash;
  claimType: SafetyClaimType;
  claimText: string;
  supportingEvidence: string[];
}

export interface DesignSearchQuery {
  text?: string;
  category?: DesignCategory;
  safetyClass?: SafetyClass;
  materialCompatible?: MaterialType[];
  limit?: number;
}

export interface PrintStatistics {
  designHash: ActionHash;
  totalPrints: number;
  successRate: number;
  averageQuality: number;
  averagePogScore: number;
  totalMyceliumEarned: number;
}
