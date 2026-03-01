/**
 * @mycelix/fabrication-sdk
 *
 * TypeScript SDK for the Mycelix Fabrication hApp
 * Distributed Manufacturing Commons for the Civilizational OS
 *
 * Key Features:
 * - HDC-Encoded Parametric Designs (Generative CAD Commons)
 * - Proof of Grounded Fabrication (PoGF) scoring
 * - Cincinnati Algorithm quality monitoring
 * - Anticipatory Repair Loop integration
 * - Cross-hApp bridges (Knowledge, Marketplace, Supply Chain)
 *
 * @example
 * ```typescript
 * import { FabricationClient } from '@mycelix/fabrication-sdk';
 *
 * const client = new FabricationClient({
 *   url: 'ws://localhost:8888',
 *   installedAppId: 'fabrication'
 * });
 *
 * await client.connect();
 *
 * // Search for designs using natural language
 * const results = await client.searchDesigns('bracket for 12mm pipe');
 *
 * // Create a design
 * const design = await client.designs.create({
 *   title: 'Pipe Bracket 12mm',
 *   description: 'Universal bracket for 12mm pipes',
 *   category: 'Parts',
 *   files: [...],
 *   license: { type: 'CreativeCommons', variant: 'BY_SA' },
 *   safetyClass: 'Class1Functional'
 * });
 *
 * // Find local printers
 * const matches = await client.printers.matchDesign(designHash);
 *
 * // Create a print job with PoGF
 * const job = await client.prints.createJob({
 *   designHash,
 *   printerHash,
 *   settings: { layerHeight: 0.2, infillPercent: 20, material: 'PETG' },
 *   groundingCertificate: { ... }
 * });
 *
 * // Calculate PoGF score
 * const pogScore = client.calculatePogScore({
 *   energyRenewableFraction: 0.8,
 *   materialCircularity: 0.6,
 *   qualityVerified: 0.9,
 *   localParticipation: 1.0
 * });
 *
 * await client.disconnect();
 * ```
 */

// Main client
export { FabricationClient } from './client';
export type { FabricationConfig } from './client';

// Sub-clients
export { DesignsClient } from './clients/designs';
export { PrintersClient } from './clients/printers';
export type {
  PrinterRequirements,
  CompatibilityResult,
  UpdatePrinterInput,
  FindNearbyInput,
  FindByCapabilityInput,
  MatchDesignInput,
  CheckCompatibilityInput,
  UpdateAvailabilityInput,
} from './clients/printers';
export { PrintsClient } from './clients/prints';
export type {
  UpdateProgressInput,
  CompletePrintInput,
  CancelPrintInput,
  AddPhotosInput,
  StartCincinnatiInput,
  ReportAnomalyInput,
  UpdateCincinnatiInput,
  DesignPrintStats,
} from './clients/prints';
export { MaterialsClient } from './clients/materials';
export type { CreateMaterialInput } from './clients/materials';
export { VerificationClient } from './clients/verification';
export { BridgeClient } from './clients/bridge';
export type {
  CreateRepairPredictionInput,
  UpdateWorkflowInput,
  EmitEventInput,
  MarketplaceListingInput,
  AuditTrailFilter,
} from './clients/bridge';
export { SymthaeaClient } from './clients/symthaea';
export type {
  CreateIntentInput,
  LateralBindInput,
  SemanticSearchInput,
  GenerateVariantInput,
  OptimizeLocalInput,
  PredictRepairInput,
} from './clients/symthaea';

// Types
export type {
  // HDC Types
  HdcHypervector,
  SemanticBinding,
  BindingRole,
  HdcMethod,

  // Design Types
  Design,
  DesignCategory,
  ParametricSchema,
  ParametricEngine,
  DesignParameter,
  ParameterType,
  ParametricConstraint,
  ConstraintGraph,
  ConstraintNode,
  ConstraintEdge,
  MaterialBinding,
  DesignFile,
  FileFormat,
  RepairManifest,
  FailureMode,
  RepairDifficulty,
  License,
  CCVariant,
  SafetyClass,
  DesignEpistemic,

  // Printer Types
  Printer,
  PrinterType,
  PrinterCapabilities,
  BuildVolume,
  PrinterFeature,
  GeoLocation,
  AvailabilityStatus,
  PrinterRates,
  PrinterMatch,

  // Print Types
  PrintJob,
  PrintSettings,
  TemperatureSettings,
  PrintJobStatus,
  PrintRecord,
  PrintResult,
  FailureReason,
  PrintIssue,
  DimensionalAccuracy,

  // PoGF Types
  GroundingCertificate,
  EnergyType,
  EnergySource,
  MaterialPassport,
  MaterialOrigin,
  EndOfLifeStrategy,

  // Cincinnati Types
  CincinnatiSession,
  CincinnatiReport,
  AnomalyEvent,
  AnomalyType,
  SensorSnapshot,
  CincinnatiAction,
  ParameterAdjustment,
  QualityPrediction,

  // Material Types
  Material,
  MaterialType,
  MaterialProperties,
  Certification,
  CertificationType,

  // Verification Types
  DesignVerification,
  VerificationType,
  VerificationResult,
  SafetyClaim,
  SafetyClaimType,
  ClaimEpistemic,
  EpistemicScore,
  VerificationSummary,

  // Bridge Types
  RepairPrediction,
  RepairAction,
  RepairWorkflow,
  RepairWorkflowStatus,
  MarketplaceListing,
  ListingType,
  SupplyChainLink,

  // Symthaea Types
  IntentResult,
  SearchResult,
  RepairPredictionResult,
  SensorReading,

  // Input Types
  CreateDesignInput,
  UpdateDesignInput,
  RegisterPrinterInput,
  CreatePrintJobInput,
  RecordPrintInput,
  SubmitVerificationInput,
  SubmitClaimInput,
  DesignSearchQuery,
  PrintStatistics,
} from './types';
