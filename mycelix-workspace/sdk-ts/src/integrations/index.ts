/**
 * @mycelix/sdk hApp Integrations
 *
 * Pre-built integration modules for all Civilizational OS domains.
 * Each integration provides domain-specific trust logic, type definitions,
 * and bridge clients for interacting with the corresponding hApp.
 *
 * @packageDocumentation
 * @module integrations
 */

// ============================================================================
// Cluster Integrations (Personal + Identity + Commons + Civic)
// ============================================================================

export {
  PersonalBridgeClient,
  createPersonalBridgeClient,
  isPersonalBridgeSignal,
  PERSONAL_DOMAINS,
  PERSONAL_ZOMES,
} from './personal/index.js';

export type {
  PersonalQueryInput,
  PersonalEventInput,
  PersonalBridgeEventSignal,
  PersonalBridgeSignalHandler,
  CrossClusterDispatchInput as PersonalCrossClusterInput,
  CredentialType,
  DisclosureScope,
  CredentialPresentation,
  SubmitPhiAttestationInput,
  QueryIdentityInput,
  CreatePresentationInput,
} from './personal/index.js';


export {
  CommonsBridgeClient,
  createCommonsBridgeClient,
  isCommonsBridgeSignal,
  COMMONS_DOMAINS,
  COMMONS_ZOMES,
} from './commons/index.js';

export type {
  CommonsQueryInput,
  CommonsEventInput,
  CommonsBridgeEventSignal,
  BridgeSignalHandler as CommonsBridgeSignalHandler,
  CrossClusterDispatchInput as CommonsCrossClusterInput,
  CheckEmergencyForAreaInput,
  EmergencyAreaCheckResult,
  CheckJusticeDisputesInput,
  JusticeDisputeCheckResult,
  PropertyOwnershipQuery,
  PropertyOwnershipResult,
  CareAvailabilityQuery,
  CareAvailabilityResult,
} from './commons/index.js';

export {
  CivicBridgeClient,
  createCivicBridgeClient,
  isCivicBridgeSignal,
  CIVIC_DOMAINS,
  CIVIC_ZOMES,
} from './civic/index.js';

export {
  HearthBridgeClient,
  createHearthBridgeClient,
  isHearthBridgeSignal,
  HEARTH_DOMAINS,
  HEARTH_ZOMES,
} from './hearth/index.js';

export type {
  HearthQueryInput,
  HearthEventInput,
  HearthBridgeEventSignal,
  HearthBridgeSignalHandler,
  SeveranceInput,
  CrossClusterDispatchInput as HearthCrossClusterInput,
} from './hearth/index.js';

export type {
  CivicQueryInput,
  CivicEventInput,
  CivicBridgeEventSignal,
  CivicBridgeSignalHandler,
  BridgeHealth,
  DispatchInput,
  DispatchResult,
  EventTypeQuery,
  CrossClusterDispatchInput as CivicCrossClusterInput,
  QueryPropertyForEnforcementInput,
  PropertyEnforcementResult,
  CheckHousingCapacityInput,
  HousingCapacityResult,
  VerifyCareCredentialsInput,
  CareCredentialVerifyResult,
  JusticeAreaQuery,
  JusticeAreaResult,
  FactcheckStatusQuery,
  FactcheckStatusResult,
  AuditTrailQuery,
  AuditTrailEntry,
  AuditTrailResult,
} from './civic/index.js';

// ============================================================================
// Community Support Integration
// ============================================================================

export { SupportClient, createSupportClient, SUPPORT_ZOMES } from './support/index.js';

export type {
  SupportCategory,
  TicketPriority,
  TicketStatus,
  AutonomyLevel,
  ActionType,
  SharingTier,
  DiagnosticType,
  DiagnosticSeverity,
  DifficultyLevel,
  ArticleSource,
  FlagReason,
  EscalationLevel,
  EpistemicStatus,
  LinkReason,
  KnowledgeArticle,
  UpdateArticleInput,
  DeprecateInput,
  ArticleFlag,
  Resolution as SupportResolution,
  LinkArticleInput,
  SupportTicket,
  UpdateTicketInput,
  TicketComment,
  AutonomousAction,
  UndoAction,
  PreemptiveAlert,
  PromoteAlertInput,
  EscalateInput,
  SatisfactionSurvey,
  DiagnosticResult,
  PrivacyPreference,
  HelperProfile,
  UpdateAvailInput,
  CognitiveUpdate,
} from './support/index.js';

// ============================================================================
// Food Sovereignty Integration
// ============================================================================

export { FoodClient, createFoodClient, FOOD_ZOMES } from './food/index.js';

export type {
  // Production types
  PlotStatus,
  Plot,
  CropStatus,
  Crop,
  QualityGrade,
  YieldRecord,
  SeasonPlan,
  // Distribution types
  MarketType,
  OrderStatus,
  Market as FoodMarket,
  Listing,
  Order as FoodOrder,
  // Preservation types
  PreservationStatus,
  StorageType,
  SkillLevel,
  PreservationBatch,
  PreservationMethod,
  StorageUnit,
  // Knowledge types
  PracticeCategory,
  SeedVariety,
  TraditionalPractice,
  Recipe,
  // Seed exchange (Phase 2)
  SeedStock,
  SeedRequestStatus,
  FoodSeedRequest,
  MatchSeedInput,
  // Nutrient tracking (Phase 2)
  NutrientProfile,
  // Allergen search (Phase 2)
  AllergenSearchInput,
  // Garden membership (Phase 2)
  GardenRole,
  GardenMembership,
  AddMemberInput,
  RemoveMemberInput,
  // Listing status (Phase 2)
  ListingStatus,
  // Update inputs (Phase 2)
  UpdateSeedVarietyInput,
  UpdateTraditionalPracticeInput,
  UpdateRecipeInput,
} from './food/index.js';

// ============================================================================
// Transport/Mobility Integration
// ============================================================================

export { TransportClient, createTransportClient, TRANSPORT_ZOMES } from './transport/index.js';

export type {
  // Route types
  VehicleType,
  VehicleStatus,
  Vehicle,
  TransportMode,
  Waypoint,
  Route,
  StopType,
  Stop,
  // Sharing types
  RideOfferStatus,
  RideRequestStatus,
  RideMatchStatus,
  MatchStatus,
  UpdateMatchStatusInput,
  RideOffer,
  RideRequest,
  RideMatch,
  CargoOffer,
  // Impact types
  TripMode,
  CreditSource,
  TripLog,
  CarbonCredit,
  EmissionsCalcInput,
  EmissionsCalcResult,
  CommunityImpactSummary,
  // Maintenance (Phase 3)
  MaintenanceType,
  MaintenanceRecord,
  VehicleFeatures,
  // Reviews (Phase 2)
  ReviewerRole,
  RideReview,
  FindNearbyInput,
  // Carbon redemption (Phase 2)
  RedeemInput,
  CreditRedemption,
  CarbonBalance,
} from './transport/index.js';

// ============================================================================
// Mail Integration
// ============================================================================

export { MailTrustService, getMailTrustService } from './mail/index.js';

// ============================================================================
// Marketplace Integration
// ============================================================================

export { MarketplaceReputationService, getMarketplaceService } from './marketplace/index.js';

// ============================================================================
// EduNet Integration
// ============================================================================

export { EduNetCredentialService, getEduNetService } from './edunet/index.js';

// ============================================================================
// SupplyChain Integration
// ============================================================================

export { SupplyChainProvenanceService, getSupplyChainService } from './supplychain/index.js';

// ============================================================================
// Identity Integration
// ============================================================================

export {
  IdentityBridgeClient,
  getIdentityBridgeClient,
  MycelixIdentityClient,
  IdentityClient,
  // MFA exports
  MfaClient,
} from './identity/index.js';

// Re-export MFA types for convenience
export type {
  FactorType,
  FactorCategory,
  AssuranceLevel,
  EnrolledFactor,
  MfaState,
  EnrollFactorInput,
  FactorProof,
  GuardianAttestation,
  VerificationChallenge,
  MfaVerificationResult,
  FlEligibilityResult,
  FlRequirement,
  FactorEnrollment,
} from './identity/index.js';

// ============================================================================
// Finance Integration
// ============================================================================

export { FinanceBridgeClient, getFinanceBridgeClient } from './finance/index.js';

// ============================================================================
// Property Integration
// ============================================================================

export { PropertyBridgeClient, getPropertyBridgeClient } from './property/index.js';

// ============================================================================
// Energy Integration
// ============================================================================

export { EnergyBridgeClient, getEnergyBridgeClient } from './energy/index.js';

// ============================================================================
// Media Integration
// ============================================================================

export { MediaBridgeClient, getMediaBridgeClient } from './media/index.js';

// ============================================================================
// Governance Integration
// ============================================================================

export { GovernanceBridgeClient, getGovernanceBridgeClient } from './governance/index.js';

// ============================================================================
// Justice Integration
// ============================================================================

export { JusticeBridgeClient, getJusticeBridgeClient } from './justice/index.js';

// ============================================================================
// Knowledge Integration
// ============================================================================

export { KnowledgeBridgeClient, getKnowledgeBridgeClient } from './knowledge/index.js';

// ============================================================================
// Fabrication Integration
// ============================================================================

export { FabricationService, getFabricationService } from './fabrication/index.js';

// Re-export fabrication types for convenience
export type {
  // Design types
  Design,
  DesignCategory,
  DesignFile,
  DesignEpistemic,
  SafetyClass,
  License as FabricationLicense,
  FileFormat,
  ParametricSchema,
  ParametricEngine,
  DesignParameter,
  ParameterType,
  RepairManifest,
  FailureMode,
  // Printer types
  Printer,
  PrinterType,
  PrinterCapabilities,
  PrinterRates,
  PrinterMatch,
  BuildVolume,
  GeoLocation,
  AvailabilityStatus,
  // Print job types
  PrintJob,
  PrintJobStatus,
  PrintRecord,
  PrintResult,
  PrintSettings,
  PrintIssue,
  TemperatureSettings,
  QualityAssessment,
  // PoGF types
  GroundingRequest,
  GroundingCertificate,
  MaterialPassport,
  MaterialOrigin,
  EndOfLifeStrategy,
  EnergyType,
  // Cincinnati types
  CincinnatiReport,
  AnomalyEvent,
  AnomalyType,
  CincinnatiAction,
  SensorSnapshot,
  // Material types
  Material,
  MaterialType,
  MaterialProperties,
  MaterialCertification,
  CertificationType,
  // Verification types
  DesignVerification,
  VerificationType,
  VerificationResult,
  // Bridge types
  RepairPrediction,
  RepairWorkflow,
  RepairAction,
  RepairWorkflowStatus,
  FabricationEvent,
  FabricationEventType,
  MarketplaceListing,
  ListingType,
} from './fabrication/index.js';

// ============================================================================
// Epistemic Markets Integration
// ============================================================================

export { EpistemicMarketsService, getEpistemicMarketsService } from './epistemic-markets/index.js';

// Re-export epistemic markets types for convenience
export type {
  // Market types
  Market,
  MarketStatus,
  CreateMarketInput,
  Outcome as MarketOutcome,
  Order,
  Visibility,
  // Epistemic types
  EpistemicPosition,
  ResolutionMechanism,
  MarketMechanism,
  // Stake types
  MultiDimensionalStake,
  MonetaryStake,
  ReputationStake,
  SocialStake,
  CommitmentStake,
  TimeStake,
  Commitment,
  // Prediction types
  Prediction,
  SubmitPredictionInput,
  PredictionUpdate,
  PredictionResolution,
  BeliefDependency,
  TemporalCommitment,
  // Reasoning types
  ReasoningTrace,
  ReasoningStep,
  Assumption,
  UpdateTrigger,
  AlternativeFraming,
  InformationSource,
  ConfidenceBreakdown,
  StepSupport,
  // Wisdom types
  WisdomSeed,
  WisdomLesson,
  LessonApplicability,
  ActivationCondition,
} from './epistemic-markets/index.js';

// Re-export epistemic enums and functions
export {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  calculateStakeValue,
  getRecommendedResolution,
  getRecommendedDurationDays,
  toClassificationCode,
  createReputationOnlyStake,
  createMonetaryOnlyStake,
} from './epistemic-markets/index.js';

// ============================================================================
// Music Integration
// ============================================================================

export { MusicRoyaltyService, getMusicService } from './music/index.js';

// Re-export music types for convenience
export type {
  Track,
  TrackArtist,
  PlayRecord,
  RoyaltyDistribution,
  RoyaltyPayment,
  ArtistProfile,
  CollaborationAgreement,
} from './music/index.js';

// ============================================================================
// DeSci (Decentralized Science) Integration
// ============================================================================

export { DeSciService, getDeSciService } from './desci/index.js';

// Re-export DeSci types for convenience
export type {
  Researcher,
  Publication,
  PeerReview,
  ResearchGrant,
  GrantMilestone,
  ResearcherProfile,
  Collaboration,
} from './desci/index.js';

// ============================================================================
// Consensus Integration
// ============================================================================

export { ConsensusService, getConsensusService } from './consensus/index.js';

// Re-export consensus types for convenience
export type {
  ConsensusTopic,
  Vote,
  ConsensusRound,
  ConsensusResult,
  ConsensusParticipant,
  SignatureShard,
  ThresholdSignature,
} from './consensus/index.js';

// ============================================================================
// Health Integration
// ============================================================================

export { HealthService } from './health/index.js';

// Re-export health types for convenience
export type {
  // Patient types
  Patient,
  BiologicalSex,
  BloodType,
  ContactInfo,
  EmergencyContact,
  Allergy,
  AllergySeverity,
  // Provider types
  Provider,
  ProviderType,
  EpistemicLevel,
  PracticeLocation,
  ProviderContact,
  License,
  LicenseType,
  LicenseStatus,
  // Medical records types
  Encounter,
  EncounterType,
  EncounterStatus,
  RecordsEpistemicLevel,
  Diagnosis,
  DiagnosisType,
  DiagnosisStatus,
  DiagnosisSeverity,
  ProcedurePerformed,
  ProcedureOutcome,
  LabResult,
  LabInterpretation,
  VitalSigns,
  // Prescription types
  Prescription,
  MedicationForm,
  AdministrationRoute,
  PrescriptionStatus,
  DrugSchedule,
  // Consent types
  Consent,
  ConsentGrantee,
  ConsentScope,
  DataCategory,
  DataPermission,
  ConsentPurpose,
  ConsentStatus,
  // Clinical trials types
  ClinicalTrial,
  TrialPhase,
  StudyType,
  TrialStatus,
  TrialEpistemicLevel,
  EligibilityCriteria,
  Intervention,
  InterventionType,
  Outcome,
  TrialParticipant,
  ParticipantStatus,
  AdverseEvent,
  AESeverity,
  SeriousnessCriteria,
  Causality,
  AEOutcome,
  ActionTaken,
} from './health/index.js';

// ============================================================================
// Health-Energy Integration (Cross-Domain Bridge)
// ============================================================================

export {
  HealthEnergyBridge,
  MedicalDeviceClient,
  MedicalPriorityClient,
  HealthcareFacilityClient,
  createHealthEnergyBridge,
  calculatePriorityScore,
  estimateRestorationTime,
  requiresImmediateResponse,
} from './health-energy/index.js';

// Re-export health-energy types for convenience
export type {
  // Medical device types
  MedicalDevice,
  MedicalDeviceCategory,
  // Priority types
  PowerPriority,
  MedicalPriorityAlert,
  // Facility types
  HealthcareFacility,
  // Event types
  PowerOutageEvent,
  FacilityOutageResponse,
} from './health-energy/index.js';

// ============================================================================
// Health-Food Integration (Cross-Domain Bridge)
// ============================================================================

export {
  HealthFoodBridge,
  DietaryRestrictionClient,
  DrugFoodInteractionClient,
  NutritionTrackingClient,
  FoodSecurityClient,
  NutritionRecommendationClient,
  createHealthFoodBridge,
  mapAllergySeverity,
  getAllergenCategories,
  calculateNutritionScore,
  checkConditionSafety,
} from './health-food/index.js';

// ============================================================================
// Food-Shelter Integration (Cross-Domain Bridge)
// ============================================================================

export {
  FoodShelterBridge,
  createFoodShelterBridge,
  isInCrisis,
  householdSize,
  qualifiesForEmergencyAssistance,
} from './food-shelter/index.js';

export type {
  // Core types
  HouseholdFoodSecurityLevel,
  HousingSecurityStatus,
  ResourceCategory,
  HouseholdType,
  // Household types
  Household,
  ResourceAllocation,
  AllocationEntry,
  AllocationWarning,
  // Security types
  HouseholdSecurityAssessment,
  SecurityRiskFactor,
  SupportProgram,
  SecurityRecommendation,
  // Community types
  CommunityResourceHub,
  CommunityFoodResource,
  CommunityShelterResource,
  // Reputation types
  FoodShelterReputation,
  DomainReputation,
  ReputationContribution,
  // Input types
  CreateHouseholdInput,
  FoodShelterBridgeConfig,
} from './food-shelter/index.js';

// ============================================================================
// Water-Energy Integration (Cross-Domain Bridge)
// ============================================================================

export {
  WaterEnergyBridge,
  createWaterEnergyBridge,
  calculatePumpingEnergy,
  calculateHydroGenerationPotential,
  calculateDesalinationEnergy,
  shouldParticipateInDemandResponse,
} from './water-energy/index.js';

export type {
  // Water types
  WaterSourceType,
  WaterQuality,
  WaterSystemStatus,
  WaterFacilityType,
  // Energy types
  EnergySourceType,
  GridLoadCategory,
  PricingPeriod,
  // Facility types
  WaterEnergyFacility,
  GeoLocation as WaterEnergyGeoLocation,
  EnergyConnection,
  // Consumption types
  WaterEnergyConsumption,
  EnergySourceBreakdown,
  ConsumptionStats,
  // Demand response types
  DemandResponseEvent,
  DemandResponseType,
  DemandResponseStatus,
  DemandResponseCapacity,
  // Optimization types
  OptimizationSchedule,
  HourlyTarget,
  ScheduleStatus,
  // Hydro types
  HydroGenerationRecord,
  // Alert types
  WaterEnergyAlert,
  AlertType as WaterEnergyAlertType,
  AlertSeverity as WaterEnergyAlertSeverity,
  AlertMetrics,
  // Reputation types
  WaterEnergyReputation,
  SustainabilityRating,
  // Config types
  WaterEnergyBridgeConfig,
  PricingSchedule,
} from './water-energy/index.js';

// Re-export health-food types for convenience
export type {
  // Restriction types
  DietaryRestriction,
  DietaryRestrictionType,
  RestrictionSeverity,
  FoodCategory,
  // Interaction types
  DrugFoodInteraction,
  // Nutrition types
  NutritionGoal,
  MealLog,
  MealItem,
  NutritionRecommendation,
  // SDOH types
  FoodSecurityStatus,
} from './health-food/index.js';

// ============================================================================
// Genetics Integration (HDC Privacy-Preserving Genetic Similarity)
// ============================================================================

export {
  GeneticsBridge,
  GeneticEncodingClient,
  GeneticSimilarityClient,
  GeneticBundlingClient,
  GeneticCodebookClient,
  createGeneticsBridge,
  interpretHlaSimilarity,
  calculateEncodingConfidence,
} from './genetics/index.js';

// Re-export genetics types for convenience
export type {
  // Hypervector types
  GeneticHypervector,
  GeneticEncodingType,
  // Similarity types
  GeneticSimilarityResult,
  SimilarityMetric,
  SimilaritySearchResult,
  // Codebook types
  GeneticCodebook,
  // Bundle types
  BundledGeneticVector,
  // Encoding input types
  EncodeDnaSequenceInput,
  EncodeSnpPanelInput,
  EncodeHlaTypingInput,
  BatchSimilaritySearchInput,
  // Source types
  GeneticSourceMetadata,
} from './genetics/index.js';

// ============================================================================
// Health-Governance Integration (Cross-Domain Bridge)
// ============================================================================

export {
  HealthGovernanceBridge,
  HealthPolicyClient,
  TrialOversightClient,
  PandemicResponseClient,
  DataPrivacyPolicyClient,
  createHealthGovernanceBridge,
  calculateVoterWeight,
  evaluateProposalOutcome,
  detectConflictOfInterest,
} from './health-governance/index.js';

// Re-export health-governance types for convenience
export type {
  // Policy types
  HealthPolicyCategory,
  HealthVoterType,
  HealthPolicyProposal,
  HealthPolicyVote,
  // Trial oversight types
  TrialOversight,
  TrialParticipantVoice,
  // Credentialing types
  CredentialingDecision,
  // Pandemic types
  PandemicResponseDecision,
  // Privacy policy types
  DataPrivacyPolicy,
} from './health-governance/index.js';

// ============================================================================
// Health-Marketplace Integration (Cross-Domain Bridge)
// ============================================================================

export {
  HealthMarketplaceBridge,
  PrescriptionMarketplaceClient,
  HealthProductClient,
  DMERentalClient,
  createHealthMarketplaceBridge,
  daysUntilRefillable,
  isPrescriptionExpiringSoon,
  calculatePotentialSavings,
  requiresPrescriptionVerification,
  formatDirections,
} from './health-marketplace/index.js';

// Re-export health-marketplace types for convenience
export type {
  // Product types
  HealthProductCategory,
  HealthProduct,
  ProductVerification,
  // Prescription types
  MarketplacePrescription,
  PrescriptionStatus as MarketplacePrescriptionStatus,
  PrescriptionPriceQuote,
  PrescriptionOrder,
  // Pharmacy types
  Pharmacy,
  // Order types
  HealthProductOrder,
  FulfillmentStatus,
  PaymentMethod,
  // Coverage types
  CoverageCheckResult,
} from './health-marketplace/index.js';

// ============================================================================
// Climate Integration
// ============================================================================

export { ClimateService, getClimateService, resetClimateService } from './climate/index.js';

// Re-export climate types for convenience
export type {
  ClimateProjectType,
  VerificationStandard,
  CreditUnit,
  GeoLocation as ClimateGeoLocation,
  ClimateProject,
  CarbonCreditBatch,
  RenewableEnergyCertificate,
  CarbonOffsetSettlement,
} from './climate/index.js';

// ============================================================================
// Mutual Aid Integration
// ============================================================================

export { MutualAidService, getMutualAidService, resetMutualAidService } from './mutualaid/index.js';

// Re-export mutual aid types for convenience
export type {
  NeedCategory,
  GiftCircle,
  NeedRequest,
  TimebankEntry,
  ResourceContribution,
  MemberSummary,
} from './mutualaid/index.js';

// ============================================================================
// Academic Credentials Integration (Legacy Bridge)
// ============================================================================

export { AcademicCredentialService, getAcademicService, resetAcademicService } from './academic/index.js';

// Re-export academic types for convenience
export type {
  DegreeType,
  DnssecStatus,
  RevocationReason,
  RevocationRequestStatus,
  ImportStatus,
  InstitutionLocation,
  Accreditation,
  InstitutionalIssuer,
  AcademicSubject,
  AcademicProof,
  AchievementMetadata,
  DnsVerificationRecord,
  DnsDid,
  AcademicCredential,
  CredentialRevocation,
  AcademicRevocationRequest,
  LegacyBridgeImport,
  ImportError,
  CsvFieldMapping,
  CredentialVerification,
} from './academic/index.js';

// ============================================================================
// Attribution Registry Integration (Open-Source Dependency Attribution)
// ============================================================================

export {
  AttributionClient,
  createAttributionClient,
  resetAttributionClient,
  isAttributionSignal,
} from './attribution/index.js';

export type {
  DependencyIdentity,
  DependencyEcosystem,
  UpdateDependencyInput,
  UsageReceipt,
  UsageType,
  UsageScale,
  UsageAttestation,
  VerifyAttestationInput,
  ReciprocityPledge,
  PledgeType,
  Currency,
  StewardshipScore,
  RegistrySignal,
  UsageSignal,
  ReciprocitySignal,
  AttributionSignal,
  PaginatedResult,
  PaginationInput,
  TopDependency,
  BulkRegisterResult,
  BulkUsageResult,
  LeaderboardEntry,
  EcosystemStat,
  EcosystemStatistics,
} from './attribution/index.js';

// ============================================================================
// Health-FHIR Integration (Interoperability)
// ============================================================================

export {
  FHIRBridge,
  FHIRExportClient,
  FHIRImportClient,
  createFHIRBridge,
  toFHIRDateTime,
  fromFHIRDateTime,
  createVitalsCoding,
  createSNOMEDCoding,
  createRxNormCoding,
  createICD10Coding,
  validateRequiredFields,
} from './health-fhir/index.js';

// Re-export FHIR types for convenience
export type {
  // Base FHIR types
  FHIRResource,
  FHIRCoding,
  FHIRCodeableConcept,
  FHIRIdentifier,
  FHIRHumanName,
  FHIRAddress,
  FHIRContactPoint,
  FHIRReference,
  FHIRQuantity,
  FHIRPeriod,
  FHIRAttachment,
  // FHIR Resources
  FHIRPatient,
  FHIRObservation,
  FHIRCondition,
  FHIRAllergyIntolerance,
  FHIRMedicationStatement,
  FHIRConsent,
  FHIRConsentProvision,
  FHIRBundle,
} from './health-fhir/index.js';
