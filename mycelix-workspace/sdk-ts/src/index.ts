/**
 * @mycelix/sdk
 *
 * TypeScript SDK for the Mycelix ecosystem.
 *
 * - **MATL**: Mycelix Adaptive Trust Layer (34% validated Byzantine tolerance)
 * - **Epistemic**: Epistemic Charter v2.0 (3D truth classification)
 * - **Bridge**: Inter-hApp communication protocol
 *
 * @example
 * ```typescript
 * import { matl, epistemic, bridge } from '@mycelix/sdk';
 *
 * // Create a trust measurement
 * const pogq = matl.createPoGQ(0.95, 0.88, 0.12);
 * const score = matl.compositeScore(pogq, 0.75);
 *
 * // Classify a claim
 * const myClaim = epistemic.claim("User verified identity")
 *   .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
 *   .withNormative(epistemic.NormativeLevel.N2_Network)
 *   .build();
 *
 * // Cross-hApp reputation
 * const localBridge = new bridge.LocalBridge();
 * localBridge.registerHapp('myapp');
 * ```
 */

// Re-export modules
export * as matl from './matl/index.js';
export * as epistemic from './epistemic/index.js';
export * as bridge from './bridge/index.js';
export * as client from './client/index.js';
export * as fl from './fl/index.js';
export * as errors from './errors.js';
export * as security from './security/index.js';
export * as config from './config/index.js';
export * as utils from './utils/index.js';
export * as validation from './validation/index.js';
export * as signals from './signals/index.js';
export * as calibration from './calibration/index.js';
export * as metabolism from './metabolism/index.js';
export * as propagation from './propagation/index.js';
export * as discovery from './discovery/index.js';
export * as rtc from './rtc/index.js';
export * as ai from './ai/index.js';
export * as fhe from './fhe/index.js';
export * as flHub from './fl-hub/index.js';
export * as mobile from './mobile/index.js';

// Core SDK - Unified Client Architecture (Phase 2)
export * as core from './core/index.js';

// Glass-Top Architecture - Unified Wallet System
export * as reactive from './reactive/index.js';
export * as vault from './vault/index.js';
export * as wallet from './wallet/index.js';

// UESS (Unified Epistemic Storage System) - Reference Implementation
export * as storage from './storage/index.js';

// SCEI (Self-Correcting Epistemic Infrastructure) - Civilizational Intent Framework
export * as scei from './scei/index.js';

// Innovations - Revolutionary new systems extending the Mycelix ecosystem
export * as innovations from './innovations/index.js';

// Re-export integration modules
export * as mail from './integrations/mail/index.js';
export * as marketplace from './integrations/marketplace/index.js';
export * as edunet from './integrations/edunet/index.js';
export * as supplychain from './integrations/supplychain/index.js';

// Bridge routing types (shared across clusters)
export * as bridgeRouting from './integrations/bridge-routing.js';

// Core hApp integrations
export * as governance from './integrations/governance/index.js';
export * as finance from './integrations/finance/index.js';
export * as identity from './integrations/identity/index.js';
export * as knowledge from './integrations/knowledge/index.js';
export * as property from './integrations/property/index.js';
export * as energy from './integrations/energy/index.js';
export * as media from './integrations/media/index.js';
export * as justice from './integrations/justice/index.js';

// Direct exports for convenience
export {
  // MATL types and functions
  type ProofOfGradientQuality,
  type ReputationScore,
  type CompositeScore,
  type AdaptiveThreshold,
  type CompositeScoreOptions,
  createPoGQ,
  compositeScore,
  isByzantine,
  createReputation,
  reputationValue,
  recordPositive,
  recordNegative,
  calculateComposite,
  isTrustworthy,
  createAdaptiveThreshold,
  observe,
  getThreshold,
  isAnomalous,
  DEFAULT_QUALITY_WEIGHT,
  DEFAULT_CONSISTENCY_WEIGHT,
  DEFAULT_REPUTATION_WEIGHT,
  DEFAULT_ENTROPY_PENALTY,
  MAX_BYZANTINE_TOLERANCE,
  DEFAULT_BYZANTINE_THRESHOLD,
  ReputationCache,
  type ReputationCacheConfig,
} from './matl/index.js';

export {
  // Epistemic types and functions
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type EpistemicClassification,
  type Evidence,
  type EpistemicClaim,
  type EpistemicBatchSummary,
  type ClaimPoolConfig,
  type ClaimPoolStats,
  ClaimBuilder,
  EpistemicBatch,
  EpistemicClaimPool,
  classificationCode,
  meetsMinimum,
  parseClassificationCode,
  parseClassificationCodeStrict,
  createClaim,
  addEvidence,
  meetsStandard,
  isExpired,
  claim,
  Standards,
  // GIS v4.0 - Graceful Ignorance System with Harmonic extensions
  Harmony,
  HarmonicLevel,
  IgnoranceType,
  MoralUncertaintyType,
  MoralActionGuidance,
  type HarmonicImpact,
  type HarmonicIgnorance,
  type MoralUncertainty,
  type HarmonicPerspective,
  type SynthesizedView,
  type GracefulResponse,
  type GracefulResponseType,
  type HarmonicProfile,
  type EpistemicClearance,
  type KosmicSongState,
  type HarmonicEpistemicClassification,
  HARMONY_WEIGHTS,
  harmonyName,
  harmonyEpistemicMode,
  harmonyAbbreviation,
  generateHCode,
  calculateHarmonicEig,
  createMoralUncertainty,
  totalMoralUncertainty,
  getMoralActionGuidance,
  shouldPauseForReflection,
  getMoralRecommendations,
  createDefaultHarmonicProfile,
  calculateHarmonicCoherence,
  identifyGrowthEdges,
  findResonantHarmony,
  extendedClassificationCode,
  parseExtendedClassificationCode,
} from './epistemic/index.js';

export {
  // Bridge types and functions
  BridgeMessageType,
  type HappReputationScore,
  type BridgeMessage,
  type ReputationQueryMessage,
  type CrossHappReputationMessage,
  type CredentialVerificationMessage,
  type VerificationResultMessage,
  type BroadcastEventMessage,
  type HappRegistrationMessage,
  type AnyBridgeMessage,
  type MessageHandler,
  type AsyncMessageHandler,
  type BridgeMiddleware,
  type BridgeRouterHandlers,
  type BridgeRouterConfig,
  calculateAggregateReputation,
  createReputationQuery,
  createCrossHappReputation,
  createCredentialVerification,
  createVerificationResult,
  createBroadcastEvent,
  createHappRegistration,
  createMessageHandler,
  LocalBridge,
  BridgeRouter,
} from './bridge/index.js';

export {
  // Client types and functions
  type MycelixClientConfig,
  type ZomeCallParams,
  type RegisterHappInput,
  type RecordReputationInput,
  type TrustCheckInput,
  type BroadcastEventInput,
  type GetEventsInput,
  type CredentialVerifyInput,
  type CrossHappReputation,
  type BridgeEventRecord,
  type ConnectionState,
  type ConnectionStateListener,
  MycelixClient,
  MockMycelixClient,
  createClient,
  createMockClient,
} from './client/index.js';

export {
  // Federated Learning types and functions
  type GradientUpdate,
  type AggregatedGradient,
  type Participant,
  type FLRound,
  type FLConfig,
  type SerializedGradientUpdate,
  type SerializedAggregatedGradient,
  AggregationMethod,
  DEFAULT_CONFIG as FL_DEFAULT_CONFIG,
  fedAvg,
  trimmedMean,
  coordinateMedian,
  krum,
  trustWeightedAggregation,
  FLCoordinator,
  serializeGradients,
  deserializeGradients,
  serializeGradientUpdate,
  deserializeGradientUpdate,
  serializeAggregatedGradient,
  deserializeAggregatedGradient,
} from './fl/index.js';

export {
  // Error types and utilities
  MycelixError,
  ValidationError,
  MATLError,
  EpistemicError,
  FederatedLearningError,
  BridgeError,
  ConnectionError,
  ErrorCode,
  type MycelixErrorContext,
  type RecoverySuggestion,
  type ValidationResult,
  type RecoveryResult,
  type RecoveryConfig,
  type RecoveryStrategy,
  type CircuitState,
  validate,
  withErrorHandling,
  withRetry,
  withRecovery,
  withRecoveryOrThrow,
  exponentialBackoffStrategy,
  fallbackChainStrategy,
  circuitBreakerStrategy,
  gracePeriodStrategy,
  combineStrategies,
  assert,
  assertDefined,
  Validator,
} from './errors.js';

export {
  // Security types and functions
  SecurityError,
  SecurityErrorCode,
  secureRandomBytes,
  secureRandomFloat,
  secureRandomInt,
  secureUUID,
  secureShuffle,
  /** @deprecated Use `secureShuffle` instead (typo fix) */
  secureShufffle,
  hash,
  hashHex,
  hmac,
  verifyHmac,
  constantTimeEqual,
  createRateLimiter,
  checkRateLimit,
  RateLimiterRegistry,
  sanitizeString,
  sanitizeId,
  sanitizeJson,
  SecureSecret,
  validateBftParams,
  maxByzantineFailures,
  hasQuorum,
  randomDelay,
  constantTime,
  SecurityAuditLog,
  securityAudit,
  SecurityEventType,
  type SecurityEvent,
  type SecurityEventInput,
  type SecurityEventHandler,
  type RateLimitConfig,
  type RateLimitState,
  type RateLimitMetrics,
  type ThresholdConfig,
  type HashAlgorithm,
} from './security/index.js';

export {
  // Configuration types and functions
  type MATLConfig,
  type FLConfig as ConfigFLConfig,
  type SecurityConfig,
  type BridgeConfig,
  type ClientConfig,
  type LogConfig,
  type MycelixConfig,
  DEFAULT_MATL_CONFIG,
  DEFAULT_FL_CONFIG as CONFIG_FL_DEFAULT,
  DEFAULT_SECURITY_CONFIG,
  DEFAULT_BRIDGE_CONFIG,
  DEFAULT_CLIENT_CONFIG,
  DEFAULT_LOG_CONFIG,
  DEFAULT_CONFIG,
  ConfigManager,
  getConfig,
  initConfig,
  resetGlobalConfig,
  DEV_CONFIG,
  PROD_CONFIG,
  TEST_CONFIG,
  HIGH_SECURITY_CONFIG,
  createPreset,
  mergePresets,
  validateConfig,
  validateMATLConfig,
  validateFLConfig,
  validateSecurityConfig,
  validatePreset,
  validateDevPreset,
  validateProdPreset,
  validateTestPreset,
  validateHighSecurityPreset,
  detectPresetType,
  selectPreset,
  type PresetValidationWarning,
  type PresetValidationResult,
} from './config/index.js';

export {
  // Utility types and functions
  type AgentId,
  type HappId,
  type Timestamp,
  type TrustScore,
  type ByzantineTolerance,
  type TrustCheckResult,
  type FLRoundSummary,
  type SecureMessage,
  type HealthCheckResult,
  // Timestamp utilities
  now,
  isValidTimestamp,
  isExpiredTimestamp,
  isFutureTimestamp,
  normalizeTimestamp,
  timestampAge,
  formatTimestamp,
  parseTimestamp,
  // Trust utilities
  checkTrust,
  buildReputation,
  createSimpleFLCoordinator,
  createGradientUpdate,
  runFLRound,
  signMessage,
  verifyMessage,
  createRateLimitedOperation,
  createHighTrustClaim,
  createMediumTrustClaim,
  createLowTrustClaim,
  ReputationAggregator,
  runHealthChecks,
  getSdkInfo,
  // Async utilities
  retry,
  parallel,
  deferred,
  timeout,
  withTimeout,
  TimeoutError,
  debounce,
  throttle,
  sleep,
  // Type guards
  isProofOfGradientQuality,
  isReputationScore,
  isEpistemicClaim,
  isGradientUpdate,
  isBridgeMessage,
  isSecureMessage,
  isTrustCheckResult,
  isHealthCheckResult,
  isConnectionState,
  isFLRound,
  isEpistemicClassification,
  isCompositeScore,
  isFLConfig,
  isMycelixConfig,
  isAggregatedGradient,
  isSerializedGradientUpdate,
  isSerializedAggregatedGradient,
  // Observable event system
  type Observer,
  type Subscription,
  type ObservableStats,
  Observable,
  // SDK Event types
  type SdkEvent,
  type MatlEvent,
  type ReputationUpdatedEvent,
  type TrustEvaluatedEvent,
  type ByzantineDetectedEvent,
  type ThresholdAdjustedEvent,
  type CacheHitEvent,
  type CacheMissEvent,
  MatlEventType,
  type BridgeEvent,
  type MessageSentEvent,
  type MessageReceivedEvent,
  type MessageRoutedEvent,
  type MessageUnhandledEvent,
  type HappRegisteredEvent,
  type ReputationAggregatedEvent,
  BridgeEventType,
  type FlEvent,
  type RoundStartedEvent,
  type UpdateSubmittedEvent,
  type UpdateRejectedEvent,
  type RoundAggregatedEvent,
  type ParticipantRegisteredEvent,
  type FlReputationUpdatedEvent,
  FlEventType,
  type EpistemicEvent,
  type ClaimCreatedEvent,
  type ClaimExpiredEvent,
  type ClaimExtendedEvent,
  type EvidenceAddedEvent,
  type PoolCleanupEvent,
  type BatchCompletedEvent,
  EpistemicEventType,
  type SdkSecurityEvent,
  type SecretCreatedEvent,
  type SecretAccessedEvent,
  type SecretExpiredEvent,
  type SecretDestroyedEvent,
  type RateLimitTriggeredEvent,
  type RateLimitResetEvent,
  SdkSecurityEventType,
  type AnySdkEvent,
  SdkEventBus,
  // Event factory functions
  createReputationUpdatedEvent,
  createTrustEvaluatedEvent,
  createMessageRoutedEvent,
  createRoundStartedEvent,
  createRoundAggregatedEvent,
  createClaimCreatedEvent,
  createPoolCleanupEvent,
  // Global event bus
  sdkEvents,
  // Batch Request Optimization
  type BatchOptions,
  type BatchStats,
  RequestBatcher,
  // Event Pipeline Composition
  type PipelineOperator,
  type WindowResult,
  EventPipeline,
  pipeline,
  // Distributed Tracing
  type TraceContext,
  type Span,
  type CompletedTrace,
  TracingManager,
  tracer,
  traced,
  // Policy-Based Authorization
  type ResourceType,
  type Action as PolicyAction,
  type AccessPolicy,
  type PolicyContext,
  type PolicyResult,
  PolicyEngine,
  policyEngine,
  // Time-Series Analytics
  type TimeSeriesPoint,
  type TrendAnalysis,
  type TimeWindow,
  type WindowStats,
  TimeSeriesAnalytics,
  reputationAnalytics,
  flAnalytics,
  // Claim Verification Framework
  type EvidenceProof,
  type ProofVerificationResult,
  type ClaimVerificationResult,
  type VerificationStrategy,
  type VerificationOptions,
  ClaimVerifier,
  claimVerifier,
  // Schema Versioning
  type SchemaVersion,
  type SchemaMigration,
  type SchemaDefinition,
  type MigrationResult,
  SchemaRegistry,
  schemaRegistry,
  // Real-time & Connectivity
  type WebSocketState,
  type WebSocketMessage,
  type WebSocketConfig,
  type WebSocketHandlers,
  type WebSocketStats,
  WebSocketManager,
  type ConnectionPoolConfig,
  type PooledConnection,
  type ConnectionPoolStats,
  ConnectionPool,
  type QueuedOperation,
  type OfflineQueueConfig,
  type OfflineQueueStats,
  OfflineQueue,
  type SignalType,
  type HolochainSignal,
  type SignalHandlerConfig,
  SignalHandler,
  // Developer Experience
  type LogLevel,
  type LogEntry,
  type LoggerConfig,
  StructuredLogger,
  DebugInspector,
  type BrandedAgentId,
  type BrandedHappId,
  type BrandedTrustScore,
  type BrandedTimestampMs,
  brandAgentId,
  brandHappId,
  brandTrustScore,
  brandTimestamp,
  sdkHelpers,
  // Production Readiness
  type MetricType,
  type MetricLabels,
  type MetricValue,
  type MetricDefinition,
  type HistogramBuckets,
  MetricsCollector,
  metrics,
  type HealthStatus,
  type HealthCheck,
  type HealthReport,
  type HealthCheckFn,
  HealthChecker,
  // Testing & Quality
  type MatlTestFixture,
  type EpistemicTestFixture,
  type FlTestFixture,
  TestFixtureFactory,
  mockFactory,
  propertyGenerators,
  // Documentation Helpers
  examples,
  // Advanced Features
  type PluginMetadata,
  type PluginHooks,
  type Plugin,
  PluginManager,
  type StateSyncConfig,
  type SyncableState,
  StateSync,
  type EncryptedStorageConfig,
  EncryptedStorage,
  MultiConductor,
} from './utils/index.js';

// Integration type exports
export {
  // Mail Integration
  type SenderTrust,
  type EmailVerification,
  type VerifiedEmail,
  type EmailClaim,
  MailTrustService,
  getMailTrustService,
} from './integrations/mail/index.js';

export {
  // Marketplace Integration
  type Transaction,
  type TransactionType,
  type SellerProfile,
  type BuyerProfile,
  type ListingVerification,
  MarketplaceReputationService,
  getMarketplaceService,
} from './integrations/marketplace/index.js';

export {
  // EduNet Integration
  type CompletionStatus,
  type Course,
  type CourseCompletion,
  type EducationalCredential,
  type SkillAssessment,
  type LearnerProfile,
  EduNetCredentialService,
  getEduNetService,
} from './integrations/edunet/index.js';

export {
  // SupplyChain Integration
  type EvidenceType,
  type CheckpointEvidence,
  type Checkpoint,
  type ProvenanceChain,
  type HandlerProfile,
  type ChainVerification,
  SupplyChainProvenanceService,
  getSupplyChainService,
} from './integrations/supplychain/index.js';

export {
  // Validation schemas and utilities
  z,
  // Common schemas
  didSchema,
  happIdSchema,
  matlScoreSchema,
  amountSchema,
  timestampSchema,
  locationSchema,
  // Identity schemas
  registerHappInputSchema,
  queryIdentityInputSchema,
  reportReputationInputSchema,
  bridgeEventTypeSchema,
  broadcastIdentityEventInputSchema,
  // Finance schemas
  creditPurposeSchema,
  queryCreditInputSchema,
  processPaymentInputSchema,
  assetTypeSchema,
  registerCollateralInputSchema,
  financeBridgeEventTypeSchema,
  // Property schemas
  verificationTypeSchema,
  verifyOwnershipInputSchema,
  pledgeCollateralInputSchema,
  propertyBridgeEventTypeSchema,
  // Energy schemas
  energySourceSchema,
  queryAvailableEnergyInputSchema,
  requestEnergyPurchaseInputSchema,
  reportProductionInputSchema,
  energyBridgeEventTypeSchema,
  // Media schemas
  contentTypeSchema,
  licenseTypeSchema,
  queryContentInputSchema,
  requestLicenseInputSchema,
  distributeRoyaltiesInputSchema,
  verifyLicenseInputSchema,
  mediaBridgeEventTypeSchema,
  // Governance schemas
  governanceQueryTypeSchema,
  queryGovernanceInputSchema,
  requestExecutionInputSchema,
  governanceBridgeEventTypeSchema,
  broadcastGovernanceEventInputSchema,
  // Justice schemas
  crossHappDisputeTypeSchema,
  enforcementTypeSchema,
  fileCrossHappDisputeInputSchema,
  requestEnforcementInputSchema,
  disputeHistoryQuerySchema,
  justiceBridgeEventTypeSchema,
  // Knowledge schemas
  knowledgeQueryTypeSchema,
  queryKnowledgeInputSchema,
  factCheckInputSchema,
  registerExternalClaimInputSchema,
  knowledgeBridgeEventTypeSchema,
  broadcastKnowledgeEventInputSchema,
  // Validation utilities
  validateOrThrow,
  validateSafe,
  withValidation,
  // Schema collections
  IdentitySchemas,
  FinanceSchemas,
  PropertySchemas,
  EnergySchemas,
  MediaSchemas,
  GovernanceSchemas,
  JusticeSchemas,
  KnowledgeSchemas,
} from './validation/index.js';

export {
  // Signal types
  type BaseSignalPayload,
  type SignalSubscriptionOptions,
  // Identity signals
  type IdentitySignalType,
  type IdentitySignalPayload,
  type IdentityCreatedPayload,
  type IdentityUpdatedPayload,
  type CredentialIssuedPayload,
  type CredentialRevokedPayload,
  type TrustAttestedPayload,
  // Finance signals
  type FinanceSignalType,
  type FinanceSignalPayload,
  type PaymentProcessedPayload,
  type LoanApprovedPayload,
  type CreditScoreUpdatedPayload,
  // Energy signals
  type EnergySignalType,
  type EnergySignalPayload,
  type EnergyListedPayload,
  type EnergyPurchasedPayload,
  type GridBalanceRequestPayload,
  // Governance signals
  type GovernanceSignalType,
  type GovernanceSignalPayload,
  type ProposalCreatedPayload,
  type ProposalPassedPayload,
  type VoteCastPayload,
  // Knowledge signals
  type KnowledgeSignalType,
  type KnowledgeSignalPayload,
  type ClaimCreatedPayload,
  type FactCheckResultPayload,
  // Signal handlers
  IdentitySignalHandler,
  FinanceSignalHandler,
  EnergySignalHandler,
  GovernanceSignalHandler,
  KnowledgeSignalHandler,
  BridgeSignalManager,
  // Factory functions
  createSignalManager,
  createIdentitySignals,
  createFinanceSignals,
  createEnergySignals,
  createGovernanceSignals,
  createKnowledgeSignals,
} from './signals/index.js';

// SCEI (Self-Correcting Epistemic Infrastructure) direct exports
export {
  // Infrastructure bundle
  type SCEIInfrastructure,
  getSCEIInfrastructure,
  resetSCEIInfrastructure,
  // Validation
  type ValidationResult as SCEIValidationResult,
  type ValidationError as SCEIValidationErrorType,
  validateConfidence,
  validateDegradationFactor,
  validatePositiveInteger,
  validateNonEmptyString,
  validateId,
  validateEnum,
  validateArray,
  combineValidations,
  assertValid,
  SCEIValidationError,
  safeLog,
  clampConfidence,
  safeDivide,
  safeWeightedAverage,
  // Event Bus
  type SCEIEventType,
  type SCEIEvent,
  type SCEIModule,
  type SCEIEventListener,
  type SCEIEventBusMetrics,
  SCEIEventBus,
  SCEIEventBuilder,
  getSCEIEventBus,
  resetSCEIEventBus,
  createSCEIEvent,
  createCorrelationId,
  // Metrics
  type SCEIMetricsSnapshot,
  SCEI_METRICS,
  SCEIMetricsCollector,
  getSCEIMetrics,
  resetSCEIMetrics,
  // Persistence
  type StorageKey,
  type SCEINamespace,
  type StorageOptions,
  type StoredItem,
  type SCEIPersistence,
  InMemorySCEIStorage,
  SCEIPersistenceManager,
  getSCEIPersistence,
  setSCEIPersistence,
  resetSCEIPersistence,
} from './scei/index.js';

// ============================================================================
// Glass-Top Wallet Architecture - Unified UX
// ============================================================================

export {
  // Reactive State (for "alive" UIs)
  BehaviorSubject,
  Subject,
  ReplaySubject,
  Observable as ReactiveObservable,
  type Observer as ReactiveObserver,
  type Subscription as ReactiveSubscription,
  type Subscribable,
  // Operators
  map as rxMap,
  filter as rxFilter,
  distinctUntilChanged,
  debounceTime,
  take,
  skip,
  // Combinators
  combineLatest,
  merge as rxMerge,
  fromPromise,
  interval,
  timer,
  // Store
  Store,
  type Action as StoreAction,
  type Reducer,
} from './reactive/index.js';

export {
  // MycelixVault - Secure Key Management (the Pen)
  MycelixVault,
  createVault,
  loadVault,
  type VaultId,
  type AccountId,
  type KeyId,
  type VaultStatus,
  type AuthMethod,
  type KeyType,
  type VaultMetadata,
  type VaultAccount,
  type VaultState,
  type CreateVaultConfig,
  type UnlockOptions,
  type VaultStorageProvider,
  type BiometricProvider,
} from './vault/index.js';

export {
  // Unified Wallet - Glass-Top Facade
  Wallet,
  createWallet,
  loadWallet,
  type Currency,
  type TransactionStatus,
  type TransactionDirection,
  type Identity,
  type Transaction as WalletTransaction,
  type Balance,
  type WalletState,
  type SendOptions,
  type WalletConfig,
  type IdentityResolver,
  type FinanceProvider,
  type WalletEvent,
} from './wallet/index.js';

export {
  // Identity Resolution - Human-Readable Everything
  HolochainIdentityResolver,
  IdentityCache,
  formatIdentity,
  truncateAgentId,
  generateAvatarUrl,
  hydrateTransaction,
  hydrateTransactions,
  type ProfileData,
  type DIDDocument,
  type ZomeCallable,
} from './wallet/identity-resolver.js';

export {
  // React Hooks & Optimistic UI
  createWalletHooks,
  OptimisticStateManager,
  createOptimisticTransaction,
  applyOptimisticBalanceChange,
  eventToToast,
  getContextAction,
  createBalanceAnimation,
  formatCurrency,
  formatRelativeTime,
  type UseWalletReturn,
  type UseBalanceReturn,
  type UseTransactionsReturn,
  type UseSendReturn,
  type UseIdentityReturn,
  type WalletContext,
  type ContextAction,
} from './wallet/react.js';

// ============================================================================
// Framework Integrations
// ============================================================================

// Legacy framework exports (domain-specific hooks/stores)
// Re-export React hooks for all 8 Civilizational OS hApps
export * as react from './react/index.js';

// Re-export Svelte stores for all 8 Civilizational OS hApps
export * as svelte from './svelte/index.js';

// Re-export Vue 3 composables for all 8 Civilizational OS hApps
export * as vue from './vue/index.js';

// New unified framework integrations (Phase 5)
// These provide framework-specific providers, hooks, and composables
// for the unified Mycelix client architecture
export * as frameworks from './frameworks/index.js';

// ============================================================================
// GraphQL API Layer
// ============================================================================

// Re-export GraphQL API for unified frontend access
export * as graphql from './graphql/index.js';

// ============================================================================
// Unified Client Convenience Exports
// ============================================================================

/**
 * Unified clients for all 8 Civilizational OS hApps.
 * These provide a unified interface for each domain.
 */
export { MycelixKnowledgeClient } from './integrations/knowledge/client';
export type { KnowledgeClientConfig, KnowledgeConnectionOptions } from './integrations/knowledge/client';

export { MycelixJusticeClient } from './justice/client';
export type { JusticeClientConfig, JusticeConnectionOptions } from './justice/client';

export { MycelixMediaClient } from './media/client';
export type { MediaClientConfig, MediaConnectionOptions } from './media/client';

export { MycelixHealthClient } from './integrations/health/client';
export type { HealthClientConfig, HealthConnectionOptions } from './integrations/health/client';

export { MycelixEnergyClient } from './integrations/energy/client';
export type { EnergyClientConfig, EnergyConnectionOptions } from './integrations/energy/client';

export { MycelixFinanceClient } from './integrations/finance/client';
export type { FinanceClientConfig, FinanceConnectionOptions } from './integrations/finance/client';

export { MycelixGovernanceClient } from './integrations/governance/client';
export type { GovernanceClientConfig as LegacyGovernanceClientConfig, GovernanceConnectionOptions } from './integrations/governance/client';

// Phase 3: Full 10-zome GovernanceClient (dao, proposals, voting, delegation, treasury, execution, bridge, councils, constitution, threshold-signing)
export { GovernanceClient, type GovernanceClientConfig } from './clients/governance/index.js';
export { DAOClient } from './clients/governance/dao';
export { ProposalsClient } from './clients/governance/proposals';
export { VotingClient } from './clients/governance/voting';
export { DelegationClient } from './clients/governance/delegation';
export { TreasuryClient } from './clients/governance/treasury';
export { ExecutionClient } from './clients/governance/execution';
export { BridgeClient as GovernanceBridgeClient } from './clients/governance/bridge';
export { CouncilsClient } from './clients/governance/councils';
export { ConstitutionClient } from './clients/governance/constitution';
export { ThresholdSigningClient } from './clients/governance/threshold-signing';

export { MycelixPropertyClient } from './integrations/property/client';
export type { PropertyClientConfig, PropertyConnectionOptions } from './integrations/property/client';

export { MycelixIdentityClient } from './integrations/identity/client';
export type { MycelixIdentityConfig } from './integrations/identity/client';

// ============================================================================
// Common Utilities
// ============================================================================

export { RetryPolicy, RetryPolicies, withRetryAndTimeout } from './common/retry';
export type { RetryOptions } from './common/retry';

export { CircuitBreaker, CircuitBreakerOpenError, CircuitBreakers, withCircuitBreaker } from './common/circuit-breaker';
export type { CircuitBreakerConfig as CommonCircuitBreakerConfig, CircuitBreakerStats as CommonCircuitBreakerStats } from './common/circuit-breaker';

export { Cache, NamespacedCache, Caches, memoize } from './common/cache';
export type { CacheOptions } from './common/cache';

export {
  RequestBatcher as BatchRequestBatcher,
  BatchExecutor,
  createBatchedFunction,
  debounce as asyncDebounce,
  throttle as asyncThrottle,
} from './common/batch';
export type { BatchOptions as CommonBatchOptions } from './common/batch';

// ============================================================================
// Core SDK - Unified Client Factory (Phase 2)
// ============================================================================

export {
  // Unified Client
  Mycelix,
  type MycelixConfig as UnifiedClientConfig,
  type MycelixClientOptions,
  type ConnectionState as CoreConnectionState,
  type SignalHandler as CoreSignalHandler,
  type ConnectionStateListener as CoreConnectionStateListener,

  // Error Handling
  SdkErrorCode,
  SdkError as CoreSdkError,
  NetworkError,
  ValidationError as CoreValidationError,
  NotFoundError,
  TimeoutError as CoreTimeoutError,
  UnauthorizedError,
  ZomeCallError,
  isRetryableCode,
  wrapError,
  type SdkErrorDetails,

  // Retry Utilities
  withRetry as coreWithRetry,
  tryWithRetry,
  withTimeout as coreWithTimeout,
  withRetryAndTimeout as coreWithRetryAndTimeout,
  createRetryWrapper,
  RetryPolicies as CoreRetryPolicies,
  DEFAULT_RETRY_CONFIG,
  type RetryConfig as CoreRetryConfig,
  type RetryResult,

  // Zome Client Base
  ZomeClient,
  createZomeClient,
  type ZomeClientConfig,
} from './core/index.js';

/**
 * SDK version
 */
export const VERSION = '0.6.0';

/**
 * Initialize the SDK (placeholder for future initialization)
 */
export function init(): void {
  // Future: Initialize logging, metrics, Holochain connection, etc.
}
