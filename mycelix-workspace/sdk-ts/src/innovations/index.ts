/**
 * @mycelix/sdk Innovations
 *
 * Revolutionary new systems that extend the Mycelix ecosystem:
 *
 * - **Trust Markets**: Prediction markets for trust claims themselves
 * - **Private Queries**: Privacy-preserving cross-hApp analytics via FHE
 * - **Epistemic Agents**: Transform agents into verified knowledge sources
 * - **Civic Feedback Loop**: Auto-propagation across Justice-Knowledge-Governance
 * - **Self-Healing Workflows**: Resilient workflow orchestration with sagas
 * - **Constitutional AI**: DAO-governed rules for AI agent behavior
 *
 * @packageDocumentation
 * @module innovations
 */

// Trust Market Protocol
export {
  TrustMarketService,
  type TrustMarket,
  type TrustClaim,
  type TrustClaimType,
  type TrustMarketMechanism,
  type TrustOutcome,
  type TrustOrder,
  type TrustMarketState,
  type TrustResolutionConfig,
  type TrustDataSource,
  type TrustMarketMetadata,
  type TrustPosition,
  type MultiDimensionalTrustStake,
  type CommitmentAction,
  type CalibrationFeedbackLoop,
  type TrustWisdom,
  type WisdomActivation,
  type CreateTrustMarketInput,
  type SubmitTrustTradeInput,
  type TrustTradeResult,
  type TrustResolutionData,
  type TrustPayout,
  type TrustMarketResolution,
  type TrustMarketStats,
} from './trust-markets/index.js';

// Privacy-Preserving Cross-hApp Queries
export {
  PrivateQueryService,
  createPrivateQueryService,
  createStrictPrivateQueryService,
  createAnalyticsQueryService,
  type PrivateQuery,
  type PrivateQueryType,
  type EncryptedQueryParams,
  type QueryFilter,
  type PrivateQueryState,
  type QueryMetrics,
  type PrivateQueryResult,
  type PSIQuery,
  type PSIResult,
  type ThresholdDecryptionRequest,
  type GuardianApproval,
  type SubmitAggregateQueryInput,
  type SubmitPSIQueryInput,
  type RequestThresholdDecryptionInput,
  type PrivateCorrelationInput,
  type PrivateCorrelationResult,
  type EncryptedStats,
  type PrivacyStats,
} from './private-queries/index.js';

// Epistemic Validator Agent System
export {
  EpistemicAgentRunner,
  EpistemicKnowledgeRetriever,
  createEpistemicAgent,
  createEpistemicAgentFromDomain,
  DEFAULT_EPISTEMIC_CONFIG,
  type EpistemicClaim,
  type DKGValidationResult,
  type EpistemicAgentResponse,
  type EpistemicResponseMetadata,
  type EpistemicConfig,
  type AgentEpistemicStats,
  type DKGClient,
} from './epistemic-agents/index.js';

// Justice-Knowledge-Governance Feedback Loop
export {
  CivicFeedbackLoop,
  createCivicFeedbackLoop,
  getCivicFeedbackLoop,
  resetCivicFeedbackLoop,
  DEFAULT_FEEDBACK_LOOP_CONFIG,
  type JusticeDecision,
  type JusticeDecisionType,
  type LegalDomain,
  type GovernanceOutcome,
  type GovernanceOutcomeType,
  type EnactedRule,
  type DerivedKnowledgeClaim,
  type LegalPrecedent,
  type FeedbackLoopEvent,
  type FeedbackLoopEventType,
  type FeedbackLoopConfig,
  type PropagationResult,
  type ConflictReport,
  type FeedbackLoopStats,
} from './civic-feedback-loop/index.js';

// Self-Healing Workflow Orchestrator
export {
  SelfHealingOrchestrator,
  WorkflowBuilder,
  createWorkflow,
  createOrchestrator,
  getOrchestrator,
  resetOrchestrator,
  DEFAULT_ORCHESTRATOR_CONFIG,
  type WorkflowStatus,
  type StepStatus,
  type WorkflowStepDefinition,
  type WorkflowDefinition,
  type WorkflowContext,
  type WorkflowStepState,
  type WorkflowState,
  type WorkflowCheckpoint,
  type WorkflowResult,
  type WorkflowStateEvent,
  type WorkflowObserver,
  type OrchestratorConfig,
} from './self-healing-workflows/index.js';

// Constitutional AI Governance
export {
  ConstitutionalGovernor,
  createConstitutionalGovernor,
  getConstitutionalGovernor,
  resetConstitutionalGovernor,
  DEFAULT_CONSTITUTIONAL_CONFIG,
  FOUNDATIONAL_RULES,
  type RuleCategory,
  type RuleSeverity,
  type ConstitutionalRule,
  type RuleCondition,
  type RuleAmendment,
  type ConstitutionalViolation,
  type ViolationAction,
  type AgentConstitutionalStatus,
  type AmendmentProposal,
  type ConstitutionalAudit,
  type ConstitutionalConfig,
  type ContentCheckResult,
} from './constitutional-ai/index.js';

// Integration Examples
export {
  trustMarketCalibrationExample,
  epistemicCivicIntegrationExample,
  workflowConstitutionalExample,
  privateQueryAnalyticsExample,
  comprehensiveCivicScenario,
  runAllExamples,
} from './examples/index.js';
