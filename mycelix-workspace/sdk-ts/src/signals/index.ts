/**
 * @mycelix/sdk Signals Module
 *
 * Typed signal handlers for real-time event subscriptions
 * from all Mycelix hApp bridge zomes.
 *
 * @packageDocumentation
 * @module signals
 */

import { type MycelixClient } from '../client/index.js';
import { SignalHandler, type HolochainSignal } from '../utils/index.js';

// ============================================================================
// Common Signal Types
// ============================================================================

/** Base signal payload with common fields */
export interface BaseSignalPayload {
  timestamp: number;
  source_happ?: string;
}

/** Signal subscription options */
export interface SignalSubscriptionOptions {
  /** Only receive signals from specific zome */
  zomeFilter?: string;
  /** Buffer size for recent signals */
  bufferSize?: number;
  /** Auto-reconnect on disconnect */
  autoReconnect?: boolean;
}

// ============================================================================
// Identity Bridge Signals
// ============================================================================

export type IdentitySignalType =
  | 'IdentityCreated'
  | 'IdentityUpdated'
  | 'CredentialIssued'
  | 'CredentialRevoked'
  | 'TrustAttested'
  | 'RecoveryInitiated'
  | 'HappRegistered'
  | 'ReputationReported';

export interface IdentityCreatedPayload extends BaseSignalPayload {
  did: string;
  initial_matl_score: number;
}

export interface IdentityUpdatedPayload extends BaseSignalPayload {
  did: string;
  updated_fields: string[];
}

export interface CredentialIssuedPayload extends BaseSignalPayload {
  credential_id: string;
  issuer_did: string;
  subject_did: string;
  credential_type: string;
}

export interface CredentialRevokedPayload extends BaseSignalPayload {
  credential_id: string;
  reason: string;
}

export interface TrustAttestedPayload extends BaseSignalPayload {
  attester_did: string;
  subject_did: string;
  trust_level: number;
}

export interface RecoveryInitiatedPayload extends BaseSignalPayload {
  did: string;
  recovery_method: string;
}

export interface HappRegisteredPayload extends BaseSignalPayload {
  happ_id: string;
  happ_name: string;
  capabilities: string[];
}

export interface ReputationReportedPayload extends BaseSignalPayload {
  did: string;
  score: number;
  interactions: number;
}

export type IdentitySignalPayload =
  | { type: 'IdentityCreated'; data: IdentityCreatedPayload }
  | { type: 'IdentityUpdated'; data: IdentityUpdatedPayload }
  | { type: 'CredentialIssued'; data: CredentialIssuedPayload }
  | { type: 'CredentialRevoked'; data: CredentialRevokedPayload }
  | { type: 'TrustAttested'; data: TrustAttestedPayload }
  | { type: 'RecoveryInitiated'; data: RecoveryInitiatedPayload }
  | { type: 'HappRegistered'; data: HappRegisteredPayload }
  | { type: 'ReputationReported'; data: ReputationReportedPayload };

// ============================================================================
// Finance Bridge Signals
// ============================================================================

export type FinanceSignalType =
  | 'PaymentProcessed'
  | 'LoanApproved'
  | 'LoanDefaulted'
  | 'CollateralPledged'
  | 'CollateralReleased'
  | 'CreditScoreUpdated';

export interface PaymentProcessedPayload extends BaseSignalPayload {
  payment_id: string;
  payer_did: string;
  payee_did: string;
  amount: number;
  currency: string;
  reference_id: string;
}

export interface LoanApprovedPayload extends BaseSignalPayload {
  loan_id: string;
  borrower_did: string;
  amount: number;
  interest_rate: number;
  term_months: number;
}

export interface LoanDefaultedPayload extends BaseSignalPayload {
  loan_id: string;
  borrower_did: string;
  amount_owed: number;
}

export interface CollateralPledgedPayload extends BaseSignalPayload {
  pledge_id: string;
  asset_id: string;
  owner_did: string;
  pledge_amount: number;
  pledge_to_happ: string;
}

export interface CollateralReleasedPayload extends BaseSignalPayload {
  pledge_id: string;
  asset_id: string;
  reason: 'repaid' | 'foreclosed' | 'released';
}

export interface CreditScoreUpdatedPayload extends BaseSignalPayload {
  did: string;
  old_score: number;
  new_score: number;
  reason: string;
}

export type FinanceSignalPayload =
  | { type: 'PaymentProcessed'; data: PaymentProcessedPayload }
  | { type: 'LoanApproved'; data: LoanApprovedPayload }
  | { type: 'LoanDefaulted'; data: LoanDefaultedPayload }
  | { type: 'CollateralPledged'; data: CollateralPledgedPayload }
  | { type: 'CollateralReleased'; data: CollateralReleasedPayload }
  | { type: 'CreditScoreUpdated'; data: CreditScoreUpdatedPayload };

// ============================================================================
// Property Bridge Signals
// ============================================================================

export type PropertySignalType =
  | 'OwnershipTransferred'
  | 'CollateralPledged'
  | 'CollateralReleased'
  | 'EncumbranceAdded'
  | 'EncumbranceRemoved'
  | 'PropertyRegistered';

export interface OwnershipTransferredPayload extends BaseSignalPayload {
  asset_id: string;
  from_did: string;
  to_did: string;
  percentage: number;
}

export interface PropertyCollateralPledgedPayload extends BaseSignalPayload {
  asset_id: string;
  owner_did: string;
  pledge_to_happ: string;
  pledge_amount: number;
}

export interface PropertyCollateralReleasedPayload extends BaseSignalPayload {
  asset_id: string;
  pledge_id: string;
  reason: string;
}

export interface EncumbranceAddedPayload extends BaseSignalPayload {
  asset_id: string;
  encumbrance_type: string;
  holder_did: string;
  amount?: number;
}

export interface EncumbranceRemovedPayload extends BaseSignalPayload {
  asset_id: string;
  encumbrance_id: string;
}

export interface PropertyRegisteredPayload extends BaseSignalPayload {
  asset_id: string;
  owner_did: string;
  asset_type: string;
  name: string;
}

export type PropertySignalPayload =
  | { type: 'OwnershipTransferred'; data: OwnershipTransferredPayload }
  | { type: 'CollateralPledged'; data: PropertyCollateralPledgedPayload }
  | { type: 'CollateralReleased'; data: PropertyCollateralReleasedPayload }
  | { type: 'EncumbranceAdded'; data: EncumbranceAddedPayload }
  | { type: 'EncumbranceRemoved'; data: EncumbranceRemovedPayload }
  | { type: 'PropertyRegistered'; data: PropertyRegisteredPayload };

// ============================================================================
// Energy Bridge Signals
// ============================================================================

export type EnergySignalType =
  | 'EnergyListed'
  | 'EnergyPurchased'
  | 'SettlementCompleted'
  | 'CarbonCreditIssued'
  | 'GridBalanceRequest'
  | 'GridBalanceFulfilled';

export interface EnergyListedPayload extends BaseSignalPayload {
  listing_id: string;
  seller_did: string;
  amount_kwh: number;
  price_per_kwh: number;
  energy_source: string;
  location?: { lat: number; lng: number };
}

export interface EnergyPurchasedPayload extends BaseSignalPayload {
  transaction_id: string;
  buyer_did: string;
  seller_did: string;
  amount_kwh: number;
  total_price: number;
}

export interface SettlementCompletedPayload extends BaseSignalPayload {
  settlement_id: string;
  transactions: string[];
  total_kwh: number;
  total_value: number;
}

export interface CarbonCreditIssuedPayload extends BaseSignalPayload {
  credit_id: string;
  recipient_did: string;
  amount_tons: number;
  source: string;
}

export interface GridBalanceRequestPayload extends BaseSignalPayload {
  request_id: string;
  requester_did: string;
  needed_kwh: number;
  max_price: number;
  deadline: number;
}

export interface GridBalanceFulfilledPayload extends BaseSignalPayload {
  request_id: string;
  fulfilled_by: string[];
  total_kwh: number;
  avg_price: number;
}

export type EnergySignalPayload =
  | { type: 'EnergyListed'; data: EnergyListedPayload }
  | { type: 'EnergyPurchased'; data: EnergyPurchasedPayload }
  | { type: 'SettlementCompleted'; data: SettlementCompletedPayload }
  | { type: 'CarbonCreditIssued'; data: CarbonCreditIssuedPayload }
  | { type: 'GridBalanceRequest'; data: GridBalanceRequestPayload }
  | { type: 'GridBalanceFulfilled'; data: GridBalanceFulfilledPayload };

// ============================================================================
// Media Bridge Signals
// ============================================================================

export type MediaSignalType =
  | 'ContentPublished'
  | 'ContentLicensed'
  | 'RoyaltyDistributed'
  | 'ContentFlagged'
  | 'ContentRemoved'
  | 'CreatorVerified';

export interface ContentPublishedPayload extends BaseSignalPayload {
  content_id: string;
  creator_did: string;
  content_type: string;
  title: string;
  license_type: string;
}

export interface ContentLicensedPayload extends BaseSignalPayload {
  license_id: string;
  content_id: string;
  licensee_did: string;
  license_type: string;
  duration_days?: number;
}

export interface RoyaltyDistributedPayload extends BaseSignalPayload {
  distribution_id: string;
  content_id: string;
  total_amount: number;
  recipients: Array<{ did: string; amount: number; percentage: number }>;
}

export interface ContentFlaggedPayload extends BaseSignalPayload {
  content_id: string;
  flagged_by: string;
  reason: string;
  severity: 'low' | 'medium' | 'high';
}

export interface ContentRemovedPayload extends BaseSignalPayload {
  content_id: string;
  reason: string;
  removed_by: string;
}

export interface CreatorVerifiedPayload extends BaseSignalPayload {
  creator_did: string;
  verification_type: string;
  verified_by: string;
}

export type MediaSignalPayload =
  | { type: 'ContentPublished'; data: ContentPublishedPayload }
  | { type: 'ContentLicensed'; data: ContentLicensedPayload }
  | { type: 'RoyaltyDistributed'; data: RoyaltyDistributedPayload }
  | { type: 'ContentFlagged'; data: ContentFlaggedPayload }
  | { type: 'ContentRemoved'; data: ContentRemovedPayload }
  | { type: 'CreatorVerified'; data: CreatorVerifiedPayload };

// ============================================================================
// Governance Bridge Signals
// ============================================================================

export type GovernanceSignalType =
  | 'ProposalCreated'
  | 'ProposalPassed'
  | 'ProposalRejected'
  | 'VoteCast'
  | 'ExecutionRequested'
  | 'ExecutionCompleted'
  | 'DelegationChanged';

export interface ProposalCreatedPayload extends BaseSignalPayload {
  proposal_hash: string;
  proposer_did: string;
  title: string;
  voting_ends_at: number;
}

export interface ProposalPassedPayload extends BaseSignalPayload {
  proposal_hash: string;
  votes_for: number;
  votes_against: number;
  participation_rate: number;
}

export interface ProposalRejectedPayload extends BaseSignalPayload {
  proposal_hash: string;
  votes_for: number;
  votes_against: number;
  reason: string;
}

export interface VoteCastPayload extends BaseSignalPayload {
  proposal_hash: string;
  voter_did: string;
  vote: 'for' | 'against' | 'abstain';
  weight: number;
}

export interface ExecutionRequestedPayload extends BaseSignalPayload {
  proposal_hash: string;
  target_happ: string;
  action: string;
}

export interface ExecutionCompletedPayload extends BaseSignalPayload {
  proposal_hash: string;
  target_happ: string;
  success: boolean;
  result?: string;
}

export interface DelegationChangedPayload extends BaseSignalPayload {
  delegator_did: string;
  delegate_did: string | null;
  previous_delegate?: string;
}

export type GovernanceSignalPayload =
  | { type: 'ProposalCreated'; data: ProposalCreatedPayload }
  | { type: 'ProposalPassed'; data: ProposalPassedPayload }
  | { type: 'ProposalRejected'; data: ProposalRejectedPayload }
  | { type: 'VoteCast'; data: VoteCastPayload }
  | { type: 'ExecutionRequested'; data: ExecutionRequestedPayload }
  | { type: 'ExecutionCompleted'; data: ExecutionCompletedPayload }
  | { type: 'DelegationChanged'; data: DelegationChangedPayload };

// ============================================================================
// Justice Bridge Signals
// ============================================================================

export type JusticeSignalType =
  | 'DisputeFiled'
  | 'DisputeResolved'
  | 'EnforcementRequested'
  | 'EnforcementExecuted'
  | 'AppealFiled'
  | 'MediationInitiated';

export interface DisputeFiledPayload extends BaseSignalPayload {
  dispute_id: string;
  complainant_did: string;
  respondent_did: string;
  dispute_type: string;
  title: string;
}

export interface DisputeResolvedPayload extends BaseSignalPayload {
  dispute_id: string;
  resolution: 'ForComplainant' | 'ForRespondent' | 'Settled' | 'Dismissed';
  damages_awarded?: number;
}

export interface EnforcementRequestedPayload extends BaseSignalPayload {
  enforcement_id: string;
  dispute_id: string;
  target_happ: string;
  target_did: string;
  action_type: string;
}

export interface EnforcementExecutedPayload extends BaseSignalPayload {
  enforcement_id: string;
  target_happ: string;
  success: boolean;
  action_taken: string;
}

export interface AppealFiledPayload extends BaseSignalPayload {
  appeal_id: string;
  dispute_id: string;
  appellant_did: string;
  grounds: string;
}

export interface MediationInitiatedPayload extends BaseSignalPayload {
  mediation_id: string;
  dispute_id: string;
  mediator_did: string;
}

export type JusticeSignalPayload =
  | { type: 'DisputeFiled'; data: DisputeFiledPayload }
  | { type: 'DisputeResolved'; data: DisputeResolvedPayload }
  | { type: 'EnforcementRequested'; data: EnforcementRequestedPayload }
  | { type: 'EnforcementExecuted'; data: EnforcementExecutedPayload }
  | { type: 'AppealFiled'; data: AppealFiledPayload }
  | { type: 'MediationInitiated'; data: MediationInitiatedPayload };

// ============================================================================
// Knowledge Bridge Signals
// ============================================================================

export type KnowledgeSignalType =
  | 'ClaimCreated'
  | 'ClaimVerified'
  | 'ClaimDisputed'
  | 'EvidenceAdded'
  | 'SynthesisCompleted'
  | 'FactCheckResult';

export interface ClaimCreatedPayload extends BaseSignalPayload {
  claim_hash: string;
  author_did: string;
  title: string;
  empirical: number;
  normative: number;
  mythic: number;
}

export interface ClaimVerifiedPayload extends BaseSignalPayload {
  claim_hash: string;
  verifier_did: string;
  verification_score: number;
  evidence_count: number;
}

export interface ClaimDisputedPayload extends BaseSignalPayload {
  claim_hash: string;
  disputer_did: string;
  counter_claim_hash?: string;
  reason: string;
}

export interface EvidenceAddedPayload extends BaseSignalPayload {
  claim_hash: string;
  evidence_hash: string;
  evidence_type: string;
  contributor_did: string;
}

export interface SynthesisCompletedPayload extends BaseSignalPayload {
  synthesis_id: string;
  input_claims: string[];
  result_claim_hash: string;
  confidence: number;
}

export interface FactCheckResultPayload extends BaseSignalPayload {
  request_id: string;
  claim_text: string;
  verdict: 'true' | 'false' | 'partially_true' | 'unverifiable';
  confidence: number;
  supporting_claims: string[];
}

export type KnowledgeSignalPayload =
  | { type: 'ClaimCreated'; data: ClaimCreatedPayload }
  | { type: 'ClaimVerified'; data: ClaimVerifiedPayload }
  | { type: 'ClaimDisputed'; data: ClaimDisputedPayload }
  | { type: 'EvidenceAdded'; data: EvidenceAddedPayload }
  | { type: 'SynthesisCompleted'; data: SynthesisCompletedPayload }
  | { type: 'FactCheckResult'; data: FactCheckResultPayload };

// ============================================================================
// Bridge Signal Handlers
// ============================================================================

/**
 * Typed signal handler for Identity bridge events
 */
export class IdentitySignalHandler {
  private handler: SignalHandler<IdentitySignalPayload>;

  constructor(options: SignalSubscriptionOptions = {}) {
    this.handler = new SignalHandler({
      zomeFilter: options.zomeFilter || 'identity_bridge',
      bufferSize: options.bufferSize,
    });
  }

  /** Subscribe to identity created events */
  onIdentityCreated(callback: (data: IdentityCreatedPayload) => void): () => void {
    return this.handler.on('IdentityCreated', (signal) => {
      if (signal.payload.type === 'IdentityCreated') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to identity updated events */
  onIdentityUpdated(callback: (data: IdentityUpdatedPayload) => void): () => void {
    return this.handler.on('IdentityUpdated', (signal) => {
      if (signal.payload.type === 'IdentityUpdated') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to credential issued events */
  onCredentialIssued(callback: (data: CredentialIssuedPayload) => void): () => void {
    return this.handler.on('CredentialIssued', (signal) => {
      if (signal.payload.type === 'CredentialIssued') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to credential revoked events */
  onCredentialRevoked(callback: (data: CredentialRevokedPayload) => void): () => void {
    return this.handler.on('CredentialRevoked', (signal) => {
      if (signal.payload.type === 'CredentialRevoked') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to trust attested events */
  onTrustAttested(callback: (data: TrustAttestedPayload) => void): () => void {
    return this.handler.on('TrustAttested', (signal) => {
      if (signal.payload.type === 'TrustAttested') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to all identity events */
  onAny(callback: (signal: HolochainSignal<IdentitySignalPayload>) => void): () => void {
    return this.handler.onAny(callback);
  }

  /** Process incoming signal from conductor */
  handle(signal: HolochainSignal<IdentitySignalPayload>): void {
    this.handler.handle(signal);
  }

  /** Get recent signals */
  getBuffer(): readonly HolochainSignal<IdentitySignalPayload>[] {
    return this.handler.getBuffer();
  }
}

/**
 * Typed signal handler for Finance bridge events
 */
export class FinanceSignalHandler {
  private handler: SignalHandler<FinanceSignalPayload>;

  constructor(options: SignalSubscriptionOptions = {}) {
    this.handler = new SignalHandler({
      zomeFilter: options.zomeFilter || 'finance_bridge',
      bufferSize: options.bufferSize,
    });
  }

  /** Subscribe to payment processed events */
  onPaymentProcessed(callback: (data: PaymentProcessedPayload) => void): () => void {
    return this.handler.on('PaymentProcessed', (signal) => {
      if (signal.payload.type === 'PaymentProcessed') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to loan approved events */
  onLoanApproved(callback: (data: LoanApprovedPayload) => void): () => void {
    return this.handler.on('LoanApproved', (signal) => {
      if (signal.payload.type === 'LoanApproved') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to loan defaulted events */
  onLoanDefaulted(callback: (data: LoanDefaultedPayload) => void): () => void {
    return this.handler.on('LoanDefaulted', (signal) => {
      if (signal.payload.type === 'LoanDefaulted') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to credit score updated events */
  onCreditScoreUpdated(callback: (data: CreditScoreUpdatedPayload) => void): () => void {
    return this.handler.on('CreditScoreUpdated', (signal) => {
      if (signal.payload.type === 'CreditScoreUpdated') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to all finance events */
  onAny(callback: (signal: HolochainSignal<FinanceSignalPayload>) => void): () => void {
    return this.handler.onAny(callback);
  }

  /** Process incoming signal */
  handle(signal: HolochainSignal<FinanceSignalPayload>): void {
    this.handler.handle(signal);
  }

  getBuffer(): readonly HolochainSignal<FinanceSignalPayload>[] {
    return this.handler.getBuffer();
  }
}

/**
 * Typed signal handler for Energy bridge events
 */
export class EnergySignalHandler {
  private handler: SignalHandler<EnergySignalPayload>;

  constructor(options: SignalSubscriptionOptions = {}) {
    this.handler = new SignalHandler({
      zomeFilter: options.zomeFilter || 'energy_bridge',
      bufferSize: options.bufferSize,
    });
  }

  /** Subscribe to energy listed events */
  onEnergyListed(callback: (data: EnergyListedPayload) => void): () => void {
    return this.handler.on('EnergyListed', (signal) => {
      if (signal.payload.type === 'EnergyListed') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to energy purchased events */
  onEnergyPurchased(callback: (data: EnergyPurchasedPayload) => void): () => void {
    return this.handler.on('EnergyPurchased', (signal) => {
      if (signal.payload.type === 'EnergyPurchased') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to grid balance request events */
  onGridBalanceRequest(callback: (data: GridBalanceRequestPayload) => void): () => void {
    return this.handler.on('GridBalanceRequest', (signal) => {
      if (signal.payload.type === 'GridBalanceRequest') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to all energy events */
  onAny(callback: (signal: HolochainSignal<EnergySignalPayload>) => void): () => void {
    return this.handler.onAny(callback);
  }

  handle(signal: HolochainSignal<EnergySignalPayload>): void {
    this.handler.handle(signal);
  }

  getBuffer(): readonly HolochainSignal<EnergySignalPayload>[] {
    return this.handler.getBuffer();
  }
}

/**
 * Typed signal handler for Governance bridge events
 */
export class GovernanceSignalHandler {
  private handler: SignalHandler<GovernanceSignalPayload>;

  constructor(options: SignalSubscriptionOptions = {}) {
    this.handler = new SignalHandler({
      zomeFilter: options.zomeFilter || 'governance_bridge',
      bufferSize: options.bufferSize,
    });
  }

  /** Subscribe to proposal created events */
  onProposalCreated(callback: (data: ProposalCreatedPayload) => void): () => void {
    return this.handler.on('ProposalCreated', (signal) => {
      if (signal.payload.type === 'ProposalCreated') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to proposal passed events */
  onProposalPassed(callback: (data: ProposalPassedPayload) => void): () => void {
    return this.handler.on('ProposalPassed', (signal) => {
      if (signal.payload.type === 'ProposalPassed') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to vote cast events */
  onVoteCast(callback: (data: VoteCastPayload) => void): () => void {
    return this.handler.on('VoteCast', (signal) => {
      if (signal.payload.type === 'VoteCast') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to all governance events */
  onAny(callback: (signal: HolochainSignal<GovernanceSignalPayload>) => void): () => void {
    return this.handler.onAny(callback);
  }

  handle(signal: HolochainSignal<GovernanceSignalPayload>): void {
    this.handler.handle(signal);
  }

  getBuffer(): readonly HolochainSignal<GovernanceSignalPayload>[] {
    return this.handler.getBuffer();
  }
}

/**
 * Typed signal handler for Knowledge bridge events
 */
export class KnowledgeSignalHandler {
  private handler: SignalHandler<KnowledgeSignalPayload>;

  constructor(options: SignalSubscriptionOptions = {}) {
    this.handler = new SignalHandler({
      zomeFilter: options.zomeFilter || 'knowledge_bridge',
      bufferSize: options.bufferSize,
    });
  }

  /** Subscribe to claim created events */
  onClaimCreated(callback: (data: ClaimCreatedPayload) => void): () => void {
    return this.handler.on('ClaimCreated', (signal) => {
      if (signal.payload.type === 'ClaimCreated') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to claim verified events */
  onClaimVerified(callback: (data: ClaimVerifiedPayload) => void): () => void {
    return this.handler.on('ClaimVerified', (signal) => {
      if (signal.payload.type === 'ClaimVerified') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to fact check result events */
  onFactCheckResult(callback: (data: FactCheckResultPayload) => void): () => void {
    return this.handler.on('FactCheckResult', (signal) => {
      if (signal.payload.type === 'FactCheckResult') {
        callback(signal.payload.data);
      }
    });
  }

  /** Subscribe to all knowledge events */
  onAny(callback: (signal: HolochainSignal<KnowledgeSignalPayload>) => void): () => void {
    return this.handler.onAny(callback);
  }

  handle(signal: HolochainSignal<KnowledgeSignalPayload>): void {
    this.handler.handle(signal);
  }

  getBuffer(): readonly HolochainSignal<KnowledgeSignalPayload>[] {
    return this.handler.getBuffer();
  }
}

// ============================================================================
// Unified Bridge Signal Manager
// ============================================================================

/**
 * Unified signal manager for all Mycelix bridge events.
 * Automatically routes signals to the appropriate domain handler.
 *
 * @example
 * ```typescript
 * import { BridgeSignalManager, createClient } from '@mycelix/sdk';
 *
 * const client = createClient({ installedAppId: 'mycelix' });
 * await client.connect();
 *
 * const signals = new BridgeSignalManager(client);
 *
 * // Subscribe to identity events
 * signals.identity.onIdentityCreated((data) => {
 *   console.log('New identity:', data.did);
 * });
 *
 * // Subscribe to finance events
 * signals.finance.onPaymentProcessed((data) => {
 *   console.log('Payment:', data.amount, data.currency);
 * });
 *
 * // Subscribe to governance events
 * signals.governance.onProposalPassed((data) => {
 *   console.log('Proposal passed:', data.proposal_hash);
 * });
 * ```
 */
export class BridgeSignalManager {
  readonly identity: IdentitySignalHandler;
  readonly finance: FinanceSignalHandler;
  readonly energy: EnergySignalHandler;
  readonly governance: GovernanceSignalHandler;
  readonly knowledge: KnowledgeSignalHandler;

  private client: MycelixClient | null = null;
  private unsubscribe: (() => void) | null = null;

  constructor(client?: MycelixClient, options: SignalSubscriptionOptions = {}) {
    this.identity = new IdentitySignalHandler(options);
    this.finance = new FinanceSignalHandler(options);
    this.energy = new EnergySignalHandler(options);
    this.governance = new GovernanceSignalHandler(options);
    this.knowledge = new KnowledgeSignalHandler(options);

    if (client) {
      this.connect(client);
    }
  }

  /**
   * Connect to a Mycelix client and start receiving signals
   */
  connect(client: MycelixClient): void {
    this.client = client;

    // Subscribe to conductor signals
    this.unsubscribe = client.onSignal((signal) => {
      this.routeSignal(signal);
    });
  }

  /**
   * Disconnect from the client
   */
  disconnect(): void {
    if (this.unsubscribe) {
      this.unsubscribe();
      this.unsubscribe = null;
    }
    this.client = null;
  }

  /**
   * Route a signal to the appropriate domain handler
   */
  private routeSignal(signal: unknown): void {
    // Type guard for HolochainSignal structure
    if (!signal || typeof signal !== 'object') return;

    const s = signal as { zomeName?: string; signalName?: string; payload?: unknown };

    if (!s.zomeName || !s.payload) return;

    // Route based on zome name
    switch (s.zomeName) {
      case 'identity_bridge':
        this.identity.handle(signal as HolochainSignal<IdentitySignalPayload>);
        break;
      case 'finance_bridge':
        this.finance.handle(signal as HolochainSignal<FinanceSignalPayload>);
        break;
      case 'energy_bridge':
        this.energy.handle(signal as HolochainSignal<EnergySignalPayload>);
        break;
      case 'governance_bridge':
        this.governance.handle(signal as HolochainSignal<GovernanceSignalPayload>);
        break;
      case 'knowledge_bridge':
        this.knowledge.handle(signal as HolochainSignal<KnowledgeSignalPayload>);
        break;
    }
  }

  /**
   * Check if connected to a client
   */
  isConnected(): boolean {
    return this.client !== null && this.unsubscribe !== null;
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new BridgeSignalManager
 */
export function createSignalManager(
  client?: MycelixClient,
  options?: SignalSubscriptionOptions
): BridgeSignalManager {
  return new BridgeSignalManager(client, options);
}

/**
 * Create an identity-only signal handler
 */
export function createIdentitySignals(options?: SignalSubscriptionOptions): IdentitySignalHandler {
  return new IdentitySignalHandler(options);
}

/**
 * Create a finance-only signal handler
 */
export function createFinanceSignals(options?: SignalSubscriptionOptions): FinanceSignalHandler {
  return new FinanceSignalHandler(options);
}

/**
 * Create an energy-only signal handler
 */
export function createEnergySignals(options?: SignalSubscriptionOptions): EnergySignalHandler {
  return new EnergySignalHandler(options);
}

/**
 * Create a governance-only signal handler
 */
export function createGovernanceSignals(options?: SignalSubscriptionOptions): GovernanceSignalHandler {
  return new GovernanceSignalHandler(options);
}

/**
 * Create a knowledge-only signal handler
 */
export function createKnowledgeSignals(options?: SignalSubscriptionOptions): KnowledgeSignalHandler {
  return new KnowledgeSignalHandler(options);
}
