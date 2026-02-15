/**
 * Finance hApp Client Types
 *
 * Type definitions for all Finance zome clients in the Mycelix ecosystem.
 *
 * @module @mycelix/sdk/clients/finance/types
 */

// ActionHash import removed (unused)

// =============================================================================
// Common Types
// =============================================================================

/** Timestamp in microseconds (Holochain format) */
export type Timestamp = number;

/** DID string identifier */
export type Did = string;

/** Currency types supported by the Finance hApp */
export type Currency = 'MYC' | 'USD' | 'EUR' | 'BTC' | 'ETH' | 'USDC' | 'ENERGY';

// =============================================================================
// Lending Types (Deprecated - use TEND/HEARTH/CGC instead)
// =============================================================================

/** Loan status */
export type LoanStatus =
  | 'Requested'
  | 'Funded'
  | 'Active'
  | 'Repaid'
  | 'Defaulted'
  | 'Cancelled';

/** A loan record */
export interface Loan {
  id: string;
  borrower_did: Did;
  lender_did: Did;
  principal: number;
  currency: string;
  interest_rate: number;
  term_days: number;
  collateral_ids: string[];
  status: LoanStatus;
  created: Timestamp;
  funded?: Timestamp;
  maturity?: Timestamp;
  repaid?: Timestamp;
}

/** Input for requesting a loan */
export interface RequestLoanInput {
  borrower_did: Did;
  amount: number;
  currency: string;
  term_days: number;
  collateral_ids: string[];
}

/** A loan offer */
export interface LoanOffer {
  id: string;
  lender_did: Did;
  max_amount: number;
  min_amount: number;
  currency: string;
  base_interest_rate: number;
  min_credit_score: number;
  max_term_days: number;
  collateral_required: boolean;
  active: boolean;
  created: Timestamp;
}

/** Input for creating a loan offer */
export interface CreateOfferInput {
  lender_did: Did;
  max_amount: number;
  min_amount: number;
  currency: string;
  base_interest_rate: number;
  min_credit_score: number;
  max_term_days: number;
  collateral_required: boolean;
}

/** Input for funding a loan */
export interface FundLoanInput {
  loan_id: string;
  lender_did: Did;
  interest_rate: number;
}

/** Payment schedule */
export interface PaymentSchedule {
  loan_id: string;
  payments: ScheduledPayment[];
}

/** A scheduled payment */
export interface ScheduledPayment {
  payment_number: number;
  due_date: Timestamp;
  principal_amount: number;
  interest_amount: number;
  total_amount: number;
  paid: boolean;
}

/** Input for creating payment schedule */
export interface CreateScheduleInput {
  loan_id: string;
  num_payments: number;
}

/** Input for matching offers */
export interface MatchOffersInput {
  credit_score: number;
  amount: number;
  currency: string;
  term_days: number;
  has_collateral: boolean;
}

/** Credit tier */
export type CreditTier = 'Excellent' | 'Good' | 'Standard' | 'BasicAccess';

/** Credit assessment result */
export interface CreditAssessmentResult {
  borrower_did: Did;
  credit_score: number;
  credit_tier: CreditTier;
  components: CreditAssessmentComponents;
  rate_adjustment: number;
  floor_applied: boolean;
  transparency: TransparencyReport;
  calculated_at: Timestamp;
  valid_until: Timestamp;
}

/** Credit assessment components */
export interface CreditAssessmentComponents {
  payment_history: ComponentDetail;
  collateral_ratio: ComponentDetail;
  account_age: ComponentDetail;
}

/** Component detail */
export interface ComponentDetail {
  score: number;
  weight: number;
  contribution: number;
  details: string;
}

/** Transparency report */
export interface TransparencyReport {
  data_sources_used: string[];
  data_sources_excluded: string[];
  decay_applied: boolean;
  decay_halflife_days: number;
  floor_applied: boolean;
  appeal_available: boolean;
}

/** Privacy proof types */
export type ProofType =
  | { MeetsMinimum: null }
  | { WithinRange: { low: number; high: number } }
  | { TierMembership: null };

/** Privacy credit proof */
export interface PrivacyCreditProof {
  proof_id: string;
  borrower_did: Did;
  proof_type: ProofType;
  threshold: number;
  result: boolean;
  commitment: string;
  proof_data: string;
  generated_at: Timestamp;
  valid_until: Timestamp;
  reveals: string[];
  conceals: string[];
}

// =============================================================================
// Credit Scoring Types (Deprecated - use CGC instead)
// =============================================================================

/** Credit profile */
export interface CreditProfile {
  did: Did;
  matl_score: number;
  payment_history_score: number;
  collateral_ratio: number;
  account_age_days: number;
  activity_score: number;
  computed_score: number;
  last_updated: Timestamp;
  version: number;
}

/** Input for creating credit profile */
export interface CreateProfileInput {
  did: Did;
  matl_score: number;
  collateral_ratio: number;
}

/** Payment status */
export type PaymentStatus = 'Pending' | 'OnTime' | 'Late' | 'Missed';

/** Payment record */
export interface PaymentRecord {
  id: string;
  profile_did: Did;
  loan_id: string;
  amount: number;
  currency: string;
  due_date: Timestamp;
  paid_date?: Timestamp;
  status: PaymentStatus;
}

/** Input for recording payment */
export interface RecordPaymentInput {
  profile_did: Did;
  loan_id: string;
  amount: number;
  currency: string;
  due_date: Timestamp;
  paid_date?: Timestamp;
}

/** Collateral type */
export type CollateralType =
  | 'RealEstate'
  | 'Vehicle'
  | 'Cryptocurrency'
  | 'NFT'
  | 'Equipment'
  | 'Inventory'
  | 'Other';

/** Collateral record */
export interface CollateralRecord {
  id: string;
  owner_did: Did;
  asset_type: CollateralType;
  asset_id: string;
  appraised_value: number;
  currency: string;
  locked_for_loan?: string;
  registered: Timestamp;
}

/** Input for registering collateral */
export interface RegisterCollateralInput {
  owner_did: Did;
  asset_type: CollateralType;
  asset_id: string;
  appraised_value: number;
  currency: string;
}

/** Input for locking collateral */
export interface LockCollateralInput {
  collateral_id: string;
  loan_id: string;
}

/** Input for updating MATL score */
export interface UpdateMatlInput {
  did: Did;
  matl_score: number;
}

/** Input for score range query */
export interface ScoreRangeInput {
  min_score: number;
  max_score: number;
}

// =============================================================================
// Payments Types
// =============================================================================

/** Payment type */
export type PaymentType =
  | 'Direct'
  | 'Scheduled'
  | { Escrow: string }
  | 'Channel';

/** Transfer status */
export type TransferStatus = 'Pending' | 'Completed' | 'Failed' | 'Refunded';

/** A payment */
export interface Payment {
  id: string;
  from_did: Did;
  to_did: Did;
  amount: number;
  currency: string;
  payment_type: PaymentType;
  status: TransferStatus;
  memo?: string;
  created: Timestamp;
  completed?: Timestamp;
}

/** Input for sending payment */
export interface SendPaymentInput {
  from_did: Did;
  to_did: Did;
  amount: number;
  currency: string;
  payment_type: PaymentType;
  memo?: string;
}

/** A payment channel */
export interface PaymentChannel {
  id: string;
  party_a: Did;
  party_b: Did;
  currency: string;
  balance_a: number;
  balance_b: number;
  opened: Timestamp;
  last_updated: Timestamp;
  closed?: Timestamp;
}

/** Input for opening channel */
export interface OpenChannelInput {
  party_a: Did;
  party_b: Did;
  currency: string;
  initial_deposit_a: number;
  initial_deposit_b: number;
}

/** Input for channel transfer */
export interface ChannelTransferInput {
  channel_id: string;
  amount: number;
  from_a: boolean;
}

/** A payment receipt */
export interface Receipt {
  payment_id: string;
  from_did: Did;
  to_did: Did;
  amount: number;
  currency: string;
  timestamp: Timestamp;
  signature: string;
}

/** Input for creating escrow */
export interface CreateEscrowPaymentInput {
  from_did: Did;
  to_did: Did;
  amount: number;
  currency: string;
  escrow_id: string;
  memo?: string;
}

// =============================================================================
// Treasury Types
// =============================================================================

/** Treasury */
export interface Treasury {
  id: string;
  name: string;
  description: string;
  currency: string;
  balance: number;
  reserve_ratio: number;
  managers: Did[];
  created: Timestamp;
  last_updated: Timestamp;
}

/** Input for creating treasury */
export interface CreateTreasuryInput {
  name: string;
  description: string;
  currency: string;
  reserve_ratio: number;
  managers: Did[];
}

/** Contribution type */
export type ContributionType = 'Donation' | 'Dues' | 'Grant' | 'Fee' | 'Other';

/** A contribution */
export interface Contribution {
  id: string;
  treasury_id: string;
  contributor_did: Did;
  amount: number;
  currency: string;
  contribution_type: ContributionType;
  timestamp: Timestamp;
}

/** Input for contributing */
export interface ContributeInput {
  treasury_id: string;
  contributor_did: Did;
  amount: number;
  currency: string;
  contribution_type: ContributionType;
}

/** Allocation status */
export type AllocationStatus =
  | 'Proposed'
  | 'Approved'
  | 'Rejected'
  | 'Executed'
  | 'Cancelled';

/** An allocation */
export interface Allocation {
  id: string;
  treasury_id: string;
  proposal_id?: string;
  recipient_did: Did;
  amount: number;
  currency: string;
  purpose: string;
  status: AllocationStatus;
  approved_by: Did[];
  created: Timestamp;
  executed?: Timestamp;
}

/** Input for proposing allocation */
export interface ProposeAllocationInput {
  treasury_id: string;
  proposal_id?: string;
  recipient_did: Did;
  amount: number;
  currency: string;
  purpose: string;
}

/** Input for approving allocation */
export interface ApproveAllocationInput {
  allocation_id: string;
  approver_did: Did;
}

/** Input for rejecting allocation */
export interface RejectAllocationInput {
  allocation_id: string;
  rejector_did: Did;
}

/** Input for cancelling allocation */
export interface CancelAllocationInput {
  allocation_id: string;
  cancelled_by_did: Did;
}

/** Input for adding manager */
export interface AddManagerInput {
  treasury_id: string;
  new_manager_did: Did;
  added_by_did: Did;
}

/** Input for removing manager */
export interface RemoveManagerInput {
  treasury_id: string;
  manager_did: Did;
  removed_by_did: Did;
}

/** Input for allocation status query */
export interface AllocationStatusQuery {
  treasury_id: string;
  status: AllocationStatus;
}

/** Savings pool */
export interface SavingsPool {
  id: string;
  treasury_id: string;
  name: string;
  target_amount: number;
  current_amount: number;
  currency: string;
  members: Did[];
  yield_rate: number;
  created: Timestamp;
}

/** Input for creating pool */
export interface CreatePoolInput {
  treasury_id: string;
  name: string;
  target_amount: number;
  currency: string;
  initial_members: Did[];
  yield_rate: number;
}

/** Input for joining pool */
export interface JoinPoolInput {
  pool_id: string;
  member_did: Did;
}

/** Input for pool contribution */
export interface PoolContributionInput {
  pool_id: string;
  contributor_did: Did;
  amount: number;
}

// =============================================================================
// CGC (Civic Gifting Credits) Types
// =============================================================================

/** CGC contribution type */
export type CgcContributionType =
  | 'CodeContribution'
  | 'Documentation'
  | 'CommunitySupport'
  | 'Mentorship'
  | 'EventOrganization'
  | 'ContentCreation'
  | 'Other';

/** CGC allocation */
export interface CgcAllocation {
  member_did: Did;
  cycle_id: string;
  total_allocated: number;
  remaining: number;
  gifted: number;
  created: Timestamp;
  expires: Timestamp;
}

/** CGC allocation status */
export interface CgcAllocationStatus {
  member_did: Did;
  cycle_id: string;
  total_allocated: number;
  remaining: number;
  gifted: number;
  expires_at: Timestamp;
}

/** CGC gift */
export interface CgcGift {
  id: string;
  giver_did: Did;
  receiver_did: Did;
  amount: number;
  gratitude_message: string;
  contribution_type: CgcContributionType;
  cultural_alias?: string;
  cycle_id: string;
  timestamp: Timestamp;
}

/** Input for gifting CGC */
export interface GiftCgcInput {
  receiver_did: Did;
  amount: number;
  gratitude_message: string;
  contribution_type: CgcContributionType;
  cultural_alias?: string;
}

/** Gift record */
export interface GiftRecord {
  id: string;
  giver_did: Did;
  receiver_did: Did;
  amount: number;
  gratitude_message: string;
  contribution_type: CgcContributionType;
  timestamp: Timestamp;
}

/** Recognition summary */
export interface RecognitionSummary {
  member_did: Did;
  cycle_id: string;
  total_received: number;
  unique_givers: number;
  by_contribution_type: [string, number][];
  is_flagged: boolean;
}

/** Cultural alias */
export interface CulturalAlias {
  alias: string;
  dao_did: Did;
  meaning: string;
  registered_at: Timestamp;
}

/** Input for registering alias */
export interface RegisterAliasInput {
  alias: string;
  dao_did: Did;
  meaning: string;
}

// =============================================================================
// TEND (Time Exchange) Types
// =============================================================================

/** Service category */
export type ServiceCategory =
  | 'Healthcare'
  | 'Education'
  | 'Technology'
  | 'Crafts'
  | 'Transport'
  | 'Household'
  | 'Professional'
  | 'Creative'
  | 'Other';

/** Exchange status */
export type ExchangeStatus =
  | 'Proposed'
  | 'Confirmed'
  | 'Disputed'
  | 'Cancelled';

/** TEND exchange */
export interface TendExchange {
  id: string;
  provider_did: Did;
  receiver_did: Did;
  hours: number;
  service_description: string;
  service_category: ServiceCategory;
  cultural_alias?: string;
  dao_did: Did;
  timestamp: Timestamp;
  status: ExchangeStatus;
  service_date?: Timestamp;
}

/** Input for recording exchange */
export interface RecordExchangeInput {
  receiver_did: Did;
  hours: number;
  service_description: string;
  service_category: ServiceCategory;
  cultural_alias?: string;
  dao_did: Did;
  service_date?: Timestamp;
}

/** Exchange record */
export interface ExchangeRecord {
  id: string;
  provider_did: Did;
  receiver_did: Did;
  hours: number;
  service_description: string;
  service_category: ServiceCategory;
  status: ExchangeStatus;
  timestamp: Timestamp;
}

/** TEND balance */
export interface TendBalance {
  member_did: Did;
  dao_did: Did;
  balance: number;
  total_provided: number;
  total_received: number;
  exchange_count: number;
  last_activity: Timestamp;
}

/** Balance info */
export interface BalanceInfo {
  member_did: Did;
  dao_did: Did;
  balance: number;
  can_provide: boolean;
  can_receive: boolean;
  total_provided: number;
  total_received: number;
  exchange_count: number;
}

/** Input for getting balance */
export interface GetBalanceInput {
  member_did: Did;
  dao_did: Did;
}

/** Service listing */
export interface ServiceListing {
  id: string;
  provider_did: Did;
  dao_did: Did;
  title: string;
  description: string;
  category: ServiceCategory;
  estimated_hours?: number;
  availability?: string;
  active: boolean;
  created: Timestamp;
}

/** Input for creating listing */
export interface CreateListingInput {
  dao_did: Did;
  title: string;
  description: string;
  category: ServiceCategory;
  estimated_hours?: number;
  availability?: string;
}

/** Urgency level */
export type Urgency = 'Low' | 'Medium' | 'High' | 'Urgent';

/** Service request */
export interface ServiceRequest {
  id: string;
  requester_did: Did;
  dao_did: Did;
  title: string;
  description: string;
  category: ServiceCategory;
  estimated_hours?: number;
  urgency: Urgency;
  open: boolean;
  created: Timestamp;
}

/** Input for creating request */
export interface CreateRequestInput {
  dao_did: Did;
  title: string;
  description: string;
  category: ServiceCategory;
  estimated_hours?: number;
  urgency: Urgency;
}

// =============================================================================
// HEARTH (Commons Pool) Types
// =============================================================================

/** Resource type */
export type ResourceType = 'Money' | 'Time' | 'Equipment' | 'Space' | 'Skills' | 'Other';

/** Pool status */
export type PoolStatus = 'Proposed' | 'Active' | 'Paused' | 'Closed';

/** Commons pool */
export interface CommonsPool {
  id: string;
  name: string;
  description: string;
  dao_did: Did;
  resource_type: ResourceType;
  cultural_alias?: string;
  total_resources: number;
  lifetime_contributions: number;
  lifetime_allocations: number;
  contributor_count: number;
  status: PoolStatus;
  governance: PoolGovernance;
  created: Timestamp;
}

/** Pool governance */
export interface PoolGovernance {
  allocation_threshold: number;
  voting_period_hours: number;
  contribution_weighted_voting: boolean;
  min_civ_to_request?: number;
  max_allocation_percent: number;
  require_justification: boolean;
  request_cooldown_hours: number;
}

/** Pool governance input */
export interface PoolGovernanceInput {
  allocation_threshold?: number;
  voting_period_hours?: number;
  contribution_weighted_voting?: boolean;
  min_civ_to_request?: number;
  max_allocation_percent?: number;
  require_justification?: boolean;
  request_cooldown_hours?: number;
}

/** Input for creating pool */
export interface CreateCommonsPoolInput {
  name: string;
  description: string;
  dao_did: Did;
  resource_type: ResourceType;
  cultural_alias?: string;
  governance?: PoolGovernanceInput;
}

/** Pool summary */
export interface PoolSummary {
  id: string;
  name: string;
  description: string;
  dao_did: Did;
  resource_type: ResourceType;
  total_resources: number;
  contributor_count: number;
  status: PoolStatus;
  pending_requests: number;
}

/** Pool contribution */
export interface PoolContribution {
  id: string;
  pool_id: string;
  contributor_did: Did;
  amount: number;
  message?: string;
  anonymous: boolean;
  timestamp: Timestamp;
}

/** Input for contributing to pool */
export interface ContributeToPoolInput {
  pool_id: string;
  amount: number;
  message?: string;
  anonymous: boolean;
}

/** Need category */
export type NeedCategory =
  | 'Emergency'
  | 'Healthcare'
  | 'Education'
  | 'Housing'
  | 'Childcare'
  | 'Transportation'
  | 'Food'
  | 'Other';

/** Request status */
export type RequestStatus = 'Voting' | 'Approved' | 'Rejected' | 'Disbursed' | 'Cancelled';

/** Allocation request */
export interface AllocationRequest {
  id: string;
  pool_id: string;
  requester_did: Did;
  recipient_did: Did;
  amount: number;
  justification: string;
  need_category: NeedCategory;
  status: RequestStatus;
  voting_deadline: Timestamp;
  votes_for: number;
  votes_against: number;
  total_vote_weight: number;
  created: Timestamp;
  resolved?: Timestamp;
}

/** Input for requesting allocation */
export interface RequestAllocationInput {
  pool_id: string;
  recipient_did?: Did;
  amount: number;
  justification: string;
  need_category: NeedCategory;
}

/** Request summary */
export interface RequestSummary {
  id: string;
  pool_id: string;
  requester_did: Did;
  recipient_did: Did;
  amount: number;
  justification: string;
  need_category: NeedCategory;
  status: RequestStatus;
  votes_for: number;
  votes_against: number;
  voting_deadline: Timestamp;
}

/** Allocation vote */
export interface AllocationVote {
  request_id: string;
  voter_did: Did;
  vote: boolean;
  weight: number;
  comment?: string;
  timestamp: Timestamp;
}

/** Input for voting */
export interface VoteInput {
  request_id: string;
  vote: boolean;
  comment?: string;
}

// =============================================================================
// Staking Types
// =============================================================================

/** Stake status */
export type StakeStatus =
  | 'Active'
  | 'Unbonding'
  | 'Withdrawn'
  | 'Slashed'
  | 'Jailed';

/** Tri-asset stake */
export interface TriAssetStake {
  id: string;
  staker_did: Did;
  myc_amount: bigint;
  eth_amount: bigint;
  usdc_amount: bigint;
  total_value_usd: number;
  kvector_trust_score: number;
  stake_weight: number;
  staked_at: Timestamp;
  unbonding_until?: Timestamp;
  status: StakeStatus;
  myc_lock_proof: Uint8Array;
  eth_lock_proof: Uint8Array;
  usdc_lock_proof: Uint8Array;
  pending_rewards: number;
  last_reward_claim: Timestamp;
}

/** Input for creating stake */
export interface CreateStakeInput {
  staker_did: Did;
  myc_amount: bigint;
  eth_amount: bigint;
  usdc_amount: bigint;
  kvector_trust_score: number;
  myc_lock_proof: Uint8Array;
  eth_lock_proof: Uint8Array;
  usdc_lock_proof: Uint8Array;
}

/** Input for updating trust */
export interface UpdateTrustInput {
  stake_id: string;
  new_trust_score: number;
}

/** Slashing reason */
export type SlashingReason =
  | 'DoubleSign'
  | 'Downtime'
  | 'InvalidBlock'
  | 'Misbehavior'
  | 'GovernanceViolation';

/** Slashing evidence */
export interface SlashingEvidence {
  evidence_type: string;
  data: Uint8Array;
  signatures: Uint8Array[];
  timestamp: Timestamp;
}

/** Input for slashing */
export interface SlashStakeInput {
  stake_id: string;
  reason: SlashingReason;
  evidence: SlashingEvidence;
  custom_slash_percentage?: number;
}

/** Slashing event */
export interface SlashingEvent {
  id: string;
  stake_id: string;
  staker_did: Did;
  reason: SlashingReason;
  slash_percentage: number;
  myc_slashed: bigint;
  eth_slashed: bigint;
  usdc_slashed: bigint;
  evidence_hash: Uint8Array;
  evidence: Uint8Array;
  slashed_at: Timestamp;
  jailed: boolean;
  jail_release?: Timestamp;
}

/** Chain type */
export type ChainType = 'Ethereum' | 'Holochain' | 'Solana' | 'Cosmos';

/** Asset type for staking */
export type StakingAssetType = 'MYC' | 'ETH' | 'USDC';

/** Cross-chain lock proof */
export interface CrossChainLockProof {
  id: string;
  stake_id: string;
  source_chain: ChainType;
  asset: StakingAssetType;
  amount: bigint;
  lock_contract: string;
  lock_tx_hash: Uint8Array;
  lock_block: number;
  merkle_proof: Uint8Array;
  block_header_hash: Uint8Array;
  validator_signatures: Uint8Array[];
  created_at: Timestamp;
  expires_at: Timestamp;
  verified: boolean;
}

/** Input for submitting proof */
export interface SubmitProofInput {
  stake_id: string;
  source_chain: ChainType;
  asset: StakingAssetType;
  amount: bigint;
  lock_contract: string;
  lock_tx_hash: Uint8Array;
  lock_block: number;
  merkle_proof: Uint8Array;
  block_header_hash: Uint8Array;
  validator_signatures: Uint8Array[];
}

/** Escrow status */
export type EscrowStatus = 'Pending' | 'Releasable' | 'Released' | 'Refunded' | 'Expired';

/** Escrow hash type */
export type EscrowHashType = 'Sha256' | 'Sha3_256' | 'Blake2b' | 'Keccak256';

/** Release condition */
export type ReleaseCondition =
  | { HashLock: { hash: Uint8Array; hash_type: EscrowHashType } }
  | { TimeLock: { release_after: Timestamp } }
  | { MultiSig: { threshold: number; signers: Did[] } }
  | { OracleAttestation: { oracle_did: Did; condition: string } };

/** Escrow signature */
export interface EscrowSignature {
  signer_did: Did;
  signature: Uint8Array;
  signed_at: number;
}

/** Crypto escrow */
export interface CryptoEscrow {
  id: string;
  depositor_did: Did;
  beneficiary_did: Did;
  myc_amount: bigint;
  eth_amount: bigint;
  usdc_amount: bigint;
  purpose: string;
  conditions: ReleaseCondition[];
  required_conditions: number;
  met_conditions: number[];
  hash_lock?: Uint8Array;
  timelock?: Timestamp;
  multisig_threshold?: number;
  multisig_signers: Did[];
  collected_signatures: EscrowSignature[];
  status: EscrowStatus;
  created_at: Timestamp;
  released_at?: Timestamp;
}

/** Input for creating escrow */
export interface CreateCryptoEscrowInput {
  depositor_did: Did;
  beneficiary_did: Did;
  myc_amount: bigint;
  eth_amount: bigint;
  usdc_amount: bigint;
  purpose: string;
  conditions: ReleaseCondition[];
  required_conditions: number;
  hash_lock?: Uint8Array;
  timelock?: number;
  multisig_threshold?: number;
  multisig_signers: Did[];
}

/** Input for revealing preimage */
export interface RevealPreimageInput {
  escrow_id: string;
  preimage: Uint8Array;
}

/** Input for adding signature */
export interface AddSignatureInput {
  escrow_id: string;
  signer_did: Did;
  signature: Uint8Array;
}

// =============================================================================
// Bridge Types
// =============================================================================

/** Credit purpose */
export type CreditPurpose = 'LoanApplication' | 'RiskAssessment' | 'ContractExecution' | 'Audit';

/** Credit query */
export interface CreditQuery {
  id: string;
  did: Did;
  source_happ: string;
  purpose: CreditPurpose;
  queried_at: Timestamp;
}

/** Credit result */
export interface CreditResult {
  id: string;
  did: Did;
  matl_score: number;
  credit_score: number;
  payment_history_score: number;
  collateral_ratio: number;
  active_loans: number;
  total_repaid: number;
  calculated_at: Timestamp;
}

/** Input for querying credit */
export interface QueryCreditInput {
  did: Did;
  source_happ: string;
  purpose: CreditPurpose;
}

/** Cross-hApp payment */
export interface CrossHappPayment {
  id: string;
  source_happ: string;
  from_did: Did;
  to_did: Did;
  amount: number;
  currency: string;
  reference: string;
  status: PaymentStatus;
  created_at: Timestamp;
  completed_at?: Timestamp;
}

/** Input for processing payment */
export interface ProcessPaymentInput {
  source_happ: string;
  from_did: Did;
  to_did: Did;
  amount: number;
  currency: string;
  reference: string;
}

/** Asset type for bridge */
export type BridgeAssetType =
  | 'RealEstate'
  | 'Vehicle'
  | 'Cryptocurrency'
  | 'NFT'
  | 'Equipment'
  | 'Inventory'
  | 'Other';

/** Collateral status */
export type CollateralStatus = 'Available' | 'Locked' | 'Released' | 'Seized';

/** Collateral registration */
export interface CollateralRegistration {
  id: string;
  owner_did: Did;
  asset_type: BridgeAssetType;
  asset_id: string;
  source_happ: string;
  value_estimate: number;
  currency: string;
  status: CollateralStatus;
  registered_at: Timestamp;
}

/** Input for registering collateral via bridge */
export interface RegisterBridgeCollateralInput {
  owner_did: Did;
  source_happ: string;
  asset_type: BridgeAssetType;
  asset_id: string;
  value_estimate: number;
  currency: string;
}

/** Finance event type */
export type FinanceEventType =
  | 'PaymentCompleted'
  | 'LoanFunded'
  | 'LoanRepaid'
  | 'CollateralLocked'
  | 'CollateralReleased'
  | 'CreditScoreUpdated';

/** Finance bridge event */
export interface FinanceBridgeEvent {
  id: string;
  event_type: FinanceEventType;
  subject_did: Did;
  amount?: number;
  payload: string;
  source_happ: string;
  timestamp: Timestamp;
}

/** Input for broadcasting event */
export interface BroadcastFinanceEventInput {
  event_type: FinanceEventType;
  subject_did: Did;
  amount?: number;
  payload: string;
}
