/**
 * Mycelix Finance SDK Types
 *
 * Type definitions for the Finance hApp SDK, covering wallets, transactions,
 * credit scoring, lending, treasury, and escrow operations.
 *
 * @module @mycelix/sdk/integrations/finance/types
 */

// ============================================================================
// Wallet Types
// ============================================================================

/** Wallet type */
export type WalletType = 'Personal' | 'Business' | 'Escrow' | 'Treasury';

/** A multi-currency wallet */
export interface Wallet {
  /** Unique wallet identifier */
  id: string;
  /** Owner's DID */
  owner_did: string;
  /** Wallet type */
  wallet_type: WalletType;
  /** Whether wallet is frozen */
  frozen: boolean;
  /** Creation timestamp (microseconds) */
  created_at: number;
}

/** Input for creating a wallet */
export interface CreateWalletInput {
  /** Unique wallet identifier */
  id: string;
  /** Owner's DID */
  owner_did: string;
  /** Wallet type */
  wallet_type: WalletType;
}

/** Balance for a specific currency in a wallet */
export interface Balance {
  /** Wallet ID */
  wallet_id: string;
  /** Currency code */
  currency: string;
  /** Amount (signed for credit/debit) */
  amount: number;
  /** Last update timestamp */
  updated_at: number;
}

/** Query for balance */
export interface BalanceQuery {
  /** Wallet ID */
  wallet_id: string;
  /** Currency code */
  currency: string;
}

// ============================================================================
// Transaction Types
// ============================================================================

/** Transaction status */
export type TransactionStatus = 'Pending' | 'Completed' | 'Failed' | 'Reversed';

/** A fund transfer transaction */
export interface Transaction {
  /** Unique transaction identifier */
  id: string;
  /** Source wallet ID */
  from_wallet: string;
  /** Destination wallet ID */
  to_wallet: string;
  /** Transfer amount */
  amount: number;
  /** Currency code */
  currency: string;
  /** Optional memo */
  memo?: string;
  /** Transaction status */
  status: TransactionStatus;
  /** Creation timestamp */
  created_at: number;
  /** Completion timestamp */
  completed_at?: number;
}

/** Input for transfer */
export interface TransferInput {
  /** Unique transaction identifier */
  id: string;
  /** Source wallet ID */
  from_wallet: string;
  /** Destination wallet ID */
  to_wallet: string;
  /** Transfer amount */
  amount: number;
  /** Currency code */
  currency: string;
  /** Optional memo */
  memo?: string;
}

/** Input for deposit */
export interface DepositInput {
  /** Wallet ID to deposit to */
  wallet_id: string;
  /** Amount to deposit */
  amount: number;
  /** Currency code */
  currency: string;
}

/** Input for withdrawal */
export interface WithdrawInput {
  /** Wallet ID to withdraw from */
  wallet_id: string;
  /** Amount to withdraw */
  amount: number;
  /** Currency code */
  currency: string;
}

// ============================================================================
// Credit Types
// ============================================================================

/** Credit tier based on score */
export type CreditTier = 'Excellent' | 'Good' | 'Fair' | 'Poor' | 'VeryPoor';

/** MATL-based credit score */
export interface CreditScore {
  /** DID of the scored entity */
  did: string;
  /** Numeric score (0-1000) */
  score: number;
  /** Payment history factor (0-1) */
  payment_history: number;
  /** Credit utilization factor (0-1) */
  credit_utilization: number;
  /** Credit age factor (0-1) */
  credit_age: number;
  /** MATL trust score contribution (0-1) */
  matl_score: number;
  /** Collateral ratio factor (0-1) */
  collateral_ratio: number;
  /** Last update timestamp */
  updated_at: number;
}

/** Credit factors breakdown */
export interface CreditFactors {
  /** Payment history contribution (weighted) */
  payment_history_score: number;
  /** Credit utilization contribution (weighted) */
  credit_utilization_score: number;
  /** Credit history length contribution (weighted) */
  credit_history_length: number;
  /** Credit mix contribution (weighted) */
  credit_mix_score: number;
  /** New credit contribution (weighted) */
  new_credit_score: number;
  /** MATL bonus points */
  matl_bonus: number;
  /** Collateral bonus points */
  collateral_bonus: number;
  /** Governance participation bonus */
  governance_participation: number;
}

/** Input for comprehensive credit calculation */
export interface CreditCalculationInput {
  /** DID to calculate credit for */
  did: string;
  /** Number of on-time payments */
  on_time_payments: number;
  /** Number of late payments */
  late_payments: number;
  /** Number of missed payments */
  missed_payments: number;
  /** Current total debt */
  current_debt: number;
  /** Total credit limit */
  total_credit_limit: number;
  /** Credit history in months */
  credit_history_months: number;
  /** Number of loans */
  loan_count: number;
  /** Number of credit lines */
  credit_line_count: number;
  /** Recent credit inquiries */
  recent_inquiries: number;
  /** MATL score (0-1) */
  matl_score: number;
  /** Total collateral value */
  collateral_value: number;
  /** Governance participation score (0-1) */
  governance_participation: number;
  /** Annual income for max loan calculation */
  annual_income: number;
}

/** Result of comprehensive credit calculation */
export interface CreditScoreResult {
  /** DID */
  did: string;
  /** Final score (0-1000) */
  score: number;
  /** Credit tier */
  tier: CreditTier;
  /** Factor breakdown */
  factors: CreditFactors;
  /** Recommended interest rate */
  recommended_rate: number;
  /** Maximum loan amount based on score and income */
  max_loan_amount: number;
  /** Calculation timestamp */
  calculated_at: number;
}

// ============================================================================
// Loan Types
// ============================================================================

/** Loan status */
export type LoanStatus = 'Pending' | 'Active' | 'Repaid' | 'Defaulted';

/** Collateral type */
export type CollateralType =
  | 'Property'
  | 'Cryptocurrency'
  | 'Equipment'
  | 'Inventory'
  | 'Receivables'
  | 'Other';

/** Loan application status */
export type ApplicationStatus =
  | 'Draft'
  | 'Submitted'
  | 'UnderReview'
  | 'Approved'
  | 'Rejected'
  | 'Withdrawn'
  | 'Funded';

/** A P2P loan */
export interface Loan {
  /** Unique loan identifier */
  id: string;
  /** Borrower's DID */
  borrower_did: string;
  /** Lender's DID */
  lender_did: string;
  /** Principal amount */
  principal: number;
  /** Currency code */
  currency: string;
  /** Annual interest rate (percentage) */
  interest_rate: number;
  /** Term in days */
  term_days: number;
  /** Collateral asset ID (optional) */
  collateral_asset_id?: string;
  /** Loan status */
  status: LoanStatus;
  /** Creation timestamp */
  created_at: number;
  /** Maturity date timestamp */
  maturity_date: number;
}

/** Input for requesting a loan */
export interface RequestLoanInput {
  /** Unique loan identifier */
  id: string;
  /** Borrower's DID */
  borrower_did: string;
  /** Lender's DID */
  lender_did: string;
  /** Principal amount */
  principal: number;
  /** Currency code */
  currency: string;
  /** Annual interest rate */
  interest_rate: number;
  /** Term in days */
  term_days: number;
  /** Collateral asset ID (optional) */
  collateral_asset_id?: string;
}

/** Loan payment record */
export interface LoanPayment {
  /** Unique payment identifier */
  id: string;
  /** Loan ID */
  loan_id: string;
  /** Payment amount */
  amount: number;
  /** Principal portion */
  principal_portion: number;
  /** Interest portion */
  interest_portion: number;
  /** Payment timestamp */
  paid_at: number;
}

/** Input for making a loan payment */
export interface MakePaymentInput {
  /** Unique payment identifier */
  id: string;
  /** Loan ID */
  loan_id: string;
  /** Payment amount */
  amount: number;
  /** Principal portion */
  principal_portion: number;
  /** Interest portion */
  interest_portion: number;
}

/** Loan application */
export interface LoanApplication {
  /** Unique application identifier */
  id: string;
  /** Borrower's DID */
  borrower_did: string;
  /** Requested amount */
  requested_amount: number;
  /** Currency code */
  currency: string;
  /** Purpose description */
  purpose: string;
  /** Requested term in days */
  term_days: number;
  /** Collateral asset ID (optional) */
  collateral_asset_id?: string;
  /** Collateral type (optional) */
  collateral_type?: CollateralType;
  /** Income verification reference (optional) */
  income_verification?: string;
  /** Application status */
  status: ApplicationStatus;
  /** Credit score at application time */
  credit_score_at_application: number;
  /** Submission timestamp */
  created_at: number;
  /** Review timestamp */
  reviewed_at?: number;
  /** Reviewer's DID */
  reviewed_by?: string;
}

/** Input for submitting a loan application */
export interface SubmitApplicationInput {
  /** Unique application identifier */
  id: string;
  /** Borrower's DID */
  borrower_did: string;
  /** Requested amount */
  requested_amount: number;
  /** Currency code */
  currency: string;
  /** Purpose description */
  purpose: string;
  /** Requested term in days */
  term_days: number;
  /** Collateral asset ID (optional) */
  collateral_asset_id?: string;
  /** Collateral type (optional) */
  collateral_type?: CollateralType;
  /** Income verification reference (optional) */
  income_verification?: string;
  /** Current credit score */
  credit_score: number;
}

/** Input for approving a loan application */
export interface ApproveLoanInput {
  /** Application ID being approved */
  application_id: string;
  /** New loan ID */
  loan_id: string;
  /** Borrower's DID */
  borrower_did: string;
  /** Lender's DID */
  lender_did: string;
  /** Approved amount */
  approved_amount: number;
  /** Currency code */
  currency: string;
  /** Interest rate */
  interest_rate: number;
  /** Term in days */
  term_days: number;
  /** Collateral asset ID (optional) */
  collateral_asset_id?: string;
}

/** Input for payment schedule calculation */
export interface PaymentScheduleInput {
  /** Loan ID */
  loan_id: string;
  /** Principal amount */
  principal: number;
  /** Annual interest rate (percentage) */
  annual_interest_rate: number;
  /** Term in months */
  term_months: number;
  /** Start date timestamp */
  start_date: number;
}

/** Scheduled payment in amortization schedule */
export interface ScheduledPayment {
  /** Payment number (1-indexed) */
  payment_number: number;
  /** Due date timestamp */
  due_date: number;
  /** Total payment amount */
  total_amount: number;
  /** Principal portion */
  principal_portion: number;
  /** Interest portion */
  interest_portion: number;
  /** Remaining balance after payment */
  remaining_balance: number;
}

/** Payment schedule result */
export interface PaymentSchedule {
  /** Loan ID */
  loan_id: string;
  /** Principal amount */
  principal: number;
  /** Annual interest rate */
  annual_interest_rate: number;
  /** Term in months */
  term_months: number;
  /** Monthly payment amount */
  monthly_payment: number;
  /** Total interest over loan term */
  total_interest: number;
  /** Total cost (principal + interest) */
  total_cost: number;
  /** Individual payment schedule */
  payments: ScheduledPayment[];
}

/** Input for loan default processing */
export interface LoanDefaultInput {
  /** Loan ID */
  loan_id: string;
  /** Borrower's DID */
  borrower_did: string;
  /** Outstanding balance */
  outstanding_balance: number;
  /** Collateral asset ID (optional) */
  collateral_asset_id?: string;
  /** Days overdue */
  days_overdue: number;
}

/** Result of loan default processing */
export interface DefaultResult {
  /** Loan ID */
  loan_id: string;
  /** Outstanding balance */
  outstanding_balance: number;
  /** Penalty amount */
  penalty_amount: number;
  /** Total owed after penalty */
  total_owed: number;
  /** Collateral action taken */
  collateral_action?: string;
  /** Processing timestamp */
  processed_at: number;
  /** Status message */
  status: string;
}

// ============================================================================
// Treasury Types
// ============================================================================

/** Treasury allocation status */
export type AllocationStatus =
  | 'Proposed'
  | 'Voting'
  | 'Approved'
  | 'Executed'
  | 'Rejected'
  | 'Cancelled';

/** Community treasury pool */
export interface Treasury {
  /** Unique treasury identifier */
  id: string;
  /** Treasury name */
  name: string;
  /** Description */
  description: string;
  /** Associated DAO ID (optional) */
  dao_id?: string;
  /** Governance threshold (0-1) for approvals */
  governance_threshold: number;
  /** Creation timestamp */
  created_at: number;
  /** Creator's DID */
  created_by: string;
}

/** Input for creating a treasury */
export interface CreateTreasuryInput {
  /** Unique treasury identifier */
  id: string;
  /** Treasury name */
  name: string;
  /** Description */
  description: string;
  /** Associated DAO ID (optional) */
  dao_id?: string;
  /** Governance threshold (optional, defaults to 0.66) */
  governance_threshold?: number;
  /** Creator's DID */
  created_by: string;
}

/** Treasury allocation */
export interface TreasuryAllocation {
  /** Unique allocation identifier */
  id: string;
  /** Treasury ID */
  treasury_id: string;
  /** Purpose description */
  purpose: string;
  /** Allocation amount */
  amount: number;
  /** Currency code */
  currency: string;
  /** Recipient DID (optional) */
  recipient_did?: string;
  /** Recipient wallet (optional) */
  recipient_wallet?: string;
  /** Associated proposal ID (optional) */
  proposal_id?: string;
  /** Allocation status */
  status: AllocationStatus;
  /** DIDs that approved */
  approved_by: string[];
  /** Creation timestamp */
  created_at: number;
  /** Execution timestamp */
  executed_at?: number;
}

/** Input for proposing an allocation */
export interface ProposeAllocationInput {
  /** Unique allocation identifier */
  id: string;
  /** Treasury ID */
  treasury_id: string;
  /** Purpose description */
  purpose: string;
  /** Allocation amount */
  amount: number;
  /** Currency code */
  currency: string;
  /** Recipient DID (optional) */
  recipient_did?: string;
  /** Recipient wallet (optional) */
  recipient_wallet?: string;
  /** Associated proposal ID (optional) */
  proposal_id?: string;
}

// ============================================================================
// Escrow Types
// ============================================================================

/** Escrow type */
export type EscrowType = 'Purchase' | 'Service' | 'Loan' | 'Dispute' | 'Milestone';

/** Escrow status */
export type EscrowStatus =
  | 'Funded'
  | 'ConditionsMet'
  | 'Released'
  | 'Disputed'
  | 'Refunded'
  | 'Expired';

/** Escrow account */
export interface Escrow {
  /** Unique escrow identifier */
  id: string;
  /** Escrow type */
  escrow_type: EscrowType;
  /** Depositor's DID */
  depositor_did: string;
  /** Beneficiary's DID */
  beneficiary_did: string;
  /** Arbiter's DID (optional) */
  arbiter_did?: string;
  /** Escrowed amount */
  amount: number;
  /** Currency code */
  currency: string;
  /** Release conditions */
  conditions: string[];
  /** Escrow status */
  status: EscrowStatus;
  /** Creation timestamp */
  created_at: number;
  /** Release timestamp */
  released_at?: number;
  /** Associated dispute case ID */
  dispute_case_id?: string;
}

/** Input for creating an escrow */
export interface CreateEscrowInput {
  /** Unique escrow identifier */
  id: string;
  /** Escrow type */
  escrow_type: EscrowType;
  /** Depositor's DID */
  depositor_did: string;
  /** Beneficiary's DID */
  beneficiary_did: string;
  /** Arbiter's DID (optional) */
  arbiter_did?: string;
  /** Escrowed amount */
  amount: number;
  /** Currency code */
  currency: string;
  /** Release conditions */
  conditions: string[];
}

/** Input for releasing an escrow */
export interface ReleaseEscrowInput {
  /** Escrow ID */
  escrow_id: string;
  /** Beneficiary's wallet ID */
  beneficiary_wallet: string;
  /** Amount to release */
  amount: number;
  /** Currency code */
  currency: string;
  /** Releaser's DID */
  released_by: string;
}

/** Input for disputing an escrow */
export interface DisputeEscrowInput {
  /** Escrow ID */
  escrow_id: string;
  /** Disputant's DID */
  disputant_did: string;
  /** Dispute reason */
  reason: string;
  /** Disputed amount */
  amount: number;
}

/** Dispute result */
export interface DisputeResult {
  /** Status message */
  status: string;
  /** Escrow ID */
  escrow_id: string;
  /** Message */
  message: string;
}

// ============================================================================
// Cross-hApp Integration Types
// ============================================================================

/** Collateral lock request */
export interface CollateralLockRequest {
  /** Asset ID */
  asset_id: string;
  /** Loan ID */
  loan_id: string;
  /** Borrower's DID */
  borrower_did: string;
  /** Lock amount */
  lock_amount: number;
}

/** Collateral lock result */
export interface CollateralLockResult {
  /** Lock ID */
  lock_id: string;
  /** Asset ID */
  asset_id: string;
  /** Loan ID */
  loan_id: string;
  /** Lock timestamp */
  locked_at: number;
  /** Status */
  status: string;
}

/** Justice enforcement request */
export interface JusticeEnforcementRequest {
  /** Case ID */
  case_id: string;
  /** Action: 'freeze', 'transfer', 'unfreeze' */
  action: string;
  /** Target wallet ID */
  target_wallet: string;
  /** Recipient wallet (for transfers) */
  recipient_wallet?: string;
  /** Amount (for transfers) */
  amount?: number;
  /** Currency (for transfers) */
  currency?: string;
  /** Reason for action */
  reason: string;
}

/** Enforcement result */
export interface EnforcementResult {
  /** Success flag */
  success: boolean;
  /** Action taken */
  action: string;
  /** Target */
  target: string;
  /** Message */
  message: string;
}

/** Financial summary for cross-hApp queries */
export interface FinancialSummary {
  /** DID */
  did: string;
  /** Total assets value */
  total_assets: number;
  /** Total liabilities */
  total_liabilities: number;
  /** Credit score */
  credit_score: number;
  /** Number of active loans */
  active_loans: number;
  /** Total loan history count */
  loan_history_count: number;
  /** On-time payment rate (0-1) */
  on_time_payment_rate: number;
  /** Number of escrows */
  escrow_count: number;
  /** Treasury memberships */
  treasury_memberships: number;
  /** Calculation timestamp */
  calculated_at: number;
}

// ============================================================================
// Error Types
// ============================================================================

/** Finance SDK error codes */
export type FinanceSdkErrorCode =
  | 'CONNECTION_ERROR'
  | 'ZOME_ERROR'
  | 'INVALID_INPUT'
  | 'WALLET_NOT_FOUND'
  | 'INSUFFICIENT_FUNDS'
  | 'WALLET_FROZEN'
  | 'LOAN_NOT_FOUND'
  | 'ESCROW_NOT_FOUND'
  | 'TREASURY_NOT_FOUND'
  | 'UNAUTHORIZED'
  | 'SERIALIZATION_ERROR'
  | 'UNKNOWN_ERROR';

/**
 * Finance SDK Error
 */
export class FinanceSdkError extends Error {
  constructor(
    public readonly code: FinanceSdkErrorCode,
    message: string,
    public readonly cause?: unknown
  ) {
    super(message);
    this.name = 'FinanceSdkError';
  }
}
