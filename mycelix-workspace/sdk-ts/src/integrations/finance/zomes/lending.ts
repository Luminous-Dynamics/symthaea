/**
 * Lending Zome Client
 *
 * Handles P2P lending operations including loans, applications, and payments.
 *
 * @module @mycelix/sdk/integrations/finance/zomes/lending
 */

import { FinanceSdkError } from '../types';

import type {
  Loan,
  LoanStatus,
  RequestLoanInput,
  LoanPayment,
  MakePaymentInput,
  LoanApplication,
  SubmitApplicationInput,
  ApproveLoanInput,
  PaymentSchedule,
  PaymentScheduleInput,
  LoanDefaultInput,
  DefaultResult,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the Lending client
 */
export interface LendingClientConfig {
  /** Role ID for the finance DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: LendingClientConfig = {
  roleId: 'finance',
  zomeName: 'finance',
};

/**
 * Client for P2P lending operations
 *
 * @example
 * ```typescript
 * const lending = new LendingClient(holochainClient);
 *
 * // Submit a loan application
 * const application = await lending.submitApplication({
 *   id: 'app-001',
 *   borrower_did: 'did:mycelix:bob',
 *   requested_amount: 10000,
 *   currency: 'MYC',
 *   purpose: 'Equipment purchase for cooperative',
 *   term_days: 365,
 *   credit_score: 750,
 * });
 *
 * // Approve and create the loan
 * const loan = await lending.approveLoan({
 *   application_id: 'app-001',
 *   loan_id: 'loan-001',
 *   borrower_did: 'did:mycelix:bob',
 *   lender_did: 'did:mycelix:alice',
 *   approved_amount: 10000,
 *   currency: 'MYC',
 *   interest_rate: 5.0,
 *   term_days: 365,
 * });
 *
 * // Calculate payment schedule
 * const schedule = await lending.calculatePaymentSchedule({
 *   loan_id: 'loan-001',
 *   principal: 10000,
 *   annual_interest_rate: 5.0,
 *   term_months: 12,
 *   start_date: Date.now() * 1000,
 * });
 * ```
 */
export class LendingClient {
  private readonly config: LendingClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<LendingClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Call a zome function with error handling
   */
  private async call<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      const result = await this.client.callZome({
        role_name: this.config.roleId,
        zome_name: this.config.zomeName,
        fn_name: fnName,
        payload,
      });
      return result as T;
    } catch (error) {
      throw new FinanceSdkError(
        'ZOME_ERROR',
        `Failed to call ${fnName}: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Extract entry from a Holochain record
   */
  private extractEntry<T>(record: HolochainRecord): T {
    if (!record.entry || !('Present' in record.entry)) {
      throw new FinanceSdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // Loan Application Operations
  // ============================================================================

  /**
   * Submit a loan application
   *
   * @param input - Application parameters
   * @returns The submitted application
   */
  async submitApplication(input: SubmitApplicationInput): Promise<LoanApplication> {
    const record = await this.call<HolochainRecord>('submit_loan_application', input);
    return this.extractEntry<LoanApplication>(record);
  }

  /**
   * Get a loan application by ID
   *
   * @param applicationId - Application identifier
   * @returns The application or null
   */
  async getApplication(applicationId: string): Promise<LoanApplication | null> {
    const record = await this.call<HolochainRecord | null>('get_loan_application', applicationId);
    if (!record) return null;
    return this.extractEntry<LoanApplication>(record);
  }

  /**
   * Get applications by borrower
   *
   * @param borrowerDid - Borrower's DID
   * @returns Array of applications
   */
  async getApplicationsByBorrower(borrowerDid: string): Promise<LoanApplication[]> {
    const records = await this.call<HolochainRecord[]>(
      'get_applications_by_borrower',
      borrowerDid
    );
    return records.map(r => this.extractEntry<LoanApplication>(r));
  }

  /**
   * Approve a loan application and create the loan
   *
   * @param input - Approval parameters
   * @returns The created loan
   */
  async approveLoan(input: ApproveLoanInput): Promise<Loan> {
    const record = await this.call<HolochainRecord>('approve_loan_application', input);
    return this.extractEntry<Loan>(record);
  }

  /**
   * Reject a loan application
   *
   * @param applicationId - Application identifier
   * @param reason - Rejection reason
   * @returns The updated application
   */
  async rejectApplication(applicationId: string, reason: string): Promise<LoanApplication> {
    const record = await this.call<HolochainRecord>('reject_loan_application', {
      application_id: applicationId,
      reason,
    });
    return this.extractEntry<LoanApplication>(record);
  }

  // ============================================================================
  // Loan Operations
  // ============================================================================

  /**
   * Request a direct loan (without application process)
   *
   * @param input - Loan request parameters
   * @returns The created loan
   */
  async requestLoan(input: RequestLoanInput): Promise<Loan> {
    const record = await this.call<HolochainRecord>('request_loan', input);
    return this.extractEntry<Loan>(record);
  }

  /**
   * Get a loan by ID
   *
   * @param loanId - Loan identifier
   * @returns The loan or null
   */
  async getLoan(loanId: string): Promise<Loan | null> {
    const record = await this.call<HolochainRecord | null>('get_loan', loanId);
    if (!record) return null;
    return this.extractEntry<Loan>(record);
  }

  /**
   * Get loans by borrower
   *
   * @param borrowerDid - Borrower's DID
   * @returns Array of loans
   */
  async getLoansByBorrower(borrowerDid: string): Promise<Loan[]> {
    const records = await this.call<HolochainRecord[]>('get_loans_by_borrower', borrowerDid);
    return records.map(r => this.extractEntry<Loan>(r));
  }

  /**
   * Get loans by lender
   *
   * @param lenderDid - Lender's DID
   * @returns Array of loans
   */
  async getLoansByLender(lenderDid: string): Promise<Loan[]> {
    const records = await this.call<HolochainRecord[]>('get_loans_by_lender', lenderDid);
    return records.map(r => this.extractEntry<Loan>(r));
  }

  /**
   * Get active loans for a DID (as borrower or lender)
   *
   * @param did - DID to query
   * @returns Array of active loans
   */
  async getActiveLoans(did: string): Promise<Loan[]> {
    const [borrowerLoans, lenderLoans] = await Promise.all([
      this.getLoansByBorrower(did),
      this.getLoansByLender(did),
    ]);

    const allLoans = [...borrowerLoans, ...lenderLoans];
    return allLoans.filter(loan => loan.status === 'Active');
  }

  // ============================================================================
  // Payment Operations
  // ============================================================================

  /**
   * Make a loan payment
   *
   * @param input - Payment parameters
   * @returns The payment record
   */
  async makePayment(input: MakePaymentInput): Promise<LoanPayment> {
    const record = await this.call<HolochainRecord>('make_payment', input);
    return this.extractEntry<LoanPayment>(record);
  }

  /**
   * Get payments for a loan
   *
   * @param loanId - Loan identifier
   * @returns Array of payments
   */
  async getPaymentsForLoan(loanId: string): Promise<LoanPayment[]> {
    const records = await this.call<HolochainRecord[]>('get_payments_for_loan', loanId);
    return records.map(r => this.extractEntry<LoanPayment>(r));
  }

  /**
   * Calculate payment schedule (amortization)
   *
   * @param input - Schedule calculation parameters
   * @returns Payment schedule with all payments
   */
  async calculatePaymentSchedule(input: PaymentScheduleInput): Promise<PaymentSchedule> {
    return this.call<PaymentSchedule>('calculate_payment_schedule', input);
  }

  // ============================================================================
  // Default Processing
  // ============================================================================

  /**
   * Process a loan default
   *
   * @param input - Default parameters
   * @returns Default result with penalties and actions
   */
  async processDefault(input: LoanDefaultInput): Promise<DefaultResult> {
    return this.call<DefaultResult>('process_loan_default', input);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Quick loan request with auto-generated IDs
   *
   * @param borrowerDid - Borrower's DID
   * @param lenderDid - Lender's DID
   * @param principal - Loan amount
   * @param interestRate - Annual interest rate
   * @param termDays - Term in days
   * @param currency - Currency code
   * @returns The created loan
   */
  async quickLoan(
    borrowerDid: string,
    lenderDid: string,
    principal: number,
    interestRate: number,
    termDays: number,
    currency: string = 'MYC'
  ): Promise<Loan> {
    const id = `loan-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    return this.requestLoan({
      id,
      borrower_did: borrowerDid,
      lender_did: lenderDid,
      principal,
      currency,
      interest_rate: interestRate,
      term_days: termDays,
    });
  }

  /**
   * Calculate total amount owed on a loan
   *
   * @param loan - The loan
   * @returns Total owed (principal + interest)
   */
  calculateTotalOwed(loan: Loan): number {
    const dailyRate = loan.interest_rate / 365 / 100;
    const interest = loan.principal * dailyRate * loan.term_days;
    return Math.round(loan.principal + interest);
  }

  /**
   * Calculate remaining balance on a loan
   *
   * @param loan - The loan
   * @param payments - Payments made
   * @returns Remaining balance
   */
  calculateRemainingBalance(loan: Loan, payments: LoanPayment[]): number {
    const totalPaid = payments.reduce((sum, p) => sum + p.principal_portion, 0);
    return Math.max(0, loan.principal - totalPaid);
  }

  /**
   * Check if loan is overdue
   *
   * @param loan - The loan
   * @returns True if past maturity date and not repaid
   */
  isOverdue(loan: Loan): boolean {
    if (loan.status !== 'Active') return false;
    return Date.now() * 1000 > loan.maturity_date;
  }

  /**
   * Calculate days until maturity
   *
   * @param loan - The loan
   * @returns Days until maturity (negative if overdue)
   */
  daysUntilMaturity(loan: Loan): number {
    const now = Date.now() * 1000; // microseconds
    const diff = loan.maturity_date - now;
    return Math.floor(diff / (86_400_000_000)); // days
  }

  /**
   * Get loan status description
   *
   * @param status - Loan status
   * @returns Human-readable description
   */
  getStatusDescription(status: LoanStatus): string {
    const descriptions: Record<LoanStatus, string> = {
      Pending: 'Loan is awaiting approval or funding',
      Active: 'Loan is active and payments are expected',
      Repaid: 'Loan has been fully repaid',
      Defaulted: 'Loan is in default status',
    };
    return descriptions[status];
  }
}
