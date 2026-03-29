// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Finance hApp Client Module
 *
 * Provides TypeScript clients for all Finance hApp zomes in the Mycelix ecosystem.
 *
 * The Finance hApp includes 9 zomes organized into three categories:
 *
 * **Legacy (Deprecated)**
 * - `lending` - P2P lending (use TEND/HEARTH instead)
 * - `credit_scoring` - Credit profiles (use CGC instead)
 *
 * **Core Payment Infrastructure**
 * - `payments` - Direct payments and channels
 * - `treasury` - DAO treasury management
 * - `staking` - Tri-asset staking and escrow
 * - `bridge` - Cross-hApp finance operations
 *
 * **Commons-Aligned (Recommended)**
 * - `cgc` - Civic Gifting Credits (recognition system)
 * - `tend` - Time Exchange (mutual credit)
 * - `hearth` - Commons Pool (community resources)
 *
 * @module @mycelix/sdk/clients/finance
 *
 * @example
 * ```typescript
 * import { FinanceClient } from '@mycelix/sdk/clients/finance';
 *
 * const finance = new FinanceClient(appClient);
 *
 * // Use commons-aligned modules (recommended)
 * const allocation = await finance.cgc.getOrCreateAllocation('did:mycelix:alice');
 * const exchange = await finance.tend.recordExchange({ ... });
 * const pool = await finance.hearth.createPool({ ... });
 *
 * // Use core payment infrastructure
 * const payment = await finance.payments.sendPayment({ ... });
 * const treasury = await finance.treasury.createTreasury({ ... });
 * const stake = await finance.staking.createStake({ ... });
 *
 * // Cross-hApp operations
 * const credit = await finance.bridge.queryCredit({ ... });
 * ```
 */

import { BridgeClient } from './bridge.js';
import { CgcClient } from './cgc.js';
import { CreditScoringClient } from './credit-scoring.js';
import { HearthClient } from './hearth.js';
import { LendingClient } from './lending.js';
import { PaymentsClient } from './payments.js';
import { TendClient } from './tend.js';
import { TreasuryClient } from './treasury.js';

import type { ZomeClientConfig } from '../../core/zome-client.js';
import type { AppClient } from '@holochain/client';

// Zome Clients
import { StakingClient } from './staking.js';

// =============================================================================
// Barrel Exports - Zome Clients
// =============================================================================

export { LendingClient } from './lending.js';
export { CreditScoringClient } from './credit-scoring.js';
export { PaymentsClient } from './payments.js';
export { TreasuryClient } from './treasury.js';
export { CgcClient } from './cgc.js';
export { TendClient } from './tend.js';
export { HearthClient } from './hearth.js';
export { StakingClient } from './staking.js';
export { BridgeClient } from './bridge.js';

// =============================================================================
// Barrel Exports - Types
// =============================================================================

export type {
  // Common
  Timestamp,
  Did,
  Currency,

  // Lending (deprecated)
  Loan,
  LoanOffer,
  LoanStatus,
  RequestLoanInput,
  CreateOfferInput,
  FundLoanInput,
  PaymentSchedule,
  ScheduledPayment,
  CreateScheduleInput,
  MatchOffersInput,
  CreditTier,
  CreditAssessmentResult,
  CreditAssessmentComponents,
  ComponentDetail,
  TransparencyReport,
  ProofType,
  PrivacyCreditProof,

  // Credit Scoring (deprecated)
  CreditProfile,
  CreateProfileInput,
  PaymentStatus,
  PaymentRecord,
  RecordPaymentInput,
  CollateralType,
  CollateralRecord,
  RegisterCollateralInput,
  LockCollateralInput,
  UpdateMatlInput,
  ScoreRangeInput,

  // Payments
  PaymentType,
  TransferStatus,
  Payment,
  SendPaymentInput,
  PaymentChannel,
  OpenChannelInput,
  ChannelTransferInput,
  Receipt,
  CreateEscrowPaymentInput,

  // Treasury
  Treasury,
  CreateTreasuryInput,
  ContributionType,
  Contribution,
  ContributeInput,
  AllocationStatus,
  Allocation,
  ProposeAllocationInput,
  ApproveAllocationInput,
  RejectAllocationInput,
  CancelAllocationInput,
  AddManagerInput,
  RemoveManagerInput,
  AllocationStatusQuery,
  SavingsPool,
  CreatePoolInput,
  JoinPoolInput,
  PoolContributionInput,

  // CGC
  CgcContributionType,
  CgcAllocation,
  CgcAllocationStatus,
  CgcGift,
  GiftCgcInput,
  GiftRecord,
  RecognitionSummary,
  CulturalAlias,
  RegisterAliasInput,

  // TEND
  ServiceCategory,
  ExchangeStatus,
  TendExchange,
  RecordExchangeInput,
  ExchangeRecord,
  TendBalance,
  BalanceInfo,
  GetBalanceInput,
  ServiceListing,
  CreateListingInput,
  Urgency,
  ServiceRequest,
  CreateRequestInput,

  // HEARTH
  ResourceType,
  PoolStatus,
  CommonsPool,
  PoolGovernance,
  PoolGovernanceInput,
  CreateCommonsPoolInput,
  PoolSummary,
  PoolContribution,
  ContributeToPoolInput,
  NeedCategory,
  RequestStatus,
  AllocationRequest,
  RequestAllocationInput,
  RequestSummary,
  AllocationVote,
  VoteInput,

  // Staking
  StakeStatus,
  TriAssetStake,
  CreateStakeInput,
  UpdateTrustInput,
  SlashingReason,
  SlashingEvidence,
  SlashStakeInput,
  SlashingEvent,
  ChainType,
  StakingAssetType,
  CrossChainLockProof,
  SubmitProofInput,
  EscrowStatus,
  EscrowHashType,
  ReleaseCondition,
  EscrowSignature,
  CryptoEscrow,
  CreateCryptoEscrowInput,
  RevealPreimageInput,
  AddSignatureInput,

  // Bridge
  CreditPurpose,
  CreditQuery,
  CreditResult,
  QueryCreditInput,
  CrossHappPayment,
  ProcessPaymentInput,
  BridgeAssetType,
  CollateralStatus,
  CollateralRegistration,
  RegisterBridgeCollateralInput,
  FinanceEventType,
  FinanceBridgeEvent,
  BroadcastFinanceEventInput,
} from './types.js';

// =============================================================================
// Aggregate Finance Client
// =============================================================================

/**
 * Configuration for the Finance client
 */
export interface FinanceClientConfig {
  /** Timeout for zome calls in milliseconds */
  timeout?: number;
  /** Retry configuration */
  retry?: {
    maxRetries?: number;
    baseDelay?: number;
    maxDelay?: number;
  };
}

/**
 * Unified Finance hApp Client
 *
 * Provides access to all Finance zome clients through a single interface.
 * Organizes clients into logical groupings for discoverability.
 *
 * @example
 * ```typescript
 * const finance = new FinanceClient(appClient);
 *
 * // Commons-aligned modules (recommended for new development)
 * finance.cgc      // Civic Gifting Credits
 * finance.tend     // Time Exchange
 * finance.hearth   // Commons Pools
 *
 * // Core infrastructure
 * finance.payments // Direct payments and channels
 * finance.treasury // DAO treasury management
 * finance.staking  // Tri-asset staking and escrow
 * finance.bridge   // Cross-hApp operations
 *
 * // Legacy (deprecated)
 * finance.lending        // P2P lending
 * finance.creditScoring  // Credit profiles
 * ```
 */
export class FinanceClient {
  // =========================================================================
  // Commons-Aligned Modules (Recommended)
  // =========================================================================

  /**
   * CGC (Civic Gifting Credits) - Recognition system
   *
   * Monthly allocation of 10 CGC per member for recognizing contributions.
   * CGCs represent recognition, not transferable value.
   */
  public readonly cgc: CgcClient;

  /**
   * TEND (Time Exchange) - Mutual credit time banking
   *
   * Interest-free time exchange with +/- 40 balance limits.
   * All hours are equal - fundamental time banking principle.
   */
  public readonly tend: TendClient;

  /**
   * HEARTH (Commons Pool) - Community resource governance
   *
   * Democratic allocation of pooled community resources.
   * "From each according to ability, to each according to need."
   */
  public readonly hearth: HearthClient;

  // =========================================================================
  // Core Payment Infrastructure
  // =========================================================================

  /**
   * Payments - Direct payments and channels
   *
   * Send payments, open channels, manage escrow.
   */
  public readonly payments: PaymentsClient;

  /**
   * Treasury - DAO treasury management
   *
   * Multi-sig treasury with contribution tracking and allocation voting.
   */
  public readonly treasury: TreasuryClient;

  /**
   * Staking - Tri-asset staking and escrow
   *
   * MYC/ETH/USDC staking with K-Vector trust weighting and crypto escrow.
   */
  public readonly staking: StakingClient;

  /**
   * Bridge - Cross-hApp finance operations
   *
   * Query credit, process payments, and register collateral from other hApps.
   */
  public readonly bridge: BridgeClient;

  // =========================================================================
  // Legacy Modules (Deprecated)
  // =========================================================================

  /**
   * Lending - P2P lending
   *
   * @deprecated Use TEND for interest-free exchange or HEARTH for community pooling.
   * Traditional lending conflicts with the Commons Charter.
   */
  public readonly lending: LendingClient;

  /**
   * Credit Scoring - Credit profiles and collateral
   *
   * @deprecated Use CGC for recognition-based systems.
   * Credit scoring conflicts with the Commons Charter.
   */
  public readonly creditScoring: CreditScoringClient;

  /**
   * Create a new Finance client
   *
   * @param client - Holochain AppClient
   * @param config - Optional configuration
   */
  constructor(client: AppClient, config: FinanceClientConfig = {}) {
    const clientConfig: Partial<ZomeClientConfig> = {
      timeout: config.timeout,
      retry: config.retry,
    };

    // Commons-aligned modules
    this.cgc = new CgcClient(client, clientConfig);
    this.tend = new TendClient(client, clientConfig);
    this.hearth = new HearthClient(client, clientConfig);

    // Core infrastructure
    this.payments = new PaymentsClient(client, clientConfig);
    this.treasury = new TreasuryClient(client, clientConfig);
    this.staking = new StakingClient(client, clientConfig);
    this.bridge = new BridgeClient(client, clientConfig);

    // Legacy (deprecated)
    this.lending = new LendingClient(client, clientConfig);
    this.creditScoring = new CreditScoringClient(client, clientConfig);
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a FinanceClient instance
 *
 * @param client - Holochain AppClient
 * @param config - Optional configuration
 * @returns FinanceClient instance
 *
 * @example
 * ```typescript
 * import { createFinanceClient } from '@mycelix/sdk/clients/finance';
 *
 * const finance = createFinanceClient(appClient, {
 *   timeout: 60000,
 *   retry: { maxRetries: 3 },
 * });
 * ```
 */
export function createFinanceClient(
  client: AppClient,
  config: FinanceClientConfig = {}
): FinanceClient {
  return new FinanceClient(client, config);
}

/**
 * Create individual Finance zome clients
 *
 * @param client - Holochain AppClient
 * @param config - Optional configuration
 * @returns Object with all individual clients
 *
 * @example
 * ```typescript
 * const clients = createFinanceClients(appClient);
 * const { cgc, tend, hearth, payments } = clients;
 * ```
 */
export function createFinanceClients(client: AppClient, config: FinanceClientConfig = {}) {
  const clientConfig: Partial<ZomeClientConfig> = {
    timeout: config.timeout,
    retry: config.retry,
  };

  return {
    // Commons-aligned
    cgc: new CgcClient(client, clientConfig),
    tend: new TendClient(client, clientConfig),
    hearth: new HearthClient(client, clientConfig),

    // Core infrastructure
    payments: new PaymentsClient(client, clientConfig),
    treasury: new TreasuryClient(client, clientConfig),
    staking: new StakingClient(client, clientConfig),
    bridge: new BridgeClient(client, clientConfig),

    // Legacy (deprecated)
    lending: new LendingClient(client, clientConfig),
    creditScoring: new CreditScoringClient(client, clientConfig),
  };
}

// Default export
export default FinanceClient;
