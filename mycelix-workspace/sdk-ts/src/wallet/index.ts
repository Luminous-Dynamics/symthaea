// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Unified Mycelix Wallet - The Glass-Top Architecture
 *
 * This is the single interface users interact with. It presents a beautiful,
 * simple surface while the complexity lives invisibly below.
 *
 * ABOVE THE GLASS:
 * - Balance: 500 MYC
 * - Send to @alice
 * - Transaction history with human names
 *
 * BELOW THE GLASS:
 * - MycelixVault: Secure key management (the Pen)
 * - Finance hApp: On-chain ledger (the Checkbook)
 * - Identity hApp: Human-readable names (the Address Book)
 *
 * @example
 * ```typescript
 * // Create/load wallet (handles all complexity internally)
 * const wallet = await Wallet.create({ name: 'My Wallet' });
 *
 * // Simple operations - no need to think about keys, chains, or hashes
 * await wallet.send('@alice', 50, 'MYC');
 * const balance = wallet.balance('MYC');
 *
 * // Reactive state - UI updates automatically
 * wallet.balance$.subscribe(bal => updateUI(bal));
 * wallet.transactions$.subscribe(txs => renderHistory(txs));
 * ```
 */

// =============================================================================
// Wallet Core (The Glass-Top Interface)
// =============================================================================

export {
  // Wallet class and factory functions
  Wallet,
  createWallet,
  loadWallet,

  // Types
  type Currency,
  type TransactionStatus,
  type TransactionDirection,
  type Identity,
  type Transaction,
  type Balance,
  type WalletState,
  type SendOptions,
  type WalletConfig,
  type IdentityResolver,
  type FinanceProvider,
  type WalletEvent,
} from './wallet.js';

// =============================================================================
// Holochain Integration (The Nervous System)
// =============================================================================

export {
  // Conductor Connection Manager (The Spine)
  ConductorManager,
  createConductorManager,
  createDevConductorManager,
  createProdConductorManager,
  type ConductorState,
  type ConductorConfig,
  type ConductorSignal,
  type SignalListener,
} from './conductor.js';

export {
  // Holochain Finance Provider (The Muscle)
  HolochainFinanceProvider,
  createHolochainFinanceProvider,
  createMockFinanceProvider,
  type FinanceState,
  type HolochainFinanceConfig,
} from './finance-provider.js';

export {
  // Holochain Wallet Factory (Wiring the Nervous System)
  createHolochainWallet,
  createDevWallet,
  isConductorAvailable,
  type HolochainWalletConfig,
  type HolochainWalletResult,
} from './holochain-wallet.js';

export {
  // Bridge Integration (Cross-hApp Reputation)
  WalletBridge,
  createWalletBridge,
  type WalletBridgeConfig,
  type TransactionRecommendation,
  type VerificationStatus,
  type ReputationChangeHandler,
} from './bridge-integration.js';

// =============================================================================
// Cross-Domain Resource Credits (The Exchange)
// =============================================================================

export {
  // Service and factory
  ResourceCreditsService,
  getCreditsService,
  resetCreditsService,
  DEFAULT_CREDITS_CONFIG,

  // Types
  type CreditType,
  type CreditStatus,
  type CreditEntry,
  type CreditSource,
  type DomainBalance,
  type Wallet as CreditWallet,
  type WalletTier,
  type WalletSummary as CreditWalletSummary,
  type CreditTransaction,
  type TransactionType as CreditTransactionType,
  type TransactionStatus as CreditTransactionStatus,
  type ExchangeRate,
  type ExchangeQuote,
  type CommunityPool,
  type DistributionRule,
  type CreditsServiceConfig,
} from './cross-domain-credits.js';
