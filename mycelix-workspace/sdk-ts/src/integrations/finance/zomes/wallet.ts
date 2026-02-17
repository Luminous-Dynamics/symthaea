/**
 * Wallet Zome Client
 *
 * Handles wallet management and fund transfers.
 *
 * @module @mycelix/sdk/integrations/finance/zomes/wallet
 */

import { FinanceSdkError } from '../types';

import type {
  Wallet,
  CreateWalletInput,


  Transaction,
  TransferInput,
  DepositInput,
  WithdrawInput,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the Wallet client
 */
export interface WalletClientConfig {
  /** Role ID for the finance DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: WalletClientConfig = {
  roleId: 'finance',
  zomeName: 'finance',
};

/**
 * Client for wallet and transaction operations
 *
 * @example
 * ```typescript
 * const wallets = new WalletClient(holochainClient);
 *
 * // Create a personal wallet
 * const wallet = await wallets.createWallet({
 *   id: 'my-wallet',
 *   owner_did: 'did:mycelix:alice',
 *   wallet_type: 'Personal',
 * });
 *
 * // Transfer funds
 * const tx = await wallets.transfer({
 *   id: 'tx-001',
 *   from_wallet: 'my-wallet',
 *   to_wallet: 'recipient-wallet',
 *   amount: 100,
 *   currency: 'MYC',
 *   memo: 'Payment for services',
 * });
 * ```
 */
export class WalletClient {
  private readonly config: WalletClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<WalletClientConfig> = {}
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
  // Wallet Operations
  // ============================================================================

  /**
   * Create a new wallet
   *
   * @param input - Wallet creation parameters
   * @returns The created wallet
   */
  async createWallet(input: CreateWalletInput): Promise<Wallet> {
    const record = await this.call<HolochainRecord>('create_wallet', {
      id: input.id,
      owner_did: input.owner_did,
      wallet_type: input.wallet_type,
      frozen: false,
      created_at: Date.now() * 1000, // microseconds
    });
    return this.extractEntry<Wallet>(record);
  }

  /**
   * Get a wallet by ID
   *
   * @param walletId - Wallet identifier
   * @returns The wallet or null if not found
   */
  async getWallet(walletId: string): Promise<Wallet | null> {
    const record = await this.call<HolochainRecord | null>('get_wallet', walletId);
    if (!record) return null;
    return this.extractEntry<Wallet>(record);
  }

  /**
   * Get all wallets owned by a DID
   *
   * @param ownerDid - Owner's DID
   * @returns Array of wallets
   */
  async getWalletsByOwner(ownerDid: string): Promise<Wallet[]> {
    const records = await this.call<HolochainRecord[]>('get_wallets_by_owner', ownerDid);
    return records.map(r => this.extractEntry<Wallet>(r));
  }

  /**
   * Freeze a wallet (prevent transactions)
   *
   * @param walletId - Wallet identifier
   * @returns Updated wallet
   */
  async freezeWallet(walletId: string): Promise<Wallet> {
    const record = await this.call<HolochainRecord>('freeze_wallet', walletId);
    return this.extractEntry<Wallet>(record);
  }

  /**
   * Unfreeze a wallet
   *
   * @param walletId - Wallet identifier
   * @returns Updated wallet
   */
  async unfreezeWallet(walletId: string): Promise<Wallet> {
    const record = await this.call<HolochainRecord>('unfreeze_wallet', walletId);
    return this.extractEntry<Wallet>(record);
  }

  // ============================================================================
  // Balance Operations
  // ============================================================================

  /**
   * Get balance for a specific currency
   *
   * @param walletId - Wallet identifier
   * @param currency - Currency code
   * @returns Balance amount
   */
  async getBalance(walletId: string, currency: string): Promise<number> {
    return this.call<number>('get_balance', { wallet_id: walletId, currency });
  }

  /**
   * Get all balances for a wallet
   *
   * @param walletId - Wallet identifier
   * @returns Map of currency to balance
   */
  async getAllBalances(walletId: string): Promise<Record<string, number>> {
    return this.call<Record<string, number>>('get_all_balances', walletId);
  }

  // ============================================================================
  // Transaction Operations
  // ============================================================================

  /**
   * Transfer funds between wallets
   *
   * @param input - Transfer parameters
   * @returns The transaction record
   */
  async transfer(input: TransferInput): Promise<Transaction> {
    const record = await this.call<HolochainRecord>('transfer', input);
    return this.extractEntry<Transaction>(record);
  }

  /**
   * Deposit funds to a wallet (from external source)
   *
   * @param input - Deposit parameters
   * @returns The transaction record
   */
  async deposit(input: DepositInput): Promise<Transaction> {
    const record = await this.call<HolochainRecord>('deposit', input);
    return this.extractEntry<Transaction>(record);
  }

  /**
   * Withdraw funds from a wallet (to external destination)
   *
   * @param input - Withdrawal parameters
   * @returns The transaction record
   */
  async withdraw(input: WithdrawInput): Promise<Transaction> {
    const record = await this.call<HolochainRecord>('withdraw', input);
    return this.extractEntry<Transaction>(record);
  }

  /**
   * Get transaction history for a wallet
   *
   * @param walletId - Wallet identifier
   * @param limit - Maximum number of transactions (default 50)
   * @returns Array of transactions
   */
  async getTransactionHistory(walletId: string, limit: number = 50): Promise<Transaction[]> {
    const records = await this.call<HolochainRecord[]>('get_transactions', walletId);
    return records.slice(0, limit).map(r => this.extractEntry<Transaction>(r));
  }

  /**
   * Get a specific transaction by ID
   *
   * @param transactionId - Transaction identifier
   * @returns The transaction or null
   */
  async getTransaction(transactionId: string): Promise<Transaction | null> {
    const record = await this.call<HolochainRecord | null>('get_transaction', transactionId);
    if (!record) return null;
    return this.extractEntry<Transaction>(record);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Quick transfer with auto-generated ID
   *
   * @param fromWallet - Source wallet ID
   * @param toWallet - Destination wallet ID
   * @param amount - Transfer amount
   * @param currency - Currency code
   * @param memo - Optional memo
   * @returns The transaction
   */
  async quickTransfer(
    fromWallet: string,
    toWallet: string,
    amount: number,
    currency: string = 'MYC',
    memo?: string
  ): Promise<Transaction> {
    const id = `tx-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    return this.transfer({
      id,
      from_wallet: fromWallet,
      to_wallet: toWallet,
      amount,
      currency,
      memo,
    });
  }

  /**
   * Create a personal wallet with auto-generated ID
   *
   * @param ownerDid - Owner's DID
   * @param name - Optional wallet name suffix
   * @returns The created wallet
   */
  async createPersonalWallet(ownerDid: string, name?: string): Promise<Wallet> {
    const suffix = name || Math.random().toString(36).slice(2, 8);
    const id = `wallet-${suffix}-${Date.now()}`;
    return this.createWallet({
      id,
      owner_did: ownerDid,
      wallet_type: 'Personal',
    });
  }

  /**
   * Check if wallet has sufficient funds
   *
   * @param walletId - Wallet identifier
   * @param amount - Required amount
   * @param currency - Currency code
   * @returns True if sufficient funds
   */
  async hasSufficientFunds(
    walletId: string,
    amount: number,
    currency: string
  ): Promise<boolean> {
    const balance = await this.getBalance(walletId, currency);
    return balance >= amount;
  }
}
