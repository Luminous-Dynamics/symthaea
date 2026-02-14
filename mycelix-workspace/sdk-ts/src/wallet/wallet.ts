/**
 * Unified Mycelix Wallet - The Glass-Top Architecture
 *
 * This is the single interface users interact with. It presents a beautiful,
 * simple surface while the complexity lives invisibly below.
 *
 * ABOVE THE GLASS (This file):
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

import {
  BehaviorSubject,
  Subject,
  type Observable,
  type Subscription,
  map,
  distinctUntilChanged,
} from '../reactive/index.js';
import {
  type MycelixVault,
  loadVault,
  createVault,
  type VaultState,
  type VaultStorageProvider,
  type AccountId,
} from '../vault/index.js';

// =============================================================================
// Types
// =============================================================================

/** Supported currencies (extensible - MYC, HOT, USD, EUR are common) */
export type Currency = string;

/** Transaction status */
export type TransactionStatus =
  | 'pending' // Submitted, not yet confirmed
  | 'confirming' // In DHT propagation
  | 'confirmed' // Fully confirmed
  | 'failed'; // Failed permanently

/** Transaction direction */
export type TransactionDirection = 'incoming' | 'outgoing';

/** Human-readable identity */
export interface Identity {
  /** Raw agent ID (public key hash) */
  agentId: string;
  /** Human-readable nickname (@alice) */
  nickname?: string;
  /** Display name (Alice Smith) */
  displayName?: string;
  /** Avatar URL or data URI */
  avatar?: string;
  /** DID (did:mycelix:...) */
  did?: string;
  /** Whether this identity is verified */
  verified?: boolean;
}

/** Transaction with hydrated identities */
export interface Transaction {
  id: string;
  /** Who sent this */
  from: Identity;
  /** Who received this */
  to: Identity;
  /** Amount transferred */
  amount: number;
  /** Currency */
  currency: Currency;
  /** Direction relative to current user */
  direction: TransactionDirection;
  /** Current status */
  status: TransactionStatus;
  /** Human-readable memo */
  memo?: string;
  /** When initiated */
  createdAt: number;
  /** When confirmed (if confirmed) */
  confirmedAt?: number;
  /** Error message (if failed) */
  error?: string;
}

/** Balance for a single currency */
export interface Balance {
  currency: Currency;
  /** Available balance (can be spent) */
  available: number;
  /** Pending balance (in transit) */
  pending: number;
  /** Total = available + pending */
  total: number;
}

/** Wallet state (reactive) */
export interface WalletState {
  /** Whether wallet is ready to use */
  ready: boolean;
  /** Whether currently loading */
  loading: boolean;
  /** Current user identity */
  identity: Identity | null;
  /** All balances by currency */
  balances: Map<Currency, Balance>;
  /** Recent transactions */
  transactions: Transaction[];
  /** Pending (optimistic) transactions */
  pendingTransactions: Transaction[];
  /** Error if any */
  error: string | null;
}

/** Options for sending funds */
export interface SendOptions {
  /** Optional memo */
  memo?: string;
  /** Callback when optimistically applied */
  onOptimistic?: () => void;
  /** Callback when confirmed */
  onConfirmed?: () => void;
  /** Callback on error */
  onError?: (error: Error) => void;
}

/** Configuration for creating a wallet */
export interface WalletConfig {
  /** Wallet name */
  name: string;
  /** PIN for encryption (4-8 digits) */
  pin?: string;
  /** Enable biometric unlock */
  biometric?: boolean;
  /** Custom storage provider */
  storage?: VaultStorageProvider;
  /** Holochain connection URL */
  holochainUrl?: string;
}

/** Identity resolver interface (pluggable) */
export interface IdentityResolver {
  /** Resolve agent ID to identity */
  resolve(agentId: string): Promise<Identity>;
  /** Resolve nickname (@alice) to identity */
  resolveNickname(nickname: string): Promise<Identity | null>;
  /** Search identities */
  search(query: string): Promise<Identity[]>;
}

/** Finance provider interface (pluggable) */
export interface FinanceProvider {
  /** Get balance for a currency */
  getBalance(agentId: string, currency: Currency): Promise<Balance>;
  /** Get all balances */
  getAllBalances(agentId: string): Promise<Map<Currency, Balance>>;
  /** Get transaction history */
  getTransactions(agentId: string, limit?: number): Promise<Transaction[]>;
  /** Send funds */
  send(
    from: string,
    to: string,
    amount: number,
    currency: Currency,
    memo?: string,
    signature?: Uint8Array
  ): Promise<string>; // Returns transaction ID
  /** Get transaction by ID */
  getTransaction(id: string): Promise<Transaction | null>;
}

// =============================================================================
// Default Implementations (Mock for development)
// =============================================================================

/**
 * Mock identity resolver for development
 */
class MockIdentityResolver implements IdentityResolver {
  private identities: Map<string, Identity> = new Map();

  async resolve(agentId: string): Promise<Identity> {
    const existing = this.identities.get(agentId);
    if (existing) return existing;

    // Generate a mock identity
    const identity: Identity = {
      agentId,
      nickname: `@user_${agentId.slice(0, 6)}`,
      displayName: `User ${agentId.slice(0, 6)}`,
      verified: false,
    };
    this.identities.set(agentId, identity);
    return identity;
  }

  async resolveNickname(nickname: string): Promise<Identity | null> {
    for (const identity of this.identities.values()) {
      if (identity.nickname === nickname) return identity;
    }
    return null;
  }

  async search(query: string): Promise<Identity[]> {
    const results: Identity[] = [];
    const lowerQuery = query.toLowerCase();
    for (const identity of this.identities.values()) {
      if (
        identity.nickname?.toLowerCase().includes(lowerQuery) ||
        identity.displayName?.toLowerCase().includes(lowerQuery)
      ) {
        results.push(identity);
      }
    }
    return results;
  }
}

/**
 * Mock finance provider for development
 */
class MockFinanceProvider implements FinanceProvider {
  private balances: Map<string, Map<Currency, Balance>> = new Map();
  private transactions: Map<string, Transaction> = new Map();
  private userTransactions: Map<string, string[]> = new Map();

  async getBalance(agentId: string, currency: Currency): Promise<Balance> {
    const userBalances = this.balances.get(agentId);
    const balance = userBalances?.get(currency);
    return (
      balance ?? {
        currency,
        available: currency === 'MYC' ? 1000 : 0, // Start with 1000 MYC for demo
        pending: 0,
        total: currency === 'MYC' ? 1000 : 0,
      }
    );
  }

  async getAllBalances(agentId: string): Promise<Map<Currency, Balance>> {
    const result = new Map<Currency, Balance>();
    result.set('MYC', await this.getBalance(agentId, 'MYC'));
    return result;
  }

  async getTransactions(agentId: string, _limit?: number): Promise<Transaction[]> {
    const txIds = this.userTransactions.get(agentId) ?? [];
    return txIds
      .map((id) => this.transactions.get(id))
      .filter((tx): tx is Transaction => tx !== undefined);
  }

  async send(
    from: string,
    to: string,
    amount: number,
    currency: Currency,
    memo?: string,
    _signature?: Uint8Array
  ): Promise<string> {
    const id = `tx-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    // Update balances
    const fromBalances = this.balances.get(from) ?? new Map();
    const toBalances = this.balances.get(to) ?? new Map();

    const fromBalance = fromBalances.get(currency) ?? {
      currency,
      available: 1000,
      pending: 0,
      total: 1000,
    };
    const toBalance = toBalances.get(currency) ?? { currency, available: 0, pending: 0, total: 0 };

    if (fromBalance.available < amount) {
      throw new Error('Insufficient funds');
    }

    fromBalance.available -= amount;
    fromBalance.total -= amount;
    toBalance.available += amount;
    toBalance.total += amount;

    fromBalances.set(currency, fromBalance);
    toBalances.set(currency, toBalance);
    this.balances.set(from, fromBalances);
    this.balances.set(to, toBalances);

    // Create transaction record
    const tx: Transaction = {
      id,
      from: { agentId: from },
      to: { agentId: to },
      amount,
      currency,
      direction: 'outgoing',
      status: 'confirmed',
      memo,
      createdAt: Date.now(),
      confirmedAt: Date.now(),
    };

    this.transactions.set(id, tx);

    // Add to user's transaction list
    const fromTxs = this.userTransactions.get(from) ?? [];
    const toTxs = this.userTransactions.get(to) ?? [];
    fromTxs.push(id);
    toTxs.push(id);
    this.userTransactions.set(from, fromTxs);
    this.userTransactions.set(to, toTxs);

    return id;
  }

  async getTransaction(id: string): Promise<Transaction | null> {
    return this.transactions.get(id) ?? null;
  }
}

// =============================================================================
// Event Types
// =============================================================================

export type WalletEvent =
  | { type: 'transaction_pending'; transaction: Transaction }
  | { type: 'transaction_confirmed'; transaction: Transaction }
  | { type: 'transaction_failed'; transaction: Transaction; error: Error }
  | { type: 'refresh_complete' }
  | { type: 'refresh_failed'; error: Error }
  | { type: 'balance_updated'; currency: Currency; balance: Balance }
  | { type: 'identity_updated'; identity: Identity };

// =============================================================================
// Wallet Class
// =============================================================================

/**
 * Unified Mycelix Wallet
 *
 * The single point of interaction for all wallet operations.
 * Handles key management, balances, transfers, and identity resolution
 * behind a simple, reactive interface.
 */
export class Wallet {
  private vault: MycelixVault;
  private identityResolver: IdentityResolver;
  private financeProvider: FinanceProvider;

  // Reactive state
  private _state$: BehaviorSubject<WalletState>;
  private _events$: Subject<WalletEvent> = new Subject();

  // Internal state
  private refreshInterval: ReturnType<typeof setInterval> | null = null;

  private constructor(
    vault: MycelixVault,
    identityResolver: IdentityResolver,
    financeProvider: FinanceProvider
  ) {
    this.vault = vault;
    this.identityResolver = identityResolver;
    this.financeProvider = financeProvider;

    this._state$ = new BehaviorSubject<WalletState>({
      ready: false,
      loading: true,
      identity: null,
      balances: new Map(),
      transactions: [],
      pendingTransactions: [],
      error: null,
    });

    // Subscribe to vault state changes
    this.vault.subscribe((vaultState) => this.handleVaultStateChange(vaultState));
  }

  // ===========================================================================
  // Static Factory Methods
  // ===========================================================================

  /**
   * Create a new wallet (fresh setup)
   */
  static async create(config: WalletConfig): Promise<{ wallet: Wallet; recoveryPhrase: string[] }> {
    const { vault, recoveryPhrase } = await createVault(
      {
        name: config.name,
        pin: config.pin,
        biometricEnabled: config.biometric,
      },
      config.storage
    );

    const wallet = new Wallet(vault, new MockIdentityResolver(), new MockFinanceProvider());

    // Initialize state
    await wallet.refresh();

    return { wallet, recoveryPhrase };
  }

  /**
   * Load an existing wallet
   */
  static async load(storage?: VaultStorageProvider): Promise<Wallet> {
    const vault = await loadVault(storage);
    return new Wallet(vault, new MockIdentityResolver(), new MockFinanceProvider());
  }

  /**
   * Connect with custom providers (for production use)
   */
  static async connect(options: {
    vault: MycelixVault;
    identityResolver: IdentityResolver;
    financeProvider: FinanceProvider;
  }): Promise<Wallet> {
    const wallet = new Wallet(options.vault, options.identityResolver, options.financeProvider);
    if (options.vault.isUnlocked) {
      await wallet.refresh();
    }
    return wallet;
  }

  // ===========================================================================
  // Reactive State
  // ===========================================================================

  /** Full state observable */
  get state$(): BehaviorSubject<WalletState> {
    return this._state$;
  }

  /** Current state */
  get state(): WalletState {
    return this._state$.value;
  }

  /** Balance observable (all currencies) */
  get balances$(): Observable<Map<string, Balance>> {
    return this._state$.asObservable().pipe(
      map((s) => s.balances),
      distinctUntilChanged<Map<string, Balance>>()
    );
  }

  /** Transactions observable */
  get transactions$(): Observable<Transaction[]> {
    return this._state$.asObservable().pipe(
      map((s) => [...s.pendingTransactions, ...s.transactions]),
      distinctUntilChanged<Transaction[]>()
    );
  }

  /** Identity observable */
  get identity$(): Observable<Identity | null> {
    return this._state$.asObservable().pipe(
      map((s) => s.identity),
      distinctUntilChanged<Identity | null>()
    );
  }

  /** Events stream (for toasts, notifications) */
  get events$(): ReturnType<typeof Subject.prototype.asObservable> {
    return this._events$.asObservable();
  }

  /** Subscribe to state changes */
  subscribe(observer: (state: WalletState) => void): Subscription {
    return this._state$.subscribe(observer);
  }

  // ===========================================================================
  // Simple Getters (Synchronous)
  // ===========================================================================

  /** Whether wallet is ready to use */
  get ready(): boolean {
    return this._state$.value.ready;
  }

  /** Whether vault is locked */
  get locked(): boolean {
    return !this.vault.isUnlocked;
  }

  /** Current identity */
  get identity(): Identity | null {
    return this._state$.value.identity;
  }

  /** Get balance for a currency */
  balance(currency: Currency = 'MYC'): number {
    return this._state$.value.balances.get(currency)?.available ?? 0;
  }

  /** Get all balances */
  get balances(): Map<Currency, Balance> {
    return this._state$.value.balances;
  }

  /** Get recent transactions */
  get transactions(): Transaction[] {
    const state = this._state$.value;
    return [...state.pendingTransactions, ...state.transactions];
  }

  // ===========================================================================
  // Authentication
  // ===========================================================================

  /**
   * Unlock wallet with PIN
   */
  async unlockWithPin(pin: string): Promise<void> {
    await this.vault.unlock({ pin });
    await this.refresh();
  }

  /**
   * Unlock wallet with biometrics
   */
  async unlockWithBiometric(): Promise<void> {
    await this.vault.unlock({ biometric: true });
    await this.refresh();
  }

  /**
   * Lock wallet
   */
  lock(): void {
    this.vault.lock();
    this.stopAutoRefresh();
    this._state$.next({
      ready: false,
      loading: false,
      identity: null,
      balances: new Map(),
      transactions: [],
      pendingTransactions: [],
      error: null,
    });
  }

  // ===========================================================================
  // Core Operations
  // ===========================================================================

  /**
   * Send funds to another user
   *
   * Supports both human-readable addresses (@alice) and raw agent IDs.
   * Uses optimistic UI - balance updates immediately, confirms in background.
   *
   * @example
   * ```typescript
   * // Send to nickname
   * await wallet.send('@alice', 50);
   *
   * // Send with memo
   * await wallet.send('@bob', 100, 'MYC', { memo: 'Thanks!' });
   *
   * // Send with callbacks
   * await wallet.send('@alice', 50, 'MYC', {
   *   onOptimistic: () => showToast('Sending...'),
   *   onConfirmed: () => showToast('Sent!'),
   *   onError: (e) => showToast(`Failed: ${e.message}`),
   * });
   * ```
   */
  async send(
    to: string,
    amount: number,
    currency: Currency = 'MYC',
    options?: SendOptions
  ): Promise<Transaction> {
    this.requireReady();

    const state = this._state$.value;
    if (!state.identity) throw new Error('No identity');

    // Resolve recipient
    let recipient: Identity;
    if (to.startsWith('@')) {
      const resolved = await this.identityResolver.resolveNickname(to);
      if (!resolved) throw new Error(`User ${to} not found`);
      recipient = resolved;
    } else {
      recipient = await this.identityResolver.resolve(to);
    }

    // Check balance
    const balance = state.balances.get(currency);
    if (!balance || balance.available < amount) {
      throw new Error('Insufficient funds');
    }

    // Create optimistic transaction
    const tempId = `pending-${Date.now()}`;
    const optimisticTx: Transaction = {
      id: tempId,
      from: state.identity,
      to: recipient,
      amount,
      currency,
      direction: 'outgoing',
      status: 'pending',
      memo: options?.memo,
      createdAt: Date.now(),
    };

    // Apply optimistically (instant UI update)
    const newBalances = new Map(state.balances);
    const newBalance = { ...balance };
    newBalance.available -= amount;
    newBalance.pending += amount;
    newBalances.set(currency, newBalance);

    this._state$.next({
      ...state,
      balances: newBalances,
      pendingTransactions: [optimisticTx, ...state.pendingTransactions],
    });

    this._events$.next({ type: 'transaction_pending', transaction: optimisticTx });
    options?.onOptimistic?.();

    try {
      // Sign the transaction
      const txData = new TextEncoder().encode(
        JSON.stringify({ to: recipient.agentId, amount, currency, memo: options?.memo })
      );
      const signature = await this.vault.sign(txData);

      // Submit to chain
      const txId = await this.financeProvider.send(
        state.identity.agentId,
        recipient.agentId,
        amount,
        currency,
        options?.memo,
        signature
      );

      // Update with real transaction
      const confirmedTx: Transaction = {
        ...optimisticTx,
        id: txId,
        status: 'confirmed',
        confirmedAt: Date.now(),
      };

      // Remove pending, add to confirmed
      const finalBalance = { ...newBalance };
      finalBalance.pending -= amount;
      finalBalance.total -= amount;
      newBalances.set(currency, finalBalance);

      this._state$.next({
        ...this._state$.value,
        balances: newBalances,
        pendingTransactions: this._state$.value.pendingTransactions.filter((t) => t.id !== tempId),
        transactions: [confirmedTx, ...this._state$.value.transactions],
      });

      this._events$.next({ type: 'transaction_confirmed', transaction: confirmedTx });
      options?.onConfirmed?.();

      return confirmedTx;
    } catch (error) {
      // Rollback optimistic update
      const rollbackBalance = { ...balance };
      newBalances.set(currency, rollbackBalance);

      const failedTx: Transaction = {
        ...optimisticTx,
        status: 'failed',
        error: error instanceof Error ? error.message : 'Unknown error',
      };

      this._state$.next({
        ...this._state$.value,
        balances: newBalances,
        pendingTransactions: this._state$.value.pendingTransactions.filter((t) => t.id !== tempId),
      });

      this._events$.next({ type: 'transaction_failed', transaction: failedTx, error: error as Error });
      options?.onError?.(error as Error);

      throw error;
    }
  }

  /**
   * Request payment from another user (generates QR/link)
   */
  async requestPayment(
    amount: number,
    currency: Currency = 'MYC',
    memo?: string
  ): Promise<{ qrData: string; link: string }> {
    this.requireReady();

    const state = this._state$.value;
    if (!state.identity) throw new Error('No identity');

    const request = {
      type: 'payment_request',
      to: state.identity.agentId,
      amount,
      currency,
      memo,
      timestamp: Date.now(),
    };

    const qrData = JSON.stringify(request);
    const link = `mycelix://pay?${new URLSearchParams({
      to: state.identity.agentId,
      amount: amount.toString(),
      currency,
      ...(memo && { memo }),
    }).toString()}`;

    return { qrData, link };
  }

  /**
   * Resolve an address to a human-readable identity
   */
  async resolveIdentity(addressOrNickname: string): Promise<Identity | null> {
    if (addressOrNickname.startsWith('@')) {
      return this.identityResolver.resolveNickname(addressOrNickname);
    }
    return this.identityResolver.resolve(addressOrNickname);
  }

  /**
   * Search for users by name or nickname
   */
  async searchUsers(query: string): Promise<Identity[]> {
    return this.identityResolver.search(query);
  }

  // ===========================================================================
  // Refresh & Sync
  // ===========================================================================

  /**
   * Refresh all wallet data from chain
   */
  async refresh(): Promise<void> {
    if (!this.vault.isUnlocked) return;

    const account = this.vault.getCurrentAccount();
    if (!account) return;

    this._state$.next({ ...this._state$.value, loading: true });

    try {
      // Get identity
      const identity = await this.identityResolver.resolve(account.agentId);
      if (account.profile) {
        identity.nickname = account.profile.nickname;
        identity.avatar = account.profile.avatar;
        identity.did = account.profile.did;
      }

      // Get balances
      const balances = await this.financeProvider.getAllBalances(account.agentId);

      // Get transactions and hydrate identities
      const rawTransactions = await this.financeProvider.getTransactions(account.agentId);
      const transactions = await Promise.all(
        rawTransactions.map(async (tx) => ({
          ...tx,
          from: await this.identityResolver.resolve(tx.from.agentId),
          to: await this.identityResolver.resolve(tx.to.agentId),
          direction: (tx.from.agentId === account.agentId ? 'outgoing' : 'incoming') as TransactionDirection,
        }))
      );

      this._state$.next({
        ready: true,
        loading: false,
        identity,
        balances,
        transactions,
        pendingTransactions: this._state$.value.pendingTransactions,
        error: null,
      });

      this._events$.next({ type: 'refresh_complete' });
    } catch (error) {
      this._state$.next({
        ...this._state$.value,
        loading: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      this._events$.next({ type: 'refresh_failed', error: error as Error });
    }
  }

  /**
   * Start auto-refresh (polls for updates)
   */
  startAutoRefresh(intervalMs: number = 30000): void {
    this.stopAutoRefresh();
    this.refreshInterval = setInterval(() => {
      void this.refresh();
    }, intervalMs);
  }

  /**
   * Stop auto-refresh
   */
  stopAutoRefresh(): void {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
      this.refreshInterval = null;
    }
  }

  // ===========================================================================
  // Vault Access (for power users)
  // ===========================================================================

  /**
   * Access to underlying vault (key management)
   */
  get vaultAccess(): MycelixVault {
    return this.vault;
  }

  /**
   * Get recovery phrase (requires unlocked vault)
   */
  async getRecoveryPhrase(): Promise<string[]> {
    return this.vault.getRecoveryPhrase();
  }

  /**
   * Create a new account in the vault
   */
  async createAccount(name: string): Promise<void> {
    await this.vault.createAccount({ name });
    await this.refresh();
  }

  /**
   * Switch to a different account
   */
  async switchAccount(accountId: AccountId): Promise<void> {
    this.vault.switchAccount(accountId);
    await this.refresh();
  }

  /**
   * Get all accounts
   */
  get accounts(): ReturnType<typeof this.vault.getAccounts> {
    return this.vault.getAccounts();
  }

  // ===========================================================================
  // Private Helpers
  // ===========================================================================

  private requireReady(): void {
    if (!this._state$.value.ready) {
      throw new Error('Wallet not ready. Unlock first.');
    }
  }

  private handleVaultStateChange(vaultState: VaultState): void {
    if (vaultState.status === 'locked') {
      this._state$.next({
        ready: false,
        loading: false,
        identity: null,
        balances: new Map(),
        transactions: [],
        pendingTransactions: [],
        error: null,
      });
    } else if (vaultState.status === 'unlocked' && !this._state$.value.ready) {
      void this.refresh();
    }
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a new wallet
 */
export async function createWallet(
  config: WalletConfig
): Promise<{ wallet: Wallet; recoveryPhrase: string[] }> {
  return Wallet.create(config);
}

/**
 * Load an existing wallet
 */
export async function loadWallet(storage?: VaultStorageProvider): Promise<Wallet> {
  return Wallet.load(storage);
}
