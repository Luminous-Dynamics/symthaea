/**
 * Holochain Finance Provider - The Muscle
 *
 * Connects the wallet to the Finance hApp zomes:
 * - Payments zome for direct payments
 * - TEND zome for time-based balances
 * - CGC zome for civic currency
 *
 * Integrates with OfflineQueue for resilient transactions.
 *
 * @example
 * ```typescript
 * const finance = new HolochainFinanceProvider(conductorManager, offlineQueue);
 *
 * // Get balances
 * const balances = await finance.getBalances();
 *
 * // Send payment (queued if offline)
 * const tx = await finance.send('@alice', 100, 'MYC', 'Coffee');
 * ```
 */

import { BehaviorSubject } from '../reactive/index.js';

import type { ConductorManager } from './conductor.js';
import type {
  Currency,
  Transaction,
  TransactionStatus,
  TransactionDirection,
  Balance,
  FinanceProvider,
  Identity,
} from './index.js';
import type {
  OfflineQueue,
  QueuedOperation,
  SendPayload,
  SyncEvent,
} from './offline-queue.js';

// =============================================================================
// Zome Types (mirroring Rust types)
// =============================================================================

/** Input for sending payment via payments zome */
interface SendPaymentInput {
  from_did: string;
  to_did: string;
  amount: number;
  currency: string;
  payment_type: PaymentType;
  memo?: string;
}

/** Payment type enum matching Rust */
type PaymentType =
  | { Direct: null }
  | { Channel: string }
  | { Escrow: string }
  | { Recurring: string };

/** Payment record from zome */
interface PaymentRecord {
  id: string;
  from_did: string;
  to_did: string;
  amount: number;
  currency: string;
  payment_type: PaymentType;
  status: ZomeTransferStatus;
  memo?: string;
  created: number; // Timestamp in microseconds
  completed?: number;
}

/** Transfer status from zome */
type ZomeTransferStatus = 'Pending' | 'Completed' | 'Failed' | 'Refunded' | 'Cancelled';

/** TEND balance info from zome */
interface TendBalanceInfo {
  member_did: string;
  dao_did: string;
  balance: number;
  can_provide: boolean;
  can_receive: boolean;
  total_provided: number;
  total_received: number;
  exchange_count: number;
}

/** Input for getting TEND balance */
interface GetBalanceInput {
  member_did: string;
  dao_did: string;
}

// =============================================================================
// Finance State
// =============================================================================

/** Finance provider state */
export interface FinanceState {
  balances: Map<string, Balance>;
  transactions: Transaction[];
  lastFetchedAt: number | null;
  isLoading: boolean;
  error: string | null;
}

/** Configuration for HolochainFinanceProvider */
export interface HolochainFinanceConfig {
  /** Current user's DID */
  myDid?: string;
  /** DAO context for TEND balances */
  daoDid?: string;
  /** Role name in conductor (default: 'finance') */
  roleName?: string;
  /** Offline queue for resilient transactions */
  offlineQueue?: OfflineQueue;
  /** Auto-refresh interval in ms (default: 30000) */
  refreshIntervalMs?: number;
}

// =============================================================================
// Holochain Finance Provider
// =============================================================================

/**
 * Finance provider that connects to the Holochain Finance hApp.
 * Implements the FinanceProvider interface for use with the Wallet.
 */
export class HolochainFinanceProvider implements FinanceProvider {
  private conductor: ConductorManager;
  private offlineQueue?: OfflineQueue;
  private myDid: string;
  private daoDid: string;
  private _state$: BehaviorSubject<FinanceState>;
  private refreshInterval?: ReturnType<typeof setInterval>;
  private unsubscribeSync?: () => void;

  // Configuration
  private roleName: string;
  private refreshIntervalMs: number;

  constructor(
    conductor: ConductorManager,
    options?: {
      myDid?: string;
      daoDid?: string;
      roleName?: string;
      offlineQueue?: OfflineQueue;
      refreshIntervalMs?: number;
    }
  ) {
    this.conductor = conductor;
    this.offlineQueue = options?.offlineQueue;
    this.myDid = options?.myDid ?? '';
    this.daoDid = options?.daoDid ?? 'default-dao';
    this.roleName = options?.roleName ?? 'finance';
    this.refreshIntervalMs = options?.refreshIntervalMs ?? 30000;

    this._state$ = new BehaviorSubject<FinanceState>({
      balances: new Map(),
      transactions: [],
      lastFetchedAt: null,
      isLoading: false,
      error: null,
    });

    // Listen for offline queue sync events to update transaction status
    if (this.offlineQueue) {
      this.unsubscribeSync = this.offlineQueue.onSyncEvent((event) => {
        this.handleSyncEvent(event);
      });
    }
  }

  // ===========================================================================
  // Reactive State
  // ===========================================================================

  /** Observable finance state */
  get state$(): BehaviorSubject<FinanceState> {
    return this._state$;
  }

  /** Current state */
  get state(): FinanceState {
    return this._state$.value;
  }

  /** Current balances */
  get balances(): Map<string, Balance> {
    return this._state$.value.balances;
  }

  /** Transaction history */
  get transactions(): Transaction[] {
    return this._state$.value.transactions;
  }

  // ===========================================================================
  // Initialization
  // ===========================================================================

  /**
   * Set the current user's DID.
   * Must be called before making transactions.
   */
  setMyDid(did: string): void {
    this.myDid = did;
  }

  /**
   * Set the DAO context for TEND balances.
   */
  setDaoDid(did: string): void {
    this.daoDid = did;
  }

  /**
   * Start periodic balance refresh.
   */
  startAutoRefresh(): void {
    if (this.refreshInterval) return;

    this.refreshInterval = setInterval(() => {
      if (this.conductor.isConnected) {
        this.refresh().catch(console.warn);
      }
    }, this.refreshIntervalMs);
  }

  /**
   * Stop periodic refresh.
   */
  stopAutoRefresh(): void {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
      this.refreshInterval = undefined;
    }
  }

  // ===========================================================================
  // FinanceProvider Interface (Wallet compatibility)
  // ===========================================================================

  /**
   * Get balance for a specific currency.
   * @param agentId - The agent ID (uses myDid if this matches)
   * @param currency - The currency to get balance for
   */
  async getBalance(agentId: string, currency: Currency): Promise<Balance> {
    // For now, we only support querying our own balance
    if (agentId !== this.myDid && this.myDid) {
      console.warn(`Querying balance for ${agentId} but myDid is ${this.myDid}`);
    }

    const balances = await this.getAllBalances(agentId);
    return (
      balances.get(currency) ?? {
        currency,
        available: 0,
        pending: 0,
        total: 0,
      }
    );
  }

  /**
   * Get all balances for an agent.
   * @param agentId - The agent ID to query
   */
  async getAllBalances(agentId: string): Promise<Map<Currency, Balance>> {
    // Use provided agentId or fall back to myDid
    const did = agentId || this.myDid;
    if (!did) {
      throw new Error('No agent ID provided and myDid not set');
    }

    this.updateState({ isLoading: true, error: null });

    try {
      const balances = new Map<Currency, Balance>();

      // Get TEND balance
      try {
        const tendBalance = await this.getTendBalance();
        balances.set('TEND', {
          currency: 'TEND',
          available: tendBalance.balance,
          pending: 0,
          total: tendBalance.balance,
        });
      } catch (e) {
        console.warn('Failed to fetch TEND balance:', e);
      }

      // Get CGC balance (if available)
      try {
        const cgcBalance = await this.getCgcBalance();
        balances.set('CGC', {
          currency: 'CGC',
          available: cgcBalance,
          pending: 0,
          total: cgcBalance,
        });
      } catch {
        // CGC not available, that's ok
      }

      // Add MYC (main currency) - aggregated or custom
      // For now, mirror TEND as MYC
      const mycBalance = balances.get('TEND');
      if (mycBalance) {
        balances.set('MYC', { ...mycBalance, currency: 'MYC' });
      }

      this.updateState({
        balances,
        lastFetchedAt: Date.now(),
        isLoading: false,
      });

      return balances;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Failed to fetch balances';
      this.updateState({ isLoading: false, error: errorMsg });
      throw error;
    }
  }

  /**
   * Get transaction history.
   * @param agentId - The agent ID to query
   * @param limit - Maximum number of transactions to return
   */
  async getTransactions(agentId: string, limit?: number): Promise<Transaction[]> {
    // Use provided agentId or fall back to myDid
    const did = agentId || this.myDid;
    if (!did) {
      throw new Error('No agent ID provided and myDid not set');
    }

    this.updateState({ isLoading: true, error: null });

    try {
      // Fetch payment history from payments zome
      const records = await this.conductor.callZome<Array<{ entry: PaymentRecord }>>({
        role_name: this.roleName,
        zome_name: 'payments',
        fn_name: 'get_payment_history',
        payload: did,
      });

      // Transform to Transaction type
      const transactions: Transaction[] = records.map((record) => {
        const payment = record.entry;
        return this.paymentToTransaction(payment);
      });

      // Sort by date descending
      transactions.sort((a, b) => b.createdAt - a.createdAt);

      // Apply limit
      const limited = limit ? transactions.slice(0, limit) : transactions;

      this.updateState({
        transactions: limited,
        lastFetchedAt: Date.now(),
        isLoading: false,
      });

      return limited;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Failed to fetch transactions';
      this.updateState({ isLoading: false, error: errorMsg });
      throw error;
    }
  }

  /**
   * Send a payment.
   * @param from - Sender agent ID (must match myDid)
   * @param to - Recipient agent ID or DID
   * @param amount - Amount to send
   * @param currency - Currency to send
   * @param memo - Optional memo
   * @param _signature - Optional signature (not used - signing handled internally)
   * @returns Transaction ID
   */
  async send(
    from: string,
    to: string,
    amount: number,
    currency: Currency,
    memo?: string,
    _signature?: Uint8Array
  ): Promise<string> {
    // Validate from matches myDid
    if (from !== this.myDid && this.myDid) {
      throw new Error(`Cannot send from ${from} - myDid is ${this.myDid}`);
    }

    // If we have an offline queue and we're offline, queue it
    if (this.offlineQueue && !this.conductor.isOnline) {
      const tx = await this.queueSend(to, amount, currency, memo);
      return tx.id;
    }

    // Otherwise, send directly
    const tx = await this.sendDirect(to, amount, currency, memo);
    return tx.id;
  }

  /**
   * Get a specific transaction by ID.
   * @param id - Transaction ID
   */
  async getTransaction(id: string): Promise<Transaction | null> {
    try {
      const record = await this.conductor.callZome<{ entry: PaymentRecord } | null>({
        role_name: this.roleName,
        zome_name: 'payments',
        fn_name: 'get_payment',
        payload: id,
      });

      if (!record) return null;
      return this.paymentToTransaction(record.entry);
    } catch {
      return null;
    }
  }

  // ===========================================================================
  // Convenience Methods (not in FinanceProvider interface)
  // ===========================================================================

  /**
   * Get all balances for the current user (convenience method).
   */
  async getBalances(): Promise<Map<string, Balance>> {
    return this.getAllBalances(this.myDid);
  }

  /**
   * Refresh all data (balances + transactions).
   */
  async refresh(): Promise<void> {
    await Promise.all([this.getBalances(), this.getTransactions(this.myDid)]);
  }

  // ===========================================================================
  // Direct Zome Calls
  // ===========================================================================

  /**
   * Send payment directly via zome call.
   */
  private async sendDirect(
    to: string,
    amount: number,
    currency: Currency,
    memo?: string
  ): Promise<Transaction> {
    const input: SendPaymentInput = {
      from_did: this.myDid,
      to_did: to,
      amount,
      currency,
      payment_type: { Direct: null },
      memo,
    };

    const record = await this.conductor.callZome<{ entry: PaymentRecord }>({
      role_name: this.roleName,
      zome_name: 'payments',
      fn_name: 'send_payment',
      payload: input,
    });

    const transaction = this.paymentToTransaction(record.entry);

    // Add to local transaction list
    const transactions = [transaction, ...this._state$.value.transactions];
    this.updateState({ transactions });

    // Refresh balances in background
    this.getBalances().catch(console.warn);

    return transaction;
  }

  /**
   * Queue a send operation for later execution.
   */
  private async queueSend(
    to: string,
    amount: number,
    currency: Currency,
    memo?: string
  ): Promise<Transaction> {
    if (!this.offlineQueue) {
      throw new Error('Offline queue not configured');
    }

    const queued = await this.offlineQueue.enqueueSend(to, amount, currency, memo);

    // Create optimistic transaction (optimistic IDs have a prefix we can track)
    const transaction: Transaction = {
      id: `optimistic-${queued.id}`,
      from: { agentId: this.myDid, did: this.myDid },
      to: { agentId: to, did: to },
      amount,
      currency,
      status: 'pending',
      direction: 'outgoing' as TransactionDirection,
      createdAt: Date.now(),
      memo,
    };

    // Add to local transaction list
    const transactions = [transaction, ...this._state$.value.transactions];
    this.updateState({ transactions });

    return transaction;
  }

  /**
   * Get TEND balance from zome.
   */
  private async getTendBalance(): Promise<TendBalanceInfo> {
    const input: GetBalanceInput = {
      member_did: this.myDid,
      dao_did: this.daoDid,
    };

    return this.conductor.callZome<TendBalanceInfo>({
      role_name: this.roleName,
      zome_name: 'tend',
      fn_name: 'get_balance',
      payload: input,
    });
  }

  /**
   * Get CGC balance from zome.
   */
  private async getCgcBalance(): Promise<number> {
    // CGC uses a simpler balance structure
    const result = await this.conductor.callZome<{ balance: number }>({
      role_name: this.roleName,
      zome_name: 'cgc',
      fn_name: 'get_balance',
      payload: this.myDid,
    });
    return result.balance;
  }

  // ===========================================================================
  // Offline Queue Integration
  // ===========================================================================

  /**
   * Handle sync events from offline queue.
   */
  private handleSyncEvent(event: SyncEvent): void {
    if (event.type === 'item_synced' && event.itemId) {
      // Find the optimistic transaction (by ID prefix) and update its status
      const optimisticId = `optimistic-${event.itemId}`;
      const transactions = this._state$.value.transactions.map((tx) => {
        if (tx.id === optimisticId) {
          return {
            ...tx,
            status: 'confirmed' as TransactionStatus,
            confirmedAt: Date.now(),
          };
        }
        return tx;
      });
      this.updateState({ transactions });

      // Refresh to get real transaction data
      this.refresh().catch(console.warn);
    }

    if (event.type === 'item_failed' && event.itemId) {
      // Mark the transaction as failed
      const optimisticId = `optimistic-${event.itemId}`;
      const transactions = this._state$.value.transactions.map((tx) => {
        if (tx.id === optimisticId) {
          return {
            ...tx,
            status: 'failed' as TransactionStatus,
            error: event.error,
          };
        }
        return tx;
      });
      this.updateState({ transactions });
    }
  }

  /**
   * Create an executor for the offline queue that uses this provider.
   */
  createQueueExecutor() {
    return {
      execute: async (operation: QueuedOperation<SendPayload>) => {
        if (operation.type !== 'send') {
          return { success: false, error: `Unknown operation type: ${operation.type}` };
        }

        try {
          const payload = operation.payload;
          await this.sendDirect(payload.to, payload.amount, payload.currency, payload.memo);
          return { success: true };
        } catch (error) {
          return {
            success: false,
            error: error instanceof Error ? error.message : 'Send failed',
            retryable: true,
          };
        }
      },
      canExecute: (operation: QueuedOperation) => {
        return operation.type === 'send' && this.conductor.isConnected;
      },
    };
  }

  // ===========================================================================
  // Helpers
  // ===========================================================================

  /**
   * Convert zome payment record to Transaction type.
   */
  private paymentToTransaction(payment: PaymentRecord): Transaction {
    const isOutgoing = payment.from_did === this.myDid;

    return {
      id: payment.id,
      from: { agentId: payment.from_did, did: payment.from_did },
      to: { agentId: payment.to_did, did: payment.to_did },
      amount: payment.amount,
      currency: payment.currency,
      status: this.mapZomeStatus(payment.status),
      direction: isOutgoing ? 'outgoing' : 'incoming',
      createdAt: Math.floor(payment.created / 1000), // Convert from microseconds
      memo: payment.memo,
      confirmedAt: payment.completed ? Math.floor(payment.completed / 1000) : undefined,
    };
  }

  /**
   * Map zome status to SDK status.
   */
  private mapZomeStatus(status: ZomeTransferStatus): TransactionStatus {
    switch (status) {
      case 'Pending':
        return 'pending';
      case 'Completed':
        return 'confirmed';
      case 'Failed':
      case 'Cancelled':
        return 'failed';
      case 'Refunded':
        return 'confirmed'; // Refunds are completed transactions
      default:
        return 'pending';
    }
  }

  private updateState(partial: Partial<FinanceState>): void {
    const current = this._state$.value;
    this._state$.next({
      ...current,
      ...partial,
      // Preserve Map reference if not changing
      balances: partial.balances ?? current.balances,
    });
  }

  // ===========================================================================
  // Cleanup
  // ===========================================================================

  /**
   * Clean up resources.
   */
  destroy(): void {
    this.stopAutoRefresh();
    if (this.unsubscribeSync) {
      this.unsubscribeSync();
    }
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a Holochain finance provider.
 */
export function createHolochainFinanceProvider(
  conductor: ConductorManager,
  options?: {
    myDid?: string;
    daoDid?: string;
    roleName?: string;
    offlineQueue?: OfflineQueue;
    refreshIntervalMs?: number;
  }
): HolochainFinanceProvider {
  return new HolochainFinanceProvider(conductor, options);
}

/**
 * Create a mock finance provider for testing.
 */
export function createMockFinanceProvider(): FinanceProvider {
  const myIdentity: Identity = { agentId: 'did:mycelix:me', did: 'did:mycelix:me' };
  const aliceIdentity: Identity = { agentId: 'did:mycelix:alice', did: 'did:mycelix:alice' };
  const bobIdentity: Identity = { agentId: 'did:mycelix:bob', did: 'did:mycelix:bob' };

  const balances = new Map<Currency, Balance>([
    ['MYC', { currency: 'MYC', available: 1000, pending: 0, total: 1000 }],
    ['TEND', { currency: 'TEND', available: 25, pending: 0, total: 25 }],
  ]);

  const transactions: Transaction[] = [
    {
      id: 'mock-tx-1',
      from: myIdentity,
      to: aliceIdentity,
      amount: 50,
      currency: 'MYC',
      status: 'confirmed',
      direction: 'outgoing',
      createdAt: Date.now() - 3600000,
      memo: 'Coffee',
    },
    {
      id: 'mock-tx-2',
      from: bobIdentity,
      to: myIdentity,
      amount: 100,
      currency: 'MYC',
      status: 'confirmed',
      direction: 'incoming',
      createdAt: Date.now() - 7200000,
      memo: 'Lunch',
    },
  ];

  return {
    getBalance: async (_agentId: string, currency: Currency): Promise<Balance> => {
      return balances.get(currency) ?? { currency, available: 0, pending: 0, total: 0 };
    },
    getAllBalances: async (_agentId: string): Promise<Map<Currency, Balance>> => {
      return balances;
    },
    getTransactions: async (_agentId: string, limit?: number): Promise<Transaction[]> => {
      return limit !== undefined ? transactions.slice(0, limit) : transactions;
    },
    send: async (_from: string, to: string, amount: number, currency: Currency, memo?: string): Promise<string> => {
      const txId = `mock-tx-${Date.now()}`;
      const tx: Transaction = {
        id: txId,
        from: myIdentity,
        to: { agentId: to, did: to },
        amount,
        currency,
        status: 'confirmed',
        direction: 'outgoing',
        createdAt: Date.now(),
        memo,
      };
      transactions.unshift(tx);
      const bal = balances.get(currency);
      if (bal) {
        bal.available -= amount;
        bal.total -= amount;
      }
      return txId; // Return transaction ID, not the transaction object
    },
    getTransaction: async (id: string): Promise<Transaction | null> => {
      return transactions.find((tx) => tx.id === id) ?? null;
    },
  };
}
