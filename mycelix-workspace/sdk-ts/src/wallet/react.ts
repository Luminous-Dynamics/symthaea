// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React Hooks for Mycelix Wallet
 *
 * Provides hooks for seamless React integration with the wallet.
 * Features:
 * - Automatic subscription management
 * - Optimistic UI patterns built-in
 * - SSR-safe (no window references)
 * - TypeScript-first with full type inference
 *
 * @example
 * ```tsx
 * function WalletBalance() {
 *   const { balance, loading } = useBalance('MYC');
 *   if (loading) return <Spinner />;
 *   return <Text>{balance} MYC</Text>;
 * }
 *
 * function SendButton() {
 *   const { send, sending } = useSend();
 *   return (
 *     <Button
 *       onClick={() => send('@alice', 50)}
 *       disabled={sending}
 *     >
 *       {sending ? 'Sending...' : 'Send 50 MYC'}
 *     </Button>
 *   );
 * }
 * ```
 */

// NOTE: This file provides hooks that work with any React-like framework.
// It doesn't import React directly to avoid dependency issues.
// Users should use their framework's state primitives.

import type {
  WalletBridge,
  TransactionRecommendation,
  VerificationStatus,
} from './bridge-integration.js';
import type { Wallet, WalletState, Currency, Balance, Transaction, Identity, WalletEvent } from './index.js';
import type { ReputationQueryResponse } from '../bridge/cross-happ.js';
import type { Subscription } from '../reactive/index.js';

// =============================================================================
// Types for Hook Return Values
// =============================================================================

export interface UseWalletReturn {
  /** Full wallet state */
  state: WalletState;
  /** Whether wallet is ready */
  ready: boolean;
  /** Whether wallet is locked */
  locked: boolean;
  /** Current identity */
  identity: Identity | null;
  /** Unlock with PIN */
  unlockWithPin: (pin: string) => Promise<void>;
  /** Unlock with biometrics */
  unlockWithBiometric: () => Promise<void>;
  /** Lock the wallet */
  lock: () => void;
  /** Refresh data */
  refresh: () => Promise<void>;
}

export interface UseBalanceReturn {
  /** Available balance */
  balance: number;
  /** Pending balance */
  pending: number;
  /** Total balance */
  total: number;
  /** Full balance object */
  balanceData: Balance | null;
  /** Whether loading */
  loading: boolean;
}

export interface UseTransactionsReturn {
  /** All transactions (pending + confirmed) */
  transactions: Transaction[];
  /** Pending transactions only */
  pending: Transaction[];
  /** Confirmed transactions only */
  confirmed: Transaction[];
  /** Whether loading */
  loading: boolean;
  /** Refresh transactions */
  refresh: () => Promise<void>;
}

export interface UseSendReturn {
  /** Send funds */
  send: (to: string, amount: number, currency?: Currency, memo?: string) => Promise<Transaction>;
  /** Whether currently sending */
  sending: boolean;
  /** Last error if any */
  error: Error | null;
  /** Clear error */
  clearError: () => void;
}

export interface UseIdentityReturn {
  /** Current user identity */
  identity: Identity | null;
  /** Resolve an address to identity */
  resolve: (addressOrNickname: string) => Promise<Identity | null>;
  /** Search for users */
  search: (query: string) => Promise<Identity[]>;
}

// =============================================================================
// Hook Factory (Framework-agnostic)
// =============================================================================

/**
 * Creates wallet hooks for your React-like framework.
 *
 * @example
 * ```tsx
 * // In your app setup
 * import { useState, useEffect, useRef, useCallback } from 'react';
 * import { createWalletHooks } from '@mycelix/sdk/wallet/react';
 * import { wallet } from './wallet-instance';
 *
 * export const {
 *   useWallet,
 *   useBalance,
 *   useTransactions,
 *   useSend,
 *   useIdentity,
 * } = createWalletHooks(wallet, { useState, useEffect, useRef, useCallback });
 * ```
 */
export function createWalletHooks(
  wallet: Wallet,
  react: {
    useState: <T>(initial: T) => [T, (value: T | ((prev: T) => T)) => void];
    useEffect: (effect: () => void | (() => void), deps?: unknown[]) => void;
    useRef: <T>(initial: T) => { current: T };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    useCallback: <T extends (...args: any[]) => any>(fn: T, deps: unknown[]) => T;
  }
) {
  const { useState, useEffect, useRef, useCallback } = react;

  /**
   * Main wallet hook - subscribes to full state
   */
  function useWallet(): UseWalletReturn {
    const [state, setState] = useState<WalletState>(wallet.state);

    useEffect(() => {
      const subscription = wallet.subscribe((newState) => {
        setState(newState);
      });
      return () => subscription.unsubscribe();
    }, []);

    const unlockWithPin = useCallback(async (pin: string) => {
      await wallet.unlockWithPin(pin);
    }, []);

    const unlockWithBiometric = useCallback(async () => {
      await wallet.unlockWithBiometric();
    }, []);

    const lock = useCallback(() => {
      wallet.lock();
    }, []);

    const refresh = useCallback(async () => {
      await wallet.refresh();
    }, []);

    return {
      state,
      ready: state.ready,
      locked: wallet.locked,
      identity: state.identity,
      unlockWithPin,
      unlockWithBiometric,
      lock,
      refresh,
    };
  }

  /**
   * Balance hook - subscribes to balance changes for a currency
   */
  function useBalance(currency: Currency = 'MYC'): UseBalanceReturn {
    const [balance, setBalance] = useState<Balance | null>(wallet.balances.get(currency) ?? null);
    const [loading, setLoading] = useState(!wallet.ready);

    useEffect(() => {
      const subscription = wallet.balances$.subscribe((balances) => {
        setBalance(balances.get(currency) ?? null);
        setLoading(false);
      });
      return () => subscription.unsubscribe();
    }, [currency]);

    return {
      balance: balance?.available ?? 0,
      pending: balance?.pending ?? 0,
      total: balance?.total ?? 0,
      balanceData: balance,
      loading,
    };
  }

  /**
   * Transactions hook - subscribes to transaction changes
   */
  function useTransactions(limit?: number): UseTransactionsReturn {
    const [transactions, setTransactions] = useState<Transaction[]>(wallet.transactions);
    const [loading, setLoading] = useState(!wallet.ready);

    useEffect(() => {
      const subscription = wallet.transactions$.subscribe((txs) => {
        setTransactions(limit ? txs.slice(0, limit) : txs);
        setLoading(false);
      });
      return () => subscription.unsubscribe();
    }, [limit]);

    const refresh = useCallback(async () => {
      setLoading(true);
      await wallet.refresh();
      setLoading(false);
    }, []);

    return {
      transactions,
      pending: transactions.filter((t) => t.status === 'pending'),
      confirmed: transactions.filter((t) => t.status === 'confirmed'),
      loading,
      refresh,
    };
  }

  /**
   * Send hook - provides send function with loading state
   */
  function useSend(): UseSendReturn {
    const [sending, setSending] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    const send = useCallback(
      async (to: string, amount: number, currency: Currency = 'MYC', memo?: string) => {
        setSending(true);
        setError(null);

        try {
          const tx = await wallet.send(to, amount, currency, { memo });
          setSending(false);
          return tx;
        } catch (err) {
          setSending(false);
          const error = err instanceof Error ? err : new Error(String(err));
          setError(error);
          throw error;
        }
      },
      []
    );

    const clearError = useCallback(() => {
      setError(null);
    }, []);

    return { send, sending, error, clearError };
  }

  /**
   * Identity hook - provides identity resolution
   */
  function useIdentity(): UseIdentityReturn {
    const [identity, setIdentity] = useState<Identity | null>(wallet.identity);

    useEffect(() => {
      const subscription = wallet.identity$.subscribe((id) => {
        setIdentity(id);
      });
      return () => subscription.unsubscribe();
    }, []);

    const resolve = useCallback(async (addressOrNickname: string) => {
      return wallet.resolveIdentity(addressOrNickname);
    }, []);

    const search = useCallback(async (query: string) => {
      return wallet.searchUsers(query);
    }, []);

    return { identity, resolve, search };
  }

  /**
   * Events hook - subscribes to wallet events (for toasts/notifications)
   */
  function useWalletEvents(handler: (event: WalletEvent) => void): void {
    const handlerRef = useRef(handler);
    handlerRef.current = handler;

    useEffect(() => {
      const subscription = wallet.events$.subscribe((event) => {
        handlerRef.current(event);
      });
      return () => subscription.unsubscribe();
    }, []);
  }

  /**
   * Optimistic send hook - applies changes immediately, reverts on failure
   */
  function useOptimisticSend() {
    const [optimisticBalance, setOptimisticBalance] = useState<number | null>(null);
    const [sending, setSending] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    const send = useCallback(
      async (to: string, amount: number, currency: Currency = 'MYC', memo?: string) => {
        const currentBalance = wallet.balance(currency);

        // Apply optimistically
        setOptimisticBalance(currentBalance - amount);
        setSending(true);
        setError(null);

        try {
          const tx = await wallet.send(to, amount, currency, { memo });
          setOptimisticBalance(null); // Real balance will update via subscription
          setSending(false);
          return tx;
        } catch (err) {
          // Revert optimistic update
          setOptimisticBalance(null);
          setSending(false);
          const error = err instanceof Error ? err : new Error(String(err));
          setError(error);
          throw error;
        }
      },
      []
    );

    return { send, sending, error, optimisticBalance };
  }

  return {
    useWallet,
    useBalance,
    useTransactions,
    useSend,
    useIdentity,
    useWalletEvents,
    useOptimisticSend,
  };
}

// =============================================================================
// Optimistic UI Utilities
// =============================================================================

/**
 * Create an optimistic state manager
 */
export class OptimisticStateManager<T> {
  private confirmed: T;
  private pending: Array<{ id: string; apply: (state: T) => T; revert: (state: T) => T }> = [];
  private listeners: Set<(state: T) => void> = new Set();

  constructor(initialState: T) {
    this.confirmed = initialState;
  }

  /** Get current optimistic state */
  get state(): T {
    return this.pending.reduce((state, op) => op.apply(state), this.confirmed);
  }

  /** Apply an optimistic update */
  applyOptimistic(
    id: string,
    apply: (state: T) => T,
    revert: (state: T) => T
  ): () => void {
    this.pending.push({ id, apply, revert });
    this.notify();

    // Return function to commit or rollback
    return () => this.rollback(id);
  }

  /** Commit an optimistic update (it succeeded) */
  commit(id: string, newConfirmedState: T): void {
    this.confirmed = newConfirmedState;
    this.pending = this.pending.filter((op) => op.id !== id);
    this.notify();
  }

  /** Rollback an optimistic update (it failed) */
  rollback(id: string): void {
    this.pending = this.pending.filter((op) => op.id !== id);
    this.notify();
  }

  /** Subscribe to state changes */
  subscribe(listener: (state: T) => void): Subscription {
    this.listeners.add(listener);
    listener(this.state);

    return {
      unsubscribe: () => {
        this.listeners.delete(listener);
      },
      closed: false,
    };
  }

  private notify(): void {
    const state = this.state;
    this.listeners.forEach((l) => l(state));
  }
}

/**
 * Helper to create optimistic transaction
 */
export function createOptimisticTransaction(
  from: Identity,
  to: Identity,
  amount: number,
  currency: Currency,
  memo?: string
): Transaction {
  return {
    id: `optimistic-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    from,
    to,
    amount,
    currency,
    direction: 'outgoing',
    status: 'pending',
    memo,
    createdAt: Date.now(),
  };
}

/**
 * Helper to create optimistic balance update
 */
export function applyOptimisticBalanceChange(
  balances: Map<Currency, Balance>,
  currency: Currency,
  change: number
): Map<Currency, Balance> {
  const result = new Map(balances);
  const current = result.get(currency) ?? {
    currency,
    available: 0,
    pending: 0,
    total: 0,
  };

  result.set(currency, {
    ...current,
    available: current.available + change,
    pending: current.pending - change,
    // total stays same until confirmed
  });

  return result;
}

// =============================================================================
// Toast/Notification Helpers
// =============================================================================

/**
 * Create a toast message from wallet event
 */
export function eventToToast(event: WalletEvent): {
  type: 'success' | 'error' | 'info';
  title: string;
  message: string;
} | null {
  switch (event.type) {
    case 'transaction_pending':
      return {
        type: 'info',
        title: 'Sending...',
        message: `Sending ${event.transaction.amount} ${event.transaction.currency} to ${
          event.transaction.to.nickname ?? event.transaction.to.agentId.slice(0, 8)
        }`,
      };

    case 'transaction_confirmed':
      return {
        type: 'success',
        title: 'Sent!',
        message: `${event.transaction.amount} ${event.transaction.currency} sent to ${
          event.transaction.to.nickname ?? event.transaction.to.agentId.slice(0, 8)
        }`,
      };

    case 'transaction_failed':
      return {
        type: 'error',
        title: 'Failed',
        message: event.error.message,
      };

    case 'refresh_complete':
      return null; // Silent

    case 'refresh_failed':
      return {
        type: 'error',
        title: 'Sync Failed',
        message: 'Could not refresh wallet data',
      };

    default:
      return null;
  }
}

// =============================================================================
// Context Action Button Helpers
// =============================================================================

/**
 * Determine the primary action based on context
 */
export type WalletContext =
  | 'home'
  | 'receive'
  | 'send'
  | 'transaction'
  | 'settings'
  | 'profile';

export interface ContextAction {
  label: string;
  icon: string;
  action: () => void;
}

/**
 * Get the intelligent action for current context
 */
export function getContextAction(
  context: WalletContext,
  handlers: {
    onScan?: () => void;
    onSend?: () => void;
    onReceive?: () => void;
    onShare?: () => void;
    onEdit?: () => void;
  }
): ContextAction | null {
  switch (context) {
    case 'home':
      return handlers.onScan
        ? { label: 'Scan', icon: 'qr-code', action: handlers.onScan }
        : null;

    case 'receive':
      return handlers.onShare
        ? { label: 'Share', icon: 'share', action: handlers.onShare }
        : null;

    case 'send':
      return handlers.onScan
        ? { label: 'Scan', icon: 'qr-code', action: handlers.onScan }
        : null;

    case 'transaction':
      return handlers.onShare
        ? { label: 'Share', icon: 'share', action: handlers.onShare }
        : null;

    case 'settings':
    case 'profile':
      return handlers.onEdit
        ? { label: 'Edit', icon: 'pencil', action: handlers.onEdit }
        : null;

    default:
      return null;
  }
}

// =============================================================================
// Animation Helpers
// =============================================================================

/**
 * Format balance with animated counting effect data
 */
export function createBalanceAnimation(
  from: number,
  to: number,
  duration: number = 500
): {
  values: number[];
  interval: number;
} {
  const steps = Math.min(60, Math.ceil(duration / 16)); // ~60fps max
  const interval = duration / steps;
  const values: number[] = [];

  for (let i = 0; i <= steps; i++) {
    const progress = i / steps;
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    values.push(from + (to - from) * eased);
  }

  return { values, interval };
}

/**
 * Format currency with proper decimals
 */
export function formatCurrency(amount: number, currency: Currency): string {
  const decimals = currency === 'MYC' || currency === 'HOT' ? 2 : 2;
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(amount);
}

/**
 * Format relative time (2 minutes ago, etc)
 */
export function formatRelativeTime(timestamp: number): string {
  const now = Date.now();
  const diff = now - timestamp;

  if (diff < 60000) return 'just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;

  return new Date(timestamp).toLocaleDateString();
}

// =============================================================================
// Wallet Bridge Hooks (Cross-hApp Reputation)
// =============================================================================

/**
 * Return type for useReputation hook
 */
export interface UseReputationReturn {
  /** Aggregate reputation data */
  reputation: ReputationQueryResponse | null;
  /** Whether loading */
  loading: boolean;
  /** Error if any */
  error: Error | null;
  /** Refresh reputation data */
  refresh: () => Promise<void>;
  /** Whether the DID is trustworthy (above threshold) */
  isTrustworthy: boolean;
}

/**
 * Return type for useTransactionRecommendation hook
 */
export interface UseTransactionRecommendationReturn {
  /** Transaction recommendation */
  recommendation: TransactionRecommendation | null;
  /** Whether loading */
  loading: boolean;
  /** Error if any */
  error: Error | null;
  /** Refresh recommendation */
  refresh: () => Promise<void>;
}

/**
 * Return type for useWalletBridge hook
 */
export interface UseWalletBridgeReturn {
  /** Query aggregate reputation */
  getReputation: (did: string, forceRefresh?: boolean) => Promise<ReputationQueryResponse>;
  /** Check if DID is trustworthy */
  isTrustworthy: (did: string, threshold?: number) => Promise<boolean>;
  /** Get transaction recommendation */
  getRecommendation: (did: string, amount: number) => Promise<TransactionRecommendation>;
  /** Report positive interaction */
  reportPositive: (did: string, txId: string) => void;
  /** Report negative interaction */
  reportNegative: (did: string, reason: 'payment_failed' | 'dispute' | 'fraud_suspected') => void;
  /** Subscribe to reputation changes */
  subscribe: (did: string, handler: (rep: ReputationQueryResponse) => void) => () => void;
  /** Request identity verification */
  verifyIdentity: (did: string) => Promise<VerificationStatus>;
  /** Verify credential */
  verifyCredential: (did: string, credentialType: string) => Promise<VerificationStatus>;
}

/**
 * Create reputation hook state manager
 *
 * Use this to build framework-specific hooks (React, Vue, Svelte)
 *
 * @example
 * ```tsx
 * // React usage
 * function useReputation(bridge: WalletBridge, did: string, threshold = 0.6) {
 *   const [state, setState] = useState<UseReputationReturn>({
 *     reputation: null,
 *     loading: true,
 *     error: null,
 *     isTrustworthy: false,
 *     refresh: async () => {},
 *   });
 *
 *   useEffect(() => {
 *     const manager = createReputationHookState(bridge, did, threshold, setState);
 *     manager.initialize();
 *     return manager.cleanup;
 *   }, [bridge, did, threshold]);
 *
 *   return state;
 * }
 * ```
 */
export function createReputationHookState(
  bridge: WalletBridge,
  did: string,
  threshold: number,
  setState: (state: UseReputationReturn) => void
): {
  initialize: () => void;
  cleanup: () => void;
} {
  let unsubscribe: (() => void) | null = null;

  const fetchReputation = async () => {
    setState({
      reputation: null,
      loading: true,
      error: null,
      isTrustworthy: false,
      refresh: fetchReputation,
    });

    try {
      const reputation = await bridge.getAggregateReputation(did);
      const isTrustworthy = reputation.aggregatedScore >= threshold && reputation.confidence >= 0.5;

      setState({
        reputation,
        loading: false,
        error: null,
        isTrustworthy,
        refresh: fetchReputation,
      });
    } catch (e) {
      setState({
        reputation: null,
        loading: false,
        error: e instanceof Error ? e : new Error(String(e)),
        isTrustworthy: false,
        refresh: fetchReputation,
      });
    }
  };

  return {
    initialize: () => {
      void fetchReputation();

      // Subscribe to reputation changes
      unsubscribe = bridge.onReputationChange(did, (rep) => {
        const isTrustworthy = rep.aggregatedScore >= threshold && rep.confidence >= 0.5;
        setState({
          reputation: rep,
          loading: false,
          error: null,
          isTrustworthy,
          refresh: fetchReputation,
        });
      });
    },
    cleanup: () => {
      if (unsubscribe) {
        unsubscribe();
        unsubscribe = null;
      }
    },
  };
}

/**
 * Create transaction recommendation hook state manager
 *
 * @example
 * ```tsx
 * // React usage
 * function useTransactionRecommendation(bridge: WalletBridge, did: string, amount: number) {
 *   const [state, setState] = useState<UseTransactionRecommendationReturn>({
 *     recommendation: null,
 *     loading: true,
 *     error: null,
 *     refresh: async () => {},
 *   });
 *
 *   useEffect(() => {
 *     const manager = createRecommendationHookState(bridge, did, amount, setState);
 *     manager.fetch();
 *   }, [bridge, did, amount]);
 *
 *   return state;
 * }
 * ```
 */
export function createRecommendationHookState(
  bridge: WalletBridge,
  did: string,
  amount: number,
  setState: (state: UseTransactionRecommendationReturn) => void
): {
  fetch: () => Promise<void>;
} {
  const fetchRecommendation = async () => {
    setState({
      recommendation: null,
      loading: true,
      error: null,
      refresh: fetchRecommendation,
    });

    try {
      const recommendation = await bridge.getTransactionRecommendation(did, amount);
      setState({
        recommendation,
        loading: false,
        error: null,
        refresh: fetchRecommendation,
      });
    } catch (e) {
      setState({
        recommendation: null,
        loading: false,
        error: e instanceof Error ? e : new Error(String(e)),
        refresh: fetchRecommendation,
      });
    }
  };

  return { fetch: fetchRecommendation };
}

/**
 * Create wallet bridge adapter for framework hooks
 *
 * Wraps WalletBridge methods for convenient use in hooks
 *
 * @example
 * ```tsx
 * // React context usage
 * const WalletBridgeContext = createContext<UseWalletBridgeReturn | null>(null);
 *
 * function WalletBridgeProvider({ children, bridge }) {
 *   const adapter = useMemo(() => createWalletBridgeAdapter(bridge), [bridge]);
 *   return (
 *     <WalletBridgeContext.Provider value={adapter}>
 *       {children}
 *     </WalletBridgeContext.Provider>
 *   );
 * }
 *
 * function useWalletBridge() {
 *   const ctx = useContext(WalletBridgeContext);
 *   if (!ctx) throw new Error('Must be used within WalletBridgeProvider');
 *   return ctx;
 * }
 * ```
 */
export function createWalletBridgeAdapter(bridge: WalletBridge): UseWalletBridgeReturn {
  return {
    getReputation: (did, forceRefresh) => bridge.getAggregateReputation(did, forceRefresh),
    isTrustworthy: (did, threshold) => bridge.isTrustworthy(did, threshold),
    getRecommendation: (did, amount) => bridge.getTransactionRecommendation(did, amount),
    reportPositive: (did, txId) => bridge.reportPositiveInteraction(did, txId),
    reportNegative: (did, reason) => bridge.reportNegativeInteraction(did, reason),
    subscribe: (did, handler) => bridge.onReputationChange(did, handler),
    verifyIdentity: (did) => bridge.requestIdentityVerification(did),
    verifyCredential: (did, credentialType) => bridge.verifyCredential(did, credentialType),
  };
}

/**
 * Format reputation score for display
 */
export function formatReputationScore(score: number): string {
  return (score * 100).toFixed(0) + '%';
}

/**
 * Get reputation badge info based on score
 */
export function getReputationBadge(score: number): {
  label: string;
  color: 'green' | 'yellow' | 'orange' | 'red' | 'gray';
  icon: string;
} {
  if (score >= 0.9) {
    return { label: 'Excellent', color: 'green', icon: 'star' };
  } else if (score >= 0.7) {
    return { label: 'Good', color: 'green', icon: 'check-circle' };
  } else if (score >= 0.5) {
    return { label: 'Fair', color: 'yellow', icon: 'minus-circle' };
  } else if (score >= 0.3) {
    return { label: 'Low', color: 'orange', icon: 'alert-circle' };
  } else if (score > 0) {
    return { label: 'Poor', color: 'red', icon: 'x-circle' };
  } else {
    return { label: 'Unknown', color: 'gray', icon: 'help-circle' };
  }
}

/**
 * Format risk level for display
 */
export function formatRiskLevel(riskLevel: TransactionRecommendation['riskLevel']): {
  label: string;
  color: 'green' | 'yellow' | 'orange' | 'red';
  description: string;
} {
  switch (riskLevel) {
    case 'low':
      return {
        label: 'Low Risk',
        color: 'green',
        description: 'This transaction appears safe.',
      };
    case 'medium':
      return {
        label: 'Medium Risk',
        color: 'yellow',
        description: 'Proceed with normal caution.',
      };
    case 'high':
      return {
        label: 'High Risk',
        color: 'orange',
        description: 'Consider verifying identity first.',
      };
    case 'very_high':
      return {
        label: 'Very High Risk',
        color: 'red',
        description: 'Verification strongly recommended.',
      };
  }
}
