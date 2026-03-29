// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React Hooks for Mycelix Mail Holochain
 *
 * Provides reactive state management for all Holochain operations:
 * - useInbox, useSent, useDrafts
 * - useTrustScore, useAttestations
 * - useSyncState, useOnlineStatus
 * - useSignals for real-time updates
 */

import {
  useState,
  useEffect,
  useCallback,
  useContext,
  createContext,
  useMemo,
  useRef,
} from 'react';
import type { ReactNode } from 'react';
import type { AgentPubKey, ActionHash } from '@holochain/client';
import type { MycelixMailClient } from '../index';
import type {
  DecryptedEmail,
  TrustScore,
  TrustCategory,
  SyncState,
  MycelixSignal,
  MessageSignal,
  TrustSignal,
  SyncSignal,
  SendEmailInput,
  CreateAttestationInput,
} from '../types';

// ==================== CONTEXT ====================

interface MycelixContextValue {
  client: MycelixMailClient | null;
  isConnected: boolean;
  isInitialized: boolean;
  myAgentPubKey: AgentPubKey | null;
  error: Error | null;
}

const MycelixContext = createContext<MycelixContextValue>({
  client: null,
  isConnected: false,
  isInitialized: false,
  myAgentPubKey: null,
  error: null,
});

export interface MycelixProviderProps {
  client: MycelixMailClient;
  children: ReactNode;
}

export function MycelixProvider({ client, children }: MycelixProviderProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [myAgentPubKey, setMyAgentPubKey] = useState<AgentPubKey | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const init = async () => {
      try {
        await client.initialize();
        setMyAgentPubKey(client.myAgentPubKey);
        setIsConnected(true);
        setIsInitialized(true);
      } catch (err) {
        setError(err as Error);
      }
    };

    init();

    // Handle online/offline
    const handleOnline = () => {
      setIsConnected(true);
      client.goOnline();
    };
    const handleOffline = () => {
      setIsConnected(false);
      client.goOffline();
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [client]);

  const value = useMemo(
    () => ({ client, isConnected, isInitialized, myAgentPubKey, error }),
    [client, isConnected, isInitialized, myAgentPubKey, error]
  );

  return (
    <MycelixContext.Provider value={value}>
      {children}
    </MycelixContext.Provider>
  );
}

export function useMycelix() {
  return useContext(MycelixContext);
}

// ==================== EMAIL HOOKS ====================

interface UseEmailsOptions {
  limit?: number;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

interface UseEmailsResult<T> {
  data: T[];
  isLoading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
  hasMore: boolean;
  loadMore: () => Promise<void>;
}

function createEmailHook<T>(
  fetchFn: (client: MycelixMailClient, limit: number) => Promise<T[]>
) {
  return function useEmails(options: UseEmailsOptions = {}): UseEmailsResult<T> {
    const { client, isInitialized } = useMycelix();
    const [data, setData] = useState<T[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<Error | null>(null);
    const [hasMore, setHasMore] = useState(true);

    const limit = options.limit ?? 50;

    const refresh = useCallback(async () => {
      if (!client || !isInitialized) return;

      setIsLoading(true);
      setError(null);

      try {
        const result = await fetchFn(client, limit);
        setData(result);
        setHasMore(result.length >= limit);
      } catch (err) {
        setError(err as Error);
      } finally {
        setIsLoading(false);
      }
    }, [client, isInitialized, limit]);

    const loadMore = useCallback(async () => {
      if (!client || !hasMore || isLoading) return;

      try {
        const more = await fetchFn(client, limit);
        setData((prev) => [...prev, ...more]);
        setHasMore(more.length >= limit);
      } catch (err) {
        setError(err as Error);
      }
    }, [client, hasMore, isLoading, limit]);

    useEffect(() => {
      refresh();
    }, [refresh]);

    // Auto-refresh
    useEffect(() => {
      if (!options.autoRefresh) return;

      const interval = setInterval(refresh, options.refreshInterval ?? 30000);
      return () => clearInterval(interval);
    }, [options.autoRefresh, options.refreshInterval, refresh]);

    return { data, isLoading, error, refresh, hasMore, loadMore };
  };
}

export const useInbox = createEmailHook(
  (client, limit) => client.messages.getInbox(limit)
);

export const useSent = createEmailHook(
  (client, limit) => client.messages.getSent(limit)
);

export const useDrafts = createEmailHook(
  (client, _limit) => client.messages.getDrafts()
);

/**
 * Hook for a single email
 */
export function useEmail(emailHash: ActionHash | null) {
  const { client, isInitialized } = useMycelix();
  const [email, setEmail] = useState<DecryptedEmail | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!client || !isInitialized || !emailHash) {
      setIsLoading(false);
      return;
    }

    const fetch = async () => {
      setIsLoading(true);
      try {
        const result = await client.messages.getEmail(emailHash);
        setEmail(result);
      } catch (err) {
        setError(err as Error);
      } finally {
        setIsLoading(false);
      }
    };

    fetch();
  }, [client, isInitialized, emailHash]);

  return { email, isLoading, error };
}

/**
 * Hook for sending emails
 */
export function useSendEmail() {
  const { client, isInitialized } = useMycelix();
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const send = useCallback(
    async (input: SendEmailInput) => {
      if (!client || !isInitialized) {
        throw new Error('Client not initialized');
      }

      setIsSending(true);
      setError(null);

      try {
        const result = await client.messages.sendEmail(input);
        return result;
      } catch (err) {
        setError(err as Error);
        throw err;
      } finally {
        setIsSending(false);
      }
    },
    [client, isInitialized]
  );

  const sendWithTrustCheck = useCallback(
    async (input: SendEmailInput & { requireTrustLevel?: number }) => {
      if (!client || !isInitialized) {
        throw new Error('Client not initialized');
      }

      setIsSending(true);
      setError(null);

      try {
        const result = await client.sendEmailWithTrustCheck(
          input.recipient,
          input.subject,
          input.body,
          {
            requireTrustLevel: input.requireTrustLevel,
            attachments: input.attachments,
            priority: input.priority,
          }
        );
        return result;
      } catch (err) {
        setError(err as Error);
        throw err;
      } finally {
        setIsSending(false);
      }
    },
    [client, isInitialized]
  );

  return { send, sendWithTrustCheck, isSending, error };
}

// ==================== TRUST HOOKS ====================

/**
 * Hook for trust score
 */
export function useTrustScore(
  subject: AgentPubKey | null,
  category?: TrustCategory
) {
  const { client, isInitialized } = useMycelix();
  const [score, setScore] = useState<TrustScore | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const refresh = useCallback(async () => {
    if (!client || !isInitialized || !subject) {
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    try {
      const result = await client.trust.getTrustScore({
        subject,
        category,
        include_transitive: true,
      });
      setScore(result);
    } catch (err) {
      setError(err as Error);
    } finally {
      setIsLoading(false);
    }
  }, [client, isInitialized, subject, category]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { score, isLoading, error, refresh };
}

/**
 * Hook for creating attestations
 */
export function useCreateAttestation() {
  const { client, isInitialized } = useMycelix();
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const create = useCallback(
    async (input: CreateAttestationInput) => {
      if (!client || !isInitialized) {
        throw new Error('Client not initialized');
      }

      setIsCreating(true);
      setError(null);

      try {
        const hash = await client.trust.createAttestation(input);
        return hash;
      } catch (err) {
        setError(err as Error);
        throw err;
      } finally {
        setIsCreating(false);
      }
    },
    [client, isInitialized]
  );

  return { create, isCreating, error };
}

/**
 * Hook for my attestations
 */
export function useMyAttestations() {
  const { client, isInitialized } = useMycelix();
  const [attestations, setAttestations] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const refresh = useCallback(async () => {
    if (!client || !isInitialized) return;

    setIsLoading(true);
    try {
      const result = await client.trust.getMyAttestations();
      setAttestations(result);
    } catch (err) {
      setError(err as Error);
    } finally {
      setIsLoading(false);
    }
  }, [client, isInitialized]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { attestations, isLoading, error, refresh };
}

// ==================== SYNC HOOKS ====================

/**
 * Hook for sync state
 */
export function useSyncState() {
  const { client, isInitialized } = useMycelix();
  const [syncState, setSyncState] = useState<SyncState | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!client || !isInitialized) return;

    const fetch = async () => {
      const state = await client.sync.getSyncState();
      setSyncState(state);
      setIsLoading(false);
    };

    fetch();

    // Subscribe to sync signals
    const unsubscribe = client.signals.onSync((signal) => {
      if (signal.type === 'SyncStateChanged') {
        fetch();
      }
    });

    return unsubscribe;
  }, [client, isInitialized]);

  return { syncState, isLoading };
}

/**
 * Hook for online status
 */
export function useOnlineStatus() {
  const { client, isConnected } = useMycelix();
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [pendingOps, setPendingOps] = useState(0);

  useEffect(() => {
    if (!client) return;

    const unsubscribe = client.signals.onSync((signal) => {
      if (signal.type === 'Online') {
        setIsOnline(true);
      } else if (signal.type === 'Offline') {
        setIsOnline(false);
      } else if (signal.type === 'SyncStateChanged') {
        setPendingOps(signal.pending_ops);
      }
    });

    return unsubscribe;
  }, [client]);

  return { isOnline: isOnline && isConnected, pendingOps };
}

/**
 * Hook for manual sync
 */
export function useSync() {
  const { client, isInitialized } = useMycelix();
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const sync = useCallback(async () => {
    if (!client || !isInitialized) return;

    setIsSyncing(true);
    setError(null);

    try {
      await client.syncWithAllPeers();
      setLastSync(new Date());
    } catch (err) {
      setError(err as Error);
    } finally {
      setIsSyncing(false);
    }
  }, [client, isInitialized]);

  return { sync, isSyncing, lastSync, error };
}

// ==================== SIGNAL HOOKS ====================

/**
 * Hook for subscribing to signals
 */
export function useSignal<T extends MycelixSignal['type']>(
  type: T,
  handler: (signal: Extract<MycelixSignal, { type: T }>) => void
) {
  const { client } = useMycelix();
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    if (!client) return;

    const unsubscribe = client.signals.on(type, (signal) => {
      handlerRef.current(signal as Extract<MycelixSignal, { type: T }>);
    });

    return unsubscribe;
  }, [client, type]);
}

/**
 * Hook for message signals with state
 */
export function useMessageSignals() {
  const { client } = useMycelix();
  const [recentSignals, setRecentSignals] = useState<MessageSignal[]>([]);

  useEffect(() => {
    if (!client) return;

    const unsubscribe = client.signals.onMessage((signal) => {
      setRecentSignals((prev) => [signal, ...prev.slice(0, 49)]);
    });

    return unsubscribe;
  }, [client]);

  const clearSignals = useCallback(() => {
    setRecentSignals([]);
  }, []);

  return { recentSignals, clearSignals };
}

/**
 * Hook for new email notifications
 */
export function useNewEmailNotification(
  onNewEmail: (email: { hash: ActionHash; sender: AgentPubKey; preview: string }) => void
) {
  useSignal('EmailReceived', (signal) => {
    onNewEmail({
      hash: signal.email_hash,
      sender: signal.sender,
      preview: signal.preview,
    });
  });
}

// ==================== UTILITY HOOKS ====================

/**
 * Hook for mailbox status
 */
export function useMailboxStatus() {
  const { client, isInitialized } = useMycelix();
  const [status, setStatus] = useState<{
    inbox_count: number;
    unread_count: number;
    sent_count: number;
    draft_count: number;
    sync_status: { is_online: boolean; pending_ops: number };
  } | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const refresh = useCallback(async () => {
    if (!client || !isInitialized) return;

    try {
      const result = await client.getMailboxStatus();
      setStatus(result);
    } catch (err) {
      console.error('Failed to get mailbox status:', err);
    } finally {
      setIsLoading(false);
    }
  }, [client, isInitialized]);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 60000);
    return () => clearInterval(interval);
  }, [refresh]);

  return { status, isLoading, refresh };
}

/**
 * Hook for search
 */
export function useSearch() {
  const { client, isInitialized } = useMycelix();
  const [results, setResults] = useState<DecryptedEmail[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const search = useCallback(
    async (query: string, options?: Parameters<typeof client.messages.searchEmails>[1]) => {
      if (!client || !isInitialized || !query.trim()) {
        setResults([]);
        return;
      }

      setIsSearching(true);
      setError(null);

      try {
        const result = await client.messages.searchEmails(query, options);
        setResults(result);
      } catch (err) {
        setError(err as Error);
      } finally {
        setIsSearching(false);
      }
    },
    [client, isInitialized]
  );

  const clear = useCallback(() => {
    setResults([]);
    setError(null);
  }, []);

  return { results, search, clear, isSearching, error };
}

/**
 * Debounced search hook
 */
export function useDebouncedSearch(delay = 300) {
  const { search, results, isSearching, error, clear } = useSearch();
  const [query, setQuery] = useState('');
  const timeoutRef = useRef<NodeJS.Timeout>();

  const debouncedSearch = useCallback(
    (newQuery: string) => {
      setQuery(newQuery);

      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      if (!newQuery.trim()) {
        clear();
        return;
      }

      timeoutRef.current = setTimeout(() => {
        search(newQuery);
      }, delay);
    },
    [search, clear, delay]
  );

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return { query, setQuery: debouncedSearch, results, isSearching, error, clear };
}

// Export all hooks
export {
  useMycelix,
  useInbox,
  useSent,
  useDrafts,
  useEmail,
  useSendEmail,
  useTrustScore,
  useCreateAttestation,
  useMyAttestations,
  useSyncState,
  useOnlineStatus,
  useSync,
  useSignal,
  useMessageSignals,
  useNewEmailNotification,
  useMailboxStatus,
  useSearch,
  useDebouncedSearch,
};
