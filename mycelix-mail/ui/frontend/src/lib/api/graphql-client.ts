// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * GraphQL Client Integration
 *
 * Unified GraphQL client with:
 * - Query/mutation/subscription support
 * - Automatic caching with normalization
 * - Optimistic updates
 * - Real-time subscriptions via WebSocket
 * - Offline queue with sync
 * - Request deduplication
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// ============================================================================
// Types
// ============================================================================

export interface GraphQLError {
  message: string;
  locations?: { line: number; column: number }[];
  path?: (string | number)[];
  extensions?: Record<string, unknown>;
}

export interface GraphQLResponse<T = unknown> {
  data?: T;
  errors?: GraphQLError[];
}

export interface QueryOptions {
  variables?: Record<string, unknown>;
  fetchPolicy?: 'cache-first' | 'network-only' | 'cache-only' | 'no-cache';
  pollInterval?: number;
  context?: Record<string, unknown>;
}

export interface MutationOptions<T = unknown> {
  variables?: Record<string, unknown>;
  optimisticResponse?: T;
  update?: (cache: CacheStore, result: GraphQLResponse<T>) => void;
  refetchQueries?: string[];
}

export interface SubscriptionOptions {
  variables?: Record<string, unknown>;
  onData?: (data: unknown) => void;
  onError?: (error: Error) => void;
  onComplete?: () => void;
}

interface CacheEntry {
  data: unknown;
  timestamp: number;
  expiresAt: number;
}

interface CacheStore {
  entries: Map<string, CacheEntry>;
  get: (key: string) => unknown | undefined;
  set: (key: string, data: unknown, ttl?: number) => void;
  invalidate: (pattern: string | RegExp) => void;
  clear: () => void;
}

interface PendingOperation {
  id: string;
  type: 'query' | 'mutation';
  query: string;
  variables?: Record<string, unknown>;
  timestamp: number;
  retries: number;
}

interface GraphQLClientState {
  isConnected: boolean;
  isAuthenticated: boolean;
  pendingOperations: PendingOperation[];
  subscriptions: Map<string, WebSocket>;
  cache: Map<string, CacheEntry>;
}

// ============================================================================
// Configuration
// ============================================================================

const DEFAULT_CONFIG = {
  httpEndpoint: import.meta.env.VITE_GRAPHQL_HTTP_URL || 'http://localhost:3001/graphql',
  wsEndpoint: import.meta.env.VITE_GRAPHQL_WS_URL || 'ws://localhost:3001/graphql',
  defaultCacheTTL: 5 * 60 * 1000, // 5 minutes
  maxRetries: 3,
  retryDelay: 1000,
  requestTimeout: 30000,
};

// ============================================================================
// Cache Implementation
// ============================================================================

const createCache = (): CacheStore => {
  const entries = new Map<string, CacheEntry>();

  return {
    entries,
    get: (key: string) => {
      const entry = entries.get(key);
      if (!entry) return undefined;
      if (Date.now() > entry.expiresAt) {
        entries.delete(key);
        return undefined;
      }
      return entry.data;
    },
    set: (key: string, data: unknown, ttl = DEFAULT_CONFIG.defaultCacheTTL) => {
      entries.set(key, {
        data,
        timestamp: Date.now(),
        expiresAt: Date.now() + ttl,
      });
    },
    invalidate: (pattern: string | RegExp) => {
      const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;
      for (const key of entries.keys()) {
        if (regex.test(key)) {
          entries.delete(key);
        }
      }
    },
    clear: () => entries.clear(),
  };
};

// ============================================================================
// GraphQL Client Store
// ============================================================================

interface GraphQLStore extends GraphQLClientState {
  // Auth
  setAuthToken: (token: string | null) => void;
  getAuthToken: () => string | null;

  // Operations
  query: <T>(query: string, options?: QueryOptions) => Promise<GraphQLResponse<T>>;
  mutate: <T>(mutation: string, options?: MutationOptions<T>) => Promise<GraphQLResponse<T>>;
  subscribe: <T>(subscription: string, options?: SubscriptionOptions) => () => void;

  // Cache
  cacheGet: (key: string) => unknown | undefined;
  cacheSet: (key: string, data: unknown, ttl?: number) => void;
  cacheInvalidate: (pattern: string | RegExp) => void;
  cacheClear: () => void;

  // Offline
  addPendingOperation: (op: Omit<PendingOperation, 'id' | 'timestamp' | 'retries'>) => void;
  processPendingOperations: () => Promise<void>;

  // Connection
  connect: () => void;
  disconnect: () => void;
}

let authToken: string | null = null;
const cache = createCache();

export const useGraphQLClient = create<GraphQLStore>()(
  persist(
    (set, get) => ({
      isConnected: false,
      isAuthenticated: false,
      pendingOperations: [],
      subscriptions: new Map(),
      cache: new Map(),

      // Auth
      setAuthToken: (token: string | null) => {
        authToken = token;
        set({ isAuthenticated: !!token });
      },

      getAuthToken: () => authToken,

      // Query
      query: async <T>(query: string, options: QueryOptions = {}): Promise<GraphQLResponse<T>> => {
        const { variables, fetchPolicy = 'cache-first' } = options;
        const cacheKey = JSON.stringify({ query, variables });

        // Check cache first
        if (fetchPolicy === 'cache-first' || fetchPolicy === 'cache-only') {
          const cached = cache.get(cacheKey);
          if (cached) {
            return { data: cached as T };
          }
          if (fetchPolicy === 'cache-only') {
            return { errors: [{ message: 'No cached data available' }] };
          }
        }

        // Network request
        try {
          const response = await fetchGraphQL<T>(query, variables);

          // Cache successful responses
          if (response.data && fetchPolicy !== 'no-cache') {
            cache.set(cacheKey, response.data);
          }

          return response;
        } catch (error) {
          // If offline, queue the operation
          if (!navigator.onLine) {
            get().addPendingOperation({ type: 'query', query, variables });
            const cached = cache.get(cacheKey);
            if (cached) {
              return { data: cached as T };
            }
          }
          throw error;
        }
      },

      // Mutation
      mutate: async <T>(mutation: string, options: MutationOptions<T> = {}): Promise<GraphQLResponse<T>> => {
        const { variables, optimisticResponse, update, refetchQueries } = options;

        // Apply optimistic response
        if (optimisticResponse && update) {
          update(cache, { data: optimisticResponse });
        }

        try {
          const response = await fetchGraphQL<T>(mutation, variables);

          // Update cache
          if (response.data && update) {
            update(cache, response);
          }

          // Refetch queries
          if (refetchQueries) {
            for (const queryName of refetchQueries) {
              cache.invalidate(new RegExp(queryName));
            }
          }

          return response;
        } catch (error) {
          // Rollback optimistic update on error
          if (optimisticResponse) {
            cache.invalidate('.*');
          }

          // Queue for retry if offline
          if (!navigator.onLine) {
            get().addPendingOperation({ type: 'mutation', query: mutation, variables });
          }

          throw error;
        }
      },

      // Subscription
      subscribe: <T>(subscription: string, options: SubscriptionOptions = {}): (() => void) => {
        const { variables, onData, onError, onComplete } = options;
        const subscriptionId = Math.random().toString(36).slice(2);

        const ws = new WebSocket(DEFAULT_CONFIG.wsEndpoint, 'graphql-ws');

        ws.onopen = () => {
          // Connection init
          ws.send(JSON.stringify({
            type: 'connection_init',
            payload: { authToken },
          }));

          // Start subscription
          ws.send(JSON.stringify({
            id: subscriptionId,
            type: 'start',
            payload: { query: subscription, variables },
          }));
        };

        ws.onmessage = (event) => {
          const message = JSON.parse(event.data);

          switch (message.type) {
            case 'data':
              if (message.id === subscriptionId && onData) {
                onData(message.payload.data as T);
              }
              break;
            case 'error':
              if (onError) {
                onError(new Error(message.payload?.message || 'Subscription error'));
              }
              break;
            case 'complete':
              if (onComplete) {
                onComplete();
              }
              break;
          }
        };

        ws.onerror = (error) => {
          if (onError) {
            onError(error as unknown as Error);
          }
        };

        // Store subscription
        const { subscriptions } = get();
        subscriptions.set(subscriptionId, ws);
        set({ subscriptions: new Map(subscriptions) });

        // Return unsubscribe function
        return () => {
          ws.send(JSON.stringify({ id: subscriptionId, type: 'stop' }));
          ws.close();
          subscriptions.delete(subscriptionId);
          set({ subscriptions: new Map(subscriptions) });
        };
      },

      // Cache methods
      cacheGet: (key: string) => cache.get(key),
      cacheSet: (key: string, data: unknown, ttl?: number) => cache.set(key, data, ttl),
      cacheInvalidate: (pattern: string | RegExp) => cache.invalidate(pattern),
      cacheClear: () => cache.clear(),

      // Offline queue
      addPendingOperation: (op) => {
        const { pendingOperations } = get();
        set({
          pendingOperations: [
            ...pendingOperations,
            {
              ...op,
              id: Math.random().toString(36).slice(2),
              timestamp: Date.now(),
              retries: 0,
            },
          ],
        });
      },

      processPendingOperations: async () => {
        const { pendingOperations, query, mutate } = get();
        const remaining: PendingOperation[] = [];

        for (const op of pendingOperations) {
          try {
            if (op.type === 'query') {
              await query(op.query, { variables: op.variables, fetchPolicy: 'network-only' });
            } else {
              await mutate(op.query, { variables: op.variables });
            }
          } catch {
            if (op.retries < DEFAULT_CONFIG.maxRetries) {
              remaining.push({ ...op, retries: op.retries + 1 });
            }
          }
        }

        set({ pendingOperations: remaining });
      },

      // Connection management
      connect: () => {
        set({ isConnected: true });

        // Process pending operations when back online
        window.addEventListener('online', () => {
          get().processPendingOperations();
        });
      },

      disconnect: () => {
        const { subscriptions } = get();
        for (const ws of subscriptions.values()) {
          ws.close();
        }
        set({ isConnected: false, subscriptions: new Map() });
      },
    }),
    {
      name: 'mycelix-graphql-client',
      partialize: (state) => ({
        pendingOperations: state.pendingOperations,
      }),
    }
  )
);

// ============================================================================
// Fetch Helper
// ============================================================================

async function fetchGraphQL<T>(
  query: string,
  variables?: Record<string, unknown>
): Promise<GraphQLResponse<T>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), DEFAULT_CONFIG.requestTimeout);

  try {
    const response = await fetch(DEFAULT_CONFIG.httpEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(authToken && { Authorization: `Bearer ${authToken}` }),
      },
      body: JSON.stringify({ query, variables }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

// ============================================================================
// Query Hooks
// ============================================================================

export function useQuery<T>(query: string, options: QueryOptions = {}) {
  const { query: executeQuery, cacheGet } = useGraphQLClient();
  const [data, setData] = useState<T | undefined>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<GraphQLError[] | undefined>();

  const cacheKey = JSON.stringify({ query, variables: options.variables });

  useEffect(() => {
    let mounted = true;
    let pollTimer: NodeJS.Timeout;

    const fetchData = async () => {
      setLoading(true);
      try {
        const result = await executeQuery<T>(query, options);
        if (mounted) {
          setData(result.data);
          setError(result.errors);
        }
      } catch (e) {
        if (mounted) {
          setError([{ message: (e as Error).message }]);
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    fetchData();

    if (options.pollInterval) {
      pollTimer = setInterval(fetchData, options.pollInterval);
    }

    return () => {
      mounted = false;
      if (pollTimer) clearInterval(pollTimer);
    };
  }, [cacheKey]);

  const refetch = async () => {
    setLoading(true);
    const result = await executeQuery<T>(query, { ...options, fetchPolicy: 'network-only' });
    setData(result.data);
    setError(result.errors);
    setLoading(false);
    return result;
  };

  return { data, loading, error, refetch };
}

export function useMutation<T, V = Record<string, unknown>>(mutation: string) {
  const { mutate } = useGraphQLClient();
  const [data, setData] = useState<T | undefined>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<GraphQLError[] | undefined>();

  const execute = async (options: MutationOptions<T> & { variables?: V } = {}) => {
    setLoading(true);
    setError(undefined);
    try {
      const result = await mutate<T>(mutation, options);
      setData(result.data);
      setError(result.errors);
      return result;
    } catch (e) {
      const err = [{ message: (e as Error).message }];
      setError(err);
      return { errors: err };
    } finally {
      setLoading(false);
    }
  };

  return { execute, data, loading, error };
}

export function useSubscription<T>(subscription: string, options: SubscriptionOptions = {}) {
  const { subscribe } = useGraphQLClient();
  const [data, setData] = useState<T | undefined>();
  const [error, setError] = useState<Error | undefined>();

  useEffect(() => {
    const unsubscribe = subscribe<T>(subscription, {
      ...options,
      onData: (newData) => {
        setData(newData as T);
        options.onData?.(newData);
      },
      onError: (err) => {
        setError(err);
        options.onError?.(err);
      },
    });

    return unsubscribe;
  }, [subscription, JSON.stringify(options.variables)]);

  return { data, error };
}

// ============================================================================
// React imports (for hooks)
// ============================================================================

import { useState, useEffect } from 'react';

// ============================================================================
// Pre-built Queries
// ============================================================================

export const QUERIES = {
  GET_EMAILS: `
    query GetEmails($folder: String!, $limit: Int, $cursor: String) {
      emails(folder: $folder, first: $limit, after: $cursor) {
        edges {
          node {
            id
            threadId
            from { email name }
            to { email name }
            subject
            preview
            isRead
            isStarred
            hasAttachments
            receivedAt
            trustScore
            labels
          }
          cursor
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  `,

  GET_EMAIL: `
    query GetEmail($id: ID!) {
      email(id: $id) {
        id
        threadId
        from { email name }
        to { email name }
        cc { email name }
        subject
        bodyHtml
        bodyText
        isRead
        isStarred
        receivedAt
        trustScore
        attachments {
          id
          filename
          contentType
          size
        }
        thread {
          id
          emails { id subject from { email name } receivedAt }
        }
      }
    }
  `,

  GET_TRUST_SCORE: `
    query GetTrustScore($email: String!) {
      trustScore(email: $email) {
        score
        attestations
        pathLength
        breakdown {
          direct
          network
          behavior
        }
      }
    }
  `,

  GET_CONTACTS: `
    query GetContacts($search: String, $limit: Int) {
      contacts(search: $search, first: $limit) {
        edges {
          node {
            id
            email
            name
            avatarUrl
            trustScore
            lastInteraction
          }
        }
      }
    }
  `,
};

export const MUTATIONS = {
  CREATE_EMAIL: `
    mutation CreateEmail($input: CreateEmailInput!) {
      createEmail(input: $input) {
        id
        subject
        bodyText
        bodyHtml
        from { email name }
        to { email name }
        isRead
        isStarred
        sentAt
      }
    }
  `,

  SEND_EMAIL: `
    mutation SendEmail($id: ID!) {
      sendEmail(id: $id) {
        id
        threadId
        sentAt
      }
    }
  `,

  UPDATE_EMAIL: `
    mutation UpdateEmail($id: ID!, $input: UpdateEmailInput!) {
      updateEmail(id: $id, input: $input) {
        id
        isRead
        isStarred
        isArchived
        labels
      }
    }
  `,

  DELETE_EMAIL: `
    mutation DeleteEmail($id: ID!) {
      deleteEmail(id: $id)
    }
  `,

  MARK_AS_READ: `
    mutation MarkAsRead($ids: [ID!]!) {
      markAsRead(ids: $ids) {
        success
        count
      }
    }
  `,

  ARCHIVE_EMAILS: `
    mutation ArchiveEmails($ids: [ID!]!) {
      archiveEmails(ids: $ids) {
        success
        count
      }
    }
  `,

  CREATE_ATTESTATION: `
    mutation CreateAttestation($input: AttestationInput!) {
      createAttestation(input: $input) {
        id
        type
        score
        createdAt
      }
    }
  `,
};

export const SUBSCRIPTIONS = {
  NEW_EMAIL: `
    subscription OnNewEmail {
      newEmail {
        id
        from { email name }
        subject
        preview
        receivedAt
        trustScore
      }
    }
  `,

  EMAIL_UPDATED: `
    subscription OnEmailUpdated($id: ID!) {
      emailUpdated(id: $id) {
        id
        isRead
        isStarred
        labels
      }
    }
  `,

  TRUST_UPDATED: `
    subscription OnTrustUpdated($email: String!) {
      trustUpdated(email: $email) {
        email
        score
        attestations
      }
    }
  `,
};

export default useGraphQLClient;

// ============================================================================
// Compatibility Aliases (for direct client access without React hooks)
// ============================================================================

// Create a client object that can be used outside React components
export const graphqlClient = {
  query: <T>(query: string, variables?: Record<string, unknown>) =>
    useGraphQLClient.getState().query<T>(query, { variables }).then(r => r.data as T),
  mutate: <T>(mutation: string, variables?: Record<string, unknown>) =>
    useGraphQLClient.getState().mutate<T>(mutation, { variables }).then(r => r.data as T),
};

// Lowercase aliases for query/mutation definitions
export const queries = QUERIES;
export const mutations = MUTATIONS;
