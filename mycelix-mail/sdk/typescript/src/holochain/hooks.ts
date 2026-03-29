// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail React Hooks
 *
 * React hooks for Holochain zome interactions
 */

import { useState, useEffect, useCallback, useContext, createContext, ReactNode } from 'react';
import type { ActionHash, AgentPubKey } from '@holochain/client';
import { MycelixHolochainClient, createHolochainClient, MycelixHolochainConfig } from './client';
import type {
  Email,
  EmailStateUpdate,
  Contact,
  TrustScore,
  CreateAttestationInput,
  SearchQuery,
  SearchResult,
  MailSignal,
} from './types';

// ==================== CONTEXT ====================

interface MycelixContextValue {
  client: MycelixHolochainClient | null;
  isConnected: boolean;
  isConnecting: boolean;
  error: Error | null;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
}

const MycelixContext = createContext<MycelixContextValue | null>(null);

interface MycelixProviderProps {
  config: MycelixHolochainConfig;
  children: ReactNode;
  autoConnect?: boolean;
}

/**
 * Provider component for Mycelix Holochain client
 */
export function MycelixProvider({ config, children, autoConnect = true }: MycelixProviderProps) {
  const [client, setClient] = useState<MycelixHolochainClient | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const connect = useCallback(async () => {
    if (isConnecting || isConnected) return;

    setIsConnecting(true);
    setError(null);

    try {
      const newClient = createHolochainClient(config);
      await newClient.connect();
      setClient(newClient);
      setIsConnected(true);
    } catch (err) {
      setError(err as Error);
    } finally {
      setIsConnecting(false);
    }
  }, [config, isConnecting, isConnected]);

  const disconnect = useCallback(async () => {
    if (client) {
      await client.disconnect();
      setClient(null);
      setIsConnected(false);
    }
  }, [client]);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      if (client) {
        client.disconnect();
      }
    };
  }, []);

  return (
    <MycelixContext.Provider
      value={{ client, isConnected, isConnecting, error, connect, disconnect }}
    >
      {children}
    </MycelixContext.Provider>
  );
}

/**
 * Hook to access the Mycelix client context
 */
export function useMycelix(): MycelixContextValue {
  const context = useContext(MycelixContext);
  if (!context) {
    throw new Error('useMycelix must be used within a MycelixProvider');
  }
  return context;
}

// ==================== EMAIL HOOKS ====================

interface UseInboxResult {
  emails: Email[];
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

/**
 * Hook to fetch inbox emails
 */
export function useInbox(): UseInboxResult {
  const { client, isConnected } = useMycelix();
  const [emails, setEmails] = useState<Email[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    if (!client || !isConnected) return;

    setIsLoading(true);
    setError(null);

    try {
      const inbox = await client.messages.getInbox();
      setEmails(inbox);
    } catch (err) {
      setError(err as Error);
    } finally {
      setIsLoading(false);
    }
  }, [client, isConnected]);

  useEffect(() => {
    refetch();
  }, [refetch]);

  // Listen for new emails
  useEffect(() => {
    if (!client) return;

    const unsubscribe = client.onSignal((signal) => {
      if (signal.type === 'EmailReceived') {
        refetch();
      }
    });

    return unsubscribe;
  }, [client, refetch]);

  return { emails, isLoading, error, refetch };
}

/**
 * Hook to fetch a single email
 */
export function useEmail(hash: ActionHash | null) {
  const { client, isConnected } = useMycelix();
  const [email, setEmail] = useState<Email | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!client || !isConnected || !hash) return;

    const fetchEmail = async () => {
      setIsLoading(true);
      try {
        const result = await client.messages.getEmail(hash);
        setEmail(result);
      } catch (err) {
        setError(err as Error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchEmail();
  }, [client, isConnected, hash]);

  return { email, isLoading, error };
}

/**
 * Hook to send an email
 */
export function useSendEmail() {
  const { client, isConnected } = useMycelix();
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const sendEmail = useCallback(
    async (input: Parameters<typeof client.messages.sendEmail>[0]) => {
      if (!client || !isConnected) {
        throw new Error('Not connected');
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
    [client, isConnected]
  );

  return { sendEmail, isSending, error };
}

/**
 * Hook to update email state (read, starred, archived, etc.)
 */
export function useEmailState(emailHash: ActionHash | null) {
  const { client, isConnected } = useMycelix();
  const [isUpdating, setIsUpdating] = useState(false);

  const updateState = useCallback(
    async (update: EmailStateUpdate) => {
      if (!client || !isConnected || !emailHash) return;

      setIsUpdating(true);
      try {
        await client.messages.updateEmailState(emailHash, update);
      } finally {
        setIsUpdating(false);
      }
    },
    [client, isConnected, emailHash]
  );

  const markAsRead = useCallback(
    (sendReceipt = false) =>
      client?.messages.markAsRead(emailHash!, sendReceipt),
    [client, emailHash]
  );

  return { updateState, markAsRead, isUpdating };
}

// ==================== CONTACTS HOOKS ====================

/**
 * Hook to fetch all contacts
 */
export function useContacts() {
  const { client, isConnected } = useMycelix();
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    if (!client || !isConnected) return;

    setIsLoading(true);
    try {
      const result = await client.contacts.getAllContacts();
      setContacts(result);
    } catch (err) {
      setError(err as Error);
    } finally {
      setIsLoading(false);
    }
  }, [client, isConnected]);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { contacts, isLoading, error, refetch };
}

/**
 * Hook to manage a single contact
 */
export function useContact(hash: ActionHash | null) {
  const { client, isConnected } = useMycelix();
  const [contact, setContact] = useState<Contact | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!client || !isConnected || !hash) return;

    const fetchContact = async () => {
      setIsLoading(true);
      try {
        const result = await client.contacts.getContact(hash);
        setContact(result);
      } finally {
        setIsLoading(false);
      }
    };

    fetchContact();
  }, [client, isConnected, hash]);

  return { contact, isLoading };
}

// ==================== TRUST HOOKS ====================

/**
 * Hook to get trust score for an agent
 */
export function useTrustScore(agent: AgentPubKey | null) {
  const { client, isConnected } = useMycelix();
  const [score, setScore] = useState<TrustScore | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!client || !isConnected || !agent) return;

    const fetchScore = async () => {
      setIsLoading(true);
      try {
        const result = await client.trust.computeTrustScore(agent);
        setScore(result);
      } finally {
        setIsLoading(false);
      }
    };

    fetchScore();
  }, [client, isConnected, agent]);

  return { score, isLoading };
}

/**
 * Hook to create trust attestations
 */
export function useCreateAttestation() {
  const { client, isConnected } = useMycelix();
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const createAttestation = useCallback(
    async (input: CreateAttestationInput) => {
      if (!client || !isConnected) {
        throw new Error('Not connected');
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
    [client, isConnected]
  );

  return { createAttestation, isCreating, error };
}

// ==================== SEARCH HOOKS ====================

/**
 * Hook for searching emails
 */
export function useSearch() {
  const { client, isConnected } = useMycelix();
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const search = useCallback(
    async (query: SearchQuery) => {
      if (!client || !isConnected) return;

      setIsSearching(true);
      setError(null);

      try {
        const searchResults = await client.search.search(query);
        setResults(searchResults);
        return searchResults;
      } catch (err) {
        setError(err as Error);
        return [];
      } finally {
        setIsSearching(false);
      }
    },
    [client, isConnected]
  );

  const clear = useCallback(() => {
    setResults([]);
  }, []);

  return { results, search, clear, isSearching, error };
}

// ==================== SYNC HOOKS ====================

/**
 * Hook to manage sync state
 */
export function useSyncState() {
  const { client, isConnected } = useMycelix();
  const [isOnline, setIsOnline] = useState(true);
  const [pendingOperations, setPendingOperations] = useState(0);

  useEffect(() => {
    if (!client || !isConnected) return;

    const fetchState = async () => {
      const state = await client.sync.getSyncState();
      if (state) {
        setPendingOperations(state.pending_operations);
      }
    };

    fetchState();
    const interval = setInterval(fetchState, 30000);

    return () => clearInterval(interval);
  }, [client, isConnected]);

  const setOnline = useCallback(
    async (online: boolean) => {
      if (!client) return;
      await client.sync.setOnlineStatus(online);
      setIsOnline(online);

      if (online) {
        await client.sync.processPendingOperations();
      }
    },
    [client]
  );

  return { isOnline, pendingOperations, setOnline };
}

// ==================== SIGNAL HOOKS ====================

/**
 * Hook to listen for specific signal types
 */
export function useSignal<T extends MailSignal['type']>(
  type: T,
  handler: (data: unknown) => void
) {
  const { client } = useMycelix();

  useEffect(() => {
    if (!client) return;

    const unsubscribe = client.onSignal((signal) => {
      if (signal.type === type) {
        handler(signal.data);
      }
    });

    return unsubscribe;
  }, [client, type, handler]);
}

export default {
  MycelixProvider,
  useMycelix,
  useInbox,
  useEmail,
  useSendEmail,
  useEmailState,
  useContacts,
  useContact,
  useTrustScore,
  useCreateAttestation,
  useSearch,
  useSyncState,
  useSignal,
};
