// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React hooks for Holochain Mail Client
 *
 * Provides React Query-based hooks for the Holochain Mail DNA.
 * Works in both mock mode (development) and real mode (production).
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState, useCallback } from 'react';
import {
  getMailClient,
  type MailMessage,
  type Contact,
  type TrustScore,
  type SendMessageInput,
} from '../api/holochain-mail-client';

// ============================================================================
// Connection Hook
// ============================================================================

export function useHolochainConnection(options?: {
  conductorUrl?: string;
  mockMode?: boolean;
}) {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const connect = useCallback(async () => {
    setIsConnecting(true);
    setError(null);
    try {
      const client = getMailClient(options);
      await client.connect();
      setIsConnected(true);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsConnecting(false);
    }
  }, [options?.conductorUrl, options?.mockMode]);

  const disconnect = useCallback(async () => {
    try {
      const client = getMailClient();
      await client.disconnect();
      setIsConnected(false);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    }
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, []);

  return {
    isConnected,
    isConnecting,
    error,
    connect,
    disconnect,
  };
}

// ============================================================================
// Message Hooks
// ============================================================================

export function useInbox(options?: { minTrust?: number }) {
  return useQuery({
    queryKey: ['holochain', 'inbox', options?.minTrust],
    queryFn: async () => {
      const client = getMailClient();
      return client.getInbox({ minTrust: options?.minTrust });
    },
    staleTime: 30 * 1000,
    refetchOnWindowFocus: true,
  });
}

export function useMessage(id: string) {
  return useQuery({
    queryKey: ['holochain', 'message', id],
    queryFn: async () => {
      const client = getMailClient();
      return client.getMessage(id);
    },
    enabled: !!id,
  });
}

export function useSendMessage() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (input: SendMessageInput) => {
      const client = getMailClient();
      return client.sendMessage(input);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['holochain', 'inbox'] });
    },
  });
}

export function useMarkAsRead() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      const client = getMailClient();
      return client.markAsRead(id);
    },
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['holochain', 'message', id] });
      queryClient.invalidateQueries({ queryKey: ['holochain', 'inbox'] });
    },
  });
}

export function useToggleStar() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      const client = getMailClient();
      return client.toggleStar(id);
    },
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: ['holochain', 'message', id] });
      queryClient.invalidateQueries({ queryKey: ['holochain', 'inbox'] });
    },
  });
}

export function useArchiveMessage() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (id: string) => {
      const client = getMailClient();
      return client.archive(id);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['holochain', 'inbox'] });
    },
  });
}

// ============================================================================
// Trust Hooks
// ============================================================================

export function useTrustScore(did: string) {
  return useQuery({
    queryKey: ['holochain', 'trust', did],
    queryFn: async () => {
      const client = getMailClient();
      return client.getTrustScore(did);
    },
    enabled: !!did,
    staleTime: 60 * 1000,
  });
}

export function useCrossHappReputation(did: string) {
  return useQuery({
    queryKey: ['holochain', 'crossHappReputation', did],
    queryFn: async () => {
      const client = getMailClient();
      return client.getCrossHappReputation(did);
    },
    enabled: !!did,
    staleTime: 60 * 1000,
  });
}

export function useRecordPositiveInteraction() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (did: string) => {
      const client = getMailClient();
      return client.recordPositiveInteraction(did);
    },
    onSuccess: (_, did) => {
      queryClient.invalidateQueries({ queryKey: ['holochain', 'trust', did] });
      queryClient.invalidateQueries({ queryKey: ['holochain', 'contacts'] });
    },
  });
}

export function useReportSpam() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ messageId, reason }: { messageId: string; reason: string }) => {
      const client = getMailClient();
      return client.reportSpam(messageId, reason);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['holochain', 'inbox'] });
      queryClient.invalidateQueries({ queryKey: ['holochain', 'trust'] });
    },
  });
}

// ============================================================================
// Contact Hooks
// ============================================================================

export function useContacts() {
  return useQuery({
    queryKey: ['holochain', 'contacts'],
    queryFn: async () => {
      const client = getMailClient();
      return client.getContacts();
    },
    staleTime: 60 * 1000,
  });
}

export function useAddContact() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (contact: Omit<Contact, 'id' | 'trust_score'>) => {
      const client = getMailClient();
      return client.addContact(contact);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['holochain', 'contacts'] });
    },
  });
}

export function useBlockContact() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (did: string) => {
      const client = getMailClient();
      return client.blockContact(did);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['holochain', 'contacts'] });
    },
  });
}

// ============================================================================
// Combined Hook for Email List Item
// ============================================================================

export function useEmailWithTrust(email: MailMessage) {
  const { data: trustData } = useTrustScore(email.from_did);
  const { data: crossHappData } = useCrossHappReputation(email.from_did);

  return {
    ...email,
    senderTrust: trustData,
    crossHappReputation: crossHappData,
    isByzantine: trustData?.is_byzantine ?? false,
    trustLevel: trustData?.score ?? email.trust_score,
  };
}
