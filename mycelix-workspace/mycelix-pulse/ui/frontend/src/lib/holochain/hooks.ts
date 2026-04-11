// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React Hooks for Holochain Integration
 *
 * Provides React hooks for accessing Holochain data and actions
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  getHolochainClient,
  getMycelixClient,
  onConnectionStateChange,
  isConnected,
  ConnectionState,
} from './index';
import type { MycelixMailClient } from '@mycelix/holochain-client';

// ==========================================
// Connection Hook
// ==========================================

export function useHolochainConnection() {
  const [state, setState] = useState<ConnectionState>('disconnected');
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    return onConnectionStateChange((newState, err) => {
      setState(newState);
      setError(err || null);
    });
  }, []);

  return {
    state,
    error,
    isConnected: state === 'connected',
    isConnecting: state === 'connecting',
    isError: state === 'error',
  };
}

// ==========================================
// Client Hook
// ==========================================

export function useHolochainClient(): MycelixMailClient | null {
  const { isConnected } = useHolochainConnection();

  return useMemo(() => {
    if (!isConnected) return null;
    try {
      return getHolochainClient();
    } catch {
      return null;
    }
  }, [isConnected]);
}

// ==========================================
// Inbox Hook
// ==========================================

export interface Email {
  hash: string;
  sender: string;
  recipient: string;
  subject: string;
  body: string;
  timestamp: number;
  isRead: boolean;
  isStarred: boolean;
  trustLevel: number;
  attachments: any[];
  labels: string[];
}

export function useInbox(limit?: number) {
  const client = useHolochainClient();
  const [emails, setEmails] = useState<Email[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchInbox = useCallback(async () => {
    if (!client) return;

    setIsLoading(true);
    setError(null);

    try {
      const inboxWithTrust = await client.getInboxWithTrust(limit);

      const mapped: Email[] = inboxWithTrust.map((item) => ({
        hash: item.email.hash?.toString() || '',
        sender: item.email.sender?.toString() || '',
        recipient: item.email.recipient?.toString() || '',
        subject: item.email.subject || '',
        body: item.email.body || '',
        timestamp: Number(item.email.timestamp) / 1000,
        isRead: item.email.is_read || false,
        isStarred: item.email.is_starred || false,
        trustLevel: item.sender_trust,
        attachments: item.email.attachments || [],
        labels: item.email.labels || [],
      }));

      setEmails(mapped);
    } catch (err) {
      setError(err as Error);
    } finally {
      setIsLoading(false);
    }
  }, [client, limit]);

  useEffect(() => {
    fetchInbox();
  }, [fetchInbox]);

  // Listen for new emails
  useEffect(() => {
    const handleSignal = (event: CustomEvent) => {
      if (event.detail.type === 'EmailReceived') {
        fetchInbox();
      }
    };

    window.addEventListener('holochain-signal', handleSignal as EventListener);
    return () => window.removeEventListener('holochain-signal', handleSignal as EventListener);
  }, [fetchInbox]);

  return {
    emails,
    isLoading,
    error,
    refresh: fetchInbox,
  };
}

// ==========================================
// Send Email Hook
// ==========================================

export interface SendEmailInput {
  to: string;
  subject: string;
  body: string;
  attachments?: any[];
  priority?: 'Low' | 'Normal' | 'High' | 'Urgent';
}

export function useSendEmail() {
  const client = useHolochainClient();
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const sendEmail = useCallback(
    async (input: SendEmailInput) => {
      if (!client) {
        throw new Error('Not connected to Holochain');
      }

      setIsSending(true);
      setError(null);

      try {
        const result = await client.sendEmailWithTrustCheck(
          input.to as any,
          input.subject,
          input.body,
          {
            attachments: input.attachments,
            priority: input.priority,
          }
        );

        if (!result.sent) {
          throw new Error(result.reason || 'Failed to send email');
        }

        return result;
      } catch (err) {
        setError(err as Error);
        throw err;
      } finally {
        setIsSending(false);
      }
    },
    [client]
  );

  return {
    sendEmail,
    isSending,
    error,
  };
}

// ==========================================
// Trust Network Hook
// ==========================================

export interface TrustNode {
  id: string;
  name?: string;
  email?: string;
  trustLevel: number;
  attestationCount: number;
  isMe?: boolean;
}

export interface TrustEdge {
  source: string;
  target: string;
  trustLevel: number;
  createdAt: number;
}

export function useTrustNetwork() {
  const client = useHolochainClient();
  const [nodes, setNodes] = useState<TrustNode[]>([]);
  const [edges, setEdges] = useState<TrustEdge[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchNetwork = useCallback(async () => {
    if (!client) return;

    setIsLoading(true);
    setError(null);

    try {
      const [attestations, myPubKey] = await Promise.all([
        client.trust.getMyAttestations(),
        Promise.resolve(client.myAgentPubKey),
      ]);

      // Build nodes from attestations
      const nodeMap = new Map<string, TrustNode>();

      // Add self
      if (myPubKey) {
        nodeMap.set(myPubKey.toString(), {
          id: myPubKey.toString(),
          name: 'You',
          trustLevel: 1,
          attestationCount: 0,
          isMe: true,
        });
      }

      // Add attestation targets
      attestations.forEach((att: any) => {
        const targetId = att.target?.toString() || '';
        if (!nodeMap.has(targetId)) {
          nodeMap.set(targetId, {
            id: targetId,
            trustLevel: att.trust_level || 0,
            attestationCount: 1,
          });
        } else {
          const node = nodeMap.get(targetId)!;
          node.attestationCount++;
        }
      });

      // Build edges
      const edgeList: TrustEdge[] = attestations.map((att: any) => ({
        source: myPubKey?.toString() || '',
        target: att.target?.toString() || '',
        trustLevel: att.trust_level || 0,
        createdAt: Number(att.created_at) || Date.now(),
      }));

      setNodes(Array.from(nodeMap.values()));
      setEdges(edgeList);
    } catch (err) {
      setError(err as Error);
    } finally {
      setIsLoading(false);
    }
  }, [client]);

  useEffect(() => {
    fetchNetwork();
  }, [fetchNetwork]);

  const createAttestation = useCallback(
    async (targetAgent: string, trustLevel: number, context?: string) => {
      if (!client) throw new Error('Not connected');

      await client.trust.createAttestation({
        target: targetAgent as any,
        trust_level: trustLevel,
        context,
        category: 'Communication',
      });

      await fetchNetwork();
    },
    [client, fetchNetwork]
  );

  return {
    nodes,
    edges,
    isLoading,
    error,
    refresh: fetchNetwork,
    createAttestation,
  };
}

// ==========================================
// Contacts Hook
// ==========================================

export interface Contact {
  id: string;
  name: string;
  email: string;
  agentPubKey?: string;
  trustLevel?: number;
  groups: string[];
}

export function useContacts() {
  const mycelixClient = useMemo(() => {
    if (!isConnected()) return null;
    try {
      return getMycelixClient();
    } catch {
      return null;
    }
  }, []);

  const [contacts, setContacts] = useState<Contact[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchContacts = useCallback(async () => {
    if (!mycelixClient) return;

    setIsLoading(true);
    setError(null);

    try {
      const contactsData = await mycelixClient.contacts.getAllContacts();

      const mapped: Contact[] = contactsData.map((c: any) => ({
        id: c.hash?.toString() || c.id || '',
        name: c.name || '',
        email: c.email || '',
        agentPubKey: c.agent_pub_key?.toString(),
        trustLevel: c.trust_level,
        groups: c.groups || [],
      }));

      setContacts(mapped);
    } catch (err) {
      setError(err as Error);
    } finally {
      setIsLoading(false);
    }
  }, [mycelixClient]);

  useEffect(() => {
    fetchContacts();
  }, [fetchContacts]);

  const createContact = useCallback(
    async (contact: Omit<Contact, 'id'>) => {
      if (!mycelixClient) throw new Error('Not connected');

      await mycelixClient.contacts.createContact(contact);
      await fetchContacts();
    },
    [mycelixClient, fetchContacts]
  );

  const updateContact = useCallback(
    async (id: string, updates: Partial<Contact>) => {
      if (!mycelixClient) throw new Error('Not connected');

      await mycelixClient.contacts.updateContact(id as any, updates);
      await fetchContacts();
    },
    [mycelixClient, fetchContacts]
  );

  const deleteContact = useCallback(
    async (id: string) => {
      if (!mycelixClient) throw new Error('Not connected');

      await mycelixClient.contacts.deleteContact(id as any);
      await fetchContacts();
    },
    [mycelixClient, fetchContacts]
  );

  return {
    contacts,
    isLoading,
    error,
    refresh: fetchContacts,
    createContact,
    updateContact,
    deleteContact,
  };
}

// ==========================================
// Mailbox Status Hook
// ==========================================

export interface MailboxStatus {
  inboxCount: number;
  unreadCount: number;
  sentCount: number;
  draftCount: number;
  isOnline: boolean;
  pendingOps: number;
  lastSync: Date | null;
}

export function useMailboxStatus() {
  const client = useHolochainClient();
  const [status, setStatus] = useState<MailboxStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchStatus = useCallback(async () => {
    if (!client) return;

    setIsLoading(true);
    setError(null);

    try {
      const data = await client.getMailboxStatus();

      setStatus({
        inboxCount: data.inbox_count,
        unreadCount: data.unread_count,
        sentCount: data.sent_count,
        draftCount: data.draft_count,
        isOnline: data.sync_status.is_online,
        pendingOps: data.sync_status.pending_ops,
        lastSync: data.sync_status.last_sync ? new Date(data.sync_status.last_sync) : null,
      });
    } catch (err) {
      setError(err as Error);
    } finally {
      setIsLoading(false);
    }
  }, [client]);

  useEffect(() => {
    fetchStatus();

    // Refresh periodically
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  return {
    status,
    isLoading,
    error,
    refresh: fetchStatus,
  };
}

// ==========================================
// Sync Hook
// ==========================================

export function useSync() {
  const client = useHolochainClient();
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const sync = useCallback(async () => {
    if (!client) return;

    setIsSyncing(true);
    setError(null);

    try {
      const result = await client.syncWithAllPeers();
      setLastSync(new Date());
      return result;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setIsSyncing(false);
    }
  }, [client]);

  return {
    sync,
    isSyncing,
    lastSync,
    error,
  };
}

export default {
  useHolochainConnection,
  useHolochainClient,
  useInbox,
  useSendEmail,
  useTrustNetwork,
  useContacts,
  useMailboxStatus,
  useSync,
};
