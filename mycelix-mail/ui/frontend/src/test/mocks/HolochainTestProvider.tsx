// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Holochain Test Provider
 *
 * React context provider for testing components that depend on Holochain hooks
 */

import React, { createContext, useContext, ReactNode, useMemo, useState, useCallback } from 'react';
import type { Email, TrustNode, TrustEdge, Contact, MailboxStatus, SendEmailInput } from '@/lib/holochain/hooks';
import {
  mockEmails,
  mockContacts,
  mockTrustNodes,
  mockTrustEdges,
  mockMailboxStatus,
  type MockConnectionState,
} from './holochain';

// ============================================================================
// Context Types
// ============================================================================

interface HolochainTestContextValue {
  // Connection state
  connectionState: MockConnectionState;
  connectionError: Error | null;
  isConnected: boolean;

  // Inbox
  emails: Email[];
  inboxLoading: boolean;
  inboxError: Error | null;
  refreshInbox: () => Promise<void>;

  // Send email
  sendEmail: (input: SendEmailInput) => Promise<{ sent: boolean; email_hash?: string; reason?: string }>;
  isSending: boolean;
  sendError: Error | null;

  // Trust network
  trustNodes: TrustNode[];
  trustEdges: TrustEdge[];
  trustLoading: boolean;
  trustError: Error | null;
  refreshTrustNetwork: () => Promise<void>;
  createAttestation: (target: string, trustLevel: number, context?: string) => Promise<void>;

  // Contacts
  contacts: Contact[];
  contactsLoading: boolean;
  contactsError: Error | null;
  refreshContacts: () => Promise<void>;
  createContact: (contact: Omit<Contact, 'id'>) => Promise<void>;
  updateContact: (id: string, updates: Partial<Contact>) => Promise<void>;
  deleteContact: (id: string) => Promise<void>;

  // Mailbox status
  mailboxStatus: MailboxStatus | null;
  statusLoading: boolean;
  statusError: Error | null;
  refreshStatus: () => Promise<void>;

  // Sync
  sync: () => Promise<{ synced: number; failed: number }>;
  isSyncing: boolean;
  lastSync: Date | null;
  syncError: Error | null;

  // Test controls
  setEmails: (emails: Email[]) => void;
  setContacts: (contacts: Contact[]) => void;
  setConnectionState: (state: MockConnectionState, error?: Error) => void;
  simulateNewEmail: (email: Email) => void;
  simulateConnectionLost: () => void;
  simulateReconnected: () => void;
}

const HolochainTestContext = createContext<HolochainTestContextValue | null>(null);

// ============================================================================
// Provider Props
// ============================================================================

interface HolochainTestProviderProps {
  children: ReactNode;
  initialEmails?: Email[];
  initialContacts?: Contact[];
  initialTrustNodes?: TrustNode[];
  initialTrustEdges?: TrustEdge[];
  initialMailboxStatus?: MailboxStatus;
  initialConnectionState?: MockConnectionState;
  initialConnectionError?: Error;
  // Behavior overrides
  sendEmailShouldFail?: boolean;
  sendEmailFailReason?: string;
  loadingShouldNeverResolve?: boolean;
}

// ============================================================================
// Provider Component
// ============================================================================

export function HolochainTestProvider({
  children,
  initialEmails = mockEmails,
  initialContacts = mockContacts,
  initialTrustNodes = mockTrustNodes,
  initialTrustEdges = mockTrustEdges,
  initialMailboxStatus = mockMailboxStatus,
  initialConnectionState = 'connected',
  initialConnectionError,
  sendEmailShouldFail = false,
  sendEmailFailReason = 'Send failed',
  loadingShouldNeverResolve = false,
}: HolochainTestProviderProps) {
  // Connection state
  const [connectionState, setConnectionStateInternal] = useState<MockConnectionState>(initialConnectionState);
  const [connectionError, setConnectionError] = useState<Error | null>(initialConnectionError || null);

  // Inbox state
  const [emails, setEmails] = useState<Email[]>(initialEmails);
  const [inboxLoading, setInboxLoading] = useState(false);
  const [inboxError, setInboxError] = useState<Error | null>(null);

  // Send state
  const [isSending, setIsSending] = useState(false);
  const [sendError, setSendError] = useState<Error | null>(null);

  // Trust state
  const [trustNodes, setTrustNodes] = useState<TrustNode[]>(initialTrustNodes);
  const [trustEdges, setTrustEdges] = useState<TrustEdge[]>(initialTrustEdges);
  const [trustLoading, setTrustLoading] = useState(false);
  const [trustError, setTrustError] = useState<Error | null>(null);

  // Contacts state
  const [contacts, setContacts] = useState<Contact[]>(initialContacts);
  const [contactsLoading, setContactsLoading] = useState(false);
  const [contactsError, setContactsError] = useState<Error | null>(null);

  // Mailbox status state
  const [mailboxStatus, setMailboxStatus] = useState<MailboxStatus | null>(initialMailboxStatus);
  const [statusLoading, setStatusLoading] = useState(false);
  const [statusError, setStatusError] = useState<Error | null>(null);

  // Sync state
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const [syncError, setSyncError] = useState<Error | null>(null);

  // Helper for simulating async operations
  const simulateAsync = useCallback(async <T,>(value: T, delay = 50): Promise<T> => {
    if (loadingShouldNeverResolve) {
      return new Promise(() => {}); // Never resolves
    }
    return new Promise((resolve) => setTimeout(() => resolve(value), delay));
  }, [loadingShouldNeverResolve]);

  // Connection state control
  const setConnectionState = useCallback((state: MockConnectionState, error?: Error) => {
    setConnectionStateInternal(state);
    setConnectionError(error || null);
  }, []);

  // Inbox operations
  const refreshInbox = useCallback(async () => {
    if (connectionState !== 'connected') {
      setInboxError(new Error('Not connected'));
      return;
    }
    setInboxLoading(true);
    setInboxError(null);
    await simulateAsync(null);
    setInboxLoading(false);
  }, [connectionState, simulateAsync]);

  // Send email
  const sendEmail = useCallback(async (input: SendEmailInput) => {
    if (connectionState !== 'connected') {
      throw new Error('Not connected to Holochain');
    }

    setIsSending(true);
    setSendError(null);

    await simulateAsync(null);

    if (sendEmailShouldFail) {
      const error = new Error(sendEmailFailReason);
      setSendError(error);
      setIsSending(false);
      throw error;
    }

    const newEmail: Email = {
      hash: `uhCkk${Date.now()}`,
      sender: 'testuser@mycelix.mail',
      recipient: input.to,
      subject: input.subject,
      body: input.body,
      timestamp: Date.now(),
      isRead: true,
      isStarred: false,
      trustLevel: 1,
      attachments: input.attachments || [],
      labels: [],
    };

    setIsSending(false);
    return { sent: true, email_hash: newEmail.hash };
  }, [connectionState, sendEmailShouldFail, sendEmailFailReason, simulateAsync]);

  // Trust network operations
  const refreshTrustNetwork = useCallback(async () => {
    if (connectionState !== 'connected') {
      setTrustError(new Error('Not connected'));
      return;
    }
    setTrustLoading(true);
    setTrustError(null);
    await simulateAsync(null);
    setTrustLoading(false);
  }, [connectionState, simulateAsync]);

  const createAttestation = useCallback(async (target: string, trustLevel: number, context?: string) => {
    if (connectionState !== 'connected') {
      throw new Error('Not connected');
    }
    await simulateAsync(null);

    // Add new edge
    const newEdge: TrustEdge = {
      source: 'uhCAkme123',
      target,
      trustLevel,
      createdAt: Date.now(),
    };
    setTrustEdges((prev) => [...prev, newEdge]);
  }, [connectionState, simulateAsync]);

  // Contacts operations
  const refreshContacts = useCallback(async () => {
    if (connectionState !== 'connected') {
      setContactsError(new Error('Not connected'));
      return;
    }
    setContactsLoading(true);
    setContactsError(null);
    await simulateAsync(null);
    setContactsLoading(false);
  }, [connectionState, simulateAsync]);

  const createContact = useCallback(async (contact: Omit<Contact, 'id'>) => {
    if (connectionState !== 'connected') {
      throw new Error('Not connected');
    }
    await simulateAsync(null);

    const newContact: Contact = {
      ...contact,
      id: `contact-${Date.now()}`,
    };
    setContacts((prev) => [...prev, newContact]);
  }, [connectionState, simulateAsync]);

  const updateContact = useCallback(async (id: string, updates: Partial<Contact>) => {
    if (connectionState !== 'connected') {
      throw new Error('Not connected');
    }
    await simulateAsync(null);

    setContacts((prev) =>
      prev.map((c) => (c.id === id ? { ...c, ...updates } : c))
    );
  }, [connectionState, simulateAsync]);

  const deleteContact = useCallback(async (id: string) => {
    if (connectionState !== 'connected') {
      throw new Error('Not connected');
    }
    await simulateAsync(null);

    setContacts((prev) => prev.filter((c) => c.id !== id));
  }, [connectionState, simulateAsync]);

  // Mailbox status operations
  const refreshStatus = useCallback(async () => {
    if (connectionState !== 'connected') {
      setStatusError(new Error('Not connected'));
      return;
    }
    setStatusLoading(true);
    setStatusError(null);
    await simulateAsync(null);
    setStatusLoading(false);
  }, [connectionState, simulateAsync]);

  // Sync operations
  const sync = useCallback(async () => {
    if (connectionState !== 'connected') {
      throw new Error('Not connected');
    }
    setIsSyncing(true);
    setSyncError(null);

    await simulateAsync(null, 100);

    setLastSync(new Date());
    setIsSyncing(false);
    return { synced: 5, failed: 0 };
  }, [connectionState, simulateAsync]);

  // Test helper: simulate new email received
  const simulateNewEmail = useCallback((email: Email) => {
    setEmails((prev) => [email, ...prev]);
    if (mailboxStatus) {
      setMailboxStatus({
        ...mailboxStatus,
        inboxCount: mailboxStatus.inboxCount + 1,
        unreadCount: email.isRead ? mailboxStatus.unreadCount : mailboxStatus.unreadCount + 1,
      });
    }
    // Dispatch holochain signal event
    window.dispatchEvent(
      new CustomEvent('holochain-signal', {
        detail: { type: 'EmailReceived', payload: { email_hash: email.hash, subject: email.subject } },
      })
    );
  }, [mailboxStatus]);

  // Test helper: simulate connection lost
  const simulateConnectionLost = useCallback(() => {
    setConnectionState('error', new Error('Connection lost'));
  }, [setConnectionState]);

  // Test helper: simulate reconnected
  const simulateReconnected = useCallback(() => {
    setConnectionState('connected');
  }, [setConnectionState]);

  const value = useMemo<HolochainTestContextValue>(() => ({
    // Connection
    connectionState,
    connectionError,
    isConnected: connectionState === 'connected',

    // Inbox
    emails,
    inboxLoading,
    inboxError,
    refreshInbox,

    // Send
    sendEmail,
    isSending,
    sendError,

    // Trust
    trustNodes,
    trustEdges,
    trustLoading,
    trustError,
    refreshTrustNetwork,
    createAttestation,

    // Contacts
    contacts,
    contactsLoading,
    contactsError,
    refreshContacts,
    createContact,
    updateContact,
    deleteContact,

    // Mailbox status
    mailboxStatus,
    statusLoading,
    statusError,
    refreshStatus,

    // Sync
    sync,
    isSyncing,
    lastSync,
    syncError,

    // Test controls
    setEmails,
    setContacts,
    setConnectionState,
    simulateNewEmail,
    simulateConnectionLost,
    simulateReconnected,
  }), [
    connectionState,
    connectionError,
    emails,
    inboxLoading,
    inboxError,
    refreshInbox,
    sendEmail,
    isSending,
    sendError,
    trustNodes,
    trustEdges,
    trustLoading,
    trustError,
    refreshTrustNetwork,
    createAttestation,
    contacts,
    contactsLoading,
    contactsError,
    refreshContacts,
    createContact,
    updateContact,
    deleteContact,
    mailboxStatus,
    statusLoading,
    statusError,
    refreshStatus,
    sync,
    isSyncing,
    lastSync,
    syncError,
    setConnectionState,
    simulateNewEmail,
    simulateConnectionLost,
    simulateReconnected,
  ]);

  return (
    <HolochainTestContext.Provider value={value}>
      {children}
    </HolochainTestContext.Provider>
  );
}

// ============================================================================
// Hook for accessing test context
// ============================================================================

export function useHolochainTestContext(): HolochainTestContextValue {
  const context = useContext(HolochainTestContext);
  if (!context) {
    throw new Error('useHolochainTestContext must be used within HolochainTestProvider');
  }
  return context;
}

// ============================================================================
// Re-export mock data for convenience
// ============================================================================

export { mockEmails, mockContacts, mockTrustNodes, mockTrustEdges, mockMailboxStatus };
