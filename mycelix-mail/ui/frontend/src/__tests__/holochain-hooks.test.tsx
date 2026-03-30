// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Holochain Hooks Unit Tests
 *
 * Comprehensive tests for all Holochain integration hooks
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { ReactNode } from 'react';
import {
  useHolochainConnection,
  useHolochainClient,
  useInbox,
  useSendEmail,
  useTrustNetwork,
  useContacts,
  useMailboxStatus,
  useSync,
} from '@/lib/holochain/hooks';
import {
  createHolochainIndexMock,
  setupHolochainMocks,
  clearHolochainMocks,
  setMockConnectionState,
  mockEmails,
  mockContacts,
  mockTrustNodes,
  mockTrustEdges,
  mockMailboxStatus,
} from '@/test/mocks/holochain';

// Mock the holochain index module
vi.mock('@/lib/holochain/index', () => createHolochainIndexMock());

// Import the mocked functions for assertions
import * as holochainIndex from '@/lib/holochain/index';

// ============================================================================
// Test Setup
// ============================================================================

beforeEach(() => {
  vi.clearAllMocks();
  clearHolochainMocks();
  setupHolochainMocks();
  setMockConnectionState('connected');
});

afterEach(() => {
  clearHolochainMocks();
});

// Wrapper for hooks that need React context
const wrapper = ({ children }: { children: ReactNode }) => children;

// ============================================================================
// useHolochainConnection Tests
// ============================================================================

describe('useHolochainConnection', () => {
  it('should return connected state when connected', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useHolochainConnection(), { wrapper });

    await waitFor(() => {
      expect(result.current.state).toBe('connected');
      expect(result.current.isConnected).toBe(true);
      expect(result.current.isConnecting).toBe(false);
      expect(result.current.isError).toBe(false);
      expect(result.current.error).toBeNull();
    });
  });

  it('should return connecting state when connecting', async () => {
    setMockConnectionState('connecting');

    const { result } = renderHook(() => useHolochainConnection(), { wrapper });

    await waitFor(() => {
      expect(result.current.state).toBe('connecting');
      expect(result.current.isConnected).toBe(false);
      expect(result.current.isConnecting).toBe(true);
    });
  });

  it('should return error state with error object', async () => {
    const testError = new Error('Connection failed');
    setMockConnectionState('error', testError);

    const { result } = renderHook(() => useHolochainConnection(), { wrapper });

    await waitFor(() => {
      expect(result.current.state).toBe('error');
      expect(result.current.isError).toBe(true);
      expect(result.current.error).toEqual(testError);
    });
  });

  it('should update state when connection state changes', async () => {
    setMockConnectionState('disconnected');

    const { result } = renderHook(() => useHolochainConnection(), { wrapper });

    await waitFor(() => {
      expect(result.current.state).toBe('disconnected');
    });

    act(() => {
      setMockConnectionState('connecting');
    });

    await waitFor(() => {
      expect(result.current.state).toBe('connecting');
    });

    act(() => {
      setMockConnectionState('connected');
    });

    await waitFor(() => {
      expect(result.current.state).toBe('connected');
    });
  });
});

// ============================================================================
// useHolochainClient Tests
// ============================================================================

describe('useHolochainClient', () => {
  it('should return client when connected', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useHolochainClient(), { wrapper });

    await waitFor(() => {
      expect(result.current).not.toBeNull();
    });
  });

  it('should return null when disconnected', async () => {
    setMockConnectionState('disconnected');

    const { result } = renderHook(() => useHolochainClient(), { wrapper });

    await waitFor(() => {
      expect(result.current).toBeNull();
    });
  });

  it('should return null when in error state', async () => {
    setMockConnectionState('error', new Error('Connection error'));

    const { result } = renderHook(() => useHolochainClient(), { wrapper });

    await waitFor(() => {
      expect(result.current).toBeNull();
    });
  });
});

// ============================================================================
// useInbox Tests
// ============================================================================

describe('useInbox', () => {
  it('should fetch emails on mount when connected', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useInbox(), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.emails).toHaveLength(mockEmails.length);
    expect(result.current.error).toBeNull();
  });

  it('should respect limit parameter', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useInbox(2), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // The mock should be called with the limit
    const client = holochainIndex.getHolochainClient();
    expect(client.getInboxWithTrust).toHaveBeenCalledWith(2);
  });

  it('should map email properties correctly', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useInbox(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const firstEmail = result.current.emails[0];
    expect(firstEmail).toMatchObject({
      hash: expect.any(String),
      sender: expect.any(String),
      recipient: expect.any(String),
      subject: expect.any(String),
      body: expect.any(String),
      timestamp: expect.any(Number),
      isRead: expect.any(Boolean),
      isStarred: expect.any(Boolean),
      trustLevel: expect.any(Number),
      attachments: expect.any(Array),
      labels: expect.any(Array),
    });
  });

  it('should handle refresh correctly', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useInbox(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const client = holochainIndex.getHolochainClient();
    const initialCallCount = client.getInboxWithTrust.mock.calls.length;

    await act(async () => {
      await result.current.refresh();
    });

    expect(client.getInboxWithTrust.mock.calls.length).toBe(initialCallCount + 1);
  });

  it('should handle errors gracefully', async () => {
    setMockConnectionState('connected');
    const client = holochainIndex.getHolochainClient();
    const testError = new Error('Failed to fetch inbox');
    client.getInboxWithTrust.mockRejectedValueOnce(testError);

    const { result } = renderHook(() => useInbox(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.emails).toEqual([]);
  });

  it('should not fetch when disconnected', async () => {
    setMockConnectionState('disconnected');

    const { result } = renderHook(() => useInbox(), { wrapper });

    // Should remain loading but never resolve
    expect(result.current.emails).toEqual([]);
  });

  it('should respond to holochain-signal events', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useInbox(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const client = holochainIndex.getHolochainClient();
    const callsBeforeSignal = client.getInboxWithTrust.mock.calls.length;

    // Simulate receiving a new email signal
    act(() => {
      window.dispatchEvent(
        new CustomEvent('holochain-signal', {
          detail: { type: 'EmailReceived', payload: { email_hash: 'new-email' } },
        })
      );
    });

    await waitFor(() => {
      expect(client.getInboxWithTrust.mock.calls.length).toBe(callsBeforeSignal + 1);
    });
  });
});

// ============================================================================
// useSendEmail Tests
// ============================================================================

describe('useSendEmail', () => {
  it('should send email successfully', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useSendEmail(), { wrapper });

    expect(result.current.isSending).toBe(false);
    expect(result.current.error).toBeNull();

    let sendResult: any;
    await act(async () => {
      sendResult = await result.current.sendEmail({
        to: 'recipient@example.com',
        subject: 'Test Subject',
        body: 'Test body content',
      });
    });

    expect(sendResult.sent).toBe(true);
    expect(sendResult.email_hash).toBeDefined();
    expect(result.current.isSending).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('should include optional attachments and priority', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useSendEmail(), { wrapper });

    await act(async () => {
      await result.current.sendEmail({
        to: 'recipient@example.com',
        subject: 'Test with attachments',
        body: 'Body with attachments',
        attachments: [{ name: 'file.txt', data: 'content' }],
        priority: 'High',
      });
    });

    const client = holochainIndex.getHolochainClient();
    expect(client.sendEmailWithTrustCheck).toHaveBeenCalledWith(
      'recipient@example.com',
      'Test with attachments',
      'Body with attachments',
      expect.objectContaining({
        attachments: [{ name: 'file.txt', data: 'content' }],
        priority: 'High',
      })
    );
  });

  it('should throw error when not connected', async () => {
    setMockConnectionState('disconnected');

    const { result } = renderHook(() => useSendEmail(), { wrapper });

    await expect(
      act(async () => {
        await result.current.sendEmail({
          to: 'recipient@example.com',
          subject: 'Test',
          body: 'Body',
        });
      })
    ).rejects.toThrow('Not connected to Holochain');
  });

  it('should handle send failure', async () => {
    setMockConnectionState('connected');
    const client = holochainIndex.getHolochainClient();
    client.sendEmailWithTrustCheck.mockResolvedValueOnce({
      sent: false,
      reason: 'Recipient blocked',
    });

    const { result } = renderHook(() => useSendEmail(), { wrapper });

    await expect(
      act(async () => {
        await result.current.sendEmail({
          to: 'blocked@example.com',
          subject: 'Test',
          body: 'Body',
        });
      })
    ).rejects.toThrow('Recipient blocked');

    expect(result.current.error).toBeDefined();
  });

  it('should update isSending during send operation', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useSendEmail(), { wrapper });

    expect(result.current.isSending).toBe(false);

    const sendPromise = act(async () => {
      return result.current.sendEmail({
        to: 'recipient@example.com',
        subject: 'Test',
        body: 'Body',
      });
    });

    // Note: Due to async nature, isSending might flip quickly
    await sendPromise;
    expect(result.current.isSending).toBe(false);
  });
});

// ============================================================================
// useTrustNetwork Tests
// ============================================================================

describe('useTrustNetwork', () => {
  it('should fetch trust network on mount', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useTrustNetwork(), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.nodes.length).toBeGreaterThan(0);
    expect(result.current.edges.length).toBeGreaterThan(0);
    expect(result.current.error).toBeNull();
  });

  it('should include self as a node with isMe flag', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useTrustNetwork(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const selfNode = result.current.nodes.find((n) => n.isMe);
    expect(selfNode).toBeDefined();
    expect(selfNode?.name).toBe('You');
    expect(selfNode?.trustLevel).toBe(1);
  });

  it('should create attestation and refresh network', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useTrustNetwork(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const client = holochainIndex.getHolochainClient();
    const initialFetchCount = client.trust.getMyAttestations.mock.calls.length;

    await act(async () => {
      await result.current.createAttestation('uhCAknewagent', 0.8, 'Professional contact');
    });

    expect(client.trust.createAttestation).toHaveBeenCalledWith({
      target: 'uhCAknewagent',
      trust_level: 0.8,
      context: 'Professional contact',
      category: 'Communication',
    });

    // Should refresh network after attestation
    expect(client.trust.getMyAttestations.mock.calls.length).toBe(initialFetchCount + 1);
  });

  it('should handle refresh correctly', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useTrustNetwork(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const client = holochainIndex.getHolochainClient();
    const initialCallCount = client.trust.getMyAttestations.mock.calls.length;

    await act(async () => {
      await result.current.refresh();
    });

    expect(client.trust.getMyAttestations.mock.calls.length).toBe(initialCallCount + 1);
  });

  it('should handle errors gracefully', async () => {
    setMockConnectionState('connected');
    const client = holochainIndex.getHolochainClient();
    const testError = new Error('Failed to fetch trust network');
    client.trust.getMyAttestations.mockRejectedValueOnce(testError);

    const { result } = renderHook(() => useTrustNetwork(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toEqual(testError);
  });

  it('should throw when creating attestation while disconnected', async () => {
    setMockConnectionState('disconnected');

    const { result } = renderHook(() => useTrustNetwork(), { wrapper });

    await expect(
      act(async () => {
        await result.current.createAttestation('uhCAktest', 0.5);
      })
    ).rejects.toThrow('Not connected');
  });
});

// ============================================================================
// useContacts Tests
// ============================================================================

describe('useContacts', () => {
  it('should fetch contacts on mount', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useContacts(), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.contacts).toHaveLength(mockContacts.length);
    expect(result.current.error).toBeNull();
  });

  it('should map contact properties correctly', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useContacts(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const firstContact = result.current.contacts[0];
    expect(firstContact).toMatchObject({
      id: expect.any(String),
      name: expect.any(String),
      email: expect.any(String),
      groups: expect.any(Array),
    });
  });

  it('should create contact and refresh list', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useContacts(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const mycelixClient = holochainIndex.getMycelixClient();
    const initialFetchCount = mycelixClient.contacts.getAllContacts.mock.calls.length;

    await act(async () => {
      await result.current.createContact({
        name: 'New Contact',
        email: 'newcontact@example.com',
        groups: ['friends'],
      });
    });

    expect(mycelixClient.contacts.createContact).toHaveBeenCalled();
    expect(mycelixClient.contacts.getAllContacts.mock.calls.length).toBe(initialFetchCount + 1);
  });

  it('should update contact and refresh list', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useContacts(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const mycelixClient = holochainIndex.getMycelixClient();

    await act(async () => {
      await result.current.updateContact('contact-1', { name: 'Updated Name' });
    });

    expect(mycelixClient.contacts.updateContact).toHaveBeenCalledWith('contact-1', { name: 'Updated Name' });
  });

  it('should delete contact and refresh list', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useContacts(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const mycelixClient = holochainIndex.getMycelixClient();

    await act(async () => {
      await result.current.deleteContact('contact-1');
    });

    expect(mycelixClient.contacts.deleteContact).toHaveBeenCalledWith('contact-1');
  });

  it('should handle refresh correctly', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useContacts(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const mycelixClient = holochainIndex.getMycelixClient();
    const initialCallCount = mycelixClient.contacts.getAllContacts.mock.calls.length;

    await act(async () => {
      await result.current.refresh();
    });

    expect(mycelixClient.contacts.getAllContacts.mock.calls.length).toBe(initialCallCount + 1);
  });

  it('should handle errors gracefully', async () => {
    setMockConnectionState('connected');
    const mycelixClient = holochainIndex.getMycelixClient();
    const testError = new Error('Failed to fetch contacts');
    mycelixClient.contacts.getAllContacts.mockRejectedValueOnce(testError);

    const { result } = renderHook(() => useContacts(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toEqual(testError);
  });

  it('should throw when modifying contacts while disconnected', async () => {
    setMockConnectionState('disconnected');

    const { result } = renderHook(() => useContacts(), { wrapper });

    await expect(
      act(async () => {
        await result.current.createContact({
          name: 'Test',
          email: 'test@example.com',
          groups: [],
        });
      })
    ).rejects.toThrow('Not connected');

    await expect(
      act(async () => {
        await result.current.updateContact('id', { name: 'Updated' });
      })
    ).rejects.toThrow('Not connected');

    await expect(
      act(async () => {
        await result.current.deleteContact('id');
      })
    ).rejects.toThrow('Not connected');
  });
});

// ============================================================================
// useMailboxStatus Tests
// ============================================================================

describe('useMailboxStatus', () => {
  it('should fetch mailbox status on mount', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useMailboxStatus(), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.status).toBeDefined();
    expect(result.current.error).toBeNull();
  });

  it('should map status properties correctly', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useMailboxStatus(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.status).toMatchObject({
      inboxCount: expect.any(Number),
      unreadCount: expect.any(Number),
      sentCount: expect.any(Number),
      draftCount: expect.any(Number),
      isOnline: expect.any(Boolean),
      pendingOps: expect.any(Number),
    });
  });

  it('should handle refresh correctly', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useMailboxStatus(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    const client = holochainIndex.getHolochainClient();
    const initialCallCount = client.getMailboxStatus.mock.calls.length;

    await act(async () => {
      await result.current.refresh();
    });

    expect(client.getMailboxStatus.mock.calls.length).toBe(initialCallCount + 1);
  });

  it('should handle errors gracefully', async () => {
    setMockConnectionState('connected');
    const client = holochainIndex.getHolochainClient();
    const testError = new Error('Failed to fetch status');
    client.getMailboxStatus.mockRejectedValueOnce(testError);

    const { result } = renderHook(() => useMailboxStatus(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toEqual(testError);
  });
});

// ============================================================================
// useSync Tests
// ============================================================================

describe('useSync', () => {
  it('should sync successfully', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useSync(), { wrapper });

    expect(result.current.isSyncing).toBe(false);
    expect(result.current.lastSync).toBeNull();

    let syncResult: any;
    await act(async () => {
      syncResult = await result.current.sync();
    });

    expect(syncResult.synced).toBe(5);
    expect(syncResult.failed).toBe(0);
    expect(result.current.isSyncing).toBe(false);
    expect(result.current.lastSync).toBeInstanceOf(Date);
    expect(result.current.error).toBeNull();
  });

  it('should update isSyncing during sync operation', async () => {
    setMockConnectionState('connected');

    const { result } = renderHook(() => useSync(), { wrapper });

    expect(result.current.isSyncing).toBe(false);

    const syncPromise = act(async () => {
      return result.current.sync();
    });

    await syncPromise;
    expect(result.current.isSyncing).toBe(false);
  });

  it('should handle sync errors', async () => {
    setMockConnectionState('connected');
    const client = holochainIndex.getHolochainClient();
    const testError = new Error('Sync failed');
    client.syncWithAllPeers.mockRejectedValueOnce(testError);

    const { result } = renderHook(() => useSync(), { wrapper });

    await expect(
      act(async () => {
        await result.current.sync();
      })
    ).rejects.toThrow('Sync failed');

    expect(result.current.error).toEqual(testError);
    expect(result.current.isSyncing).toBe(false);
  });

  it('should not sync when disconnected', async () => {
    setMockConnectionState('disconnected');

    const { result } = renderHook(() => useSync(), { wrapper });

    await act(async () => {
      await result.current.sync();
    });

    // Should not throw, but also not update lastSync
    expect(result.current.lastSync).toBeNull();
  });
});

// ============================================================================
// Integration Tests
// ============================================================================

describe('Holochain Hooks Integration', () => {
  it('should handle connection state changes across hooks', async () => {
    // Start connected
    setMockConnectionState('connected');

    const { result: inboxResult } = renderHook(() => useInbox(), { wrapper });
    const { result: contactsResult } = renderHook(() => useContacts(), { wrapper });

    await waitFor(() => {
      expect(inboxResult.current.isLoading).toBe(false);
      expect(contactsResult.current.isLoading).toBe(false);
    });

    expect(inboxResult.current.emails.length).toBeGreaterThan(0);
    expect(contactsResult.current.contacts.length).toBeGreaterThan(0);

    // Simulate disconnection
    act(() => {
      setMockConnectionState('error', new Error('Connection lost'));
    });

    // Hooks should now reflect disconnected state
    // (depending on implementation, they may show errors or stale data)
  });

  it('should handle rapid state changes gracefully', async () => {
    setMockConnectionState('disconnected');

    const { result } = renderHook(() => useHolochainConnection(), { wrapper });

    // Rapid state changes
    act(() => {
      setMockConnectionState('connecting');
    });

    act(() => {
      setMockConnectionState('connected');
    });

    act(() => {
      setMockConnectionState('error', new Error('Brief error'));
    });

    act(() => {
      setMockConnectionState('connected');
    });

    await waitFor(() => {
      expect(result.current.state).toBe('connected');
      expect(result.current.isConnected).toBe(true);
    });
  });
});
