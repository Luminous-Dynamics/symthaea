// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Holochain Mock Providers
 *
 * Provides mock implementations for testing Holochain hooks without a real connection
 */

import { vi } from 'vitest';
import type { Email, TrustNode, TrustEdge, Contact, MailboxStatus } from '@/lib/holochain/hooks';

// ============================================================================
// Mock Data
// ============================================================================

export const mockEmails: Email[] = [
  {
    hash: 'uhCkk1234567890abcdef',
    sender: 'alice@holochain.test',
    recipient: 'testuser@mycelix.mail',
    subject: 'Test Email from Alice',
    body: 'Hello! This is a test email from Alice.',
    timestamp: Date.now() - 3600000,
    isRead: false,
    isStarred: true,
    trustLevel: 0.85,
    attachments: [],
    labels: ['important'],
  },
  {
    hash: 'uhCkk0987654321fedcba',
    sender: 'bob@holochain.test',
    recipient: 'testuser@mycelix.mail',
    subject: 'Meeting Tomorrow',
    body: 'Just confirming our meeting for tomorrow at 2pm.',
    timestamp: Date.now() - 7200000,
    isRead: true,
    isStarred: false,
    trustLevel: 0.65,
    attachments: [],
    labels: [],
  },
  {
    hash: 'uhCkkabcdef1234567890',
    sender: 'unknown@suspicious.net',
    recipient: 'testuser@mycelix.mail',
    subject: 'Suspicious Message',
    body: 'This is from an unknown sender.',
    timestamp: Date.now() - 86400000,
    isRead: false,
    isStarred: false,
    trustLevel: 0.15,
    attachments: [],
    labels: ['spam'],
  },
];

export const mockContacts: Contact[] = [
  {
    id: 'contact-1',
    name: 'Alice Johnson',
    email: 'alice@holochain.test',
    agentPubKey: 'uhCAkalice123',
    trustLevel: 0.9,
    groups: ['friends', 'work'],
  },
  {
    id: 'contact-2',
    name: 'Bob Smith',
    email: 'bob@holochain.test',
    agentPubKey: 'uhCAkbob456',
    trustLevel: 0.75,
    groups: ['work'],
  },
  {
    id: 'contact-3',
    name: 'Charlie Brown',
    email: 'charlie@holochain.test',
    agentPubKey: 'uhCAkcharlie789',
    trustLevel: 0.6,
    groups: [],
  },
];

export const mockTrustNodes: TrustNode[] = [
  {
    id: 'uhCAkme123',
    name: 'You',
    email: 'testuser@mycelix.mail',
    trustLevel: 1,
    attestationCount: 0,
    isMe: true,
  },
  {
    id: 'uhCAkalice123',
    name: 'Alice Johnson',
    email: 'alice@holochain.test',
    trustLevel: 0.9,
    attestationCount: 5,
  },
  {
    id: 'uhCAkbob456',
    name: 'Bob Smith',
    email: 'bob@holochain.test',
    trustLevel: 0.75,
    attestationCount: 3,
  },
];

export const mockTrustEdges: TrustEdge[] = [
  {
    source: 'uhCAkme123',
    target: 'uhCAkalice123',
    trustLevel: 0.9,
    createdAt: Date.now() - 86400000 * 30,
  },
  {
    source: 'uhCAkme123',
    target: 'uhCAkbob456',
    trustLevel: 0.75,
    createdAt: Date.now() - 86400000 * 15,
  },
];

export const mockMailboxStatus: MailboxStatus = {
  inboxCount: 42,
  unreadCount: 7,
  sentCount: 156,
  draftCount: 3,
  isOnline: true,
  pendingOps: 0,
  lastSync: new Date(),
};

// ============================================================================
// Mock Holochain Client
// ============================================================================

export interface MockHolochainClient {
  myAgentPubKey: string;
  getInboxWithTrust: ReturnType<typeof vi.fn>;
  sendEmailWithTrustCheck: ReturnType<typeof vi.fn>;
  getMailboxStatus: ReturnType<typeof vi.fn>;
  syncWithAllPeers: ReturnType<typeof vi.fn>;
  goOnline: ReturnType<typeof vi.fn>;
  goOffline: ReturnType<typeof vi.fn>;
  signals: {
    on: ReturnType<typeof vi.fn>;
    onAny: ReturnType<typeof vi.fn>;
    off: ReturnType<typeof vi.fn>;
  };
  trust: {
    getMyAttestations: ReturnType<typeof vi.fn>;
    createAttestation: ReturnType<typeof vi.fn>;
    getSenderTrust: ReturnType<typeof vi.fn>;
  };
}

export function createMockHolochainClient(overrides?: Partial<MockHolochainClient>): MockHolochainClient {
  return {
    myAgentPubKey: 'uhCAkme123',
    getInboxWithTrust: vi.fn().mockResolvedValue(
      mockEmails.map((email) => ({
        email: {
          hash: email.hash,
          sender: email.sender,
          recipient: email.recipient,
          subject: email.subject,
          body: email.body,
          timestamp: email.timestamp * 1000,
          is_read: email.isRead,
          is_starred: email.isStarred,
          attachments: email.attachments,
          labels: email.labels,
        },
        sender_trust: email.trustLevel,
      }))
    ),
    sendEmailWithTrustCheck: vi.fn().mockResolvedValue({
      sent: true,
      email_hash: 'uhCkknew123',
      reason: null,
    }),
    getMailboxStatus: vi.fn().mockResolvedValue({
      inbox_count: mockMailboxStatus.inboxCount,
      unread_count: mockMailboxStatus.unreadCount,
      sent_count: mockMailboxStatus.sentCount,
      draft_count: mockMailboxStatus.draftCount,
      sync_status: {
        is_online: mockMailboxStatus.isOnline,
        pending_ops: mockMailboxStatus.pendingOps,
        last_sync: mockMailboxStatus.lastSync?.toISOString() || null,
      },
    }),
    syncWithAllPeers: vi.fn().mockResolvedValue({
      synced: 5,
      failed: 0,
    }),
    goOnline: vi.fn().mockResolvedValue(undefined),
    goOffline: vi.fn().mockResolvedValue(undefined),
    signals: {
      on: vi.fn(),
      onAny: vi.fn(),
      off: vi.fn(),
    },
    trust: {
      getMyAttestations: vi.fn().mockResolvedValue(
        mockTrustEdges.map((edge) => ({
          target: edge.target,
          trust_level: edge.trustLevel,
          created_at: edge.createdAt,
          context: 'communication',
          category: 'Communication',
        }))
      ),
      createAttestation: vi.fn().mockResolvedValue({
        hash: 'uhCkkatt123',
        success: true,
      }),
      getSenderTrust: vi.fn().mockResolvedValue(0.75),
    },
    ...overrides,
  };
}

// ============================================================================
// Mock Mycelix Client
// ============================================================================

export interface MockMycelixClient {
  contacts: {
    getAllContacts: ReturnType<typeof vi.fn>;
    createContact: ReturnType<typeof vi.fn>;
    updateContact: ReturnType<typeof vi.fn>;
    deleteContact: ReturnType<typeof vi.fn>;
    getContact: ReturnType<typeof vi.fn>;
  };
  shutdown: ReturnType<typeof vi.fn>;
}

export function createMockMycelixClient(overrides?: Partial<MockMycelixClient>): MockMycelixClient {
  return {
    contacts: {
      getAllContacts: vi.fn().mockResolvedValue(
        mockContacts.map((c) => ({
          hash: c.id,
          id: c.id,
          name: c.name,
          email: c.email,
          agent_pub_key: c.agentPubKey,
          trust_level: c.trustLevel,
          groups: c.groups,
        }))
      ),
      createContact: vi.fn().mockResolvedValue({ hash: 'new-contact-id' }),
      updateContact: vi.fn().mockResolvedValue({ success: true }),
      deleteContact: vi.fn().mockResolvedValue({ success: true }),
      getContact: vi.fn().mockImplementation((id: string) => {
        const contact = mockContacts.find((c) => c.id === id);
        if (contact) {
          return Promise.resolve({
            hash: contact.id,
            id: contact.id,
            name: contact.name,
            email: contact.email,
            agent_pub_key: contact.agentPubKey,
            trust_level: contact.trustLevel,
            groups: contact.groups,
          });
        }
        return Promise.reject(new Error('Contact not found'));
      }),
    },
    shutdown: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  };
}

// ============================================================================
// Connection State Mock
// ============================================================================

export type MockConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

let mockConnectionState: MockConnectionState = 'connected';
let mockConnectionError: Error | null = null;
const mockConnectionListeners: Set<(state: MockConnectionState, error?: Error) => void> = new Set();

export function setMockConnectionState(state: MockConnectionState, error?: Error): void {
  mockConnectionState = state;
  mockConnectionError = error || null;
  mockConnectionListeners.forEach((listener) => listener(state, error));
}

export function getMockConnectionState(): { state: MockConnectionState; error: Error | null } {
  return { state: mockConnectionState, error: mockConnectionError };
}

export function onMockConnectionStateChange(
  listener: (state: MockConnectionState, error?: Error) => void
): () => void {
  mockConnectionListeners.add(listener);
  listener(mockConnectionState, mockConnectionError || undefined);
  return () => mockConnectionListeners.delete(listener);
}

export function resetMockConnectionState(): void {
  mockConnectionState = 'connected';
  mockConnectionError = null;
  mockConnectionListeners.clear();
}

// ============================================================================
// Mock Module Factory
// ============================================================================

let mockClient: MockHolochainClient | null = null;
let mockMycelixClientInstance: MockMycelixClient | null = null;

export function setupHolochainMocks(
  clientOverrides?: Partial<MockHolochainClient>,
  mycelixOverrides?: Partial<MockMycelixClient>
): {
  holochainClient: MockHolochainClient;
  mycelixClient: MockMycelixClient;
} {
  mockClient = createMockHolochainClient(clientOverrides);
  mockMycelixClientInstance = createMockMycelixClient(mycelixOverrides);

  return {
    holochainClient: mockClient,
    mycelixClient: mockMycelixClientInstance,
  };
}

export function getMockHolochainClient(): MockHolochainClient | null {
  return mockClient;
}

export function getMockMycelixClient(): MockMycelixClient | null {
  return mockMycelixClientInstance;
}

export function clearHolochainMocks(): void {
  mockClient = null;
  mockMycelixClientInstance = null;
  resetMockConnectionState();
}

// ============================================================================
// Vitest Mock Factory for Holochain Module
// ============================================================================

/**
 * Creates a mock for the @/lib/holochain/index module
 * Use with vi.mock('@/lib/holochain/index', () => createHolochainIndexMock())
 */
export function createHolochainIndexMock() {
  const mocks = setupHolochainMocks();

  return {
    connectToHolochain: vi.fn().mockResolvedValue(mocks.holochainClient),
    disconnectFromHolochain: vi.fn().mockResolvedValue(undefined),
    getHolochainClient: vi.fn(() => {
      if (mockConnectionState !== 'connected' || !mockClient) {
        throw new Error('Holochain client not connected');
      }
      return mockClient;
    }),
    getMycelixClient: vi.fn(() => {
      if (mockConnectionState !== 'connected' || !mockMycelixClientInstance) {
        throw new Error('Mycelix client not connected');
      }
      return mockMycelixClientInstance;
    }),
    isConnected: vi.fn(() => mockConnectionState === 'connected'),
    getConnectionState: vi.fn(() => getMockConnectionState()),
    onConnectionStateChange: vi.fn((listener: (state: MockConnectionState, error?: Error) => void) =>
      onMockConnectionStateChange(listener)
    ),
    initializeHolochain: vi.fn().mockResolvedValue(undefined),
    cleanupHolochain: vi.fn().mockResolvedValue(undefined),
  };
}
