// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Holochain Mail Client
 *
 * Uses the Mycelix SDK to connect to the Mail DNA on Holochain.
 * Supports both real conductor connections and mock mode for development.
 */

import {
  createClient,
  createMockClient,
  type MycelixClient,
  type MockMycelixClient,
} from '@mycelix/sdk';
import { matl as matlModule } from '@mycelix/sdk/matl';

// ============================================================================
// Types matching the Mail DNA
// ============================================================================

export interface MailMessage {
  id: string;
  from_did: string;
  to_did: string;
  subject: string;
  body: string;
  timestamp: string;
  thread_id?: string;
  is_read: boolean;
  is_starred: boolean;
  is_archived: boolean;
  trust_score: number;
  epistemic_tier: string;
  attachments: Attachment[];
}

export interface Attachment {
  id: string;
  filename: string;
  content_type: string;
  size: number;
  cid: string; // IPFS CID
}

export interface Contact {
  id: string;
  did: string;
  email: string;
  name?: string;
  avatar_url?: string;
  trust_score: number;
  is_blocked: boolean;
  is_favorite: boolean;
  tags: string[];
  last_interaction?: string;
}

export interface TrustScore {
  did: string;
  score: number;
  is_byzantine: boolean;
  total_interactions: number;
  breakdown: {
    direct: number;
    network: number;
    behavior: number;
  };
}

export interface SendMessageInput {
  to_did: string;
  subject: string;
  body: string;
  thread_id?: string;
}

// ============================================================================
// Mock Data for Development
// ============================================================================

const MOCK_CONTACTS: Contact[] = [
  {
    id: 'contact-1',
    did: 'did:mycelix:bob',
    email: 'bob@example.com',
    name: 'Bob Wilson',
    trust_score: 0.92,
    is_blocked: false,
    is_favorite: true,
    tags: ['work', 'trusted'],
    last_interaction: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'contact-2',
    did: 'did:mycelix:carol',
    email: 'carol@trusted.org',
    name: 'Carol Martinez',
    trust_score: 0.88,
    is_blocked: false,
    is_favorite: false,
    tags: ['friends'],
    last_interaction: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'contact-3',
    did: 'did:mycelix:david',
    email: 'david@newconnection.io',
    name: 'David Kim',
    trust_score: 0.45,
    is_blocked: false,
    is_favorite: false,
    tags: ['new'],
    last_interaction: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
  },
];

const MOCK_MESSAGES: MailMessage[] = [
  {
    id: 'msg-1',
    from_did: 'did:mycelix:bob',
    to_did: 'did:mycelix:me',
    subject: 'Project Update: Mycelix Integration Complete',
    body: `Hi there,

Great news! The Holochain integration is now working perfectly. All tests are passing and the trust scoring system is giving accurate results.

Key achievements:
- MATL spam filtering at 99.2% accuracy
- Trust attestations propagating within 2 seconds
- Byzantine fault detection working correctly

Let me know when you'd like to do a demo!

Best,
Bob`,
    timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    is_read: false,
    is_starred: true,
    is_archived: false,
    trust_score: 0.92,
    epistemic_tier: 'Tier2PrivatelyVerifiable',
    attachments: [],
  },
  {
    id: 'msg-2',
    from_did: 'did:mycelix:carol',
    to_did: 'did:mycelix:me',
    subject: 'Re: Dinner plans for Saturday?',
    body: `Saturday works great for me! How about that new place downtown?

Carol`,
    timestamp: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
    is_read: true,
    is_starred: false,
    is_archived: false,
    trust_score: 0.88,
    epistemic_tier: 'Tier1Testimonial',
    attachments: [],
  },
  {
    id: 'msg-3',
    from_did: 'did:mycelix:david',
    to_did: 'did:mycelix:me',
    subject: 'Introduction from the conference',
    body: `Hi,

It was great meeting you at the Holochain conference last week. I'd love to continue the discussion.

David`,
    timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    is_read: false,
    is_starred: false,
    is_archived: false,
    trust_score: 0.45,
    epistemic_tier: 'Tier1Testimonial',
    attachments: [],
  },
];

// ============================================================================
// Holochain Mail Client
// ============================================================================

export class HolochainMailClient {
  private client: MycelixClient | MockMycelixClient;
  private isMockMode: boolean;
  private myDid: string = 'did:mycelix:me';

  // Mock state for development
  private mockMessages: MailMessage[] = [...MOCK_MESSAGES];
  private mockContacts: Contact[] = [...MOCK_CONTACTS];

  constructor(options: {
    conductorUrl?: string;
    mockMode?: boolean;
  } = {}) {
    this.isMockMode = options.mockMode ?? true; // Default to mock mode

    if (this.isMockMode) {
      this.client = createMockClient();
      console.log('🍄 Holochain Mail Client (Mock Mode)');
    } else {
      this.client = createClient({
        installedAppId: 'mycelix-mail',
        appUrl: options.conductorUrl || 'ws://localhost:8888',
      });
      console.log('🍄 Holochain Mail Client (Real Mode)');
    }
  }

  async connect(): Promise<void> {
    await this.client.connect();
  }

  async disconnect(): Promise<void> {
    await this.client.disconnect();
  }

  isConnected(): boolean {
    return this.client.isConnected();
  }

  // =========================================================================
  // DID Management
  // =========================================================================

  async registerDid(did: string): Promise<void> {
    if (this.isMockMode) {
      this.myDid = did;
      return;
    }

    await (this.client as MycelixClient).callZome({
      role_name: 'mycelix-mail',
      zome_name: 'mail_messages',
      fn_name: 'register_my_did',
      payload: { did },
    });
    this.myDid = did;
  }

  getMyDid(): string {
    return this.myDid;
  }

  // =========================================================================
  // Message Operations
  // =========================================================================

  async getInbox(options: {
    minTrust?: number;
    limit?: number;
  } = {}): Promise<MailMessage[]> {
    if (this.isMockMode) {
      let messages = this.mockMessages.filter(m => !m.is_archived);
      if (options.minTrust !== undefined) {
        messages = messages.filter(m => m.trust_score >= options.minTrust!);
      }
      return messages.sort((a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
    }

    const minTrust = options.minTrust ?? 0.3;
    return (this.client as MycelixClient).callZome({
      role_name: 'mycelix-mail',
      zome_name: 'trust_filter',
      fn_name: 'filter_inbox',
      payload: minTrust,
    });
  }

  async getMessage(id: string): Promise<MailMessage | null> {
    if (this.isMockMode) {
      return this.mockMessages.find(m => m.id === id) || null;
    }

    return (this.client as MycelixClient).callZome({
      role_name: 'mycelix-mail',
      zome_name: 'mail_messages',
      fn_name: 'get_message',
      payload: id,
    });
  }

  async sendMessage(input: SendMessageInput): Promise<string> {
    if (this.isMockMode) {
      const newMessage: MailMessage = {
        id: `msg-${Date.now()}`,
        from_did: this.myDid,
        to_did: input.to_did,
        subject: input.subject,
        body: input.body,
        timestamp: new Date().toISOString(),
        thread_id: input.thread_id,
        is_read: true,
        is_starred: false,
        is_archived: false,
        trust_score: 1.0,
        epistemic_tier: 'Tier2PrivatelyVerifiable',
        attachments: [],
      };
      this.mockMessages.unshift(newMessage);
      return newMessage.id;
    }

    const result = await (this.client as MycelixClient).callZome<{ id: string }>({
      role_name: 'mycelix-mail',
      zome_name: 'mail_messages',
      fn_name: 'send_message',
      payload: {
        from_did: this.myDid,
        to_did: input.to_did,
        subject_encrypted: new TextEncoder().encode(input.subject),
        body_cid: input.body, // In real impl, would upload to IPFS first
        timestamp: { secs: Math.floor(Date.now() / 1000), nanos: 0 },
        thread_id: input.thread_id || null,
        epistemic_tier: 'Tier2PrivatelyVerifiable',
      },
    });
    return result.id;
  }

  async markAsRead(id: string): Promise<void> {
    if (this.isMockMode) {
      const msg = this.mockMessages.find(m => m.id === id);
      if (msg) msg.is_read = true;
      return;
    }

    // Would call update_message in real implementation
  }

  async toggleStar(id: string): Promise<void> {
    if (this.isMockMode) {
      const msg = this.mockMessages.find(m => m.id === id);
      if (msg) msg.is_starred = !msg.is_starred;
      return;
    }
  }

  async archive(id: string): Promise<void> {
    if (this.isMockMode) {
      const msg = this.mockMessages.find(m => m.id === id);
      if (msg) msg.is_archived = true;
      return;
    }
  }

  // =========================================================================
  // Trust Operations
  // =========================================================================

  async getTrustScore(did: string): Promise<TrustScore> {
    if (this.isMockMode) {
      const contact = this.mockContacts.find(c => c.did === did);
      const score = contact?.trust_score ?? 0.5;
      return {
        did,
        score,
        is_byzantine: score < 0.2,
        total_interactions: Math.floor(Math.random() * 50),
        breakdown: {
          direct: score * 0.4,
          network: score * 0.35,
          behavior: score * 0.25,
        },
      };
    }

    return (this.client as MycelixClient).callZome({
      role_name: 'mycelix-mail',
      zome_name: 'trust_filter',
      fn_name: 'evaluate_trust_matl',
      payload: did,
    });
  }

  async recordPositiveInteraction(did: string): Promise<void> {
    if (this.isMockMode) {
      const contact = this.mockContacts.find(c => c.did === did);
      if (contact) {
        // Use MATL's Bayesian update
        contact.trust_score = Math.min(1, contact.trust_score + 0.02);
      }
      return;
    }

    await (this.client as MycelixClient).callZome({
      role_name: 'mycelix-mail',
      zome_name: 'trust_filter',
      fn_name: 'record_positive_interaction',
      payload: did,
    });
  }

  async reportSpam(messageId: string, reason: string): Promise<void> {
    if (this.isMockMode) {
      const msg = this.mockMessages.find(m => m.id === messageId);
      if (msg) {
        // Reduce trust score of sender
        const contact = this.mockContacts.find(c => c.did === msg.from_did);
        if (contact) {
          contact.trust_score = Math.max(0, contact.trust_score - 0.1);
        }
        // Archive the spam message
        msg.is_archived = true;
      }
      return;
    }

    await (this.client as MycelixClient).callZome({
      role_name: 'mycelix-mail',
      zome_name: 'trust_filter',
      fn_name: 'report_spam',
      payload: { message_hash: messageId, reason },
    });
  }

  // =========================================================================
  // Contact Operations
  // =========================================================================

  async getContacts(): Promise<Contact[]> {
    if (this.isMockMode) {
      return this.mockContacts;
    }

    // Would fetch from address book zome
    return [];
  }

  async addContact(contact: Omit<Contact, 'id' | 'trust_score'>): Promise<string> {
    if (this.isMockMode) {
      const newContact: Contact = {
        ...contact,
        id: `contact-${Date.now()}`,
        trust_score: 0.5, // Default for new contacts
      };
      this.mockContacts.push(newContact);
      return newContact.id;
    }

    // Would call create_contact in real implementation
    return '';
  }

  async blockContact(did: string): Promise<void> {
    if (this.isMockMode) {
      const contact = this.mockContacts.find(c => c.did === did);
      if (contact) contact.is_blocked = true;
      return;
    }
  }

  // =========================================================================
  // Cross-hApp Reputation (via SDK Bridge)
  // =========================================================================

  async getCrossHappReputation(did: string): Promise<{
    local_score: number;
    cross_happ_score: number;
    happ_count: number;
    confidence: number;
  }> {
    const localScore = (await this.getTrustScore(did)).score;

    if (this.isMockMode) {
      return {
        local_score: localScore,
        cross_happ_score: localScore * 0.95, // Simulate slight variance
        happ_count: 3,
        confidence: 0.85,
      };
    }

    const crossRep = await this.client.queryCrossHappReputation(did);
    return {
      local_score: localScore,
      cross_happ_score: crossRep.aggregate,
      happ_count: crossRep.scores.length,
      confidence: Math.min(1, crossRep.total_interactions / 100),
    };
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let clientInstance: HolochainMailClient | null = null;

export function getMailClient(options?: {
  conductorUrl?: string;
  mockMode?: boolean;
}): HolochainMailClient {
  if (!clientInstance) {
    clientInstance = new HolochainMailClient(options);
  }
  return clientInstance;
}

export function resetMailClient(): void {
  clientInstance = null;
}
