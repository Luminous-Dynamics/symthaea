// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail - Mock GraphQL Server
 *
 * Provides realistic demo data for the Mail frontend without
 * needing the full Rust backend or Holochain.
 */

import express from 'express';
import cors from 'cors';

const app = express();
app.use(cors());
app.use(express.json());

// ============================================================================
// Mock Data
// ============================================================================

const mockUsers = {
  'alice@mycelix.net': {
    id: 'user-alice',
    email: 'alice@mycelix.net',
    name: 'Alice Chen',
    did: 'did:mycelix:alice',
    avatarUrl: null,
  },
  'demo@mycelix.net': {
    id: 'user-demo',
    email: 'demo@mycelix.net',
    name: 'Demo User',
    did: 'did:mycelix:demo',
    avatarUrl: null,
  }
};

const mockContacts = [
  {
    id: 'contact-1',
    email: 'bob@example.com',
    displayName: 'Bob Wilson',
    avatarUrl: null,
    holochainAgentId: 'did:mycelix:bob',
    trustScore: 0.92,
    interactionCount: 47,
    lastInteractionAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    isBlocked: false,
    isFavorite: true,
    tags: ['work', 'trusted'],
  },
  {
    id: 'contact-2',
    email: 'carol@trusted.org',
    displayName: 'Carol Martinez',
    avatarUrl: null,
    holochainAgentId: 'did:mycelix:carol',
    trustScore: 0.88,
    interactionCount: 23,
    lastInteractionAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    isBlocked: false,
    isFavorite: false,
    tags: ['friends'],
  },
  {
    id: 'contact-3',
    email: 'david@newconnection.io',
    displayName: 'David Kim',
    avatarUrl: null,
    holochainAgentId: 'did:mycelix:david',
    trustScore: 0.45,
    interactionCount: 3,
    lastInteractionAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    isBlocked: false,
    isFavorite: false,
    tags: ['new'],
  },
  {
    id: 'contact-4',
    email: 'spam@suspicious.xyz',
    displayName: null,
    avatarUrl: null,
    holochainAgentId: null,
    trustScore: 0.08,
    interactionCount: 1,
    lastInteractionAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
    isBlocked: true,
    isFavorite: false,
    tags: [],
  },
];

const mockEmails = [
  {
    id: 'email-1',
    threadId: 'thread-1',
    from: { email: 'bob@example.com', name: 'Bob Wilson' },
    to: [{ email: 'demo@mycelix.net', name: 'Demo User' }],
    cc: [],
    subject: 'Project Update: Mycelix Integration Complete',
    preview: 'Great news! The Holochain integration is now working perfectly. All tests passing...',
    bodyText: `Hi there,

Great news! The Holochain integration is now working perfectly. All tests are passing and the trust scoring system is giving accurate results.

Key achievements:
- MATL spam filtering at 99.2% accuracy
- Trust attestations propagating within 2 seconds
- Byzantine fault detection working correctly

Let me know when you'd like to do a demo!

Best,
Bob`,
    bodyHtml: null,
    isRead: false,
    isStarred: true,
    isArchived: false,
    hasAttachments: false,
    trustScore: 0.92,
    labels: ['important', 'work'],
    receivedAt: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    attachments: [],
  },
  {
    id: 'email-2',
    threadId: 'thread-2',
    from: { email: 'carol@trusted.org', name: 'Carol Martinez' },
    to: [{ email: 'demo@mycelix.net', name: 'Demo User' }],
    cc: [],
    subject: 'Re: Dinner plans for Saturday?',
    preview: 'Saturday works great for me! How about that new place downtown...',
    bodyText: `Saturday works great for me! How about that new place downtown that just opened?

I heard they have amazing food and it's walking distance from my place.

Let me know what time works for you!

Carol`,
    bodyHtml: null,
    isRead: true,
    isStarred: false,
    isArchived: false,
    hasAttachments: false,
    trustScore: 0.88,
    labels: ['personal'],
    receivedAt: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
    attachments: [],
  },
  {
    id: 'email-3',
    threadId: 'thread-3',
    from: { email: 'david@newconnection.io', name: 'David Kim' },
    to: [{ email: 'demo@mycelix.net', name: 'Demo User' }],
    cc: [],
    subject: 'Introduction from the conference',
    preview: 'It was great meeting you at the Holochain conference last week...',
    bodyText: `Hi,

It was great meeting you at the Holochain conference last week. I really enjoyed our conversation about decentralized identity.

I'd love to continue the discussion - maybe we could schedule a call?

David`,
    bodyHtml: null,
    isRead: false,
    isStarred: false,
    isArchived: false,
    hasAttachments: false,
    trustScore: 0.45,
    labels: [],
    receivedAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    attachments: [],
  },
  {
    id: 'email-4',
    threadId: 'thread-4',
    from: { email: 'noreply@mycelix.net', name: 'Mycelix Network' },
    to: [{ email: 'demo@mycelix.net', name: 'Demo User' }],
    cc: [],
    subject: 'Welcome to Mycelix Mail!',
    preview: 'Your trust-first email experience begins now. Here\'s how to get started...',
    bodyText: `Welcome to Mycelix Mail!

Your trust-first email experience begins now. Here's what makes Mycelix different:

1. Trust Scoring - Every sender has a trust score based on your network's attestations
2. MATL Filtering - Advanced spam protection that learns from the collective
3. E2E Encryption - Your messages are encrypted by default
4. Decentralized - No central server controls your data

Get started by adding your first trusted contacts!

The Mycelix Team`,
    bodyHtml: null,
    isRead: true,
    isStarred: true,
    isArchived: false,
    hasAttachments: false,
    trustScore: 0.95,
    labels: ['mycelix'],
    receivedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    attachments: [],
  },
  {
    id: 'email-5',
    threadId: 'thread-5',
    from: { email: 'unknown@sketchy.biz', name: null },
    to: [{ email: 'demo@mycelix.net', name: 'Demo User' }],
    cc: [],
    subject: 'URGENT: Verify your account immediately!!!',
    preview: 'Your account will be suspended unless you click this link...',
    bodyText: `URGENT!!!

Your account will be suspended unless you verify immediately!

Click here: [suspicious-link]

Act now or lose access forever!`,
    bodyHtml: null,
    isRead: false,
    isStarred: false,
    isArchived: false,
    hasAttachments: false,
    trustScore: 0.03,
    labels: ['spam'],
    receivedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    attachments: [],
  },
];

const mockTrustNetwork = {
  nodes: [
    { id: 'did:mycelix:demo', label: 'You', email: 'demo@mycelix.net', trustScore: 1.0 },
    { id: 'did:mycelix:bob', label: 'Bob W.', email: 'bob@example.com', trustScore: 0.92 },
    { id: 'did:mycelix:carol', label: 'Carol M.', email: 'carol@trusted.org', trustScore: 0.88 },
    { id: 'did:mycelix:david', label: 'David K.', email: 'david@newconnection.io', trustScore: 0.45 },
    { id: 'did:mycelix:eve', label: 'Eve L.', email: 'eve@network.org', trustScore: 0.78 },
    { id: 'did:mycelix:frank', label: 'Frank H.', email: 'frank@trusted.net', trustScore: 0.65 },
  ],
  edges: [
    { from: 'did:mycelix:demo', to: 'did:mycelix:bob', weight: 0.9, context: 'professional' },
    { from: 'did:mycelix:demo', to: 'did:mycelix:carol', weight: 0.85, context: 'personal' },
    { from: 'did:mycelix:bob', to: 'did:mycelix:carol', weight: 0.7, context: 'professional' },
    { from: 'did:mycelix:bob', to: 'did:mycelix:eve', weight: 0.8, context: 'professional' },
    { from: 'did:mycelix:carol', to: 'did:mycelix:david', weight: 0.4, context: 'general' },
    { from: 'did:mycelix:eve', to: 'did:mycelix:frank', weight: 0.6, context: 'verified' },
    { from: 'did:mycelix:demo', to: 'did:mycelix:david', weight: 0.3, context: 'general' },
  ],
};

const mockAttestations = {
  'did:mycelix:bob': [
    { id: 'att-1', fromAgentId: 'did:mycelix:demo', toAgentId: 'did:mycelix:bob', trustLevel: 0.9, context: 'professional', createdAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString() },
    { id: 'att-2', fromAgentId: 'did:mycelix:carol', toAgentId: 'did:mycelix:bob', trustLevel: 0.85, context: 'verified', createdAt: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString() },
  ],
  'did:mycelix:carol': [
    { id: 'att-3', fromAgentId: 'did:mycelix:demo', toAgentId: 'did:mycelix:carol', trustLevel: 0.88, context: 'personal', createdAt: new Date(Date.now() - 45 * 24 * 60 * 60 * 1000).toISOString() },
  ],
};

// Current authenticated user
let currentUser = null;

// ============================================================================
// GraphQL Handler
// ============================================================================

function handleGraphQL(query, variables = {}) {
  // Simple pattern matching for GraphQL operations
  // Note: More specific matches should come before general matches

  // Contact queries (check before 'me' since 'contacts' contains no 'me')
  if (query.includes('GetContacts') || (query.includes('contacts') && !query.includes('createContact'))) {
    return { contacts: mockContacts };
  }

  // Trust queries (check early - specific)
  if (query.includes('GetTrustScore') || query.includes('trustScore(')) {
    const email = variables.email;
    const contact = mockContacts.find(c => c.email === email);
    return {
      trustScore: {
        score: contact?.trustScore || 0.5,
        attestations: 3,
        pathLength: 2,
        breakdown: { direct: 0.4, network: 0.3, behavior: 0.3 },
      }
    };
  }

  if (query.includes('GetTrustNetwork') || query.includes('trustNetwork')) {
    return { trustNetwork: mockTrustNetwork };
  }

  if (query.includes('attestations') || query.includes('Attestations')) {
    const agentId = variables.agentId || variables.id;
    return { attestations: mockAttestations[agentId] || [] };
  }

  // Auth queries
  if (query.includes('login') || query.includes('Login')) {
    currentUser = mockUsers['demo@mycelix.net'];
    return {
      login: {
        token: 'mock-jwt-token-' + Date.now(),
        user: currentUser,
      }
    };
  }

  if (query.includes('register') || query.includes('Register')) {
    return {
      register: {
        token: 'mock-jwt-token-' + Date.now(),
        user: mockUsers['demo@mycelix.net'],
      }
    };
  }

  // Current user query - must be specific to avoid false matches
  if (query.match(/\bme\b/) || query.includes('currentUser')) {
    return { me: currentUser || mockUsers['demo@mycelix.net'] };
  }

  // Email queries
  if (query.includes('GetEmails') || query.includes('emails(')) {
    const folder = variables.folder || 'inbox';
    let emails = [...mockEmails];

    if (folder === 'starred') {
      emails = emails.filter(e => e.isStarred);
    } else if (folder === 'spam') {
      emails = emails.filter(e => e.trustScore < 0.2);
    } else if (folder === 'inbox') {
      emails = emails.filter(e => !e.isArchived && e.trustScore >= 0.2);
    }

    return {
      emails: {
        edges: emails.map(e => ({ node: e, cursor: e.id })),
        pageInfo: { hasNextPage: false, endCursor: emails[emails.length - 1]?.id },
      }
    };
  }

  if (query.includes('GetEmail') || query.includes('email(id:')) {
    const email = mockEmails.find(e => e.id === variables.id);
    return { email };
  }

  // Mutations
  if (query.includes('SendEmail') || query.includes('sendEmail')) {
    return {
      sendEmail: {
        id: 'email-new-' + Date.now(),
        threadId: 'thread-new-' + Date.now(),
        sentAt: new Date().toISOString(),
      }
    };
  }

  if (query.includes('MarkAsRead') || query.includes('markAsRead')) {
    return { markAsRead: { success: true, count: variables.ids?.length || 1 } };
  }

  if (query.includes('ArchiveEmails') || query.includes('archiveEmails')) {
    return { archiveEmails: { success: true, count: variables.ids?.length || 1 } };
  }

  if (query.includes('CreateAttestation') || query.includes('createAttestation') || query.includes('createTrustAttestation')) {
    return {
      createTrustAttestation: {
        id: 'att-new-' + Date.now(),
        type: variables.input?.context || 'general',
        score: variables.input?.trustLevel || 0.5,
        createdAt: new Date().toISOString(),
      }
    };
  }

  if (query.includes('updateEmail') || query.includes('UpdateEmail')) {
    const email = mockEmails.find(e => e.id === variables.id);
    if (email && variables.input) {
      Object.assign(email, variables.input);
    }
    return { updateEmail: email };
  }

  if (query.includes('deleteEmail') || query.includes('DeleteEmail')) {
    return { deleteEmail: true };
  }

  if (query.includes('createEmail') || query.includes('CreateEmail')) {
    const newEmail = {
      id: 'email-draft-' + Date.now(),
      ...variables.input,
      isRead: true,
      isStarred: false,
      receivedAt: new Date().toISOString(),
    };
    return { createEmail: newEmail };
  }

  if (query.includes('createContact') || query.includes('AddContact')) {
    return { createContact: { id: 'contact-new-' + Date.now() } };
  }

  if (query.includes('updateContact') || query.includes('blockContact') || query.includes('BlockContact')) {
    return { updateContact: { id: variables.id } };
  }

  if (query.includes('deleteContact') || query.includes('DeleteContact')) {
    return { deleteContact: true };
  }

  // Default: return empty
  console.log('Unhandled query:', query.substring(0, 200));
  return {};
}

// ============================================================================
// Routes
// ============================================================================

// GraphQL endpoint
app.post('/graphql', (req, res) => {
  const { query, variables } = req.body;

  console.log(`[GraphQL] ${query?.substring(0, 60)}...`);

  try {
    const data = handleGraphQL(query, variables);
    res.json({ data });
  } catch (error) {
    console.error('GraphQL error:', error);
    res.json({ errors: [{ message: error.message }] });
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', mode: 'mock', timestamp: new Date().toISOString() });
});

// WebSocket endpoint (simple polling fallback)
app.get('/ws/events', (req, res) => {
  res.json({ message: 'WebSocket not implemented in mock server - use polling' });
});

// ============================================================================
// Start Server
// ============================================================================

const PORT = process.env.PORT || 3001;

app.listen(PORT, '0.0.0.0', () => {
  console.log(`
  ╔════════════════════════════════════════════════════════════╗
  ║                                                            ║
  ║   🍄 Mycelix Mail Mock Server                              ║
  ║                                                            ║
  ║   GraphQL:  http://localhost:${PORT}/graphql                  ║
  ║   Health:   http://localhost:${PORT}/health                   ║
  ║                                                            ║
  ║   Demo Accounts:                                           ║
  ║   - demo@mycelix.net (any password)                        ║
  ║   - alice@mycelix.net (any password)                       ║
  ║                                                            ║
  ║   5 sample emails with trust scores 0.03 - 0.95            ║
  ║   4 contacts including 1 blocked spammer                   ║
  ║   Trust network with 6 nodes                               ║
  ║                                                            ║
  ╚════════════════════════════════════════════════════════════╝
  `);
});
