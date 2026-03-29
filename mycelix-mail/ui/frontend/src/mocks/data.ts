// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Mock data for Storybook development
// Provides realistic sample data for component stories

export interface MockEmail {
  id: string;
  subject: string;
  from: MockContact;
  to: MockContact[];
  body: string;
  bodyPreview: string;
  date: string;
  isRead: boolean;
  isStarred: boolean;
  labels: string[];
  trustScore: number;
  attachments?: MockAttachment[];
}

export interface MockContact {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  trustScore: number;
}

export interface MockAttachment {
  id: string;
  name: string;
  size: number;
  type: string;
}

// Sample contacts with varying trust levels
export const mockContacts: MockContact[] = [
  {
    id: 'contact-1',
    name: 'Alice Chen',
    email: 'alice@mycelix.network',
    trustScore: 0.95,
  },
  {
    id: 'contact-2',
    name: 'Bob Wilson',
    email: 'bob@trusted-org.com',
    trustScore: 0.82,
  },
  {
    id: 'contact-3',
    name: 'Carol Martinez',
    email: 'carol@example.com',
    trustScore: 0.65,
  },
  {
    id: 'contact-4',
    name: 'David Unknown',
    email: 'david@suspicious-domain.xyz',
    trustScore: 0.35,
  },
  {
    id: 'contact-5',
    name: 'Unknown Sender',
    email: 'random@new-contact.net',
    trustScore: 0.1,
  },
];

export const mockCurrentUser: MockContact = {
  id: 'user-1',
  name: 'Test User',
  email: 'test@mycelix.mail',
  trustScore: 1.0,
};

// Sample emails
export const mockEmails: MockEmail[] = [
  {
    id: 'email-1',
    subject: 'Welcome to the Mycelix Network',
    from: mockContacts[0],
    to: [mockCurrentUser],
    body: `<p>Hello!</p>
<p>Welcome to the Mycelix decentralized email network. Your account has been set up and your trust attestations are ready.</p>
<p>Here are some things you can do:</p>
<ul>
  <li>Build your trust network by connecting with verified contacts</li>
  <li>Enable post-quantum encryption for sensitive communications</li>
  <li>Explore the AI-powered features for email management</li>
</ul>
<p>Best regards,<br>Alice</p>`,
    bodyPreview: 'Welcome to the Mycelix decentralized email network. Your account has been set up...',
    date: new Date(Date.now() - 1000 * 60 * 30).toISOString(), // 30 min ago
    isRead: false,
    isStarred: true,
    labels: ['important', 'mycelix'],
    trustScore: 0.95,
  },
  {
    id: 'email-2',
    subject: 'Re: Project Update - Q1 Review',
    from: mockContacts[1],
    to: [mockCurrentUser],
    body: `<p>Thanks for the update!</p>
<p>The quarterly numbers look great. Let's schedule a follow-up call to discuss the roadmap.</p>
<p>Best,<br>Bob</p>`,
    bodyPreview: 'Thanks for the update! The quarterly numbers look great...',
    date: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(), // 2 hours ago
    isRead: true,
    isStarred: false,
    labels: ['work'],
    trustScore: 0.82,
    attachments: [
      { id: 'att-1', name: 'Q1-Report.pdf', size: 2450000, type: 'application/pdf' },
    ],
  },
  {
    id: 'email-3',
    subject: 'Meeting invitation: Team sync',
    from: mockContacts[2],
    to: [mockCurrentUser],
    body: `<p>Hi there,</p>
<p>Would you be available for a quick sync tomorrow at 2pm?</p>
<p>Let me know what works for you.</p>
<p>Carol</p>`,
    bodyPreview: 'Would you be available for a quick sync tomorrow at 2pm?',
    date: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(), // 1 day ago
    isRead: true,
    isStarred: false,
    labels: ['meetings'],
    trustScore: 0.65,
  },
  {
    id: 'email-4',
    subject: 'You have won a prize!!!',
    from: mockContacts[3],
    to: [mockCurrentUser],
    body: `<p>Congratulations! You have been selected...</p>`,
    bodyPreview: 'Congratulations! You have been selected...',
    date: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString(), // 2 days ago
    isRead: false,
    isStarred: false,
    labels: ['suspicious'],
    trustScore: 0.35,
  },
  {
    id: 'email-5',
    subject: 'Invoice #12345',
    from: mockContacts[4],
    to: [mockCurrentUser],
    body: `<p>Please find attached invoice...</p>`,
    bodyPreview: 'Please find attached invoice...',
    date: new Date(Date.now() - 1000 * 60 * 60 * 72).toISOString(), // 3 days ago
    isRead: false,
    isStarred: false,
    labels: [],
    trustScore: 0.1,
    attachments: [
      { id: 'att-2', name: 'invoice.pdf', size: 45000, type: 'application/pdf' },
    ],
  },
];

// Trust attestation mock data
export interface MockAttestation {
  id: string;
  from: string;
  to: string;
  trustLevel: number;
  context: string;
  timestamp: string;
  signature: string;
}

export const mockAttestations: MockAttestation[] = [
  {
    id: 'att-1',
    from: mockContacts[0].id,
    to: mockCurrentUser.id,
    trustLevel: 0.9,
    context: 'professional',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24 * 30).toISOString(),
    signature: 'dilithium-sig-abc123...',
  },
  {
    id: 'att-2',
    from: mockContacts[1].id,
    to: mockCurrentUser.id,
    trustLevel: 0.85,
    context: 'verified',
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24 * 15).toISOString(),
    signature: 'dilithium-sig-def456...',
  },
];

// Helper to get a random email
export const getRandomEmail = (): MockEmail => {
  return mockEmails[Math.floor(Math.random() * mockEmails.length)];
};

// Helper to get emails by trust level
export const getEmailsByTrustLevel = (minTrust: number): MockEmail[] => {
  return mockEmails.filter(e => e.trustScore >= minTrust);
};

// Helper to format file size
export const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

// Helper to format relative time
export const formatRelativeTime = (date: string): string => {
  const now = Date.now();
  const then = new Date(date).getTime();
  const diff = now - then;

  const minutes = Math.floor(diff / (1000 * 60));
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));

  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return new Date(date).toLocaleDateString();
};
