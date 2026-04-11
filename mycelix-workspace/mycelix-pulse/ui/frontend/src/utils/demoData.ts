// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Demo Data Generators
 *
 * Generates realistic mock data for development and testing:
 * - Contacts with trust relationships
 * - Emails with epistemic metadata
 * - Attestations and credentials
 * - Trust network graph
 *
 * Use these when the backend is unavailable or for demos.
 */

import type { AssuranceLevel, RelationType, TrustEdge, VerifiableClaim } from '@/services/api';
import type { Email } from '@/types';

// Random utilities
const randomId = () => Math.random().toString(36).substring(2, 15);
const randomChoice = <T>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];
const randomInt = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;
const randomFloat = (min: number, max: number) => Math.random() * (max - min) + min;

// Sample data
const firstNames = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack', 'Kate', 'Leo', 'Maya', 'Noah', 'Olivia', 'Paul'];
const lastNames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Anderson', 'Taylor', 'Thomas', 'Moore', 'Jackson'];
const domains = ['example.com', 'company.io', 'startup.co', 'enterprise.org', 'dev.tech', 'agency.net'];
const companies = ['Acme Corp', 'TechStart', 'Innovation Labs', 'Digital Solutions', 'Cloud Systems', 'Data Dynamics', 'Future Tech', 'Smart Solutions'];

const subjectPrefixes = ['Re:', 'Fwd:', 'Urgent:', 'Follow up:', 'Question about', 'Update on', 'Meeting:', 'Proposal:', 'Feedback on'];
const subjectTopics = ['Q4 Planning', 'Project Alpha', 'Budget Review', 'Team Sync', 'Client Meeting', 'Product Launch', 'Design Review', 'Tech Stack', 'Hiring Update', 'Strategy Session'];

const bodyTemplates = [
  'Hi {name},\n\nHope this email finds you well. I wanted to follow up on our previous discussion about {topic}.\n\nLet me know your thoughts.\n\nBest,\n{sender}',
  'Hello,\n\nPlease find attached the documents we discussed. The deadline is {date}.\n\nThanks,\n{sender}',
  'Hey {name},\n\nQuick question - when is the best time to schedule our call about {topic}?\n\nCheers,\n{sender}',
  'Hi team,\n\nJust a reminder about the upcoming {topic} meeting on {date}. Please come prepared with your updates.\n\nBest,\n{sender}',
  'Dear {name},\n\nThank you for your interest in {topic}. I\'d be happy to discuss this further.\n\nPlease let me know your availability.\n\nRegards,\n{sender}',
];

const intentTypes = ['question', 'request', 'information', 'action_required', 'follow_up', 'scheduling', 'introduction'];
const relationshipTypes: RelationType[] = ['direct_trust', 'introduction', 'organization_member', 'credential_issuer', 'transitive_trust', 'vouch'];
const assuranceLevels: AssuranceLevel[] = ['e0_anonymous', 'e1_verified_email', 'e2_gitcoin_passport', 'e3_multi_factor', 'e4_constitutional'];

// Generate a random contact
export interface DemoContact {
  did: string;
  email: string;
  name: string;
  company?: string;
  trustScore: number;
  assuranceLevel: AssuranceLevel;
  pathLength: number;
  relationship: RelationType;
  lastContact: string;
  verified: boolean;
}

export function generateContact(options?: Partial<DemoContact>): DemoContact {
  const firstName = randomChoice(firstNames);
  const lastName = randomChoice(lastNames);
  const name = `${firstName} ${lastName}`;
  const email = `${firstName.toLowerCase()}.${lastName.toLowerCase()}@${randomChoice(domains)}`;
  const did = `did:mycelix:${randomId()}`;

  return {
    did,
    email,
    name,
    company: Math.random() > 0.3 ? randomChoice(companies) : undefined,
    trustScore: randomFloat(0.2, 1.0),
    assuranceLevel: randomChoice(assuranceLevels),
    pathLength: randomInt(1, 4),
    relationship: randomChoice(relationshipTypes),
    lastContact: new Date(Date.now() - randomInt(0, 30 * 24 * 60 * 60 * 1000)).toISOString(),
    verified: Math.random() > 0.4,
    ...options,
  };
}

export function generateContacts(count: number): DemoContact[] {
  return Array.from({ length: count }, () => generateContact());
}

// Generate a random email
export interface DemoEmail extends Email {
  senderDid: string;
  epistemicTier: number;
  trustScore: number;
  assuranceLevel: AssuranceLevel;
  claims: VerifiableClaim[];
  aiInsights: {
    intent: string;
    priority: number;
    summary?: string;
    actionItems?: string[];
  };
}

export function generateEmail(contacts: DemoContact[], options?: Partial<DemoEmail>): DemoEmail {
  const sender = randomChoice(contacts);
  const recipients = [randomChoice(contacts)];
  const cc = Math.random() > 0.7 ? [randomChoice(contacts)] : undefined;

  const hasPrefix = Math.random() > 0.5;
  const subject = hasPrefix
    ? `${randomChoice(subjectPrefixes)} ${randomChoice(subjectTopics)}`
    : randomChoice(subjectTopics);

  const bodyTemplate = randomChoice(bodyTemplates);
  const body = bodyTemplate
    .replace('{name}', recipients[0].name.split(' ')[0])
    .replace('{topic}', randomChoice(subjectTopics))
    .replace('{date}', new Date(Date.now() + randomInt(1, 14) * 24 * 60 * 60 * 1000).toLocaleDateString())
    .replace('{sender}', sender.name.split(' ')[0]);

  const date = new Date(Date.now() - randomInt(0, 7 * 24 * 60 * 60 * 1000)).toISOString();
  const isRead = Math.random() > 0.3;
  const isStarred = Math.random() > 0.8;

  return {
    id: randomId(),
    accountId: 'demo-account',
    subject,
    from: { address: sender.email, name: sender.name },
    to: recipients.map((r) => ({ address: r.email, name: r.name })),
    cc: cc?.map((r) => ({ address: r.email, name: r.name })),
    date,
    bodyText: body,
    bodyHtml: `<p>${body.replace(/\n/g, '</p><p>')}</p>`,
    isRead,
    isStarred,
    attachments: Math.random() > 0.8 ? [{ id: randomId(), filename: 'document.pdf', size: randomInt(10000, 500000), mimeType: 'application/pdf' }] : [],
    size: randomInt(1000, 50000),
    senderDid: sender.did,
    epistemicTier: randomInt(0, 4),
    trustScore: sender.trustScore,
    assuranceLevel: sender.assuranceLevel,
    claims: Math.random() > 0.6 ? generateClaims(randomInt(1, 3)) : [],
    aiInsights: {
      intent: randomChoice(intentTypes),
      priority: randomFloat(0.3, 1.0),
      summary: Math.random() > 0.5 ? `This email is about ${randomChoice(subjectTopics).toLowerCase()}.` : undefined,
      actionItems: Math.random() > 0.7 ? ['Review the attached document', 'Schedule follow-up meeting'] : undefined,
    },
    ...options,
  };
}

export function generateEmails(count: number, contacts?: DemoContact[]): DemoEmail[] {
  const demoContacts = contacts || generateContacts(10);
  return Array.from({ length: count }, () => generateEmail(demoContacts));
}

// Generate a random claim
export function generateClaim(): VerifiableClaim {
  const proofTypes = ['email_verification', 'github_verification', 'gitcoin_passport', 'organization_membership'];
  const proofType = randomChoice(proofTypes);

  return {
    id: randomId(),
    type: proofType as any,
    issuer: `did:mycelix:issuer-${randomId()}`,
    subject: `did:mycelix:subject-${randomId()}`,
    issued_at: new Date(Date.now() - randomInt(30, 365) * 24 * 60 * 60 * 1000).toISOString(),
    expires_at: new Date(Date.now() + randomInt(30, 365) * 24 * 60 * 60 * 1000).toISOString(),
    proof: {
      type: 'cryptographic_signature',
      signature: `sig_${randomId()}`,
      verified: Math.random() > 0.2,
    },
    assurance_level: randomChoice(assuranceLevels),
    display_name: proofType.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
  };
}

export function generateClaims(count: number): VerifiableClaim[] {
  return Array.from({ length: count }, () => generateClaim());
}

// Generate a trust edge
export function generateTrustEdge(fromDid: string, toDid: string): TrustEdge {
  return {
    from_did: fromDid,
    to_did: toDid,
    relationship: randomChoice(relationshipTypes),
    trust_score: randomFloat(0.3, 1.0),
    reason: Math.random() > 0.5 ? randomChoice(['Professional colleague', 'Personal friend', 'Community member', 'Business partner']) : undefined,
    established_at: new Date(Date.now() - randomInt(30, 365) * 24 * 60 * 60 * 1000).toISOString(),
    decays_at: Math.random() > 0.7 ? new Date(Date.now() + randomInt(30, 365) * 24 * 60 * 60 * 1000).toISOString() : undefined,
  };
}

// Generate a trust network
export interface DemoTrustNetwork {
  userDid: string;
  contacts: DemoContact[];
  edges: TrustEdge[];
  stats: {
    totalConnections: number;
    directTrust: number;
    transitiveTrust: number;
    avgTrustScore: number;
  };
}

export function generateTrustNetwork(userDid: string, size: number = 15): DemoTrustNetwork {
  const contacts = generateContacts(size);
  const edges: TrustEdge[] = [];

  // Create direct trust edges from user
  const directCount = Math.min(5, contacts.length);
  for (let i = 0; i < directCount; i++) {
    edges.push(generateTrustEdge(userDid, contacts[i].did));
    contacts[i].pathLength = 1;
  }

  // Create transitive edges
  for (let i = directCount; i < contacts.length; i++) {
    const introducer = contacts[randomInt(0, directCount - 1)];
    edges.push(generateTrustEdge(introducer.did, contacts[i].did));
    contacts[i].pathLength = 2;
  }

  // Add some inter-contact edges
  for (let i = 0; i < size / 2; i++) {
    const from = randomChoice(contacts);
    const to = randomChoice(contacts.filter((c) => c.did !== from.did));
    if (to) {
      edges.push(generateTrustEdge(from.did, to.did));
    }
  }

  const avgTrustScore = edges.reduce((sum, e) => sum + e.trust_score, 0) / edges.length;

  return {
    userDid,
    contacts,
    edges,
    stats: {
      totalConnections: contacts.length,
      directTrust: directCount,
      transitiveTrust: contacts.length - directCount,
      avgTrustScore,
    },
  };
}

// Generate attestation history
export interface DemoAttestation {
  id: string;
  attestorDid: string;
  attestorName: string;
  subjectDid: string;
  subjectName: string;
  relationship: RelationType;
  trustScore: number;
  message: string;
  createdAt: string;
  expiresAt?: string;
  revoked: boolean;
}

export function generateAttestation(contacts: DemoContact[]): DemoAttestation {
  const attestor = randomChoice(contacts);
  const subject = randomChoice(contacts.filter((c) => c.did !== attestor.did));

  return {
    id: randomId(),
    attestorDid: attestor.did,
    attestorName: attestor.name,
    subjectDid: subject?.did || `did:mycelix:${randomId()}`,
    subjectName: subject?.name || 'Unknown',
    relationship: randomChoice(relationshipTypes),
    trustScore: randomFloat(0.5, 1.0),
    message: randomChoice([
      'Trusted colleague',
      'Long-time friend',
      'Professional reference',
      'Community member in good standing',
      'Verified business partner',
    ]),
    createdAt: new Date(Date.now() - randomInt(1, 365) * 24 * 60 * 60 * 1000).toISOString(),
    expiresAt: Math.random() > 0.3 ? new Date(Date.now() + randomInt(30, 365) * 24 * 60 * 60 * 1000).toISOString() : undefined,
    revoked: Math.random() > 0.95,
  };
}

export function generateAttestations(count: number, contacts?: DemoContact[]): DemoAttestation[] {
  const demoContacts = contacts || generateContacts(10);
  return Array.from({ length: count }, () => generateAttestation(demoContacts));
}

// Demo data hook
export function useDemoData() {
  const contacts = generateContacts(20);
  const emails = generateEmails(50, contacts);
  const network = generateTrustNetwork('did:mycelix:self', 15);
  const attestations = generateAttestations(30, contacts);

  return {
    contacts,
    emails,
    network,
    attestations,
    regenerate: () => ({
      contacts: generateContacts(20),
      emails: generateEmails(50),
      network: generateTrustNetwork('did:mycelix:self', 15),
      attestations: generateAttestations(30),
    }),
  };
}

export default {
  generateContact,
  generateContacts,
  generateEmail,
  generateEmails,
  generateClaim,
  generateClaims,
  generateTrustEdge,
  generateTrustNetwork,
  generateAttestation,
  generateAttestations,
  useDemoData,
};
