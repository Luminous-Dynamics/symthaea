// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mock credentials data
 */

export interface VerifiableCredential {
  '@context': string[];
  type: string[];
  issuer: string;
  issuanceDate: string;
  expirationDate?: string;
  credentialSubject: CredentialSubject;
  proof?: Proof;
}

export interface CredentialSubject {
  id: string;
  courseId: string;
  modelId?: string;
  score: number;
  scoreBand: string;
  skills: string[];
  completionDate: string;
  flContributions?: number;
}

export interface Proof {
  type: string;
  created: string;
  verificationMethod: string;
  proofPurpose: string;
  proofValue: string;
}

export const mockCredentials: VerifiableCredential[] = [
  {
    '@context': [
      'https://www.w3.org/2018/credentials/v1',
      'https://mycelix.network/credentials/v1',
    ],
    type: ['VerifiableCredential', 'EduAchievementCredential'],
    issuer: 'did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK',
    issuanceDate: '2025-11-15T14:30:00Z',
    credentialSubject: {
      id: 'did:key:z6MkrJVnaZkeFzdQyQssqGeLQEcnLZLqPaJhKP2CpBNGdE3F',
      courseId: 'rust-fundamentals-2025',
      modelId: 'rust-fundamentals-model-v1',
      score: 92.5,
      scoreBand: 'A',
      skills: [
        'Ownership and Borrowing',
        'Error Handling',
        'Concurrent Programming',
        'Type System Design',
      ],
      completionDate: '2025-11-15',
      flContributions: 5,
    },
    proof: {
      type: 'Ed25519Signature2020',
      created: '2025-11-15T14:30:00Z',
      verificationMethod: 'did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK#keys-1',
      proofPurpose: 'assertionMethod',
      proofValue:
        'z58DAdFfa9SkqZMVPxAQpic7ndSayn1PzZs6ZjWp1CktyGesjuTRelDqLhCXwxbFcbqB5u2hhKHZkSj8MUd2M5BXZ',
    },
  },
  {
    '@context': [
      'https://www.w3.org/2018/credentials/v1',
      'https://mycelix.network/credentials/v1',
    ],
    type: ['VerifiableCredential', 'EduAchievementCredential'],
    issuer: 'did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK',
    issuanceDate: '2025-10-20T10:15:00Z',
    credentialSubject: {
      id: 'did:key:z6MkrJVnaZkeFzdQyQssqGeLQEcnLZLqPaJhKP2CpBNGdE3F',
      courseId: 'spanish-beginner-2025',
      score: 88.0,
      scoreBand: 'B',
      skills: ['Conversational Spanish', 'Grammar Fundamentals', 'Pronunciation'],
      completionDate: '2025-10-20',
      flContributions: 3,
    },
    proof: {
      type: 'Ed25519Signature2020',
      created: '2025-10-20T10:15:00Z',
      verificationMethod: 'did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK#keys-1',
      proofPurpose: 'assertionMethod',
      proofValue:
        'z4oJ5fPzGxoZ8ybqbAJMiWt6Ps3vjW6MQZcqVf4r6PZdgzLqJM5UzSFdQxWNy8A3TvKcKjJgQp2kZvXyT9rGbPpQA',
    },
  },
  {
    '@context': [
      'https://www.w3.org/2018/credentials/v1',
      'https://mycelix.network/credentials/v1',
    ],
    type: ['VerifiableCredential', 'EduParticipationCredential'],
    issuer: 'did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK',
    issuanceDate: '2025-11-16T12:00:00Z',
    credentialSubject: {
      id: 'did:key:z6MkrJVnaZkeFzdQyQssqGeLQEcnLZLqPaJhKP2CpBNGdE3F',
      courseId: 'machine-learning-intro',
      score: 0,
      scoreBand: 'N/A',
      skills: ['Active Participation', 'Peer Collaboration'],
      completionDate: '2025-11-16',
      flContributions: 1,
    },
    proof: {
      type: 'Ed25519Signature2020',
      created: '2025-11-16T12:00:00Z',
      verificationMethod: 'did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK#keys-1',
      proofPurpose: 'assertionMethod',
      proofValue:
        'z2vXdLQpN5bJhW7QsKvPm3jRt8xY4aZcF9uGHp1qLbN6dMrK3wE5vT8pCsX2fJ4gY7iL1oQbR9aU6nM8kV3hD5eB',
    },
  },
];

export function getCredentialById(id: string): VerifiableCredential | undefined {
  return mockCredentials.find(c => c.credentialSubject.id === id);
}

export function getCredentialsByCourse(courseId: string): VerifiableCredential[] {
  return mockCredentials.filter(c => c.credentialSubject.courseId === courseId);
}

export function getCredentialsByType(type: string): VerifiableCredential[] {
  return mockCredentials.filter(c => c.type.includes(type));
}

/**
 * Verify a credential (mock implementation)
 */
export function verifyCredential(credential: VerifiableCredential): {
  valid: boolean;
  reason?: string;
} {
  // Mock verification logic
  if (!credential.proof) {
    return { valid: false, reason: 'No proof found' };
  }

  if (!credential.credentialSubject) {
    return { valid: false, reason: 'No credential subject found' };
  }

  // Check if expired
  if (credential.expirationDate) {
    const expiry = new Date(credential.expirationDate);
    if (expiry < new Date()) {
      return { valid: false, reason: 'Credential expired' };
    }
  }

  // All checks passed
  return { valid: true };
}

/**
 * Format score band as color
 */
export function getScoreBandColor(band: string): string {
  const colors: Record<string, string> = {
    A: '#10b981', // green
    B: '#3b82f6', // blue
    C: '#f59e0b', // amber
    D: '#f97316', // orange
    F: '#ef4444', // red
    'N/A': '#6b7280', // gray
  };
  return colors[band] || colors['N/A'];
}

/**
 * Get credential type icon
 */
export function getCredentialTypeIcon(type: string[]): string {
  if (type.includes('EduAchievementCredential')) return '🎓';
  if (type.includes('EduParticipationCredential')) return '✨';
  if (type.includes('EduCompletionCredential')) return '🏆';
  return '📜';
}
