// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Holochain Client Integration
 *
 * Client for interacting with Holochain DNA for trust operations:
 * - Attestation management
 * - Trust score calculation
 * - Profile management
 * - Introduction requests
 * - Trust path discovery
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// ============================================================================
// Types
// ============================================================================

export interface AgentPubKey {
  hash: Uint8Array;
  toString(): string;
}

export interface ActionHash {
  hash: Uint8Array;
  toString(): string;
}

export interface TrustAttestation {
  id: string;
  attester: string;
  subject: string;
  attestationType: 'professional' | 'personal' | 'verified' | 'vouched';
  score: number;
  comment?: string;
  evidence?: string;
  createdAt: number;
  expiresAt?: number;
  revoked: boolean;
}

export interface Profile {
  agentPubKey: string;
  email: string;
  displayName?: string;
  avatarUrl?: string;
  bio?: string;
  publicKey?: string;
  metadata: Record<string, unknown>;
  createdAt: number;
  updatedAt: number;
}

export interface TrustScore {
  subject: string;
  score: number;
  directScore: number;
  networkScore: number;
  behaviorScore: number;
  attestationCount: number;
  pathLength?: number;
  lastCalculated: number;
}

export interface TrustPath {
  found: boolean;
  length: number;
  path: Array<{
    agent: string;
    email: string;
    trustScore: number;
  }>;
}

export interface IntroductionRequest {
  id: string;
  requester: string;
  target: string;
  introducer: string;
  message: string;
  status: 'pending' | 'accepted' | 'rejected';
  createdAt: number;
}

export interface HolochainConfig {
  appId: string;
  adminUrl: string;
  appUrl: string;
  installedAppId?: string;
}

// ============================================================================
// Holochain Client Store
// ============================================================================

interface HolochainState {
  isConnected: boolean;
  isInitialized: boolean;
  agentPubKey: string | null;
  profile: Profile | null;
  attestations: TrustAttestation[];
  trustScores: Map<string, TrustScore>;
  pendingIntroductions: IntroductionRequest[];
  error: string | null;
}

interface HolochainActions {
  // Connection
  connect: (config: HolochainConfig) => Promise<void>;
  disconnect: () => void;

  // Profile
  getProfile: (agentPubKey?: string) => Promise<Profile | null>;
  setProfile: (profile: Partial<Profile>) => Promise<Profile>;

  // Attestations
  createAttestation: (attestation: Omit<TrustAttestation, 'id' | 'createdAt' | 'revoked' | 'attester'>) => Promise<TrustAttestation>;
  getAttestationsBy: (attester: string) => Promise<TrustAttestation[]>;
  getAttestationsAbout: (subject: string) => Promise<TrustAttestation[]>;
  revokeAttestation: (id: string) => Promise<void>;

  // Trust Scores
  getTrustScore: (email: string) => Promise<TrustScore>;
  calculateTrustScore: (subject: string) => Promise<TrustScore>;
  findTrustPath: (from: string, to: string) => Promise<TrustPath>;

  // Introductions
  requestIntroduction: (target: string, introducer: string, message: string) => Promise<IntroductionRequest>;
  respondToIntroduction: (id: string, accept: boolean) => Promise<void>;
  getPendingIntroductions: () => Promise<IntroductionRequest[]>;
}

type HolochainStore = HolochainState & HolochainActions;

// ============================================================================
// Mock Holochain Client (for development)
// ============================================================================

class MockHolochainClient {
  private profiles: Map<string, Profile> = new Map();
  private attestations: TrustAttestation[] = [];
  private introductions: IntroductionRequest[] = [];
  private agentPubKey: string;

  constructor() {
    this.agentPubKey = this.generateMockPubKey();
    this.seedMockData();
  }

  private generateMockPubKey(): string {
    return 'uhCAk' + Array.from({ length: 48 }, () =>
      Math.floor(Math.random() * 16).toString(16)
    ).join('');
  }

  private seedMockData(): void {
    // Create mock profiles
    const mockEmails = [
      'alice@example.com',
      'bob@company.org',
      'charlie@trusted.net',
      'diana@partner.io',
    ];

    mockEmails.forEach((email, i) => {
      const pubKey = this.generateMockPubKey();
      this.profiles.set(pubKey, {
        agentPubKey: pubKey,
        email,
        displayName: email.split('@')[0].charAt(0).toUpperCase() + email.split('@')[0].slice(1),
        metadata: {},
        createdAt: Date.now() - (i * 86400000),
        updatedAt: Date.now(),
      });
    });

    // Create mock attestations
    const profileList = Array.from(this.profiles.values());
    for (let i = 0; i < profileList.length - 1; i++) {
      this.attestations.push({
        id: `attest-${i}`,
        attester: profileList[i].agentPubKey,
        subject: profileList[i + 1].email,
        attestationType: 'professional',
        score: 0.7 + Math.random() * 0.3,
        comment: 'Trusted colleague',
        createdAt: Date.now() - Math.random() * 86400000 * 30,
        revoked: false,
      });
    }
  }

  getAgentPubKey(): string {
    return this.agentPubKey;
  }

  async getProfile(pubKey?: string): Promise<Profile | null> {
    const key = pubKey || this.agentPubKey;
    return this.profiles.get(key) || null;
  }

  async setProfile(profile: Partial<Profile>): Promise<Profile> {
    const existing = this.profiles.get(this.agentPubKey) || {
      agentPubKey: this.agentPubKey,
      email: '',
      metadata: {},
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    const updated: Profile = {
      ...existing,
      ...profile,
      agentPubKey: this.agentPubKey,
      updatedAt: Date.now(),
    };

    this.profiles.set(this.agentPubKey, updated);
    return updated;
  }

  async createAttestation(
    attestation: Omit<TrustAttestation, 'id' | 'createdAt' | 'revoked' | 'attester'>
  ): Promise<TrustAttestation> {
    const newAttestation: TrustAttestation = {
      ...attestation,
      id: `attest-${Date.now()}`,
      attester: this.agentPubKey,
      createdAt: Date.now(),
      revoked: false,
    };

    this.attestations.push(newAttestation);
    return newAttestation;
  }

  async getAttestationsBy(attester: string): Promise<TrustAttestation[]> {
    return this.attestations.filter(a => a.attester === attester && !a.revoked);
  }

  async getAttestationsAbout(subject: string): Promise<TrustAttestation[]> {
    return this.attestations.filter(a => a.subject === subject && !a.revoked);
  }

  async revokeAttestation(id: string): Promise<void> {
    const attestation = this.attestations.find(a => a.id === id);
    if (attestation && attestation.attester === this.agentPubKey) {
      attestation.revoked = true;
    }
  }

  async calculateTrustScore(subject: string): Promise<TrustScore> {
    const attestations = await this.getAttestationsAbout(subject);

    if (attestations.length === 0) {
      return {
        subject,
        score: 0,
        directScore: 0,
        networkScore: 0,
        behaviorScore: 0,
        attestationCount: 0,
        lastCalculated: Date.now(),
      };
    }

    // Calculate weighted average
    const directScore = attestations.reduce((sum, a) => sum + a.score, 0) / attestations.length;

    // Simulate network score (would traverse trust graph in real implementation)
    const networkScore = Math.min(directScore * 1.1, 1);

    // Behavior score (would analyze interaction patterns)
    const behaviorScore = 0.5 + Math.random() * 0.3;

    // Combined score with weights
    const score = directScore * 0.5 + networkScore * 0.3 + behaviorScore * 0.2;

    return {
      subject,
      score: Math.min(score, 1),
      directScore,
      networkScore,
      behaviorScore,
      attestationCount: attestations.length,
      lastCalculated: Date.now(),
    };
  }

  async findTrustPath(from: string, to: string): Promise<TrustPath> {
    // Simplified BFS for trust path (real implementation would use Holochain DHT)
    const visited = new Set<string>();
    const queue: Array<{ agent: string; path: string[] }> = [{ agent: from, path: [from] }];

    while (queue.length > 0) {
      const { agent, path } = queue.shift()!;

      if (agent === to) {
        return {
          found: true,
          length: path.length - 1,
          path: path.map(a => ({
            agent: a,
            email: this.profiles.get(a)?.email || a,
            trustScore: 0.8,
          })),
        };
      }

      if (visited.has(agent)) continue;
      visited.add(agent);

      // Find attestations from this agent
      const attestations = this.attestations.filter(
        a => a.attester === agent && !a.revoked
      );

      for (const attestation of attestations) {
        // Find profile by email
        for (const [pubKey, profile] of this.profiles) {
          if (profile.email === attestation.subject && !visited.has(pubKey)) {
            queue.push({ agent: pubKey, path: [...path, pubKey] });
          }
        }
      }
    }

    return { found: false, length: -1, path: [] };
  }

  async requestIntroduction(
    target: string,
    introducer: string,
    message: string
  ): Promise<IntroductionRequest> {
    const request: IntroductionRequest = {
      id: `intro-${Date.now()}`,
      requester: this.agentPubKey,
      target,
      introducer,
      message,
      status: 'pending',
      createdAt: Date.now(),
    };

    this.introductions.push(request);
    return request;
  }

  async respondToIntroduction(id: string, accept: boolean): Promise<void> {
    const intro = this.introductions.find(i => i.id === id);
    if (intro && intro.introducer === this.agentPubKey) {
      intro.status = accept ? 'accepted' : 'rejected';
    }
  }

  async getPendingIntroductions(): Promise<IntroductionRequest[]> {
    return this.introductions.filter(
      i => i.status === 'pending' &&
        (i.requester === this.agentPubKey || i.introducer === this.agentPubKey)
    );
  }
}

// ============================================================================
// Zustand Store
// ============================================================================

let mockClient: MockHolochainClient | null = null;

export const useHolochainClient = create<HolochainStore>()(
  persist(
    (set, get) => ({
      // State
      isConnected: false,
      isInitialized: false,
      agentPubKey: null,
      profile: null,
      attestations: [],
      trustScores: new Map(),
      pendingIntroductions: [],
      error: null,

      // Connection
      connect: async (config: HolochainConfig) => {
        try {
          // In development, use mock client
          // In production, would connect to actual Holochain conductor
          mockClient = new MockHolochainClient();

          set({
            isConnected: true,
            isInitialized: true,
            agentPubKey: mockClient.getAgentPubKey(),
            error: null,
          });

          // Load initial profile
          const profile = await mockClient.getProfile();
          if (profile) {
            set({ profile });
          }
        } catch (error) {
          set({ error: (error as Error).message });
          throw error;
        }
      },

      disconnect: () => {
        mockClient = null;
        set({
          isConnected: false,
          agentPubKey: null,
          profile: null,
          attestations: [],
          trustScores: new Map(),
          pendingIntroductions: [],
        });
      },

      // Profile
      getProfile: async (agentPubKey?: string) => {
        if (!mockClient) throw new Error('Not connected');
        return mockClient.getProfile(agentPubKey);
      },

      setProfile: async (profile: Partial<Profile>) => {
        if (!mockClient) throw new Error('Not connected');
        const updated = await mockClient.setProfile(profile);
        set({ profile: updated });
        return updated;
      },

      // Attestations
      createAttestation: async (attestation) => {
        if (!mockClient) throw new Error('Not connected');
        const created = await mockClient.createAttestation(attestation);
        set(state => ({
          attestations: [...state.attestations, created],
        }));
        return created;
      },

      getAttestationsBy: async (attester: string) => {
        if (!mockClient) throw new Error('Not connected');
        return mockClient.getAttestationsBy(attester);
      },

      getAttestationsAbout: async (subject: string) => {
        if (!mockClient) throw new Error('Not connected');
        return mockClient.getAttestationsAbout(subject);
      },

      revokeAttestation: async (id: string) => {
        if (!mockClient) throw new Error('Not connected');
        await mockClient.revokeAttestation(id);
        set(state => ({
          attestations: state.attestations.map(a =>
            a.id === id ? { ...a, revoked: true } : a
          ),
        }));
      },

      // Trust Scores
      getTrustScore: async (email: string) => {
        const cached = get().trustScores.get(email);
        if (cached && Date.now() - cached.lastCalculated < 300000) {
          return cached;
        }
        return get().calculateTrustScore(email);
      },

      calculateTrustScore: async (subject: string) => {
        if (!mockClient) throw new Error('Not connected');
        const score = await mockClient.calculateTrustScore(subject);
        set(state => {
          const newScores = new Map(state.trustScores);
          newScores.set(subject, score);
          return { trustScores: newScores };
        });
        return score;
      },

      findTrustPath: async (from: string, to: string) => {
        if (!mockClient) throw new Error('Not connected');
        return mockClient.findTrustPath(from, to);
      },

      // Introductions
      requestIntroduction: async (target, introducer, message) => {
        if (!mockClient) throw new Error('Not connected');
        const request = await mockClient.requestIntroduction(target, introducer, message);
        set(state => ({
          pendingIntroductions: [...state.pendingIntroductions, request],
        }));
        return request;
      },

      respondToIntroduction: async (id, accept) => {
        if (!mockClient) throw new Error('Not connected');
        await mockClient.respondToIntroduction(id, accept);
        set(state => ({
          pendingIntroductions: state.pendingIntroductions.map(i =>
            i.id === id ? { ...i, status: accept ? 'accepted' : 'rejected' } : i
          ),
        }));
      },

      getPendingIntroductions: async () => {
        if (!mockClient) throw new Error('Not connected');
        const intros = await mockClient.getPendingIntroductions();
        set({ pendingIntroductions: intros });
        return intros;
      },
    }),
    {
      name: 'mycelix-holochain',
      partialize: (state) => ({
        agentPubKey: state.agentPubKey,
        profile: state.profile,
      }),
    }
  )
);

// ============================================================================
// React Hooks
// ============================================================================

import { useEffect, useState, useCallback } from 'react';

export function useTrustScore(email: string) {
  const { getTrustScore, isConnected } = useHolochainClient();
  const [score, setScore] = useState<TrustScore | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!isConnected || !email) return;

    setLoading(true);
    getTrustScore(email)
      .then(setScore)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [email, isConnected, getTrustScore]);

  return { score, loading, error };
}

export function useAttestations(options: { by?: string; about?: string }) {
  const { getAttestationsBy, getAttestationsAbout, isConnected } = useHolochainClient();
  const [attestations, setAttestations] = useState<TrustAttestation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const refresh = useCallback(async () => {
    if (!isConnected) return;

    setLoading(true);
    try {
      if (options.by) {
        setAttestations(await getAttestationsBy(options.by));
      } else if (options.about) {
        setAttestations(await getAttestationsAbout(options.about));
      }
    } catch (e) {
      setError(e as Error);
    } finally {
      setLoading(false);
    }
  }, [isConnected, options.by, options.about, getAttestationsBy, getAttestationsAbout]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { attestations, loading, error, refresh };
}

export function useTrustPath(from: string, to: string) {
  const { findTrustPath, isConnected } = useHolochainClient();
  const [path, setPath] = useState<TrustPath | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!isConnected || !from || !to) return;

    setLoading(true);
    findTrustPath(from, to)
      .then(setPath)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [from, to, isConnected, findTrustPath]);

  return { path, loading, error };
}

export default useHolochainClient;
