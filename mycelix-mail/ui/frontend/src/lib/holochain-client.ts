// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Holochain Trust Network Client
 *
 * Real integration with Holochain conductor for decentralized trust attestations.
 * Connects to the Mycelix DNA for trust graph operations.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// ============================================
// Types
// ============================================

export interface AgentPubKey {
  hash: Uint8Array;
  toString(): string;
}

export interface ActionHash {
  hash: Uint8Array;
  toString(): string;
}

export interface EntryHash {
  hash: Uint8Array;
  toString(): string;
}

export interface HolochainConfig {
  conductorUrl: string;
  installedAppId: string;
  cellId?: [Uint8Array, Uint8Array]; // [DnaHash, AgentPubKey]
}

export interface TrustAttestation {
  id: string;
  attestor: string;
  subject: string;
  trustLevel: number; // 0-100
  category: 'identity' | 'professional' | 'social' | 'domain' | 'content';
  claim: string;
  evidence?: string;
  expiresAt?: string;
  createdAt: string;
  signature: string;
  actionHash?: string;
}

export interface TrustPath {
  from: string;
  to: string;
  path: TrustPathNode[];
  aggregateScore: number;
  decay: number;
}

export interface TrustPathNode {
  agent: string;
  attestation: TrustAttestation;
  contribution: number;
}

export interface TrustQuery {
  subject: string;
  maxPathLength?: number;
  minTrustLevel?: number;
  categories?: string[];
  includeExpired?: boolean;
}

export interface TrustGraphNode {
  agentPubKey: string;
  displayName?: string;
  trustScore: number;
  attestationsGiven: number;
  attestationsReceived: number;
  connections: string[];
}

export interface TrustGraphEdge {
  from: string;
  to: string;
  weight: number;
  attestations: TrustAttestation[];
}

export interface TrustNetwork {
  nodes: Map<string, TrustGraphNode>;
  edges: Map<string, TrustGraphEdge>;
  myAgentPubKey: string;
}

// ============================================
// Holochain Connection Manager
// ============================================

type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

interface HolochainState {
  status: ConnectionStatus;
  config: HolochainConfig | null;
  agentPubKey: string | null;
  error: string | null;
  network: TrustNetwork | null;
  pendingAttestations: TrustAttestation[];
}

interface HolochainActions {
  connect: (config: HolochainConfig) => Promise<void>;
  disconnect: () => void;
  setError: (error: string | null) => void;
  setNetwork: (network: TrustNetwork) => void;
  addPendingAttestation: (attestation: TrustAttestation) => void;
  removePendingAttestation: (id: string) => void;
}

export const useHolochainStore = create<HolochainState & HolochainActions>()(
  persist(
    (set, get) => ({
      status: 'disconnected',
      config: null,
      agentPubKey: null,
      error: null,
      network: null,
      pendingAttestations: [],

      connect: async (config: HolochainConfig) => {
        set({ status: 'connecting', config, error: null });

        try {
          const client = await HolochainClient.connect(config);
          const agentPubKey = await client.getAgentPubKey();

          set({
            status: 'connected',
            agentPubKey,
            error: null,
          });

          // Sync any pending attestations
          const pending = get().pendingAttestations;
          for (const attestation of pending) {
            try {
              await client.createAttestation(attestation);
              get().removePendingAttestation(attestation.id);
            } catch (err) {
              console.error('Failed to sync attestation:', attestation.id, err);
            }
          }
        } catch (err) {
          set({
            status: 'error',
            error: err instanceof Error ? err.message : 'Connection failed',
          });
        }
      },

      disconnect: () => {
        HolochainClient.disconnect();
        set({
          status: 'disconnected',
          agentPubKey: null,
          error: null,
        });
      },

      setError: (error) => set({ error }),

      setNetwork: (network) => set({ network }),

      addPendingAttestation: (attestation) => {
        set((state) => ({
          pendingAttestations: [...state.pendingAttestations, attestation],
        }));
      },

      removePendingAttestation: (id) => {
        set((state) => ({
          pendingAttestations: state.pendingAttestations.filter((a) => a.id !== id),
        }));
      },
    }),
    {
      name: 'holochain-connection',
      partialize: (state) => ({
        config: state.config,
        pendingAttestations: state.pendingAttestations,
      }),
    }
  )
);

// ============================================
// Holochain Client
// ============================================

class HolochainClient {
  private static instance: HolochainClient | null = null;
  private ws: WebSocket | null = null;
  private config: HolochainConfig;
  private messageId = 0;
  private pendingCalls = new Map<number, {
    resolve: (value: unknown) => void;
    reject: (reason: unknown) => void;
  }>();
  private eventHandlers = new Map<string, Set<(data: unknown) => void>>();

  private constructor(config: HolochainConfig) {
    this.config = config;
  }

  static async connect(config: HolochainConfig): Promise<HolochainClient> {
    if (HolochainClient.instance) {
      HolochainClient.disconnect();
    }

    const client = new HolochainClient(config);
    await client.initConnection();
    HolochainClient.instance = client;
    return client;
  }

  static disconnect(): void {
    if (HolochainClient.instance) {
      HolochainClient.instance.close();
      HolochainClient.instance = null;
    }
  }

  static getInstance(): HolochainClient | null {
    return HolochainClient.instance;
  }

  private async initConnection(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.config.conductorUrl);

      this.ws.onopen = () => {
        console.log('Connected to Holochain conductor');
        resolve();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(new Error('Failed to connect to Holochain conductor'));
      };

      this.ws.onclose = () => {
        console.log('Disconnected from Holochain conductor');
        useHolochainStore.getState().disconnect();
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };
    });
  }

  private handleMessage(data: string): void {
    try {
      const message = JSON.parse(data);

      // Handle RPC responses
      if (message.id !== undefined && this.pendingCalls.has(message.id)) {
        const { resolve, reject } = this.pendingCalls.get(message.id)!;
        this.pendingCalls.delete(message.id);

        if (message.error) {
          reject(new Error(message.error.message || 'Unknown error'));
        } else {
          resolve(message.result);
        }
      }

      // Handle signals/events
      if (message.type === 'signal') {
        const handlers = this.eventHandlers.get(message.signal_type);
        if (handlers) {
          handlers.forEach((handler) => handler(message.data));
        }
      }
    } catch (err) {
      console.error('Failed to parse message:', err);
    }
  }

  private async call<T>(zomeName: string, fnName: string, payload: unknown): Promise<T> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('Not connected to Holochain');
    }

    const id = ++this.messageId;

    return new Promise((resolve, reject) => {
      this.pendingCalls.set(id, { resolve: resolve as (v: unknown) => void, reject });

      const message = {
        id,
        type: 'call_zome',
        data: {
          cell_id: this.config.cellId,
          zome_name: zomeName,
          fn_name: fnName,
          payload,
          provenance: this.config.cellId?.[1],
        },
      };

      this.ws!.send(JSON.stringify(message));

      // Timeout after 30 seconds
      setTimeout(() => {
        if (this.pendingCalls.has(id)) {
          this.pendingCalls.delete(id);
          reject(new Error('Zome call timeout'));
        }
      }, 30000);
    });
  }

  private close(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.pendingCalls.clear();
    this.eventHandlers.clear();
  }

  on(event: string, handler: (data: unknown) => void): () => void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event)!.add(handler);

    return () => {
      this.eventHandlers.get(event)?.delete(handler);
    };
  }

  // ============================================
  // Trust Zome Functions
  // ============================================

  async getAgentPubKey(): Promise<string> {
    const result = await this.call<{ agent_pub_key: string }>('trust', 'get_my_pub_key', null);
    return result.agent_pub_key;
  }

  async createAttestation(attestation: Omit<TrustAttestation, 'id' | 'createdAt' | 'signature' | 'actionHash'>): Promise<TrustAttestation> {
    const result = await this.call<TrustAttestation>('trust', 'create_attestation', {
      subject: attestation.subject,
      trust_level: attestation.trustLevel,
      category: attestation.category,
      claim: attestation.claim,
      evidence: attestation.evidence,
      expires_at: attestation.expiresAt,
    });
    return result;
  }

  async revokeAttestation(attestationId: string, reason: string): Promise<void> {
    await this.call('trust', 'revoke_attestation', {
      attestation_id: attestationId,
      reason,
    });
  }

  async getAttestationsFor(subject: string): Promise<TrustAttestation[]> {
    const result = await this.call<{ attestations: TrustAttestation[] }>('trust', 'get_attestations_for', {
      subject,
    });
    return result.attestations;
  }

  async getAttestationsBy(attestor: string): Promise<TrustAttestation[]> {
    const result = await this.call<{ attestations: TrustAttestation[] }>('trust', 'get_attestations_by', {
      attestor,
    });
    return result.attestations;
  }

  async queryTrust(query: TrustQuery): Promise<TrustPath[]> {
    const result = await this.call<{ paths: TrustPath[] }>('trust', 'query_trust', {
      subject: query.subject,
      max_path_length: query.maxPathLength ?? 6,
      min_trust_level: query.minTrustLevel ?? 0,
      categories: query.categories,
      include_expired: query.includeExpired ?? false,
    });
    return result.paths;
  }

  async calculateTrustScore(subject: string): Promise<{
    score: number;
    tier: 'high' | 'medium' | 'low' | 'unknown';
    paths: TrustPath[];
    attestationCount: number;
  }> {
    const result = await this.call<{
      score: number;
      tier: string;
      paths: TrustPath[];
      attestation_count: number;
    }>('trust', 'calculate_trust_score', { subject });

    return {
      score: result.score,
      tier: result.tier as 'high' | 'medium' | 'low' | 'unknown',
      paths: result.paths,
      attestationCount: result.attestation_count,
    };
  }

  async getTrustNetwork(depth: number = 2): Promise<TrustNetwork> {
    const result = await this.call<{
      nodes: Array<{
        agent_pub_key: string;
        display_name?: string;
        trust_score: number;
        attestations_given: number;
        attestations_received: number;
        connections: string[];
      }>;
      edges: Array<{
        from: string;
        to: string;
        weight: number;
        attestations: TrustAttestation[];
      }>;
      my_agent_pub_key: string;
    }>('trust', 'get_trust_network', { depth });

    const nodes = new Map<string, TrustGraphNode>();
    const edges = new Map<string, TrustGraphEdge>();

    for (const node of result.nodes) {
      nodes.set(node.agent_pub_key, {
        agentPubKey: node.agent_pub_key,
        displayName: node.display_name,
        trustScore: node.trust_score,
        attestationsGiven: node.attestations_given,
        attestationsReceived: node.attestations_received,
        connections: node.connections,
      });
    }

    for (const edge of result.edges) {
      const key = `${edge.from}->${edge.to}`;
      edges.set(key, {
        from: edge.from,
        to: edge.to,
        weight: edge.weight,
        attestations: edge.attestations,
      });
    }

    return {
      nodes,
      edges,
      myAgentPubKey: result.my_agent_pub_key,
    };
  }

  async requestIntroduction(params: {
    target: string;
    introducer: string;
    message: string;
    context?: string;
  }): Promise<string> {
    const result = await this.call<{ request_id: string }>('trust', 'request_introduction', params);
    return result.request_id;
  }

  async respondToIntroduction(params: {
    requestId: string;
    accept: boolean;
    message?: string;
    attestation?: Omit<TrustAttestation, 'id' | 'createdAt' | 'signature' | 'actionHash'>;
  }): Promise<void> {
    await this.call('trust', 'respond_to_introduction', {
      request_id: params.requestId,
      accept: params.accept,
      message: params.message,
      attestation: params.attestation,
    });
  }

  async getPendingIntroductions(): Promise<Array<{
    id: string;
    requester: string;
    target: string;
    message: string;
    context?: string;
    createdAt: string;
  }>> {
    const result = await this.call<{ introductions: Array<{
      id: string;
      requester: string;
      target: string;
      message: string;
      context?: string;
      created_at: string;
    }> }>('trust', 'get_pending_introductions', null);

    return result.introductions.map((intro) => ({
      id: intro.id,
      requester: intro.requester,
      target: intro.target,
      message: intro.message,
      context: intro.context,
      createdAt: intro.created_at,
    }));
  }

  // ============================================
  // Profile Zome Functions
  // ============================================

  async setProfile(profile: {
    displayName: string;
    bio?: string;
    avatar?: string;
    publicKeys?: { type: string; key: string }[];
  }): Promise<void> {
    await this.call('profiles', 'set_profile', {
      display_name: profile.displayName,
      bio: profile.bio,
      avatar: profile.avatar,
      public_keys: profile.publicKeys,
    });
  }

  async getProfile(agentPubKey: string): Promise<{
    displayName: string;
    bio?: string;
    avatar?: string;
    publicKeys?: { type: string; key: string }[];
  } | null> {
    const result = await this.call<{
      profile: {
        display_name: string;
        bio?: string;
        avatar?: string;
        public_keys?: { type: string; key: string }[];
      } | null;
    }>('profiles', 'get_profile', { agent_pub_key: agentPubKey });

    if (!result.profile) return null;

    return {
      displayName: result.profile.display_name,
      bio: result.profile.bio,
      avatar: result.profile.avatar,
      publicKeys: result.profile.public_keys,
    };
  }

  async searchProfiles(query: string): Promise<Array<{
    agentPubKey: string;
    displayName: string;
    bio?: string;
  }>> {
    const result = await this.call<{
      profiles: Array<{
        agent_pub_key: string;
        display_name: string;
        bio?: string;
      }>;
    }>('profiles', 'search_profiles', { query });

    return result.profiles.map((p) => ({
      agentPubKey: p.agent_pub_key,
      displayName: p.display_name,
      bio: p.bio,
    }));
  }
}

// ============================================
// React Hooks
// ============================================

export function useHolochain() {
  const store = useHolochainStore();

  return {
    status: store.status,
    agentPubKey: store.agentPubKey,
    error: store.error,
    network: store.network,
    isConnected: store.status === 'connected',
    connect: store.connect,
    disconnect: store.disconnect,
  };
}

export function useTrustQuery(subject: string | null, options?: Partial<TrustQuery>) {
  const { isConnected } = useHolochain();
  const [data, setData] = useState<TrustPath[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!subject || !isConnected) {
      setData(null);
      return;
    }

    const client = HolochainClient.getInstance();
    if (!client) return;

    setIsLoading(true);
    setError(null);

    client
      .queryTrust({ subject, ...options })
      .then(setData)
      .catch(setError)
      .finally(() => setIsLoading(false));
  }, [subject, isConnected, options?.maxPathLength, options?.minTrustLevel]);

  return { data, isLoading, error };
}

export function useTrustScore(subject: string | null) {
  const { isConnected } = useHolochain();
  const [data, setData] = useState<{
    score: number;
    tier: 'high' | 'medium' | 'low' | 'unknown';
    paths: TrustPath[];
    attestationCount: number;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!subject || !isConnected) {
      setData(null);
      return;
    }

    const client = HolochainClient.getInstance();
    if (!client) return;

    setIsLoading(true);
    setError(null);

    client
      .calculateTrustScore(subject)
      .then(setData)
      .catch(setError)
      .finally(() => setIsLoading(false));
  }, [subject, isConnected]);

  return { data, isLoading, error };
}

export function useCreateAttestation() {
  const { isConnected } = useHolochain();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const createAttestation = async (
    attestation: Omit<TrustAttestation, 'id' | 'createdAt' | 'signature' | 'actionHash' | 'attestor'>
  ): Promise<TrustAttestation | null> => {
    const client = HolochainClient.getInstance();

    if (!isConnected || !client) {
      // Queue for later if offline
      const pendingAttestation: TrustAttestation = {
        ...attestation,
        id: crypto.randomUUID(),
        attestor: 'pending',
        createdAt: new Date().toISOString(),
        signature: 'pending',
      };
      useHolochainStore.getState().addPendingAttestation(pendingAttestation);
      return pendingAttestation;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await client.createAttestation(attestation);
      return result;
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to create attestation'));
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  return { createAttestation, isLoading, error };
}

export function useTrustNetwork(depth: number = 2) {
  const { isConnected } = useHolochain();
  const setNetwork = useHolochainStore((s) => s.setNetwork);
  const network = useHolochainStore((s) => s.network);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const refresh = async () => {
    const client = HolochainClient.getInstance();
    if (!isConnected || !client) return;

    setIsLoading(true);
    setError(null);

    try {
      const result = await client.getTrustNetwork(depth);
      setNetwork(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch trust network'));
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isConnected) {
      refresh();
    }
  }, [isConnected, depth]);

  return { network, isLoading, error, refresh };
}

// Import for useState/useEffect
import { useState, useEffect } from 'react';

export default HolochainClient;
