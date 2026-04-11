// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Vue Composable - useTrust
 *
 * Reactive trust network management with MATL algorithm.
 */

import { ref, computed, onMounted, onUnmounted } from 'vue';
import type { Ref, ComputedRef } from 'vue';
import { getMycelixClient } from '../../bootstrap';

export interface TrustScore {
  agentPubKey: string;
  email?: string;
  name?: string;
  directTrust: number;
  transitiveTrust: number;
  combinedTrust: number;
  attestationCount: number;
  lastUpdated: number;
}

export interface Attestation {
  hash: string;
  from: string;
  to: string;
  trustLevel: number;
  context?: string;
  createdAt: number;
  expiresAt?: number;
}

export interface UseTrustReturn {
  // State
  myTrustNetwork: Ref<TrustScore[]>;
  pendingAttestations: Ref<Attestation[]>;
  loading: Ref<boolean>;
  error: Ref<string | null>;

  // Computed
  trustedCount: ComputedRef<number>;
  averageTrust: ComputedRef<number>;

  // Actions
  refresh: () => Promise<void>;
  getTrustScore: (agentPubKey: string) => Promise<TrustScore | null>;
  getTrustScoreByEmail: (email: string) => Promise<TrustScore | null>;
  createAttestation: (
    toAgent: string,
    trustLevel: number,
    context?: string
  ) => Promise<string>;
  revokeAttestation: (attestationHash: string) => Promise<void>;
  getAttestationsFor: (agentPubKey: string) => Promise<Attestation[]>;
  getAttestationsBy: (agentPubKey: string) => Promise<Attestation[]>;
  calculateTransitiveTrust: (
    fromAgent: string,
    toAgent: string,
    maxDepth?: number
  ) => Promise<number>;
}

export function useTrust(): UseTrustReturn {
  const myTrustNetwork = ref<TrustScore[]>([]);
  const pendingAttestations = ref<Attestation[]>([]);
  const loading = ref(false);
  const error = ref<string | null>(null);

  let unsubscribe: (() => void) | null = null;

  const trustedCount = computed(() =>
    myTrustNetwork.value.filter((t) => t.combinedTrust >= 0.3).length
  );

  const averageTrust = computed(() => {
    if (myTrustNetwork.value.length === 0) return 0;
    const sum = myTrustNetwork.value.reduce((acc, t) => acc + t.combinedTrust, 0);
    return sum / myTrustNetwork.value.length;
  });

  async function refresh() {
    loading.value = true;
    error.value = null;

    try {
      const client = getMycelixClient();
      const services = client.getServices();

      const [network, pending] = await Promise.all([
        services.trust.getMyTrustNetwork(),
        services.trust.getPendingAttestations(),
      ]);

      myTrustNetwork.value = network;
      pendingAttestations.value = pending;
    } catch (e) {
      error.value = String(e);
    } finally {
      loading.value = false;
    }
  }

  async function getTrustScore(agentPubKey: string): Promise<TrustScore | null> {
    try {
      const client = getMycelixClient();
      return await client.getServices().trust.getTrustScore(agentPubKey);
    } catch (e) {
      error.value = String(e);
      return null;
    }
  }

  async function getTrustScoreByEmail(email: string): Promise<TrustScore | null> {
    try {
      const client = getMycelixClient();
      return await client.getServices().trust.getTrustScoreByEmail(email);
    } catch (e) {
      error.value = String(e);
      return null;
    }
  }

  async function createAttestation(
    toAgent: string,
    trustLevel: number,
    context?: string
  ): Promise<string> {
    try {
      const client = getMycelixClient();
      const hash = await client.getServices().trust.createAttestation({
        toAgent,
        trustLevel,
        context,
      });

      // Refresh network after creating attestation
      await refresh();

      return hash;
    } catch (e) {
      error.value = String(e);
      throw e;
    }
  }

  async function revokeAttestation(attestationHash: string): Promise<void> {
    try {
      const client = getMycelixClient();
      await client.getServices().trust.revokeAttestation(attestationHash);

      // Refresh network after revoking
      await refresh();
    } catch (e) {
      error.value = String(e);
      throw e;
    }
  }

  async function getAttestationsFor(agentPubKey: string): Promise<Attestation[]> {
    try {
      const client = getMycelixClient();
      return await client.getServices().trust.getAttestationsFor(agentPubKey);
    } catch (e) {
      error.value = String(e);
      return [];
    }
  }

  async function getAttestationsBy(agentPubKey: string): Promise<Attestation[]> {
    try {
      const client = getMycelixClient();
      return await client.getServices().trust.getAttestationsBy(agentPubKey);
    } catch (e) {
      error.value = String(e);
      return [];
    }
  }

  async function calculateTransitiveTrust(
    fromAgent: string,
    toAgent: string,
    maxDepth = 5
  ): Promise<number> {
    try {
      const client = getMycelixClient();
      return await client.getServices().trust.calculateTransitiveTrust(
        fromAgent,
        toAgent,
        maxDepth
      );
    } catch (e) {
      error.value = String(e);
      return 0;
    }
  }

  onMounted(async () => {
    await refresh();

    // Subscribe to trust updates
    const client = getMycelixClient();
    unsubscribe = client.getServices().signalHub.on('attestation_received', () => {
      refresh();
    });
  });

  onUnmounted(() => {
    if (unsubscribe) unsubscribe();
  });

  return {
    myTrustNetwork,
    pendingAttestations,
    loading,
    error,
    trustedCount,
    averageTrust,
    refresh,
    getTrustScore,
    getTrustScoreByEmail,
    createAttestation,
    revokeAttestation,
    getAttestationsFor,
    getAttestationsBy,
    calculateTransitiveTrust,
  };
}

export default useTrust;
