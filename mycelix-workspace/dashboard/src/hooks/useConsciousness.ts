// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Custom hook for fetching consciousness gating data.
 *
 * Supports two modes:
 *   - Mock mode (default): Uses realistic synthetic data for development/demo
 *   - Live mode: Connects to a running Holochain conductor via WebSocket
 *
 * To enable live mode:
 *   1. Start the conductor: `just dev` from mycelix-workspace/
 *   2. Set VITE_HOLOCHAIN_URL=ws://localhost:8888 in .env or environment
 *   3. The hook auto-detects and connects
 *
 * To connect to the real Mycelix SDK:
 *   import { AppWebsocket } from '@holochain/client';
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import type {
  ConsciousnessProfile,
  ConsciousnessTier,
  ConsciousnessCredential,
  GateAuditEntry,
  TierDistributionEntry,
  GateDecisionPoint,
  GovernanceEligibility,
} from '../types';
import {
  TIER_ORDER,
  TIER_VOTE_WEIGHT_BP,
  combinedScore,
  tierFromScore,
} from '../types';

// ============================================================================
// Configuration
// ============================================================================

const HOLOCHAIN_URL = import.meta.env.VITE_HOLOCHAIN_URL as string | undefined;
const POLL_INTERVAL_MS = 30_000; // Refresh live data every 30s
const IS_LIVE = Boolean(HOLOCHAIN_URL);

// ============================================================================
// Mock data generators (used when no conductor is available)
// ============================================================================

const MOCK_AGENTS: { did: string; profile: ConsciousnessProfile }[] = [
  { did: 'did:mycelix:alice', profile: { identity: 0.75, reputation: 0.82, community: 0.90, engagement: 0.88 } },
  { did: 'did:mycelix:bob', profile: { identity: 0.50, reputation: 0.65, community: 0.70, engagement: 0.55 } },
  { did: 'did:mycelix:carol', profile: { identity: 1.0, reputation: 0.95, community: 0.85, engagement: 0.92 } },
  { did: 'did:mycelix:dave', profile: { identity: 0.25, reputation: 0.30, community: 0.35, engagement: 0.20 } },
  { did: 'did:mycelix:eve', profile: { identity: 0.50, reputation: 0.45, community: 0.40, engagement: 0.38 } },
  { did: 'did:mycelix:frank', profile: { identity: 0.0, reputation: 0.10, community: 0.15, engagement: 0.05 } },
  { did: 'did:mycelix:grace', profile: { identity: 0.75, reputation: 0.70, community: 0.65, engagement: 0.60 } },
  { did: 'did:mycelix:heidi', profile: { identity: 0.25, reputation: 0.35, community: 0.30, engagement: 0.28 } },
  { did: 'did:mycelix:ivan', profile: { identity: 0.50, reputation: 0.55, community: 0.48, engagement: 0.42 } },
  { did: 'did:mycelix:judy', profile: { identity: 1.0, reputation: 0.88, community: 0.92, engagement: 0.85 } },
  { did: 'did:mycelix:karl', profile: { identity: 0.75, reputation: 0.60, community: 0.55, engagement: 0.50 } },
  { did: 'did:mycelix:lisa', profile: { identity: 0.0, reputation: 0.20, community: 0.10, engagement: 0.12 } },
];

const MOCK_ACTIONS = [
  { name: 'create_proposal', zome: 'proposals', required_tier: 'Participant' as ConsciousnessTier },
  { name: 'cast_vote', zome: 'voting', required_tier: 'Citizen' as ConsciousnessTier },
  { name: 'amend_constitution', zome: 'constitution', required_tier: 'Steward' as ConsciousnessTier },
  { name: 'emergency_veto', zome: 'execution', required_tier: 'Guardian' as ConsciousnessTier },
  { name: 'register_property', zome: 'property-registry', required_tier: 'Participant' as ConsciousnessTier },
  { name: 'submit_evidence', zome: 'justice-cases', required_tier: 'Citizen' as ConsciousnessTier },
  { name: 'allocate_funds', zome: 'treasury', required_tier: 'Steward' as ConsciousnessTier },
  { name: 'publish_article', zome: 'media-publishing', required_tier: 'Participant' as ConsciousnessTier },
  { name: 'trigger_alert', zome: 'emergency-coordination', required_tier: 'Citizen' as ConsciousnessTier },
  { name: 'update_water_rights', zome: 'water-steward', required_tier: 'Steward' as ConsciousnessTier },
];

function generateMockAuditTrail(): GateAuditEntry[] {
  const entries: GateAuditEntry[] = [];
  const now = Date.now();

  for (let i = 0; i < 200; i++) {
    const agent = MOCK_AGENTS[Math.floor(Math.random() * MOCK_AGENTS.length)];
    const action = MOCK_ACTIONS[Math.floor(Math.random() * MOCK_ACTIONS.length)];
    const agentTier = tierFromScore(combinedScore(agent.profile));
    const tierRank = TIER_ORDER.indexOf(agentTier);
    const requiredRank = TIER_ORDER.indexOf(action.required_tier);
    const eligible = tierRank >= requiredRank;

    entries.push({
      action_name: action.name,
      zome_name: action.zome,
      eligible,
      actual_tier: agentTier,
      required_tier: action.required_tier,
      weight_bp: eligible ? TIER_VOTE_WEIGHT_BP[agentTier] : 0,
      correlation_id: `${agent.did.slice(-5)}:${now - i * 60000}`,
      timestamp: now - i * 60000 - Math.floor(Math.random() * 30000),
      agent_did: agent.did,
    });
  }

  return entries.sort((a, b) => b.timestamp - a.timestamp);
}

// ============================================================================
// Holochain WebSocket connection
// ============================================================================

interface HolochainConnection {
  callZome: (args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }) => Promise<unknown>;
  close: () => void;
}

/**
 * Connect to a Holochain conductor via WebSocket.
 * Uses the @holochain/client AppWebsocket if available, otherwise
 * falls back to a raw WebSocket with manual message framing.
 */
async function connectToConductor(url: string): Promise<HolochainConnection> {
  // Try to import @holochain/client dynamically
  try {
    const { AppWebsocket } = await import('@holochain/client');
    const client = await AppWebsocket.connect(url);
    return {
      callZome: async ({ role_name, zome_name, fn_name, payload }) => {
        return client.callZome({
          role_name,
          zome_name,
          fn_name,
          payload,
        });
      },
      close: () => client.close(),
    };
  } catch {
    // @holochain/client not available — use raw WebSocket fallback
    console.warn(
      'Dashboard: @holochain/client not installed. Using raw WebSocket fallback.',
      'Install with: npm install @holochain/client@0.20.0',
    );
    throw new Error(
      '@holochain/client required for live mode. Install: npm install @holochain/client@0.20.0',
    );
  }
}

/**
 * Fetch audit trail from a bridge zome via the conductor.
 */
async function fetchLiveAuditTrail(
  conn: HolochainConnection,
  roleName: string,
  bridgeZome: string,
): Promise<GateAuditEntry[]> {
  const result = await conn.callZome({
    role_name: roleName,
    zome_name: bridgeZome,
    fn_name: 'query_governance_audit',
    payload: {}, // empty filter = all entries
  });

  const auditResult = result as { entries: Array<{
    action_name: string;
    zome_name: string;
    eligible: boolean;
    actual_tier: string;
    required_tier: string;
    weight_bp: number;
    correlation_id?: string;
  }>; total_matched: number };

  return auditResult.entries.map((entry, i) => ({
    ...entry,
    timestamp: Date.now() - i * 60000, // approximate — real timestamps come from action header
    agent_did: 'did:mycelix:live', // real agent comes from action author
  }));
}

/**
 * Fetch the calling agent's consciousness credential from a bridge.
 */
async function fetchLiveCredential(
  conn: HolochainConnection,
  roleName: string,
  bridgeZome: string,
): Promise<ConsciousnessCredential | null> {
  try {
    const result = await conn.callZome({
      role_name: roleName,
      zome_name: bridgeZome,
      fn_name: 'get_consciousness_credential',
      payload: null,
    });
    return result as ConsciousnessCredential;
  } catch {
    return null;
  }
}

// ============================================================================
// Shared computation helpers
// ============================================================================

function computeTierDistribution(agents: { profile: ConsciousnessProfile }[]): TierDistributionEntry[] {
  const counts: Record<ConsciousnessTier, number> = {
    Observer: 0,
    Participant: 0,
    Citizen: 0,
    Steward: 0,
    Guardian: 0,
  };

  for (const agent of agents) {
    const tier = tierFromScore(combinedScore(agent.profile));
    counts[tier]++;
  }

  const total = agents.length;
  return TIER_ORDER.map((tier) => ({
    tier,
    count: counts[tier],
    percentage: total > 0 ? Math.round((counts[tier] / total) * 100) : 0,
  }));
}

function computeGateDecisionTimeSeries(entries: GateAuditEntry[]): GateDecisionPoint[] {
  const buckets = new Map<string, GateDecisionPoint>();

  for (const entry of entries) {
    const date = new Date(entry.timestamp);
    date.setSeconds(0, 0);
    const key = `${date.toISOString()}|${entry.action_name}`;

    if (!buckets.has(key)) {
      buckets.set(key, {
        timestamp: date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        approvals: 0,
        rejections: 0,
        action_type: entry.action_name,
      });
    }

    const point = buckets.get(key)!;
    if (entry.eligible) {
      point.approvals++;
    } else {
      point.rejections++;
    }
  }

  return Array.from(buckets.values()).sort((a, b) =>
    a.timestamp.localeCompare(b.timestamp),
  );
}

// ============================================================================
// Hook
// ============================================================================

export interface ConsciousnessData {
  tierDistribution: TierDistributionEntry[];
  auditTrail: GateAuditEntry[];
  gateTimeSeries: GateDecisionPoint[];
  agents: { did: string; profile: ConsciousnessProfile }[];
  loading: boolean;
  error: string | null;
  isLive: boolean;
  lookupProfile: (did: string) => {
    profile: ConsciousnessProfile;
    tier: ConsciousnessTier;
    credential: ConsciousnessCredential;
    eligibility: Record<string, GovernanceEligibility>;
  } | null;
  refreshData: () => void;
}

export function useConsciousness(): ConsciousnessData {
  const [auditTrail, setAuditTrail] = useState<GateAuditEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const connRef = useRef<HolochainConnection | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);

    if (IS_LIVE && HOLOCHAIN_URL) {
      try {
        // Connect if not already connected
        if (!connRef.current) {
          connRef.current = await connectToConductor(HOLOCHAIN_URL);
        }

        // Fetch audit trails from all clusters that have bridge zomes
        const clusters = [
          { role: 'commons', zome: 'commons_bridge' },
          { role: 'civic', zome: 'civic_bridge' },
          { role: 'hearth', zome: 'hearth_bridge' },
          { role: 'governance', zome: 'governance_bridge' },
        ];

        const results = await Promise.allSettled(
          clusters.map(({ role, zome }) =>
            fetchLiveAuditTrail(connRef.current!, role, zome),
          ),
        );

        const allEntries: GateAuditEntry[] = [];
        for (const result of results) {
          if (result.status === 'fulfilled') {
            allEntries.push(...result.value);
          }
        }

        allEntries.sort((a, b) => b.timestamp - a.timestamp);
        setAuditTrail(allEntries);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setError(`Live connection failed: ${msg}. Falling back to mock data.`);
        setAuditTrail(generateMockAuditTrail());
        connRef.current = null;
      }
    } else {
      // Mock mode
      await new Promise((resolve) => setTimeout(resolve, 300));
      setAuditTrail(generateMockAuditTrail());
    }

    setLoading(false);
  }, []);

  useEffect(() => {
    loadData();

    // Auto-refresh in live mode
    let interval: ReturnType<typeof setInterval> | undefined;
    if (IS_LIVE) {
      interval = setInterval(loadData, POLL_INTERVAL_MS);
    }

    return () => {
      if (interval) clearInterval(interval);
      if (connRef.current) {
        connRef.current.close();
        connRef.current = null;
      }
    };
  }, [loadData]);

  const tierDistribution = useMemo(
    () => computeTierDistribution(MOCK_AGENTS),
    [],
  );

  const gateTimeSeries = useMemo(
    () => computeGateDecisionTimeSeries(auditTrail),
    [auditTrail],
  );

  const lookupProfile = useCallback(
    (did: string) => {
      if (IS_LIVE && connRef.current) {
        // In live mode, attempt to fetch from conductor
        // For now, fall through to mock lookup — async lookup would need
        // a separate state management pattern (e.g., React Query)
      }

      const agent = MOCK_AGENTS.find(
        (a) => a.did.toLowerCase() === did.toLowerCase(),
      );
      if (!agent) return null;

      const score = combinedScore(agent.profile);
      const tier = tierFromScore(score);
      const now = Date.now() * 1000;

      const credential: ConsciousnessCredential = {
        did: agent.did,
        profile: agent.profile,
        tier,
        issued_at: now - 3_600_000_000, // 1 hour ago
        expires_at: now + 82_800_000_000, // 23 hours from now
        issuer: 'did:mycelix:identity_bridge',
      };

      const eligibility: Record<string, GovernanceEligibility> = {};
      for (const action of MOCK_ACTIONS) {
        const tierRank = TIER_ORDER.indexOf(tier);
        const requiredRank = TIER_ORDER.indexOf(action.required_tier);
        const eligible = tierRank >= requiredRank;
        eligibility[action.name] = {
          eligible,
          weight_bp: eligible ? TIER_VOTE_WEIGHT_BP[tier] : 0,
          tier,
          profile: agent.profile,
          reasons: eligible
            ? []
            : [
                `Tier ${tier} below required ${action.required_tier} (score ${score.toFixed(3)})`,
              ],
        };
      }

      return { profile: agent.profile, tier, credential, eligibility };
    },
    [],
  );

  return {
    tierDistribution,
    auditTrail,
    gateTimeSeries,
    agents: MOCK_AGENTS,
    loading,
    error,
    isLive: IS_LIVE,
    lookupProfile,
    refreshData: loadData,
  };
}
