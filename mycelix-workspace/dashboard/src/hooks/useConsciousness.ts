/**
 * Custom hook for fetching consciousness gating data.
 *
 * Currently uses mock data that matches the real Rust/SDK types.
 * Comments indicate where to wire in real SDK calls.
 *
 * To connect to the real Mycelix SDK:
 *   import { MycelixEcosystemClient } from '@mycelix/sdk';
 *   import { canPerform, queryGovernanceAudit } from '@mycelix/sdk/core/consciousness-gate';
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
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
// Mock data generators
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

function computeTierDistribution(agents: typeof MOCK_AGENTS): TierDistributionEntry[] {
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
  // Group by minute and action type
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
  const [error] = useState<string | null>(null);

  const loadData = useCallback(() => {
    setLoading(true);

    // TODO: Wire in real SDK calls:
    //   const client = await MycelixEcosystemClient.connect({ url: 'ws://localhost:8888' });
    //   const auditResult = await queryGovernanceAudit(client, 'commons_bridge', 'commons', {});
    //   setAuditTrail(auditResult.entries);

    // Using mock data for now
    setTimeout(() => {
      setAuditTrail(generateMockAuditTrail());
      setLoading(false);
    }, 300);
  }, []);

  useEffect(() => {
    loadData();
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
      // TODO: Wire in real SDK call:
      //   const credential = await client.callZome({
      //     role_name: 'commons',
      //     zome_name: 'commons_bridge',
      //     fn_name: 'get_consciousness_credential',
      //     payload: null,
      //   });
      //   const eligibility = await canPerform(client, 'commons_bridge', 'commons', requiredTier);

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
    lookupProfile,
    refreshData: loadData,
  };
}
