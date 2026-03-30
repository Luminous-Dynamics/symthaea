// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Consciousness Gate Unit Tests
 *
 * Tests for the consciousness gate error handling, score computation,
 * tier derivation, pre-flight eligibility checks, and retry middleware.
 */

import { describe, it, expect, vi } from 'vitest';

import {
  ConsciousnessGateError,
  combinedScore,
  tierFromScore,
  canPerform,
  withGateRetry,
  queryGovernanceAudit,
  type ConsciousnessProfile,
  type ConsciousnessCredential,
  type ZomeCallable,
  type GovernanceAuditFilter,
  type GovernanceAuditResult,
  type GateAuditEntry,
} from '../src/core/consciousness-gate';

// ============================================================================
// ConsciousnessGateError.fromWasmError()
// ============================================================================

describe('ConsciousnessGateError.fromWasmError()', () => {
  it('parses a gate rejection with tier and reasons', () => {
    const err = new Error(
      'Consciousness gate: tier Observer insufficient. Reasons: Tier Observer below required Participant (score 0.100, need >= 0.300)'
    );
    const gateErr = ConsciousnessGateError.fromWasmError(err);
    expect(gateErr).not.toBeNull();
    expect(gateErr!.rejection.actualTier).toBe('Observer');
    expect(gateErr!.rejection.requiredTier).toBe('Participant');
    expect(gateErr!.rejection.expired).toBe(false);
    expect(gateErr!.rejection.reasons.length).toBeGreaterThan(0);
  });

  it('parses an expired credential error', () => {
    const err = new Error(
      'Consciousness gate: tier Observer insufficient. Reasons: Credential expired at 1700000000000000 (now 1700001000000000)'
    );
    const gateErr = ConsciousnessGateError.fromWasmError(err);
    expect(gateErr).not.toBeNull();
    expect(gateErr!.rejection.expired).toBe(true);
  });

  it('parses a standalone credential expired pattern', () => {
    const err = new Error('Credential expired at 1700000000000000');
    const gateErr = ConsciousnessGateError.fromWasmError(err);
    expect(gateErr).not.toBeNull();
    expect(gateErr!.rejection.expired).toBe(true);
    expect(gateErr!.rejection.actualTier).toBe('Observer');
  });

  it('returns null for non-gate errors', () => {
    const err = new Error('Network timeout');
    expect(ConsciousnessGateError.fromWasmError(err)).toBeNull();
  });

  it('returns null for string errors without gate pattern', () => {
    expect(ConsciousnessGateError.fromWasmError('random error')).toBeNull();
  });

  it('extracts tier from reasons string', () => {
    const err = new Error(
      'Consciousness gate: tier Citizen insufficient. Reasons: Tier Citizen below required Steward (score 0.450, need >= 0.600), Identity 0.200 below required 0.500'
    );
    const gateErr = ConsciousnessGateError.fromWasmError(err);
    expect(gateErr).not.toBeNull();
    expect(gateErr!.rejection.actualTier).toBe('Citizen');
    expect(gateErr!.rejection.requiredTier).toBe('Steward');
    // Comma inside parenthetical "(score 0.450, need >= 0.600)" causes 3 splits
    expect(gateErr!.rejection.reasons.length).toBe(3);
  });
});

// ============================================================================
// combinedScore
// ============================================================================

describe('combinedScore', () => {
  it('computes weighted average correctly', () => {
    const profile: ConsciousnessProfile = {
      identity: 0.5,
      reputation: 0.5,
      community: 0.5,
      engagement: 0.5,
    };
    // 0.5*0.25 + 0.5*0.25 + 0.5*0.30 + 0.5*0.20 = 0.5
    expect(combinedScore(profile)).toBeCloseTo(0.5, 10);
  });

  it('returns 1.0 for all-ones profile', () => {
    const profile: ConsciousnessProfile = {
      identity: 1.0,
      reputation: 1.0,
      community: 1.0,
      engagement: 1.0,
    };
    expect(combinedScore(profile)).toBeCloseTo(1.0, 10);
  });

  it('returns 0.0 for all-zeros profile', () => {
    const profile: ConsciousnessProfile = {
      identity: 0.0,
      reputation: 0.0,
      community: 0.0,
      engagement: 0.0,
    };
    expect(combinedScore(profile)).toBeCloseTo(0.0, 10);
  });

  it('weights community highest at 30%', () => {
    const communityOnly: ConsciousnessProfile = {
      identity: 0,
      reputation: 0,
      community: 1.0,
      engagement: 0,
    };
    expect(combinedScore(communityOnly)).toBeCloseTo(0.3, 10);
  });

  it('weights engagement lowest at 20%', () => {
    const engagementOnly: ConsciousnessProfile = {
      identity: 0,
      reputation: 0,
      community: 0,
      engagement: 1.0,
    };
    expect(combinedScore(engagementOnly)).toBeCloseTo(0.2, 10);
  });
});

// ============================================================================
// tierFromScore
// ============================================================================

describe('tierFromScore', () => {
  it('maps score boundaries correctly', () => {
    expect(tierFromScore(0.0)).toBe('Observer');
    expect(tierFromScore(0.29)).toBe('Observer');
    expect(tierFromScore(0.3)).toBe('Participant');
    expect(tierFromScore(0.39)).toBe('Participant');
    expect(tierFromScore(0.4)).toBe('Citizen');
    expect(tierFromScore(0.59)).toBe('Citizen');
    expect(tierFromScore(0.6)).toBe('Steward');
    expect(tierFromScore(0.79)).toBe('Steward');
    expect(tierFromScore(0.8)).toBe('Guardian');
    expect(tierFromScore(1.0)).toBe('Guardian');
  });

  it('tier ordering is monotonically increasing', () => {
    const tiers = ['Observer', 'Participant', 'Citizen', 'Steward', 'Guardian'];
    const scores = [0.0, 0.3, 0.4, 0.6, 0.8];
    for (let i = 0; i < scores.length; i++) {
      expect(tierFromScore(scores[i])).toBe(tiers[i]);
    }
  });
});

// ============================================================================
// canPerform
// ============================================================================

describe('canPerform', () => {
  function makeMockClient(credential: ConsciousnessCredential): ZomeCallable {
    return {
      callZome: vi.fn().mockResolvedValue(credential),
    };
  }

  function makeExpiredClient(): ZomeCallable {
    return {
      callZome: vi.fn().mockResolvedValue({
        did: 'did:mycelix:test',
        profile: { identity: 1.0, reputation: 1.0, community: 1.0, engagement: 1.0 },
        tier: 'Guardian',
        issued_at: 1_000_000,
        expires_at: 1_000_000, // already expired
        issuer: 'test',
      } as ConsciousnessCredential),
    };
  }

  function makeFailingClient(): ZomeCallable {
    return {
      callZome: vi.fn().mockRejectedValue(new Error('Network timeout')),
    };
  }

  it('returns eligible when tier is sufficient', async () => {
    const cred: ConsciousnessCredential = {
      did: 'did:mycelix:alice',
      profile: { identity: 0.5, reputation: 0.5, community: 0.5, engagement: 0.5 },
      tier: 'Citizen',
      issued_at: Date.now() * 1000 - 60_000_000,
      expires_at: Date.now() * 1000 + 86_400_000_000,
      issuer: 'test',
    };
    const result = await canPerform(makeMockClient(cred), 'commons_bridge', 'commons', 'Participant');
    expect(result.eligible).toBe(true);
    expect(result.tier).toBe('Citizen');
    expect(result.reasons).toHaveLength(0);
  });

  it('returns ineligible when tier is insufficient', async () => {
    const cred: ConsciousnessCredential = {
      did: 'did:mycelix:bob',
      profile: { identity: 0.2, reputation: 0.2, community: 0.2, engagement: 0.2 },
      tier: 'Observer',
      issued_at: Date.now() * 1000 - 60_000_000,
      expires_at: Date.now() * 1000 + 86_400_000_000,
      issuer: 'test',
    };
    const result = await canPerform(makeMockClient(cred), 'commons_bridge', 'commons', 'Citizen');
    expect(result.eligible).toBe(false);
    expect(result.reasons.length).toBeGreaterThan(0);
  });

  it('detects expired credential', async () => {
    const result = await canPerform(makeExpiredClient(), 'commons_bridge', 'commons', 'Participant');
    expect(result.eligible).toBe(false);
    expect(result.nearingExpiry).toBe(true);
    expect(result.reasons.some((r) => r.includes('expired'))).toBe(true);
  });

  it('detects credential nearing expiry', async () => {
    const cred: ConsciousnessCredential = {
      did: 'did:mycelix:carol',
      profile: { identity: 0.5, reputation: 0.5, community: 0.5, engagement: 0.5 },
      tier: 'Citizen',
      issued_at: Date.now() * 1000 - 86_000_000_000,
      expires_at: Date.now() * 1000 + 3_600_000_000, // 1 hour left (within 2h window)
      issuer: 'test',
    };
    const result = await canPerform(makeMockClient(cred), 'commons_bridge', 'commons', 'Participant');
    expect(result.eligible).toBe(true);
    expect(result.nearingExpiry).toBe(true);
  });

  it('returns ineligible on call failure', async () => {
    const result = await canPerform(makeFailingClient(), 'commons_bridge', 'commons', 'Participant');
    expect(result.eligible).toBe(false);
    expect(result.tier).toBe('Observer');
    expect(result.reasons.some((r) => r.includes('Failed'))).toBe(true);
  });
});

// ============================================================================
// withGateRetry
// ============================================================================

describe('withGateRetry', () => {
  it('returns result on first success', async () => {
    const callFn = vi.fn().mockResolvedValue('ok');
    const refreshFn = vi.fn().mockResolvedValue(undefined);
    const result = await withGateRetry(callFn, refreshFn);
    expect(result).toBe('ok');
    expect(callFn).toHaveBeenCalledTimes(1);
    expect(refreshFn).not.toHaveBeenCalled();
  });

  it('retries after expired credential error', async () => {
    const expiredError = new Error(
      'Consciousness gate: tier Observer insufficient. Reasons: Credential expired at 1700000000000000 (now 1700001000000000)'
    );
    const callFn = vi
      .fn()
      .mockRejectedValueOnce(expiredError)
      .mockResolvedValueOnce('retry-ok');
    const refreshFn = vi.fn().mockResolvedValue(undefined);

    const result = await withGateRetry(callFn, refreshFn);
    expect(result).toBe('retry-ok');
    expect(callFn).toHaveBeenCalledTimes(2);
    expect(refreshFn).toHaveBeenCalledTimes(1);
  });

  it('rethrows non-gate errors without retry', async () => {
    const networkError = new Error('Network timeout');
    const callFn = vi.fn().mockRejectedValue(networkError);
    const refreshFn = vi.fn().mockResolvedValue(undefined);

    await expect(withGateRetry(callFn, refreshFn)).rejects.toThrow('Network timeout');
    expect(callFn).toHaveBeenCalledTimes(1);
    expect(refreshFn).not.toHaveBeenCalled();
  });

  it('calls refreshFn before retry attempt', async () => {
    const order: string[] = [];
    const expiredError = new Error('Credential expired at 1700000000000000');
    const callFn = vi
      .fn()
      .mockImplementationOnce(async () => {
        order.push('call-1');
        throw expiredError;
      })
      .mockImplementationOnce(async () => {
        order.push('call-2');
        return 'ok';
      });
    const refreshFn = vi.fn().mockImplementation(async () => {
      order.push('refresh');
    });

    await withGateRetry(callFn, refreshFn);
    expect(order).toEqual(['call-1', 'refresh', 'call-2']);
  });

  it('rethrows gate errors that are not expired', async () => {
    const tierError = new Error(
      'Consciousness gate: tier Observer insufficient. Reasons: Tier Observer below required Citizen (score 0.100, need >= 0.400)'
    );
    const callFn = vi.fn().mockRejectedValue(tierError);
    const refreshFn = vi.fn().mockResolvedValue(undefined);

    await expect(withGateRetry(callFn, refreshFn)).rejects.toThrow('Consciousness gate');
    expect(callFn).toHaveBeenCalledTimes(1);
    expect(refreshFn).not.toHaveBeenCalled();
  });
});

// ============================================================================
// queryGovernanceAudit
// ============================================================================

describe('queryGovernanceAudit', () => {
  const mockEntries: GateAuditEntry[] = [
    {
      action_name: 'create_proposal',
      zome_name: 'housing_governance',
      eligible: true,
      actual_tier: 'Citizen',
      required_tier: 'Participant',
      weight_bp: 4500,
    },
    {
      action_name: 'cast_vote',
      zome_name: 'housing_governance',
      eligible: false,
      actual_tier: 'Observer',
      required_tier: 'Citizen',
      weight_bp: 1000,
    },
    {
      action_name: 'submit_reading',
      zome_name: 'water_purity',
      eligible: true,
      actual_tier: 'Steward',
      required_tier: 'Participant',
      weight_bp: 7000,
      correlation_id: 'corr-123',
    },
  ];

  const mockResult: GovernanceAuditResult = {
    entries: mockEntries,
    total_matched: 3,
  };

  function makeAuditClient(result: GovernanceAuditResult): ZomeCallable {
    return {
      callZome: vi.fn().mockResolvedValue(result),
    };
  }

  it('calls the correct bridge extern with filter', async () => {
    const client = makeAuditClient(mockResult);
    const filter: GovernanceAuditFilter = { action_name: 'create_proposal' };
    await queryGovernanceAudit(client, 'commons_bridge', 'commons', filter);

    expect(client.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'commons_bridge',
      fn_name: 'get_governance_audit_trail',
      payload: filter,
    });
  });

  it('returns typed GovernanceAuditResult', async () => {
    const client = makeAuditClient(mockResult);
    const result = await queryGovernanceAudit(client, 'commons_bridge', 'commons', {});
    expect(result.total_matched).toBe(3);
    expect(result.entries).toHaveLength(3);
    expect(result.entries[0].action_name).toBe('create_proposal');
  });

  it('passes empty filter for unfiltered queries', async () => {
    const client = makeAuditClient(mockResult);
    const filter: GovernanceAuditFilter = {};
    await queryGovernanceAudit(client, 'civic_bridge', 'civic', filter);

    expect(client.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'civic_bridge',
      fn_name: 'get_governance_audit_trail',
      payload: {},
    });
  });

  it('handles empty result set', async () => {
    const emptyResult: GovernanceAuditResult = { entries: [], total_matched: 0 };
    const client = makeAuditClient(emptyResult);
    const result = await queryGovernanceAudit(client, 'commons_bridge', 'commons', {
      eligible: false,
    });
    expect(result.entries).toHaveLength(0);
    expect(result.total_matched).toBe(0);
  });

  it('supports all filter fields', async () => {
    const client = makeAuditClient(mockResult);
    const filter: GovernanceAuditFilter = {
      action_name: 'cast_vote',
      zome_name: 'housing_governance',
      eligible: false,
      from_us: 1700000000000000,
      to_us: 1700001000000000,
    };
    await queryGovernanceAudit(client, 'commons_bridge', 'commons', filter);

    expect(client.callZome).toHaveBeenCalledWith({
      role_name: 'commons',
      zome_name: 'commons_bridge',
      fn_name: 'get_governance_audit_trail',
      payload: filter,
    });
  });

  it('preserves correlation_id in entries', async () => {
    const client = makeAuditClient(mockResult);
    const result = await queryGovernanceAudit(client, 'commons_bridge', 'commons', {});
    const withCorrelation = result.entries.find((e) => e.correlation_id);
    expect(withCorrelation).toBeDefined();
    expect(withCorrelation!.correlation_id).toBe('corr-123');
  });

  it('works with hearth bridge', async () => {
    const client = makeAuditClient({ entries: [mockEntries[0]], total_matched: 1 });
    const result = await queryGovernanceAudit(client, 'hearth_bridge', 'hearth', {
      zome_name: 'hearth_care',
    });
    expect(result.total_matched).toBe(1);
    expect(client.callZome).toHaveBeenCalledWith({
      role_name: 'hearth',
      zome_name: 'hearth_bridge',
      fn_name: 'get_governance_audit_trail',
      payload: { zome_name: 'hearth_care' },
    });
  });
});
