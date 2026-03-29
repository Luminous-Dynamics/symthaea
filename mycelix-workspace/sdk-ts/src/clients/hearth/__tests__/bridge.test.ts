// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Bridge Client Tests
 *
 * Verifies zome call arguments and payload passthrough for the BridgeClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BridgeClient } from '../bridge';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockClient() {
  return { callZome: vi.fn() } as any;
}

const HASH = new Uint8Array(32);
const AGENT = new Uint8Array(39);
const MOCK_RECORD = { entry: { Present: {} }, signed_action: { hashed: { hash: HASH } } };
const MOCK_DISPATCH_RESULT = { success: true, response: 'ok' };

// ============================================================================
// TESTS
// ============================================================================

describe('BridgeClient', () => {
  let client: BridgeClient;
  let mockClient: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    mockClient = createMockClient();
    client = new BridgeClient(mockClient, { roleName: 'hearth', timeout: 30000 });
  });

  // --------------------------------------------------------------------------
  // Intra-Cluster Dispatch
  // --------------------------------------------------------------------------

  describe('intra-cluster dispatch', () => {
    it('dispatchCall calls hearth_bridge/dispatch_call with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_DISPATCH_RESULT);
      const input = { target_zome: 'hearth_kinship', fn_name: 'get_my_hearths', payload: null };

      await client.dispatchCall(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'hearth',
          zome_name: 'hearth_bridge',
          fn_name: 'dispatch_call',
          payload: input,
        })
      );
    });

    it('queryHearth calls query_hearth with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { target_zome: 'hearth_stories', fn_name: 'get_hearth_stories', payload: HASH };

      await client.queryHearth(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'query_hearth',
          payload: input,
        })
      );
    });

    it('resolveQuery calls resolve_query with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { query_hash: HASH, result: 'resolved' };

      await client.resolveQuery(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'resolve_query',
          payload: input,
        })
      );
    });

    it('broadcastEvent calls broadcast_event with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { event_type: 'MilestoneReached', data: '{"member":"alice"}' };

      await client.broadcastEvent(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'broadcast_event',
          payload: input,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Cross-Cluster Dispatch
  // --------------------------------------------------------------------------

  describe('cross-cluster dispatch', () => {
    it('dispatchPersonalCall calls dispatch_personal_call with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_DISPATCH_RESULT);
      const input = { target_zome: 'identity_profiles', fn_name: 'get_profile', payload: AGENT };

      await client.dispatchPersonalCall(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'dispatch_personal_call',
          payload: input,
        })
      );
    });

    it('dispatchIdentityCall calls dispatch_identity_call with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_DISPATCH_RESULT);
      const input = { target_zome: 'identity_credentials', fn_name: 'verify', payload: 'did:mycelix:alice' };

      await client.dispatchIdentityCall(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'dispatch_identity_call',
          payload: input,
        })
      );
    });

    it('dispatchCommonsCall calls dispatch_commons_call with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_DISPATCH_RESULT);
      const input = { target_zome: 'mutualaid_timebank', fn_name: 'get_balance', payload: AGENT };

      await client.dispatchCommonsCall(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'dispatch_commons_call',
          payload: input,
        })
      );
    });

    it('dispatchCivicCall calls dispatch_civic_call with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_DISPATCH_RESULT);
      const input = { target_zome: 'emergency_alerts', fn_name: 'get_active', payload: null };

      await client.dispatchCivicCall(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'dispatch_civic_call',
          payload: input,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Convenience Cross-Cluster Methods
  // --------------------------------------------------------------------------

  describe('convenience cross-cluster methods', () => {
    it('verifyMemberIdentity calls verify_member_identity with agent', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_DISPATCH_RESULT);

      await client.verifyMemberIdentity(AGENT);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'verify_member_identity',
          payload: AGENT,
        })
      );
    });

    it('escalateEmergency calls escalate_emergency with string', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_DISPATCH_RESULT);

      await client.escalateEmergency('House fire at 123 Main St');

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'escalate_emergency',
          payload: 'House fire at 123 Main St',
        })
      );
    });

    it('queryTimebankBalance calls query_timebank_balance with agent', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_DISPATCH_RESULT);

      await client.queryTimebankBalance(AGENT);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'query_timebank_balance',
          payload: AGENT,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Governance & Consciousness
  // --------------------------------------------------------------------------

  describe('governance and consciousness', () => {
    it('getConsciousnessCredential calls get_consciousness_credential with did string', async () => {
      const credential = { did: 'did:mycelix:alice', level: 3, score: 0.85 };
      mockClient.callZome.mockResolvedValueOnce(credential);

      const result = await client.getConsciousnessCredential('did:mycelix:alice');

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_consciousness_credential',
          payload: 'did:mycelix:alice',
        })
      );
      expect(result).toEqual(credential);
    });

    it('logGovernanceGate calls log_governance_gate with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(undefined);
      const input = { gate_type: 'DecisionVote', decision_hash: HASH, passed: true };

      await client.logGovernanceGate(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'log_governance_gate',
          payload: input,
        })
      );
    });

    it('getGovernanceAuditTrail calls get_governance_audit_trail with filter', async () => {
      const auditResult = { entries: [], total: 0 };
      mockClient.callZome.mockResolvedValueOnce(auditResult);
      const filter = { gate_type: 'DecisionVote', limit: 50 };

      const result = await client.getGovernanceAuditTrail(filter as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_governance_audit_trail',
          payload: filter,
        })
      );
      expect(result).toEqual(auditResult);
    });
  });

  // --------------------------------------------------------------------------
  // Events & Queries
  // --------------------------------------------------------------------------

  describe('events and queries', () => {
    it('getDomainEvents calls get_domain_events with domain string', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getDomainEvents('kinship');

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_domain_events',
          payload: 'kinship',
        })
      );
    });

    it('getAllEvents calls get_all_events with null payload', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getAllEvents();

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'hearth_bridge',
          fn_name: 'get_all_events',
          payload: null,
        })
      );
    });

    it('getEventsByType calls get_events_by_type with query', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);
      const query = { event_type: 'MilestoneReached', limit: 10 };

      await client.getEventsByType(query as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_events_by_type',
          payload: query,
        })
      );
    });

    it('getMyQueries calls get_my_queries with null payload', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getMyQueries();

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_my_queries',
          payload: null,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Lifecycle
  // --------------------------------------------------------------------------

  describe('lifecycle', () => {
    it('initiateSeverance calls initiate_severance with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, reason: 'Moving away', effective_date: 1709000000 };

      await client.initiateSeverance(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'initiate_severance',
          payload: input,
        })
      );
    });

    it('hearthSync calls hearth_sync with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, sync_type: 'Full' };

      await client.hearthSync(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'hearth_sync',
          payload: input,
        })
      );
    });

    it('getWeeklyDigests calls get_weekly_digests with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getWeeklyDigests(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_weekly_digests',
          payload: HASH,
        })
      );
    });

    it('healthCheck calls health_check with null payload', async () => {
      const health = { status: 'healthy', connected_zomes: 10 };
      mockClient.callZome.mockResolvedValueOnce(health);

      const result = await client.healthCheck();

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'hearth_bridge',
          fn_name: 'health_check',
          payload: null,
        })
      );
      expect(result).toEqual(health);
    });
  });
});
