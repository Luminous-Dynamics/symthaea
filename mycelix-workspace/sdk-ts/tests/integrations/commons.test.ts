/**
 * Commons Cluster Bridge Integration Tests
 *
 * Tests for CommonsBridgeClient -- the cluster-level SDK client for the
 * mycelix-commons DNA which unifies property, housing, care, mutualaid,
 * water, food, and transport domains into a single Holochain DNA.
 *
 * Covers cross-domain dispatch, audited queries, event broadcasting,
 * cross-cluster (commons->civic) calls, typed convenience functions,
 * audit trail, health checks, and signal type guards.
 *
 * All calls are dispatched through the commons role to the commons_bridge
 * coordinator zome.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  CommonsBridgeClient,
  createCommonsBridgeClient,
  COMMONS_DOMAINS,
  COMMONS_ZOMES,
  isCommonsBridgeSignal,
  type DispatchInput,
  type DispatchResult,
  type CommonsQueryInput,
  type CommonsEventInput,
  type BridgeHealth,
  type EventTypeQuery,
  type CrossClusterDispatchInput,
  type CheckEmergencyForAreaInput,
  type EmergencyAreaCheckResult,
  type CheckJusticeDisputesInput,
  type JusticeDisputeCheckResult,
  type PropertyOwnershipQuery,
  type PropertyOwnershipResult,
  type CareAvailabilityQuery,
  type CareAvailabilityResult,
  type AuditTrailQuery,
  type AuditTrailResult,
  type AuditTrailEntry,
  type CommonsBridgeEventSignal,
} from '../../src/integrations/commons/index.js';

// ============================================================================
// Mock Holochain client
// ============================================================================

function createMockClient() {
  return {
    callZome: vi.fn().mockResolvedValue({}),
  };
}

// ============================================================================
// Test helpers
// ============================================================================

function fakeAgentKey(): Uint8Array {
  return new Uint8Array(39).fill(0xca);
}

function fakeActionHash(): Uint8Array {
  return new Uint8Array(39).fill(0xab);
}

function fakePayload(): Uint8Array {
  return new Uint8Array([0x01, 0x02, 0x03, 0x04]);
}

// ============================================================================
// Constants
// ============================================================================

describe('Commons Constants', () => {
  it('should export all commons domain names', () => {
    expect(COMMONS_DOMAINS).toEqual([
      'property', 'housing', 'care', 'mutualaid', 'water', 'food', 'transport', 'support', 'space',
    ]);
    expect(COMMONS_DOMAINS).toHaveLength(9);
  });

  it('should export all commons zome names', () => {
    expect(COMMONS_ZOMES).toContain('property_registry');
    expect(COMMONS_ZOMES).toContain('housing_clt');
    expect(COMMONS_ZOMES).toContain('care_timebank');
    expect(COMMONS_ZOMES).toContain('mutualaid_needs');
    expect(COMMONS_ZOMES).toContain('water_flow');
    expect(COMMONS_ZOMES).toContain('food_production');
    expect(COMMONS_ZOMES).toContain('transport_routes');
    expect(COMMONS_ZOMES).toContain('support_knowledge');
    expect(COMMONS_ZOMES).toContain('space');
    // 4 property + 6 housing + 5 care + 7 mutualaid + 5 water + 4 food + 3 transport + 3 support + 1 space = 38
    expect(COMMONS_ZOMES).toHaveLength(38);
  });
});

// ============================================================================
// Type construction tests
// ============================================================================

describe('Commons Types', () => {
  describe('CommonsQueryInput', () => {
    it('should construct with all valid domain variants', () => {
      const domains: CommonsQueryInput['domain'][] = [
        'property', 'housing', 'care', 'mutualaid', 'water', 'food', 'transport',
      ];
      domains.forEach((d) => {
        const input: CommonsQueryInput = {
          domain: d,
          query_type: 'test_query',
          params: '{}',
        };
        expect(input.domain).toBe(d);
      });
    });
  });

  describe('CommonsEventInput', () => {
    it('should construct with optional related_hashes', () => {
      const input: CommonsEventInput = {
        domain: 'care',
        event_type: 'match_completed',
        payload: JSON.stringify({ match_id: '123' }),
        related_hashes: ['hash1', 'hash2'],
      };
      expect(input.related_hashes).toHaveLength(2);
    });

    it('should allow omitting related_hashes', () => {
      const input: CommonsEventInput = {
        domain: 'property',
        event_type: 'transfer_completed',
        payload: '{}',
      };
      expect(input.related_hashes).toBeUndefined();
    });
  });

  describe('BridgeHealth', () => {
    it('should construct a valid BridgeHealth', () => {
      const health: BridgeHealth = {
        healthy: true,
        agent: 'agent-123',
        total_events: 42,
        total_queries: 15,
        domains: ['property', 'housing', 'care', 'mutualaid', 'water', 'food', 'transport'],
      };
      expect(health.healthy).toBe(true);
      expect(health.domains).toHaveLength(7);
    });
  });

  describe('DispatchResult', () => {
    it('should construct a successful result', () => {
      const result: DispatchResult = {
        success: true,
        response: new Uint8Array([0x01, 0x02]),
      };
      expect(result.success).toBe(true);
      expect(result.response).toBeInstanceOf(Uint8Array);
    });

    it('should construct a failed result with error', () => {
      const result: DispatchResult = {
        success: false,
        error: 'Target zome not found',
      };
      expect(result.success).toBe(false);
      expect(result.error).toBe('Target zome not found');
    });
  });

  describe('AuditTrailQuery', () => {
    it('should construct with optional domain and event_type filters', () => {
      const query: AuditTrailQuery = {
        from_us: 1000000,
        to_us: 2000000,
        domain: 'property',
        event_type: 'transfer',
      };
      expect(query.domain).toBe('property');
      expect(query.event_type).toBe('transfer');
    });

    it('should allow omitting optional filters', () => {
      const query: AuditTrailQuery = {
        from_us: 1000000,
        to_us: 2000000,
      };
      expect(query.domain).toBeUndefined();
      expect(query.event_type).toBeUndefined();
    });
  });

  describe('EmergencyAreaCheckResult', () => {
    it('should construct with active emergencies', () => {
      const result: EmergencyAreaCheckResult = {
        has_active_emergencies: true,
        active_count: 3,
        recommendation: 'Avoid area',
      };
      expect(result.has_active_emergencies).toBe(true);
      expect(result.active_count).toBe(3);
    });

    it('should construct with optional error', () => {
      const result: EmergencyAreaCheckResult = {
        has_active_emergencies: false,
        active_count: 0,
        recommendation: 'Area is clear',
        error: 'Partial data available',
      };
      expect(result.error).toBe('Partial data available');
    });
  });

  describe('PropertyOwnershipResult', () => {
    it('should construct an ownership result', () => {
      const result: PropertyOwnershipResult = {
        is_owner: true,
        owner_did: 'did:key:z6Mk...',
      };
      expect(result.is_owner).toBe(true);
      expect(result.owner_did).toBe('did:key:z6Mk...');
    });
  });

  describe('CareAvailabilityResult', () => {
    it('should construct an availability result', () => {
      const result: CareAvailabilityResult = {
        available_count: 5,
        recommendation: 'Multiple providers available',
      };
      expect(result.available_count).toBe(5);
    });
  });
});

// ============================================================================
// CommonsBridgeClient -- Factory
// ============================================================================

describe('CommonsBridgeClient', () => {
  let client: ReturnType<typeof createMockClient>;
  let bridge: CommonsBridgeClient;

  beforeEach(() => {
    client = createMockClient();
    bridge = createCommonsBridgeClient(client);
  });

  describe('factory', () => {
    it('should create via createCommonsBridgeClient', () => {
      expect(bridge).toBeInstanceOf(CommonsBridgeClient);
    });

    it('should also be constructable directly', () => {
      const direct = new CommonsBridgeClient(client);
      expect(direct).toBeInstanceOf(CommonsBridgeClient);
    });
  });

  // ==========================================================================
  // Cross-Domain Dispatch
  // ==========================================================================

  describe('Cross-Domain Dispatch', () => {
    describe('dispatch', () => {
      it('should call commons_bridge.dispatch_call with zome, fn_name, and serialized payload', async () => {
        const payload = fakePayload();
        const mockResult: DispatchResult = { success: true, response: new Uint8Array([0x05]) };
        client.callZome.mockResolvedValue(mockResult);

        const result = await bridge.dispatch('property_registry', 'get_asset', payload);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'dispatch_call',
          payload: {
            zome: 'property_registry',
            fn_name: 'get_asset',
            payload: Array.from(payload),
          },
        });
        expect(result.success).toBe(true);
      });

      it('should convert Uint8Array payload to Array.from for serialization', async () => {
        const payload = new Uint8Array([0xaa, 0xbb, 0xcc]);
        await bridge.dispatch('housing_units', 'get_unit', payload);

        const lastCall = client.callZome.mock.calls[0][0];
        expect(lastCall.payload.payload).toEqual([0xaa, 0xbb, 0xcc]);
        expect(Array.isArray(lastCall.payload.payload)).toBe(true);
      });
    });
  });

  // ==========================================================================
  // Audited Queries
  // ==========================================================================

  describe('Audited Queries', () => {
    describe('query', () => {
      it('should call commons_bridge.query_commons with expanded payload', async () => {
        const input: CommonsQueryInput = {
          domain: 'housing',
          query_type: 'get_clt_lease',
          params: JSON.stringify({ unit_id: 'unit-001' }),
        };
        client.callZome.mockResolvedValue({});

        await bridge.query(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'query_commons',
          payload: {
            domain: 'housing',
            query_type: 'get_clt_lease',
            requester: null,
            params: JSON.stringify({ unit_id: 'unit-001' }),
            result: null,
            created_at: null,
            resolved_at: null,
            success: null,
          },
        });
      });

      it('should set requester, result, created_at, resolved_at, and success to null', async () => {
        const input: CommonsQueryInput = {
          domain: 'water',
          query_type: 'get_flow',
          params: '{}',
        };
        await bridge.query(input);

        const payload = client.callZome.mock.calls[0][0].payload;
        expect(payload.requester).toBeNull();
        expect(payload.result).toBeNull();
        expect(payload.created_at).toBeNull();
        expect(payload.resolved_at).toBeNull();
        expect(payload.success).toBeNull();
      });
    });

    describe('resolveQuery', () => {
      it('should call commons_bridge.resolve_query with query hash, result, and success', async () => {
        const queryHash = fakeActionHash();
        client.callZome.mockResolvedValue({});

        await bridge.resolveQuery(queryHash, 'query result data', true);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'resolve_query',
          payload: { query_hash: queryHash, result: 'query result data', success: true },
        });
      });

      it('should support resolving a failed query', async () => {
        const queryHash = fakeActionHash();
        await bridge.resolveQuery(queryHash, 'error: not found', false);

        const payload = client.callZome.mock.calls[0][0].payload;
        expect(payload.success).toBe(false);
        expect(payload.result).toBe('error: not found');
      });
    });

    describe('getDomainQueries', () => {
      it('should call commons_bridge.get_domain_queries with domain string', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await bridge.getDomainQueries('property');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'get_domain_queries',
          payload: 'property',
        });
        expect(result).toEqual([]);
      });
    });

    describe('getMyQueries', () => {
      it('should call commons_bridge.get_my_queries on both roles and merge results', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await bridge.getMyQueries();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'get_my_queries',
          payload: null,
        });
        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'commons_bridge',
          fn_name: 'get_my_queries',
          payload: null,
        });
        expect(result).toEqual([]);
      });
    });
  });

  // ==========================================================================
  // Event Broadcasting
  // ==========================================================================

  describe('Event Broadcasting', () => {
    describe('broadcastEvent', () => {
      it('should call commons_bridge.broadcast_event with expanded payload', async () => {
        const input: CommonsEventInput = {
          domain: 'care',
          event_type: 'match_completed',
          payload: JSON.stringify({ match_id: 'm-001' }),
          related_hashes: ['hash1'],
        };
        client.callZome.mockResolvedValue({});

        await bridge.broadcastEvent(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'commons_bridge',
          fn_name: 'broadcast_event',
          payload: {
            domain: 'care',
            event_type: 'match_completed',
            source_agent: null,
            payload: JSON.stringify({ match_id: 'm-001' }),
            created_at: null,
            related_hashes: ['hash1'],
          },
        });
      });

      it('should default related_hashes to empty array when omitted', async () => {
        const input: CommonsEventInput = {
          domain: 'food',
          event_type: 'harvest_recorded',
          payload: '{}',
        };
        await bridge.broadcastEvent(input);

        const payload = client.callZome.mock.calls[0][0].payload;
        expect(payload.related_hashes).toEqual([]);
      });

      it('should set source_agent and created_at to null (filled by zome)', async () => {
        const input: CommonsEventInput = {
          domain: 'transport',
          event_type: 'trip_logged',
          payload: '{}',
        };
        await bridge.broadcastEvent(input);

        const payload = client.callZome.mock.calls[0][0].payload;
        expect(payload.source_agent).toBeNull();
        expect(payload.created_at).toBeNull();
      });
    });

    describe('getDomainEvents', () => {
      it('should call commons_bridge.get_domain_events with domain string', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await bridge.getDomainEvents('water');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'get_domain_events',
          payload: 'water',
        });
        expect(result).toEqual([]);
      });
    });

    describe('getEventsByType', () => {
      it('should call commons_bridge.get_events_by_type with query', async () => {
        const query: EventTypeQuery = { domain: 'property', event_type: 'transfer' };
        client.callZome.mockResolvedValue([]);

        const result = await bridge.getEventsByType(query);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'get_events_by_type',
          payload: query,
        });
        expect(result).toEqual([]);
      });
    });

    describe('getAllEvents', () => {
      it('should call commons_bridge.get_all_events on both roles', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await bridge.getAllEvents();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'get_all_events',
          payload: null,
        });
        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'commons_bridge',
          fn_name: 'get_all_events',
          payload: null,
        });
        expect(result).toEqual([]);
      });
    });

    describe('getMyEvents', () => {
      it('should call commons_bridge.get_my_events on both roles', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await bridge.getMyEvents();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'get_my_events',
          payload: null,
        });
        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'commons_bridge',
          fn_name: 'get_my_events',
          payload: null,
        });
        expect(result).toEqual([]);
      });
    });
  });

  // ==========================================================================
  // Cross-Cluster (Commons -> Civic)
  // ==========================================================================

  describe('Cross-Cluster (Commons -> Civic)', () => {
    describe('dispatchCivicCall', () => {
      it('should call commons_bridge.dispatch_civic_call with civic role, zome, fn_name, and serialized payload', async () => {
        const payload = fakePayload();
        const mockResult: DispatchResult = { success: true };
        client.callZome.mockResolvedValue(mockResult);

        const result = await bridge.dispatchCivicCall('justice_cases', 'file_case', payload);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'dispatch_civic_call',
          payload: {
            role: 'civic',
            zome: 'justice_cases',
            fn_name: 'file_case',
            payload: Array.from(payload),
          },
        });
        expect(result.success).toBe(true);
      });
    });

    describe('checkEmergencyForArea', () => {
      it('should call commons_bridge.check_emergency_for_area with lat/lon', async () => {
        const input: CheckEmergencyForAreaInput = { lat: 32.95, lon: -96.75 };
        const mockResult: EmergencyAreaCheckResult = {
          has_active_emergencies: false,
          active_count: 0,
          recommendation: 'Area is clear',
        };
        client.callZome.mockResolvedValue(mockResult);

        const result = await bridge.checkEmergencyForArea(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'check_emergency_for_area',
          payload: input,
        });
        expect(result.has_active_emergencies).toBe(false);
        expect(result.active_count).toBe(0);
      });
    });

    describe('checkJusticeDisputesForProperty', () => {
      it('should call commons_bridge.check_justice_disputes_for_property with resource_id', async () => {
        const input: CheckJusticeDisputesInput = { resource_id: 'prop-001' };
        const mockResult: JusticeDisputeCheckResult = {
          has_pending_cases: true,
          recommendation: 'Proceed with caution',
        };
        client.callZome.mockResolvedValue(mockResult);

        const result = await bridge.checkJusticeDisputesForProperty(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'check_justice_disputes_for_property',
          payload: input,
        });
        expect(result.has_pending_cases).toBe(true);
      });
    });
  });

  // ==========================================================================
  // Typed Convenience Functions
  // ==========================================================================

  describe('Typed Convenience Functions', () => {
    describe('verifyPropertyOwnership', () => {
      it('should call commons_bridge.verify_property_ownership with query', async () => {
        const input: PropertyOwnershipQuery = {
          property_id: 'prop-001',
          requester_did: 'did:key:z6Mk...',
        };
        const mockResult: PropertyOwnershipResult = {
          is_owner: true,
          owner_did: 'did:key:z6Mk...',
        };
        client.callZome.mockResolvedValue(mockResult);

        const result = await bridge.verifyPropertyOwnership(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'verify_property_ownership',
          payload: input,
        });
        expect(result.is_owner).toBe(true);
      });
    });

    describe('checkCareAvailability', () => {
      it('should call commons_bridge.check_care_availability with query', async () => {
        const input: CareAvailabilityQuery = {
          skill_needed: 'nursing',
          location: 'Richardson, TX',
        };
        const mockResult: CareAvailabilityResult = {
          available_count: 3,
          recommendation: 'Multiple providers available',
        };
        client.callZome.mockResolvedValue(mockResult);

        const result = await bridge.checkCareAvailability(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_care',
          zome_name: 'commons_bridge',
          fn_name: 'check_care_availability',
          payload: input,
        });
        expect(result.available_count).toBe(3);
      });

      it('should allow omitting optional location', async () => {
        const input: CareAvailabilityQuery = { skill_needed: 'counseling' };
        await bridge.checkCareAvailability(input);

        const payload = client.callZome.mock.calls[0][0].payload;
        expect(payload.location).toBeUndefined();
      });
    });
  });

  // ==========================================================================
  // Audit Trail
  // ==========================================================================

  describe('Audit Trail', () => {
    describe('queryAuditTrail', () => {
      it('should call commons_bridge.query_audit_trail with time range and optional filters', async () => {
        const query: AuditTrailQuery = {
          from_us: 1000000,
          to_us: 2000000,
          domain: 'property',
          event_type: 'transfer',
        };
        const mockResult: AuditTrailResult = {
          entries: [],
          total_matched: 0,
          query_from_us: 1000000,
          query_to_us: 2000000,
        };
        client.callZome.mockResolvedValue(mockResult);

        const result = await bridge.queryAuditTrail(query);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'query_audit_trail',
          payload: {
            from_us: 1000000,
            to_us: 2000000,
            domain: 'property',
            event_type: 'transfer',
          },
        });
        expect(result.total_matched).toBe(0);
      });

      it('should default optional domain and event_type to null when omitted', async () => {
        const query: AuditTrailQuery = {
          from_us: 1000000,
          to_us: 2000000,
        };
        await bridge.queryAuditTrail(query);

        const payload = client.callZome.mock.calls[0][0].payload;
        expect(payload.domain).toBeNull();
        expect(payload.event_type).toBeNull();
      });
    });
  });

  // ==========================================================================
  // Health
  // ==========================================================================

  describe('Health', () => {
    describe('healthCheck', () => {
      it('should call commons_bridge.health_check with null payload', async () => {
        const mockHealth: BridgeHealth = {
          healthy: true,
          agent: 'agent-001',
          total_events: 100,
          total_queries: 50,
          domains: ['property', 'housing', 'care', 'mutualaid', 'water', 'food', 'transport'],
        };
        client.callZome.mockResolvedValue(mockHealth);

        const result = await bridge.healthCheck();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'commons_land',
          zome_name: 'commons_bridge',
          fn_name: 'health_check',
          payload: null,
        });
        expect(result.healthy).toBe(true);
        expect(result.domains).toHaveLength(7);
      });
    });
  });

  // ==========================================================================
  // All calls use commons role and commons_bridge zome
  // ==========================================================================

  describe('All calls dispatch through commons_land or commons_care role and commons_bridge zome', () => {
    it('should use correct role and commons_bridge zome for every method', async () => {
      // Call one method from each category
      client.callZome.mockResolvedValue([]);
      await bridge.dispatch('property_registry', 'get_asset', fakePayload());
      await bridge.query({ domain: 'housing', query_type: 'test', params: '{}' });
      await bridge.broadcastEvent({ domain: 'care', event_type: 'test', payload: '{}' });
      await bridge.getDomainEvents('water');
      await bridge.getMyQueries();
      await bridge.getAllEvents();
      await bridge.getMyEvents();
      await bridge.healthCheck();

      for (const call of client.callZome.mock.calls) {
        const roleName = (call[0] as { role_name: string }).role_name;
        expect(['commons_land', 'commons_care']).toContain(roleName);
        expect((call[0] as { zome_name: string }).zome_name).toBe('commons_bridge');
      }
    });
  });

  // ==========================================================================
  // Error propagation
  // ==========================================================================

  describe('Error propagation', () => {
    it('should propagate zome call errors for dispatch', async () => {
      client.callZome.mockRejectedValue(new Error('Conductor unavailable'));
      await expect(bridge.dispatch('property_registry', 'get_asset', fakePayload())).rejects.toThrow('Conductor unavailable');
    });

    it('should propagate zome call errors for query', async () => {
      client.callZome.mockRejectedValue(new Error('Validation failed'));
      await expect(bridge.query({ domain: 'housing', query_type: 'test', params: '{}' })).rejects.toThrow('Validation failed');
    });

    it('should propagate zome call errors for broadcastEvent', async () => {
      client.callZome.mockRejectedValue(new Error('Rate limited'));
      await expect(bridge.broadcastEvent({ domain: 'care', event_type: 'test', payload: '{}' })).rejects.toThrow('Rate limited');
    });

    it('should propagate zome call errors for dispatchCivicCall', async () => {
      client.callZome.mockRejectedValue(new Error('Cross-cluster unreachable'));
      await expect(bridge.dispatchCivicCall('justice_cases', 'file_case', fakePayload())).rejects.toThrow('Cross-cluster unreachable');
    });

    it('should propagate zome call errors for healthCheck', async () => {
      client.callZome.mockRejectedValue(new Error('Network error'));
      await expect(bridge.healthCheck()).rejects.toThrow('Network error');
    });

    it('should propagate zome call errors for checkEmergencyForArea', async () => {
      client.callZome.mockRejectedValue(new Error('Timeout'));
      await expect(bridge.checkEmergencyForArea({ lat: 0, lon: 0 })).rejects.toThrow('Timeout');
    });

    it('should propagate zome call errors for verifyPropertyOwnership', async () => {
      client.callZome.mockRejectedValue(new Error('Not authorized'));
      await expect(
        bridge.verifyPropertyOwnership({ property_id: 'x', requester_did: 'y' }),
      ).rejects.toThrow('Not authorized');
    });

    it('should propagate zome call errors for queryAuditTrail', async () => {
      client.callZome.mockRejectedValue(new Error('Invalid time range'));
      await expect(
        bridge.queryAuditTrail({ from_us: 0, to_us: 100 }),
      ).rejects.toThrow('Invalid time range');
    });
  });
});

// ============================================================================
// Signal Type Guards
// ============================================================================

describe('isCommonsBridgeSignal', () => {
  it('should return true for valid commons bridge event signals', () => {
    const signal: CommonsBridgeEventSignal = {
      signal_type: 'commons_bridge_event',
      domain: 'property',
      event_type: 'transfer_completed',
      payload: '{}',
      action_hash: fakeActionHash(),
    };
    expect(isCommonsBridgeSignal(signal)).toBe(true);
  });

  it('should return false for null', () => {
    expect(isCommonsBridgeSignal(null)).toBe(false);
  });

  it('should return false for undefined', () => {
    expect(isCommonsBridgeSignal(undefined)).toBe(false);
  });

  it('should return false for non-object types', () => {
    expect(isCommonsBridgeSignal('string')).toBe(false);
    expect(isCommonsBridgeSignal(42)).toBe(false);
    expect(isCommonsBridgeSignal(true)).toBe(false);
  });

  it('should return false for objects without signal_type', () => {
    expect(isCommonsBridgeSignal({ domain: 'property' })).toBe(false);
  });

  it('should return false for objects with wrong signal_type', () => {
    expect(isCommonsBridgeSignal({ signal_type: 'civic_bridge_event' })).toBe(false);
    expect(isCommonsBridgeSignal({ signal_type: 'something_else' })).toBe(false);
  });
});
