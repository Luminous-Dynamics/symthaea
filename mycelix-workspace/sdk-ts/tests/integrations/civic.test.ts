/**
 * Civic Cluster Bridge Integration Tests
 *
 * Tests for CivicBridgeClient -- the cluster-level SDK client for the
 * mycelix-civic DNA which unifies justice, emergency, and media domains
 * into a single Holochain DNA with cross-domain dispatch.
 *
 * Covers cross-domain dispatch, audited queries, event broadcasting,
 * cross-cluster (civic->commons) calls, typed convenience functions,
 * audit trail, health checks, and signal type guards.
 *
 * All calls are dispatched through the civic role to the civic_bridge
 * coordinator zome.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  CivicBridgeClient,
  createCivicBridgeClient,
  CIVIC_DOMAINS,
  CIVIC_ZOMES,
  isCivicBridgeSignal,
  type DispatchInput,
  type DispatchResult,
  type CivicQueryInput,
  type CivicEventInput,
  type BridgeHealth,
  type EventTypeQuery,
  type CrossClusterDispatchInput,
  type QueryPropertyForEnforcementInput,
  type PropertyEnforcementResult,
  type CheckHousingCapacityInput,
  type HousingCapacityResult,
  type VerifyCareCredentialsInput,
  type CareCredentialVerifyResult,
  type JusticeAreaQuery,
  type JusticeAreaResult,
  type FactcheckStatusQuery,
  type FactcheckStatusResult,
  type AuditTrailQuery,
  type AuditTrailResult,
  type AuditTrailEntry,
  type CivicBridgeEventSignal,
} from '../../src/integrations/civic/index.js';

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

describe('Civic Constants', () => {
  it('should export all 3 civic domain names', () => {
    expect(CIVIC_DOMAINS).toEqual(['justice', 'emergency', 'media']);
    expect(CIVIC_DOMAINS).toHaveLength(3);
  });

  it('should export all civic zome names', () => {
    expect(CIVIC_ZOMES).toContain('justice_cases');
    expect(CIVIC_ZOMES).toContain('justice_evidence');
    expect(CIVIC_ZOMES).toContain('justice_arbitration');
    expect(CIVIC_ZOMES).toContain('justice_restorative');
    expect(CIVIC_ZOMES).toContain('justice_enforcement');
    expect(CIVIC_ZOMES).toContain('emergency_incidents');
    expect(CIVIC_ZOMES).toContain('emergency_triage');
    expect(CIVIC_ZOMES).toContain('emergency_resources');
    expect(CIVIC_ZOMES).toContain('emergency_coordination');
    expect(CIVIC_ZOMES).toContain('emergency_shelters');
    expect(CIVIC_ZOMES).toContain('emergency_comms');
    expect(CIVIC_ZOMES).toContain('media_publication');
    expect(CIVIC_ZOMES).toContain('media_attribution');
    expect(CIVIC_ZOMES).toContain('media_factcheck');
    expect(CIVIC_ZOMES).toContain('media_curation');
    // 5 justice + 6 emergency + 4 media = 15
    expect(CIVIC_ZOMES).toHaveLength(15);
  });
});

// ============================================================================
// Type construction tests
// ============================================================================

describe('Civic Types', () => {
  describe('CivicQueryInput', () => {
    it('should construct with all valid domain variants', () => {
      const domains: CivicQueryInput['domain'][] = ['justice', 'emergency', 'media'];
      domains.forEach((d) => {
        const input: CivicQueryInput = {
          domain: d,
          query_type: 'test_query',
          params: '{}',
        };
        expect(input.domain).toBe(d);
      });
    });
  });

  describe('CivicEventInput', () => {
    it('should construct with optional related_hashes', () => {
      const input: CivicEventInput = {
        domain: 'justice',
        event_type: 'case_filed',
        payload: JSON.stringify({ case_id: '123' }),
        related_hashes: ['hash1', 'hash2'],
      };
      expect(input.related_hashes).toHaveLength(2);
    });

    it('should allow omitting related_hashes', () => {
      const input: CivicEventInput = {
        domain: 'emergency',
        event_type: 'incident_reported',
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
        total_events: 20,
        total_queries: 8,
        domains: ['justice', 'emergency', 'media'],
      };
      expect(health.healthy).toBe(true);
      expect(health.domains).toHaveLength(3);
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
    it('should construct with optional filters', () => {
      const query: AuditTrailQuery = {
        from_us: 1000000,
        to_us: 2000000,
        domain: 'justice',
        event_type: 'case_filed',
      };
      expect(query.domain).toBe('justice');
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

  describe('PropertyEnforcementResult', () => {
    it('should construct a valid result', () => {
      const result: PropertyEnforcementResult = {
        property_found: true,
        enforcement_advisory: 'Property confirmed, proceed with enforcement',
      };
      expect(result.property_found).toBe(true);
    });

    it('should support optional error', () => {
      const result: PropertyEnforcementResult = {
        property_found: false,
        enforcement_advisory: 'Cannot verify',
        error: 'Commons unreachable',
      };
      expect(result.error).toBe('Commons unreachable');
    });
  });

  describe('HousingCapacityResult', () => {
    it('should construct a valid result', () => {
      const result: HousingCapacityResult = {
        commons_reachable: true,
        recommendation: '15 units available for emergency sheltering',
      };
      expect(result.commons_reachable).toBe(true);
    });
  });

  describe('CareCredentialVerifyResult', () => {
    it('should construct a valid result', () => {
      const result: CareCredentialVerifyResult = {
        commons_reachable: true,
        recommendation: 'Credentials verified, admissible as evidence',
      };
      expect(result.commons_reachable).toBe(true);
    });
  });

  describe('JusticeAreaResult', () => {
    it('should construct a valid result', () => {
      const result: JusticeAreaResult = {
        active_cases: 5,
        recommendation: 'Moderate activity in area',
      };
      expect(result.active_cases).toBe(5);
    });
  });

  describe('FactcheckStatusResult', () => {
    it('should construct with verdict', () => {
      const result: FactcheckStatusResult = {
        has_factcheck: true,
        verdict: 'verified',
      };
      expect(result.has_factcheck).toBe(true);
      expect(result.verdict).toBe('verified');
    });

    it('should construct without verdict', () => {
      const result: FactcheckStatusResult = {
        has_factcheck: false,
      };
      expect(result.verdict).toBeUndefined();
    });
  });
});

// ============================================================================
// CivicBridgeClient -- Factory
// ============================================================================

describe('CivicBridgeClient', () => {
  let client: ReturnType<typeof createMockClient>;
  let bridge: CivicBridgeClient;

  beforeEach(() => {
    client = createMockClient();
    bridge = createCivicBridgeClient(client);
  });

  describe('factory', () => {
    it('should create via createCivicBridgeClient', () => {
      expect(bridge).toBeInstanceOf(CivicBridgeClient);
    });

    it('should also be constructable directly', () => {
      const direct = new CivicBridgeClient(client);
      expect(direct).toBeInstanceOf(CivicBridgeClient);
    });
  });

  // ==========================================================================
  // Cross-Domain Dispatch
  // ==========================================================================

  describe('Cross-Domain Dispatch', () => {
    describe('dispatch', () => {
      it('should call civic_bridge.dispatch_call with zome, fn_name, and serialized payload', async () => {
        const payload = fakePayload();
        const mockResult: DispatchResult = { success: true, response: new Uint8Array([0x05]) };
        client.callZome.mockResolvedValue(mockResult);

        const result = await bridge.dispatch('justice_cases', 'file_case', payload);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'dispatch_call',
          payload: {
            zome: 'justice_cases',
            fn_name: 'file_case',
            payload: Array.from(payload),
          },
        });
        expect(result.success).toBe(true);
      });

      it('should convert Uint8Array payload to Array.from for serialization', async () => {
        const payload = new Uint8Array([0xaa, 0xbb, 0xcc]);
        await bridge.dispatch('emergency_incidents', 'report_incident', payload);

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
      it('should call civic_bridge.query_civic with expanded payload', async () => {
        const input: CivicQueryInput = {
          domain: 'emergency',
          query_type: 'get_active_incidents',
          params: '{}',
        };
        client.callZome.mockResolvedValue({});

        await bridge.query(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'query_civic',
          payload: {
            domain: 'emergency',
            query_type: 'get_active_incidents',
            requester: null,
            params: '{}',
            result: null,
            created_at: null,
            resolved_at: null,
            success: null,
          },
        });
      });

      it('should set requester, result, created_at, resolved_at, and success to null', async () => {
        const input: CivicQueryInput = {
          domain: 'justice',
          query_type: 'get_case',
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
      it('should call civic_bridge.resolve_query with query hash, result, and success', async () => {
        const queryHash = fakeActionHash();
        client.callZome.mockResolvedValue({});

        await bridge.resolveQuery(queryHash, 'case data', true);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'resolve_query',
          payload: { query_hash: queryHash, result: 'case data', success: true },
        });
      });

      it('should support resolving a failed query', async () => {
        const queryHash = fakeActionHash();
        await bridge.resolveQuery(queryHash, 'error: not found', false);

        const payload = client.callZome.mock.calls[0][0].payload;
        expect(payload.success).toBe(false);
      });
    });

    describe('getMyQueries', () => {
      it('should call civic_bridge.get_my_queries with null payload', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await bridge.getMyQueries();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
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
      it('should call civic_bridge.broadcast_event with expanded payload', async () => {
        const input: CivicEventInput = {
          domain: 'media',
          event_type: 'factcheck_completed',
          payload: JSON.stringify({ article_id: 'art-001', verdict: 'verified' }),
          related_hashes: ['hash1'],
        };
        client.callZome.mockResolvedValue({});

        await bridge.broadcastEvent(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'broadcast_event',
          payload: {
            domain: 'media',
            event_type: 'factcheck_completed',
            source_agent: null,
            payload: JSON.stringify({ article_id: 'art-001', verdict: 'verified' }),
            created_at: null,
            related_hashes: ['hash1'],
          },
        });
      });

      it('should default related_hashes to empty array when omitted', async () => {
        const input: CivicEventInput = {
          domain: 'justice',
          event_type: 'case_resolved',
          payload: '{}',
        };
        await bridge.broadcastEvent(input);

        const payload = client.callZome.mock.calls[0][0].payload;
        expect(payload.related_hashes).toEqual([]);
      });

      it('should set source_agent and created_at to null (filled by zome)', async () => {
        const input: CivicEventInput = {
          domain: 'emergency',
          event_type: 'incident_closed',
          payload: '{}',
        };
        await bridge.broadcastEvent(input);

        const payload = client.callZome.mock.calls[0][0].payload;
        expect(payload.source_agent).toBeNull();
        expect(payload.created_at).toBeNull();
      });
    });

    describe('getDomainEvents', () => {
      it('should call civic_bridge.get_domain_events with domain string', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await bridge.getDomainEvents('justice');

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'get_domain_events',
          payload: 'justice',
        });
        expect(result).toEqual([]);
      });
    });

    describe('getEventsByType', () => {
      it('should call civic_bridge.get_events_by_type with query', async () => {
        const query: EventTypeQuery = { domain: 'media', event_type: 'factcheck_completed' };
        client.callZome.mockResolvedValue([]);

        const result = await bridge.getEventsByType(query);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'get_events_by_type',
          payload: query,
        });
        expect(result).toEqual([]);
      });
    });

    describe('getAllEvents', () => {
      it('should call civic_bridge.get_all_events with null payload', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await bridge.getAllEvents();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'get_all_events',
          payload: null,
        });
        expect(result).toEqual([]);
      });
    });

    describe('getMyEvents', () => {
      it('should call civic_bridge.get_my_events with null payload', async () => {
        client.callZome.mockResolvedValue([]);

        const result = await bridge.getMyEvents();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'get_my_events',
          payload: null,
        });
        expect(result).toEqual([]);
      });
    });
  });

  // ==========================================================================
  // Cross-Cluster (Civic -> Commons)
  // ==========================================================================

  describe('Cross-Cluster (Civic -> Commons)', () => {
    describe('dispatchCommonsCall', () => {
      it('should call civic_bridge.dispatch_commons_call with commons role, zome, fn_name, and serialized payload', async () => {
        const payload = fakePayload();
        const mockResult: DispatchResult = { success: true };
        client.callZome.mockResolvedValue(mockResult);

        const result = await bridge.dispatchCommonsCall('property_registry', 'get_asset', payload);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'dispatch_commons_call',
          payload: {
            role: 'commons',
            zome: 'property_registry',
            fn_name: 'get_asset',
            payload: Array.from(payload),
          },
        });
        expect(result.success).toBe(true);
      });
    });

    describe('queryPropertyForEnforcement', () => {
      it('should call civic_bridge.query_property_for_enforcement with property and case ids', async () => {
        const input: QueryPropertyForEnforcementInput = {
          property_id: 'prop-001',
          case_id: 'case-001',
        };
        const mockResult: PropertyEnforcementResult = {
          property_found: true,
          enforcement_advisory: 'Property confirmed, proceed',
        };
        client.callZome.mockResolvedValue(mockResult);

        const result = await bridge.queryPropertyForEnforcement(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'query_property_for_enforcement',
          payload: input,
        });
        expect(result.property_found).toBe(true);
      });
    });

    describe('checkHousingCapacityForSheltering', () => {
      it('should call civic_bridge.check_housing_capacity_for_sheltering with disaster and area', async () => {
        const input: CheckHousingCapacityInput = {
          disaster_id: 'disaster-001',
          area: 'Richardson, TX',
        };
        const mockResult: HousingCapacityResult = {
          commons_reachable: true,
          recommendation: '15 units available',
        };
        client.callZome.mockResolvedValue(mockResult);

        const result = await bridge.checkHousingCapacityForSheltering(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'check_housing_capacity_for_sheltering',
          payload: input,
        });
        expect(result.commons_reachable).toBe(true);
      });
    });

    describe('verifyCareCredentialsForEvidence', () => {
      it('should call civic_bridge.verify_care_credentials_for_evidence with provider and case', async () => {
        const input: VerifyCareCredentialsInput = {
          provider_did: 'did:key:z6Mk...',
          case_id: 'case-001',
        };
        const mockResult: CareCredentialVerifyResult = {
          commons_reachable: true,
          recommendation: 'Credentials verified',
        };
        client.callZome.mockResolvedValue(mockResult);

        const result = await bridge.verifyCareCredentialsForEvidence(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'verify_care_credentials_for_evidence',
          payload: input,
        });
        expect(result.commons_reachable).toBe(true);
      });
    });
  });

  // ==========================================================================
  // Typed Convenience Functions
  // ==========================================================================

  describe('Typed Convenience Functions', () => {
    describe('getActiveCasesForArea', () => {
      it('should call civic_bridge.get_active_cases_for_area with query', async () => {
        const input: JusticeAreaQuery = {
          area: 'Richardson, TX',
          case_type: 'civil',
        };
        const mockResult: JusticeAreaResult = {
          active_cases: 3,
          recommendation: 'Low activity',
        };
        client.callZome.mockResolvedValue(mockResult);

        const result = await bridge.getActiveCasesForArea(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'get_active_cases_for_area',
          payload: input,
        });
        expect(result.active_cases).toBe(3);
      });

      it('should allow omitting optional case_type', async () => {
        const input: JusticeAreaQuery = { area: 'Dallas, TX' };
        await bridge.getActiveCasesForArea(input);

        const payload = client.callZome.mock.calls[0][0].payload;
        expect(payload.case_type).toBeUndefined();
      });
    });

    describe('checkFactcheckStatus', () => {
      it('should call civic_bridge.check_factcheck_status with claim_id', async () => {
        const input: FactcheckStatusQuery = { claim_id: 'claim-001' };
        const mockResult: FactcheckStatusResult = {
          has_factcheck: true,
          verdict: 'verified',
        };
        client.callZome.mockResolvedValue(mockResult);

        const result = await bridge.checkFactcheckStatus(input);

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'check_factcheck_status',
          payload: input,
        });
        expect(result.has_factcheck).toBe(true);
        expect(result.verdict).toBe('verified');
      });
    });
  });

  // ==========================================================================
  // Audit Trail
  // ==========================================================================

  describe('Audit Trail', () => {
    describe('queryAuditTrail', () => {
      it('should call civic_bridge.query_audit_trail with time range and optional filters', async () => {
        const query: AuditTrailQuery = {
          from_us: 1000000,
          to_us: 2000000,
          domain: 'justice',
          event_type: 'case_filed',
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
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'query_audit_trail',
          payload: {
            from_us: 1000000,
            to_us: 2000000,
            domain: 'justice',
            event_type: 'case_filed',
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
      it('should call civic_bridge.health_check with null payload', async () => {
        const mockHealth: BridgeHealth = {
          healthy: true,
          agent: 'agent-001',
          total_events: 50,
          total_queries: 25,
          domains: ['justice', 'emergency', 'media'],
        };
        client.callZome.mockResolvedValue(mockHealth);

        const result = await bridge.healthCheck();

        expect(client.callZome).toHaveBeenCalledWith({
          role_name: 'civic',
          zome_name: 'civic_bridge',
          fn_name: 'health_check',
          payload: null,
        });
        expect(result.healthy).toBe(true);
        expect(result.domains).toHaveLength(3);
      });
    });
  });

  // ==========================================================================
  // All calls use civic role and civic_bridge zome
  // ==========================================================================

  describe('All calls dispatch through civic role and civic_bridge zome', () => {
    it('should use civic role and civic_bridge zome for every method', async () => {
      await bridge.dispatch('justice_cases', 'file_case', fakePayload());
      await bridge.query({ domain: 'emergency', query_type: 'test', params: '{}' });
      await bridge.broadcastEvent({ domain: 'media', event_type: 'test', payload: '{}' });
      await bridge.getDomainEvents('justice');
      await bridge.getMyQueries();
      await bridge.getAllEvents();
      await bridge.getMyEvents();
      await bridge.healthCheck();

      for (const call of client.callZome.mock.calls) {
        expect((call[0] as { role_name: string }).role_name).toBe('civic');
        expect((call[0] as { zome_name: string }).zome_name).toBe('civic_bridge');
      }
    });
  });

  // ==========================================================================
  // Error propagation
  // ==========================================================================

  describe('Error propagation', () => {
    it('should propagate zome call errors for dispatch', async () => {
      client.callZome.mockRejectedValue(new Error('Conductor unavailable'));
      await expect(bridge.dispatch('justice_cases', 'file_case', fakePayload())).rejects.toThrow('Conductor unavailable');
    });

    it('should propagate zome call errors for query', async () => {
      client.callZome.mockRejectedValue(new Error('Validation failed'));
      await expect(bridge.query({ domain: 'emergency', query_type: 'test', params: '{}' })).rejects.toThrow('Validation failed');
    });

    it('should propagate zome call errors for broadcastEvent', async () => {
      client.callZome.mockRejectedValue(new Error('Rate limited'));
      await expect(bridge.broadcastEvent({ domain: 'media', event_type: 'test', payload: '{}' })).rejects.toThrow('Rate limited');
    });

    it('should propagate zome call errors for dispatchCommonsCall', async () => {
      client.callZome.mockRejectedValue(new Error('Cross-cluster unreachable'));
      await expect(bridge.dispatchCommonsCall('property_registry', 'get_asset', fakePayload())).rejects.toThrow('Cross-cluster unreachable');
    });

    it('should propagate zome call errors for healthCheck', async () => {
      client.callZome.mockRejectedValue(new Error('Network error'));
      await expect(bridge.healthCheck()).rejects.toThrow('Network error');
    });

    it('should propagate zome call errors for queryPropertyForEnforcement', async () => {
      client.callZome.mockRejectedValue(new Error('Commons offline'));
      await expect(
        bridge.queryPropertyForEnforcement({ property_id: 'x', case_id: 'y' }),
      ).rejects.toThrow('Commons offline');
    });

    it('should propagate zome call errors for checkHousingCapacityForSheltering', async () => {
      client.callZome.mockRejectedValue(new Error('Timeout'));
      await expect(
        bridge.checkHousingCapacityForSheltering({ disaster_id: 'd1', area: 'TX' }),
      ).rejects.toThrow('Timeout');
    });

    it('should propagate zome call errors for checkFactcheckStatus', async () => {
      client.callZome.mockRejectedValue(new Error('Not authorized'));
      await expect(
        bridge.checkFactcheckStatus({ claim_id: 'c1' }),
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

describe('isCivicBridgeSignal', () => {
  it('should return true for valid civic bridge event signals', () => {
    const signal: CivicBridgeEventSignal = {
      signal_type: 'civic_bridge_event',
      domain: 'justice',
      event_type: 'case_filed',
      payload: '{}',
      action_hash: fakeActionHash(),
    };
    expect(isCivicBridgeSignal(signal)).toBe(true);
  });

  it('should return false for null', () => {
    expect(isCivicBridgeSignal(null)).toBe(false);
  });

  it('should return false for undefined', () => {
    expect(isCivicBridgeSignal(undefined)).toBe(false);
  });

  it('should return false for non-object types', () => {
    expect(isCivicBridgeSignal('string')).toBe(false);
    expect(isCivicBridgeSignal(42)).toBe(false);
    expect(isCivicBridgeSignal(true)).toBe(false);
  });

  it('should return false for objects without signal_type', () => {
    expect(isCivicBridgeSignal({ domain: 'justice' })).toBe(false);
  });

  it('should return false for objects with wrong signal_type', () => {
    expect(isCivicBridgeSignal({ signal_type: 'commons_bridge_event' })).toBe(false);
    expect(isCivicBridgeSignal({ signal_type: 'something_else' })).toBe(false);
  });
});
