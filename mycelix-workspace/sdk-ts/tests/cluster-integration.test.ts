/**
 * Cluster Integration Module Tests
 *
 * Tests for CommonsBridgeClient and CivicBridgeClient — the cluster-level
 * SDK clients that wrap cross-domain dispatch, audited queries, event
 * broadcasting, and health monitoring.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  CommonsBridgeClient,
  createCommonsBridgeClient,
  isCommonsBridgeSignal,
  COMMONS_DOMAINS,
  COMMONS_ZOMES,
  CivicBridgeClient,
  createCivicBridgeClient,
  isCivicBridgeSignal,
  CIVIC_DOMAINS,
  CIVIC_ZOMES,
} from '../src/integrations/index.js';

import type {
  CommonsQueryInput,
  CommonsEventInput,
  CommonsBridgeEventSignal,
  CivicQueryInput,
  CivicEventInput,
  CivicBridgeEventSignal,
  BridgeHealth,
  DispatchResult,
  CheckEmergencyForAreaInput,
  EmergencyAreaCheckResult,
  CheckJusticeDisputesInput,
  JusticeDisputeCheckResult,
  QueryPropertyForEnforcementInput,
  PropertyEnforcementResult,
  CheckHousingCapacityInput,
  HousingCapacityResult,
  VerifyCareCredentialsInput,
  CareCredentialVerifyResult,
  PropertyOwnershipQuery,
  PropertyOwnershipResult,
  CareAvailabilityQuery,
  CareAvailabilityResult,
  JusticeAreaQuery,
  JusticeAreaResult,
  FactcheckStatusQuery,
  FactcheckStatusResult,
  AuditTrailQuery,
  AuditTrailEntry,
  AuditTrailResult,
} from '../src/integrations/index.js';

// ============================================================================
// Mock Holochain client
// ============================================================================

function createMockClient() {
  return {
    callZome: vi.fn().mockResolvedValue({}),
  };
}

// ============================================================================
// Constants
// ============================================================================

describe('Cluster Constants', () => {
  it('should export all 9 commons domains', () => {
    expect(COMMONS_DOMAINS).toEqual(['property', 'housing', 'care', 'mutualaid', 'water', 'food', 'transport', 'support', 'space']);
    expect(COMMONS_DOMAINS).toHaveLength(9);
  });

  it('should export all 3 civic domains', () => {
    expect(CIVIC_DOMAINS).toEqual(['justice', 'emergency', 'media']);
    expect(CIVIC_DOMAINS).toHaveLength(3);
  });

  it('should export commons zomes with domain prefixes', () => {
    expect(COMMONS_ZOMES).toContain('property_registry');
    expect(COMMONS_ZOMES).toContain('housing_clt');
    expect(COMMONS_ZOMES).toContain('care_timebank');
    expect(COMMONS_ZOMES).toContain('mutualaid_needs');
    expect(COMMONS_ZOMES).toContain('water_flow');
    expect(COMMONS_ZOMES.length).toBeGreaterThanOrEqual(27);
  });

  it('should export civic zomes with domain prefixes', () => {
    expect(CIVIC_ZOMES).toContain('justice_cases');
    expect(CIVIC_ZOMES).toContain('emergency_incidents');
    expect(CIVIC_ZOMES).toContain('media_publication');
    expect(CIVIC_ZOMES.length).toBeGreaterThanOrEqual(15);
  });
});

// ============================================================================
// CommonsBridgeClient
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
  });

  describe('dispatch', () => {
    it('should call commons_bridge.dispatch_call with correct params', async () => {
      const payload = new Uint8Array([1, 2, 3]);
      const mockResult: DispatchResult = { success: true, response: new Uint8Array([4, 5]) };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.dispatch('property_registry', 'get_asset', payload);

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_land',
        zome_name: 'commons_bridge',
        fn_name: 'dispatch_call',
        payload: { zome: 'property_registry', fn_name: 'get_asset', payload: [1, 2, 3] },
      });
      expect(result).toEqual(mockResult);
    });
  });

  describe('query', () => {
    it('should submit an audited cross-domain query', async () => {
      const input: CommonsQueryInput = {
        domain: 'housing',
        query_type: 'get_clt_lease',
        params: '{"unit_id": "123"}',
      };

      await bridge.query(input);

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_land',
        zome_name: 'commons_bridge',
        fn_name: 'query_commons',
        payload: expect.objectContaining({
          domain: 'housing',
          query_type: 'get_clt_lease',
          params: '{"unit_id": "123"}',
        }),
      });
    });
  });

  describe('resolveQuery', () => {
    it('should resolve a pending query', async () => {
      const hash = new Uint8Array([10, 20, 30]);
      await bridge.resolveQuery(hash, '{"status":"ok"}', true);

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_land',
        zome_name: 'commons_bridge',
        fn_name: 'resolve_query',
        payload: { query_hash: hash, result: '{"status":"ok"}', success: true },
      });
    });
  });

  describe('event broadcasting', () => {
    it('should broadcast a cross-domain event', async () => {
      const input: CommonsEventInput = {
        domain: 'care',
        event_type: 'match_completed',
        payload: '{"match_id": "abc"}',
        related_hashes: ['hash1'],
      };

      await bridge.broadcastEvent(input);

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_care',
        zome_name: 'commons_bridge',
        fn_name: 'broadcast_event',
        payload: expect.objectContaining({
          domain: 'care',
          event_type: 'match_completed',
          payload: '{"match_id": "abc"}',
          related_hashes: ['hash1'],
        }),
      });
    });

    it('should default related_hashes to empty array', async () => {
      const input: CommonsEventInput = {
        domain: 'water',
        event_type: 'purity_alert',
        payload: '{}',
      };

      await bridge.broadcastEvent(input);

      const call = client.callZome.mock.calls[0][0];
      expect(call.payload.related_hashes).toEqual([]);
    });

    it('should get domain events', async () => {
      await bridge.getDomainEvents('property');
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_land',
        zome_name: 'commons_bridge',
        fn_name: 'get_domain_events',
        payload: 'property',
      });
    });

    it('should get events by type', async () => {
      await bridge.getEventsByType({ domain: 'mutualaid', event_type: 'resource_shared' });
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_care',
        zome_name: 'commons_bridge',
        fn_name: 'get_events_by_type',
        payload: { domain: 'mutualaid', event_type: 'resource_shared' },
      });
    });

    it('should get all events (merges both DNAs)', async () => {
      client.callZome.mockResolvedValue([]);
      await bridge.getAllEvents();
      expect(client.callZome).toHaveBeenCalledTimes(2);
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
    });

    it('should get my events (merges both DNAs)', async () => {
      client.callZome.mockResolvedValue([]);
      await bridge.getMyEvents();
      expect(client.callZome).toHaveBeenCalledTimes(2);
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
    });
  });

  describe('queries', () => {
    it('should get my queries (merges both DNAs)', async () => {
      client.callZome.mockResolvedValue([]);
      await bridge.getMyQueries();
      expect(client.callZome).toHaveBeenCalledTimes(2);
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
    });

    it('should get domain queries', async () => {
      await bridge.getDomainQueries('housing');
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_land',
        zome_name: 'commons_bridge',
        fn_name: 'get_domain_queries',
        payload: 'housing',
      });
    });
  });

  describe('cross-cluster dispatch (commons → civic)', () => {
    it('should dispatch a call to the civic DNA', async () => {
      const payload = new Uint8Array([7, 8, 9]);
      const mockResult: DispatchResult = { success: true, response: new Uint8Array([10]) };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.dispatchCivicCall('justice_cases', 'get_case', payload);

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_land',
        zome_name: 'commons_bridge',
        fn_name: 'dispatch_civic_call',
        payload: { role: 'civic', zome: 'justice_cases', fn_name: 'get_case', payload: [7, 8, 9] },
      });
      expect(result.success).toBe(true);
    });

    it('should check emergency for area', async () => {
      const mockResult: EmergencyAreaCheckResult = {
        has_active_emergencies: true,
        active_count: 2,
        recommendation: 'Defer transfer until emergencies resolved',
      };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.checkEmergencyForArea({ lat: 32.9, lon: -96.7 });

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_land',
        zome_name: 'commons_bridge',
        fn_name: 'check_emergency_for_area',
        payload: { lat: 32.9, lon: -96.7 },
      });
      expect(result.has_active_emergencies).toBe(true);
      expect(result.active_count).toBe(2);
    });

    it('should check justice disputes for property', async () => {
      const mockResult: JusticeDisputeCheckResult = {
        has_pending_cases: false,
        recommendation: 'No pending disputes — transfer may proceed',
      };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.checkJusticeDisputesForProperty({ resource_id: 'prop_123' });

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_land',
        zome_name: 'commons_bridge',
        fn_name: 'check_justice_disputes_for_property',
        payload: { resource_id: 'prop_123' },
      });
      expect(result.has_pending_cases).toBe(false);
    });
  });

  describe('typed convenience functions (intra-cluster)', () => {
    it('should verify property ownership', async () => {
      const mockResult: PropertyOwnershipResult = {
        is_owner: true,
        owner_did: 'did:mycelix:owner_abc',
      };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.verifyPropertyOwnership({
        property_id: 'PROP-001',
        requester_did: 'did:mycelix:requester_xyz',
      });

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_land',
        zome_name: 'commons_bridge',
        fn_name: 'verify_property_ownership',
        payload: { property_id: 'PROP-001', requester_did: 'did:mycelix:requester_xyz' },
      });
      expect(result.is_owner).toBe(true);
      expect(result.owner_did).toBe('did:mycelix:owner_abc');
    });

    it('should check care availability', async () => {
      const mockResult: CareAvailabilityResult = {
        available_count: 5,
        recommendation: '5 providers available for nursing in downtown',
      };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.checkCareAvailability({
        skill_needed: 'nursing',
        location: 'downtown',
      });

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_care',
        zome_name: 'commons_bridge',
        fn_name: 'check_care_availability',
        payload: { skill_needed: 'nursing', location: 'downtown' },
      });
      expect(result.available_count).toBe(5);
    });
  });

  describe('audit trail', () => {
    it('should query audit trail with time range only', async () => {
      const mockResult: AuditTrailResult = {
        entries: [
          {
            domain: 'property',
            event_type: 'ownership_transferred',
            source_agent: 'uhCAk...',
            payload_preview: '{"property_id":"P-1"}',
            created_at_us: 1700000000000000,
            action_hash: new Uint8Array([1, 2, 3]),
          },
        ],
        total_matched: 1,
        query_from_us: 1700000000000000,
        query_to_us: 1700001000000000,
      };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.queryAuditTrail({
        from_us: 1700000000000000,
        to_us: 1700001000000000,
      });

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_land',
        zome_name: 'commons_bridge',
        fn_name: 'query_audit_trail',
        payload: {
          from_us: 1700000000000000,
          to_us: 1700001000000000,
          domain: null,
          event_type: null,
        },
      });
      expect(result.entries).toHaveLength(1);
      expect(result.entries[0].domain).toBe('property');
      expect(result.total_matched).toBe(1);
    });

    it('should query audit trail with domain and event_type filters', async () => {
      const mockResult: AuditTrailResult = {
        entries: [],
        total_matched: 0,
        query_from_us: 1700000000000000,
        query_to_us: 1700001000000000,
      };
      client.callZome.mockResolvedValue(mockResult);

      await bridge.queryAuditTrail({
        from_us: 1700000000000000,
        to_us: 1700001000000000,
        domain: 'care',
        event_type: 'match_completed',
      });

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons_care',
        zome_name: 'commons_bridge',
        fn_name: 'query_audit_trail',
        payload: {
          from_us: 1700000000000000,
          to_us: 1700001000000000,
          domain: 'care',
          event_type: 'match_completed',
        },
      });
    });
  });

  describe('health', () => {
    it('should perform health check', async () => {
      const mockHealth: BridgeHealth = {
        healthy: true,
        agent: 'uhCAk...',
        total_events: 42,
        total_queries: 7,
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

// ============================================================================
// CivicBridgeClient
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
  });

  describe('dispatch', () => {
    it('should call civic_bridge.dispatch_call with correct params', async () => {
      const payload = new Uint8Array([10, 20]);
      const mockResult: DispatchResult = { success: true };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.dispatch('justice_cases', 'file_case', payload);

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'dispatch_call',
        payload: { zome: 'justice_cases', fn_name: 'file_case', payload: [10, 20] },
      });
      expect(result.success).toBe(true);
    });
  });

  describe('query', () => {
    it('should submit an audited cross-domain query', async () => {
      const input: CivicQueryInput = {
        domain: 'emergency',
        query_type: 'get_active_incidents',
        params: '{}',
      };

      await bridge.query(input);

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'query_civic',
        payload: expect.objectContaining({
          domain: 'emergency',
          query_type: 'get_active_incidents',
          params: '{}',
        }),
      });
    });
  });

  describe('event broadcasting', () => {
    it('should broadcast a cross-domain event', async () => {
      const input: CivicEventInput = {
        domain: 'media',
        event_type: 'factcheck_completed',
        payload: '{"article_id": "xyz", "verdict": "verified"}',
      };

      await bridge.broadcastEvent(input);

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'broadcast_event',
        payload: expect.objectContaining({
          domain: 'media',
          event_type: 'factcheck_completed',
          related_hashes: [],
        }),
      });
    });

    it('should get domain events', async () => {
      await bridge.getDomainEvents('justice');
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'get_domain_events',
        payload: 'justice',
      });
    });

    it('should get events by type', async () => {
      await bridge.getEventsByType({ domain: 'emergency', event_type: 'shelter_opened' });
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'get_events_by_type',
        payload: { domain: 'emergency', event_type: 'shelter_opened' },
      });
    });

    it('should get all events', async () => {
      await bridge.getAllEvents();
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'get_all_events',
        payload: null,
      });
    });

    it('should get my events', async () => {
      await bridge.getMyEvents();
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'get_my_events',
        payload: null,
      });
    });
  });

  describe('cross-cluster dispatch (civic → commons)', () => {
    it('should dispatch a call to the commons DNA', async () => {
      const payload = new Uint8Array([11, 12]);
      const mockResult: DispatchResult = { success: true };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.dispatchCommonsCall('property_registry', 'get_asset', payload);

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'dispatch_commons_call',
        payload: { role: 'commons', zome: 'property_registry', fn_name: 'get_asset', payload: [11, 12] },
      });
      expect(result.success).toBe(true);
    });

    it('should query property for enforcement', async () => {
      const mockResult: PropertyEnforcementResult = {
        property_found: true,
        enforcement_advisory: 'Property verified — enforcement may proceed',
      };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.queryPropertyForEnforcement({
        property_id: 'prop_456',
        case_id: 'case_789',
      });

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'query_property_for_enforcement',
        payload: { property_id: 'prop_456', case_id: 'case_789' },
      });
      expect(result.property_found).toBe(true);
    });

    it('should check housing capacity for sheltering', async () => {
      const mockResult: HousingCapacityResult = {
        commons_reachable: true,
        recommendation: 'Housing units available — sheltering possible',
      };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.checkHousingCapacityForSheltering({
        disaster_id: 'disaster_001',
        area: 'downtown',
      });

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'check_housing_capacity_for_sheltering',
        payload: { disaster_id: 'disaster_001', area: 'downtown' },
      });
      expect(result.commons_reachable).toBe(true);
    });

    it('should verify care credentials for evidence', async () => {
      const mockResult: CareCredentialVerifyResult = {
        commons_reachable: true,
        recommendation: 'Credentials verified — evidence admissible',
      };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.verifyCareCredentialsForEvidence({
        provider_did: 'did:mycelix:care_provider_123',
        case_id: 'case_456',
      });

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'verify_care_credentials_for_evidence',
        payload: { provider_did: 'did:mycelix:care_provider_123', case_id: 'case_456' },
      });
      expect(result.commons_reachable).toBe(true);
    });
  });

  describe('typed convenience functions (intra-cluster)', () => {
    it('should get active cases for area', async () => {
      const mockResult: JusticeAreaResult = {
        active_cases: 3,
        recommendation: '3 active civil cases in north-side',
      };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.getActiveCasesForArea({
        area: 'north-side',
        case_type: 'civil',
      });

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'get_active_cases_for_area',
        payload: { area: 'north-side', case_type: 'civil' },
      });
      expect(result.active_cases).toBe(3);
    });

    it('should check factcheck status', async () => {
      const mockResult: FactcheckStatusResult = {
        has_factcheck: true,
        verdict: 'verified',
      };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.checkFactcheckStatus({
        claim_id: 'CL-42',
      });

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'check_factcheck_status',
        payload: { claim_id: 'CL-42' },
      });
      expect(result.has_factcheck).toBe(true);
      expect(result.verdict).toBe('verified');
    });
  });

  describe('audit trail', () => {
    it('should query audit trail with time range only', async () => {
      const mockResult: AuditTrailResult = {
        entries: [
          {
            domain: 'justice',
            event_type: 'case_filed',
            source_agent: 'uhCAk...',
            payload_preview: '{"case_id":"C-1"}',
            created_at_us: 1700000500000000,
            action_hash: new Uint8Array([4, 5, 6]),
          },
        ],
        total_matched: 1,
        query_from_us: 1700000000000000,
        query_to_us: 1700001000000000,
      };
      client.callZome.mockResolvedValue(mockResult);

      const result = await bridge.queryAuditTrail({
        from_us: 1700000000000000,
        to_us: 1700001000000000,
      });

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'query_audit_trail',
        payload: {
          from_us: 1700000000000000,
          to_us: 1700001000000000,
          domain: null,
          event_type: null,
        },
      });
      expect(result.entries).toHaveLength(1);
      expect(result.entries[0].event_type).toBe('case_filed');
    });

    it('should query audit trail with domain filter', async () => {
      const mockResult: AuditTrailResult = {
        entries: [],
        total_matched: 0,
        query_from_us: 1700000000000000,
        query_to_us: 1700001000000000,
      };
      client.callZome.mockResolvedValue(mockResult);

      await bridge.queryAuditTrail({
        from_us: 1700000000000000,
        to_us: 1700001000000000,
        domain: 'emergency',
      });

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'civic',
        zome_name: 'civic_bridge',
        fn_name: 'query_audit_trail',
        payload: {
          from_us: 1700000000000000,
          to_us: 1700001000000000,
          domain: 'emergency',
          event_type: null,
        },
      });
    });
  });

  describe('health', () => {
    it('should perform health check', async () => {
      const mockHealth: BridgeHealth = {
        healthy: true,
        agent: 'uhCAk...',
        total_events: 10,
        total_queries: 3,
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

// ============================================================================
// Bridge Event Signal Type Guards
// ============================================================================

describe('Bridge Event Signals', () => {
  describe('isCommonsBridgeSignal', () => {
    it('should return true for valid commons bridge signal', () => {
      const signal: CommonsBridgeEventSignal = {
        signal_type: 'commons_bridge_event',
        domain: 'property',
        event_type: 'ownership_transferred',
        payload: '{"property_id":"P-1"}',
        action_hash: new Uint8Array([1, 2, 3]),
      };
      expect(isCommonsBridgeSignal(signal)).toBe(true);
    });

    it('should return false for civic bridge signal', () => {
      const signal: CivicBridgeEventSignal = {
        signal_type: 'civic_bridge_event',
        domain: 'justice',
        event_type: 'case_filed',
        payload: '{}',
        action_hash: new Uint8Array([4, 5, 6]),
      };
      expect(isCommonsBridgeSignal(signal)).toBe(false);
    });

    it('should return false for null/undefined/primitives', () => {
      expect(isCommonsBridgeSignal(null)).toBe(false);
      expect(isCommonsBridgeSignal(undefined)).toBe(false);
      expect(isCommonsBridgeSignal('string')).toBe(false);
      expect(isCommonsBridgeSignal(42)).toBe(false);
    });

    it('should return false for object without signal_type', () => {
      expect(isCommonsBridgeSignal({ domain: 'property' })).toBe(false);
    });
  });

  describe('isCivicBridgeSignal', () => {
    it('should return true for valid civic bridge signal', () => {
      const signal: CivicBridgeEventSignal = {
        signal_type: 'civic_bridge_event',
        domain: 'emergency',
        event_type: 'incident_reported',
        payload: '{"severity":"high"}',
        action_hash: new Uint8Array([7, 8, 9]),
      };
      expect(isCivicBridgeSignal(signal)).toBe(true);
    });

    it('should return false for commons bridge signal', () => {
      const signal: CommonsBridgeEventSignal = {
        signal_type: 'commons_bridge_event',
        domain: 'water',
        event_type: 'purity_alert',
        payload: '{}',
        action_hash: new Uint8Array([10, 11, 12]),
      };
      expect(isCivicBridgeSignal(signal)).toBe(false);
    });

    it('should return false for null/undefined/primitives', () => {
      expect(isCivicBridgeSignal(null)).toBe(false);
      expect(isCivicBridgeSignal(undefined)).toBe(false);
      expect(isCivicBridgeSignal('string')).toBe(false);
      expect(isCivicBridgeSignal(42)).toBe(false);
    });

    it('should return false for wrong signal_type value', () => {
      expect(isCivicBridgeSignal({ signal_type: 'other_signal' })).toBe(false);
    });
  });
});
