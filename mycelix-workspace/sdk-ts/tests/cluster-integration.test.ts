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
  COMMONS_DOMAINS,
  COMMONS_ZOMES,
  CivicBridgeClient,
  createCivicBridgeClient,
  CIVIC_DOMAINS,
  CIVIC_ZOMES,
} from '../src/integrations/index.js';

import type {
  CommonsQueryInput,
  CommonsEventInput,
  CivicQueryInput,
  CivicEventInput,
  BridgeHealth,
  DispatchResult,
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
  it('should export all 5 commons domains', () => {
    expect(COMMONS_DOMAINS).toEqual(['property', 'housing', 'care', 'mutualaid', 'water']);
    expect(COMMONS_DOMAINS).toHaveLength(5);
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
        role_name: 'commons',
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
        role_name: 'commons',
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
        role_name: 'commons',
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
        role_name: 'commons',
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
        role_name: 'commons',
        zome_name: 'commons_bridge',
        fn_name: 'get_domain_events',
        payload: 'property',
      });
    });

    it('should get events by type', async () => {
      await bridge.getEventsByType({ domain: 'mutualaid', event_type: 'resource_shared' });
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons',
        zome_name: 'commons_bridge',
        fn_name: 'get_events_by_type',
        payload: { domain: 'mutualaid', event_type: 'resource_shared' },
      });
    });

    it('should get all events', async () => {
      await bridge.getAllEvents();
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons',
        zome_name: 'commons_bridge',
        fn_name: 'get_all_events',
        payload: null,
      });
    });

    it('should get my events', async () => {
      await bridge.getMyEvents();
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons',
        zome_name: 'commons_bridge',
        fn_name: 'get_my_events',
        payload: null,
      });
    });
  });

  describe('queries', () => {
    it('should get my queries', async () => {
      await bridge.getMyQueries();
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons',
        zome_name: 'commons_bridge',
        fn_name: 'get_my_queries',
        payload: null,
      });
    });

    it('should get domain queries', async () => {
      await bridge.getDomainQueries('housing');
      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons',
        zome_name: 'commons_bridge',
        fn_name: 'get_domain_queries',
        payload: 'housing',
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
        domains: ['property', 'housing', 'care', 'mutualaid', 'water'],
      };
      client.callZome.mockResolvedValue(mockHealth);

      const result = await bridge.healthCheck();

      expect(client.callZome).toHaveBeenCalledWith({
        role_name: 'commons',
        zome_name: 'commons_bridge',
        fn_name: 'health_check',
        payload: null,
      });
      expect(result.healthy).toBe(true);
      expect(result.domains).toHaveLength(5);
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
