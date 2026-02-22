/**
 * Cluster Bridge Conductor Integration Tests
 *
 * Tests CommonsBridgeClient and CivicBridgeClient against mock clients
 * that simulate the Rust bridge coordinator behavior, catching
 * serialization mismatches between TS types and Rust structs.
 *
 * The mock client validates:
 * 1. Correct role_name/zome_name/fn_name parameters
 * 2. Payload shape matches Rust struct expectations
 * 3. Rate limit error responses are handled properly
 * 4. Typed convenience function responses decode correctly
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  CommonsBridgeClient,
  createCommonsBridgeClient,
} from '../../src/integrations/commons/index.js';

import {
  CivicBridgeClient,
  createCivicBridgeClient,
} from '../../src/integrations/civic/index.js';

import type {
  PropertyOwnershipQuery,
  PropertyOwnershipResult,
  CareAvailabilityQuery,
  CareAvailabilityResult,
} from '../../src/integrations/commons/index.js';

import type {
  JusticeAreaQuery,
  JusticeAreaResult,
  FactcheckStatusQuery,
  FactcheckStatusResult,
} from '../../src/integrations/civic/index.js';

// ============================================================================
// Mock client that simulates Rust bridge behavior
// ============================================================================

/** Event tracking for audit verification */
interface AuditLog {
  zome: string;
  fn_name: string;
  payload: unknown;
  timestamp: number;
}

function createBridgeMockClient(role: 'commons' | 'civic') {
  const auditLog: AuditLog[] = [];
  let dispatchCount = 0;
  const RATE_LIMIT_MAX = 100;

  return {
    auditLog,
    get dispatchCount() { return dispatchCount; },
    resetDispatchCount() { dispatchCount = 0; },

    callZome: vi.fn(async <T>({ role_name, zome_name, fn_name, payload }: {
      role_name: string;
      zome_name: string;
      fn_name: string;
      payload: unknown;
    }): Promise<T> => {
      // Validate role name matches the expected cluster
      const validRoles = role === 'commons'
        ? ['commons_land', 'commons_care']
        : ['civic'];
      const expectedBridge = role === 'commons' ? 'commons_bridge' : 'civic_bridge';

      if (!validRoles.includes(role_name)) {
        throw new Error(`Wrong role_name: expected one of ${JSON.stringify(validRoles)}, got '${role_name}'`);
      }
      if (zome_name !== expectedBridge) {
        throw new Error(`Wrong zome_name: expected '${expectedBridge}', got '${zome_name}'`);
      }

      auditLog.push({ zome: zome_name, fn_name, payload, timestamp: Date.now() });

      // Simulate dispatch with rate limiting
      if (fn_name === 'dispatch_call' || fn_name === 'dispatch_civic_call' || fn_name === 'dispatch_commons_call') {
        dispatchCount++;
        if (dispatchCount > RATE_LIMIT_MAX) {
          return {
            success: false,
            response: null,
            error: `Rate limit exceeded: ${dispatchCount} dispatches in 60s (max ${RATE_LIMIT_MAX})`,
          } as T;
        }

        const p = payload as { zome: string; fn_name: string; payload: number[] };
        const allowedZomes = role === 'commons'
          ? ['property_registry', 'housing_clt', 'care_matching', 'mutualaid_needs', 'water_flow',
             'property_transfer', 'property_disputes', 'property_commons',
             'housing_units', 'housing_membership', 'housing_finances', 'housing_maintenance', 'housing_governance',
             'care_timebank', 'care_circles', 'care_plans', 'care_credentials',
             'mutualaid_circles', 'mutualaid_governance', 'mutualaid_pools', 'mutualaid_requests', 'mutualaid_resources', 'mutualaid_timebank',
             'water_purity', 'water_capture', 'water_steward', 'water_wisdom']
          : ['justice_cases', 'justice_evidence', 'justice_arbitration', 'justice_restorative', 'justice_enforcement',
             'emergency_incidents', 'emergency_triage', 'emergency_resources', 'emergency_coordination', 'emergency_shelters', 'emergency_comms',
             'media_publication', 'media_attribution', 'media_factcheck', 'media_curation'];

        if (!allowedZomes.includes(p.zome)) {
          return {
            success: false,
            response: null,
            error: `Zome '${p.zome}' is not in the allowed dispatch list. Valid zomes: ${JSON.stringify(allowedZomes)}`,
          } as T;
        }

        return { success: true, response: [0, 0, 0], error: null } as T;
      }

      // Health check
      if (fn_name === 'health_check') {
        const domains = role === 'commons'
          ? ['property', 'housing', 'care', 'mutualaid', 'water']
          : ['justice', 'emergency', 'media'];
        return {
          healthy: true,
          agent: 'uhCAk_test_agent',
          total_events: auditLog.filter(a => a.fn_name === 'broadcast_event').length,
          total_queries: auditLog.filter(a => a.fn_name.startsWith('query_')).length,
          domains,
        } as T;
      }

      // Typed convenience functions
      if (fn_name === 'verify_property_ownership') {
        const p = payload as PropertyOwnershipQuery;
        return {
          is_owner: p.property_id === 'PROP-OWNED',
          owner_did: p.property_id === 'PROP-OWNED' ? p.requester_did : null,
          error: null,
        } as T;
      }

      if (fn_name === 'check_care_availability') {
        const p = payload as CareAvailabilityQuery;
        return {
          available_count: p.skill_needed === 'nursing' ? 3 : 0,
          recommendation: p.skill_needed === 'nursing'
            ? '3 nursing providers available'
            : 'No providers available for this skill',
          error: null,
        } as T;
      }

      if (fn_name === 'get_active_cases_for_area') {
        const p = payload as JusticeAreaQuery;
        return {
          active_cases: p.area === 'downtown' ? 2 : 0,
          recommendation: p.area === 'downtown'
            ? '2 active cases in downtown area'
            : 'No active cases in this area',
          error: null,
        } as T;
      }

      if (fn_name === 'check_factcheck_status') {
        const p = payload as FactcheckStatusQuery;
        return {
          has_factcheck: p.claim_id === 'CL-VERIFIED',
          verdict: p.claim_id === 'CL-VERIFIED' ? 'verified' : null,
          error: null,
        } as T;
      }

      // Cross-cluster typed helpers (commons → civic)
      if (fn_name === 'check_emergency_for_area') {
        const p = payload as { lat: number; lon: number };
        return {
          has_active_emergencies: false,
          active_count: 0,
          recommendation: `No active emergencies near (${p.lat}, ${p.lon})`,
          error: null,
        } as T;
      }

      if (fn_name === 'check_justice_disputes_for_property') {
        const p = payload as { resource_id: string };
        return {
          has_pending_cases: false,
          recommendation: `No disputes for ${p.resource_id}`,
          error: null,
        } as T;
      }

      // Cross-cluster typed helpers (civic → commons)
      if (fn_name === 'query_property_for_enforcement') {
        const p = payload as { property_id: string; case_id: string };
        return {
          property_found: true,
          enforcement_advisory: `Property ${p.property_id} found for case ${p.case_id}`,
          error: null,
        } as T;
      }

      if (fn_name === 'check_housing_capacity_for_sheltering') {
        return {
          commons_reachable: true,
          recommendation: 'Housing capacity available',
          error: null,
        } as T;
      }

      if (fn_name === 'verify_care_credentials_for_evidence') {
        return {
          commons_reachable: true,
          recommendation: 'Credentials verified',
          error: null,
        } as T;
      }

      // Query/event operations — return empty arrays or simple acks
      if (fn_name.startsWith('query_') || fn_name === 'resolve_query') {
        return {} as T;
      }

      if (fn_name === 'broadcast_event') {
        return {} as T;
      }

      if (fn_name.startsWith('get_domain_') || fn_name.startsWith('get_my_') ||
          fn_name === 'get_all_events' || fn_name.startsWith('get_events_')) {
        return [] as T;
      }

      throw new Error(`Unknown function: ${fn_name}`);
    }),
  };
}

// ============================================================================
// Commons Bridge — Typed Convenience Functions
// ============================================================================

describe('Commons Bridge — Typed Convenience (mock conductor)', () => {
  let mockClient: ReturnType<typeof createBridgeMockClient>;
  let bridge: CommonsBridgeClient;

  beforeEach(() => {
    mockClient = createBridgeMockClient('commons');
    bridge = createCommonsBridgeClient(mockClient);
  });

  it('should verify property ownership — owner found', async () => {
    const result: PropertyOwnershipResult = await bridge.verifyPropertyOwnership({
      property_id: 'PROP-OWNED',
      requester_did: 'did:mycelix:alice',
    });

    expect(result.is_owner).toBe(true);
    expect(result.owner_did).toBe('did:mycelix:alice');
    expect(result.error).toBeNull();
  });

  it('should verify property ownership — not owner', async () => {
    const result: PropertyOwnershipResult = await bridge.verifyPropertyOwnership({
      property_id: 'PROP-OTHER',
      requester_did: 'did:mycelix:bob',
    });

    expect(result.is_owner).toBe(false);
    expect(result.owner_did).toBeNull();
  });

  it('should check care availability — matches found', async () => {
    const result: CareAvailabilityResult = await bridge.checkCareAvailability({
      skill_needed: 'nursing',
      location: 'downtown',
    });

    expect(result.available_count).toBe(3);
    expect(result.recommendation).toContain('nursing');
  });

  it('should check care availability — no matches', async () => {
    const result: CareAvailabilityResult = await bridge.checkCareAvailability({
      skill_needed: 'quantum_healing',
    });

    expect(result.available_count).toBe(0);
    expect(result.recommendation).toContain('No providers');
  });

  it('should check emergency for area (cross-cluster)', async () => {
    const result = await bridge.checkEmergencyForArea({ lat: 32.9, lon: -96.7 });
    expect(result.has_active_emergencies).toBe(false);
    expect(result.active_count).toBe(0);
  });

  it('should check justice disputes for property (cross-cluster)', async () => {
    const result = await bridge.checkJusticeDisputesForProperty({ resource_id: 'prop_123' });
    expect(result.has_pending_cases).toBe(false);
  });
});

// ============================================================================
// Civic Bridge — Typed Convenience Functions
// ============================================================================

describe('Civic Bridge — Typed Convenience (mock conductor)', () => {
  let mockClient: ReturnType<typeof createBridgeMockClient>;
  let bridge: CivicBridgeClient;

  beforeEach(() => {
    mockClient = createBridgeMockClient('civic');
    bridge = createCivicBridgeClient(mockClient);
  });

  it('should get active cases for area — matches found', async () => {
    const result: JusticeAreaResult = await bridge.getActiveCasesForArea({
      area: 'downtown',
      case_type: 'civil',
    });

    expect(result.active_cases).toBe(2);
    expect(result.recommendation).toContain('downtown');
  });

  it('should get active cases for area — no matches', async () => {
    const result: JusticeAreaResult = await bridge.getActiveCasesForArea({
      area: 'rural-outskirts',
    });

    expect(result.active_cases).toBe(0);
  });

  it('should check factcheck status — verified', async () => {
    const result: FactcheckStatusResult = await bridge.checkFactcheckStatus({
      claim_id: 'CL-VERIFIED',
    });

    expect(result.has_factcheck).toBe(true);
    expect(result.verdict).toBe('verified');
  });

  it('should check factcheck status — not found', async () => {
    const result: FactcheckStatusResult = await bridge.checkFactcheckStatus({
      claim_id: 'CL-UNKNOWN',
    });

    expect(result.has_factcheck).toBe(false);
    expect(result.verdict).toBeNull();
  });

  it('should query property for enforcement (cross-cluster)', async () => {
    const result = await bridge.queryPropertyForEnforcement({
      property_id: 'prop_456',
      case_id: 'case_789',
    });

    expect(result.property_found).toBe(true);
    expect(result.enforcement_advisory).toContain('prop_456');
  });

  it('should check housing capacity for sheltering (cross-cluster)', async () => {
    const result = await bridge.checkHousingCapacityForSheltering({
      disaster_id: 'dis_001',
      area: 'city-center',
    });

    expect(result.commons_reachable).toBe(true);
  });

  it('should verify care credentials for evidence (cross-cluster)', async () => {
    const result = await bridge.verifyCareCredentialsForEvidence({
      provider_did: 'did:mycelix:provider_abc',
      case_id: 'case_123',
    });

    expect(result.commons_reachable).toBe(true);
  });
});

// ============================================================================
// Rate Limiting Behavior
// ============================================================================

describe('Bridge Rate Limiting (mock conductor)', () => {
  it('should handle rate limit errors from commons bridge', async () => {
    const mockClient = createBridgeMockClient('commons');
    const bridge = createCommonsBridgeClient(mockClient);

    // Exhaust the rate limit
    for (let i = 0; i < 101; i++) {
      await bridge.dispatch('property_registry', 'get_property', new Uint8Array([1]));
    }

    // The 102nd call should return a rate limit error
    const result = await bridge.dispatch('property_registry', 'get_property', new Uint8Array([1]));
    expect(result.success).toBe(false);
    expect(result.error).toContain('Rate limit exceeded');
  });

  it('should handle rate limit errors from civic bridge', async () => {
    const mockClient = createBridgeMockClient('civic');
    const bridge = createCivicBridgeClient(mockClient);

    // Exhaust the rate limit
    for (let i = 0; i < 101; i++) {
      await bridge.dispatch('justice_cases', 'get_case', new Uint8Array([1]));
    }

    const result = await bridge.dispatch('justice_cases', 'get_case', new Uint8Array([1]));
    expect(result.success).toBe(false);
    expect(result.error).toContain('Rate limit exceeded');
  });

  it('should count dispatches correctly across multiple zomes', async () => {
    const mockClient = createBridgeMockClient('commons');
    const bridge = createCommonsBridgeClient(mockClient);

    // Mix dispatch targets
    for (let i = 0; i < 50; i++) {
      await bridge.dispatch('property_registry', 'fn', new Uint8Array([]));
      await bridge.dispatch('housing_clt', 'fn', new Uint8Array([]));
    }

    // 100 calls total — at the limit
    expect(mockClient.dispatchCount).toBe(100);

    // 101st should be rate limited
    const result = await bridge.dispatch('care_matching', 'fn', new Uint8Array([]));
    expect(result.success).toBe(false);
    expect(result.error).toContain('Rate limit exceeded');
  });
});

// ============================================================================
// Dispatch Allowlist Enforcement
// ============================================================================

describe('Bridge Allowlist Enforcement (mock conductor)', () => {
  it('should reject dispatch to disallowed commons zome', async () => {
    const mockClient = createBridgeMockClient('commons');
    const bridge = createCommonsBridgeClient(mockClient);

    const result = await bridge.dispatch('evil_zome', 'steal_data', new Uint8Array([]));
    expect(result.success).toBe(false);
    expect(result.error).toContain('not in the allowed dispatch list');
  });

  it('should reject dispatch to disallowed civic zome', async () => {
    const mockClient = createBridgeMockClient('civic');
    const bridge = createCivicBridgeClient(mockClient);

    const result = await bridge.dispatch('evil_zome', 'steal_data', new Uint8Array([]));
    expect(result.success).toBe(false);
    expect(result.error).toContain('not in the allowed dispatch list');
  });

  it('should accept dispatch to allowed commons zome', async () => {
    const mockClient = createBridgeMockClient('commons');
    const bridge = createCommonsBridgeClient(mockClient);

    const result = await bridge.dispatch('water_flow', 'get_readings', new Uint8Array([1, 2]));
    expect(result.success).toBe(true);
  });

  it('should accept dispatch to allowed civic zome', async () => {
    const mockClient = createBridgeMockClient('civic');
    const bridge = createCivicBridgeClient(mockClient);

    const result = await bridge.dispatch('media_factcheck', 'check_claim', new Uint8Array([3, 4]));
    expect(result.success).toBe(true);
  });
});

// ============================================================================
// Audit Trail
// ============================================================================

describe('Bridge Audit Trail (mock conductor)', () => {
  it('should track all operations in audit log', async () => {
    const mockClient = createBridgeMockClient('commons');
    const bridge = createCommonsBridgeClient(mockClient);

    await bridge.healthCheck();
    await bridge.dispatch('property_registry', 'get_property', new Uint8Array([1]));
    await bridge.broadcastEvent({
      domain: 'housing',
      event_type: 'unit_created',
      payload: '{}',
    });

    expect(mockClient.auditLog).toHaveLength(3);
    expect(mockClient.auditLog[0].fn_name).toBe('health_check');
    expect(mockClient.auditLog[1].fn_name).toBe('dispatch_call');
    expect(mockClient.auditLog[2].fn_name).toBe('broadcast_event');
  });

  it('should track health check event/query counts', async () => {
    const mockClient = createBridgeMockClient('commons');
    const bridge = createCommonsBridgeClient(mockClient);

    await bridge.broadcastEvent({ domain: 'care', event_type: 'match', payload: '{}' });
    await bridge.broadcastEvent({ domain: 'water', event_type: 'alert', payload: '{}' });
    await bridge.query({ domain: 'property', query_type: 'check', params: '{}' });

    const health = await bridge.healthCheck();
    expect(health.total_events).toBe(2);
    expect(health.total_queries).toBe(1);
  });
});
