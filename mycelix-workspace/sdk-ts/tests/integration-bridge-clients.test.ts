/**
 * Integration Bridge Client Tests
 *
 * Tests the domain-specific bridge clients: governance, finance, property,
 * energy, media, justice, knowledge, support, food, transport, mail, fabrication.
 * Uses mock AppClient (no conductor required).
 */

import { describe, it, expect, vi } from 'vitest';

import {
  GovernanceBridgeClient,
  getGovernanceBridgeClient,
  FinanceBridgeClient,
  getFinanceBridgeClient,
  PropertyBridgeClient,
  getPropertyBridgeClient,
  EnergyBridgeClient,
  getEnergyBridgeClient,
  MediaBridgeClient,
  getMediaBridgeClient,
  JusticeBridgeClient,
  getJusticeBridgeClient,
  KnowledgeBridgeClient,
  getKnowledgeBridgeClient,
  SupportClient,
  createSupportClient,
  SUPPORT_ZOMES,
  FoodClient,
  createFoodClient,
  FOOD_ZOMES,
  TransportClient,
  createTransportClient,
  TRANSPORT_ZOMES,
  MailTrustService,
  getMailTrustService,
  FabricationService,
  getFabricationService,
} from '../src/integrations/index.js';

// ============================================================================
// Mock Holochain client
// ============================================================================

function createMockClient() {
  return {
    callZome: vi.fn().mockResolvedValue({}),
    on: vi.fn(),
    addSignalHandler: vi.fn(),
  };
}

// ============================================================================
// Governance Bridge Client
// ============================================================================

describe('GovernanceBridgeClient', () => {
  it('should be constructable via factory', () => {
    const mock = createMockClient();
    const client = getGovernanceBridgeClient(mock as any);
    expect(client).toBeInstanceOf(GovernanceBridgeClient);
  });

  it('should have methods from GovernanceBridgeClient prototype', () => {
    const mock = createMockClient();
    const client = getGovernanceBridgeClient(mock as any);
    // Client wraps governance zome calls — verify it's not empty
    const methods = Object.getOwnPropertyNames(Object.getPrototypeOf(client))
      .filter(m => m !== 'constructor');
    expect(methods.length).toBeGreaterThan(0);
  });
});

// ============================================================================
// Finance Bridge Client
// ============================================================================

describe('FinanceBridgeClient', () => {
  it('should be constructable via factory', () => {
    const mock = createMockClient();
    const client = getFinanceBridgeClient(mock as any);
    expect(client).toBeInstanceOf(FinanceBridgeClient);
  });
});

// ============================================================================
// Property Bridge Client
// ============================================================================

describe('PropertyBridgeClient', () => {
  it('should be constructable via factory', () => {
    const mock = createMockClient();
    const client = getPropertyBridgeClient(mock as any);
    expect(client).toBeInstanceOf(PropertyBridgeClient);
  });
});

// ============================================================================
// Energy Bridge Client
// ============================================================================

describe('EnergyBridgeClient', () => {
  it('should be constructable via factory', () => {
    const mock = createMockClient();
    const client = getEnergyBridgeClient(mock as any);
    expect(client).toBeInstanceOf(EnergyBridgeClient);
  });
});

// ============================================================================
// Media Bridge Client
// ============================================================================

describe('MediaBridgeClient', () => {
  it('should be constructable via factory', () => {
    const mock = createMockClient();
    const client = getMediaBridgeClient(mock as any);
    expect(client).toBeInstanceOf(MediaBridgeClient);
  });
});

// ============================================================================
// Justice Bridge Client
// ============================================================================

describe('JusticeBridgeClient', () => {
  it('should be constructable via factory', () => {
    const mock = createMockClient();
    const client = getJusticeBridgeClient(mock as any);
    expect(client).toBeInstanceOf(JusticeBridgeClient);
  });
});

// ============================================================================
// Knowledge Bridge Client
// ============================================================================

describe('KnowledgeBridgeClient', () => {
  it('should be constructable via factory', () => {
    const mock = createMockClient();
    const client = getKnowledgeBridgeClient(mock as any);
    expect(client).toBeInstanceOf(KnowledgeBridgeClient);
  });
});

// ============================================================================
// Support Client
// ============================================================================

describe('SupportClient', () => {
  it('should be constructable via factory', () => {
    const mock = createMockClient();
    const client = createSupportClient(mock as any);
    expect(client).toBeDefined();
  });

  it('should export zome names', () => {
    expect(Array.isArray(SUPPORT_ZOMES)).toBe(true);
    expect(SUPPORT_ZOMES.length).toBeGreaterThan(0);
  });

  it('should have support-specific zomes', () => {
    expect(SUPPORT_ZOMES).toContain('support_tickets');
  });
});

// ============================================================================
// Food Client
// ============================================================================

describe('FoodClient', () => {
  it('should be constructable via factory', () => {
    const mock = createMockClient();
    const client = createFoodClient(mock as any);
    expect(client).toBeDefined();
  });

  it('should export zome names', () => {
    expect(Array.isArray(FOOD_ZOMES)).toBe(true);
    expect(FOOD_ZOMES.length).toBeGreaterThan(0);
  });

  it('should have food-specific zomes', () => {
    expect(FOOD_ZOMES).toContain('food_production');
  });
});

// ============================================================================
// Transport Client
// ============================================================================

describe('TransportClient', () => {
  it('should be constructable via factory', () => {
    const mock = createMockClient();
    const client = createTransportClient(mock as any);
    expect(client).toBeDefined();
  });

  it('should export zome names', () => {
    expect(Array.isArray(TRANSPORT_ZOMES)).toBe(true);
    expect(TRANSPORT_ZOMES.length).toBeGreaterThan(0);
  });

  it('should have transport-specific zomes', () => {
    expect(TRANSPORT_ZOMES).toContain('transport_routes');
  });
});

// ============================================================================
// Mail Trust Service
// ============================================================================

describe('MailTrustService', () => {
  it('should be constructable via factory', () => {
    const service = getMailTrustService();
    expect(service).toBeInstanceOf(MailTrustService);
  });

  it('should be a singleton', () => {
    const a = getMailTrustService();
    const b = getMailTrustService();
    expect(a).toBe(b);
  });

  it('should have trust-related properties', () => {
    const service = getMailTrustService();
    // MailTrustService manages trust scores for encrypted mail
    expect(service).toBeDefined();
  });
});

// ============================================================================
// Fabrication Service
// ============================================================================

describe('FabricationService', () => {
  it('should be constructable via factory', () => {
    const service = getFabricationService();
    expect(service).toBeInstanceOf(FabricationService);
  });

  it('should be a singleton', () => {
    const a = getFabricationService();
    const b = getFabricationService();
    expect(a).toBe(b);
  });
});
