/**
 * Water Client Tests
 *
 * Verifies zome call arguments, response mapping, and operations for
 * FlowClient, PurityClient, and StewardClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { FlowClient } from '../flow';
import { PurityClient } from '../purity';
import { StewardClient } from '../steward';
import type { AppClient } from '@holochain/client';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockClient(): AppClient {
  return {
    callZome: vi.fn(),
  } as unknown as AppClient;
}

function mockRecord(entry: Record<string, unknown>) {
  return {
    entry: { Present: entry },
    signed_action: { hashed: { hash: new Uint8Array(32) } },
  };
}

// ============================================================================
// MOCK ENTRIES (snake_case, matching Rust serde output)
// ============================================================================

const SOURCE_ENTRY = {
  name: 'Community Well',
  description: 'Deep well serving 50 households',
  source_type: 'Well',
  owner_did: 'did:mycelix:alice',
  latitude: 32.948,
  longitude: -96.729,
  watershed_id: new Uint8Array(32),
  estimated_yield_liters_per_day: 5000,
  available_allocation: 3000,
  status: 'Active',
  created_at: 1708200000,
  updated_at: 1708400000,
};

const SHARE_ENTRY = {
  source_id: new Uint8Array(32),
  holder_did: 'did:mycelix:alice',
  daily_allocation_liters: 500,
  priority: 3,
  purpose: 'Household',
  active: true,
  granted_at: 1708200000,
  expires_at: 1740000000,
};

const READING_ENTRY = {
  source_id: new Uint8Array(32),
  tester_did: 'did:mycelix:alice',
  ph: 7.2,
  turbidity: 0.5,
  tds: 250,
  dissolved_oxygen: 8.1,
  e_coli_count: 0,
  temperature: 18.5,
  contaminants: {},
  potable: true,
  notes: 'All parameters within safe range',
  tested_at: 1708400000,
};

const ALERT_ENTRY = {
  source_id: new Uint8Array(32),
  raised_by_did: 'did:mycelix:alice',
  severity: 'Warning',
  description: 'Turbidity reading above threshold',
  contaminant: 'Sediment',
  reading_id: new Uint8Array(32),
  status: 'Active',
  raised_at: 1708400000,
  resolved_at: null,
};

const WATERSHED_ENTRY = {
  name: 'Cedar Creek Watershed',
  description: 'Watershed covering the Cedar Creek drainage basin',
  steward_did: 'did:mycelix:alice',
  boundary_geo_json: '{"type":"Polygon","coordinates":[]}',
  annual_yield_megaliters: 125.5,
  source_count: 5,
  active_rights_count: 12,
  created_at: 1708200000,
  updated_at: 1708200000,
};

const WATER_RIGHT_ENTRY = {
  watershed_id: new Uint8Array(32),
  holder_did: 'did:mycelix:alice',
  annual_allocation_megaliters: 10.5,
  right_type: 'Riparian',
  priority: 1,
  purpose: 'Agriculture',
  conditions: 'Must maintain minimum flow',
  active: true,
  granted_at: 1708200000,
  expires_at: 1740000000,
};

const DISPUTE_ENTRY = {
  watershed_id: new Uint8Array(32),
  filed_by_did: 'did:mycelix:alice',
  against_did: 'did:mycelix:bob',
  right_ids: [new Uint8Array(32)],
  description: 'Respondent exceeding allocated daily limit',
  status: 'Open',
  resolution: null,
  filed_at: 1708400000,
  resolved_at: null,
};

// ============================================================================
// FLOW CLIENT TESTS
// ============================================================================

describe('FlowClient', () => {
  let client: FlowClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new FlowClient(mockAppClient);
  });

  describe('initialization', () => {
    it('should use commons role and water_flow zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(SOURCE_ENTRY)
      );

      await client.registerSource({
        name: 'Test Well',
        sourceType: 'Well',
        latitude: 32.948,
        longitude: -96.729,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'commons',
          zome_name: 'water_flow',
        })
      );
    });
  });

  describe('registerSource', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(SOURCE_ENTRY)
      );

      const result = await client.registerSource({
        name: 'Community Well',
        description: 'Deep well',
        sourceType: 'Well',
        latitude: 32.948,
        longitude: -96.729,
        watershedId: new Uint8Array(32),
        estimatedYieldLitersPerDay: 5000,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'register_source',
          payload: expect.objectContaining({
            source_type: 'Well',
            estimated_yield_liters_per_day: 5000,
          }),
        })
      );
      expect(result.name).toBe('Community Well');
      expect(result.sourceType).toBe('Well');
      expect(result.estimatedYieldLitersPerDay).toBe(5000);
    });
  });

  describe('getSource', () => {
    it('should return source for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(SOURCE_ENTRY)
      );

      const result = await client.getSource(new Uint8Array(32));
      expect(result).not.toBeNull();
      expect(result!.name).toBe('Community Well');
    });

    it('should return null for missing source', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getSource(new Uint8Array(32));
      expect(result).toBeNull();
    });
  });

  describe('getMySources', () => {
    it('should return array of sources', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(SOURCE_ENTRY),
        mockRecord({ ...SOURCE_ENTRY, name: 'Garden Spring' }),
      ]);

      const result = await client.getMySources();
      expect(result).toHaveLength(2);
      expect(result[1].name).toBe('Garden Spring');
    });
  });

  describe('allocateShares', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(SHARE_ENTRY)
      );

      const result = await client.allocateShares({
        sourceId: new Uint8Array(32),
        holderDid: 'did:mycelix:alice',
        dailyAllocationLiters: 500,
        priority: 3,
        purpose: 'Household',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'allocate_shares',
          payload: expect.objectContaining({
            source_id: new Uint8Array(32),
            holder_did: 'did:mycelix:alice',
            daily_allocation_liters: 500,
          }),
        })
      );
      expect(result.dailyAllocationLiters).toBe(500);
      expect(result.purpose).toBe('Household');
    });
  });

  describe('transferCredits', () => {
    it('should call zome and return void', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(undefined);

      await client.transferCredits({
        sourceId: new Uint8Array(32),
        fromDid: 'did:mycelix:alice',
        toDid: 'did:mycelix:bob',
        liters: 100,
        reason: 'Sharing surplus',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'transfer_credits',
          payload: expect.objectContaining({
            from_did: 'did:mycelix:alice',
            to_did: 'did:mycelix:bob',
          }),
        })
      );
    });
  });

  describe('getMyBalance', () => {
    it('should return balance directly', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        did: 'did:mycelix:alice',
        total_daily_allocation: 500,
        used_today: 120,
        remaining_today: 380,
        transferable_credits: 200,
      });

      const result = await client.getMyBalance();
      expect(result.totalDailyAllocation).toBe(500);
      expect(result.usedToday).toBe(120);
      expect(result.remainingToday).toBe(380);
    });
  });

  describe('recordUsage', () => {
    it('should pass parameters correctly', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({
          source_id: new Uint8Array(32),
          user_did: 'did:mycelix:alice',
          liters_used: 50,
          purpose: 'Household',
          recorded_at: 1708400000,
        })
      );

      const result = await client.recordUsage(new Uint8Array(32), 50, 'Household');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'record_usage',
          payload: expect.objectContaining({
            source_id: new Uint8Array(32),
            liters_used: 50,
          }),
        })
      );
      expect(result.litersUsed).toBe(50);
    });
  });

  describe('error handling', () => {
    it('should propagate zome errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Source depleted')
      );

      await expect(
        client.registerSource({ name: 'T', sourceType: 'Well', latitude: 0, longitude: 0 })
      ).rejects.toThrow();
    });
  });
});

// ============================================================================
// PURITY CLIENT TESTS
// ============================================================================

describe('PurityClient', () => {
  let client: PurityClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new PurityClient(mockAppClient);
  });

  describe('initialization', () => {
    it('should use commons role and water_purity zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(READING_ENTRY)
      );

      await client.submitReading({
        sourceId: new Uint8Array(32),
        ph: 7.0,
        turbidity: 0.5,
        tds: 200,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'commons',
          zome_name: 'water_purity',
        })
      );
    });
  });

  describe('submitReading', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(READING_ENTRY)
      );

      const result = await client.submitReading({
        sourceId: new Uint8Array(32),
        ph: 7.2,
        turbidity: 0.5,
        tds: 250,
        dissolvedOxygen: 8.1,
        eColiCount: 0,
        temperature: 18.5,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'submit_reading',
          payload: expect.objectContaining({
            source_id: new Uint8Array(32),
            dissolved_oxygen: 8.1,
            e_coli_count: 0,
          }),
        })
      );
      expect(result.ph).toBe(7.2);
      expect(result.potable).toBe(true);
    });
  });

  describe('getReadingsForSource', () => {
    it('should return array of readings', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(READING_ENTRY),
      ]);

      const result = await client.getReadingsForSource(new Uint8Array(32));
      expect(result).toHaveLength(1);
      expect(result[0].ph).toBe(7.2);
    });
  });

  describe('checkPotability', () => {
    it('should return potability result directly', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        potable: true,
        issues: [],
      });

      const result = await client.checkPotability(new Uint8Array(32));
      expect(result.potable).toBe(true);
      expect(result.issues).toHaveLength(0);
    });
  });

  describe('raiseAlert', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(ALERT_ENTRY)
      );

      const result = await client.raiseAlert({
        sourceId: new Uint8Array(32),
        severity: 'Warning',
        description: 'Turbidity above threshold',
        contaminant: 'Sediment',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'raise_alert',
          payload: expect.objectContaining({
            source_id: new Uint8Array(32),
            severity: 'Warning',
          }),
        })
      );
      expect(result.severity).toBe('Warning');
      expect(result.status).toBe('Active');
    });
  });

  describe('resolveAlert', () => {
    it('should pass alert ID and resolution', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...ALERT_ENTRY, status: 'Resolved', resolved_at: 1709000000 })
      );

      const result = await client.resolveAlert(new Uint8Array(32), 'Sediment filter installed');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'resolve_alert',
          payload: expect.objectContaining({
            alert_id: new Uint8Array(32),
            resolution: 'Sediment filter installed',
          }),
        })
      );
      expect(result.status).toBe('Resolved');
    });
  });

  describe('error handling', () => {
    it('should propagate zome errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Source not found')
      );

      await expect(
        client.submitReading({ sourceId: new Uint8Array(32), ph: 7.0, turbidity: 0.5, tds: 200 })
      ).rejects.toThrow();
    });
  });
});

// ============================================================================
// STEWARD CLIENT TESTS
// ============================================================================

describe('StewardClient', () => {
  let client: StewardClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new StewardClient(mockAppClient);
  });

  describe('initialization', () => {
    it('should use commons role and water_steward zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(WATERSHED_ENTRY)
      );

      await client.defineWatershed({
        name: 'Test',
        description: 'Test watershed',
        boundaryGeoJson: '{}',
        annualYieldMegaliters: 10,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'commons',
          zome_name: 'water_steward',
        })
      );
    });
  });

  describe('defineWatershed', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(WATERSHED_ENTRY)
      );

      const result = await client.defineWatershed({
        name: 'Cedar Creek Watershed',
        description: 'Cedar Creek drainage basin',
        boundaryGeoJson: '{"type":"Polygon"}',
        annualYieldMegaliters: 125.5,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'define_watershed',
          payload: expect.objectContaining({
            boundary_geo_json: '{"type":"Polygon"}',
            annual_yield_megaliters: 125.5,
          }),
        })
      );
      expect(result.name).toBe('Cedar Creek Watershed');
      expect(result.annualYieldMegaliters).toBe(125.5);
    });
  });

  describe('getWatershed', () => {
    it('should return watershed for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(WATERSHED_ENTRY)
      );

      const result = await client.getWatershed(new Uint8Array(32));
      expect(result).not.toBeNull();
      expect(result!.name).toBe('Cedar Creek Watershed');
    });

    it('should return null for missing watershed', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getWatershed(new Uint8Array(32));
      expect(result).toBeNull();
    });
  });

  describe('listWatersheds', () => {
    it('should return array of watersheds', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(WATERSHED_ENTRY),
        mockRecord({ ...WATERSHED_ENTRY, name: 'Oak River Watershed' }),
      ]);

      const result = await client.listWatersheds();
      expect(result).toHaveLength(2);
    });
  });

  describe('registerWaterRight', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(WATER_RIGHT_ENTRY)
      );

      const result = await client.registerWaterRight({
        watershedId: new Uint8Array(32),
        holderDid: 'did:mycelix:alice',
        annualAllocationMegaliters: 10.5,
        rightType: 'Riparian',
        priority: 1,
        purpose: 'Agriculture',
        conditions: 'Must maintain minimum flow',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'register_water_right',
          payload: expect.objectContaining({
            watershed_id: new Uint8Array(32),
            holder_did: 'did:mycelix:alice',
            annual_allocation_megaliters: 10.5,
          }),
        })
      );
      expect(result.rightType).toBe('Riparian');
      expect(result.annualAllocationMegaliters).toBe(10.5);
    });
  });

  describe('getWaterRight', () => {
    it('should return water right for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(WATER_RIGHT_ENTRY)
      );

      const result = await client.getWaterRight(new Uint8Array(32));
      expect(result).not.toBeNull();
      expect(result!.holderDid).toBe('did:mycelix:alice');
    });
  });

  describe('transferRight', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...WATER_RIGHT_ENTRY, holder_did: 'did:mycelix:bob' })
      );

      const result = await client.transferRight({
        rightId: new Uint8Array(32),
        toDid: 'did:mycelix:bob',
        reason: 'Property transfer',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'transfer_right',
          payload: expect.objectContaining({
            right_id: new Uint8Array(32),
            to_did: 'did:mycelix:bob',
          }),
        })
      );
      expect(result.holderDid).toBe('did:mycelix:bob');
    });
  });

  describe('fileDispute', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DISPUTE_ENTRY)
      );

      const result = await client.fileDispute({
        watershedId: new Uint8Array(32),
        againstDid: 'did:mycelix:bob',
        rightIds: [new Uint8Array(32)],
        description: 'Exceeding daily limit',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'file_dispute',
          payload: expect.objectContaining({
            watershed_id: new Uint8Array(32),
            against_did: 'did:mycelix:bob',
          }),
        })
      );
      expect(result.status).toBe('Open');
      expect(result.againstDid).toBe('did:mycelix:bob');
    });
  });

  describe('getDispute', () => {
    it('should return dispute for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DISPUTE_ENTRY)
      );

      const result = await client.getDispute(new Uint8Array(32));
      expect(result).not.toBeNull();
      expect(result!.description).toBe('Respondent exceeding allocated daily limit');
    });
  });

  describe('getDisputesForWatershed', () => {
    it('should return array of disputes', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(DISPUTE_ENTRY),
      ]);

      const result = await client.getDisputesForWatershed(new Uint8Array(32));
      expect(result).toHaveLength(1);
    });
  });

  describe('error handling', () => {
    it('should propagate zome errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Right expired')
      );

      await expect(
        client.transferRight({ rightId: new Uint8Array(32), toDid: 'did:mycelix:bob', reason: 'X' })
      ).rejects.toThrow();
    });
  });
});
