// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Emergency Client Tests
 *
 * Verifies zome call arguments, response mapping, and operations for
 * IncidentsClient, SheltersClient, and CoordinationClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { IncidentsClient } from '../incidents';
import { SheltersClient } from '../shelters';
import { CoordinationClient } from '../coordination';
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

const DISASTER_ENTRY = {
  title: 'Flash Flood Warning',
  description: 'Severe flooding in Cedar Creek area',
  disaster_type: 'Flood',
  severity: 'Warning',
  declared_by_did: 'did:mycelix:alice',
  latitude: 32.948,
  longitude: -96.729,
  radius_km: 15,
  estimated_affected: 2500,
  status: 'Active',
  declared_at: 1708300000,
  updated_at: 1708300000,
  ended_at: null,
};

const SHELTER_ENTRY = {
  disaster_id: new Uint8Array(32),
  name: 'Community Center Shelter',
  address: '456 Main St',
  latitude: 32.95,
  longitude: -96.73,
  capacity: 200,
  current_occupancy: 45,
  manager_did: 'did:mycelix:bob',
  amenities: ['beds', 'food', 'medical'],
  pet_friendly: true,
  accessible: true,
  status: 'Open',
  created_at: 1708300000,
  updated_at: 1708300000,
};

const TEAM_ENTRY = {
  disaster_id: new Uint8Array(32),
  name: 'Search & Rescue Alpha',
  leader_did: 'did:mycelix:alice',
  members: ['did:mycelix:alice', 'did:mycelix:bob'],
  assigned_zone: 'Zone A',
  specialization: 'Water Rescue',
  status: 'Deployed',
  created_at: 1708300000,
  updated_at: 1708300000,
};

const SITREP_ENTRY = {
  disaster_id: new Uint8Array(32),
  team_id: new Uint8Array(32),
  author_did: 'did:mycelix:alice',
  summary: 'Rescued 12 people from flooded area. 3 need medical attention.',
  casualties: 0,
  rescued: 12,
  resources_needed: ['Medical team', 'Additional boats'],
  created_at: 1708400000,
};

const CHECKIN_ENTRY = {
  disaster_id: new Uint8Array(32),
  agent_did: 'did:mycelix:alice',
  team_id: new Uint8Array(32),
  latitude: 32.952,
  longitude: -96.731,
  status: 'Active',
  message: 'Proceeding to sector B',
  checked_in_at: 1708400000,
};

// ============================================================================
// INCIDENTS CLIENT TESTS
// ============================================================================

describe('IncidentsClient', () => {
  let client: IncidentsClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new IncidentsClient(mockAppClient);
  });

  describe('initialization', () => {
    it('should use civic role and emergency_incidents zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DISASTER_ENTRY)
      );

      await client.declareDisaster({
        title: 'Test',
        description: 'Test disaster',
        disasterType: 'Flood',
        severity: 'Advisory',
        latitude: 0,
        longitude: 0,
        radiusKm: 1,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'civic',
          zome_name: 'emergency_incidents',
        })
      );
    });
  });

  describe('declareDisaster', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DISASTER_ENTRY)
      );

      const result = await client.declareDisaster({
        title: 'Flash Flood Warning',
        description: 'Severe flooding in Cedar Creek',
        disasterType: 'Flood',
        severity: 'Warning',
        latitude: 32.948,
        longitude: -96.729,
        radiusKm: 15,
        estimatedAffected: 2500,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'declare_disaster',
          payload: expect.objectContaining({
            disaster_type: 'Flood',
            radius_km: 15,
            estimated_affected: 2500,
          }),
        })
      );
      expect(result.title).toBe('Flash Flood Warning');
      expect(result.disasterType).toBe('Flood');
      expect(result.severity).toBe('Warning');
      expect(result.estimatedAffected).toBe(2500);
      expect(result.status).toBe('Active');
    });
  });

  describe('getDisaster', () => {
    it('should return disaster for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DISASTER_ENTRY)
      );

      const result = await client.getDisaster(new Uint8Array(32));
      expect(result).not.toBeNull();
      expect(result!.title).toBe('Flash Flood Warning');
    });

    it('should return null for missing disaster', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getDisaster(new Uint8Array(32));
      expect(result).toBeNull();
    });
  });

  describe('getActiveDisasters', () => {
    it('should return array of active disasters', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(DISASTER_ENTRY),
        mockRecord({ ...DISASTER_ENTRY, title: 'Tornado Watch' }),
      ]);

      const result = await client.getActiveDisasters();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_active_disasters',
          payload: null,
        })
      );
      expect(result).toHaveLength(2);
    });
  });

  describe('updateDisasterStatus', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...DISASTER_ENTRY, severity: 'Emergency' })
      );

      const result = await client.updateDisasterStatus({
        disasterId: new Uint8Array(32),
        status: 'Active',
        severity: 'Emergency',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'update_disaster_status',
          payload: expect.objectContaining({
            disaster_id: new Uint8Array(32),
          }),
        })
      );
      expect(result.severity).toBe('Emergency');
    });
  });

  describe('endDisaster', () => {
    it('should pass disaster ID and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...DISASTER_ENTRY, status: 'Ended', ended_at: 1709000000 })
      );

      const result = await client.endDisaster(new Uint8Array(32));
      expect(result.status).toBe('Ended');
    });
  });

  describe('getDisasterHistory', () => {
    it('should return array of past disasters', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord({ ...DISASTER_ENTRY, status: 'Ended' }),
      ]);

      const result = await client.getDisasterHistory();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_disaster_history',
          payload: null,
        })
      );
      expect(result).toHaveLength(1);
    });
  });

  describe('error handling', () => {
    it('should propagate zome errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Unauthorized')
      );

      await expect(
        client.declareDisaster({
          title: 'Test', description: 'Test', disasterType: 'Flood',
          severity: 'Advisory', latitude: 0, longitude: 0, radiusKm: 1,
        })
      ).rejects.toThrow();
    });
  });
});

// ============================================================================
// SHELTERS CLIENT TESTS
// ============================================================================

describe('SheltersClient', () => {
  let client: SheltersClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new SheltersClient(mockAppClient);
  });

  describe('initialization', () => {
    it('should use civic role and emergency_shelters zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(SHELTER_ENTRY)
      );

      await client.registerShelter({
        name: 'Test Shelter',
        address: '123 St',
        latitude: 0,
        longitude: 0,
        capacity: 100,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'civic',
          zome_name: 'emergency_shelters',
        })
      );
    });
  });

  describe('registerShelter', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(SHELTER_ENTRY)
      );

      const result = await client.registerShelter({
        disasterId: new Uint8Array(32),
        name: 'Community Center Shelter',
        address: '456 Main St',
        latitude: 32.95,
        longitude: -96.73,
        capacity: 200,
        amenities: ['beds', 'food'],
        petFriendly: true,
        accessible: true,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'register_shelter',
          payload: expect.objectContaining({
            disaster_id: new Uint8Array(32),
            pet_friendly: true,
            accessible: true,
          }),
        })
      );
      expect(result.name).toBe('Community Center Shelter');
      expect(result.capacity).toBe(200);
      expect(result.petFriendly).toBe(true);
      expect(result.status).toBe('Open');
    });
  });

  describe('getShelter', () => {
    it('should return shelter for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(SHELTER_ENTRY)
      );

      const result = await client.getShelter(new Uint8Array(32));
      expect(result).not.toBeNull();
      expect(result!.name).toBe('Community Center Shelter');
    });

    it('should return null for missing shelter', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getShelter(new Uint8Array(32));
      expect(result).toBeNull();
    });
  });

  describe('checkInPerson', () => {
    it('should pass shelter ID and person identifier', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...SHELTER_ENTRY, current_occupancy: 46 })
      );

      const result = await client.checkInPerson(new Uint8Array(32), 'John Doe');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'check_in_person',
          payload: expect.objectContaining({
            shelter_id: new Uint8Array(32),
            person_identifier: 'John Doe',
          }),
        })
      );
      expect(result.currentOccupancy).toBe(46);
    });
  });

  describe('findNearbyShelters', () => {
    it('should pass coordinates and radius', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(SHELTER_ENTRY),
      ]);

      const result = await client.findNearbyShelters(32.95, -96.73, 10);

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'find_nearby_shelters',
          payload: expect.objectContaining({
            latitude: 32.95,
            longitude: -96.73,
            radius_km: 10,
          }),
        })
      );
      expect(result).toHaveLength(1);
    });
  });

  describe('updateShelterStatus', () => {
    it('should pass shelter ID and new status', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...SHELTER_ENTRY, status: 'Full' })
      );

      const result = await client.updateShelterStatus(new Uint8Array(32), 'Full');
      expect(result.status).toBe('Full');
    });
  });

  describe('error handling', () => {
    it('should propagate zome errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Shelter at capacity')
      );

      await expect(
        client.registerShelter({ name: 'T', address: '1', latitude: 0, longitude: 0, capacity: 10 })
      ).rejects.toThrow();
    });
  });
});

// ============================================================================
// COORDINATION CLIENT TESTS
// ============================================================================

describe('CoordinationClient', () => {
  let client: CoordinationClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new CoordinationClient(mockAppClient);
  });

  describe('initialization', () => {
    it('should use civic role and emergency_coordination zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TEAM_ENTRY)
      );

      await client.formTeam({
        disasterId: new Uint8Array(32),
        name: 'Test Team',
        specialization: 'Search',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'civic',
          zome_name: 'emergency_coordination',
        })
      );
    });
  });

  describe('formTeam', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TEAM_ENTRY)
      );

      const result = await client.formTeam({
        disasterId: new Uint8Array(32),
        name: 'Search & Rescue Alpha',
        specialization: 'Water Rescue',
        initialMembers: ['did:mycelix:alice', 'did:mycelix:bob'],
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'form_team',
          payload: expect.objectContaining({
            disaster_id: new Uint8Array(32),
            initial_members: ['did:mycelix:alice', 'did:mycelix:bob'],
          }),
        })
      );
      expect(result.name).toBe('Search & Rescue Alpha');
      expect(result.specialization).toBe('Water Rescue');
      expect(result.status).toBe('Deployed');
    });
  });

  describe('getTeam', () => {
    it('should return team for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TEAM_ENTRY)
      );

      const result = await client.getTeam(new Uint8Array(32));
      expect(result).not.toBeNull();
      expect(result!.name).toBe('Search & Rescue Alpha');
    });

    it('should return null for missing team', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getTeam(new Uint8Array(32));
      expect(result).toBeNull();
    });
  });

  describe('assignToZone', () => {
    it('should pass team ID and zone', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...TEAM_ENTRY, assigned_zone: 'Zone B' })
      );

      const result = await client.assignToZone(new Uint8Array(32), 'Zone B');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'assign_to_zone',
          payload: expect.objectContaining({
            team_id: new Uint8Array(32),
            zone: 'Zone B',
          }),
        })
      );
      expect(result.assignedZone).toBe('Zone B');
    });
  });

  describe('submitSitrep', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(SITREP_ENTRY)
      );

      const result = await client.submitSitrep({
        disasterId: new Uint8Array(32),
        teamId: new Uint8Array(32),
        summary: 'Rescued 12 people',
        casualties: 0,
        rescued: 12,
        resourcesNeeded: ['Medical team'],
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'submit_sitrep',
          payload: expect.objectContaining({
            disaster_id: new Uint8Array(32),
            team_id: new Uint8Array(32),
            resources_needed: ['Medical team'],
          }),
        })
      );
      expect(result.rescued).toBe(12);
      expect(result.casualties).toBe(0);
    });
  });

  describe('checkin', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(CHECKIN_ENTRY)
      );

      const result = await client.checkin({
        disasterId: new Uint8Array(32),
        teamId: new Uint8Array(32),
        latitude: 32.952,
        longitude: -96.731,
        status: 'Active',
        message: 'Proceeding to sector B',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'checkin',
          payload: expect.objectContaining({
            disaster_id: new Uint8Array(32),
            team_id: new Uint8Array(32),
          }),
        })
      );
      expect(result.status).toBe('Active');
      expect(result.message).toBe('Proceeding to sector B');
    });
  });

  describe('error handling', () => {
    it('should propagate zome errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Disaster not found')
      );

      await expect(
        client.formTeam({ disasterId: new Uint8Array(32), name: 'T', specialization: 'S' })
      ).rejects.toThrow();
    });
  });
});
