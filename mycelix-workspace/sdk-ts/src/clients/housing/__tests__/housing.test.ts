// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Housing Client Tests
 *
 * Verifies zome call arguments, response mapping, and operations for
 * UnitsClient, MembershipClient, HousingGovernanceClient, and FinancesClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { UnitsClient } from '../units';
import { MembershipClient } from '../membership';
import { HousingGovernanceClient } from '../governance';
import { FinancesClient } from '../finances';
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

const BUILDING_ENTRY = {
  name: 'Harmony Housing Co-op',
  address: '123 Community Ln',
  latitude: 32.948,
  longitude: -96.729,
  owner_did: 'did:mycelix:alice',
  building_type: 'Cooperative',
  total_units: 24,
  occupied_units: 18,
  year_built: 2020,
  amenities: ['garden', 'workshop', 'kitchen'],
  active: true,
  created_at: 1708200000,
  updated_at: 1708200000,
};

const UNIT_ENTRY = {
  building_id: new Uint8Array(32),
  unit_number: 'A101',
  floor_plan: 'Two bedroom',
  square_meters: 79,
  bedrooms: 2,
  bathrooms: 1,
  monthly_charge: 1200,
  occupant_did: 'did:mycelix:alice',
  status: 'Occupied',
  created_at: 1708200000,
  updated_at: 1708200000,
};

const MEMBER_ENTRY = {
  building_id: new Uint8Array(32),
  member_did: 'did:mycelix:alice',
  member_pub_key: new Uint8Array(32),
  name: 'Alice Johnson',
  unit_id: new Uint8Array(32),
  membership_type: 'Full',
  equity_share: 0.042,
  equity_accumulated: 5000,
  voting_weight: 1.0,
  status: 'Active',
  joined_at: 1708200000,
  updated_at: 1708200000,
};

const RENT_TO_OWN_ENTRY = {
  member_id: new Uint8Array(32),
  unit_id: new Uint8Array(32),
  total_equity_target: 50000,
  monthly_equity_contribution: 500,
  equity_accumulated: 6000,
  term_months: 120,
  start_date: 1708200000,
  projected_completion_date: 1740000000,
  status: 'Active',
  created_at: 1708200000,
};

const MEETING_ENTRY = {
  building_id: new Uint8Array(32),
  title: 'Monthly General Meeting',
  description: 'Regular monthly meeting for all members',
  meeting_type: 'General',
  scheduled_at: 1709000000,
  location: 'Community Room',
  agenda: ['Budget review', 'Garden project'],
  minutes_taker_did: null,
  quorum_required: 0.5,
  status: 'Scheduled',
  created_at: 1708300000,
};

const RESOLUTION_ENTRY = {
  building_id: new Uint8Array(32),
  meeting_id: new Uint8Array(32),
  title: 'Install Solar Panels',
  description: 'Proposal to install solar panels on the roof',
  proposer_did: 'did:mycelix:alice',
  approve_votes: 15,
  reject_votes: 3,
  abstain_votes: 2,
  quorum_required: 0.5,
  threshold: 0.5,
  status: 'Passed',
  voting_ends_at: 1709604800,
  created_at: 1708300000,
};

const ELECTION_ENTRY = {
  building_id: new Uint8Array(32),
  title: 'Board Election 2026',
  positions: ['President', 'Treasurer', 'Secretary'],
  candidates: [],
  status: 'Open',
  voting_starts_at: 1709000000,
  voting_ends_at: 1709604800,
  created_at: 1708300000,
};

const CHARGE_ENTRY = {
  building_id: new Uint8Array(32),
  unit_id: new Uint8Array(32),
  member_did: 'did:mycelix:alice',
  period: '2026-02',
  base_rent: 1000,
  maintenance_fee: 150,
  utilities: 50,
  special_assessment: 0,
  total_due: 1200,
  total_paid: 0,
  status: 'Pending',
  due_date: 1709251200,
  created_at: 1708200000,
};

const PAYMENT_ENTRY = {
  charge_id: new Uint8Array(32),
  member_did: 'did:mycelix:alice',
  amount: 1200,
  method: 'MYC',
  reference: 'tx-123',
  paid_at: 1709000000,
};

// ============================================================================
// UNITS CLIENT TESTS
// ============================================================================

describe('UnitsClient', () => {
  let client: UnitsClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new UnitsClient(mockAppClient);
  });

  describe('initialization', () => {
    it('should create client', () => {
      expect(client).toBeInstanceOf(UnitsClient);
    });

    it('should use commons role and housing_units zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(BUILDING_ENTRY)
      );

      await client.registerBuilding({
        name: 'Test',
        address: '123 St',
        totalUnits: 10,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'commons',
          zome_name: 'housing_units',
        })
      );
    });
  });

  describe('registerBuilding', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(BUILDING_ENTRY)
      );

      await client.registerBuilding({
        name: 'Harmony Housing Co-op',
        address: '123 Community Ln',
        latitude: 32.948,
        longitude: -96.729,
        buildingType: 'Cooperative',
        totalUnits: 24,
        yearBuilt: 2020,
        amenities: ['garden', 'workshop'],
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'register_building',
          payload: expect.objectContaining({
            name: 'Harmony Housing Co-op',
            building_type: 'Cooperative',
            total_units: 24,
            year_built: 2020,
          }),
        })
      );
    });

    it('should map response to Building', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(BUILDING_ENTRY)
      );

      const result = await client.registerBuilding({
        name: 'Test',
        address: '123 St',
        totalUnits: 24,
      });

      expect(result.name).toBe('Harmony Housing Co-op');
      expect(result.totalUnits).toBe(24);
      expect(result.buildingType).toBe('Cooperative');
      expect(result.active).toBe(true);
    });
  });

  describe('getBuilding', () => {
    it('should return building for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(BUILDING_ENTRY)
      );

      const result = await client.getBuilding(new Uint8Array(32));
      expect(result).not.toBeNull();
      expect(result!.name).toBe('Harmony Housing Co-op');
    });

    it('should return null for missing building', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getBuilding(new Uint8Array(32));
      expect(result).toBeNull();
    });
  });

  describe('listBuildings', () => {
    it('should return array of buildings', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(BUILDING_ENTRY),
        mockRecord({ ...BUILDING_ENTRY, name: 'Second Building' }),
      ]);

      const result = await client.listBuildings();
      expect(result).toHaveLength(2);
      expect(result[1].name).toBe('Second Building');
    });
  });

  describe('registerUnit', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(UNIT_ENTRY)
      );

      await client.registerUnit({
        buildingId: new Uint8Array(32),
        unitNumber: 'A101',
        floorPlan: 'Two bedroom',
        squareMeters: 79,
        bedrooms: 2,
        bathrooms: 1,
        monthlyCharge: 1200,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'register_unit',
          payload: expect.objectContaining({
            building_id: new Uint8Array(32),
            unit_number: 'A101',
            square_meters: 79,
            monthly_charge: 1200,
          }),
        })
      );
    });
  });

  describe('getUnit', () => {
    it('should return unit for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(UNIT_ENTRY)
      );

      const result = await client.getUnit(new Uint8Array(32));
      expect(result).not.toBeNull();
      expect(result!.unitNumber).toBe('A101');
      expect(result!.squareMeters).toBe(79);
      expect(result!.status).toBe('Occupied');
    });
  });

  describe('getUnitsForBuilding', () => {
    it('should return array of units', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(UNIT_ENTRY),
        mockRecord({ ...UNIT_ENTRY, unit_number: 'A102' }),
      ]);

      const result = await client.getUnitsForBuilding(new Uint8Array(32));
      expect(result).toHaveLength(2);
    });
  });

  describe('assignOccupant', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(UNIT_ENTRY)
      );

      await client.assignOccupant({
        unitId: new Uint8Array(32),
        occupantDid: 'did:mycelix:alice',
        moveInDate: 1709000000,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'assign_occupant',
          payload: expect.objectContaining({
            unit_id: new Uint8Array(32),
            occupant_did: 'did:mycelix:alice',
            move_in_date: 1709000000,
          }),
        })
      );
    });
  });

  describe('vacateUnit', () => {
    it('should pass unit ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...UNIT_ENTRY, occupant_did: null, status: 'Vacant' })
      );

      const result = await client.vacateUnit(new Uint8Array(32));
      expect(result.status).toBe('Vacant');
    });
  });

  describe('error handling', () => {
    it('should propagate zome errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Building not found')
      );

      await expect(
        client.registerBuilding({ name: 'Test', address: '123', totalUnits: 1 })
      ).rejects.toThrow();
    });
  });
});

// ============================================================================
// MEMBERSHIP CLIENT TESTS
// ============================================================================

describe('MembershipClient', () => {
  let client: MembershipClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new MembershipClient(mockAppClient);
  });

  describe('initialization', () => {
    it('should use commons role and housing_membership zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEMBER_ENTRY)
      );

      await client.submitApplication({
        buildingId: new Uint8Array(32),
        name: 'Alice',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'commons',
          zome_name: 'housing_membership',
        })
      );
    });
  });

  describe('submitApplication', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEMBER_ENTRY)
      );

      const result = await client.submitApplication({
        buildingId: new Uint8Array(32),
        name: 'Alice Johnson',
        membershipType: 'Full',
        statement: 'I want to join the cooperative',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'submit_application',
          payload: expect.objectContaining({
            building_id: new Uint8Array(32),
            name: 'Alice Johnson',
            membership_type: 'Full',
          }),
        })
      );
      expect(result.memberDid).toBe('did:mycelix:alice');
      expect(result.equityShare).toBe(0.042);
      expect(result.status).toBe('Active');
    });
  });

  describe('approveMember', () => {
    it('should pass approve input', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEMBER_ENTRY)
      );

      await client.approveMember({
        applicationId: new Uint8Array(32),
        unitId: new Uint8Array(32),
        equityShare: 0.042,
        votingWeight: 1.0,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'approve_member',
          payload: expect.objectContaining({
            application_id: new Uint8Array(32),
            unit_id: new Uint8Array(32),
            equity_share: 0.042,
            voting_weight: 1.0,
          }),
        })
      );
    });
  });

  describe('getMember', () => {
    it('should return member for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEMBER_ENTRY)
      );

      const result = await client.getMember(new Uint8Array(32));
      expect(result).not.toBeNull();
      expect(result!.name).toBe('Alice Johnson');
    });

    it('should return null for missing member', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getMember(new Uint8Array(32));
      expect(result).toBeNull();
    });
  });

  describe('getMembersForBuilding', () => {
    it('should return array of members', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(MEMBER_ENTRY),
        mockRecord({ ...MEMBER_ENTRY, member_did: 'did:mycelix:bob' }),
      ]);

      const result = await client.getMembersForBuilding(new Uint8Array(32));
      expect(result).toHaveLength(2);
    });
  });

  describe('suspendMember', () => {
    it('should pass member ID and reason', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...MEMBER_ENTRY, status: 'Suspended' })
      );

      const result = await client.suspendMember(new Uint8Array(32), 'Non-payment');
      expect(result.status).toBe('Suspended');
    });
  });

  describe('createRentToOwn', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(RENT_TO_OWN_ENTRY)
      );

      await client.createRentToOwn({
        memberId: new Uint8Array(32),
        unitId: new Uint8Array(32),
        totalEquityTarget: 50000,
        monthlyEquityContribution: 500,
        termMonths: 120,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_rent_to_own',
          payload: expect.objectContaining({
            member_id: new Uint8Array(32),
            total_equity_target: 50000,
            monthly_equity_contribution: 500,
            term_months: 120,
          }),
        })
      );
    });
  });

  describe('recordRentPayment', () => {
    it('should call zome and return void', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(undefined);

      await client.recordRentPayment({
        memberId: new Uint8Array(32),
        amount: 1500,
        period: '2026-02',
        notes: 'Monthly payment',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'record_rent_payment',
          payload: expect.objectContaining({
            member_id: new Uint8Array(32),
            amount: 1500,
            period: '2026-02',
          }),
        })
      );
    });
  });

  describe('error handling', () => {
    it('should propagate zome errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Not authorized')
      );

      await expect(
        client.submitApplication({ buildingId: new Uint8Array(32), name: 'Alice' })
      ).rejects.toThrow();
    });
  });
});

// ============================================================================
// HOUSING GOVERNANCE CLIENT TESTS
// ============================================================================

describe('HousingGovernanceClient', () => {
  let client: HousingGovernanceClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new HousingGovernanceClient(mockAppClient);
  });

  describe('initialization', () => {
    it('should use commons role and housing_governance zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEETING_ENTRY)
      );

      await client.scheduleMeeting({
        buildingId: new Uint8Array(32),
        title: 'Test',
        description: 'Test meeting',
        scheduledAt: 1709000000,
        location: 'Room A',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'commons',
          zome_name: 'housing_governance',
        })
      );
    });
  });

  describe('scheduleMeeting', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEETING_ENTRY)
      );

      const result = await client.scheduleMeeting({
        buildingId: new Uint8Array(32),
        title: 'Monthly General Meeting',
        description: 'Regular monthly meeting',
        meetingType: 'General',
        scheduledAt: 1709000000,
        location: 'Community Room',
        agenda: ['Budget review'],
        quorumRequired: 0.5,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'schedule_meeting',
          payload: expect.objectContaining({
            building_id: new Uint8Array(32),
            meeting_type: 'General',
            scheduled_at: 1709000000,
            quorum_required: 0.5,
          }),
        })
      );
      expect(result.title).toBe('Monthly General Meeting');
      expect(result.status).toBe('Scheduled');
    });
  });

  describe('getMeeting', () => {
    it('should return meeting for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEETING_ENTRY)
      );

      const result = await client.getMeeting(new Uint8Array(32));
      expect(result).not.toBeNull();
      expect(result!.location).toBe('Community Room');
    });

    it('should return null for missing meeting', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getMeeting(new Uint8Array(32));
      expect(result).toBeNull();
    });
  });

  describe('proposeResolution', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(RESOLUTION_ENTRY)
      );

      await client.proposeResolution({
        buildingId: new Uint8Array(32),
        meetingId: new Uint8Array(32),
        title: 'Install Solar Panels',
        description: 'Install solar on roof',
        quorumRequired: 0.5,
        threshold: 0.5,
        votingPeriodHours: 168,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'propose_resolution',
          payload: expect.objectContaining({
            building_id: new Uint8Array(32),
            meeting_id: new Uint8Array(32),
            quorum_required: 0.5,
            threshold: 0.5,
            voting_period_hours: 168,
          }),
        })
      );
    });

    it('should map response to Resolution', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(RESOLUTION_ENTRY)
      );

      const result = await client.proposeResolution({
        buildingId: new Uint8Array(32),
        title: 'Test',
        description: 'Test',
      });

      expect(result.title).toBe('Install Solar Panels');
      expect(result.proposerDid).toBe('did:mycelix:alice');
      expect(result.approveVotes).toBe(15);
      expect(result.status).toBe('Passed');
    });
  });

  describe('voteOnResolution', () => {
    it('should pass resolution ID, choice, and reason', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(undefined);

      await client.voteOnResolution({
        resolutionId: new Uint8Array(32),
        choice: 'Approve',
        reason: 'Good for the environment',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'vote_on_resolution',
          payload: expect.objectContaining({
            resolution_id: new Uint8Array(32),
            choice: 'Approve',
            reason: 'Good for the environment',
          }),
        })
      );
    });
  });

  describe('createElection', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(ELECTION_ENTRY)
      );

      await client.createElection({
        buildingId: new Uint8Array(32),
        title: 'Board Election 2026',
        positions: ['President', 'Treasurer'],
        votingStartsAt: 1709000000,
        votingEndsAt: 1709604800,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_election',
          payload: expect.objectContaining({
            building_id: new Uint8Array(32),
            voting_starts_at: 1709000000,
            voting_ends_at: 1709604800,
          }),
        })
      );
    });
  });

  describe('getElection', () => {
    it('should return election for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(ELECTION_ENTRY)
      );

      const result = await client.getElection(new Uint8Array(32));
      expect(result).not.toBeNull();
      expect(result!.title).toBe('Board Election 2026');
    });
  });

  describe('castBallot', () => {
    it('should pass election ID and selections', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(undefined);

      await client.castBallot({
        electionId: new Uint8Array(32),
        selections: { President: 'did:mycelix:bob' },
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'cast_ballot',
          payload: expect.objectContaining({
            election_id: new Uint8Array(32),
            selections: { President: 'did:mycelix:bob' },
          }),
        })
      );
    });
  });

  describe('error handling', () => {
    it('should propagate zome errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Resolution not found')
      );

      await expect(
        client.proposeResolution({
          buildingId: new Uint8Array(32),
          title: 'Test',
          description: 'Test',
        })
      ).rejects.toThrow();
    });
  });
});

// ============================================================================
// FINANCES CLIENT TESTS
// ============================================================================

describe('FinancesClient', () => {
  let client: FinancesClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new FinancesClient(mockAppClient);
  });

  describe('initialization', () => {
    it('should use commons role and housing_finances zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(CHARGE_ENTRY),
      ]);

      await client.generateMonthlyCharges({
        buildingId: new Uint8Array(32),
        period: '2026-02',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'commons',
          zome_name: 'housing_finances',
        })
      );
    });
  });

  describe('generateMonthlyCharges', () => {
    it('should return array of charges', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(CHARGE_ENTRY),
        mockRecord({ ...CHARGE_ENTRY, member_did: 'did:mycelix:bob' }),
      ]);

      const result = await client.generateMonthlyCharges({
        buildingId: new Uint8Array(32),
        period: '2026-02',
      });

      expect(result).toHaveLength(2);
      expect(result[0].baseRent).toBe(1000);
      expect(result[0].totalDue).toBe(1200);
    });
  });

  describe('getChargesForMember', () => {
    it('should pass member DID and optional period', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(CHARGE_ENTRY),
      ]);

      const result = await client.getChargesForMember('did:mycelix:alice', '2026-02');
      expect(result).toHaveLength(1);
      expect(result[0].totalDue).toBe(1200);
    });
  });

  describe('recordPayment', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(PAYMENT_ENTRY)
      );

      const result = await client.recordPayment({
        chargeId: new Uint8Array(32),
        amount: 1200,
        method: 'MYC',
        reference: 'tx-123',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'record_payment',
          payload: expect.objectContaining({
            charge_id: new Uint8Array(32),
            amount: 1200,
            method: 'MYC',
          }),
        })
      );
      expect(result.amount).toBe(1200);
      expect(result.method).toBe('MYC');
    });
  });

  describe('getFinancialSummary', () => {
    it('should return computed summary directly', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        building_id: new Uint8Array(32),
        period: '2026-02',
        total_revenue: 28800,
        total_expenses: 15000,
        net_income: 13800,
        outstanding_charges: 4800,
        reserve_fund: 50000,
        occupancy_rate: 0.75,
      });

      const result = await client.getFinancialSummary(new Uint8Array(32), '2026-02');
      expect(result.totalRevenue).toBe(28800);
      expect(result.netIncome).toBe(13800);
      expect(result.occupancyRate).toBe(0.75);
    });
  });

  describe('error handling', () => {
    it('should propagate zome errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Building not found')
      );

      await expect(
        client.generateMonthlyCharges({ buildingId: new Uint8Array(32), period: '2026-02' })
      ).rejects.toThrow();
    });
  });
});
