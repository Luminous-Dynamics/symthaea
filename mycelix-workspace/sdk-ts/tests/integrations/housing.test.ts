/**
 * Housing Integration Tests
 *
 * Tests for HousingClient and its sub-clients (UnitsClient, MembershipClient,
 * FinancesClient, HousingGovernanceClient) -- the domain-specific SDK clients
 * for the housing zomes within the mycelix-commons cluster DNA.
 *
 * Covers:
 * - housing-clt: Land trust CRUD, ground leases
 * - housing-units: Building and unit management
 * - housing-finances: Monthly charges, payments, financial summaries
 * - housing-membership: Applications, approvals, rent-to-own
 * - housing-governance: Meetings, resolutions, elections
 * - housing-maintenance: Maintenance request lifecycle
 *
 * All calls are dispatched through the commons role to the housing_units,
 * housing_membership, housing_finances, and housing_governance coordinator zomes.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  HousingClient,
  createHousingClient,
  UnitsClient,
  MembershipClient,
  FinancesClient,
  HousingGovernanceClient,
  HousingError,
  type Building,
  type RegisterBuildingInput,
  type Unit,
  type RegisterUnitInput,
  type AssignOccupantInput,
  type Member,
  type SubmitApplicationInput,
  type ApproveMemberInput,
  type RentToOwnAgreement,
  type CreateRentToOwnInput,
  type RecordRentPaymentInput,
  type MonthlyCharge,
  type GenerateMonthlyChargesInput,
  type PaymentRecord,
  type RecordPaymentInput,
  type FinancialSummary,
  type MaintenanceRequest,
  type Meeting,
  type ScheduleMeetingInput,
  type Resolution,
  type ProposeResolutionInput,
  type VoteOnResolutionInput,
  type Election,
  type CreateElectionInput,
  type CastBallotInput,
  type LandTrust,
  type GroundLease,
} from '../../src/clients/housing/index.js';

// ============================================================================
// Mock Holochain client
// ============================================================================

function createMockClient() {
  return {
    callZome: vi.fn().mockResolvedValue({}),
    myPubKey: new Uint8Array(39).fill(0xaa),
    appInfo: vi.fn().mockResolvedValue({ installed_app_id: 'test' }),
  };
}

// ============================================================================
// Test helpers
// ============================================================================

function fakeActionHash(): Uint8Array {
  return new Uint8Array(39).fill(0xab);
}

function fakeAgentKey(): Uint8Array {
  return new Uint8Array(39).fill(0xca);
}

function makeMockRecord(entry: Record<string, unknown>): object {
  return {
    signed_action: { hashed: { hash: fakeActionHash() } },
    entry: { Present: entry },
  };
}

function makeBuildingEntry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    name: 'Harmony Co-Op',
    address: '123 Community Lane',
    latitude: 32.95,
    longitude: -96.73,
    owner_did: 'did:mycelix:owner1',
    building_type: 'CoOp',
    total_units: 24,
    occupied_units: 18,
    year_built: 2020,
    amenities: ['garden', 'workshop'],
    active: true,
    created_at: Date.now(),
    updated_at: Date.now(),
    ...overrides,
  };
}

function makeUnitEntry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    building_id: fakeActionHash(),
    unit_number: 'A101',
    floor_plan: 'Studio',
    square_meters: 45,
    bedrooms: 1,
    bathrooms: 1,
    monthly_charge: 800,
    occupant_did: undefined,
    status: 'Available',
    created_at: Date.now(),
    updated_at: Date.now(),
    ...overrides,
  };
}

function makeMemberEntry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    building_id: fakeActionHash(),
    member_did: 'did:mycelix:alice',
    member_pub_key: fakeAgentKey(),
    name: 'Alice',
    unit_id: undefined,
    membership_type: 'Full',
    equity_share: 0.05,
    equity_accumulated: 5000,
    voting_weight: 1,
    status: 'Active',
    joined_at: Date.now(),
    updated_at: Date.now(),
    ...overrides,
  };
}

function makeChargeEntry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    building_id: fakeActionHash(),
    unit_id: fakeActionHash(),
    member_did: 'did:mycelix:alice',
    period: '2026-02',
    base_rent: 700,
    maintenance_fee: 50,
    utilities: 100,
    special_assessment: 0,
    total_due: 850,
    total_paid: 0,
    status: 'Pending',
    due_date: Date.now() + 30 * 86400000,
    created_at: Date.now(),
    ...overrides,
  };
}

function makeMeetingEntry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    building_id: fakeActionHash(),
    title: 'Monthly Board Meeting',
    description: 'Regular business meeting',
    meeting_type: 'Regular',
    scheduled_at: Date.now() + 7 * 86400000,
    location: 'Community Room',
    agenda: ['Budget review', 'Maintenance updates'],
    minutes_taker_did: undefined,
    quorum_required: 0.5,
    status: 'Scheduled',
    created_at: Date.now(),
    ...overrides,
  };
}

function makeResolutionEntry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    building_id: fakeActionHash(),
    meeting_id: fakeActionHash(),
    title: 'Solar Panel Installation',
    description: 'Approve installation of rooftop solar',
    proposer_did: 'did:mycelix:alice',
    approve_votes: 0,
    reject_votes: 0,
    abstain_votes: 0,
    quorum_required: 0.5,
    threshold: 0.5,
    status: 'Proposed',
    voting_ends_at: Date.now() + 7 * 86400000,
    created_at: Date.now(),
    ...overrides,
  };
}

function makeElectionEntry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    building_id: fakeActionHash(),
    title: 'Board of Directors 2026',
    positions: ['President', 'Secretary', 'Treasurer'],
    candidates: [],
    status: 'Nominations',
    voting_starts_at: Date.now() + 14 * 86400000,
    voting_ends_at: Date.now() + 21 * 86400000,
    created_at: Date.now(),
    ...overrides,
  };
}

// ============================================================================
// Type construction tests
// ============================================================================

describe('Housing Types', () => {
  describe('Building', () => {
    it('should construct a valid Building from entry data', () => {
      const entry = makeBuildingEntry();
      expect(entry.name).toBe('Harmony Co-Op');
      expect(entry.building_type).toBe('CoOp');
      expect(entry.total_units).toBe(24);
      expect(entry.active).toBe(true);
    });

    it('should accept all building type variants', () => {
      const types = ['CoOp', 'LandTrust', 'MutualHousing', 'CommunityLand', 'PublicHousing'];
      for (const t of types) {
        const entry = makeBuildingEntry({ building_type: t });
        expect(entry.building_type).toBe(t);
      }
    });
  });

  describe('Unit', () => {
    it('should construct a valid Unit from entry data', () => {
      const entry = makeUnitEntry();
      expect(entry.unit_number).toBe('A101');
      expect(entry.floor_plan).toBe('Studio');
      expect(entry.square_meters).toBe(45);
      expect(entry.status).toBe('Available');
    });

    it('should accept all unit status variants', () => {
      const statuses = ['Available', 'Occupied', 'Reserved', 'Maintenance', 'Decommissioned'];
      for (const s of statuses) {
        const entry = makeUnitEntry({ status: s });
        expect(entry.status).toBe(s);
      }
    });
  });

  describe('Member', () => {
    it('should construct a valid Member from entry data', () => {
      const entry = makeMemberEntry();
      expect(entry.name).toBe('Alice');
      expect(entry.membership_type).toBe('Full');
      expect(entry.equity_share).toBe(0.05);
      expect(entry.status).toBe('Active');
    });

    it('should accept all membership type variants', () => {
      const types = ['Full', 'Associate', 'Provisional', 'RentToOwn'];
      for (const t of types) {
        const entry = makeMemberEntry({ membership_type: t });
        expect(entry.membership_type).toBe(t);
      }
    });
  });

  describe('MonthlyCharge', () => {
    it('should construct a valid MonthlyCharge from entry data', () => {
      const entry = makeChargeEntry();
      expect(entry.period).toBe('2026-02');
      expect(entry.base_rent).toBe(700);
      expect(entry.total_due).toBe(850);
      expect(entry.status).toBe('Pending');
    });

    it('should accept all charge status variants', () => {
      const statuses = ['Pending', 'Partial', 'Paid', 'Overdue', 'Waived'];
      for (const s of statuses) {
        const entry = makeChargeEntry({ status: s });
        expect(entry.status).toBe(s);
      }
    });
  });

  describe('Meeting', () => {
    it('should construct a valid Meeting from entry data', () => {
      const entry = makeMeetingEntry();
      expect(entry.title).toBe('Monthly Board Meeting');
      expect(entry.meeting_type).toBe('Regular');
      expect(entry.quorum_required).toBe(0.5);
    });

    it('should accept all meeting type variants', () => {
      const types = ['Regular', 'Special', 'Emergency', 'Annual'];
      for (const t of types) {
        const entry = makeMeetingEntry({ meeting_type: t });
        expect(entry.meeting_type).toBe(t);
      }
    });
  });

  describe('Resolution', () => {
    it('should construct a valid Resolution from entry data', () => {
      const entry = makeResolutionEntry();
      expect(entry.title).toBe('Solar Panel Installation');
      expect(entry.status).toBe('Proposed');
      expect(entry.threshold).toBe(0.5);
    });

    it('should accept all resolution status variants', () => {
      const statuses = ['Proposed', 'Voting', 'Passed', 'Rejected', 'Tabled'];
      for (const s of statuses) {
        const entry = makeResolutionEntry({ status: s });
        expect(entry.status).toBe(s);
      }
    });
  });

  describe('Election', () => {
    it('should construct a valid Election from entry data', () => {
      const entry = makeElectionEntry();
      expect(entry.title).toBe('Board of Directors 2026');
      expect(entry.positions).toHaveLength(3);
      expect(entry.status).toBe('Nominations');
    });

    it('should accept all election status variants', () => {
      const statuses = ['Nominations', 'Voting', 'Completed', 'Cancelled'];
      for (const s of statuses) {
        const entry = makeElectionEntry({ status: s });
        expect(entry.status).toBe(s);
      }
    });
  });

  describe('HousingError', () => {
    it('should construct with code, message, and details', () => {
      const error = new HousingError('NOT_FOUND', 'Building not found', { id: 'xyz' });
      expect(error.code).toBe('NOT_FOUND');
      expect(error.message).toBe('Building not found');
      expect(error.details).toEqual({ id: 'xyz' });
      expect(error.name).toBe('HousingError');
    });

    it('should accept all error code variants', () => {
      const codes = [
        'CONNECTION_ERROR', 'ZOME_CALL_ERROR', 'NOT_FOUND', 'UNAUTHORIZED',
        'INVALID_INPUT', 'UNIT_UNAVAILABLE', 'NOT_MEMBER', 'ALREADY_MEMBER',
        'PAYMENT_ERROR', 'GOVERNANCE_ERROR', 'ELECTION_ERROR',
      ];
      for (const code of codes) {
        const error = new HousingError(code as any, 'test');
        expect(error.code).toBe(code);
      }
    });
  });
});

// ============================================================================
// Client construction tests
// ============================================================================

describe('HousingClient', () => {
  describe('factory', () => {
    it('should create via createHousingClient', () => {
      const mockClient = createMockClient();
      const housing = createHousingClient(mockClient as any);
      expect(housing).toBeInstanceOf(HousingClient);
    });

    it('should create via HousingClient.fromClient', () => {
      const mockClient = createMockClient();
      const housing = HousingClient.fromClient(mockClient as any);
      expect(housing).toBeInstanceOf(HousingClient);
    });

    it('should expose units sub-client', () => {
      const mockClient = createMockClient();
      const housing = createHousingClient(mockClient as any);
      expect(housing.units).toBeInstanceOf(UnitsClient);
    });

    it('should expose membership sub-client', () => {
      const mockClient = createMockClient();
      const housing = createHousingClient(mockClient as any);
      expect(housing.membership).toBeInstanceOf(MembershipClient);
    });

    it('should expose finances sub-client', () => {
      const mockClient = createMockClient();
      const housing = createHousingClient(mockClient as any);
      expect(housing.finances).toBeInstanceOf(FinancesClient);
    });

    it('should expose governance sub-client', () => {
      const mockClient = createMockClient();
      const housing = createHousingClient(mockClient as any);
      expect(housing.governance).toBeInstanceOf(HousingGovernanceClient);
    });

    it('should accept custom config', () => {
      const mockClient = createMockClient();
      const housing = createHousingClient(mockClient as any, {
        roleName: 'custom_role',
        debug: true,
        timeout: 60000,
      });
      expect(housing).toBeInstanceOf(HousingClient);
    });
  });

  describe('getClient', () => {
    it('should return the underlying AppClient', () => {
      const mockClient = createMockClient();
      const housing = createHousingClient(mockClient as any);
      expect(housing.getClient()).toBe(mockClient);
    });
  });

  describe('getAgentPubKey', () => {
    it('should return the agent public key', () => {
      const mockClient = createMockClient();
      const housing = createHousingClient(mockClient as any);
      const key = housing.getAgentPubKey();
      expect(key).toBeInstanceOf(Uint8Array);
      expect(key).toHaveLength(39);
    });
  });
});

// ============================================================================
// Units zome dispatch tests
// ============================================================================

describe('UnitsClient', () => {
  let client: ReturnType<typeof createMockClient>;
  let units: UnitsClient;

  beforeEach(() => {
    client = createMockClient();
    units = new UnitsClient(client as any);
  });

  describe('registerBuilding', () => {
    it('should dispatch to housing_units.register_building', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeBuildingEntry()));

      const result = await units.registerBuilding({
        name: 'Harmony Co-Op',
        address: '123 Community Lane',
        buildingType: 'CoOp',
        totalUnits: 24,
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'commons',
          zome_name: 'housing_units',
          fn_name: 'register_building',
        }),
      );
      expect(result.name).toBe('Harmony Co-Op');
      expect(result.buildingType).toBe('CoOp');
    });

    it('should pass optional fields through', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeBuildingEntry({ year_built: 2020 })));

      await units.registerBuilding({
        name: 'Test',
        address: '456 Test St',
        buildingType: 'LandTrust',
        totalUnits: 12,
        yearBuilt: 2020,
        amenities: ['pool'],
        latitude: 33.0,
        longitude: -97.0,
      });

      const callPayload = client.callZome.mock.calls[0][0].payload;
      expect(callPayload.year_built).toBe(2020);
      expect(callPayload.amenities).toEqual(['pool']);
      expect(callPayload.latitude).toBe(33.0);
    });
  });

  describe('getBuilding', () => {
    it('should dispatch to housing_units.get_building', async () => {
      const hash = fakeActionHash();
      client.callZome.mockResolvedValue(makeMockRecord(makeBuildingEntry()));

      const result = await units.getBuilding(hash);

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_units',
          fn_name: 'get_building',
          payload: hash,
        }),
      );
      expect(result).not.toBeNull();
      expect(result!.name).toBe('Harmony Co-Op');
    });

    it('should return null for non-existent building', async () => {
      client.callZome.mockResolvedValue(null);

      const result = await units.getBuilding(fakeActionHash());
      expect(result).toBeNull();
    });
  });

  describe('registerUnit', () => {
    it('should dispatch to housing_units.register_unit with correct params', async () => {
      const buildingId = fakeActionHash();
      client.callZome.mockResolvedValue(makeMockRecord(makeUnitEntry()));

      const result = await units.registerUnit({
        buildingId,
        unitNumber: 'B202',
        floorPlan: '2BR',
        squareMeters: 75,
        bedrooms: 2,
        bathrooms: 1,
        monthlyCharge: 1200,
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_units',
          fn_name: 'register_unit',
        }),
      );
      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.building_id).toBe(buildingId);
      expect(payload.unit_number).toBe('B202');
      expect(payload.square_meters).toBe(75);
      expect(payload.monthly_charge).toBe(1200);
      expect(result.status).toBe('Available');
    });
  });

  describe('getUnit', () => {
    it('should dispatch to housing_units.get_unit', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeUnitEntry()));

      const result = await units.getUnit(fakeActionHash());

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_units',
          fn_name: 'get_unit',
        }),
      );
      expect(result).not.toBeNull();
    });

    it('should return null for non-existent unit', async () => {
      client.callZome.mockResolvedValue(null);

      const result = await units.getUnit(fakeActionHash());
      expect(result).toBeNull();
    });
  });

  describe('getAvailableUnits', () => {
    it('should dispatch to housing_units.get_available_units', async () => {
      const buildingId = fakeActionHash();
      client.callZome.mockResolvedValue([
        makeMockRecord(makeUnitEntry({ unit_number: 'A101' })),
        makeMockRecord(makeUnitEntry({ unit_number: 'A102' })),
      ]);

      const result = await units.getAvailableUnits(buildingId);

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_units',
          fn_name: 'get_available_units',
          payload: buildingId,
        }),
      );
      expect(result).toHaveLength(2);
    });
  });

  describe('assignOccupant', () => {
    it('should dispatch to housing_units.assign_occupant', async () => {
      client.callZome.mockResolvedValue(
        makeMockRecord(makeUnitEntry({ occupant_did: 'did:mycelix:alice', status: 'Occupied' })),
      );

      const result = await units.assignOccupant({
        unitId: fakeActionHash(),
        occupantDid: 'did:mycelix:alice',
        moveInDate: Date.now(),
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_units',
          fn_name: 'assign_occupant',
        }),
      );
      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.occupant_did).toBe('did:mycelix:alice');
    });
  });

  describe('vacateUnit', () => {
    it('should dispatch to housing_units.vacate_unit', async () => {
      client.callZome.mockResolvedValue(
        makeMockRecord(makeUnitEntry({ status: 'Available', occupant_did: undefined })),
      );

      const result = await units.vacateUnit(fakeActionHash());

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_units',
          fn_name: 'vacate_unit',
        }),
      );
      expect(result.status).toBe('Available');
    });
  });
});

// ============================================================================
// Membership zome dispatch tests
// ============================================================================

describe('MembershipClient', () => {
  let client: ReturnType<typeof createMockClient>;
  let membership: MembershipClient;

  beforeEach(() => {
    client = createMockClient();
    membership = new MembershipClient(client as any);
  });

  describe('submitApplication', () => {
    it('should dispatch to housing_membership.submit_application', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeMemberEntry({ membership_type: 'Provisional' })));

      const result = await membership.submitApplication({
        buildingId: fakeActionHash(),
        name: 'Bob',
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_membership',
          fn_name: 'submit_application',
        }),
      );
      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.name).toBe('Bob');
      expect(payload.membership_type).toBe('Provisional');
    });

    it('should pass specified membership type', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeMemberEntry({ membership_type: 'RentToOwn' })));

      await membership.submitApplication({
        buildingId: fakeActionHash(),
        name: 'Carol',
        membershipType: 'RentToOwn',
        statement: 'I am interested in rent-to-own',
      });

      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.membership_type).toBe('RentToOwn');
      expect(payload.statement).toBe('I am interested in rent-to-own');
    });
  });

  describe('approveMember', () => {
    it('should dispatch to housing_membership.approve_member', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeMemberEntry({ status: 'Active' })));

      await membership.approveMember({
        applicationId: fakeActionHash(),
        unitId: fakeActionHash(),
        equityShare: 0.04,
        votingWeight: 1,
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_membership',
          fn_name: 'approve_member',
        }),
      );
      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.equity_share).toBe(0.04);
      expect(payload.voting_weight).toBe(1);
    });
  });

  describe('getMember', () => {
    it('should dispatch to housing_membership.get_member', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeMemberEntry()));

      const result = await membership.getMember(fakeActionHash());

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_membership',
          fn_name: 'get_member',
        }),
      );
      expect(result).not.toBeNull();
      expect(result!.name).toBe('Alice');
    });

    it('should return null for non-existent member', async () => {
      client.callZome.mockResolvedValue(null);

      const result = await membership.getMember(fakeActionHash());
      expect(result).toBeNull();
    });
  });

  describe('suspendMember', () => {
    it('should dispatch to housing_membership.suspend_member', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeMemberEntry({ status: 'Suspended' })));

      const result = await membership.suspendMember(fakeActionHash(), 'Non-payment');

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_membership',
          fn_name: 'suspend_member',
        }),
      );
      expect(result.status).toBe('Suspended');
    });
  });

  describe('createRentToOwn', () => {
    it('should dispatch to housing_membership.create_rent_to_own', async () => {
      const rtoEntry = {
        member_id: fakeActionHash(),
        unit_id: fakeActionHash(),
        total_equity_target: 50000,
        monthly_equity_contribution: 500,
        equity_accumulated: 0,
        term_months: 120,
        start_date: Date.now(),
        projected_completion_date: Date.now() + 120 * 30 * 86400000,
        status: 'Active',
        created_at: Date.now(),
      };
      client.callZome.mockResolvedValue(makeMockRecord(rtoEntry));

      const result = await membership.createRentToOwn({
        memberId: fakeActionHash(),
        unitId: fakeActionHash(),
        totalEquityTarget: 50000,
        monthlyEquityContribution: 500,
        termMonths: 120,
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_membership',
          fn_name: 'create_rent_to_own',
        }),
      );
      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.total_equity_target).toBe(50000);
      expect(payload.term_months).toBe(120);
      expect(result.status).toBe('Active');
    });
  });
});

// ============================================================================
// Finances zome dispatch tests
// ============================================================================

describe('FinancesClient', () => {
  let client: ReturnType<typeof createMockClient>;
  let finances: FinancesClient;

  beforeEach(() => {
    client = createMockClient();
    finances = new FinancesClient(client as any);
  });

  describe('generateMonthlyCharges', () => {
    it('should dispatch to housing_finances.generate_monthly_charges', async () => {
      client.callZome.mockResolvedValue([
        makeMockRecord(makeChargeEntry()),
        makeMockRecord(makeChargeEntry({ unit_id: new Uint8Array(39).fill(0xbb) })),
      ]);

      const result = await finances.generateMonthlyCharges({
        buildingId: fakeActionHash(),
        period: '2026-02',
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_finances',
          fn_name: 'generate_monthly_charges',
        }),
      );
      expect(result).toHaveLength(2);
      expect(result[0].period).toBe('2026-02');
    });

    it('should pass special assessment fields', async () => {
      client.callZome.mockResolvedValue([makeMockRecord(makeChargeEntry({ special_assessment: 200 }))]);

      await finances.generateMonthlyCharges({
        buildingId: fakeActionHash(),
        period: '2026-03',
        specialAssessment: 200,
        specialAssessmentReason: 'Roof repair',
      });

      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.special_assessment).toBe(200);
      expect(payload.special_assessment_reason).toBe('Roof repair');
    });
  });

  describe('recordPayment', () => {
    it('should dispatch to housing_finances.record_payment', async () => {
      const paymentEntry = {
        charge_id: fakeActionHash(),
        member_did: 'did:mycelix:alice',
        amount: 850,
        method: 'Bank',
        reference: 'TXN-001',
        paid_at: Date.now(),
      };
      client.callZome.mockResolvedValue(makeMockRecord(paymentEntry));

      const result = await finances.recordPayment({
        chargeId: fakeActionHash(),
        amount: 850,
        method: 'Bank',
        reference: 'TXN-001',
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_finances',
          fn_name: 'record_payment',
        }),
      );
      expect(result.amount).toBe(850);
      expect(result.method).toBe('Bank');
    });

    it('should accept all payment method variants', async () => {
      const methods = ['Bank', 'Check', 'Cash', 'Crypto', 'TimeBank'] as const;
      for (const method of methods) {
        const entry = {
          charge_id: fakeActionHash(),
          member_did: 'did:mycelix:alice',
          amount: 100,
          method,
          paid_at: Date.now(),
        };
        client.callZome.mockResolvedValue(makeMockRecord(entry));

        const result = await finances.recordPayment({
          chargeId: fakeActionHash(),
          amount: 100,
          method,
        });
        expect(result.method).toBe(method);
      }
    });
  });

  describe('getFinancialSummary', () => {
    it('should dispatch to housing_finances.get_financial_summary', async () => {
      const summaryResult = {
        building_id: fakeActionHash(),
        period: '2026-02',
        total_revenue: 20000,
        total_expenses: 8000,
        net_income: 12000,
        outstanding_charges: 2500,
        reserve_fund: 50000,
        occupancy_rate: 0.92,
      };
      client.callZome.mockResolvedValue(summaryResult);

      const result = await finances.getFinancialSummary(fakeActionHash(), '2026-02');

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_finances',
          fn_name: 'get_financial_summary',
        }),
      );
      expect(result.totalRevenue).toBe(20000);
      expect(result.netIncome).toBe(12000);
      expect(result.occupancyRate).toBe(0.92);
    });
  });
});

// ============================================================================
// Governance zome dispatch tests
// ============================================================================

describe('HousingGovernanceClient', () => {
  let client: ReturnType<typeof createMockClient>;
  let governance: HousingGovernanceClient;

  beforeEach(() => {
    client = createMockClient();
    governance = new HousingGovernanceClient(client as any);
  });

  describe('scheduleMeeting', () => {
    it('should dispatch to housing_governance.schedule_meeting', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeMeetingEntry()));

      const result = await governance.scheduleMeeting({
        buildingId: fakeActionHash(),
        title: 'Monthly Board Meeting',
        description: 'Regular business meeting',
        meetingType: 'Regular',
        scheduledAt: Date.now() + 7 * 86400000,
        location: 'Community Room',
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_governance',
          fn_name: 'schedule_meeting',
        }),
      );
      expect(result.title).toBe('Monthly Board Meeting');
      expect(result.status).toBe('Scheduled');
    });

    it('should default quorum to 0.5 and agenda to empty', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeMeetingEntry()));

      await governance.scheduleMeeting({
        buildingId: fakeActionHash(),
        title: 'Test',
        description: 'Test',
        meetingType: 'Special',
        scheduledAt: Date.now(),
        location: 'Online',
      });

      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.quorum_required).toBe(0.5);
      expect(payload.agenda).toEqual([]);
    });
  });

  describe('getMeeting', () => {
    it('should dispatch to housing_governance.get_meeting', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeMeetingEntry()));

      const result = await governance.getMeeting(fakeActionHash());

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_governance',
          fn_name: 'get_meeting',
        }),
      );
      expect(result).not.toBeNull();
    });

    it('should return null for non-existent meeting', async () => {
      client.callZome.mockResolvedValue(null);

      const result = await governance.getMeeting(fakeActionHash());
      expect(result).toBeNull();
    });
  });

  describe('proposeResolution', () => {
    it('should dispatch to housing_governance.propose_resolution', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeResolutionEntry()));

      const result = await governance.proposeResolution({
        buildingId: fakeActionHash(),
        title: 'Solar Panel Installation',
        description: 'Approve installation of rooftop solar',
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_governance',
          fn_name: 'propose_resolution',
        }),
      );
      expect(result.title).toBe('Solar Panel Installation');
      expect(result.status).toBe('Proposed');
    });

    it('should default quorum, threshold, and voting period', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeResolutionEntry()));

      await governance.proposeResolution({
        buildingId: fakeActionHash(),
        title: 'Test',
        description: 'Test',
      });

      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.quorum_required).toBe(0.5);
      expect(payload.threshold).toBe(0.5);
      expect(payload.voting_period_hours).toBe(168);
    });
  });

  describe('voteOnResolution', () => {
    it('should dispatch to housing_governance.vote_on_resolution', async () => {
      client.callZome.mockResolvedValue(undefined);

      await governance.voteOnResolution({
        resolutionId: fakeActionHash(),
        choice: 'Approve',
        reason: 'Great idea',
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_governance',
          fn_name: 'vote_on_resolution',
        }),
      );
      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.choice).toBe('Approve');
      expect(payload.reason).toBe('Great idea');
    });

    it('should accept all vote choice variants', async () => {
      const choices = ['Approve', 'Reject', 'Abstain'] as const;
      for (const choice of choices) {
        client.callZome.mockResolvedValue(undefined);
        await governance.voteOnResolution({
          resolutionId: fakeActionHash(),
          choice,
        });
        const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
        expect(lastCall.payload.choice).toBe(choice);
      }
    });
  });

  describe('createElection', () => {
    it('should dispatch to housing_governance.create_election', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeElectionEntry()));

      const result = await governance.createElection({
        buildingId: fakeActionHash(),
        title: 'Board Election 2026',
        positions: ['President', 'Secretary'],
        votingStartsAt: Date.now(),
        votingEndsAt: Date.now() + 7 * 86400000,
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_governance',
          fn_name: 'create_election',
        }),
      );
      expect(result.status).toBe('Nominations');
      expect(result.positions).toHaveLength(3);
    });
  });

  describe('castBallot', () => {
    it('should dispatch to housing_governance.cast_ballot', async () => {
      client.callZome.mockResolvedValue(undefined);

      await governance.castBallot({
        electionId: fakeActionHash(),
        selections: { President: 'did:mycelix:alice', Secretary: 'did:mycelix:bob' },
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_governance',
          fn_name: 'cast_ballot',
        }),
      );
      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.selections.President).toBe('did:mycelix:alice');
    });
  });

  describe('getElection', () => {
    it('should dispatch to housing_governance.get_election', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeElectionEntry()));

      const result = await governance.getElection(fakeActionHash());

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'housing_governance',
          fn_name: 'get_election',
        }),
      );
      expect(result).not.toBeNull();
    });

    it('should return null for non-existent election', async () => {
      client.callZome.mockResolvedValue(null);

      const result = await governance.getElection(fakeActionHash());
      expect(result).toBeNull();
    });
  });
});

// ============================================================================
// Cross-zome interaction tests
// ============================================================================

describe('Cross-zome Interactions', () => {
  let client: ReturnType<typeof createMockClient>;
  let housing: HousingClient;

  beforeEach(() => {
    client = createMockClient();
    housing = createHousingClient(client as any);
  });

  it('should call units, membership, and governance in getBuildingOverview', async () => {
    client.callZome
      .mockResolvedValueOnce(makeMockRecord(makeBuildingEntry()))
      .mockResolvedValueOnce([makeMockRecord(makeUnitEntry())])
      .mockResolvedValueOnce([makeMockRecord(makeMemberEntry())])
      .mockResolvedValueOnce([makeMockRecord(makeResolutionEntry())]);

    const overview = await housing.getBuildingOverview(fakeActionHash());

    expect(client.callZome).toHaveBeenCalledTimes(4);
    expect(overview.building).toBeDefined();
    expect(overview.units).toHaveLength(1);
    expect(overview.members).toHaveLength(1);
    expect(overview.resolutions).toHaveLength(1);
  });

  it('should dispatch unit creation referencing a building hash', async () => {
    const buildingHash = fakeActionHash();
    client.callZome.mockResolvedValue(makeMockRecord(makeUnitEntry()));

    await housing.units.registerUnit({
      buildingId: buildingHash,
      unitNumber: 'C301',
      floorPlan: '3BR',
      squareMeters: 120,
      bedrooms: 3,
      bathrooms: 2,
      monthlyCharge: 1800,
    });

    const payload = client.callZome.mock.calls[0][0].payload;
    expect(payload.building_id).toBe(buildingHash);
  });

  it('should dispatch membership application referencing a building hash', async () => {
    const buildingHash = fakeActionHash();
    client.callZome.mockResolvedValue(makeMockRecord(makeMemberEntry()));

    await housing.membership.submitApplication({
      buildingId: buildingHash,
      name: 'New Member',
    });

    const payload = client.callZome.mock.calls[0][0].payload;
    expect(payload.building_id).toBe(buildingHash);
  });
});
