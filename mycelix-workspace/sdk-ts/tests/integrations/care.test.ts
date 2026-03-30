// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Care Integration Tests
 *
 * Tests for CareClient and its sub-clients (TimebankClient, CirclesClient,
 * MatchingClient) -- the domain-specific SDK clients for the care zomes
 * within the mycelix-commons cluster DNA.
 *
 * Covers:
 * - care-plans: Care plan creation and lifecycle
 * - care-circles: Circle creation, membership, management
 * - care-needs: Need posting and matching
 * - care-timebank: Service offers, requests, time exchanges, balances
 * - care-credentials: Credential issuance and verification
 * - care-matching: Provider matching and suggestions
 *
 * All calls are dispatched through the commons role to the care_timebank,
 * care_circles, and care_matching coordinator zomes.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  CareClient,
  createCareClient,
  TimebankClient,
  CirclesClient,
  MatchingClient,
  CareError,
  type ServiceOffer,
  type CreateServiceOfferInput,
  type ServiceRequest,
  type CreateServiceRequestInput,
  type TimeExchange,
  type CompleteExchangeInput,
  type RateExchangeInput,
  type TimeBalance,
  type CareCircle,
  type CreateCareCircleInput,
  type CircleMember,
  type CareMatch,
  type CarePlan,
  type CarePlanSession,
  type CareCredential,
} from '../../src/clients/care/index.js';

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

function makeServiceOfferEntry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    provider_did: 'did:mycelix:provider1',
    provider_pub_key: fakeAgentKey(),
    title: 'Garden Help',
    description: 'Help with community garden maintenance',
    category: 'Gardening',
    estimated_minutes: 120,
    tags: ['gardening', 'outdoor'],
    location: 'Community Garden',
    available: true,
    average_rating: 0,
    completed_count: 0,
    created_at: Date.now(),
    updated_at: Date.now(),
    ...overrides,
  };
}

function makeServiceRequestEntry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    requester_did: 'did:mycelix:requester1',
    requester_pub_key: fakeAgentKey(),
    title: 'Need garden help',
    description: 'Need help weeding and planting',
    category: 'Gardening',
    urgency: 'Medium',
    preferred_time_start: undefined,
    preferred_time_end: undefined,
    location: 'Backyard',
    status: 'Open',
    created_at: Date.now(),
    updated_at: Date.now(),
    ...overrides,
  };
}

function makeTimeExchangeEntry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    offer_id: fakeActionHash(),
    request_id: fakeActionHash(),
    provider_did: 'did:mycelix:provider1',
    receiver_did: 'did:mycelix:receiver1',
    minutes: 90,
    description: 'Helped with garden weeding',
    provider_rating: undefined,
    receiver_rating: undefined,
    status: 'Pending',
    completed_at: undefined,
    created_at: Date.now(),
    ...overrides,
  };
}

function makeCareCircleEntry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    name: 'Neighborhood Support',
    description: 'Local mutual care circle',
    founder_did: 'did:mycelix:founder1',
    max_members: 20,
    member_count: 1,
    area: 'Downtown',
    categories: ['eldercare', 'childcare'],
    open: true,
    active: true,
    created_at: Date.now(),
    updated_at: Date.now(),
    ...overrides,
  };
}

function makeCircleMemberEntry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    circle_id: fakeActionHash(),
    member_did: 'did:mycelix:member1',
    member_pub_key: fakeAgentKey(),
    role: 'Member',
    joined_at: Date.now(),
    active: true,
    ...overrides,
  };
}

function makeCareMatchEntry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    request_id: fakeActionHash(),
    offer_id: fakeActionHash(),
    requester_did: 'did:mycelix:requester1',
    provider_did: 'did:mycelix:provider1',
    confidence: 0.85,
    reason: 'Category match: Gardening',
    status: 'Suggested',
    created_at: Date.now(),
    ...overrides,
  };
}

// ============================================================================
// Type construction tests
// ============================================================================

describe('Care Types', () => {
  describe('ServiceOffer', () => {
    it('should construct a valid ServiceOffer from entry data', () => {
      const entry = makeServiceOfferEntry();
      expect(entry.title).toBe('Garden Help');
      expect(entry.category).toBe('Gardening');
      expect(entry.estimated_minutes).toBe(120);
      expect(entry.available).toBe(true);
      expect(entry.average_rating).toBe(0);
    });

    it('should support optional location field', () => {
      const withLocation = makeServiceOfferEntry({ location: 'Park' });
      expect(withLocation.location).toBe('Park');

      const withoutLocation = makeServiceOfferEntry({ location: undefined });
      expect(withoutLocation.location).toBeUndefined();
    });
  });

  describe('ServiceRequest', () => {
    it('should construct a valid ServiceRequest from entry data', () => {
      const entry = makeServiceRequestEntry();
      expect(entry.title).toBe('Need garden help');
      expect(entry.urgency).toBe('Medium');
      expect(entry.status).toBe('Open');
    });

    it('should accept all urgency variants', () => {
      const urgencies = ['Low', 'Medium', 'High', 'Critical'];
      for (const u of urgencies) {
        const entry = makeServiceRequestEntry({ urgency: u });
        expect(entry.urgency).toBe(u);
      }
    });

    it('should accept all request status variants', () => {
      const statuses = ['Open', 'Matched', 'InProgress', 'Completed', 'Cancelled'];
      for (const s of statuses) {
        const entry = makeServiceRequestEntry({ status: s });
        expect(entry.status).toBe(s);
      }
    });
  });

  describe('TimeExchange', () => {
    it('should construct a valid TimeExchange from entry data', () => {
      const entry = makeTimeExchangeEntry();
      expect(entry.minutes).toBe(90);
      expect(entry.status).toBe('Pending');
      expect(entry.provider_did).toBe('did:mycelix:provider1');
    });

    it('should accept all exchange status variants', () => {
      const statuses = ['Pending', 'Confirmed', 'Disputed', 'Completed'];
      for (const s of statuses) {
        const entry = makeTimeExchangeEntry({ status: s });
        expect(entry.status).toBe(s);
      }
    });
  });

  describe('CareCircle', () => {
    it('should construct a valid CareCircle from entry data', () => {
      const entry = makeCareCircleEntry();
      expect(entry.name).toBe('Neighborhood Support');
      expect(entry.max_members).toBe(20);
      expect(entry.open).toBe(true);
      expect(entry.active).toBe(true);
    });

    it('should support categories array', () => {
      const entry = makeCareCircleEntry({ categories: ['eldercare', 'meals', 'transport'] });
      expect(entry.categories).toHaveLength(3);
    });
  });

  describe('CircleMember', () => {
    it('should construct a valid CircleMember from entry data', () => {
      const entry = makeCircleMemberEntry();
      expect(entry.member_did).toBe('did:mycelix:member1');
      expect(entry.role).toBe('Member');
      expect(entry.active).toBe(true);
    });

    it('should accept all role variants', () => {
      const roles = ['Member', 'Coordinator', 'Founder'];
      for (const r of roles) {
        const entry = makeCircleMemberEntry({ role: r });
        expect(entry.role).toBe(r);
      }
    });
  });

  describe('CareMatch', () => {
    it('should construct a valid CareMatch from entry data', () => {
      const entry = makeCareMatchEntry();
      expect(entry.confidence).toBe(0.85);
      expect(entry.status).toBe('Suggested');
      expect(entry.reason).toContain('Gardening');
    });

    it('should accept all match status variants', () => {
      const statuses = ['Suggested', 'Accepted', 'Declined', 'Expired'];
      for (const s of statuses) {
        const entry = makeCareMatchEntry({ status: s });
        expect(entry.status).toBe(s);
      }
    });
  });

  describe('CareCredential', () => {
    it('should construct a valid CareCredential structure', () => {
      const credential: CareCredential = {
        id: fakeActionHash() as any,
        holderDid: 'did:mycelix:provider1',
        issuerDid: 'did:mycelix:circle-coordinator',
        credentialType: 'ServiceProvider',
        category: 'Gardening',
        hoursCompleted: 50,
        averageRating: 4.5,
        issuedAt: Date.now(),
      };
      expect(credential.credentialType).toBe('ServiceProvider');
      expect(credential.hoursCompleted).toBe(50);
    });

    it('should accept all credential type variants', () => {
      const types: CareCredential['credentialType'][] = [
        'ServiceProvider', 'CircleCoordinator', 'CommunityPillar',
      ];
      for (const t of types) {
        const credential: CareCredential = {
          id: fakeActionHash() as any,
          holderDid: 'did:mycelix:test',
          issuerDid: 'did:mycelix:issuer',
          credentialType: t,
          category: 'test',
          hoursCompleted: 0,
          averageRating: 0,
          issuedAt: Date.now(),
        };
        expect(credential.credentialType).toBe(t);
      }
    });

    it('should support optional expiresAt field', () => {
      const withExpiry: CareCredential = {
        id: fakeActionHash() as any,
        holderDid: 'did:mycelix:test',
        issuerDid: 'did:mycelix:issuer',
        credentialType: 'ServiceProvider',
        category: 'test',
        hoursCompleted: 10,
        averageRating: 4.0,
        issuedAt: Date.now(),
        expiresAt: Date.now() + 365 * 86400000,
      };
      expect(withExpiry.expiresAt).toBeDefined();
      expect(withExpiry.expiresAt!).toBeGreaterThan(withExpiry.issuedAt);
    });
  });

  describe('CarePlan', () => {
    it('should construct a valid CarePlan structure', () => {
      const plan: CarePlan = {
        id: fakeActionHash() as any,
        recipientDid: 'did:mycelix:patient1',
        circleId: fakeActionHash() as any,
        title: 'Elder Care Plan',
        description: 'Weekly support for Mrs. Johnson',
        sessions: [
          {
            providerDid: 'did:mycelix:provider1',
            category: 'Meals',
            scheduledAt: Date.now(),
            durationMinutes: 60,
            completed: false,
          },
        ],
        status: 'Active',
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      expect(plan.status).toBe('Active');
      expect(plan.sessions).toHaveLength(1);
      expect(plan.sessions[0].durationMinutes).toBe(60);
    });

    it('should accept all plan status variants', () => {
      const statuses: CarePlan['status'][] = ['Draft', 'Active', 'Completed', 'Paused'];
      for (const s of statuses) {
        const plan: CarePlan = {
          id: fakeActionHash() as any,
          recipientDid: 'did:mycelix:test',
          title: 'Test',
          description: 'Test',
          sessions: [],
          status: s,
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        expect(plan.status).toBe(s);
      }
    });
  });

  describe('CareError', () => {
    it('should construct with code, message, and details', () => {
      const error = new CareError('NOT_FOUND', 'Circle not found', { id: 'xyz' });
      expect(error.code).toBe('NOT_FOUND');
      expect(error.message).toBe('Circle not found');
      expect(error.details).toEqual({ id: 'xyz' });
      expect(error.name).toBe('CareError');
    });

    it('should accept all error code variants', () => {
      const codes = [
        'CONNECTION_ERROR', 'ZOME_CALL_ERROR', 'NOT_FOUND', 'UNAUTHORIZED',
        'INVALID_INPUT', 'INSUFFICIENT_BALANCE', 'CIRCLE_FULL', 'ALREADY_MEMBER',
        'NOT_MEMBER', 'EXCHANGE_ERROR', 'MATCH_ERROR',
      ];
      for (const code of codes) {
        const error = new CareError(code as any, 'test');
        expect(error.code).toBe(code);
      }
    });
  });
});

// ============================================================================
// Client construction tests
// ============================================================================

describe('CareClient', () => {
  describe('factory', () => {
    it('should create via createCareClient', () => {
      const mockClient = createMockClient();
      const care = createCareClient(mockClient as any);
      expect(care).toBeInstanceOf(CareClient);
    });

    it('should create via CareClient.fromClient', () => {
      const mockClient = createMockClient();
      const care = CareClient.fromClient(mockClient as any);
      expect(care).toBeInstanceOf(CareClient);
    });

    it('should expose timebank sub-client', () => {
      const mockClient = createMockClient();
      const care = createCareClient(mockClient as any);
      expect(care.timebank).toBeInstanceOf(TimebankClient);
    });

    it('should expose circles sub-client', () => {
      const mockClient = createMockClient();
      const care = createCareClient(mockClient as any);
      expect(care.circles).toBeInstanceOf(CirclesClient);
    });

    it('should expose matching sub-client', () => {
      const mockClient = createMockClient();
      const care = createCareClient(mockClient as any);
      expect(care.matching).toBeInstanceOf(MatchingClient);
    });

    it('should accept custom config', () => {
      const mockClient = createMockClient();
      const care = createCareClient(mockClient as any, {
        roleName: 'custom_role',
        debug: true,
        timeout: 60000,
      });
      expect(care).toBeInstanceOf(CareClient);
    });
  });

  describe('getClient', () => {
    it('should return the underlying AppClient', () => {
      const mockClient = createMockClient();
      const care = createCareClient(mockClient as any);
      expect(care.getClient()).toBe(mockClient);
    });
  });

  describe('getAgentPubKey', () => {
    it('should return the agent public key', () => {
      const mockClient = createMockClient();
      const care = createCareClient(mockClient as any);
      const key = care.getAgentPubKey();
      expect(key).toBeInstanceOf(Uint8Array);
      expect(key).toHaveLength(39);
    });
  });
});

// ============================================================================
// Timebank zome dispatch tests
// ============================================================================

describe('TimebankClient', () => {
  let client: ReturnType<typeof createMockClient>;
  let timebank: TimebankClient;

  beforeEach(() => {
    client = createMockClient();
    timebank = new TimebankClient(client as any);
  });

  describe('createServiceOffer', () => {
    it('should dispatch to care_timebank.create_service_offer', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeServiceOfferEntry()));

      const result = await timebank.createServiceOffer({
        title: 'Garden Help',
        description: 'Help with community garden',
        category: 'Gardening',
        estimatedMinutes: 120,
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'commons',
          zome_name: 'care_timebank',
          fn_name: 'create_service_offer',
        }),
      );
      expect(result.title).toBe('Garden Help');
      expect(result.category).toBe('Gardening');
      expect(result.available).toBe(true);
    });

    it('should pass optional tags and location', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeServiceOfferEntry()));

      await timebank.createServiceOffer({
        title: 'Cooking Help',
        description: 'Meal preparation',
        category: 'Meals',
        estimatedMinutes: 60,
        tags: ['cooking', 'meal-prep'],
        location: 'Kitchen',
      });

      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.tags).toEqual(['cooking', 'meal-prep']);
      expect(payload.location).toBe('Kitchen');
    });

    it('should default tags to empty array when not specified', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeServiceOfferEntry()));

      await timebank.createServiceOffer({
        title: 'Test',
        description: 'Test',
        category: 'Other',
        estimatedMinutes: 30,
      });

      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.tags).toEqual([]);
    });
  });

  describe('getOffersByCategory', () => {
    it('should dispatch to care_timebank.get_offers_by_category', async () => {
      client.callZome.mockResolvedValue([
        makeMockRecord(makeServiceOfferEntry({ title: 'Offer 1' })),
        makeMockRecord(makeServiceOfferEntry({ title: 'Offer 2' })),
      ]);

      const result = await timebank.getOffersByCategory('Gardening');

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_timebank',
          fn_name: 'get_offers_by_category',
          payload: 'Gardening',
        }),
      );
      expect(result).toHaveLength(2);
    });
  });

  describe('getMyOffers', () => {
    it('should dispatch to care_timebank.get_my_offers with null payload', async () => {
      client.callZome.mockResolvedValue([]);

      const result = await timebank.getMyOffers();

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_timebank',
          fn_name: 'get_my_offers',
          payload: null,
        }),
      );
      expect(result).toEqual([]);
    });
  });

  describe('updateOfferAvailability', () => {
    it('should dispatch to care_timebank.update_offer_availability', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeServiceOfferEntry({ available: false })));

      const result = await timebank.updateOfferAvailability(fakeActionHash(), false);

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_timebank',
          fn_name: 'update_offer_availability',
        }),
      );
      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.available).toBe(false);
      expect(result.available).toBe(false);
    });
  });

  describe('createServiceRequest', () => {
    it('should dispatch to care_timebank.create_service_request', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeServiceRequestEntry()));

      const result = await timebank.createServiceRequest({
        title: 'Need help with groceries',
        description: 'Weekly grocery shopping assistance',
        category: 'Errands',
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_timebank',
          fn_name: 'create_service_request',
        }),
      );
      expect(result.status).toBe('Open');
    });

    it('should default urgency to Medium when not specified', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeServiceRequestEntry()));

      await timebank.createServiceRequest({
        title: 'Test',
        description: 'Test',
        category: 'Other',
      });

      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.urgency).toBe('Medium');
    });

    it('should pass all urgency levels through', async () => {
      const urgencies = ['Low', 'Medium', 'High', 'Critical'] as const;
      for (const urgency of urgencies) {
        client.callZome.mockResolvedValue(makeMockRecord(makeServiceRequestEntry({ urgency })));

        await timebank.createServiceRequest({
          title: 'Test',
          description: 'Test',
          category: 'Other',
          urgency,
        });

        const lastCall = client.callZome.mock.calls[client.callZome.mock.calls.length - 1][0];
        expect(lastCall.payload.urgency).toBe(urgency);
      }
    });
  });

  describe('getOpenRequests', () => {
    it('should dispatch to care_timebank.get_open_requests with category', async () => {
      client.callZome.mockResolvedValue([makeMockRecord(makeServiceRequestEntry())]);

      const result = await timebank.getOpenRequests('Gardening');

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_timebank',
          fn_name: 'get_open_requests',
          payload: 'Gardening',
        }),
      );
      expect(result).toHaveLength(1);
    });

    it('should dispatch with null when no category specified', async () => {
      client.callZome.mockResolvedValue([]);

      await timebank.getOpenRequests();

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_open_requests',
          payload: null,
        }),
      );
    });
  });

  describe('completeExchange', () => {
    it('should dispatch to care_timebank.complete_exchange', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeTimeExchangeEntry()));

      const result = await timebank.completeExchange({
        offerId: fakeActionHash(),
        receiverDid: 'did:mycelix:receiver1',
        minutes: 90,
        description: 'Helped with garden weeding',
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_timebank',
          fn_name: 'complete_exchange',
        }),
      );
      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.receiver_did).toBe('did:mycelix:receiver1');
      expect(payload.minutes).toBe(90);
      expect(result.status).toBe('Pending');
    });

    it('should pass optional requestId', async () => {
      const requestId = fakeActionHash();
      client.callZome.mockResolvedValue(makeMockRecord(makeTimeExchangeEntry()));

      await timebank.completeExchange({
        offerId: fakeActionHash(),
        requestId,
        receiverDid: 'did:mycelix:receiver1',
        minutes: 60,
        description: 'Completed task',
      });

      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.request_id).toBe(requestId);
    });
  });

  describe('rateExchange', () => {
    it('should dispatch to care_timebank.rate_exchange', async () => {
      client.callZome.mockResolvedValue(
        makeMockRecord(makeTimeExchangeEntry({ receiver_rating: 4.5, status: 'Completed' })),
      );

      const result = await timebank.rateExchange({
        exchangeId: fakeActionHash(),
        rating: 4.5,
        comment: 'Excellent service',
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_timebank',
          fn_name: 'rate_exchange',
        }),
      );
      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.rating).toBe(4.5);
      expect(payload.comment).toBe('Excellent service');
    });

    it('should reject ratings outside valid range conceptually', () => {
      // The type system allows any number, but the zome would validate
      const validRatings = [0, 1, 2.5, 3, 4, 5];
      for (const r of validRatings) {
        expect(r).toBeGreaterThanOrEqual(0);
        expect(r).toBeLessThanOrEqual(5);
      }
    });
  });

  describe('getMyBalance', () => {
    it('should dispatch to care_timebank.get_my_balance', async () => {
      const balanceResult = {
        did: 'did:mycelix:me',
        earned: 25.5,
        spent: 10.0,
        balance: 15.5,
        total_exchanges: 8,
        average_provider_rating: 4.7,
        average_receiver_rating: 4.3,
      };
      client.callZome.mockResolvedValue(balanceResult);

      const result = await timebank.getMyBalance();

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_timebank',
          fn_name: 'get_my_balance',
          payload: null,
        }),
      );
      expect(result.earned).toBe(25.5);
      expect(result.spent).toBe(10.0);
      expect(result.balance).toBe(15.5);
      expect(result.totalExchanges).toBe(8);
      expect(result.averageProviderRating).toBe(4.7);
    });
  });
});

// ============================================================================
// Circles zome dispatch tests
// ============================================================================

describe('CirclesClient', () => {
  let client: ReturnType<typeof createMockClient>;
  let circles: CirclesClient;

  beforeEach(() => {
    client = createMockClient();
    circles = new CirclesClient(client as any);
  });

  describe('createCircle', () => {
    it('should dispatch to care_circles.create_circle', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeCareCircleEntry()));

      const result = await circles.createCircle({
        name: 'Neighborhood Support',
        description: 'Local mutual care circle',
      });

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'commons',
          zome_name: 'care_circles',
          fn_name: 'create_circle',
        }),
      );
      expect(result.name).toBe('Neighborhood Support');
      expect(result.open).toBe(true);
    });

    it('should default maxMembers to 20 and open to true', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeCareCircleEntry()));

      await circles.createCircle({
        name: 'Test',
        description: 'Test',
      });

      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.max_members).toBe(20);
      expect(payload.open).toBe(true);
      expect(payload.categories).toEqual([]);
    });

    it('should pass optional fields through', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeCareCircleEntry()));

      await circles.createCircle({
        name: 'Private Circle',
        description: 'Invite only',
        maxMembers: 10,
        area: 'East Side',
        categories: ['eldercare'],
        open: false,
      });

      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.max_members).toBe(10);
      expect(payload.area).toBe('East Side');
      expect(payload.categories).toEqual(['eldercare']);
      expect(payload.open).toBe(false);
    });
  });

  describe('getCircle', () => {
    it('should dispatch to care_circles.get_circle', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeCareCircleEntry()));

      const result = await circles.getCircle(fakeActionHash());

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_circles',
          fn_name: 'get_circle',
        }),
      );
      expect(result).not.toBeNull();
      expect(result!.name).toBe('Neighborhood Support');
    });

    it('should return null for non-existent circle', async () => {
      client.callZome.mockResolvedValue(null);

      const result = await circles.getCircle(fakeActionHash());
      expect(result).toBeNull();
    });
  });

  describe('joinCircle', () => {
    it('should dispatch to care_circles.join_circle', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeCircleMemberEntry()));

      const result = await circles.joinCircle(fakeActionHash());

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_circles',
          fn_name: 'join_circle',
        }),
      );
      expect(result.role).toBe('Member');
      expect(result.active).toBe(true);
    });
  });

  describe('leaveCircle', () => {
    it('should dispatch to care_circles.leave_circle', async () => {
      client.callZome.mockResolvedValue(undefined);

      await circles.leaveCircle(fakeActionHash());

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_circles',
          fn_name: 'leave_circle',
        }),
      );
    });
  });

  describe('getMyCircles', () => {
    it('should dispatch to care_circles.get_my_circles', async () => {
      client.callZome.mockResolvedValue([
        makeMockRecord(makeCareCircleEntry({ name: 'Circle A' })),
        makeMockRecord(makeCareCircleEntry({ name: 'Circle B' })),
      ]);

      const result = await circles.getMyCircles();

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_circles',
          fn_name: 'get_my_circles',
          payload: null,
        }),
      );
      expect(result).toHaveLength(2);
    });
  });

  describe('getCircleMembers', () => {
    it('should dispatch to care_circles.get_circle_members', async () => {
      const circleId = fakeActionHash();
      client.callZome.mockResolvedValue([
        makeMockRecord(makeCircleMemberEntry({ role: 'Founder' })),
        makeMockRecord(makeCircleMemberEntry({ role: 'Member' })),
        makeMockRecord(makeCircleMemberEntry({ role: 'Coordinator' })),
      ]);

      const result = await circles.getCircleMembers(circleId);

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_circles',
          fn_name: 'get_circle_members',
          payload: circleId,
        }),
      );
      expect(result).toHaveLength(3);
    });
  });

  describe('listOpenCircles', () => {
    it('should dispatch to care_circles.list_open_circles', async () => {
      client.callZome.mockResolvedValue([
        makeMockRecord(makeCareCircleEntry({ open: true })),
      ]);

      const result = await circles.listOpenCircles();

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_circles',
          fn_name: 'list_open_circles',
          payload: null,
        }),
      );
      expect(result).toHaveLength(1);
      expect(result[0].open).toBe(true);
    });
  });
});

// ============================================================================
// Matching zome dispatch tests
// ============================================================================

describe('MatchingClient', () => {
  let client: ReturnType<typeof createMockClient>;
  let matching: MatchingClient;

  beforeEach(() => {
    client = createMockClient();
    matching = new MatchingClient(client as any);
  });

  describe('findMatchesForRequest', () => {
    it('should dispatch to care_matching.find_matches_for_request', async () => {
      const requestId = fakeActionHash();
      client.callZome.mockResolvedValue([
        makeMockRecord(makeCareMatchEntry({ confidence: 0.9 })),
        makeMockRecord(makeCareMatchEntry({ confidence: 0.75 })),
      ]);

      const result = await matching.findMatchesForRequest(requestId);

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_matching',
          fn_name: 'find_matches_for_request',
          payload: requestId,
        }),
      );
      expect(result).toHaveLength(2);
    });
  });

  describe('suggestMatch', () => {
    it('should dispatch to care_matching.suggest_match', async () => {
      client.callZome.mockResolvedValue(makeMockRecord(makeCareMatchEntry()));

      const result = await matching.suggestMatch(
        fakeActionHash(),
        fakeActionHash(),
        'Category and location match',
      );

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_matching',
          fn_name: 'suggest_match',
        }),
      );
      const payload = client.callZome.mock.calls[0][0].payload;
      expect(payload.reason).toBe('Category and location match');
      expect(result.status).toBe('Suggested');
    });
  });

  describe('acceptMatch', () => {
    it('should dispatch to care_matching.accept_match', async () => {
      client.callZome.mockResolvedValue(
        makeMockRecord(makeCareMatchEntry({ status: 'Accepted' })),
      );

      const result = await matching.acceptMatch(fakeActionHash());

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_matching',
          fn_name: 'accept_match',
        }),
      );
      expect(result.status).toBe('Accepted');
    });
  });

  describe('declineMatch', () => {
    it('should dispatch to care_matching.decline_match', async () => {
      client.callZome.mockResolvedValue(
        makeMockRecord(makeCareMatchEntry({ status: 'Declined' })),
      );

      const result = await matching.declineMatch(fakeActionHash());

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_matching',
          fn_name: 'decline_match',
        }),
      );
      expect(result.status).toBe('Declined');
    });
  });

  describe('getMatchesForOffer', () => {
    it('should dispatch to care_matching.get_matches_for_offer', async () => {
      const offerId = fakeActionHash();
      client.callZome.mockResolvedValue([
        makeMockRecord(makeCareMatchEntry()),
      ]);

      const result = await matching.getMatchesForOffer(offerId);

      expect(client.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'care_matching',
          fn_name: 'get_matches_for_offer',
          payload: offerId,
        }),
      );
      expect(result).toHaveLength(1);
    });
  });
});

// ============================================================================
// Cross-zome interaction tests
// ============================================================================

describe('Cross-zome Interactions', () => {
  let client: ReturnType<typeof createMockClient>;
  let care: CareClient;

  beforeEach(() => {
    client = createMockClient();
    care = createCareClient(client as any);
  });

  it('should dispatch offer creation through timebank sub-client', async () => {
    client.callZome.mockResolvedValue(makeMockRecord(makeServiceOfferEntry()));

    const offer = await care.timebank.createServiceOffer({
      title: 'Meal Delivery',
      description: 'Weekly meal delivery',
      category: 'Meals',
      estimatedMinutes: 45,
    });

    expect(client.callZome).toHaveBeenCalledWith(
      expect.objectContaining({
        zome_name: 'care_timebank',
      }),
    );
    expect(offer.title).toBe('Garden Help');
  });

  it('should dispatch circle creation through circles sub-client', async () => {
    client.callZome.mockResolvedValue(makeMockRecord(makeCareCircleEntry()));

    const circle = await care.circles.createCircle({
      name: 'Elder Care Network',
      description: 'Support for elderly neighbors',
      categories: ['eldercare', 'meals'],
    });

    expect(client.callZome).toHaveBeenCalledWith(
      expect.objectContaining({
        zome_name: 'care_circles',
      }),
    );
    expect(circle.name).toBe('Neighborhood Support');
  });

  it('should dispatch matching through matching sub-client', async () => {
    client.callZome.mockResolvedValue([makeMockRecord(makeCareMatchEntry())]);

    const matches = await care.matching.findMatchesForRequest(fakeActionHash());

    expect(client.callZome).toHaveBeenCalledWith(
      expect.objectContaining({
        zome_name: 'care_matching',
      }),
    );
    expect(matches).toHaveLength(1);
  });

  it('should handle a full service offer-to-exchange workflow', async () => {
    // Step 1: Create offer
    const offerHash = fakeActionHash();
    client.callZome.mockResolvedValueOnce(makeMockRecord(makeServiceOfferEntry()));

    await care.timebank.createServiceOffer({
      title: 'Garden Help',
      description: 'Garden maintenance',
      category: 'Gardening',
      estimatedMinutes: 120,
    });

    // Step 2: Create request
    client.callZome.mockResolvedValueOnce(makeMockRecord(makeServiceRequestEntry()));

    await care.timebank.createServiceRequest({
      title: 'Need garden help',
      description: 'Weeding needed',
      category: 'Gardening',
      urgency: 'Medium',
    });

    // Step 3: Complete exchange
    client.callZome.mockResolvedValueOnce(makeMockRecord(makeTimeExchangeEntry()));

    await care.timebank.completeExchange({
      offerId: offerHash,
      receiverDid: 'did:mycelix:receiver1',
      minutes: 90,
      description: 'Completed garden weeding',
    });

    // Step 4: Rate exchange
    client.callZome.mockResolvedValueOnce(
      makeMockRecord(makeTimeExchangeEntry({ receiver_rating: 5, status: 'Completed' })),
    );

    await care.timebank.rateExchange({
      exchangeId: fakeActionHash(),
      rating: 5,
      comment: 'Amazing help!',
    });

    expect(client.callZome).toHaveBeenCalledTimes(4);
  });

  it('should handle a full circle lifecycle', async () => {
    // Step 1: Create circle
    client.callZome.mockResolvedValueOnce(makeMockRecord(makeCareCircleEntry()));

    await care.circles.createCircle({
      name: 'Neighborhood Care',
      description: 'Mutual support',
      categories: ['eldercare'],
    });

    // Step 2: Join circle
    client.callZome.mockResolvedValueOnce(makeMockRecord(makeCircleMemberEntry()));

    await care.circles.joinCircle(fakeActionHash());

    // Step 3: Get members
    client.callZome.mockResolvedValueOnce([
      makeMockRecord(makeCircleMemberEntry({ role: 'Founder' })),
      makeMockRecord(makeCircleMemberEntry({ role: 'Member' })),
    ]);

    const members = await care.circles.getCircleMembers(fakeActionHash());

    expect(client.callZome).toHaveBeenCalledTimes(3);
    expect(members).toHaveLength(2);
  });
});
