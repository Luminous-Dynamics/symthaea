// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Care Client Tests
 *
 * Verifies zome call arguments, response mapping, and operations for
 * CirclesClient, TimebankClient, and MatchingClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { CirclesClient } from '../circles';
import { TimebankClient } from '../timebank';
import { MatchingClient } from '../matching';
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

const CIRCLE_ENTRY = {
  name: 'Elder Care Circle',
  description: 'Support circle for elderly neighbors',
  founder_did: 'did:mycelix:alice',
  max_members: 12,
  member_count: 5,
  area: 'Downtown',
  categories: ['Elder Care', 'Companionship'],
  open: true,
  active: true,
  created_at: 1708200000,
  updated_at: 1708200000,
};

const CIRCLE_MEMBER_ENTRY = {
  circle_id: new Uint8Array(32),
  member_did: 'did:mycelix:alice',
  member_pub_key: new Uint8Array(32),
  role: 'Facilitator',
  joined_at: 1708200000,
  active: true,
};

const SERVICE_OFFER_ENTRY = {
  provider_did: 'did:mycelix:alice',
  provider_pub_key: new Uint8Array(32),
  title: 'Home meal preparation',
  description: 'Weekly meal prep for those who need it',
  category: 'Meals',
  estimated_minutes: 180,
  tags: ['cooking', 'nutrition'],
  location: 'Downtown',
  available: true,
  average_rating: 4.8,
  completed_count: 12,
  created_at: 1708200000,
  updated_at: 1708200000,
};

const SERVICE_REQUEST_ENTRY = {
  requester_did: 'did:mycelix:bob',
  requester_pub_key: new Uint8Array(32),
  title: 'Need help with garden maintenance',
  description: 'Weekly garden care while recovering from surgery',
  category: 'Gardening',
  urgency: 'Medium',
  preferred_time_start: null,
  preferred_time_end: null,
  location: 'Suburb',
  status: 'Open',
  created_at: 1708300000,
  updated_at: 1708300000,
};

const TIME_EXCHANGE_ENTRY = {
  offer_id: new Uint8Array(32),
  request_id: new Uint8Array(32),
  provider_did: 'did:mycelix:alice',
  receiver_did: 'did:mycelix:bob',
  minutes: 180,
  description: 'Prepared meals for the week',
  provider_rating: null,
  receiver_rating: null,
  status: 'Completed',
  completed_at: 1708400000,
  created_at: 1708400000,
};

const MATCH_ENTRY = {
  request_id: new Uint8Array(32),
  offer_id: new Uint8Array(32),
  requester_did: 'did:mycelix:bob',
  provider_did: 'did:mycelix:alice',
  confidence: 0.92,
  reason: 'Skills match and proximity',
  status: 'Suggested',
  created_at: 1708300000,
};

// ============================================================================
// CIRCLES CLIENT TESTS
// ============================================================================

describe('CirclesClient', () => {
  let client: CirclesClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new CirclesClient(mockAppClient);
  });

  describe('initialization', () => {
    it('should use commons role and care_circles zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(CIRCLE_ENTRY)
      );

      await client.createCircle({
        name: 'Test Circle',
        description: 'Test',
        maxMembers: 10,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'commons',
          zome_name: 'care_circles',
        })
      );
    });
  });

  describe('createCircle', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(CIRCLE_ENTRY)
      );

      const result = await client.createCircle({
        name: 'Elder Care Circle',
        description: 'Support circle for elderly neighbors',
        maxMembers: 12,
        area: 'Downtown',
        categories: ['Elder Care'],
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_circle',
          payload: expect.objectContaining({
            max_members: 12,
            area: 'Downtown',
          }),
        })
      );
      expect(result.name).toBe('Elder Care Circle');
      expect(result.maxMembers).toBe(12);
      expect(result.memberCount).toBe(5);
      expect(result.open).toBe(true);
    });
  });

  describe('getCircle', () => {
    it('should return circle for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(CIRCLE_ENTRY)
      );

      const result = await client.getCircle(new Uint8Array(32));
      expect(result).not.toBeNull();
      expect(result!.name).toBe('Elder Care Circle');
    });

    it('should return null for missing circle', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getCircle(new Uint8Array(32));
      expect(result).toBeNull();
    });
  });

  describe('joinCircle', () => {
    it('should pass circle ID and map member response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(CIRCLE_MEMBER_ENTRY)
      );

      const result = await client.joinCircle(new Uint8Array(32));
      expect(result.memberDid).toBe('did:mycelix:alice');
      expect(result.role).toBe('Facilitator');
      expect(result.active).toBe(true);
    });
  });

  describe('leaveCircle', () => {
    it('should call zome and return void', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(undefined);

      await client.leaveCircle(new Uint8Array(32));

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'leave_circle',
        })
      );
    });
  });

  describe('getMyCircles', () => {
    it('should return array of circles', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(CIRCLE_ENTRY),
        mockRecord({ ...CIRCLE_ENTRY, name: 'Child Care Circle' }),
      ]);

      const result = await client.getMyCircles();
      expect(result).toHaveLength(2);
      expect(result[1].name).toBe('Child Care Circle');
    });
  });

  describe('getCircleMembers', () => {
    it('should return array of members', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(CIRCLE_MEMBER_ENTRY),
        mockRecord({ ...CIRCLE_MEMBER_ENTRY, member_did: 'did:mycelix:bob', role: 'Member' }),
      ]);

      const result = await client.getCircleMembers(new Uint8Array(32));
      expect(result).toHaveLength(2);
    });
  });

  describe('listOpenCircles', () => {
    it('should return array of open circles', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(CIRCLE_ENTRY),
      ]);

      const result = await client.listOpenCircles();
      expect(result).toHaveLength(1);
    });
  });

  describe('error handling', () => {
    it('should propagate zome errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Circle is full')
      );

      await expect(client.joinCircle(new Uint8Array(32))).rejects.toThrow();
    });
  });
});

// ============================================================================
// TIMEBANK CLIENT TESTS
// ============================================================================

describe('TimebankClient', () => {
  let client: TimebankClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new TimebankClient(mockAppClient);
  });

  describe('initialization', () => {
    it('should use commons role and care_timebank zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(SERVICE_OFFER_ENTRY)
      );

      await client.createServiceOffer({
        title: 'Test',
        description: 'Test',
        category: 'General',
        estimatedMinutes: 60,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'commons',
          zome_name: 'care_timebank',
        })
      );
    });
  });

  describe('createServiceOffer', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(SERVICE_OFFER_ENTRY)
      );

      const result = await client.createServiceOffer({
        title: 'Home meal preparation',
        description: 'Weekly meal prep',
        category: 'Meals',
        estimatedMinutes: 180,
        tags: ['cooking', 'nutrition'],
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_service_offer',
          payload: expect.objectContaining({
            estimated_minutes: 180,
            tags: ['cooking', 'nutrition'],
          }),
        })
      );
      expect(result.title).toBe('Home meal preparation');
      expect(result.estimatedMinutes).toBe(180);
      expect(result.available).toBe(true);
    });
  });

  describe('getOffersByCategory', () => {
    it('should return array of offers', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(SERVICE_OFFER_ENTRY),
      ]);

      const result = await client.getOffersByCategory('Meals');
      expect(result).toHaveLength(1);
    });
  });

  describe('getMyOffers', () => {
    it('should return array of my offers', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(SERVICE_OFFER_ENTRY),
      ]);

      const result = await client.getMyOffers();
      expect(result).toHaveLength(1);
    });
  });

  describe('updateOfferAvailability', () => {
    it('should pass offer ID and availability flag', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...SERVICE_OFFER_ENTRY, available: false })
      );

      const result = await client.updateOfferAvailability(new Uint8Array(32), false);
      expect(result.available).toBe(false);
    });
  });

  describe('createServiceRequest', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(SERVICE_REQUEST_ENTRY)
      );

      const result = await client.createServiceRequest({
        title: 'Need help with garden',
        description: 'Weekly garden care',
        category: 'Gardening',
        urgency: 'Medium',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_service_request',
          payload: expect.objectContaining({
            urgency: 'Medium',
          }),
        })
      );
      expect(result.title).toBe('Need help with garden maintenance');
      expect(result.urgency).toBe('Medium');
      expect(result.status).toBe('Open');
    });
  });

  describe('getOpenRequests', () => {
    it('should return array of open requests', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(SERVICE_REQUEST_ENTRY),
      ]);

      const result = await client.getOpenRequests('Gardening');
      expect(result).toHaveLength(1);
    });
  });

  describe('completeExchange', () => {
    it('should pass snake_case payload and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TIME_EXCHANGE_ENTRY)
      );

      const result = await client.completeExchange({
        offerId: new Uint8Array(32),
        requestId: new Uint8Array(32),
        receiverDid: 'did:mycelix:bob',
        minutes: 180,
        description: 'Prepared meals for the week',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'complete_exchange',
          payload: expect.objectContaining({
            offer_id: new Uint8Array(32),
            request_id: new Uint8Array(32),
            receiver_did: 'did:mycelix:bob',
            minutes: 180,
          }),
        })
      );
      expect(result.minutes).toBe(180);
      expect(result.status).toBe('Completed');
    });
  });

  describe('rateExchange', () => {
    it('should pass exchange ID, rating, and comment', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...TIME_EXCHANGE_ENTRY, provider_rating: 5 })
      );

      const result = await client.rateExchange({
        exchangeId: new Uint8Array(32),
        rating: 5,
        comment: 'Excellent service',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'rate_exchange',
          payload: expect.objectContaining({
            exchange_id: new Uint8Array(32),
            rating: 5,
            comment: 'Excellent service',
          }),
        })
      );
      expect(result.providerRating).toBe(5);
    });
  });

  describe('getMyBalance', () => {
    it('should return balance directly (no extractEntry)', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        did: 'did:mycelix:alice',
        earned: 900,
        spent: 480,
        balance: 420,
        total_exchanges: 5,
        average_provider_rating: 4.8,
        average_receiver_rating: 4.5,
      });

      const result = await client.getMyBalance();
      expect(result.earned).toBe(900);
      expect(result.spent).toBe(480);
      expect(result.balance).toBe(420);
      expect(result.totalExchanges).toBe(5);
    });
  });

  describe('error handling', () => {
    it('should propagate zome errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Insufficient balance')
      );

      await expect(
        client.completeExchange({
          offerId: new Uint8Array(32),
          requestId: new Uint8Array(32),
          receiverDid: 'did:mycelix:bob',
          minutes: 180,
          description: 'Test',
        })
      ).rejects.toThrow();
    });
  });
});

// ============================================================================
// MATCHING CLIENT TESTS
// ============================================================================

describe('MatchingClient', () => {
  let client: MatchingClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new MatchingClient(mockAppClient);
  });

  describe('initialization', () => {
    it('should use commons role and care_matching zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MATCH_ENTRY)
      );

      await client.suggestMatch(new Uint8Array(32), new Uint8Array(32), 'Skills match');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'commons',
          zome_name: 'care_matching',
        })
      );
    });
  });

  describe('findMatchesForRequest', () => {
    it('should return array of matches', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(MATCH_ENTRY),
        mockRecord({ ...MATCH_ENTRY, confidence: 0.85 }),
      ]);

      const result = await client.findMatchesForRequest(new Uint8Array(32));
      expect(result).toHaveLength(2);
      expect(result[0].confidence).toBe(0.92);
    });
  });

  describe('suggestMatch', () => {
    it('should pass request ID, offer ID, and reason', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MATCH_ENTRY)
      );

      const result = await client.suggestMatch(
        new Uint8Array(32),
        new Uint8Array(32),
        'Skills match and proximity'
      );

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'suggest_match',
          payload: expect.objectContaining({
            request_id: new Uint8Array(32),
            offer_id: new Uint8Array(32),
            reason: 'Skills match and proximity',
          }),
        })
      );
      expect(result.status).toBe('Suggested');
      expect(result.confidence).toBe(0.92);
    });
  });

  describe('acceptMatch', () => {
    it('should pass match ID and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...MATCH_ENTRY, status: 'Accepted' })
      );

      const result = await client.acceptMatch(new Uint8Array(32));
      expect(result.status).toBe('Accepted');
    });
  });

  describe('declineMatch', () => {
    it('should pass match ID and map response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...MATCH_ENTRY, status: 'Declined' })
      );

      const result = await client.declineMatch(new Uint8Array(32));
      expect(result.status).toBe('Declined');
    });
  });

  describe('getMatchesForOffer', () => {
    it('should return array of matches', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(MATCH_ENTRY),
      ]);

      const result = await client.getMatchesForOffer(new Uint8Array(32));
      expect(result).toHaveLength(1);
    });
  });

  describe('error handling', () => {
    it('should propagate zome errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Request not found')
      );

      await expect(
        client.findMatchesForRequest(new Uint8Array(32))
      ).rejects.toThrow();
    });
  });
});
