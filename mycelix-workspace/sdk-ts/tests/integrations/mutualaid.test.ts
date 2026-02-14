/**
 * Mutual Aid Integration Tests
 *
 * Tests for MutualAidService -- the domain-specific SDK service for
 * gift circle coordination and timebanking within the mycelix ecosystem.
 * Covers circle management, need posting/claiming/fulfilling,
 * resource contributions, timebank balances, and member summaries.
 *
 * Unlike the zome-dispatch clients (FoodClient, TransportClient, etc.),
 * MutualAidService uses an in-memory data model with MATL reputation
 * scoring via LocalBridge, so tests verify local state management
 * rather than zome call dispatch.
 */

import { describe, it, expect, beforeEach } from 'vitest';

import {
  MutualAidService,
  getMutualAidService,
  resetMutualAidService,
  type GiftCircle,
  type NeedRequest,
  type NeedCategory,
  type TimebankEntry,
  type ResourceContribution,
  type MemberSummary,
} from '../../src/integrations/mutualaid/index.js';

// ============================================================================
// Test helpers
// ============================================================================

function createTestCircle(service: MutualAidService, overrides: { id?: string; name?: string } = {}): GiftCircle {
  return service.createCircle({
    id: overrides.id ?? 'circle-001',
    name: overrides.name ?? 'Neighborhood Helpers',
    description: 'Mutual support for our block',
  });
}

function joinMembers(service: MutualAidService, circleId: string, memberIds: string[]): void {
  for (const id of memberIds) {
    service.joinCircle(circleId, id);
  }
}

function postTestNeed(
  service: MutualAidService,
  overrides: {
    circleId?: string;
    requesterId?: string;
    title?: string;
    category?: NeedCategory;
    estimatedHours?: number;
    urgent?: boolean;
  } = {},
): NeedRequest {
  return service.postNeed({
    circleId: overrides.circleId ?? 'circle-001',
    requesterId: overrides.requesterId ?? 'member-001',
    title: overrides.title ?? 'Help moving furniture',
    category: overrides.category ?? 'labor',
    estimatedHours: overrides.estimatedHours ?? 3,
    urgent: overrides.urgent,
  });
}

// ============================================================================
// Type construction tests
// ============================================================================

describe('Mutual Aid Types', () => {
  describe('NeedCategory', () => {
    it('should accept all NeedCategory variants', () => {
      const categories: NeedCategory[] = [
        'labor', 'transport', 'childcare', 'eldercare', 'food',
        'housing', 'education', 'emotional_support', 'technology',
        'medical', 'legal', 'other',
      ];
      expect(categories).toHaveLength(12);
      categories.forEach((c) => {
        expect(typeof c).toBe('string');
      });
    });
  });

  describe('GiftCircle', () => {
    it('should have all expected fields', () => {
      const circle: GiftCircle = {
        id: 'circle-001',
        name: 'Test Circle',
        description: 'A test circle',
        memberIds: ['m1', 'm2'],
        createdAt: Date.now(),
        active: true,
      };
      expect(circle.id).toBe('circle-001');
      expect(circle.memberIds).toHaveLength(2);
      expect(circle.active).toBe(true);
    });

    it('should allow optional description', () => {
      const circle: GiftCircle = {
        id: 'circle-002',
        name: 'Minimal Circle',
        memberIds: [],
        createdAt: Date.now(),
        active: true,
      };
      expect(circle.description).toBeUndefined();
    });
  });

  describe('NeedRequest', () => {
    it('should have all expected fields', () => {
      const need: NeedRequest = {
        id: 'need-001',
        circleId: 'circle-001',
        requesterId: 'member-001',
        title: 'Test Need',
        category: 'labor',
        estimatedHours: 2,
        urgent: false,
        status: 'open',
        createdAt: Date.now(),
      };
      expect(need.status).toBe('open');
      expect(need.urgent).toBe(false);
    });

    it('should accept all status variants', () => {
      const statuses: NeedRequest['status'][] = ['open', 'claimed', 'fulfilled', 'cancelled'];
      expect(statuses).toHaveLength(4);
    });
  });

  describe('TimebankEntry', () => {
    it('should have all expected fields', () => {
      const entry: TimebankEntry = {
        id: 'tb-001',
        circleId: 'circle-001',
        needId: 'need-001',
        giverId: 'member-002',
        receiverId: 'member-001',
        hours: 3,
        category: 'labor',
        timestamp: Date.now(),
        note: 'Helped with moving',
      };
      expect(entry.hours).toBe(3);
      expect(entry.note).toBe('Helped with moving');
    });

    it('should allow optional note', () => {
      const entry: TimebankEntry = {
        id: 'tb-002',
        circleId: 'circle-001',
        needId: 'need-002',
        giverId: 'member-003',
        receiverId: 'member-001',
        hours: 1,
        category: 'food',
        timestamp: Date.now(),
      };
      expect(entry.note).toBeUndefined();
    });
  });

  describe('ResourceContribution', () => {
    it('should have all expected fields', () => {
      const resource: ResourceContribution = {
        id: 'res-001',
        circleId: 'circle-001',
        contributorId: 'member-001',
        resourceType: 'tools',
        quantity: 5,
        unit: 'items',
        availableFrom: Date.now(),
        availableUntil: Date.now() + 30 * 24 * 3600_000,
        claimed: false,
      };
      expect(resource.claimed).toBe(false);
      expect(resource.quantity).toBe(5);
    });

    it('should allow optional availableUntil', () => {
      const resource: ResourceContribution = {
        id: 'res-002',
        circleId: 'circle-001',
        contributorId: 'member-002',
        resourceType: 'space',
        quantity: 1,
        unit: 'rooms',
        availableFrom: Date.now(),
        claimed: false,
      };
      expect(resource.availableUntil).toBeUndefined();
    });
  });

  describe('MemberSummary', () => {
    it('should have all expected fields', () => {
      const summary: MemberSummary = {
        memberId: 'member-001',
        hoursGiven: 10,
        hoursReceived: 5,
        needsFulfilled: 3,
        needsRequested: 2,
        reputation: 0.75,
      };
      expect(summary.hoursGiven).toBe(10);
      expect(summary.reputation).toBe(0.75);
    });
  });
});

// ============================================================================
// Singleton
// ============================================================================

describe('MutualAidService Singleton', () => {
  beforeEach(() => {
    resetMutualAidService();
  });

  it('should return same instance on repeated calls to getMutualAidService', () => {
    const a = getMutualAidService();
    const b = getMutualAidService();
    expect(a).toBe(b);
  });

  it('should return new instance after resetMutualAidService', () => {
    const a = getMutualAidService();
    resetMutualAidService();
    const b = getMutualAidService();
    expect(a).not.toBe(b);
  });

  it('should return an instance of MutualAidService', () => {
    const service = getMutualAidService();
    expect(service).toBeInstanceOf(MutualAidService);
  });
});

// ============================================================================
// MutualAidService
// ============================================================================

describe('MutualAidService', () => {
  let service: MutualAidService;

  beforeEach(() => {
    service = new MutualAidService();
  });

  // ==========================================================================
  // Circle Management
  // ==========================================================================

  describe('Circle Management', () => {
    describe('createCircle', () => {
      it('should create a circle with correct fields', () => {
        const circle = createTestCircle(service);

        expect(circle.id).toBe('circle-001');
        expect(circle.name).toBe('Neighborhood Helpers');
        expect(circle.description).toBe('Mutual support for our block');
        expect(circle.memberIds).toEqual([]);
        expect(circle.active).toBe(true);
        expect(circle.createdAt).toBeGreaterThan(0);
      });

      it('should allow creating multiple circles', () => {
        const c1 = createTestCircle(service, { id: 'c1', name: 'Circle A' });
        const c2 = createTestCircle(service, { id: 'c2', name: 'Circle B' });

        expect(c1.id).toBe('c1');
        expect(c2.id).toBe('c2');
        expect(service.getCircle('c1')).toBeDefined();
        expect(service.getCircle('c2')).toBeDefined();
      });
    });

    describe('getCircle', () => {
      it('should return the circle by ID', () => {
        createTestCircle(service);
        const circle = service.getCircle('circle-001');
        expect(circle).toBeDefined();
        expect(circle!.name).toBe('Neighborhood Helpers');
      });

      it('should return undefined for non-existent circle', () => {
        const circle = service.getCircle('non-existent');
        expect(circle).toBeUndefined();
      });
    });

    describe('joinCircle', () => {
      it('should add member to circle', () => {
        createTestCircle(service);
        const updated = service.joinCircle('circle-001', 'member-001');

        expect(updated.memberIds).toContain('member-001');
        expect(updated.memberIds).toHaveLength(1);
      });

      it('should add multiple members to circle', () => {
        createTestCircle(service);
        joinMembers(service, 'circle-001', ['m1', 'm2', 'm3']);

        const circle = service.getCircle('circle-001');
        expect(circle!.memberIds).toHaveLength(3);
      });

      it('should throw for non-existent circle', () => {
        expect(() => {
          service.joinCircle('non-existent', 'member-001');
        }).toThrow('Circle not found: non-existent');
      });

      it('should throw if member already in circle', () => {
        createTestCircle(service);
        service.joinCircle('circle-001', 'member-001');

        expect(() => {
          service.joinCircle('circle-001', 'member-001');
        }).toThrow('Already a member of circle: circle-001');
      });

      it('should initialize reputation for new members', () => {
        createTestCircle(service);
        service.joinCircle('circle-001', 'member-001');

        const summary = service.getMemberSummary('member-001');
        expect(summary.reputation).toBeGreaterThanOrEqual(0);
      });
    });
  });

  // ==========================================================================
  // Need Management
  // ==========================================================================

  describe('Need Management', () => {
    beforeEach(() => {
      createTestCircle(service);
      joinMembers(service, 'circle-001', ['member-001', 'member-002']);
    });

    describe('postNeed', () => {
      it('should create a need with correct fields', () => {
        const need = postTestNeed(service);

        expect(need.circleId).toBe('circle-001');
        expect(need.requesterId).toBe('member-001');
        expect(need.title).toBe('Help moving furniture');
        expect(need.category).toBe('labor');
        expect(need.estimatedHours).toBe(3);
        expect(need.status).toBe('open');
        expect(need.urgent).toBe(false);
        expect(need.id).toBeTruthy();
        expect(need.createdAt).toBeGreaterThan(0);
      });

      it('should create an urgent need', () => {
        const need = postTestNeed(service, { urgent: true });
        expect(need.urgent).toBe(true);
      });

      it('should accept all NeedCategory variants', () => {
        const categories: NeedCategory[] = [
          'labor', 'transport', 'childcare', 'eldercare', 'food',
          'housing', 'education', 'emotional_support', 'technology',
          'medical', 'legal', 'other',
        ];
        for (const cat of categories) {
          const need = postTestNeed(service, { category: cat, requesterId: 'member-001' });
          expect(need.category).toBe(cat);
        }
      });

      it('should throw for non-existent circle', () => {
        expect(() => {
          service.postNeed({
            circleId: 'non-existent',
            requesterId: 'member-001',
            title: 'Test',
            category: 'labor',
            estimatedHours: 1,
          });
        }).toThrow('Circle not found: non-existent');
      });

      it('should generate unique IDs for each need', () => {
        const n1 = postTestNeed(service, { title: 'Need 1' });
        const n2 = postTestNeed(service, { title: 'Need 2' });
        expect(n1.id).not.toBe(n2.id);
      });
    });

    describe('getOpenNeeds', () => {
      it('should return only open needs for a circle', () => {
        postTestNeed(service, { title: 'Open need 1' });
        postTestNeed(service, { title: 'Open need 2' });

        const openNeeds = service.getOpenNeeds('circle-001');
        expect(openNeeds).toHaveLength(2);
        expect(openNeeds.every((n) => n.status === 'open')).toBe(true);
      });

      it('should not return claimed or fulfilled needs', () => {
        const need = postTestNeed(service);
        service.claimNeed(need.id, 'member-002');

        const openNeeds = service.getOpenNeeds('circle-001');
        expect(openNeeds).toHaveLength(0);
      });

      it('should return empty array for circle with no needs', () => {
        const openNeeds = service.getOpenNeeds('circle-001');
        expect(openNeeds).toEqual([]);
      });

      it('should scope needs to the specified circle', () => {
        createTestCircle(service, { id: 'circle-002', name: 'Other Circle' });
        postTestNeed(service, { circleId: 'circle-001' });

        const openNeeds = service.getOpenNeeds('circle-002');
        expect(openNeeds).toHaveLength(0);
      });
    });

    describe('claimNeed', () => {
      it('should change need status to claimed and set claimedBy', () => {
        const need = postTestNeed(service);
        const claimed = service.claimNeed(need.id, 'member-002');

        expect(claimed.status).toBe('claimed');
        expect(claimed.claimedBy).toBe('member-002');
      });

      it('should throw for non-existent need', () => {
        expect(() => {
          service.claimNeed('non-existent', 'member-002');
        }).toThrow('Need not found: non-existent');
      });

      it('should throw if need is already claimed', () => {
        const need = postTestNeed(service);
        service.claimNeed(need.id, 'member-002');

        expect(() => {
          service.claimNeed(need.id, 'member-003');
        }).toThrow(/Need not open/);
      });
    });

    describe('fulfillNeed', () => {
      it('should change need status to fulfilled and return timebank entry', () => {
        const need = postTestNeed(service);
        service.claimNeed(need.id, 'member-002');

        const entry = service.fulfillNeed(need.id, 'member-002', 3);

        expect(entry.giverId).toBe('member-002');
        expect(entry.receiverId).toBe('member-001');
        expect(entry.hours).toBe(3);
        expect(entry.category).toBe('labor');
        expect(entry.circleId).toBe('circle-001');
        expect(entry.needId).toBe(need.id);
        expect(entry.id).toBeTruthy();
        expect(entry.timestamp).toBeGreaterThan(0);
      });

      it('should allow fulfilling an open need directly (without claiming first)', () => {
        const need = postTestNeed(service);
        const entry = service.fulfillNeed(need.id, 'member-002', 2);

        expect(entry.giverId).toBe('member-002');
        expect(entry.hours).toBe(2);
      });

      it('should throw for non-existent need', () => {
        expect(() => {
          service.fulfillNeed('non-existent', 'member-002', 1);
        }).toThrow('Need not found: non-existent');
      });

      it('should throw for already fulfilled need', () => {
        const need = postTestNeed(service);
        service.fulfillNeed(need.id, 'member-002', 3);

        expect(() => {
          service.fulfillNeed(need.id, 'member-002', 3);
        }).toThrow(/Need cannot be fulfilled/);
      });

      it('should update giver reputation after fulfillment', () => {
        const need = postTestNeed(service);
        const summaryBefore = service.getMemberSummary('member-002');
        const repBefore = summaryBefore.reputation;

        service.fulfillNeed(need.id, 'member-002', 3);

        const summaryAfter = service.getMemberSummary('member-002');
        expect(summaryAfter.reputation).toBeGreaterThanOrEqual(repBefore);
      });
    });
  });

  // ==========================================================================
  // Resource Contributions
  // ==========================================================================

  describe('Resource Contributions', () => {
    it('should create a resource contribution', () => {
      const resource = service.contributeResource({
        circleId: 'circle-001',
        contributorId: 'member-001',
        resourceType: 'tools',
        quantity: 5,
        unit: 'items',
        availableFrom: Date.now(),
      });

      expect(resource.id).toBeTruthy();
      expect(resource.claimed).toBe(false);
      expect(resource.resourceType).toBe('tools');
      expect(resource.quantity).toBe(5);
      expect(resource.unit).toBe('items');
    });

    it('should generate unique IDs for each resource', () => {
      const r1 = service.contributeResource({
        circleId: 'circle-001',
        contributorId: 'member-001',
        resourceType: 'tools',
        quantity: 1,
        unit: 'items',
        availableFrom: Date.now(),
      });
      const r2 = service.contributeResource({
        circleId: 'circle-001',
        contributorId: 'member-001',
        resourceType: 'space',
        quantity: 1,
        unit: 'rooms',
        availableFrom: Date.now(),
      });
      expect(r1.id).not.toBe(r2.id);
    });

    it('should allow optional availableUntil', () => {
      const resource = service.contributeResource({
        circleId: 'circle-001',
        contributorId: 'member-001',
        resourceType: 'food',
        quantity: 10,
        unit: 'kg',
        availableFrom: Date.now(),
        availableUntil: Date.now() + 7 * 24 * 3600_000,
      });
      expect(resource.availableUntil).toBeGreaterThan(resource.availableFrom);
    });
  });

  // ==========================================================================
  // Member Summary & Timebank
  // ==========================================================================

  describe('Member Summary & Timebank', () => {
    beforeEach(() => {
      createTestCircle(service);
      joinMembers(service, 'circle-001', ['member-001', 'member-002', 'member-003']);
    });

    describe('getMemberSummary', () => {
      it('should return zero-value summary for new member', () => {
        const summary = service.getMemberSummary('member-001');

        expect(summary.memberId).toBe('member-001');
        expect(summary.hoursGiven).toBe(0);
        expect(summary.hoursReceived).toBe(0);
        expect(summary.needsFulfilled).toBe(0);
        expect(summary.needsRequested).toBe(0);
      });

      it('should track hours given and received after fulfillment', () => {
        const need = postTestNeed(service, { requesterId: 'member-001', estimatedHours: 3 });
        service.fulfillNeed(need.id, 'member-002', 3);

        const giverSummary = service.getMemberSummary('member-002');
        expect(giverSummary.hoursGiven).toBe(3);
        expect(giverSummary.needsFulfilled).toBe(1);

        const receiverSummary = service.getMemberSummary('member-001');
        expect(receiverSummary.hoursReceived).toBe(3);
        expect(receiverSummary.needsRequested).toBe(1);
      });

      it('should accumulate across multiple fulfillments', () => {
        const need1 = postTestNeed(service, { requesterId: 'member-001', category: 'labor' });
        const need2 = postTestNeed(service, { requesterId: 'member-001', category: 'transport' });

        service.fulfillNeed(need1.id, 'member-002', 3);
        service.fulfillNeed(need2.id, 'member-002', 2);

        const summary = service.getMemberSummary('member-002');
        expect(summary.hoursGiven).toBe(5);
        expect(summary.needsFulfilled).toBe(2);
      });

      it('should return 0 reputation for member without reputation record', () => {
        const summary = service.getMemberSummary('unknown-member');
        expect(summary.reputation).toBe(0);
      });
    });

    describe('getTimebankBalance', () => {
      it('should return 0 for new member', () => {
        const balance = service.getTimebankBalance('member-001');
        expect(balance).toBe(0);
      });

      it('should return positive balance for net giver', () => {
        const need = postTestNeed(service, { requesterId: 'member-001' });
        service.fulfillNeed(need.id, 'member-002', 5);

        const balance = service.getTimebankBalance('member-002');
        expect(balance).toBe(5);
      });

      it('should return negative balance for net receiver', () => {
        const need = postTestNeed(service, { requesterId: 'member-001' });
        service.fulfillNeed(need.id, 'member-002', 5);

        const balance = service.getTimebankBalance('member-001');
        expect(balance).toBe(-5);
      });

      it('should compute net balance from giving and receiving', () => {
        // member-002 gives 5 hours to member-001
        const need1 = postTestNeed(service, { requesterId: 'member-001' });
        service.fulfillNeed(need1.id, 'member-002', 5);

        // member-001 gives 3 hours to member-002
        const need2 = postTestNeed(service, { requesterId: 'member-002' });
        service.fulfillNeed(need2.id, 'member-001', 3);

        const balance1 = service.getTimebankBalance('member-001');
        expect(balance1).toBe(-2); // gave 3, received 5

        const balance2 = service.getTimebankBalance('member-002');
        expect(balance2).toBe(2); // gave 5, received 3
      });
    });
  });

  // ==========================================================================
  // Full Lifecycle (create circle -> join -> post -> claim -> fulfill -> balance)
  // ==========================================================================

  describe('Full Lifecycle', () => {
    it('should support the complete mutual aid workflow', () => {
      // Step 1: Create a circle
      const circle = service.createCircle({
        id: 'lifecycle-circle',
        name: 'Lifecycle Test Circle',
      });
      expect(circle.active).toBe(true);

      // Step 2: Members join
      service.joinCircle('lifecycle-circle', 'alice');
      service.joinCircle('lifecycle-circle', 'bob');
      service.joinCircle('lifecycle-circle', 'carol');
      expect(service.getCircle('lifecycle-circle')!.memberIds).toHaveLength(3);

      // Step 3: Alice posts a need
      const need = service.postNeed({
        circleId: 'lifecycle-circle',
        requesterId: 'alice',
        title: 'Need help with gardening',
        category: 'labor',
        estimatedHours: 4,
      });
      expect(need.status).toBe('open');

      // Step 4: Bob claims the need
      const claimed = service.claimNeed(need.id, 'bob');
      expect(claimed.status).toBe('claimed');
      expect(claimed.claimedBy).toBe('bob');

      // Step 5: Bob fulfills the need (took 3 hours)
      const entry = service.fulfillNeed(need.id, 'bob', 3);
      expect(entry.giverId).toBe('bob');
      expect(entry.receiverId).toBe('alice');
      expect(entry.hours).toBe(3);

      // Step 6: Verify balances
      const bobBalance = service.getTimebankBalance('bob');
      expect(bobBalance).toBe(3);

      const aliceBalance = service.getTimebankBalance('alice');
      expect(aliceBalance).toBe(-3);

      // Step 7: Verify member summaries
      const bobSummary = service.getMemberSummary('bob');
      expect(bobSummary.hoursGiven).toBe(3);
      expect(bobSummary.needsFulfilled).toBe(1);
      expect(bobSummary.reputation).toBeGreaterThan(0);

      const aliceSummary = service.getMemberSummary('alice');
      expect(aliceSummary.hoursReceived).toBe(3);
      expect(aliceSummary.needsRequested).toBe(1);

      // Step 8: Carol is uninvolved
      const carolBalance = service.getTimebankBalance('carol');
      expect(carolBalance).toBe(0);
    });
  });

  // ==========================================================================
  // Error handling
  // ==========================================================================

  describe('Error handling', () => {
    it('should throw when posting need to non-existent circle', () => {
      expect(() => {
        service.postNeed({
          circleId: 'ghost',
          requesterId: 'member-001',
          title: 'Test',
          category: 'labor',
          estimatedHours: 1,
        });
      }).toThrow('Circle not found: ghost');
    });

    it('should throw when claiming non-existent need', () => {
      expect(() => {
        service.claimNeed('ghost-need', 'member-001');
      }).toThrow('Need not found: ghost-need');
    });

    it('should throw when fulfilling non-existent need', () => {
      expect(() => {
        service.fulfillNeed('ghost-need', 'member-001', 1);
      }).toThrow('Need not found: ghost-need');
    });

    it('should throw when joining non-existent circle', () => {
      expect(() => {
        service.joinCircle('ghost-circle', 'member-001');
      }).toThrow('Circle not found: ghost-circle');
    });

    it('should throw when claiming already claimed need', () => {
      createTestCircle(service);
      const need = postTestNeed(service);
      service.claimNeed(need.id, 'member-002');

      expect(() => {
        service.claimNeed(need.id, 'member-003');
      }).toThrow(/Need not open/);
    });

    it('should throw when fulfilling already fulfilled need', () => {
      createTestCircle(service);
      const need = postTestNeed(service);
      service.fulfillNeed(need.id, 'member-002', 2);

      expect(() => {
        service.fulfillNeed(need.id, 'member-002', 2);
      }).toThrow(/Need cannot be fulfilled/);
    });

    it('should throw when joining circle twice', () => {
      createTestCircle(service);
      service.joinCircle('circle-001', 'member-001');

      expect(() => {
        service.joinCircle('circle-001', 'member-001');
      }).toThrow('Already a member of circle: circle-001');
    });
  });
});
