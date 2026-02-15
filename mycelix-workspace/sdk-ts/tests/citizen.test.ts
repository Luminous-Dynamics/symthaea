/**
 * Citizen Module Tests
 *
 * Tests for:
 * - CitizenDashboardService (API fallback mode)
 * - Utility functions (calculateTrustScore, canAppeal, getDaysUntilDeadline)
 * - ProactiveOutreachEngine
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  CitizenDashboardService,
  calculateTrustScore,
  canAppeal,
  getDaysUntilDeadline,
  createCitizenService,
  ProactiveOutreachEngine,
  type CitizenServiceConfig,
  type RecentDecision,
  type LifeEvent,
  type OutreachMessage,
} from '../src/citizen/index.js';

// =============================================================================
// Utility Function Tests
// =============================================================================

describe('Citizen Utilities', () => {
  describe('calculateTrustScore', () => {
    it('should calculate weighted average of components', () => {
      const score = calculateTrustScore([
        { score: 0.9, weight: 0.4 },
        { score: 0.8, weight: 0.3 },
        { score: 0.7, weight: 0.3 },
      ]);

      // (0.9*0.4 + 0.8*0.3 + 0.7*0.3) / (0.4+0.3+0.3) = 0.81
      expect(score).toBeCloseTo(0.81, 2);
    });

    it('should return 0 for empty components', () => {
      expect(calculateTrustScore([])).toBe(0);
    });

    it('should handle single component', () => {
      const score = calculateTrustScore([{ score: 0.95, weight: 1.0 }]);
      expect(score).toBeCloseTo(0.95, 2);
    });

    it('should handle unequal weights correctly', () => {
      const score = calculateTrustScore([
        { score: 1.0, weight: 0.0 },
        { score: 0.5, weight: 1.0 },
      ]);
      // (1.0*0 + 0.5*1.0) / (0+1.0) = 0.5
      expect(score).toBeCloseTo(0.5, 2);
    });
  });

  describe('canAppeal', () => {
    it('should return true when deadline is in the future', () => {
      const decision: RecentDecision = {
        id: 'dec-1',
        type: 'proposal',
        title: 'Test Decision',
        decidedAt: new Date(),
        outcome: 'denied',
        explanation: 'Insufficient documentation',
        appealDeadline: new Date(Date.now() + 86400000), // tomorrow
        algorithmUsed: 'standard',
        inputData: {},
        outputScore: 0.4,
      };

      expect(canAppeal(decision)).toBe(true);
    });

    it('should return false when deadline has passed', () => {
      const decision: RecentDecision = {
        id: 'dec-2',
        type: 'application',
        title: 'Expired',
        decidedAt: new Date(),
        outcome: 'denied',
        explanation: 'Expired',
        appealDeadline: new Date(Date.now() - 86400000), // yesterday
        algorithmUsed: 'standard',
        inputData: {},
        outputScore: 0.3,
      };

      expect(canAppeal(decision)).toBe(false);
    });

    it('should return false when no appeal deadline', () => {
      const decision: RecentDecision = {
        id: 'dec-3',
        type: 'case',
        title: 'No deadline',
        decidedAt: new Date(),
        outcome: 'approved',
        explanation: 'Approved',
        algorithmUsed: 'standard',
        inputData: {},
        outputScore: 0.9,
      };

      expect(canAppeal(decision)).toBe(false);
    });
  });

  describe('getDaysUntilDeadline', () => {
    it('should return positive number for future deadline', () => {
      const twoDaysFromNow = new Date(Date.now() + 2 * 86400000);
      const days = getDaysUntilDeadline(twoDaysFromNow);
      expect(days).toBe(2);
    });

    it('should return negative number for past deadline', () => {
      const twoDaysAgo = new Date(Date.now() - 2 * 86400000);
      const days = getDaysUntilDeadline(twoDaysAgo);
      expect(days).toBe(-2);
    });

    it('should return 1 for deadline tomorrow', () => {
      const tomorrow = new Date(Date.now() + 86400000);
      expect(getDaysUntilDeadline(tomorrow)).toBe(1);
    });
  });
});

// =============================================================================
// CitizenDashboardService Tests
// =============================================================================

describe('CitizenDashboardService', () => {
  describe('constructor', () => {
    it('should create service with agentId', () => {
      const service = createCitizenService({ agentId: 'agent-123' });
      expect(service).toBeInstanceOf(CitizenDashboardService);
    });
  });

  describe('API fallback mode', () => {
    it('should throw when no client or API URL configured', async () => {
      const service = new CitizenDashboardService({ agentId: 'agent-123' });

      await expect(service.getProfile()).rejects.toThrow(
        'No Mycelix client or API URL configured'
      );
    });

    it('should use API URL when provided', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            id: 'agent-123',
            did: 'did:mycelix:abc',
            name: 'Test User',
            trustScore: 0.85,
            memberSince: new Date().toISOString(),
            verificationLevel: 'verified',
            credentials: [],
            delegations: [],
          }),
      });

      vi.stubGlobal('fetch', mockFetch);

      const service = new CitizenDashboardService({
        agentId: 'agent-123',
        apiBaseUrl: 'https://api.example.com',
      });

      const profile = await service.getProfile();
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/citizen/profile',
        expect.objectContaining({
          headers: { 'X-Agent-Id': 'agent-123' },
        })
      );

      vi.unstubAllGlobals();
    });

    it('should throw on API error response', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
      });

      vi.stubGlobal('fetch', mockFetch);

      const service = new CitizenDashboardService({
        agentId: 'agent-123',
        apiBaseUrl: 'https://api.example.com',
      });

      await expect(service.getProfile()).rejects.toThrow('API error: 500');

      vi.unstubAllGlobals();
    });
  });

  describe('Holochain client mode', () => {
    it('should call identity zome for getProfile', async () => {
      const mockClient = {
        callZome: vi.fn().mockResolvedValue({
          agent_id: 'agent-123',
          did: 'did:mycelix:abc',
          display_name: 'Test',
          trust_score: 0.9,
          created_at: Date.now(),
          verification_level: 'verified',
          credentials: [],
          delegations: [],
        }),
      };

      const service = new CitizenDashboardService({
        client: mockClient as any,
        agentId: 'agent-123',
      });

      const profile = await service.getProfile();
      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'identity',
          zome_name: 'identity',
          fn_name: 'get_profile',
        })
      );
      expect(profile.did).toBe('did:mycelix:abc');
      expect(profile.name).toBe('Test');
    });

    it('should call multiple zomes for getPendingDecisions', async () => {
      const mockClient = {
        callZome: vi.fn().mockResolvedValue([]),
      };

      const service = new CitizenDashboardService({
        client: mockClient as any,
        agentId: 'agent-123',
      });

      await service.getPendingDecisions();

      // Should call governance proposals, applications, and justice cases
      expect(mockClient.callZome).toHaveBeenCalledTimes(3);
    });

    it('should aggregate dashboard data from all sources', async () => {
      const mockClient = {
        callZome: vi.fn().mockImplementation(({ fn_name }: { fn_name: string }) => {
          switch (fn_name) {
            case 'get_profile':
              return {
                agent_id: 'agent-123',
                did: 'did:mycelix:abc',
                display_name: 'Test',
                trust_score: 0.9,
                created_at: Date.now(),
                verification_level: 'basic',
                credentials: [],
                delegations: [],
              };
            case 'get_trust_breakdown':
              return {
                composite_score: 0.85,
                pogq: 0.9,
                consistency: 0.8,
                reputation: 0.85,
                last_verified: Date.now(),
                history: [],
              };
            case 'get_balance':
              return {
                balance: 500,
                monthly_allocation: 100,
                last_allocation: Date.now(),
                expires_at: Date.now() + 86400000,
                spent_this_month: [],
              };
            default:
              return [];
          }
        }),
      };

      const service = new CitizenDashboardService({
        client: mockClient as any,
        agentId: 'agent-123',
      });

      const dashboard = await service.getDashboardData();
      expect(dashboard.profile.did).toBe('did:mycelix:abc');
      expect(dashboard.trustBreakdown.overall).toBe(0.85);
      expect(dashboard.civicCredits.balance).toBe(500);
    });
  });
});

// =============================================================================
// ProactiveOutreachEngine Tests
// =============================================================================

describe('ProactiveOutreachEngine', () => {
  it('should handle life event and send outreach for eligible programs', async () => {
    const sentMessages: OutreachMessage[] = [];
    const mockClient = {
      callZome: vi.fn().mockImplementation(({ fn_name }: { fn_name: string }) => {
        if (fn_name === 'check') {
          return {
            program: 'child_tax_credit',
            eligible: true,
            matchScore: 0.95,
            requirements: [{ name: 'has_child', met: true }],
            estimatedValue: 3000,
          };
        }
        return null; // log_proactive_contact
      }),
    };

    const mockNotification = {
      send: vi.fn().mockImplementation(async (msg: OutreachMessage) => {
        sentMessages.push(msg);
      }),
    };

    const engine = new ProactiveOutreachEngine(mockClient as any, mockNotification);

    const event: LifeEvent = {
      type: 'birth',
      citizenId: 'citizen-1',
      timestamp: new Date(),
      data: { childName: 'Baby' },
    };

    await engine.handleLifeEvent(event);

    expect(mockNotification.send).toHaveBeenCalled();
    const msg = sentMessages[0];
    expect(msg.citizenId).toBe('citizen-1');
    expect(msg.channel).toBe('all');
    expect(msg.subject).toContain('eligible');
  });

  it('should not send outreach when no programs are eligible', async () => {
    const mockClient = {
      callZome: vi.fn().mockResolvedValue({
        program: 'some_program',
        eligible: false,
        matchScore: 0.2,
        requirements: [{ name: 'income_check', met: false }],
      }),
    };

    const mockNotification = {
      send: vi.fn(),
    };

    const engine = new ProactiveOutreachEngine(mockClient as any, mockNotification);

    await engine.handleLifeEvent({
      type: 'address_change',
      citizenId: 'citizen-2',
      timestamp: new Date(),
      data: {},
    });

    expect(mockNotification.send).not.toHaveBeenCalled();
  });
});
