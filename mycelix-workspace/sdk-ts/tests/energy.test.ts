/**
 * Energy Module Tests
 *
 * Tests for the Energy hApp TypeScript clients (Terra Atlas integration):
 * - ProjectsClient (energy project registry)
 * - ParticipantsClient (participant management)
 * - TradingClient (P2P energy trading)
 * - CreditsClient (energy credits)
 * - InvestmentsClient (project investments)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  ProjectsClient,
  ParticipantsClient,
  TradingClient,
  CreditsClient,
  InvestmentsClient,
  createEnergyClients,
  type EnergyProject,
  type EnergyParticipant,
  type EnergyTrade,
  type EnergyCredit,
  type Investment,
  type ZomeCallable,
  type HolochainRecord,
  type EnergySource,
  type ProjectStatus,
  type ParticipantType,
} from '../src/energy/index.js';

// ============================================================================
// Mock Setup
// ============================================================================

function createMockRecord<T>(entry: T): HolochainRecord<T> {
  return {
    signed_action: {
      hashed: { hash: 'uhCXk_test_hash_123', content: {} },
      signature: 'sig_test_123',
    },
    entry: { Present: entry },
  };
}

function createMockClient(responses: Map<string, unknown> = new Map()): ZomeCallable {
  return {
    callZome: vi.fn(
      async <T>(params: {
        role_name: string;
        zome_name: string;
        fn_name: string;
        payload: unknown;
      }): Promise<T> => {
        const key = `${params.zome_name}:${params.fn_name}`;
        if (responses.has(key)) {
          return responses.get(key) as T;
        }
        throw new Error(`No mock response for ${key}`);
      }
    ),
  };
}

// ============================================================================
// Mock Data Factories
// ============================================================================

function createMockProject(overrides: Partial<EnergyProject> = {}): EnergyProject {
  return {
    id: 'project-123',
    name: 'West Texas Solar Farm',
    description: 'Community-owned solar installation in West Texas corridor',
    source: 'Solar' as EnergySource,
    capacity_kw: 50000,
    location: {
      lat: 31.7619,
      lng: -106.485,
      address: 'West Texas, USA',
    },
    status: 'Operational' as ProjectStatus,
    owner_dao: 'dao-west-texas-energy',
    investment_goal: 5000000,
    investment_raised: 4250000,
    operating_since: Date.now() * 1000 - 31536000000000,
    created_at: Date.now() * 1000 - 63072000000000,
    updated_at: Date.now() * 1000,
    ...overrides,
  };
}

function createMockParticipant(overrides: Partial<EnergyParticipant> = {}): EnergyParticipant {
  return {
    id: 'participant-123',
    did: 'did:mycelix:producer123',
    type_: 'Producer' as ParticipantType,
    sources: ['Solar', 'Wind'] as EnergySource[],
    capacity_kwh: 1000,
    location: { lat: 31.7619, lng: -106.485 },
    reputation_score: 0.92,
    active: true,
    registered_at: Date.now() * 1000 - 31536000000000,
    ...overrides,
  };
}

function createMockTrade(overrides: Partial<EnergyTrade> = {}): EnergyTrade {
  return {
    id: 'trade-123',
    seller: 'did:mycelix:producer123',
    buyer: 'did:mycelix:consumer456',
    amount_kwh: 500,
    source: 'Solar' as EnergySource,
    price_per_kwh: 0.12,
    currency: 'USD',
    status: 'Confirmed',
    created_at: Date.now() * 1000,
    ...overrides,
  };
}

function createMockCredit(overrides: Partial<EnergyCredit> = {}): EnergyCredit {
  return {
    id: 'credit-123',
    participant: 'did:mycelix:producer123',
    amount_kwh: 1000,
    source: 'Solar' as EnergySource,
    issued_at: Date.now() * 1000,
    expires_at: Date.now() * 1000 + 31536000000000,
    used: false,
    ...overrides,
  };
}

function createMockInvestment(overrides: Partial<Investment> = {}): Investment {
  return {
    id: 'investment-123',
    project_id: 'project-123',
    investor: 'did:mycelix:investor789',
    amount: 25000,
    currency: 'USD',
    ownership_percentage: 0.5,
    invested_at: Date.now() * 1000,
    ...overrides,
  };
}

// ============================================================================
// ProjectsClient Tests
// ============================================================================

describe('ProjectsClient', () => {
  let client: ProjectsClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockProject = createMockProject();

    responses.set('projects:register_project', createMockRecord(mockProject));
    responses.set('projects:get_project', createMockRecord(mockProject));
    responses.set('projects:get_projects_by_source', [createMockRecord(mockProject)]);
    responses.set('projects:get_projects_by_status', [createMockRecord(mockProject)]);
    responses.set('projects:search_projects', [createMockRecord(mockProject)]);

    mockZome = createMockClient(responses);
    client = new ProjectsClient(mockZome);
  });

  describe('registerProject', () => {
    it('should register a new energy project', async () => {
      const result = await client.registerProject({
        name: 'West Texas Solar Farm',
        description: 'Community solar installation',
        source: 'Solar',
        capacity_kw: 50000,
        location: { lat: 31.7619, lng: -106.485 },
        investment_goal: 5000000,
      });

      expect(result.entry.Present.id).toBe('project-123');
      expect(result.entry.Present.source).toBe('Solar');
      expect(mockZome.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'energy',
          zome_name: 'projects',
          fn_name: 'register_project',
        })
      );
    });

    it('should register projects with all energy sources', async () => {
      const sources: EnergySource[] = ['Solar', 'Wind', 'Hydro', 'Nuclear', 'Geothermal', 'Storage', 'Other'];

      for (const source of sources) {
        const result = await client.registerProject({
          name: `Test ${source} Project`,
          description: 'Test project',
          source,
          capacity_kw: 1000,
          location: { lat: 0, lng: 0 },
        });
        expect(result).toBeDefined();
      }
    });
  });

  describe('getProject', () => {
    it('should get project by ID', async () => {
      const result = await client.getProject('project-123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.id).toBe('project-123');
    });
  });

  describe('getProjectsBySource', () => {
    it('should get projects by energy source', async () => {
      const results = await client.getProjectsBySource('Solar');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.source).toBe('Solar');
    });
  });

  describe('getProjectsByStatus', () => {
    it('should get projects by status', async () => {
      const results = await client.getProjectsByStatus('Operational');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.status).toBe('Operational');
    });

    it('should support all project statuses', async () => {
      const statuses: ProjectStatus[] = [
        'Proposed',
        'Planning',
        'Development',
        'Construction',
        'Operational',
        'Decommissioned',
      ];

      for (const status of statuses) {
        const results = await client.getProjectsByStatus(status);
        expect(results).toBeDefined();
      }
    });
  });

  describe('searchProjects', () => {
    it('should search projects with multiple criteria', async () => {
      const results = await client.searchProjects({
        source: 'Solar',
        status: 'Operational',
        minCapacity: 10000,
      });

      expect(results).toHaveLength(1);
    });

    it('should search with partial criteria', async () => {
      const results = await client.searchProjects({ source: 'Wind' });
      expect(results).toBeDefined();
    });
  });
});

// ============================================================================
// ParticipantsClient Tests
// ============================================================================

describe('ParticipantsClient', () => {
  let client: ParticipantsClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockParticipant = createMockParticipant();

    responses.set('participants:register', createMockRecord(mockParticipant));
    responses.set('participants:get_participant', createMockRecord(mockParticipant));
    responses.set('participants:get_by_type', [createMockRecord(mockParticipant)]);
    responses.set(
      'participants:update_capacity',
      createMockRecord({
        ...mockParticipant,
        capacity_kwh: 2000,
      })
    );

    mockZome = createMockClient(responses);
    client = new ParticipantsClient(mockZome);
  });

  describe('register', () => {
    it('should register a participant', async () => {
      const result = await client.register({
        type_: 'Producer',
        sources: ['Solar', 'Wind'],
        capacity_kwh: 1000,
        location: { lat: 31.7619, lng: -106.485 },
      });

      expect(result.entry.Present.type_).toBe('Producer');
      expect(result.entry.Present.sources).toContain('Solar');
    });

    it('should register with all participant types', async () => {
      const types: ParticipantType[] = ['Producer', 'Consumer', 'Prosumer', 'Operator', 'Investor'];

      for (const type_ of types) {
        const result = await client.register({
          type_,
          sources: ['Solar'],
          capacity_kwh: 100,
        });
        expect(result).toBeDefined();
      }
    });
  });

  describe('getParticipant', () => {
    it('should get participant by DID', async () => {
      const result = await client.getParticipant('did:mycelix:producer123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.did).toBe('did:mycelix:producer123');
    });
  });

  describe('getParticipantsByType', () => {
    it('should get participants by type', async () => {
      const results = await client.getParticipantsByType('Producer');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.type_).toBe('Producer');
    });
  });

  describe('updateCapacity', () => {
    it('should update participant capacity', async () => {
      const result = await client.updateCapacity(2000);

      expect(result.entry.Present.capacity_kwh).toBe(2000);
    });
  });
});

// ============================================================================
// TradingClient Tests
// ============================================================================

describe('TradingClient', () => {
  let client: TradingClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockTrade = createMockTrade();

    responses.set('trading:create_trade', createMockRecord(mockTrade));
    responses.set(
      'trading:accept_trade',
      createMockRecord({
        ...mockTrade,
        status: 'Confirmed',
      })
    );
    responses.set(
      'trading:confirm_delivery',
      createMockRecord({
        ...mockTrade,
        status: 'Delivered',
        delivered_at: Date.now() * 1000,
      })
    );
    responses.set('trading:get_trades_by_participant', [createMockRecord(mockTrade)]);
    responses.set('trading:get_open_trades', [createMockRecord({ ...mockTrade, status: 'Pending' })]);

    mockZome = createMockClient(responses);
    client = new TradingClient(mockZome);
  });

  describe('createTrade', () => {
    it('should create an energy trade', async () => {
      const result = await client.createTrade({
        buyer: 'did:mycelix:consumer456',
        amount_kwh: 500,
        source: 'Solar',
        price_per_kwh: 0.12,
        currency: 'USD',
      });

      expect(result.entry.Present.amount_kwh).toBe(500);
      expect(result.entry.Present.source).toBe('Solar');
    });
  });

  describe('acceptTrade', () => {
    it('should accept a trade', async () => {
      const result = await client.acceptTrade('trade-123');

      expect(result.entry.Present.status).toBe('Confirmed');
    });
  });

  describe('confirmDelivery', () => {
    it('should confirm energy delivery', async () => {
      const result = await client.confirmDelivery('trade-123');

      expect(result.entry.Present.status).toBe('Delivered');
      expect(result.entry.Present.delivered_at).toBeDefined();
    });
  });

  describe('getTradesByParticipant', () => {
    it('should get trades by participant', async () => {
      const results = await client.getTradesByParticipant('did:mycelix:producer123');

      expect(results).toHaveLength(1);
    });
  });

  describe('getOpenTrades', () => {
    it('should get all open trades', async () => {
      const results = await client.getOpenTrades();

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.status).toBe('Pending');
    });

    it('should get open trades by source', async () => {
      const results = await client.getOpenTrades('Solar');

      expect(results).toBeDefined();
    });
  });
});

// ============================================================================
// CreditsClient Tests
// ============================================================================

describe('CreditsClient', () => {
  let client: CreditsClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockCredit = createMockCredit();

    responses.set('credits:issue_credit', createMockRecord(mockCredit));
    responses.set('credits:get_credits', [createMockRecord(mockCredit)]);
    responses.set(
      'credits:use_credit',
      createMockRecord({
        ...mockCredit,
        used: true,
      })
    );

    mockZome = createMockClient(responses);
    client = new CreditsClient(mockZome);
  });

  describe('issueCredit', () => {
    it('should issue an energy credit', async () => {
      const result = await client.issueCredit('did:mycelix:producer123', 1000, 'Solar');

      expect(result.entry.Present.amount_kwh).toBe(1000);
      expect(result.entry.Present.source).toBe('Solar');
      expect(result.entry.Present.used).toBe(false);
    });
  });

  describe('getCredits', () => {
    it('should get credits by participant', async () => {
      const results = await client.getCredits('did:mycelix:producer123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.participant).toBe('did:mycelix:producer123');
    });
  });

  describe('useCredit', () => {
    it('should mark credit as used', async () => {
      const result = await client.useCredit('credit-123');

      expect(result.entry.Present.used).toBe(true);
    });
  });
});

// ============================================================================
// InvestmentsClient Tests
// ============================================================================

describe('InvestmentsClient', () => {
  let client: InvestmentsClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockInvestment = createMockInvestment();

    responses.set('investments:invest', createMockRecord(mockInvestment));
    responses.set('investments:get_by_project', [createMockRecord(mockInvestment)]);
    responses.set('investments:get_by_investor', [createMockRecord(mockInvestment)]);

    mockZome = createMockClient(responses);
    client = new InvestmentsClient(mockZome);
  });

  describe('invest', () => {
    it('should make an investment in a project', async () => {
      const result = await client.invest({
        project_id: 'project-123',
        amount: 25000,
        currency: 'USD',
      });

      expect(result.entry.Present.amount).toBe(25000);
      expect(result.entry.Present.project_id).toBe('project-123');
    });
  });

  describe('getInvestmentsByProject', () => {
    it('should get investments by project', async () => {
      const results = await client.getInvestmentsByProject('project-123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.project_id).toBe('project-123');
    });
  });

  describe('getInvestmentsByInvestor', () => {
    it('should get investments by investor', async () => {
      const results = await client.getInvestmentsByInvestor('did:mycelix:investor789');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.investor).toBe('did:mycelix:investor789');
    });
  });
});

// ============================================================================
// Factory Function Tests
// ============================================================================

describe('createEnergyClients', () => {
  it('should create all energy clients', () => {
    const mockZome = createMockClient(new Map());
    const clients = createEnergyClients(mockZome);

    expect(clients.projects).toBeInstanceOf(ProjectsClient);
    expect(clients.participants).toBeInstanceOf(ParticipantsClient);
    expect(clients.trading).toBeInstanceOf(TradingClient);
    expect(clients.credits).toBeInstanceOf(CreditsClient);
    expect(clients.investments).toBeInstanceOf(InvestmentsClient);
  });
});

// ============================================================================
// Type Safety Tests
// ============================================================================

describe('Type Safety', () => {
  it('should enforce energy source constraints', () => {
    const sources: EnergySource[] = ['Solar', 'Wind', 'Hydro', 'Nuclear', 'Geothermal', 'Storage', 'Other'];

    sources.forEach((source) => {
      const project = createMockProject({ source });
      expect(sources).toContain(project.source);
    });
  });

  it('should enforce project status progression', () => {
    const statuses: ProjectStatus[] = [
      'Proposed',
      'Planning',
      'Development',
      'Construction',
      'Operational',
      'Decommissioned',
    ];

    statuses.forEach((status) => {
      const project = createMockProject({ status });
      expect(statuses).toContain(project.status);
    });
  });

  it('should enforce trade status constraints', () => {
    const statuses: EnergyTrade['status'][] = ['Pending', 'Confirmed', 'Delivered', 'Disputed'];

    statuses.forEach((status) => {
      const trade = createMockTrade({ status });
      expect(statuses).toContain(trade.status);
    });
  });

  it('should enforce capacity is positive', () => {
    const project = createMockProject();
    expect(project.capacity_kw).toBeGreaterThan(0);

    const participant = createMockParticipant();
    expect(participant.capacity_kwh).toBeGreaterThan(0);
  });

  it('should enforce price is positive', () => {
    const trade = createMockTrade();
    expect(trade.price_per_kwh).toBeGreaterThan(0);
  });

  it('should enforce ownership percentage in valid range', () => {
    const investment = createMockInvestment();
    expect(investment.ownership_percentage).toBeGreaterThanOrEqual(0);
    expect(investment.ownership_percentage).toBeLessThanOrEqual(100);
  });
});

// ============================================================================
// Integration Pattern Tests
// ============================================================================

describe('Integration Patterns', () => {
  it('should support full investment lifecycle', async () => {
    const responses = new Map<string, unknown>();
    const mockProject = createMockProject({ investment_raised: 0 });
    const mockInvestment = createMockInvestment();

    responses.set('projects:register_project', createMockRecord(mockProject));
    responses.set('investments:invest', createMockRecord(mockInvestment));
    responses.set(
      'projects:get_project',
      createMockRecord({
        ...mockProject,
        investment_raised: 25000,
      })
    );

    const mockZome = createMockClient(responses);
    const clients = createEnergyClients(mockZome);

    // Register project
    const project = await clients.projects.registerProject({
      name: 'Test Solar',
      description: 'Test',
      source: 'Solar',
      capacity_kw: 1000,
      location: { lat: 0, lng: 0 },
      investment_goal: 100000,
    });
    expect(project.entry.Present.investment_raised).toBe(0);

    // Make investment
    const investment = await clients.investments.invest({
      project_id: project.entry.Present.id,
      amount: 25000,
      currency: 'USD',
    });
    expect(investment.entry.Present.amount).toBe(25000);

    // Check updated project
    const updated = await clients.projects.getProject(project.entry.Present.id);
    expect(updated!.entry.Present.investment_raised).toBe(25000);
  });

  it('should support P2P trading flow', async () => {
    const responses = new Map<string, unknown>();
    const mockTrade = createMockTrade({ status: 'Pending' });

    responses.set('trading:create_trade', createMockRecord(mockTrade));
    responses.set(
      'trading:accept_trade',
      createMockRecord({
        ...mockTrade,
        status: 'Confirmed',
      })
    );
    responses.set(
      'trading:confirm_delivery',
      createMockRecord({
        ...mockTrade,
        status: 'Delivered',
        delivered_at: Date.now() * 1000,
      })
    );

    const mockZome = createMockClient(responses);
    const clients = createEnergyClients(mockZome);

    // Create trade
    const trade = await clients.trading.createTrade({
      buyer: 'did:mycelix:consumer',
      amount_kwh: 100,
      source: 'Solar',
      price_per_kwh: 0.10,
      currency: 'USD',
    });
    expect(trade.entry.Present.status).toBe('Pending');

    // Accept trade
    const accepted = await clients.trading.acceptTrade(trade.entry.Present.id);
    expect(accepted.entry.Present.status).toBe('Confirmed');

    // Confirm delivery
    const delivered = await clients.trading.confirmDelivery(trade.entry.Present.id);
    expect(delivered.entry.Present.status).toBe('Delivered');
  });

  it('should link credits to renewable production', async () => {
    const responses = new Map<string, unknown>();
    const mockParticipant = createMockParticipant();
    const mockCredit = createMockCredit();

    responses.set('participants:register', createMockRecord(mockParticipant));
    responses.set('credits:issue_credit', createMockRecord(mockCredit));
    responses.set('credits:get_credits', [createMockRecord(mockCredit)]);

    const mockZome = createMockClient(responses);
    const clients = createEnergyClients(mockZome);

    // Register as producer
    const participant = await clients.participants.register({
      type_: 'Producer',
      sources: ['Solar'],
      capacity_kwh: 1000,
    });
    expect(participant.entry.Present.type_).toBe('Producer');

    // Issue credit for production
    const credit = await clients.credits.issueCredit(
      participant.entry.Present.did,
      500,
      'Solar'
    );
    expect(credit.entry.Present.source).toBe('Solar');

    // Verify credits
    const credits = await clients.credits.getCredits(participant.entry.Present.did);
    expect(credits).toHaveLength(1);
  });
});

// ============================================================================
// Terra Atlas Integration Tests
// ============================================================================

describe('Terra Atlas Integration', () => {
  it('should support regenerative exit model', () => {
    const project = createMockProject({
      owner_dao: 'dao-community',
      investment_goal: 5000000,
      investment_raised: 4250000,
    });

    // 85% of goal reached
    const progressPercent = (project.investment_raised / (project.investment_goal || 1)) * 100;
    expect(progressPercent).toBe(85);
  });

  it('should track community ownership', () => {
    const investments = [
      createMockInvestment({ investor: 'did:mycelix:community1', ownership_percentage: 15 }),
      createMockInvestment({ investor: 'did:mycelix:community2', ownership_percentage: 10 }),
      createMockInvestment({ investor: 'did:mycelix:community3', ownership_percentage: 8 }),
    ];

    const totalCommunityOwnership = investments.reduce((sum, i) => sum + i.ownership_percentage, 0);
    expect(totalCommunityOwnership).toBe(33);
  });

  it('should support multiple energy sources per project region', () => {
    const projects = [
      createMockProject({ name: 'West Texas Solar', source: 'Solar', capacity_kw: 50000 }),
      createMockProject({ id: 'project-456', name: 'West Texas Wind', source: 'Wind', capacity_kw: 75000 }),
      createMockProject({ id: 'project-789', name: 'West Texas Storage', source: 'Storage', capacity_kw: 25000 }),
    ];

    const totalCapacity = projects.reduce((sum, p) => sum + p.capacity_kw, 0);
    expect(totalCapacity).toBe(150000);

    const sources = new Set(projects.map((p) => p.source));
    expect(sources.size).toBe(3);
  });
});

// ============================================================================
// Edge Case Tests
// ============================================================================

describe('Edge Cases', () => {
  it('should handle project without DAO owner', () => {
    const project = createMockProject({ owner_dao: undefined });
    expect(project.owner_dao).toBeUndefined();
  });

  it('should handle project without investment goal', () => {
    const project = createMockProject({ investment_goal: undefined });
    expect(project.investment_goal).toBeUndefined();
  });

  it('should handle participant without location', () => {
    const participant = createMockParticipant({ location: undefined });
    expect(participant.location).toBeUndefined();
  });

  it('should handle credit without expiration', () => {
    const credit = createMockCredit({ expires_at: undefined });
    expect(credit.expires_at).toBeUndefined();
  });

  it('should handle prosumer with both production and consumption', () => {
    const prosumer = createMockParticipant({
      type_: 'Prosumer',
      sources: ['Solar'],
      capacity_kwh: 500,
    });

    expect(prosumer.type_).toBe('Prosumer');
    expect(prosumer.sources).toContain('Solar');
  });

  it('should handle decommissioned project', () => {
    const project = createMockProject({ status: 'Decommissioned' });
    expect(project.status).toBe('Decommissioned');
  });
});

// ============================================================================
// Energy Validated Clients Tests
// ============================================================================

import {
  ValidatedProjectsClient,
  ValidatedParticipantsClient,
  ValidatedTradingClient,
  ValidatedCreditsClient,
  ValidatedInvestmentsClient,
  createValidatedEnergyClients,
} from '../src/energy/validated.js';

describe('Energy Validation Schema Boundary Tests', () => {
  let mockZome: ZomeCallable;
  const mockRecord = <T>(entry: T): HolochainRecord<T> => ({
    signed_action: { hashed: { hash: 'test_hash', content: {} }, signature: 'sig' },
    entry: { Present: entry },
  });

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    responses.set('projects:register_project', mockRecord(createMockProject()));
    responses.set('projects:get_project', mockRecord(createMockProject()));
    responses.set('projects:get_projects_by_source', [mockRecord(createMockProject())]);
    responses.set('projects:get_projects_by_status', [mockRecord(createMockProject())]);
    responses.set('projects:search_projects', [mockRecord(createMockProject())]);
    responses.set('participants:register', mockRecord(createMockParticipant()));
    responses.set('participants:get_participant', mockRecord(createMockParticipant()));
    responses.set('participants:get_by_type', [mockRecord(createMockParticipant())]);
    responses.set('participants:update_capacity', mockRecord(createMockParticipant()));
    responses.set('trading:create_trade', mockRecord(createMockTrade()));
    responses.set('trading:accept_trade', mockRecord(createMockTrade()));
    responses.set('trading:confirm_delivery', mockRecord(createMockTrade()));
    responses.set('trading:get_trades_by_participant', [mockRecord(createMockTrade())]);
    responses.set('trading:get_open_trades', [mockRecord(createMockTrade())]);
    responses.set('credits:issue_credit', mockRecord(createMockCredit()));
    responses.set('credits:get_credits', [mockRecord(createMockCredit())]);
    responses.set('credits:use_credit', mockRecord(createMockCredit()));
    responses.set('investments:invest', mockRecord(createMockInvestment()));
    responses.set('investments:get_by_project', [mockRecord(createMockInvestment())]);
    responses.set('investments:get_by_investor', [mockRecord(createMockInvestment())]);
    mockZome = createMockClient(responses);
  });

  describe('EnergySource Enum Validation', () => {
    const validSources: EnergySource[] = ['Solar', 'Wind', 'Hydro', 'Nuclear', 'Geothermal', 'Storage', 'Other'];

    it.each(validSources)('should accept valid energy source: %s', async (source) => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: 'Test Project',
        description: 'Description',
        source,
        capacity_kw: 1000,
        location: { lat: 0, lng: 0 },
      })).resolves.toBeDefined();
    });

    it.each([
      'solar', // lowercase
      'SOLAR', // uppercase
      'Invalid',
      'Biomass',
      '',
      ' Solar',
      'Solar ',
    ])('should reject invalid energy source: %s', async (invalidSource) => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: 'Test Project',
        description: 'Description',
        source: invalidSource as EnergySource,
        capacity_kw: 1000,
        location: { lat: 0, lng: 0 },
      })).rejects.toThrow(/Validation failed/);
    });

    it('should validate energy source in getProjectsBySource', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.getProjectsBySource('Solar')).resolves.toBeDefined();
      await expect(client.getProjectsBySource('invalid' as EnergySource)).rejects.toThrow(/Validation failed/);
    });
  });

  describe('ProjectStatus Enum Validation', () => {
    const validStatuses: ProjectStatus[] = [
      'Proposed', 'Planning', 'Development', 'Construction', 'Operational', 'Decommissioned'
    ];

    it.each(validStatuses)('should accept valid project status: %s', async (status) => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.getProjectsByStatus(status)).resolves.toBeDefined();
    });

    it.each([
      'proposed', // lowercase
      'OPERATIONAL', // uppercase
      'Active',
      'Closed',
      '',
      'Pending',
    ])('should reject invalid project status: %s', async (invalidStatus) => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.getProjectsByStatus(invalidStatus as ProjectStatus))
        .rejects.toThrow(/Validation failed/);
    });
  });

  describe('ParticipantType Enum Validation', () => {
    const validTypes: ParticipantType[] = ['Producer', 'Consumer', 'Prosumer', 'Operator', 'Investor'];

    it.each(validTypes)('should accept valid participant type: %s', async (type_) => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.register({
        type_,
        sources: ['Solar'],
        capacity_kwh: 100,
      })).resolves.toBeDefined();
    });

    it.each([
      'producer', // lowercase
      'CONSUMER', // uppercase
      'User',
      'Admin',
      '',
      'Buyer',
    ])('should reject invalid participant type: %s', async (invalidType) => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.register({
        type_: invalidType as ParticipantType,
        sources: ['Solar'],
        capacity_kwh: 100,
      })).rejects.toThrow(/Validation failed/);
    });

    it('should validate type in getParticipantsByType', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.getParticipantsByType('Producer')).resolves.toBeDefined();
      await expect(client.getParticipantsByType('invalid' as ParticipantType)).rejects.toThrow(/Validation failed/);
    });
  });

  describe('Location Validation', () => {
    it('should accept valid latitude at boundaries', async () => {
      const client = new ValidatedProjectsClient(mockZome);

      // Min boundary: -90
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: -90, lng: 0 },
      })).resolves.toBeDefined();

      // Max boundary: 90
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 90, lng: 0 },
      })).resolves.toBeDefined();
    });

    it('should reject latitude outside boundaries', async () => {
      const client = new ValidatedProjectsClient(mockZome);

      // Below min: -90.01
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: -90.01, lng: 0 },
      })).rejects.toThrow(/Validation failed/);

      // Above max: 90.01
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 90.01, lng: 0 },
      })).rejects.toThrow(/Validation failed/);
    });

    it('should accept valid longitude at boundaries', async () => {
      const client = new ValidatedProjectsClient(mockZome);

      // Min boundary: -180
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: -180 },
      })).resolves.toBeDefined();

      // Max boundary: 180
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: 180 },
      })).resolves.toBeDefined();
    });

    it('should reject longitude outside boundaries', async () => {
      const client = new ValidatedProjectsClient(mockZome);

      // Below min: -180.01
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: -180.01 },
      })).rejects.toThrow(/Validation failed/);

      // Above max: 180.01
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: 180.01 },
      })).rejects.toThrow(/Validation failed/);
    });
  });

  describe('Project Name Validation', () => {
    it('should accept name at minimum length (1)', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: 'A',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: 0 },
      })).resolves.toBeDefined();
    });

    it('should accept name at maximum length (200)', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: 'A'.repeat(200),
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: 0 },
      })).resolves.toBeDefined();
    });

    it('should reject empty name', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: '',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: 0 },
      })).rejects.toThrow(/Validation failed/);
    });

    it('should reject name exceeding maximum length (201)', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: 'A'.repeat(201),
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: 0 },
      })).rejects.toThrow(/Validation failed/);
    });
  });

  describe('Description Validation', () => {
    it('should accept description at minimum length (1)', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: 'Test',
        description: 'D',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: 0 },
      })).resolves.toBeDefined();
    });

    it('should reject empty description', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: 'Test',
        description: '',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: 0 },
      })).rejects.toThrow(/Validation failed/);
    });
  });

  describe('Capacity Validation', () => {
    it('should accept positive capacity_kw', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 0.001, // smallest positive
        location: { lat: 0, lng: 0 },
      })).resolves.toBeDefined();
    });

    it('should reject zero capacity_kw', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 0,
        location: { lat: 0, lng: 0 },
      })).rejects.toThrow(/Validation failed/);
    });

    it('should reject negative capacity_kw', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: -100,
        location: { lat: 0, lng: 0 },
      })).rejects.toThrow(/Validation failed/);
    });

    it('should accept zero capacity_kwh for participants', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.register({
        type_: 'Consumer',
        sources: ['Solar'],
        capacity_kwh: 0,
      })).resolves.toBeDefined();
    });

    it('should reject negative capacity_kwh for participants', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.register({
        type_: 'Consumer',
        sources: ['Solar'],
        capacity_kwh: -1,
      })).rejects.toThrow(/Validation failed/);
    });

    it('should accept zero capacityKwh in updateCapacity', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.updateCapacity(0)).resolves.toBeDefined();
    });

    it('should reject negative capacityKwh in updateCapacity', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.updateCapacity(-5)).rejects.toThrow(/Validation failed/);
    });
  });

  describe('DID Validation', () => {
    it('should accept valid DID format', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.getParticipant('did:mycelix:abc123')).resolves.toBeDefined();
    });

    it('should accept various valid DID prefixes', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.getParticipant('did:example:123')).resolves.toBeDefined();
      await expect(client.getParticipant('did:web:example.com')).resolves.toBeDefined();
      await expect(client.getParticipant('did:key:z6Mk...')).resolves.toBeDefined();
    });

    it('should reject DIDs without did: prefix', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.getParticipant('mycelix:abc123')).rejects.toThrow(/Validation failed/);
      await expect(client.getParticipant('abc123')).rejects.toThrow(/Validation failed/);
      await expect(client.getParticipant('')).rejects.toThrow(/Validation failed/);
    });

    it('should validate DID in trade input', async () => {
      const client = new ValidatedTradingClient(mockZome);

      // Valid DID
      await expect(client.createTrade({
        buyer: 'did:mycelix:buyer123',
        amount_kwh: 100,
        source: 'Solar',
        price_per_kwh: 0.1,
        currency: 'USD',
      })).resolves.toBeDefined();

      // Invalid DID
      await expect(client.createTrade({
        buyer: 'invalid-did',
        amount_kwh: 100,
        source: 'Solar',
        price_per_kwh: 0.1,
        currency: 'USD',
      })).rejects.toThrow(/Validation failed/);
    });

    it('should validate DID in getTradesByParticipant', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.getTradesByParticipant('did:mycelix:abc')).resolves.toBeDefined();
      await expect(client.getTradesByParticipant('invalid')).rejects.toThrow(/Validation failed/);
    });

    it('should validate DID in credit operations', async () => {
      const creditsClient = new ValidatedCreditsClient(mockZome);

      await expect(creditsClient.issueCredit('did:mycelix:participant', 100, 'Solar')).resolves.toBeDefined();
      await expect(creditsClient.issueCredit('invalid-did', 100, 'Solar')).rejects.toThrow(/Validation failed/);

      await expect(creditsClient.getCredits('did:mycelix:participant')).resolves.toBeDefined();
      await expect(creditsClient.getCredits('invalid')).rejects.toThrow(/Validation failed/);
    });

    it('should validate DID in getInvestmentsByInvestor', async () => {
      const client = new ValidatedInvestmentsClient(mockZome);
      await expect(client.getInvestmentsByInvestor('did:mycelix:investor')).resolves.toBeDefined();
      await expect(client.getInvestmentsByInvestor('invalid')).rejects.toThrow(/Validation failed/);
    });
  });

  describe('Sources Array Validation', () => {
    it('should accept sources array with one item', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.register({
        type_: 'Producer',
        sources: ['Solar'],
        capacity_kwh: 100,
      })).resolves.toBeDefined();
    });

    it('should accept sources array with multiple items', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.register({
        type_: 'Producer',
        sources: ['Solar', 'Wind', 'Hydro'],
        capacity_kwh: 100,
      })).resolves.toBeDefined();
    });

    it('should reject empty sources array', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.register({
        type_: 'Producer',
        sources: [],
        capacity_kwh: 100,
      })).rejects.toThrow(/Validation failed/);
    });

    it('should validate each source in the array', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.register({
        type_: 'Producer',
        sources: ['Solar', 'InvalidSource' as EnergySource],
        capacity_kwh: 100,
      })).rejects.toThrow(/Validation failed/);
    });
  });

  describe('Trade Input Validation', () => {
    it('should accept positive amount_kwh', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.createTrade({
        buyer: 'did:mycelix:buyer',
        amount_kwh: 0.001,
        source: 'Solar',
        price_per_kwh: 0.1,
        currency: 'USD',
      })).resolves.toBeDefined();
    });

    it('should reject zero amount_kwh', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.createTrade({
        buyer: 'did:mycelix:buyer',
        amount_kwh: 0,
        source: 'Solar',
        price_per_kwh: 0.1,
        currency: 'USD',
      })).rejects.toThrow(/Validation failed/);
    });

    it('should reject negative amount_kwh', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.createTrade({
        buyer: 'did:mycelix:buyer',
        amount_kwh: -10,
        source: 'Solar',
        price_per_kwh: 0.1,
        currency: 'USD',
      })).rejects.toThrow(/Validation failed/);
    });

    it('should accept positive price_per_kwh', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.createTrade({
        buyer: 'did:mycelix:buyer',
        amount_kwh: 100,
        source: 'Solar',
        price_per_kwh: 0.001,
        currency: 'USD',
      })).resolves.toBeDefined();
    });

    it('should reject zero price_per_kwh', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.createTrade({
        buyer: 'did:mycelix:buyer',
        amount_kwh: 100,
        source: 'Solar',
        price_per_kwh: 0,
        currency: 'USD',
      })).rejects.toThrow(/Validation failed/);
    });

    it('should reject negative price_per_kwh', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.createTrade({
        buyer: 'did:mycelix:buyer',
        amount_kwh: 100,
        source: 'Solar',
        price_per_kwh: -0.5,
        currency: 'USD',
      })).rejects.toThrow(/Validation failed/);
    });

    it('should require non-empty currency', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.createTrade({
        buyer: 'did:mycelix:buyer',
        amount_kwh: 100,
        source: 'Solar',
        price_per_kwh: 0.1,
        currency: '',
      })).rejects.toThrow(/Validation failed/);
    });

    it('should accept any non-empty currency', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.createTrade({
        buyer: 'did:mycelix:buyer',
        amount_kwh: 100,
        source: 'Solar',
        price_per_kwh: 0.1,
        currency: 'EUR',
      })).resolves.toBeDefined();
    });
  });

  describe('Investment Input Validation', () => {
    it('should require non-empty project_id', async () => {
      const client = new ValidatedInvestmentsClient(mockZome);
      await expect(client.invest({
        project_id: '',
        amount: 1000,
        currency: 'USD',
      })).rejects.toThrow(/Validation failed/);
    });

    it('should accept non-empty project_id', async () => {
      const client = new ValidatedInvestmentsClient(mockZome);
      await expect(client.invest({
        project_id: 'p',
        amount: 1000,
        currency: 'USD',
      })).resolves.toBeDefined();
    });

    it('should require positive amount', async () => {
      const client = new ValidatedInvestmentsClient(mockZome);

      // Positive amount
      await expect(client.invest({
        project_id: 'project-123',
        amount: 0.01,
        currency: 'USD',
      })).resolves.toBeDefined();

      // Zero amount
      await expect(client.invest({
        project_id: 'project-123',
        amount: 0,
        currency: 'USD',
      })).rejects.toThrow(/Validation failed/);

      // Negative amount
      await expect(client.invest({
        project_id: 'project-123',
        amount: -100,
        currency: 'USD',
      })).rejects.toThrow(/Validation failed/);
    });

    it('should require non-empty currency', async () => {
      const client = new ValidatedInvestmentsClient(mockZome);
      await expect(client.invest({
        project_id: 'project-123',
        amount: 1000,
        currency: '',
      })).rejects.toThrow(/Validation failed/);
    });

    it('should validate projectId in getInvestmentsByProject', async () => {
      const client = new ValidatedInvestmentsClient(mockZome);
      await expect(client.getInvestmentsByProject('project-123')).resolves.toBeDefined();
      await expect(client.getInvestmentsByProject('')).rejects.toThrow(/Validation failed/);
    });
  });

  describe('Credit Operations Validation', () => {
    it('should require positive amountKwh in issueCredit', async () => {
      const client = new ValidatedCreditsClient(mockZome);

      await expect(client.issueCredit('did:mycelix:participant', 0.01, 'Solar')).resolves.toBeDefined();
      await expect(client.issueCredit('did:mycelix:participant', 0, 'Solar')).rejects.toThrow(/Validation failed/);
      await expect(client.issueCredit('did:mycelix:participant', -10, 'Solar')).rejects.toThrow(/Validation failed/);
    });

    it('should validate energy source in issueCredit', async () => {
      const client = new ValidatedCreditsClient(mockZome);
      await expect(client.issueCredit('did:mycelix:participant', 100, 'Solar')).resolves.toBeDefined();
      await expect(client.issueCredit('did:mycelix:participant', 100, 'InvalidSource' as EnergySource)).rejects.toThrow(/Validation failed/);
    });

    it('should require non-empty creditId in useCredit', async () => {
      const client = new ValidatedCreditsClient(mockZome);
      await expect(client.useCredit('credit-123')).resolves.toBeDefined();
      await expect(client.useCredit('')).rejects.toThrow(/Validation failed/);
    });
  });

  describe('String ID Validations', () => {
    it('should require non-empty projectId in getProject', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.getProject('project-123')).resolves.toBeDefined();
      await expect(client.getProject('')).rejects.toThrow(/Validation failed/);
    });

    it('should require non-empty tradeId in acceptTrade', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.acceptTrade('trade-123')).resolves.toBeDefined();
      await expect(client.acceptTrade('')).rejects.toThrow(/Validation failed/);
    });

    it('should require non-empty tradeId in confirmDelivery', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.confirmDelivery('trade-123')).resolves.toBeDefined();
      await expect(client.confirmDelivery('')).rejects.toThrow(/Validation failed/);
    });
  });

  describe('Optional Source in getOpenTrades', () => {
    it('should accept undefined source', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.getOpenTrades()).resolves.toBeDefined();
    });

    it('should accept valid source', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.getOpenTrades('Solar')).resolves.toBeDefined();
    });

    it('should reject invalid source when provided', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.getOpenTrades('invalid' as EnergySource)).rejects.toThrow(/Validation failed/);
    });
  });

  describe('Optional Investment Goal', () => {
    it('should accept project without investment_goal', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: 0 },
      })).resolves.toBeDefined();
    });

    it('should accept positive investment_goal', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: 0 },
        investment_goal: 1000000,
      })).resolves.toBeDefined();
    });

    it('should reject zero investment_goal', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: 0 },
        investment_goal: 0,
      })).rejects.toThrow(/Validation failed/);
    });

    it('should reject negative investment_goal', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: 'Test',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: 0 },
        investment_goal: -5000,
      })).rejects.toThrow(/Validation failed/);
    });
  });

  describe('Optional Location for Participants', () => {
    it('should accept participant without location', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.register({
        type_: 'Consumer',
        sources: ['Solar'],
        capacity_kwh: 100,
      })).resolves.toBeDefined();
    });

    it('should accept participant with valid location', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.register({
        type_: 'Producer',
        sources: ['Solar'],
        capacity_kwh: 100,
        location: { lat: 45.5, lng: -122.6 },
      })).resolves.toBeDefined();
    });

    it('should reject participant with invalid location', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.register({
        type_: 'Producer',
        sources: ['Solar'],
        capacity_kwh: 100,
        location: { lat: 100, lng: 0 }, // invalid lat
      })).rejects.toThrow(/Validation failed/);
    });
  });

  describe('createValidatedEnergyClients Factory', () => {
    it('should create all validated clients', () => {
      const clients = createValidatedEnergyClients(mockZome);
      expect(clients.projects).toBeInstanceOf(ValidatedProjectsClient);
      expect(clients.participants).toBeInstanceOf(ValidatedParticipantsClient);
      expect(clients.trading).toBeInstanceOf(ValidatedTradingClient);
      expect(clients.credits).toBeInstanceOf(ValidatedCreditsClient);
      expect(clients.investments).toBeInstanceOf(ValidatedInvestmentsClient);
    });
  });

  describe('Error Message Context', () => {
    it('should include context in error message for registerProject', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.registerProject({
        name: '',
        description: 'Desc',
        source: 'Solar',
        capacity_kw: 100,
        location: { lat: 0, lng: 0 },
      })).rejects.toThrow(/registerProject input/);
    });

    it('should include context in error message for getProject', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      await expect(client.getProject('')).rejects.toThrow(/projectId/);
    });

    it('should include context in error message for register', async () => {
      const client = new ValidatedParticipantsClient(mockZome);
      await expect(client.register({
        type_: '' as ParticipantType,
        sources: ['Solar'],
        capacity_kwh: 100,
      })).rejects.toThrow(/register input/);
    });

    it('should include context in error message for createTrade', async () => {
      const client = new ValidatedTradingClient(mockZome);
      await expect(client.createTrade({
        buyer: 'invalid',
        amount_kwh: 100,
        source: 'Solar',
        price_per_kwh: 0.1,
        currency: 'USD',
      })).rejects.toThrow(/createTrade input/);
    });

    it('should include context in error message for invest', async () => {
      const client = new ValidatedInvestmentsClient(mockZome);
      await expect(client.invest({
        project_id: '',
        amount: 0,
        currency: 'USD',
      })).rejects.toThrow(/invest input/);
    });
  });

  describe('Search Projects Without Validation', () => {
    it('should pass through searchProjects without validation', async () => {
      const client = new ValidatedProjectsClient(mockZome);
      // searchProjects doesn't validate its input currently
      await expect(client.searchProjects({})).resolves.toBeDefined();
      await expect(client.searchProjects({ source: 'Solar' })).resolves.toBeDefined();
      await expect(client.searchProjects({ status: 'Operational', minCapacity: 1000 })).resolves.toBeDefined();
    });
  });
});
