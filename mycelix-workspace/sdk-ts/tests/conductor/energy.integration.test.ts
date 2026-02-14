/**
 * Energy hApp Conductor Integration Tests
 *
 * These tests verify the Energy clients work correctly with a real
 * Holochain conductor. Integrates with Terra Atlas for project discovery.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  ProjectsClient,
  ParticipantsClient,
  TradingClient,
  CreditsClient,
  InvestmentsClient,
  createEnergyClients,
  type ZomeCallable,
  type EnergyProject,
  type EnergyParticipant,
  type EnergyTrade,
  type EnergyCredit,
  type Investment,
} from '../../src/energy/index.js';
import { createValidatedEnergyClients } from '../../src/energy/validated.js';
import { MycelixError } from '../../src/errors.js';
import { CONDUCTOR_ENABLED, generateTestAgentId } from './conductor-harness.js';

function createMockClient(): ZomeCallable {
  const projects = new Map<string, EnergyProject>();
  const participants = new Map<string, EnergyParticipant>();
  const trades = new Map<string, EnergyTrade>();
  const credits = new Map<string, EnergyCredit[]>();
  const investments = new Map<string, Investment[]>();
  // Generate agent ID once to ensure consistency across all calls
  const agentId = `did:mycelix:${generateTestAgentId()}`;
  let idCounter = 0;

  return {
    async callZome({
      fn_name,
      payload,
    }: {
      role_name: string;
      zome_name: string;
      fn_name: string;
      payload: unknown;
    }) {

      switch (fn_name) {
        case 'register_project': {
          const input = payload as {
            name: string;
            description: string;
            source: string;
            capacity_kw: number;
            location: any;
            investment_goal?: number;
          };
          const project: EnergyProject = {
            id: `project-${Date.now()}-${++idCounter}`,
            owner: agentId,
            name: input.name,
            description: input.description,
            source: input.source as EnergyProject['source'],
            capacity_kw: input.capacity_kw,
            location: input.location,
            status: 'Proposed',
            investment_goal: input.investment_goal,
            investment_raised: 0,
            terra_atlas_id: undefined,
            registered_at: Date.now(),
            updated_at: Date.now(),
          };
          projects.set(project.id, project);
          return {
            signed_action: { hashed: { hash: project.id, content: {} }, signature: 'sig' },
            entry: { Present: project },
          };
        }

        case 'get_project': {
          const id = payload as string;
          const p = projects.get(id);
          return p
            ? {
                signed_action: { hashed: { hash: id, content: {} }, signature: 'sig' },
                entry: { Present: p },
              }
            : null;
        }

        case 'get_projects_by_source': {
          const source = payload as string;
          return Array.from(projects.values())
            .filter((p) => p.source === source)
            .map((p) => ({
              signed_action: { hashed: { hash: p.id, content: {} }, signature: 'sig' },
              entry: { Present: p },
            }));
        }

        case 'get_projects_by_status': {
          const status = payload as string;
          return Array.from(projects.values())
            .filter((p) => p.status === status)
            .map((p) => ({
              signed_action: { hashed: { hash: p.id, content: {} }, signature: 'sig' },
              entry: { Present: p },
            }));
        }

        case 'search_projects': {
          const query = payload as { source?: string; status?: string; minCapacity?: number };
          return Array.from(projects.values())
            .filter((p) => {
              if (query.source && p.source !== query.source) return false;
              if (query.status && p.status !== query.status) return false;
              if (query.minCapacity && p.capacity_kw < query.minCapacity) return false;
              return true;
            })
            .map((p) => ({
              signed_action: { hashed: { hash: p.id, content: {} }, signature: 'sig' },
              entry: { Present: p },
            }));
        }

        case 'register': {
          const input = payload as {
            type_: string;
            sources: string[];
            capacity_kwh: number;
            location?: any;
          };
          const participant: EnergyParticipant = {
            did: agentId,
            type_: input.type_ as EnergyParticipant['type_'],
            sources: input.sources as EnergyParticipant['sources'],
            capacity_kwh: input.capacity_kwh,
            location: input.location,
            reputation_score: 0.5,
            total_energy_traded: 0,
            total_credits_earned: 0,
            registered_at: Date.now(),
            updated_at: Date.now(),
          };
          participants.set(agentId, participant);
          return {
            signed_action: { hashed: { hash: agentId, content: {} }, signature: 'sig' },
            entry: { Present: participant },
          };
        }

        case 'get_participant': {
          const did = payload as string;
          const p = participants.get(did);
          return p
            ? {
                signed_action: { hashed: { hash: did, content: {} }, signature: 'sig' },
                entry: { Present: p },
              }
            : null;
        }

        case 'get_participants_by_type': {
          const type = payload as string;
          return Array.from(participants.values())
            .filter((p) => p.type_ === type)
            .map((p) => ({
              signed_action: { hashed: { hash: p.did, content: {} }, signature: 'sig' },
              entry: { Present: p },
            }));
        }

        case 'update_capacity': {
          const capacityKwh = payload as number;
          const p = participants.get(agentId);
          if (p) {
            p.capacity_kwh = capacityKwh;
            p.updated_at = Date.now();
          }
          return p
            ? {
                signed_action: { hashed: { hash: agentId, content: {} }, signature: 'sig' },
                entry: { Present: p },
              }
            : null;
        }

        case 'create_trade': {
          const input = payload as {
            buyer: string;
            amount_kwh: number;
            source: string;
            price_per_kwh: number;
            currency: string;
          };
          const trade: EnergyTrade = {
            id: `trade-${Date.now()}-${++idCounter}`,
            seller: agentId,
            buyer: input.buyer,
            amount_kwh: input.amount_kwh,
            source: input.source as EnergyTrade['source'],
            price_per_kwh: input.price_per_kwh,
            currency: input.currency,
            status: 'Open',
            created_at: Date.now(),
          };
          trades.set(trade.id, trade);
          return {
            signed_action: { hashed: { hash: trade.id, content: {} }, signature: 'sig' },
            entry: { Present: trade },
          };
        }

        case 'accept_trade': {
          const tradeId = payload as string;
          const t = trades.get(tradeId);
          if (t) {
            t.status = 'Accepted';
            t.accepted_at = Date.now();
          }
          return t
            ? {
                signed_action: { hashed: { hash: tradeId, content: {} }, signature: 'sig' },
                entry: { Present: t },
              }
            : null;
        }

        case 'confirm_delivery': {
          const tradeId = payload as string;
          const t = trades.get(tradeId);
          if (t) {
            t.status = 'Delivered';
            t.delivered_at = Date.now();
            // Update participant stats
            const seller = participants.get(t.seller);
            const buyer = participants.get(t.buyer);
            if (seller) {
              seller.total_energy_traded += t.amount_kwh;
            }
            if (buyer) {
              buyer.total_energy_traded += t.amount_kwh;
            }
          }
          return t
            ? {
                signed_action: { hashed: { hash: tradeId, content: {} }, signature: 'sig' },
                entry: { Present: t },
              }
            : null;
        }

        case 'get_trades_by_participant': {
          const did = payload as string;
          return Array.from(trades.values())
            .filter((t) => t.seller === did || t.buyer === did)
            .map((t) => ({
              signed_action: { hashed: { hash: t.id, content: {} }, signature: 'sig' },
              entry: { Present: t },
            }));
        }

        case 'get_open_trades': {
          const source = payload as string | undefined;
          return Array.from(trades.values())
            .filter((t) => t.status === 'Open' && (!source || t.source === source))
            .map((t) => ({
              signed_action: { hashed: { hash: t.id, content: {} }, signature: 'sig' },
              entry: { Present: t },
            }));
        }

        case 'issue_credit': {
          const { participant_did, amount_kwh, source } = payload as {
            participant_did: string;
            amount_kwh: number;
            source: string;
          };
          const credit: EnergyCredit = {
            id: `credit-${Date.now()}-${++idCounter}`,
            participant: participant_did,
            amount_kwh,
            source: source as EnergyCredit['source'],
            issued_at: Date.now(),
            valid_until: Date.now() + 365 * 24 * 60 * 60 * 1000,
            used: false,
          };
          const existing = credits.get(participant_did) || [];
          existing.push(credit);
          credits.set(participant_did, existing);
          // Update participant stats
          const p = participants.get(participant_did);
          if (p) {
            p.total_credits_earned += amount_kwh;
          }
          return {
            signed_action: { hashed: { hash: credit.id, content: {} }, signature: 'sig' },
            entry: { Present: credit },
          };
        }

        case 'get_credits': {
          const participantDid = payload as string;
          const cs = credits.get(participantDid) || [];
          return cs.map((c) => ({
            signed_action: { hashed: { hash: c.id, content: {} }, signature: 'sig' },
            entry: { Present: c },
          }));
        }

        case 'use_credit': {
          const creditId = payload as string;
          for (const cs of credits.values()) {
            const c = cs.find((cr) => cr.id === creditId);
            if (c) {
              c.used = true;
              c.used_at = Date.now();
              return {
                signed_action: { hashed: { hash: creditId, content: {} }, signature: 'sig' },
                entry: { Present: c },
              };
            }
          }
          throw new Error('Credit not found');
        }

        case 'invest': {
          const input = payload as { project_id: string; amount: number; currency: string };
          const project = projects.get(input.project_id);
          if (!project) throw new Error('Project not found');
          const investment: Investment = {
            id: `inv-${Date.now()}-${++idCounter}`,
            project_id: input.project_id,
            investor: agentId,
            amount: input.amount,
            currency: input.currency,
            status: 'Pledged',
            ownership_percentage: 0,
            invested_at: Date.now(),
          };
          const existing = investments.get(input.project_id) || [];
          existing.push(investment);
          investments.set(input.project_id, existing);
          project.investment_raised = (project.investment_raised || 0) + input.amount;
          return {
            signed_action: { hashed: { hash: investment.id, content: {} }, signature: 'sig' },
            entry: { Present: investment },
          };
        }

        case 'get_investments_by_project': {
          const projectId = payload as string;
          const invs = investments.get(projectId) || [];
          return invs.map((i) => ({
            signed_action: { hashed: { hash: i.id, content: {} }, signature: 'sig' },
            entry: { Present: i },
          }));
        }

        case 'get_investments_by_investor': {
          const investorDid = payload as string;
          const allInvs: Investment[] = [];
          for (const invs of investments.values()) {
            for (const i of invs) {
              if (i.investor === investorDid) {
                allInvs.push(i);
              }
            }
          }
          return allInvs.map((i) => ({
            signed_action: { hashed: { hash: i.id, content: {} }, signature: 'sig' },
            entry: { Present: i },
          }));
        }

        default:
          throw new Error(`Unknown function: ${fn_name}`);
      }
    },
  };
}

const describeConductor = CONDUCTOR_ENABLED ? describe : describe.skip;

describe('Energy Clients (Mock)', () => {
  let mockClient: ZomeCallable;
  let projectsClient: ProjectsClient;
  let participantsClient: ParticipantsClient;
  let tradingClient: TradingClient;
  let creditsClient: CreditsClient;
  let investmentsClient: InvestmentsClient;

  beforeAll(() => {
    mockClient = createMockClient();
    const clients = createEnergyClients(mockClient);
    projectsClient = clients.projects;
    participantsClient = clients.participants;
    tradingClient = clients.trading;
    creditsClient = clients.credits;
    investmentsClient = clients.investments;
  });

  describe('ProjectsClient', () => {
    it('should register a project', async () => {
      const result = await projectsClient.registerProject({
        name: 'Solar Farm Alpha',
        description: 'A 5MW community solar installation',
        source: 'Solar',
        capacity_kw: 5000,
        location: { lat: 35.6762, lng: 139.6503, address: 'Tokyo, Japan' },
        investment_goal: 2000000,
      });

      expect(result).toBeDefined();
      const project = result.entry.Present as EnergyProject;
      expect(project.name).toBe('Solar Farm Alpha');
      expect(project.source).toBe('Solar');
      expect(project.status).toBe('Proposed');
    });

    it('should search projects', async () => {
      await projectsClient.registerProject({
        name: 'Wind Farm Beta',
        description: 'Offshore wind installation',
        source: 'Wind',
        capacity_kw: 10000,
        location: { lat: 51.5074, lng: -0.1278 },
      });

      const result = await projectsClient.searchProjects({ source: 'Wind', minCapacity: 5000 });

      expect(result.length).toBeGreaterThan(0);
      expect((result[0].entry.Present as EnergyProject).source).toBe('Wind');
    });
  });

  describe('ParticipantsClient', () => {
    it('should register as a participant', async () => {
      const result = await participantsClient.register({
        type_: 'Prosumer',
        sources: ['Solar', 'Storage'],
        capacity_kwh: 50,
        location: { lat: 40.7128, lng: -74.006 },
      });

      expect(result).toBeDefined();
      const participant = result.entry.Present as EnergyParticipant;
      expect(participant.type_).toBe('Prosumer');
      expect(participant.sources).toContain('Solar');
    });

    it('should update capacity', async () => {
      await participantsClient.register({
        type_: 'Producer',
        sources: ['Wind'],
        capacity_kwh: 1000,
      });

      const result = await participantsClient.updateCapacity(1500);

      expect(result).toBeDefined();
      expect((result!.entry.Present as EnergyParticipant).capacity_kwh).toBe(1500);
    });
  });

  describe('TradingClient', () => {
    it('should create a trade', async () => {
      const result = await tradingClient.createTrade({
        buyer: 'did:mycelix:buyer123',
        amount_kwh: 100,
        source: 'Solar',
        price_per_kwh: 0.12,
        currency: 'USD',
      });

      expect(result).toBeDefined();
      const trade = result.entry.Present as EnergyTrade;
      expect(trade.amount_kwh).toBe(100);
      expect(trade.status).toBe('Open');
    });

    it('should complete a trade lifecycle', async () => {
      const created = await tradingClient.createTrade({
        buyer: 'did:mycelix:buyer456',
        amount_kwh: 50,
        source: 'Wind',
        price_per_kwh: 0.1,
        currency: 'EUR',
      });
      const tradeId = (created.entry.Present as EnergyTrade).id;

      await tradingClient.acceptTrade(tradeId);
      const delivered = await tradingClient.confirmDelivery(tradeId);

      expect((delivered.entry.Present as EnergyTrade).status).toBe('Delivered');
    });

    it('should get open trades', async () => {
      await tradingClient.createTrade({
        buyer: 'did:mycelix:open1',
        amount_kwh: 25,
        source: 'Hydro',
        price_per_kwh: 0.08,
        currency: 'USD',
      });

      const result = await tradingClient.getOpenTrades('Hydro');

      expect(result.length).toBeGreaterThan(0);
    });
  });

  describe('CreditsClient', () => {
    it('should issue a credit', async () => {
      const result = await creditsClient.issueCredit('did:mycelix:greenprod', 500, 'Solar');

      expect(result).toBeDefined();
      const credit = result.entry.Present as EnergyCredit;
      expect(credit.amount_kwh).toBe(500);
      expect(credit.used).toBe(false);
    });

    it('should use a credit', async () => {
      const issued = await creditsClient.issueCredit('did:mycelix:usecredit', 100, 'Wind');
      const creditId = (issued.entry.Present as EnergyCredit).id;

      const result = await creditsClient.useCredit(creditId);

      expect((result.entry.Present as EnergyCredit).used).toBe(true);
    });
  });

  describe('InvestmentsClient', () => {
    it('should invest in a project', async () => {
      const project = await projectsClient.registerProject({
        name: 'Community Solar',
        description: 'Local solar project',
        source: 'Solar',
        capacity_kw: 1000,
        location: { lat: 0, lng: 0 },
        investment_goal: 500000,
      });
      const projectId = (project.entry.Present as EnergyProject).id;

      const result = await investmentsClient.invest({
        project_id: projectId,
        amount: 10000,
        currency: 'USD',
      });

      expect(result).toBeDefined();
      const investment = result.entry.Present as Investment;
      expect(investment.amount).toBe(10000);
      expect(investment.status).toBe('Pledged');
    });

    it('should track investment raised', async () => {
      const project = await projectsClient.registerProject({
        name: 'Wind Cooperative',
        description: 'Community wind project',
        source: 'Wind',
        capacity_kw: 2000,
        location: { lat: 0, lng: 0 },
        investment_goal: 1000000,
      });
      const projectId = (project.entry.Present as EnergyProject).id;

      await investmentsClient.invest({ project_id: projectId, amount: 50000, currency: 'USD' });
      await investmentsClient.invest({ project_id: projectId, amount: 25000, currency: 'USD' });

      const updated = await projectsClient.getProject(projectId);
      expect((updated!.entry.Present as EnergyProject).investment_raised).toBe(75000);
    });
  });
});

describe('Validated Energy Clients', () => {
  let mockClient: ZomeCallable;
  let validatedClients: ReturnType<typeof createValidatedEnergyClients>;

  beforeAll(() => {
    mockClient = createMockClient();
    validatedClients = createValidatedEnergyClients(mockClient);
  });

  it('should reject invalid project registration', async () => {
    await expect(
      validatedClients.projects.registerProject({
        name: '',
        description: '',
        source: 'Solar',
        capacity_kw: -100, // Negative capacity
        location: { lat: 200, lng: 0 }, // Invalid latitude
      })
    ).rejects.toThrow(MycelixError);
  });

  it('should reject invalid participant registration', async () => {
    await expect(
      validatedClients.participants.register({
        type_: 'Prosumer',
        sources: [], // Empty sources
        capacity_kwh: -50, // Negative capacity
      })
    ).rejects.toThrow(MycelixError);
  });

  it('should reject invalid trade', async () => {
    await expect(
      validatedClients.trading.createTrade({
        buyer: 'not-a-did',
        amount_kwh: 0, // Zero amount
        source: 'Solar',
        price_per_kwh: -1, // Negative price
        currency: '',
      })
    ).rejects.toThrow(MycelixError);
  });
});

describeConductor('Energy Conductor Integration Tests', () => {
  it.todo('should integrate with Terra Atlas for project discovery');
  it.todo('should execute P2P energy trades with settlement');
  it.todo('should track regenerative exit progress');
  it.todo('should manage community ownership transitions');
});

// =============================================================================
// Bridge Zome Bidirectional Sync Tests
// =============================================================================

describe('Energy Bridge Bidirectional Sync (Mock)', () => {
  // Extended mock client with new bridge functions
  function createBridgeMockClient(): ZomeCallable {
    const syncedProjects = new Map<string, any>();
    const bridgeInvestments = new Map<string, any[]>();
    const productionRecords = new Map<string, any[]>();
    const pendingSyncs: any[] = [];
    const transitions = new Map<string, any>();
    const events: any[] = [];
    const milestones = new Map<string, any[]>();

    const agentId = `did:mycelix:${generateTestAgentId()}`;
    let idCounter = 0;

    return {
      async callZome({
        fn_name,
        payload,
      }: {
        role_name: string;
        zome_name: string;
        fn_name: string;
        payload: unknown;
      }) {
        const now = Date.now();

        switch (fn_name) {
          // Project sync from Terra Atlas
          case 'sync_terra_atlas_project': {
            const input = payload as {
              terra_atlas_id: string;
              name: string;
              project_type: string;
              location: { latitude: number; longitude: number; region: string; country: string };
              capacity_mw: number;
              total_investment: number;
              current_investment: number;
              status: string;
            };
            const project = {
              id: `project:${input.terra_atlas_id}:${now}`,
              terra_atlas_id: input.terra_atlas_id,
              name: input.name,
              project_type: input.project_type,
              location: input.location,
              capacity_mw: input.capacity_mw,
              total_investment: input.total_investment,
              current_investment: input.current_investment,
              status: input.status,
              regenerative_progress: 0.0,
              synced_at: now,
            };
            syncedProjects.set(input.terra_atlas_id, project);

            events.push({
              id: `event:ProjectDiscovered:${now}`,
              event_type: 'ProjectDiscovered',
              project_id: input.terra_atlas_id,
              payload: '{}',
              source: 'mycelix-energy',
              timestamp: now,
            });

            return {
              signed_action: { hashed: { hash: project.id, content: {} }, signature: 'sig' },
              entry: { Present: project },
            };
          }

          // Record investment (cross-hApp)
          case 'record_investment': {
            const input = payload as {
              project_id: string;
              investor_did: string;
              amount: number;
              currency: string;
              source_happ: string;
              investment_type: string;
            };
            const investment = {
              id: `invest:${input.project_id}:${input.investor_did}:${now}:${++idCounter}`,
              project_id: input.project_id,
              investor_did: input.investor_did,
              amount: input.amount,
              currency: input.currency,
              source_happ: input.source_happ,
              investment_type: input.investment_type,
              created_at: now,
            };
            const existing = bridgeInvestments.get(input.project_id) || [];
            existing.push(investment);
            bridgeInvestments.set(input.project_id, existing);

            events.push({
              id: `event:InvestmentReceived:${now}`,
              event_type: 'InvestmentReceived',
              project_id: input.project_id,
              payload: JSON.stringify({ amount: input.amount, investor: input.investor_did }),
              source: 'mycelix-energy',
              timestamp: now,
            });

            return {
              signed_action: { hashed: { hash: investment.id, content: {} }, signature: 'sig' },
              entry: { Present: investment },
            };
          }

          // NEW: Record production metrics
          case 'record_production_update': {
            const input = payload as {
              project_id: string;
              terra_atlas_id: string;
              period_start: number;
              period_end: number;
              energy_generated_mwh: number;
              capacity_factor: number;
              revenue_generated: number;
              currency: string;
              grid_injection_mwh: number;
              self_consumption_mwh: number;
              verified_by?: string;
            };

            // Validation
            if (input.energy_generated_mwh < 0) {
              throw new Error('Energy generated must be non-negative');
            }
            if (input.capacity_factor < 0 || input.capacity_factor > 1) {
              throw new Error('Capacity factor must be 0.0-1.0');
            }

            const record = {
              id: `prod:${input.project_id}:${input.period_start}:${now}`,
              project_id: input.project_id,
              terra_atlas_id: input.terra_atlas_id,
              period_start: input.period_start,
              period_end: input.period_end,
              energy_generated_mwh: input.energy_generated_mwh,
              capacity_factor: input.capacity_factor,
              revenue_generated: input.revenue_generated,
              currency: input.currency,
              grid_injection_mwh: input.grid_injection_mwh,
              self_consumption_mwh: input.self_consumption_mwh,
              recorded_at: now,
              verified_by: input.verified_by,
            };

            const existing = productionRecords.get(input.project_id) || [];
            existing.push(record);
            productionRecords.set(input.project_id, existing);

            // Queue for sync
            pendingSyncs.push({
              id: `sync:ProductionMetrics:${input.project_id}:${now}`,
              sync_type: 'ProductionMetrics',
              target_system: 'terra-atlas',
              payload: JSON.stringify({
                terra_atlas_id: input.terra_atlas_id,
                energy_generated_mwh: input.energy_generated_mwh,
                capacity_factor: input.capacity_factor,
                revenue_generated: input.revenue_generated,
              }),
              created_at: now,
              synced_at: null,
              retry_count: 0,
              last_error: null,
            });

            events.push({
              id: `event:ProductionUpdate:${now}`,
              event_type: 'ProductionUpdate',
              project_id: input.project_id,
              payload: JSON.stringify({
                energy_mwh: input.energy_generated_mwh,
                capacity_factor: input.capacity_factor,
                revenue: input.revenue_generated,
              }),
              source: 'mycelix-energy',
              timestamp: now,
            });

            return {
              signed_action: { hashed: { hash: record.id, content: {} }, signature: 'sig' },
              entry: { Present: record },
            };
          }

          // NEW: Get project production history
          case 'get_project_production_history': {
            const projectId = payload as string;
            const records = productionRecords.get(projectId) || [];
            return records.map(r => ({
              signed_action: { hashed: { hash: r.id, content: {} }, signature: 'sig' },
              entry: { Present: r },
            }));
          }

          // NEW: Get investment totals
          case 'get_project_investment_total': {
            const projectId = payload as string;
            const investments = bridgeInvestments.get(projectId) || [];
            const investors = new Set(investments.map(i => i.investor_did));
            const byType = new Map<string, number>();
            let total = 0;

            for (const inv of investments) {
              total += inv.amount;
              byType.set(inv.investment_type, (byType.get(inv.investment_type) || 0) + inv.amount);
            }

            return {
              project_id: projectId,
              total_amount: total,
              investor_count: investors.size,
              by_type: Array.from(byType.entries()),
            };
          }

          // NEW: Queue sync to Terra Atlas
          case 'queue_sync_to_terra_atlas': {
            const input = payload as { sync_type: string; project_id: string; payload: string };
            const record = {
              id: `sync:${input.sync_type}:${input.project_id}:${now}`,
              sync_type: input.sync_type,
              target_system: 'terra-atlas',
              payload: input.payload,
              created_at: now,
              synced_at: null,
              retry_count: 0,
              last_error: null,
            };
            pendingSyncs.push(record);
            return {
              signed_action: { hashed: { hash: record.id, content: {} }, signature: 'sig' },
              entry: { Present: record },
            };
          }

          // NEW: Get pending syncs
          case 'get_pending_syncs': {
            return pendingSyncs
              .filter(s => s.synced_at === null)
              .map(s => ({
                signed_action: { hashed: { hash: s.id, content: {} }, signature: 'sig' },
                entry: { Present: s },
              }));
          }

          // NEW: Mark sync complete
          case 'mark_sync_complete': {
            const input = payload as { sync_id: string; success: boolean; error?: string };
            const sync = pendingSyncs.find(s => s.id === input.sync_id);
            if (sync) {
              if (input.success) {
                sync.synced_at = now;
              } else {
                sync.retry_count++;
                sync.last_error = input.error;
              }
            }
            return true;
          }

          // Record milestone
          case 'record_milestone': {
            const input = payload as {
              project_id: string;
              milestone_type: string;
              community_readiness: number;
              operator_certification: boolean;
              financial_sustainability: number;
              verified_by?: string;
            };
            const milestone = {
              id: `milestone:${input.project_id}:${input.milestone_type}:${now}`,
              project_id: input.project_id,
              milestone_type: input.milestone_type,
              community_readiness: input.community_readiness,
              operator_certification: input.operator_certification,
              financial_sustainability: input.financial_sustainability,
              achieved_at: now,
              verified_by: input.verified_by,
            };
            const existing = milestones.get(input.project_id) || [];
            existing.push(milestone);
            milestones.set(input.project_id, existing);

            events.push({
              id: `event:MilestoneAchieved:${now}`,
              event_type: 'MilestoneAchieved',
              project_id: input.project_id,
              payload: JSON.stringify({
                milestone_type: input.milestone_type,
                community_readiness: input.community_readiness,
              }),
              source: 'mycelix-energy',
              timestamp: now,
            });

            return {
              signed_action: { hashed: { hash: milestone.id, content: {} }, signature: 'sig' },
              entry: { Present: milestone },
            };
          }

          // Get project milestones
          case 'get_project_milestones': {
            const projectId = payload as string;
            const ms = milestones.get(projectId) || [];
            return ms.map(m => ({
              signed_action: { hashed: { hash: m.id, content: {} }, signature: 'sig' },
              entry: { Present: m },
            }));
          }

          // NEW: Initiate regenerative transition
          case 'initiate_transition': {
            const input = payload as {
              project_id: string;
              terra_atlas_id: string;
              community_did: string;
              current_community_ownership: number;
              target_ownership_pct: number;
              reserve_account_balance: number;
              conditions_met: string[];
              conditions_pending: string[];
            };

            // Validation
            if (input.target_ownership_pct <= input.current_community_ownership) {
              throw new Error('Transition must increase community ownership');
            }
            if (!input.community_did.startsWith('did:mycelix:')) {
              throw new Error('Community must have valid DID');
            }

            const transition = {
              id: `transition:${input.project_id}:${now}`,
              project_id: input.project_id,
              terra_atlas_id: input.terra_atlas_id,
              from_ownership_pct: input.current_community_ownership,
              to_ownership_pct: input.target_ownership_pct,
              community_did: input.community_did,
              reserve_account_balance: input.reserve_account_balance,
              conditions_met: input.conditions_met,
              conditions_pending: input.conditions_pending,
              status: 'Proposed',
              initiated_at: now,
              completed_at: null,
            };
            transitions.set(transition.id, transition);

            pendingSyncs.push({
              id: `sync:TransitionProgress:${input.project_id}:${now}`,
              sync_type: 'TransitionProgress',
              target_system: 'terra-atlas',
              payload: JSON.stringify({
                terra_atlas_id: input.terra_atlas_id,
                current_ownership: input.current_community_ownership,
                target_ownership: input.target_ownership_pct,
                status: 'Proposed',
              }),
              created_at: now,
              synced_at: null,
              retry_count: 0,
              last_error: null,
            });

            events.push({
              id: `event:TransitionInitiated:${now}`,
              event_type: 'TransitionInitiated',
              project_id: input.project_id,
              payload: JSON.stringify({
                from_pct: input.current_community_ownership,
                to_pct: input.target_ownership_pct,
                community: input.community_did,
              }),
              source: 'mycelix-energy',
              timestamp: now,
            });

            return {
              signed_action: { hashed: { hash: transition.id, content: {} }, signature: 'sig' },
              entry: { Present: transition },
            };
          }

          // NEW: Complete transition
          case 'complete_transition': {
            const input = payload as {
              transition_id: string;
              project_id: string;
              terra_atlas_id: string;
              community_did: string;
              final_ownership_pct: number;
            };
            const transition = transitions.get(input.transition_id);
            if (transition) {
              transition.status = 'Completed';
              transition.completed_at = now;
            }

            pendingSyncs.push({
              id: `sync:TransitionProgress:${input.project_id}:${now}`,
              sync_type: 'TransitionProgress',
              target_system: 'terra-atlas',
              payload: JSON.stringify({
                terra_atlas_id: input.terra_atlas_id,
                final_ownership: input.final_ownership_pct,
                status: 'Completed',
              }),
              created_at: now,
              synced_at: null,
              retry_count: 0,
              last_error: null,
            });

            events.push({
              id: `event:CommunityOwnershipComplete:${now}`,
              event_type: 'CommunityOwnershipComplete',
              project_id: input.project_id,
              payload: JSON.stringify({
                final_ownership: input.final_ownership_pct,
                community: input.community_did,
              }),
              source: 'mycelix-energy',
              timestamp: now,
            });

            return {
              signed_action: { hashed: { hash: `event:${now}`, content: {} }, signature: 'sig' },
              entry: { Present: events[events.length - 1] },
            };
          }

          // Get active transitions
          case 'get_active_transitions': {
            return Array.from(transitions.values())
              .filter(t => t.status !== 'Completed' && t.status !== 'Cancelled')
              .map(t => ({
                signed_action: { hashed: { hash: t.id, content: {} }, signature: 'sig' },
                entry: { Present: t },
              }));
          }

          // Get recent events
          case 'get_recent_events': {
            const limit = (payload as number) || 10;
            return events
              .slice(-limit)
              .reverse()
              .map(e => ({
                signed_action: { hashed: { hash: e.id, content: {} }, signature: 'sig' },
                entry: { Present: e },
              }));
          }

          // Get all projects
          case 'get_all_projects': {
            return Array.from(syncedProjects.values()).map(p => ({
              signed_action: { hashed: { hash: p.id, content: {} }, signature: 'sig' },
              entry: { Present: p },
            }));
          }

          default:
            throw new Error(`Unknown function: ${fn_name}`);
        }
      },
    };
  }

  let bridgeClient: ZomeCallable;

  beforeAll(() => {
    bridgeClient = createBridgeMockClient();
  });

  describe('Terra Atlas Project Sync', () => {
    it('should sync a project from Terra Atlas', async () => {
      const result = await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'sync_terra_atlas_project',
        payload: {
          terra_atlas_id: 'ta-solar-001',
          name: 'Sunrise Solar Farm',
          project_type: 'Solar',
          location: { latitude: 35.6762, longitude: 139.6503, region: 'Kanto', country: 'Japan' },
          capacity_mw: 50.0,
          total_investment: 10000000,
          current_investment: 2500000,
          status: 'Funding',
        },
      });

      const project = result.entry.Present;
      expect(project.terra_atlas_id).toBe('ta-solar-001');
      expect(project.name).toBe('Sunrise Solar Farm');
      expect(project.capacity_mw).toBe(50.0);
      expect(project.regenerative_progress).toBe(0.0);
    });

    it('should list all synced projects', async () => {
      const result = await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'get_all_projects',
        payload: null,
      });

      expect(result.length).toBeGreaterThan(0);
    });
  });

  describe('Production Metrics (Bidirectional Sync)', () => {
    it('should record production metrics', async () => {
      // First sync a project
      await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'sync_terra_atlas_project',
        payload: {
          terra_atlas_id: 'ta-wind-001',
          name: 'Coastal Wind',
          project_type: 'Wind',
          location: { latitude: 51.5, longitude: -0.1, region: 'South East', country: 'UK' },
          capacity_mw: 100.0,
          total_investment: 25000000,
          current_investment: 25000000,
          status: 'Operational',
        },
      });

      // Record monthly production
      const result = await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'record_production_update',
        payload: {
          project_id: 'ta-wind-001',
          terra_atlas_id: 'ta-wind-001',
          period_start: Date.now() - 30 * 24 * 60 * 60 * 1000,
          period_end: Date.now(),
          energy_generated_mwh: 15000.5,
          capacity_factor: 0.35,
          revenue_generated: 750000,
          currency: 'GBP',
          grid_injection_mwh: 14500.0,
          self_consumption_mwh: 500.5,
          verified_by: 'did:mycelix:grid-operator-uk',
        },
      });

      const record = result.entry.Present;
      expect(record.energy_generated_mwh).toBe(15000.5);
      expect(record.capacity_factor).toBe(0.35);
      expect(record.verified_by).toBe('did:mycelix:grid-operator-uk');
    });

    it('should reject invalid production metrics', async () => {
      await expect(
        bridgeClient.callZome({
          role_name: 'energy',
          zome_name: 'energy_bridge',
          fn_name: 'record_production_update',
          payload: {
            project_id: 'ta-wind-001',
            terra_atlas_id: 'ta-wind-001',
            period_start: Date.now(),
            period_end: Date.now(),
            energy_generated_mwh: -100, // Invalid: negative
            capacity_factor: 0.35,
            revenue_generated: 0,
            currency: 'GBP',
            grid_injection_mwh: 0,
            self_consumption_mwh: 0,
          },
        })
      ).rejects.toThrow('Energy generated must be non-negative');
    });

    it('should get production history', async () => {
      const result = await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'get_project_production_history',
        payload: 'ta-wind-001',
      });

      expect(result.length).toBeGreaterThan(0);
      expect(result[0].entry.Present.project_id).toBe('ta-wind-001');
    });

    it('should queue production metrics for Terra Atlas sync', async () => {
      const pendingSyncs = await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'get_pending_syncs',
        payload: null,
      });

      const productionSyncs = pendingSyncs.filter(
        (s: any) => s.entry.Present.sync_type === 'ProductionMetrics'
      );
      expect(productionSyncs.length).toBeGreaterThan(0);
    });
  });

  describe('Investment Aggregation', () => {
    it('should aggregate investments correctly', async () => {
      // Record multiple investments
      await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'record_investment',
        payload: {
          project_id: 'ta-solar-001',
          investor_did: 'did:mycelix:investor1',
          amount: 50000,
          currency: 'USD',
          source_happ: 'mycelix-finance',
          investment_type: 'Equity',
        },
      });

      await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'record_investment',
        payload: {
          project_id: 'ta-solar-001',
          investor_did: 'did:mycelix:investor2',
          amount: 25000,
          currency: 'USD',
          source_happ: 'mycelix-finance',
          investment_type: 'CommunityShare',
        },
      });

      const summary = await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'get_project_investment_total',
        payload: 'ta-solar-001',
      });

      expect(summary.total_amount).toBe(75000);
      expect(summary.investor_count).toBe(2);
      expect(summary.by_type.length).toBe(2);
    });
  });

  describe('Sync Queue Management', () => {
    it('should mark sync as complete', async () => {
      // Get pending syncs
      const before = await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'get_pending_syncs',
        payload: null,
      });

      if (before.length > 0) {
        const syncId = before[0].entry.Present.id;

        // Mark as complete
        await bridgeClient.callZome({
          role_name: 'energy',
          zome_name: 'energy_bridge',
          fn_name: 'mark_sync_complete',
          payload: { sync_id: syncId, success: true },
        });

        // Verify it's no longer pending
        const after = await bridgeClient.callZome({
          role_name: 'energy',
          zome_name: 'energy_bridge',
          fn_name: 'get_pending_syncs',
          payload: null,
        });

        expect(after.length).toBeLessThan(before.length);
      }
    });

    it('should track failed syncs with retry count', async () => {
      // Queue a new sync
      await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'queue_sync_to_terra_atlas',
        payload: {
          sync_type: 'StatusChange',
          project_id: 'ta-test-001',
          payload: JSON.stringify({ status: 'Operational' }),
        },
      });

      const pending = await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'get_pending_syncs',
        payload: null,
      });

      const testSync = pending.find(
        (s: any) => s.entry.Present.project_id === 'ta-test-001'
      );

      if (testSync) {
        // Mark as failed
        await bridgeClient.callZome({
          role_name: 'energy',
          zome_name: 'energy_bridge',
          fn_name: 'mark_sync_complete',
          payload: {
            sync_id: testSync.entry.Present.id,
            success: false,
            error: 'Connection timeout',
          },
        });
      }
    });
  });

  describe('Regenerative Transition (Luminous Chimera Model)', () => {
    it('should initiate a regenerative transition', async () => {
      const result = await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'initiate_transition',
        payload: {
          project_id: 'ta-wind-001',
          terra_atlas_id: 'ta-wind-001',
          community_did: 'did:mycelix:coastal-community',
          current_community_ownership: 10.0,
          target_ownership_pct: 51.0,
          reserve_account_balance: 250000,
          conditions_met: ['CommunityFormation', 'OperatorTraining'],
          conditions_pending: ['FinancialIndependence', 'GovernanceEstablished'],
        },
      });

      const transition = result.entry.Present;
      expect(transition.status).toBe('Proposed');
      expect(transition.from_ownership_pct).toBe(10.0);
      expect(transition.to_ownership_pct).toBe(51.0);
      expect(transition.conditions_met).toContain('CommunityFormation');
    });

    it('should reject invalid transition (no ownership increase)', async () => {
      await expect(
        bridgeClient.callZome({
          role_name: 'energy',
          zome_name: 'energy_bridge',
          fn_name: 'initiate_transition',
          payload: {
            project_id: 'ta-wind-001',
            terra_atlas_id: 'ta-wind-001',
            community_did: 'did:mycelix:bad-community',
            current_community_ownership: 51.0,
            target_ownership_pct: 30.0, // Invalid: decreasing
            reserve_account_balance: 0,
            conditions_met: [],
            conditions_pending: [],
          },
        })
      ).rejects.toThrow('Transition must increase community ownership');
    });

    it('should complete a regenerative transition', async () => {
      // Get active transitions
      const active = await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'get_active_transitions',
        payload: null,
      });

      expect(active.length).toBeGreaterThan(0);

      const transition = active[0].entry.Present;

      // Complete the transition
      await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'complete_transition',
        payload: {
          transition_id: transition.id,
          project_id: transition.project_id,
          terra_atlas_id: transition.terra_atlas_id,
          community_did: transition.community_did,
          final_ownership_pct: transition.to_ownership_pct,
        },
      });

      // Verify event was created
      const events = await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'get_recent_events',
        payload: 5,
      });

      const completionEvent = events.find(
        (e: any) => e.entry.Present.event_type === 'CommunityOwnershipComplete'
      );
      expect(completionEvent).toBeDefined();
    });
  });

  describe('Milestone Tracking', () => {
    it('should record regenerative milestones', async () => {
      const result = await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'record_milestone',
        payload: {
          project_id: 'ta-solar-001',
          milestone_type: 'CommunityFormation',
          community_readiness: 0.85,
          operator_certification: true,
          financial_sustainability: 0.6,
          verified_by: 'did:mycelix:auditor-001',
        },
      });

      const milestone = result.entry.Present;
      expect(milestone.milestone_type).toBe('CommunityFormation');
      expect(milestone.community_readiness).toBe(0.85);
    });

    it('should get project milestones', async () => {
      const result = await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'get_project_milestones',
        payload: 'ta-solar-001',
      });

      expect(result.length).toBeGreaterThan(0);
    });
  });

  describe('Event Broadcasting', () => {
    it('should broadcast events for all operations', async () => {
      const events = await bridgeClient.callZome({
        role_name: 'energy',
        zome_name: 'energy_bridge',
        fn_name: 'get_recent_events',
        payload: 20,
      });

      const eventTypes = events.map((e: any) => e.entry.Present.event_type);

      // Should have various event types from our tests
      expect(eventTypes).toContain('ProjectDiscovered');
      expect(eventTypes).toContain('InvestmentReceived');
      expect(eventTypes).toContain('ProductionUpdate');
      expect(eventTypes).toContain('MilestoneAchieved');
      expect(eventTypes).toContain('TransitionInitiated');
    });
  });
});
