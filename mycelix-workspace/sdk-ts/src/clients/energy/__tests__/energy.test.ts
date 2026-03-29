// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Energy Client Tests
 *
 * Verifies zome call arguments and response pass-through for
 * ProjectsClient, GridClient, InvestmentsClient, RegenerativeClient, and BridgeClient.
 *
 * Energy clients use ZomeCallable interface (not ZomeClient base class)
 * and return HolochainRecord<T> directly (no extractEntry).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  ProjectsClient,
  GridClient,
  InvestmentsClient,
  RegenerativeClient,
  BridgeClient,
  type ZomeCallable,
} from '../index';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockCallable(): ZomeCallable {
  return {
    callZome: vi.fn(),
  } as unknown as ZomeCallable;
}

function mockRecord(entry: Record<string, unknown>) {
  return {
    entry: { Present: entry },
    signed_action: { hashed: { hash: 'mock-hash', content: {} }, signature: 'mock-sig' },
  };
}

// ============================================================================
// MOCK ENTRIES (snake_case, matching Rust serde output)
// ============================================================================

const PROJECT_ENTRY = {
  id: 'proj-001',
  terra_atlas_id: 'ta-solar-001',
  name: 'Richardson Solar Farm',
  description: 'Community solar installation',
  project_type: 'Solar',
  location: { latitude: 32.948, longitude: -96.729, country: 'US', region: 'Texas' },
  capacity_mw: 5.0,
  status: 'Operational',
  developer_did: 'did:mycelix:developer',
  community_did: 'did:mycelix:community',
  financials: {
    total_cost: 8000000,
    funded_amount: 6000000,
    currency: 'USD',
    target_irr: 0.08,
    payback_years: 12,
    annual_revenue_estimate: 750000,
  },
  created: 1708200000,
  updated: 1708200000,
};

const MILESTONE_ENTRY = {
  id: 'mile-001',
  project_id: 'proj-001',
  name: 'Grid connection complete',
  description: 'Successfully connected to local grid',
  target_date: 1710000000,
  completed_date: 1709500000,
  verification_evidence: 'Utility agreement signed',
};

const PRODUCTION_ENTRY = {
  id: 'prod-001',
  producer_did: 'did:mycelix:producer',
  project_id: 'proj-001',
  amount_kwh: 1200,
  timestamp: 1708200000,
  period_hours: 24,
  meter_reading: 45678,
  verified: true,
};

const OFFER_ENTRY = {
  id: 'offer-001',
  seller_did: 'did:mycelix:producer',
  project_id: 'proj-001',
  amount_kwh: 500,
  price_per_kwh: 0.12,
  currency: 'USD',
  available_from: 1708200000,
  available_until: 1708300000,
  status: 'Active',
  created: 1708200000,
};

const TRADE_ENTRY = {
  id: 'trade-001',
  offer_id: 'offer-001',
  seller_did: 'did:mycelix:producer',
  buyer_did: 'did:mycelix:consumer',
  amount_kwh: 200,
  price_per_kwh: 0.12,
  total_price: 24.0,
  currency: 'USD',
  executed: 1708200000,
  settled: false,
};

const INVESTMENT_ENTRY = {
  id: 'inv-001',
  project_id: 'proj-001',
  investor_did: 'did:mycelix:investor',
  amount: 50000,
  currency: 'USD',
  shares: 500,
  share_percentage: 0.625,
  investment_type: 'Equity',
  status: 'Confirmed',
  pledged: 1708100000,
  confirmed: 1708200000,
};

const DIVIDEND_ENTRY = {
  id: 'div-001',
  project_id: 'proj-001',
  investor_did: 'did:mycelix:investor',
  amount: 2500,
  currency: 'USD',
  period_start: 1706745600,
  period_end: 1709424000,
  distributed: 1709500000,
};

const CONTRACT_ENTRY = {
  id: 'contract-001',
  project_id: 'proj-001',
  community_did: 'did:mycelix:community',
  conditions: [
    {
      condition_type: 'CommunityReadiness',
      threshold: 0.8,
      current_value: 0.65,
      weight: 0.3,
      satisfied: false,
    },
  ],
  current_ownership_percentage: 0.25,
  target_ownership_percentage: 1.0,
  reserve_account_balance: 200000,
  currency: 'USD',
  status: 'Active',
  created: 1708200000,
  last_assessment: 1708200000,
};

const ASSESSMENT_ENTRY = {
  id: 'assess-001',
  contract_id: 'contract-001',
  assessor_did: 'did:mycelix:assessor',
  scores: [{ condition_type: 'CommunityReadiness', score: 0.7, evidence: 'Training complete' }],
  overall_readiness: 0.65,
  recommendations: ['Complete financial audit'],
  assessed: 1708200000,
};

const TERRA_PROJECT_ENTRY = {
  id: 'ta-001',
  terra_atlas_id: 'ta-solar-001',
  name: 'Richardson Solar',
  project_type: 'Solar',
  location: { latitude: 32.948, longitude: -96.729, region: 'Texas', country: 'US' },
  capacity_mw: 5.0,
  total_investment: 8000000,
  current_investment: 6000000,
  status: 'Operational',
  regenerative_progress: 0.25,
  synced_at: 1708200000,
};

const BRIDGE_EVENT_ENTRY = {
  id: 'evt-001',
  event_type: 'ProductionUpdate',
  project_id: 'proj-001',
  payload: '{"kwh": 1200}',
  source: 'grid-monitor',
  timestamp: 1708200000,
};

// ============================================================================
// PROJECTS CLIENT TESTS
// ============================================================================

describe('ProjectsClient', () => {
  let mockCallable: ZomeCallable;
  let projects: ProjectsClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    projects = new ProjectsClient(mockCallable);
  });

  it('registerProject calls correct zome', async () => {
    const record = mockRecord(PROJECT_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      terra_atlas_id: 'ta-solar-001',
      name: 'Richardson Solar Farm',
      description: 'Community solar installation',
      project_type: 'Solar' as const,
      location: { latitude: 32.948, longitude: -96.729, country: 'US', region: 'Texas' },
      capacity_mw: 5.0,
      developer_did: 'did:mycelix:developer',
      financials: PROJECT_ENTRY.financials,
    };
    const result = await projects.registerProject(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'projects',
      fn_name: 'register_project',
      payload: input,
    });
    expect(result.entry.Present.name).toBe('Richardson Solar Farm');
    expect(result.entry.Present.capacity_mw).toBe(5.0);
  });

  it('getDeveloperProjects returns project list', async () => {
    const record = mockRecord(PROJECT_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([record]);

    const result = await projects.getDeveloperProjects('did:mycelix:developer');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'projects',
      fn_name: 'get_developer_projects',
      payload: 'did:mycelix:developer',
    });
    expect(result).toHaveLength(1);
  });

  it('addMilestone returns milestone record', async () => {
    const record = mockRecord(MILESTONE_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      project_id: 'proj-001',
      name: 'Grid connection complete',
      description: 'Successfully connected to local grid',
      target_date: 1710000000,
    };
    const result = await projects.addMilestone(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'projects',
      fn_name: 'add_milestone',
      payload: input,
    });
    expect(result.entry.Present.name).toBe('Grid connection complete');
  });
});

// ============================================================================
// GRID CLIENT TESTS
// ============================================================================

describe('GridClient', () => {
  let mockCallable: ZomeCallable;
  let grid: GridClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    grid = new GridClient(mockCallable);
  });

  it('recordProduction calls correct zome', async () => {
    const record = mockRecord(PRODUCTION_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      producer_did: 'did:mycelix:producer',
      project_id: 'proj-001',
      amount_kwh: 1200,
      period_hours: 24,
      meter_reading: 45678,
    };
    const result = await grid.recordProduction(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'grid',
      fn_name: 'record_production',
      payload: input,
    });
    expect(result.entry.Present.amount_kwh).toBe(1200);
    expect(result.entry.Present.verified).toBe(true);
  });

  it('createTradeOffer returns offer record', async () => {
    const record = mockRecord(OFFER_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      seller_did: 'did:mycelix:producer',
      project_id: 'proj-001',
      amount_kwh: 500,
      price_per_kwh: 0.12,
      currency: 'USD',
      available_from: 1708200000,
      available_until: 1708300000,
    };
    const result = await grid.createTradeOffer(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'grid',
      fn_name: 'create_trade_offer',
      payload: input,
    });
    expect(result.entry.Present.status).toBe('Active');
  });

  it('executeTrade returns trade record', async () => {
    const record = mockRecord(TRADE_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      offer_id: 'offer-001',
      buyer_did: 'did:mycelix:consumer',
      amount_kwh: 200,
    };
    const result = await grid.executeTrade(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'grid',
      fn_name: 'execute_trade',
      payload: input,
    });
    expect(result.entry.Present.total_price).toBe(24.0);
    expect(result.entry.Present.settled).toBe(false);
  });

  it('getGridSummary returns scalar data', async () => {
    const summary = {
      active_offers: 15,
      total_kwh_available: 5000,
      total_trades: 42,
      total_kwh_traded: 12000,
      total_value_traded: 1440,
    };
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(summary);

    const result = await grid.getGridSummary();

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'grid',
      fn_name: 'get_grid_summary',
      payload: null,
    });
    expect(result.active_offers).toBe(15);
    expect(result.total_trades).toBe(42);
  });
});

// ============================================================================
// INVESTMENTS CLIENT TESTS
// ============================================================================

describe('InvestmentsClient', () => {
  let mockCallable: ZomeCallable;
  let investments: InvestmentsClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    investments = new InvestmentsClient(mockCallable);
  });

  it('pledgeInvestment calls correct zome', async () => {
    const record = mockRecord(INVESTMENT_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      project_id: 'proj-001',
      investor_did: 'did:mycelix:investor',
      amount: 50000,
      currency: 'USD',
      shares: 500,
      share_percentage: 0.625,
      investment_type: 'Equity' as const,
    };
    const result = await investments.pledgeInvestment(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'investments',
      fn_name: 'pledge_investment',
      payload: input,
    });
    expect(result.entry.Present.amount).toBe(50000);
    expect(result.entry.Present.status).toBe('Confirmed');
  });

  it('getInvestorPortfolio returns investment list', async () => {
    const record = mockRecord(INVESTMENT_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([record]);

    const result = await investments.getInvestorPortfolio('did:mycelix:investor');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'investments',
      fn_name: 'get_investor_portfolio',
      payload: 'did:mycelix:investor',
    });
    expect(result).toHaveLength(1);
  });

  it('distributeDividend returns dividend record', async () => {
    const record = mockRecord(DIVIDEND_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      project_id: 'proj-001',
      investor_did: 'did:mycelix:investor',
      amount: 2500,
      currency: 'USD',
      period_start: 1706745600,
      period_end: 1709424000,
    };
    const result = await investments.distributeDividend(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'investments',
      fn_name: 'distribute_dividend',
      payload: input,
    });
    expect(result.entry.Present.amount).toBe(2500);
  });

  it('getProjectTotalInvestment returns scalar data', async () => {
    const total = {
      project_id: 'proj-001',
      total_amount: 6000000,
      total_shares: 60000,
      investor_count: 120,
    };
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(total);

    const result = await investments.getProjectTotalInvestment('proj-001');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'investments',
      fn_name: 'get_project_total_investment',
      payload: 'proj-001',
    });
    expect(result.total_amount).toBe(6000000);
    expect(result.investor_count).toBe(120);
  });
});

// ============================================================================
// REGENERATIVE CLIENT TESTS
// ============================================================================

describe('RegenerativeClient', () => {
  let mockCallable: ZomeCallable;
  let regenerative: RegenerativeClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    regenerative = new RegenerativeClient(mockCallable);
  });

  it('createRegenerativeContract calls correct zome', async () => {
    const record = mockRecord(CONTRACT_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      project_id: 'proj-001',
      community_did: 'did:mycelix:community',
      conditions: CONTRACT_ENTRY.conditions,
      target_ownership_percentage: 1.0,
      currency: 'USD',
    };
    const result = await regenerative.createRegenerativeContract(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'regenerative',
      fn_name: 'create_regenerative_contract',
      payload: input,
    });
    expect(result.entry.Present.status).toBe('Active');
    expect(result.entry.Present.current_ownership_percentage).toBe(0.25);
  });

  it('submitReadinessAssessment returns assessment', async () => {
    const record = mockRecord(ASSESSMENT_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      contract_id: 'contract-001',
      assessor_did: 'did:mycelix:assessor',
      scores: [{ condition_type: 'CommunityReadiness' as const, score: 0.7, evidence: 'Training complete' }],
      recommendations: ['Complete financial audit'],
    };
    const result = await regenerative.submitReadinessAssessment(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'regenerative',
      fn_name: 'submit_readiness_assessment',
      payload: input,
    });
    expect(result.entry.Present.overall_readiness).toBe(0.65);
  });

  it('getReadinessProgress returns scalar data', async () => {
    const progress = {
      contract_id: 'contract-001',
      current_ownership_percentage: 0.25,
      target_ownership_percentage: 1.0,
      reserve_balance: 200000,
      overall_readiness: 0.65,
      conditions_met: 2,
      total_conditions: 5,
      status: 'Active',
    };
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(progress);

    const result = await regenerative.getReadinessProgress('contract-001');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'regenerative',
      fn_name: 'get_readiness_progress',
      payload: 'contract-001',
    });
    expect(result.conditions_met).toBe(2);
    expect(result.overall_readiness).toBe(0.65);
  });
});

// ============================================================================
// BRIDGE CLIENT TESTS
// ============================================================================

describe('BridgeClient', () => {
  let mockCallable: ZomeCallable;
  let bridge: BridgeClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    bridge = new BridgeClient(mockCallable);
  });

  it('syncTerraAtlasProject calls correct zome', async () => {
    const record = mockRecord(TERRA_PROJECT_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      terra_atlas_id: 'ta-solar-001',
      name: 'Richardson Solar',
      project_type: 'Solar' as const,
      location: { latitude: 32.948, longitude: -96.729, region: 'Texas', country: 'US' },
      capacity_mw: 5.0,
      total_investment: 8000000,
      current_investment: 6000000,
      status: 'Operational' as const,
    };
    const result = await bridge.syncTerraAtlasProject(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'bridge',
      fn_name: 'sync_terra_atlas_project',
      payload: input,
    });
    expect(result.entry.Present.terra_atlas_id).toBe('ta-solar-001');
    expect(result.entry.Present.regenerative_progress).toBe(0.25);
  });

  it('broadcastEnergyEvent returns event record', async () => {
    const record = mockRecord(BRIDGE_EVENT_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      event_type: 'ProductionUpdate' as const,
      project_id: 'proj-001',
      payload: '{"kwh": 1200}',
    };
    const result = await bridge.broadcastEnergyEvent(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'energy',
      zome_name: 'bridge',
      fn_name: 'broadcast_energy_event',
      payload: input,
    });
    expect(result.entry.Present.event_type).toBe('ProductionUpdate');
  });
});
