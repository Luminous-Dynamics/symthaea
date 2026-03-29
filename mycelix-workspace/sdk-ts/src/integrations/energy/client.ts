// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Energy Client
 *
 * Unified client for the Energy hApp SDK, providing access to
 * projects, participants, trading, credits, investments, and grid operations.
 *
 * @module @mycelix/sdk/integrations/energy
 */

import {
  type AppClient,
  AppWebsocket,
} from '@holochain/client';

import { EnergySdkError } from './types';
import { CreditsClient } from './zomes/credits';
import { GridClient } from './zomes/grid';
import { InvestmentsClient } from './zomes/investments';
import { ParticipantsClient } from './zomes/participants';
import { ProjectsClient } from './zomes/projects';
import { TradingClient } from './zomes/trading';
import { RetryPolicy, RetryPolicies, type RetryOptions } from '../../common/retry';

import type {
  EnergyProject,
  EnergyParticipant,
  EnergyTrade,
  EnergyCredit,
  Investment,
  GridStatus,
  GridStatistics,
  CommunityEnergySummary,
  ParticipantStatistics,
  EnergySource,
  EnergyEvent,
} from './types';


/**
 * Configuration for the Energy client
 */
export interface EnergyClientConfig {
  /** Role ID for the energy DNA */
  roleId: string;
  /** Projects zome name */
  projectsZome: string;
  /** Participants zome name */
  participantsZome: string;
  /** Trading zome name */
  tradingZome: string;
  /** Credits zome name */
  creditsZome: string;
  /** Investments zome name */
  investmentsZome: string;
  /** Grid zome name */
  gridZome: string;
  /** Retry configuration for zome calls */
  retry?: RetryOptions | RetryPolicy;
}

const DEFAULT_CONFIG: EnergyClientConfig = {
  roleId: 'energy',
  projectsZome: 'projects',
  participantsZome: 'participants',
  tradingZome: 'trading',
  creditsZome: 'credits',
  investmentsZome: 'investments',
  gridZome: 'grid',
};

/**
 * Connection options for creating a new client
 */
export interface EnergyConnectionOptions {
  /** WebSocket URL to connect to */
  url: string;
  /** Optional timeout in milliseconds */
  timeout?: number;
  /** Retry configuration for connection and zome calls */
  retry?: RetryOptions | RetryPolicy;
}

/**
 * Unified Mycelix Energy Client
 *
 * Provides access to all energy functionality through a single interface.
 *
 * @example
 * ```typescript
 * import { MycelixEnergyClient } from '@mycelix/sdk/integrations/energy';
 *
 * // Connect to Holochain
 * const energy = await MycelixEnergyClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Register a solar project
 * const project = await energy.projects.registerProject({
 *   name: 'Community Solar Farm',
 *   description: 'A 500kW community-owned solar installation',
 *   source: 'Solar',
 *   capacity_kw: 500,
 *   location: { lat: 32.95, lng: -96.73 },
 *   investment_goal: 750000,
 * });
 *
 * // Register as a prosumer
 * const participant = await energy.participants.registerProsumer(
 *   ['Solar'],
 *   50,
 *   { lat: 32.95, lng: -96.73 }
 * );
 *
 * // Trade energy
 * const trade = await energy.trading.createTrade({
 *   buyer_did: 'did:mycelix:consumer',
 *   amount_kwh: 25,
 *   source: 'Solar',
 *   price_per_kwh: 0.12,
 *   currency: 'USD',
 * });
 *
 * // Check grid status
 * const gridStatus = await energy.grid.getGridStatus('ERCOT');
 * ```
 */
export class MycelixEnergyClient {
  /** Energy project operations */
  public readonly projects: ProjectsClient;

  /** Participant operations */
  public readonly participants: ParticipantsClient;

  /** Energy trading operations */
  public readonly trading: TradingClient;

  /** Energy credit operations */
  public readonly credits: CreditsClient;

  /** Investment operations */
  public readonly investments: InvestmentsClient;

  /** Grid management operations */
  public readonly grid: GridClient;

  private readonly config: EnergyClientConfig;

  /** Retry policy for zome calls */
  private readonly retryPolicy: RetryPolicy;

  /**
   * Create an energy client from an existing Holochain client
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   */
  constructor(
    private readonly client: AppClient,
    config: Partial<EnergyClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    // Set up retry policy
    if (config.retry instanceof RetryPolicy) {
      this.retryPolicy = config.retry;
    } else if (config.retry) {
      this.retryPolicy = new RetryPolicy(config.retry);
    } else {
      this.retryPolicy = RetryPolicies.standard;
    }

    // Initialize sub-clients
    this.projects = new ProjectsClient(client, {
      roleId: this.config.roleId,
      zomeName: this.config.projectsZome,
    });
    this.participants = new ParticipantsClient(client, {
      roleId: this.config.roleId,
      zomeName: this.config.participantsZome,
    });
    this.trading = new TradingClient(client, {
      roleId: this.config.roleId,
      zomeName: this.config.tradingZome,
    });
    this.credits = new CreditsClient(client, {
      roleId: this.config.roleId,
      zomeName: this.config.creditsZome,
    });
    this.investments = new InvestmentsClient(client, {
      roleId: this.config.roleId,
      zomeName: this.config.investmentsZome,
    });
    this.grid = new GridClient(client, {
      roleId: this.config.roleId,
      zomeName: this.config.gridZome,
    });
  }

  /**
   * Connect to Holochain and create an energy client
   *
   * @param options - Connection options
   * @returns Connected energy client
   *
   * @example
   * ```typescript
   * const energy = await MycelixEnergyClient.connect({
   *   url: 'ws://localhost:8888',
   *   timeout: 30000,
   * });
   * ```
   */
  static async connect(
    options: EnergyConnectionOptions
  ): Promise<MycelixEnergyClient> {
    // Set up retry for connection
    const retryPolicy = options.retry instanceof RetryPolicy
      ? options.retry
      : options.retry
        ? new RetryPolicy(options.retry)
        : RetryPolicies.network;

    const connectFn = async () => {
      try {
        const client = await AppWebsocket.connect({
          url: new URL(options.url),
          ...((options as any).appClientArgs ?? {}),
        } as Parameters<typeof AppWebsocket.connect>[0]);

        return new MycelixEnergyClient(client, { retry: retryPolicy });
      } catch (error) {
        throw new EnergySdkError(
          'CONNECTION_ERROR',
          `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
          error
        );
      }
    };

    return retryPolicy.execute(connectFn);
  }

  /**
   * Create an energy client from an existing AppClient
   *
   * Use this when you already have a Holochain connection.
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   * @returns Energy client
   */
  static fromClient(
    client: AppClient,
    config: Partial<EnergyClientConfig> = {}
  ): MycelixEnergyClient {
    return new MycelixEnergyClient(client, config);
  }

  // ============================================================================
  // Cross-Module Integration
  // ============================================================================

  /**
   * Get comprehensive energy status for a DID
   *
   * @param did - DID to query
   * @returns Complete energy status
   */
  async getEnergyStatus(did: string): Promise<{
    participant: EnergyParticipant | null;
    statistics: ParticipantStatistics | null;
    creditBalance: number;
    activeInvestments: Investment[];
    activeTrades: EnergyTrade[];
  }> {
    const participant = await this.participants.getParticipantByDid(did);

    let statistics: ParticipantStatistics | null = null;
    if (participant) {
      statistics = await this.participants.getStatistics(participant.id);
    }

    const [creditBalance, investments, trades] = await Promise.all([
      this.credits.getCreditBalance(did),
      this.investments.getActiveInvestments(did),
      this.trading.getActiveTrades(did),
    ]);

    return {
      participant,
      statistics,
      creditBalance,
      activeInvestments: investments,
      activeTrades: trades,
    };
  }

  /**
   * Get comprehensive project details
   *
   * @param projectId - Project identifier
   * @returns Complete project information
   */
  async getProjectDetails(projectId: string): Promise<{
    project: EnergyProject | null;
    investors: Investment[];
    totalInvested: number;
    investorCount: number;
    statistics: {
      capacityFactor: number;
      estimatedAnnualGeneration: number;
      investmentProgress: number;
    };
  } | null> {
    const project = await this.projects.getProject(projectId);
    if (!project) {
      return null;
    }

    const investors = await this.investments.getInvestmentsByProject(projectId);
    const totalInvested = investors.reduce((sum, i) => sum + i.amount, 0);
    const investorCount = new Set(investors.map(i => i.investor_did)).size;

    return {
      project,
      investors,
      totalInvested,
      investorCount,
      statistics: {
        capacityFactor: this.projects.calculateCapacityFactor(project),
        estimatedAnnualGeneration: this.projects.estimateAnnualGeneration(project),
        investmentProgress: this.projects.getInvestmentProgress(project),
      },
    };
  }

  /**
   * Get grid overview
   *
   * @param region - Grid region
   * @param communityId - Optional community filter
   * @returns Grid overview
   */
  async getGridOverview(region: string, communityId?: string): Promise<{
    gridStatus: GridStatus;
    gridStatistics: GridStatistics;
    communityStatus: CommunityEnergySummary | null;
    openRequests: number;
    activeAlerts: number;
  }> {
    const [gridStatus, gridStatistics, alerts, requests] = await Promise.all([
      this.grid.getGridStatus(region),
      this.grid.getGridStatistics(),
      this.grid.getActiveAlerts(region),
      this.grid.getOpenBalanceRequests(),
    ]);

    let communityStatus: CommunityEnergySummary | null = null;
    if (communityId) {
      communityStatus = await this.grid.getCommunityEnergySummary(communityId);
    }

    return {
      gridStatus,
      gridStatistics,
      communityStatus,
      openRequests: requests.length,
      activeAlerts: alerts.length,
    };
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Quick energy sale with automatic credit transfer
   *
   * @param sellerDid - Seller's DID
   * @param buyerDid - Buyer's DID
   * @param amountKwh - Amount to sell
   * @param source - Energy source
   * @param pricePerKwh - Price per kWh
   * @param currency - Currency code
   * @returns The trade and transferred credit
   */
  async sellEnergy(
    _sellerDid: string,
    buyerDid: string,
    amountKwh: number,
    source: EnergySource,
    pricePerKwh: number,
    currency: string = 'USD'
  ): Promise<{
    trade: EnergyTrade;
    creditIssued: EnergyCredit;
  }> {
    // Create trade
    const trade = await this.trading.createTrade({
      buyer_did: buyerDid,
      amount_kwh: amountKwh,
      source,
      price_per_kwh: pricePerKwh,
      currency,
    });

    // Issue credit to buyer
    const creditIssued = await this.credits.issueCredit({
      holder_did: buyerDid,
      amount_kwh: amountKwh,
      source,
      transferable: true,
    });

    return { trade, creditIssued };
  }

  /**
   * Invest in a project and get estimated returns
   *
   * @param projectId - Project identifier
   * @param amount - Investment amount
   * @param currency - Currency code
   * @returns Investment with estimated returns
   */
  async investWithEstimate(
    projectId: string,
    amount: number,
    currency: string = 'USD'
  ): Promise<{
    investment: Investment;
    ownershipPercent: number;
    estimatedAnnualDividend: number;
  }> {
    const project = await this.projects.getProject(projectId);
    if (!project) {
      throw new EnergySdkError('NOT_FOUND', 'Project not found');
    }

    const investment = await this.investments.invest({
      project_id: projectId,
      amount,
      currency,
    });

    const ownershipPercent = this.investments.calculateOwnershipPercentage(amount, project);

    // Estimate based on typical solar economics
    const estimatedAnnualGeneration = this.projects.estimateAnnualGeneration(project);
    const averageEnergyPrice = 0.10; // USD/kWh
    const annualRevenue = estimatedAnnualGeneration * 1000 * averageEnergyPrice;
    const estimatedAnnualDividend = this.investments.calculateExpectedDividend(
      investment,
      annualRevenue,
      0.8
    );

    return {
      investment,
      ownershipPercent,
      estimatedAnnualDividend,
    };
  }

  /**
   * Get energy sources summary
   *
   * @returns Summary by energy source
   */
  async getSourcesSummary(): Promise<Record<EnergySource, {
    projectCount: number;
    totalCapacityKw: number;
    participantCount: number;
  }>> {
    const sources: EnergySource[] = ['Solar', 'Wind', 'Hydro', 'Nuclear', 'Geothermal', 'Storage', 'Biomass', 'Grid', 'Other'];
    const summary: Record<string, { projectCount: number; totalCapacityKw: number; participantCount: number }> = {};

    for (const source of sources) {
      const [projects, participants] = await Promise.all([
        this.projects.getProjectsBySource(source),
        this.participants.getParticipantsBySource(source),
      ]);

      summary[source] = {
        projectCount: projects.length,
        totalCapacityKw: projects.reduce((sum, p) => sum + p.capacity_kw, 0),
        participantCount: participants.length,
      };
    }

    return summary as Record<EnergySource, { projectCount: number; totalCapacityKw: number; participantCount: number }>;
  }

  /**
   * Get carbon offset summary for a DID
   *
   * @param did - DID to query
   * @returns Carbon offset summary
   */
  async getCarbonOffsetSummary(did: string): Promise<{
    totalOffsetKg: number;
    fromCredits: number;
    fromCertificates: number;
    equivalentTreesPlanted: number;
    equivalentMilesDriven: number;
  }> {
    const [creditSummary, certificates] = await Promise.all([
      this.credits.getCreditSummary(did),
      this.credits.getCarbonCertificates(did),
    ]);

    const fromCertificates = certificates.reduce((sum, c) => sum + c.amount_kg_co2, 0);
    const totalOffsetKg = creditSummary.carbonOffsetKg + fromCertificates;

    // Conversions
    const equivalentTreesPlanted = totalOffsetKg / 21; // ~21 kg CO2 per tree per year
    const equivalentMilesDriven = totalOffsetKg / 0.4; // ~0.4 kg CO2 per mile

    return {
      totalOffsetKg,
      fromCredits: creditSummary.carbonOffsetKg,
      fromCertificates,
      equivalentTreesPlanted: Math.round(equivalentTreesPlanted),
      equivalentMilesDriven: Math.round(equivalentMilesDriven),
    };
  }

  /**
   * Broadcast an energy event
   *
   * @param eventType - Event type
   * @param payload - Event payload
   * @returns The created event
   */
  async broadcastEvent(
    eventType: EnergyEvent['event_type'],
    payload: Record<string, unknown>
  ): Promise<EnergyEvent> {
    return this.callZome<EnergyEvent>('broadcast_event', {
      event_type: eventType,
      payload,
      timestamp: Date.now() * 1000,
    });
  }

  /**
   * Get recent energy events
   *
   * @param limit - Maximum number of events
   * @returns Array of events
   */
  async getRecentEvents(limit: number = 50): Promise<EnergyEvent[]> {
    return this.callZome<EnergyEvent[]>('get_recent_events', limit);
  }

  /**
   * Get the underlying Holochain client
   */
  getClient(): AppClient {
    return this.client;
  }

  /**
   * Helper to call zome functions with automatic retry
   */
  private async callZome<T>(fnName: string, payload: unknown): Promise<T> {
    return this.retryPolicy.execute(async () => {
      try {
        const result = await this.client.callZome({
          role_name: this.config.roleId,
          zome_name: 'energy',
          fn_name: fnName,
          payload,
        });
        return result as T;
      } catch (error) {
        throw new EnergySdkError(
          'ZOME_ERROR',
          `Failed to call ${fnName}: ${error instanceof Error ? error.message : String(error)}`,
          error
        );
      }
    });
  }

  /**
   * Get the current retry policy
   */
  getRetryPolicy(): RetryPolicy {
    return this.retryPolicy;
  }

  /**
   * Create a new client with a different retry policy
   *
   * @param retry - New retry configuration
   * @returns New client instance with updated retry policy
   */
  withRetry(retry: RetryOptions | RetryPolicy): MycelixEnergyClient {
    return new MycelixEnergyClient(this.client, { ...this.config, retry });
  }
}
