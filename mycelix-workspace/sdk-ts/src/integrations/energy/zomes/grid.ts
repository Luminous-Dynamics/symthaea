/**
 * Grid Zome Client
 *
 * Handles grid management, balance requests, and community energy.
 *
 * @module @mycelix/sdk/integrations/energy/zomes/grid
 */

import { EnergySdkError } from '../types';

import type {
  GridStatus,
  GridAlert,
  GridBalanceRequest,
  RequestGridBalanceInput,
  GridStatistics,
  CommunityEnergySummary,
  EnergyPriority,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Configuration for the Grid client
 */
export interface GridClientConfig {
  roleId: string;
  zomeName: string;
}

const DEFAULT_CONFIG: GridClientConfig = {
  roleId: 'energy',
  zomeName: 'grid',
};

/**
 * Client for grid management operations
 *
 * @example
 * ```typescript
 * const grid = new GridClient(holochainClient);
 *
 * // Get grid status
 * const status = await grid.getGridStatus('ERCOT');
 *
 * // Request grid balance
 * const request = await grid.requestBalance({
 *   type: 'supply',
 *   amount_kwh: 100,
 *   priority: 'standard',
 * });
 *
 * // Get community summary
 * const summary = await grid.getCommunityEnergySummary('my-community');
 * ```
 */
export class GridClient {
  private readonly config: GridClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<GridClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Call a zome function with error handling
   */
  private async call<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      const result = await this.client.callZome({
        role_name: this.config.roleId,
        zome_name: this.config.zomeName,
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
  }

  /**
   * Extract entry from a Holochain record
   */
  private extractEntry<T>(record: HolochainRecord): T {
    if (!record.entry || !('Present' in record.entry)) {
      throw new EnergySdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // Grid Status Operations
  // ============================================================================

  /**
   * Get grid status for a region
   *
   * @param region - Grid region identifier (e.g., 'ERCOT')
   * @returns Grid status
   */
  async getGridStatus(region: string): Promise<GridStatus> {
    return this.call<GridStatus>('get_grid_status', region);
  }

  /**
   * Get active grid alerts
   *
   * @param region - Optional region filter
   * @returns Array of active alerts
   */
  async getActiveAlerts(region?: string): Promise<GridAlert[]> {
    return this.call<GridAlert[]>('get_active_alerts', region ?? null);
  }

  /**
   * Get grid statistics
   *
   * @returns Aggregate grid statistics
   */
  async getGridStatistics(): Promise<GridStatistics> {
    return this.call<GridStatistics>('get_grid_statistics', null);
  }

  /**
   * Report grid alert
   *
   * @param alert - Alert information
   * @returns The created alert
   */
  async reportAlert(alert: Omit<GridAlert, 'id'>): Promise<GridAlert> {
    const record = await this.call<HolochainRecord>('report_alert', {
      ...alert,
      start_time: Date.now() * 1000,
    });
    return this.extractEntry<GridAlert>(record);
  }

  /**
   * Resolve a grid alert
   *
   * @param alertId - Alert identifier
   * @returns The updated alert
   */
  async resolveAlert(alertId: string): Promise<GridAlert> {
    const record = await this.call<HolochainRecord>('resolve_alert', {
      alert_id: alertId,
      end_time: Date.now() * 1000,
    });
    return this.extractEntry<GridAlert>(record);
  }

  // ============================================================================
  // Balance Request Operations
  // ============================================================================

  /**
   * Request grid balance (supply or demand)
   *
   * @param input - Balance request parameters
   * @returns The created request
   */
  async requestBalance(input: RequestGridBalanceInput): Promise<GridBalanceRequest> {
    const expiryHours = {
      critical_medical: 0.5,
      essential_safety: 1,
      food_preservation: 2,
      water_systems: 2,
      climate_extreme: 4,
      standard: 12,
      deferrable: 24,
      luxury: 48,
    };
    const hours = expiryHours[input.priority || 'standard'] || 12;

    const record = await this.call<HolochainRecord>('request_balance', {
      ...input,
      priority: input.priority || 'standard',
      expires_at: Date.now() * 1000 + (input.expires_in_hours || hours) * 60 * 60 * 1000 * 1000,
      fulfilled: false,
    });
    return this.extractEntry<GridBalanceRequest>(record);
  }

  /**
   * Get a balance request by ID
   *
   * @param requestId - Request identifier
   * @returns The request or null
   */
  async getBalanceRequest(requestId: string): Promise<GridBalanceRequest | null> {
    const record = await this.call<HolochainRecord | null>('get_balance_request', requestId);
    if (!record) return null;
    return this.extractEntry<GridBalanceRequest>(record);
  }

  /**
   * Get open balance requests
   *
   * @param type - Optional filter by request type
   * @returns Array of open requests
   */
  async getOpenBalanceRequests(type?: 'supply' | 'demand'): Promise<GridBalanceRequest[]> {
    const records = await this.call<HolochainRecord[]>('get_open_balance_requests', type ?? null);
    return records.map(r => this.extractEntry<GridBalanceRequest>(r));
  }

  /**
   * Fulfill a balance request
   *
   * @param requestId - Request identifier
   * @param fulfillerDid - DID of the fulfiller
   * @param amountKwh - Amount fulfilled
   * @returns The updated request
   */
  async fulfillBalanceRequest(
    requestId: string,
    fulfillerDid: string,
    amountKwh: number
  ): Promise<GridBalanceRequest> {
    const record = await this.call<HolochainRecord>('fulfill_balance_request', {
      request_id: requestId,
      fulfiller_did: fulfillerDid,
      amount_kwh: amountKwh,
    });
    return this.extractEntry<GridBalanceRequest>(record);
  }

  /**
   * Cancel a balance request
   *
   * @param requestId - Request identifier
   * @returns The updated request
   */
  async cancelBalanceRequest(requestId: string): Promise<GridBalanceRequest> {
    const record = await this.call<HolochainRecord>('cancel_balance_request', requestId);
    return this.extractEntry<GridBalanceRequest>(record);
  }

  // ============================================================================
  // Community Energy Operations
  // ============================================================================

  /**
   * Get community energy summary
   *
   * @param communityId - Community identifier
   * @returns Community energy summary
   */
  async getCommunityEnergySummary(communityId: string): Promise<CommunityEnergySummary> {
    return this.call<CommunityEnergySummary>('get_community_summary', communityId);
  }

  /**
   * Get energy status for multiple communities
   *
   * @param communityIds - Array of community identifiers
   * @returns Array of community summaries
   */
  async getCommunitiesStatus(communityIds: string[]): Promise<CommunityEnergySummary[]> {
    return this.call<CommunityEnergySummary[]>('get_communities_status', communityIds);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Request emergency supply
   *
   * @param amountKwh - Amount needed
   * @param priority - Emergency priority level
   * @param maxPrice - Maximum price willing to pay
   * @returns The created request
   */
  async requestEmergencySupply(
    amountKwh: number,
    priority: EnergyPriority = 'essential_safety',
    maxPrice?: number
  ): Promise<GridBalanceRequest> {
    return this.requestBalance({
      type: 'supply',
      amount_kwh: amountKwh,
      priority,
      max_price: maxPrice,
    });
  }

  /**
   * Offer surplus energy
   *
   * @param amountKwh - Amount available
   * @param maxPrice - Maximum price expecting (optional)
   * @returns The created request
   */
  async offerSurplusEnergy(
    amountKwh: number,
    maxPrice?: number
  ): Promise<GridBalanceRequest> {
    return this.requestBalance({
      type: 'demand',
      amount_kwh: amountKwh,
      priority: 'standard',
      max_price: maxPrice,
    });
  }

  /**
   * Get priority description
   *
   * @param priority - Energy priority
   * @returns Human-readable description
   */
  getPriorityDescription(priority: EnergyPriority): string {
    const descriptions: Record<EnergyPriority, string> = {
      critical_medical: 'Critical medical equipment requiring immediate power',
      essential_safety: 'Essential safety systems and life support',
      food_preservation: 'Food storage and preservation',
      water_systems: 'Water pumping and treatment',
      climate_extreme: 'Climate control during extreme weather',
      standard: 'Standard household needs',
      deferrable: 'Non-urgent, can be delayed',
      luxury: 'Comfort and convenience items',
    };
    return descriptions[priority];
  }

  /**
   * Get grid condition description
   *
   * @param status - Grid status
   * @returns Human-readable description and recommendation
   */
  getConditionAdvice(status: GridStatus): { description: string; recommendation: string } {
    const advice: Record<string, { description: string; recommendation: string }> = {
      normal: {
        description: 'Grid operating normally',
        recommendation: 'Normal energy usage is fine',
      },
      watch: {
        description: 'Grid load is elevated',
        recommendation: 'Consider reducing non-essential usage',
      },
      advisory: {
        description: 'Grid experiencing stress',
        recommendation: 'Reduce energy usage where possible',
      },
      emergency: {
        description: 'Grid emergency in progress',
        recommendation: 'Minimize all non-essential energy use immediately',
      },
      critical: {
        description: 'Critical grid condition - rolling blackouts possible',
        recommendation: 'Use only essential equipment, prepare backup power',
      },
      blackout: {
        description: 'Grid blackout in progress',
        recommendation: 'Use backup power for essential needs only',
      },
    };
    return advice[status.condition] || advice['normal'];
  }

  /**
   * Check if grid is stressed
   *
   * @param status - Grid status
   * @returns True if grid is under stress
   */
  isGridStressed(status: GridStatus): boolean {
    return ['advisory', 'emergency', 'critical', 'blackout'].includes(status.condition);
  }

  /**
   * Check if conservation is needed
   *
   * @param status - Grid status
   * @returns True if conservation is recommended
   */
  needsConservation(status: GridStatus): boolean {
    return status.reserve_margin_percent < 20 || status.condition !== 'normal';
  }

  /**
   * Get requests matching criteria for fulfillment
   *
   * @param type - 'supply' to find who needs energy, 'demand' to find who has surplus
   * @param maxAmount - Maximum amount to fulfill
   * @param priorities - Acceptable priority levels
   * @returns Array of matching requests
   */
  async findMatchingRequests(
    type: 'supply' | 'demand',
    maxAmount: number,
    priorities?: EnergyPriority[]
  ): Promise<GridBalanceRequest[]> {
    const requests = await this.getOpenBalanceRequests(type);
    return requests.filter(r =>
      r.amount_kwh <= maxAmount &&
      (!priorities || priorities.includes(r.priority))
    );
  }
}
