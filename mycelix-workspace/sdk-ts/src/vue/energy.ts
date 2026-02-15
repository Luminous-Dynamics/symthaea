/**
 * @mycelix/sdk Vue 3 Composables for Energy Module
 *
 * Provides Vue 3 Composition API composables for the Energy hApp integration.
 * Includes Terra Atlas integration for project discovery and investment.
 *
 * @packageDocumentation
 * @module vue/energy
 */

import type { UseQueryReturn, UseMutationReturn, UseQueryOptions, Ref } from './index.js';
import type { HolochainRecord } from '../identity/index.js';

// ============================================================================
// Types
// ============================================================================

export interface EnergyProject {
  id: string;
  name: string;
  description: string;
  type: 'solar' | 'wind' | 'hydro' | 'nuclear' | 'storage' | 'grid' | 'other';
  location: {
    latitude: number;
    longitude: number;
    address?: string;
    region?: string;
  };
  capacity_kw: number;
  status: 'proposed' | 'funding' | 'construction' | 'operational' | 'decommissioned';
  owner_did: string;
  created_at: number;
  operational_since?: number;
}

export interface Participant {
  id: string;
  did: string;
  type: 'producer' | 'consumer' | 'prosumer' | 'grid_operator';
  sources: string[];
  capacity_kwh: number;
  location: {
    latitude: number;
    longitude: number;
  };
  active: boolean;
  created_at: number;
}

export interface EnergyTrade {
  id: string;
  seller_id: string;
  buyer_id: string;
  amount_kwh: number;
  source: string;
  price_per_kwh: number;
  currency: string;
  status: 'pending' | 'confirmed' | 'delivered' | 'settled' | 'cancelled';
  created_at: number;
  delivered_at?: number;
  settled_at?: number;
}

export interface EnergyCredit {
  id: string;
  owner_did: string;
  type: 'REC' | 'carbon_offset' | 'capacity' | 'demand_response';
  amount: number;
  unit: string;
  source_project_id?: string;
  vintage_year: number;
  expiry_date?: number;
  retired: boolean;
  created_at: number;
}

export interface Investment {
  id: string;
  project_id: string;
  investor_did: string;
  amount: number;
  currency: string;
  ownership_percentage: number;
  investment_type: 'equity' | 'debt' | 'revenue_share' | 'community';
  status: 'pledged' | 'confirmed' | 'active' | 'exited';
  returns_to_date: number;
  created_at: number;
}

export interface RegenerativeExit {
  id: string;
  project_id: string;
  from_investor: string;
  to_community: string;
  percentage: number;
  trigger: 'time' | 'profitability' | 'community_readiness' | 'manual';
  status: 'proposed' | 'approved' | 'executed';
  executed_at?: number;
}

// ============================================================================
// Project Composables
// ============================================================================

export interface UseProjectOptions extends UseQueryOptions {
  /** Include investment details */
  includeInvestments?: boolean;
  /** Include production data */
  includeProduction?: boolean;
}

/**
 * Composable to register an energy project
 */
export function useRegisterProject(): UseMutationReturn<
  HolochainRecord<EnergyProject>,
  {
    name: string;
    description: string;
    type: EnergyProject['type'];
    location: EnergyProject['location'];
    capacity_kw: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get a project
 */
export function useProject(
  _projectId: string,
  _options?: UseProjectOptions
): UseQueryReturn<HolochainRecord<EnergyProject> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get projects by owner
 */
export function useProjectsByOwner(
  _ownerDid: string,
  _options?: UseProjectOptions
): UseQueryReturn<HolochainRecord<EnergyProject>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get projects by type
 */
export function useProjectsByType(
  _type: EnergyProject['type'],
  _options?: UseProjectOptions
): UseQueryReturn<HolochainRecord<EnergyProject>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to search projects by location (Terra Atlas integration)
 */
export function useProjectsByLocation(): UseMutationReturn<
  HolochainRecord<EnergyProject>[],
  {
    center: { latitude: number; longitude: number };
    radius_km: number;
    type?: EnergyProject['type'];
    status?: EnergyProject['status'];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to update project status
 */
export function useUpdateProjectStatus(): UseMutationReturn<
  HolochainRecord<EnergyProject>,
  {
    project_id: string;
    status: EnergyProject['status'];
    notes?: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Participant Composables
// ============================================================================

/**
 * Composable to register as participant
 */
export function useRegisterParticipant(): UseMutationReturn<
  HolochainRecord<Participant>,
  {
    type: Participant['type'];
    sources: string[];
    capacity_kwh: number;
    location: Participant['location'];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get participant
 */
export function useParticipant(
  _participantId: string
): UseQueryReturn<HolochainRecord<Participant> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get my participant profile
 */
export function useMyParticipant(
  _did: string
): UseQueryReturn<HolochainRecord<Participant> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to find nearby participants
 */
export function useNearbyParticipants(): UseMutationReturn<
  HolochainRecord<Participant>[],
  {
    center: { latitude: number; longitude: number };
    radius_km: number;
    type?: Participant['type'];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to update participant capacity
 */
export function useUpdateCapacity(): UseMutationReturn<
  HolochainRecord<Participant>,
  {
    participant_id: string;
    capacity_kwh: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Trading Composables
// ============================================================================

/**
 * Composable to create energy trade offer
 */
export function useCreateTradeOffer(): UseMutationReturn<
  HolochainRecord<EnergyTrade>,
  {
    amount_kwh: number;
    source: string;
    price_per_kwh: number;
    currency: string;
    buyer_id?: string; // Optional - if not specified, it's an open offer
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to accept trade offer
 */
export function useAcceptTrade(): UseMutationReturn<HolochainRecord<EnergyTrade>, string> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to confirm delivery
 */
export function useConfirmDelivery(): UseMutationReturn<
  HolochainRecord<EnergyTrade>,
  {
    trade_id: string;
    meter_reading?: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to settle trade
 */
export function useSettleTrade(): UseMutationReturn<
  HolochainRecord<EnergyTrade>,
  {
    trade_id: string;
    payment_transaction_id: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get trade
 */
export function useTrade(_tradeId: string): UseQueryReturn<HolochainRecord<EnergyTrade> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get trades by participant
 */
export function useTradesByParticipant(
  _participantId: string
): UseQueryReturn<HolochainRecord<EnergyTrade>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get open trade offers
 */
export function useOpenTradeOffers(_location?: {
  latitude: number;
  longitude: number;
  radius_km: number;
}): UseQueryReturn<HolochainRecord<EnergyTrade>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Credit Composables
// ============================================================================

/**
 * Composable to issue energy credit
 */
export function useIssueCredit(): UseMutationReturn<
  HolochainRecord<EnergyCredit>,
  {
    type: EnergyCredit['type'];
    amount: number;
    unit: string;
    source_project_id?: string;
    vintage_year: number;
    expiry_date?: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to transfer credit
 */
export function useTransferCredit(): UseMutationReturn<
  HolochainRecord<EnergyCredit>,
  {
    credit_id: string;
    to_did: string;
    amount?: number; // Partial transfer
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to retire credit
 */
export function useRetireCredit(): UseMutationReturn<
  HolochainRecord<EnergyCredit>,
  {
    credit_id: string;
    reason: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get credits by owner
 */
export function useCreditsByOwner(
  _ownerDid: string
): UseQueryReturn<HolochainRecord<EnergyCredit>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get credit balance summary
 */
export function useCreditBalance(_ownerDid: string): UseQueryReturn<
  Record<
    EnergyCredit['type'],
    {
      total: number;
      available: number;
      retired: number;
    }
  >
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Investment Composables (Terra Atlas Integration)
// ============================================================================

export interface UseInvestmentReturn extends UseQueryReturn<HolochainRecord<Investment> | null> {
  returns: Ref<number>;
}

/**
 * Composable to make investment
 */
export function useInvest(): UseMutationReturn<
  HolochainRecord<Investment>,
  {
    project_id: string;
    amount: number;
    currency: string;
    investment_type: Investment['investment_type'];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get investment
 */
export function useInvestment(_investmentId: string): UseInvestmentReturn {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get investments by investor
 */
export function useInvestmentsByInvestor(
  _investorDid: string
): UseQueryReturn<HolochainRecord<Investment>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get investments in project
 */
export function useInvestmentsInProject(
  _projectId: string
): UseQueryReturn<HolochainRecord<Investment>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get project funding status
 */
export function useProjectFunding(_projectId: string): UseQueryReturn<{
  target: number;
  raised: number;
  percentage: number;
  investor_count: number;
}> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Regenerative Exit Composables
// ============================================================================

/**
 * Composable to propose regenerative exit
 */
export function useProposeRegenerativeExit(): UseMutationReturn<
  HolochainRecord<RegenerativeExit>,
  {
    project_id: string;
    to_community: string;
    percentage: number;
    trigger: RegenerativeExit['trigger'];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to approve regenerative exit
 */
export function useApproveRegenerativeExit(): UseMutationReturn<
  HolochainRecord<RegenerativeExit>,
  string
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to execute regenerative exit
 */
export function useExecuteRegenerativeExit(): UseMutationReturn<
  HolochainRecord<RegenerativeExit>,
  string
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get regenerative exit status
 */
export function useRegenerativeExitStatus(_projectId: string): UseQueryReturn<{
  community_owned_percentage: number;
  pending_exits: RegenerativeExit[];
  completed_exits: RegenerativeExit[];
}> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}
