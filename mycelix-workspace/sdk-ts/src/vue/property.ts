/**
 * @mycelix/sdk Vue 3 Composables for Property Module
 *
 * Provides Vue 3 Composition API composables for the Property hApp integration.
 *
 * @packageDocumentation
 * @module vue/property
 */

import type { UseQueryReturn, UseMutationReturn, UseQueryOptions } from './index.js';
import type { HolochainRecord } from '../identity/index.js';

// ============================================================================
// Types
// ============================================================================

export interface Asset {
  id: string;
  type: 'land' | 'building' | 'vehicle' | 'equipment' | 'intellectual' | 'digital' | 'other';
  name: string;
  description: string;
  ownership: Ownership[];
  location?: GeoLocation;
  valuation?: Valuation;
  liens: Lien[];
  created_at: number;
  updated_at: number;
}

export interface Ownership {
  owner_did: string;
  percentage: number;
  type: 'sole' | 'joint' | 'fractional' | 'trust' | 'corporate';
  acquired_at: number;
  transfer_id?: string;
}

export interface GeoLocation {
  latitude: number;
  longitude: number;
  address?: string;
  parcel_id?: string;
  boundaries?: string; // GeoJSON
}

export interface Valuation {
  amount: number;
  currency: string;
  method: 'market' | 'income' | 'cost' | 'self_declared';
  appraiser_did?: string;
  date: number;
}

export interface Lien {
  id: string;
  asset_id: string;
  holder_did: string;
  amount: number;
  currency: string;
  type: 'mortgage' | 'mechanic' | 'tax' | 'judgment' | 'security_interest';
  priority: number;
  created_at: number;
  released_at?: number;
}

export interface Transfer {
  id: string;
  asset_id: string;
  from_owner: string;
  to_owner: string;
  percentage: number;
  consideration?: number;
  currency?: string;
  escrow_wallet_id?: string;
  status: 'pending' | 'completed' | 'cancelled';
  created_at: number;
  completed_at?: number;
}

export interface Commons {
  id: string;
  name: string;
  description: string;
  asset_ids: string[];
  stewards: string[];
  governance_rules: string;
  created_at: number;
}

// ============================================================================
// Asset Composables
// ============================================================================

export interface UseAssetOptions extends UseQueryOptions {
  /** Include ownership history */
  includeHistory?: boolean;
  /** Include current valuation */
  includeValuation?: boolean;
  /** Include liens */
  includeLiens?: boolean;
}

/**
 * Composable to register an asset
 */
export function useRegisterAsset(): UseMutationReturn<
  HolochainRecord<Asset>,
  {
    type: Asset['type'];
    name: string;
    description: string;
    ownership_type: Ownership['type'];
    location?: GeoLocation;
    initial_valuation?: Omit<Valuation, 'date'>;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get an asset
 */
export function useAsset(
  _assetId: string,
  _options?: UseAssetOptions
): UseQueryReturn<HolochainRecord<Asset> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get assets by owner
 */
export function useAssetsByOwner(
  _ownerDid: string,
  _options?: UseAssetOptions
): UseQueryReturn<HolochainRecord<Asset>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get assets by type
 */
export function useAssetsByType(
  _type: Asset['type'],
  _options?: UseAssetOptions
): UseQueryReturn<HolochainRecord<Asset>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to search assets by location
 */
export function useAssetsByLocation(): UseMutationReturn<
  HolochainRecord<Asset>[],
  {
    center: { latitude: number; longitude: number };
    radius_km: number;
    type?: Asset['type'];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to update asset details
 */
export function useUpdateAsset(): UseMutationReturn<
  HolochainRecord<Asset>,
  {
    asset_id: string;
    name?: string;
    description?: string;
    location?: GeoLocation;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to add valuation
 */
export function useAddValuation(): UseMutationReturn<
  HolochainRecord<Asset>,
  {
    asset_id: string;
    amount: number;
    currency: string;
    method: Valuation['method'];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Transfer Composables
// ============================================================================

/**
 * Composable to initiate transfer
 */
export function useInitiateTransfer(): UseMutationReturn<
  HolochainRecord<Transfer>,
  {
    asset_id: string;
    to_owner: string;
    percentage: number;
    consideration?: number;
    currency?: string;
    use_escrow?: boolean;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to accept transfer
 */
export function useAcceptTransfer(): UseMutationReturn<HolochainRecord<Transfer>, string> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to cancel transfer
 */
export function useCancelTransfer(): UseMutationReturn<
  HolochainRecord<Transfer>,
  {
    transfer_id: string;
    reason: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get asset transfer record
 */
export function useAssetTransfer(
  _transferId: string
): UseQueryReturn<HolochainRecord<Transfer> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get transfers for an asset
 */
export function useTransfersForAsset(
  _assetId: string
): UseQueryReturn<HolochainRecord<Transfer>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get ownership history
 */
export function useOwnershipHistory(
  _assetId: string
): UseQueryReturn<Array<{ ownership: Ownership; transfer_id: string }>> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Lien Composables
// ============================================================================

/**
 * Composable to place a lien
 */
export function usePlaceLien(): UseMutationReturn<
  HolochainRecord<Lien>,
  {
    asset_id: string;
    amount: number;
    currency: string;
    type: Lien['type'];
    priority?: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to release a lien
 */
export function useReleaseLien(): UseMutationReturn<
  HolochainRecord<Lien>,
  {
    lien_id: string;
    release_reason: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get liens for an asset
 */
export function useLiensForAsset(_assetId: string): UseQueryReturn<HolochainRecord<Lien>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get liens held by entity
 */
export function useLiensHeldBy(_holderDid: string): UseQueryReturn<HolochainRecord<Lien>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to check if asset has encumbrances
 */
export function useAssetEncumbrances(_assetId: string): UseQueryReturn<{
  has_liens: boolean;
  total_lien_amount: number;
  pending_transfers: number;
}> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Commons Composables
// ============================================================================

/**
 * Composable to create commons
 */
export function useCreateCommons(): UseMutationReturn<
  HolochainRecord<Commons>,
  {
    name: string;
    description: string;
    initial_assets?: string[];
    governance_rules: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get commons
 */
export function useCommons(_commonsId: string): UseQueryReturn<HolochainRecord<Commons> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to add asset to commons
 */
export function useAddToCommons(): UseMutationReturn<
  HolochainRecord<Commons>,
  {
    commons_id: string;
    asset_id: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to remove asset from commons
 */
export function useRemoveFromCommons(): UseMutationReturn<
  HolochainRecord<Commons>,
  {
    commons_id: string;
    asset_id: string;
    reason: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to add steward
 */
export function useAddSteward(): UseMutationReturn<
  HolochainRecord<Commons>,
  {
    commons_id: string;
    steward_did: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get commons by steward
 */
export function useCommonsBySteward(
  _stewardDid: string
): UseQueryReturn<HolochainRecord<Commons>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}
