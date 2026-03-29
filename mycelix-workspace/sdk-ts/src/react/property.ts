// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React Hooks for Mycelix Property Module
 */

import type { QueryState, MutationState } from './index.js';
import type {
  Asset,
  RegisterAssetInput,
  TitleTransfer,
  InitiateTransferInput,
  Lien,
  FileLienInput,
  Commons,
  CreateCommonsInput,
  AssetType,
  HolochainRecord,
} from '../property/index.js';

// Asset Hooks
export function useAssetById(_assetId: string): QueryState<HolochainRecord<Asset> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useAssetsByOwnerDid(_ownerDid: string): QueryState<HolochainRecord<Asset>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useAssetsByTypeProp(_assetType: AssetType): QueryState<HolochainRecord<Asset>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useAssetHistoryProp(_assetId: string): QueryState<HolochainRecord<Asset>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useRegisterAssetMutation(): MutationState<
  HolochainRecord<Asset>,
  RegisterAssetInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useUpdateAssetMutation(): MutationState<
  HolochainRecord<Asset>,
  { assetId: string; updates: Partial<RegisterAssetInput> }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

// Transfer Hooks
export function useTransferById(
  _transferId: string
): QueryState<HolochainRecord<TitleTransfer> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useTransfersForAsset(
  _assetId: string
): QueryState<HolochainRecord<TitleTransfer>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function usePendingTransfersProp(
  _did: string
): QueryState<HolochainRecord<TitleTransfer>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useInitiateTransferMutation(): MutationState<
  HolochainRecord<TitleTransfer>,
  InitiateTransferInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useAcceptTransfer(): MutationState<HolochainRecord<TitleTransfer>, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useCancelTransfer(): MutationState<HolochainRecord<TitleTransfer>, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

// Lien Hooks
export function useLienById(_lienId: string): QueryState<HolochainRecord<Lien> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useLiensForAsset(_assetId: string): QueryState<HolochainRecord<Lien>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useFileLienMutation(): MutationState<HolochainRecord<Lien>, FileLienInput> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useSatisfyLien(): MutationState<HolochainRecord<Lien>, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useReleaseLien(): MutationState<HolochainRecord<Lien>, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

// Commons Hooks
export function useCommonsById(_commonsId: string): QueryState<HolochainRecord<Commons> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useCommonsByDAO(_daoId: string): QueryState<HolochainRecord<Commons>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useCreateCommonsMutation(): MutationState<
  HolochainRecord<Commons>,
  CreateCommonsInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useJoinCommons(): MutationState<HolochainRecord<Commons>, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useUpdateCommonsRules(): MutationState<
  HolochainRecord<Commons>,
  { commonsId: string; rules: string }
> {
  throw new Error('React hooks require React. This is a type stub.');
}
