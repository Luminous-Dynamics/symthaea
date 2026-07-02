// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Contract Hooks for Mycelix
 *
 * React hooks for interacting with Mycelix smart contracts.
 */

'use client';

import { useCallback, useMemo } from 'react';
import {
  useAccount,
  useChainId,
  useReadContract,
  useWriteContract,
  useWaitForTransactionReceipt,
  usePublicClient,
  useWalletClient,
} from 'wagmi';
import { getContractAddress, CONTRACT_ADDRESSES } from './config';
import { SOUL_REGISTRY_ABI, PROOF_OF_PRESENCE_ABI } from './abis';

// ============================================================================
// Types
// ============================================================================

export interface Soul {
  id: bigint;
  owner: `0x${string}`;
  profileUri: string;
  totalResonance: bigint;
  connections: bigint;
  bornAt: bigint;
  exists: boolean;
}

export interface Presence {
  id: bigint;
  holder: `0x${string}`;
  presenceType: number;
  eventId: bigint;
  eventName: string;
  timestamp: bigint;
  duration: bigint;
  artistId: string;
  collaboratorIds: string[];
  location: string;
  experienceHash: `0x${string}`;
  metadataUri: string;
  resonanceLevel: bigint;
}

export type PresenceType =
  | 'LIVE_CONCERT'
  | 'VIRTUAL_CONCERT'
  | 'ALBUM_RELEASE'
  | 'FIRST_LISTEN'
  | 'LISTENING_CIRCLE'
  | 'ARTIST_MILESTONE'
  | 'COLLABORATIVE_SESSION'
  | 'COMMUNITY_EVENT'
  | 'SEASONAL_GATHERING'
  | 'PATRONAGE_RENEWAL';

const PRESENCE_TYPE_MAP: Record<PresenceType, number> = {
  LIVE_CONCERT: 0,
  VIRTUAL_CONCERT: 1,
  ALBUM_RELEASE: 2,
  FIRST_LISTEN: 3,
  LISTENING_CIRCLE: 4,
  ARTIST_MILESTONE: 5,
  COLLABORATIVE_SESSION: 6,
  COMMUNITY_EVENT: 7,
  SEASONAL_GATHERING: 8,
  PATRONAGE_RENEWAL: 9,
};

// ============================================================================
// Soul Registry Hooks
// ============================================================================

/**
 * Hook for reading and writing to SoulRegistry
 */
export function useSoulRegistry() {
  const chainId = useChainId();
  const { address } = useAccount();
  const publicClient = usePublicClient();

  const contractAddress = getContractAddress(chainId, 'SoulRegistry');

  // Read: Check if user has a soul
  const { data: hasSoul, isLoading: isCheckingHasSoul, refetch: refetchHasSoul } = useReadContract({
    address: contractAddress,
    abi: SOUL_REGISTRY_ABI,
    functionName: 'hasSoul',
    args: address ? [address] : undefined,
    query: { enabled: !!address && !!contractAddress },
  });

  // Read: Get user's soul
  const { data: soul, isLoading: isLoadingSoul, refetch: refetchSoul } = useReadContract({
    address: contractAddress,
    abi: SOUL_REGISTRY_ABI,
    functionName: 'getSoul',
    args: address ? [address] : undefined,
    query: { enabled: !!address && !!contractAddress && hasSoul === true },
  });

  // Write: Create soul
  const { writeContract: createSoulWrite, data: createSoulHash, isPending: isCreatingSoul } = useWriteContract();

  const { isLoading: isConfirmingCreate, isSuccess: createConfirmed } = useWaitForTransactionReceipt({
    hash: createSoulHash,
  });

  const createSoul = useCallback(
    async (profileUri: string) => {
      if (!contractAddress) throw new Error('Contract not deployed on this chain');

      createSoulWrite({
        address: contractAddress,
        abi: SOUL_REGISTRY_ABI,
        functionName: 'createSoul',
        args: [profileUri],
      });
    },
    [contractAddress, createSoulWrite]
  );

  // Write: Update profile
  const { writeContract: updateProfileWrite, data: updateProfileHash, isPending: isUpdatingProfile } = useWriteContract();

  const { isLoading: isConfirmingUpdate, isSuccess: updateConfirmed } = useWaitForTransactionReceipt({
    hash: updateProfileHash,
  });

  const updateProfile = useCallback(
    async (newProfileUri: string) => {
      if (!contractAddress) throw new Error('Contract not deployed on this chain');

      updateProfileWrite({
        address: contractAddress,
        abi: SOUL_REGISTRY_ABI,
        functionName: 'updateProfile',
        args: [newProfileUri],
      });
    },
    [contractAddress, updateProfileWrite]
  );

  // Write: Connect to another soul
  const { writeContract: connectWrite, data: connectHash, isPending: isConnecting } = useWriteContract();

  const { isLoading: isConfirmingConnect, isSuccess: connectConfirmed } = useWaitForTransactionReceipt({
    hash: connectHash,
  });

  const connect = useCallback(
    async (otherSoulId: bigint) => {
      if (!contractAddress) throw new Error('Contract not deployed on this chain');

      connectWrite({
        address: contractAddress,
        abi: SOUL_REGISTRY_ABI,
        functionName: 'connect',
        args: [otherSoulId],
      });
    },
    [contractAddress, connectWrite]
  );

  // Read: Get soul by ID
  const getSoulById = useCallback(
    async (soulId: bigint): Promise<Soul | null> => {
      if (!publicClient || !contractAddress) return null;

      try {
        const result = await publicClient.readContract({
          address: contractAddress,
          abi: SOUL_REGISTRY_ABI,
          functionName: 'getSoulById',
          args: [soulId],
        });
        return result as Soul;
      } catch {
        return null;
      }
    },
    [publicClient, contractAddress]
  );

  // Read: Check if two souls are connected
  const areConnected = useCallback(
    async (soulId1: bigint, soulId2: bigint): Promise<boolean> => {
      if (!publicClient || !contractAddress) return false;

      try {
        const result = await publicClient.readContract({
          address: contractAddress,
          abi: SOUL_REGISTRY_ABI,
          functionName: 'areConnected',
          args: [soulId1, soulId2],
        });
        return result as boolean;
      } catch {
        return false;
      }
    },
    [publicClient, contractAddress]
  );

  return {
    // State
    hasSoul: hasSoul ?? false,
    soul: soul as Soul | undefined,
    isLoading: isCheckingHasSoul || isLoadingSoul,
    contractAddress,

    // Create soul
    createSoul,
    isCreatingSoul: isCreatingSoul || isConfirmingCreate,
    createConfirmed,

    // Update profile
    updateProfile,
    isUpdatingProfile: isUpdatingProfile || isConfirmingUpdate,
    updateConfirmed,

    // Connect
    connect,
    isConnecting: isConnecting || isConfirmingConnect,
    connectConfirmed,

    // Queries
    getSoulById,
    areConnected,

    // Refetch
    refetch: () => {
      refetchHasSoul();
      refetchSoul();
    },
  };
}

// ============================================================================
// Proof of Presence Hooks
// ============================================================================

/**
 * Hook for reading and minting Proof of Presence tokens
 */
export function useProofOfPresence() {
  const chainId = useChainId();
  const { address } = useAccount();
  const publicClient = usePublicClient();

  const contractAddress = getContractAddress(chainId, 'ProofOfPresence');

  // Read: Get user's presences
  const { data: presences, isLoading: isLoadingPresences, refetch: refetchPresences } = useReadContract({
    address: contractAddress,
    abi: PROOF_OF_PRESENCE_ABI,
    functionName: 'getPresencesByHolder',
    args: address ? [address] : undefined,
    query: { enabled: !!address && !!contractAddress },
  });

  // Read: Get total resonance
  const { data: totalResonance, refetch: refetchResonance } = useReadContract({
    address: contractAddress,
    abi: PROOF_OF_PRESENCE_ABI,
    functionName: 'getTotalResonance',
    args: address ? [address] : undefined,
    query: { enabled: !!address && !!contractAddress },
  });

  // Read: Check if has presence type
  const hasPresenceType = useCallback(
    async (type: PresenceType): Promise<boolean> => {
      if (!publicClient || !contractAddress || !address) return false;

      try {
        const result = await publicClient.readContract({
          address: contractAddress,
          abi: PROOF_OF_PRESENCE_ABI,
          functionName: 'hasPresenceType',
          args: [address, PRESENCE_TYPE_MAP[type]],
        });
        return result as boolean;
      } catch {
        return false;
      }
    },
    [publicClient, contractAddress, address]
  );

  // Read: Get presences for an event
  const getPresencesByEvent = useCallback(
    async (eventId: bigint): Promise<Presence[]> => {
      if (!publicClient || !contractAddress) return [];

      try {
        const result = await publicClient.readContract({
          address: contractAddress,
          abi: PROOF_OF_PRESENCE_ABI,
          functionName: 'getPresencesByEvent',
          args: [eventId],
        });
        return result as Presence[];
      } catch {
        return [];
      }
    },
    [publicClient, contractAddress]
  );

  // Read: Get artist presence count
  const getArtistPresenceCount = useCallback(
    async (artistId: string): Promise<bigint> => {
      if (!publicClient || !contractAddress || !address) return 0n;

      try {
        const result = await publicClient.readContract({
          address: contractAddress,
          abi: PROOF_OF_PRESENCE_ABI,
          functionName: 'getArtistPresenceCount',
          args: [address, artistId],
        });
        return result as bigint;
      } catch {
        return 0n;
      }
    },
    [publicClient, contractAddress, address]
  );

  // Read: Get unlocked abilities
  const getUnlockedAbilities = useCallback(
    async (tokenId: bigint): Promise<string[]> => {
      if (!publicClient || !contractAddress) return [];

      try {
        const result = await publicClient.readContract({
          address: contractAddress,
          abi: PROOF_OF_PRESENCE_ABI,
          functionName: 'getUnlockedAbilities',
          args: [tokenId],
        });
        return result as string[];
      } catch {
        return [];
      }
    },
    [publicClient, contractAddress]
  );

  // Computed values
  const presenceCount = useMemo(() => {
    return (presences as Presence[] | undefined)?.length ?? 0;
  }, [presences]);

  const highestResonancePresence = useMemo(() => {
    const p = presences as Presence[] | undefined;
    if (!p || p.length === 0) return null;

    return p.reduce((highest, current) =>
      current.resonanceLevel > highest.resonanceLevel ? current : highest
    );
  }, [presences]);

  return {
    // State
    presences: (presences as Presence[] | undefined) ?? [],
    totalResonance: totalResonance as bigint | undefined,
    isLoading: isLoadingPresences,
    contractAddress,

    // Computed
    presenceCount,
    highestResonancePresence,

    // Queries
    hasPresenceType,
    getPresencesByEvent,
    getArtistPresenceCount,
    getUnlockedAbilities,

    // Refetch
    refetch: () => {
      refetchPresences();
      refetchResonance();
    },
  };
}

// ============================================================================
// Combined Hook
// ============================================================================

/**
 * Combined hook for all Mycelix contract interactions
 */
export function useMycelixContracts() {
  const chainId = useChainId();
  const { address, isConnected } = useAccount();

  const soulRegistry = useSoulRegistry();
  const proofOfPresence = useProofOfPresence();

  const isSupported = useMemo(() => {
    return chainId in CONTRACT_ADDRESSES;
  }, [chainId]);

  return {
    // Connection state
    isConnected,
    address,
    chainId,
    isSupported,

    // Soul Registry
    ...soulRegistry,

    // Proof of Presence (prefixed to avoid conflicts)
    presences: proofOfPresence.presences,
    totalResonance: proofOfPresence.totalResonance,
    presenceCount: proofOfPresence.presenceCount,
    highestResonancePresence: proofOfPresence.highestResonancePresence,
    hasPresenceType: proofOfPresence.hasPresenceType,
    getPresencesByEvent: proofOfPresence.getPresencesByEvent,
    getArtistPresenceCount: proofOfPresence.getArtistPresenceCount,
    getUnlockedAbilities: proofOfPresence.getUnlockedAbilities,
    refetchPresences: proofOfPresence.refetch,

    // Loading state
    isLoading: soulRegistry.isLoading || proofOfPresence.isLoading,
  };
}
