// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/identity/react - React Hooks
 *
 * React hooks for accessing Mycelix Identity state and actions.
 */

'use client';

import { useEffect, useMemo, useCallback } from 'react';
import { useStore } from 'zustand';
import { useShallow } from 'zustand/react/shallow';
import { identityStore, type IdentityStore } from '../store';
import type {
  MycelixSoul,
  SoulProfile,
  SoulCredential,
  WalletConnection,
  AuthMethod,
  MycelixApp,
  IdentityEvent,
} from '../types';

// ============================================================================
// Core Hooks
// ============================================================================

/**
 * Access the entire identity store
 */
export function useIdentityStore<T>(selector: (state: IdentityStore) => T): T {
  return useStore(identityStore, selector);
}

/**
 * Main hook for accessing identity state and actions
 */
export function useIdentity() {
  const state = useIdentityStore(
    useShallow((s) => ({
      soul: s.soul,
      isAuthenticated: s.isAuthenticated,
      authMethod: s.authMethod,
      isInitializing: s.isInitializing,
      error: s.error,
    }))
  );

  const actions = useIdentityStore(
    useShallow((s) => ({
      signIn: s.signIn,
      signOut: s.signOut,
      updateSoul: s.updateSoul,
      clearError: s.clearError,
    }))
  );

  return { ...state, ...actions };
}

/**
 * Hook for accessing and managing the current soul
 */
export function useSoul() {
  const soul = useIdentityStore((s) => s.soul);
  const updateSoul = useIdentityStore((s) => s.updateSoul);
  const addResonance = useIdentityStore((s) => s.addResonance);
  const getResonance = useIdentityStore((s) => s.getResonance);

  return {
    soul,
    profile: soul?.profile ?? null,
    resonance: soul?.totalResonance ?? 0,
    connections: soul?.connections ?? 0,
    bornAt: soul?.bornAt ?? null,
    updateProfile: updateSoul,
    addResonance,
    getResonance,
  };
}

/**
 * Hook for wallet connection management
 */
export function useWallet() {
  const wallet = useIdentityStore((s) => s.wallet);
  const isConnecting = useIdentityStore((s) => s.isConnecting);
  const isSigning = useIdentityStore((s) => s.isSigning);

  const connect = useIdentityStore((s) => s.connectWallet);
  const disconnect = useIdentityStore((s) => s.disconnectWallet);
  const switchChain = useIdentityStore((s) => s.switchChain);
  const signMessage = useIdentityStore((s) => s.signMessage);

  const isConnected = wallet.status === 'connected';
  const address = wallet.connection?.address;
  const chainId = wallet.connection?.chainId;

  return {
    // State
    status: wallet.status,
    isConnected,
    isConnecting,
    isSigning,
    address,
    chainId,
    connection: wallet.connection,
    supportedChains: wallet.supportedChains,
    ensName: wallet.connection?.ensName,
    ensAvatar: wallet.connection?.ensAvatar,
    isSmartAccount: wallet.connection?.isSmartAccount ?? false,

    // Actions
    connect,
    disconnect,
    switchChain,
    signMessage,
  };
}

/**
 * Hook for managing soulbound credentials
 */
export function useCredentials() {
  const soul = useIdentityStore((s) => s.soul);
  const addCredential = useIdentityStore((s) => s.addCredential);
  const getCredentials = useIdentityStore((s) => s.getCredentials);
  const hasCredential = useIdentityStore((s) => s.hasCredential);

  const credentials = soul?.credentials ?? [];

  // Computed credential checks
  const isVerifiedArtist = useMemo(
    () => hasCredential('artist_verified'),
    [hasCredential]
  );

  const isMentor = useMemo(
    () => hasCredential('mentor_status'),
    [hasCredential]
  );

  const patronTier = useMemo(() => {
    const patronCredentials = getCredentials('patron_tier');
    if (patronCredentials.length === 0) return null;
    return patronCredentials[patronCredentials.length - 1].data.tier as string;
  }, [getCredentials]);

  return {
    credentials,
    addCredential,
    getCredentials,
    hasCredential,
    isVerifiedArtist,
    isMentor,
    patronTier,
  };
}

/**
 * Hook for subscribing to identity events
 */
export function useIdentityEvents(
  handler: (event: IdentityEvent) => void,
  deps: React.DependencyList = []
) {
  const subscribe = useIdentityStore((s) => s.subscribe);

  useEffect(() => {
    const unsubscribe = subscribe(handler);
    return unsubscribe;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subscribe, ...deps]);
}

/**
 * Hook for initializing identity in an app
 */
export function useIdentityInit(appId: MycelixApp) {
  const initialize = useIdentityStore((s) => s.initialize);
  const isInitializing = useIdentityStore((s) => s.isInitializing);

  useEffect(() => {
    initialize(appId);
  }, [initialize, appId]);

  return { isInitializing };
}

// ============================================================================
// Authentication Hooks
// ============================================================================

/**
 * Hook for authentication state and actions
 */
export function useAuth() {
  const isAuthenticated = useIdentityStore((s) => s.isAuthenticated);
  const authMethod = useIdentityStore((s) => s.authMethod);
  const isConnecting = useIdentityStore((s) => s.isConnecting);
  const error = useIdentityStore((s) => s.error);

  const signIn = useIdentityStore((s) => s.signIn);
  const signOut = useIdentityStore((s) => s.signOut);
  const clearError = useIdentityStore((s) => s.clearError);

  // Convenience sign-in methods
  const signInWithWallet = useCallback(
    (provider: WalletConnection['provider'] = 'metamask') =>
      signIn({ type: 'wallet', provider }),
    [signIn]
  );

  const signInWithEmail = useCallback(
    (email: string) => signIn({ type: 'email', email }),
    [signIn]
  );

  const signInWithSocial = useCallback(
    (provider: 'google' | 'apple' | 'discord') =>
      signIn({ type: 'social', provider }),
    [signIn]
  );

  const signInWithPasskey = useCallback(
    (credentialId: string) => signIn({ type: 'passkey', credentialId }),
    [signIn]
  );

  return {
    isAuthenticated,
    authMethod,
    isLoading: isConnecting,
    error,

    signIn,
    signOut,
    signInWithWallet,
    signInWithEmail,
    signInWithSocial,
    signInWithPasskey,
    clearError,
  };
}

/**
 * Hook that requires authentication
 * Throws if user is not authenticated (use with error boundary or Suspense)
 */
export function useRequireAuth(): MycelixSoul {
  const { soul, isAuthenticated, isInitializing } = useIdentity();

  if (isInitializing) {
    throw new Promise(() => {}); // Suspend
  }

  if (!isAuthenticated || !soul) {
    throw new Error('Authentication required');
  }

  return soul;
}

// ============================================================================
// Session Hooks
// ============================================================================

/**
 * Hook for managing sessions
 */
export function useSessions() {
  const soul = useIdentityStore((s) => s.soul);
  const startSession = useIdentityStore((s) => s.startSession);
  const endSession = useIdentityStore((s) => s.endSession);
  const getActiveSessions = useIdentityStore((s) => s.getActiveSessions);

  const sessions = soul?.sessions ?? [];
  const activeSessionCount = sessions.length;

  return {
    sessions,
    activeSessionCount,
    startSession,
    endSession,
    getActiveSessions,
  };
}

// ============================================================================
// Utility Hooks
// ============================================================================

/**
 * Hook for formatted wallet address
 */
export function useFormattedAddress() {
  const address = useIdentityStore((s) => s.wallet.connection?.address);
  const ensName = useIdentityStore((s) => s.wallet.connection?.ensName);

  if (ensName) return ensName;
  if (!address) return null;

  return `${address.slice(0, 6)}...${address.slice(-4)}`;
}

/**
 * Hook for soul avatar
 */
export function useSoulAvatar() {
  const soul = useIdentityStore((s) => s.soul);
  const ensAvatar = useIdentityStore((s) => s.wallet.connection?.ensAvatar);

  // Priority: ENS avatar > soul avatar > generated avatar
  if (ensAvatar) return ensAvatar;
  if (soul?.profile.avatar) return soul.profile.avatar;

  // Generate deterministic avatar based on soul ID
  if (soul?.id) {
    return `https://api.dicebear.com/7.x/shapes/svg?seed=${soul.id}`;
  }

  return null;
}

/**
 * Hook for checking specific credentials
 */
export function useHasCredential(type: SoulCredential['type']): boolean {
  const hasCredential = useIdentityStore((s) => s.hasCredential);
  return hasCredential(type);
}

/**
 * Hook for resonance with animation support
 */
export function useResonance() {
  const resonance = useIdentityStore((s) => s.soul?.totalResonance ?? 0);
  const addResonance = useIdentityStore((s) => s.addResonance);

  // Format resonance for display
  const formatted = useMemo(() => {
    if (resonance >= 1000000) {
      return `${(resonance / 1000000).toFixed(1)}M`;
    }
    if (resonance >= 1000) {
      return `${(resonance / 1000).toFixed(1)}K`;
    }
    return resonance.toString();
  }, [resonance]);

  return {
    value: resonance,
    formatted,
    add: addResonance,
  };
}
