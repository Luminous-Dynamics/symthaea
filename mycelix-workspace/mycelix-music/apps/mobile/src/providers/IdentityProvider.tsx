// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Identity Provider for Mycelix Mobile
 *
 * Bridges the shared @mycelix/identity package with React Native.
 * Provides authentication and soul management.
 */

import { createContext, useContext, useState, useCallback, useEffect, ReactNode } from 'react';
import * as SecureStore from 'expo-secure-store';
import * as WebBrowser from 'expo-web-browser';
import * as Haptics from 'expo-haptics';

// Types from @mycelix/identity (in production, import from package)
export interface SoulProfile {
  displayName: string;
  bio?: string;
  avatarUri?: string;
  location?: string;
  genres: string[];
  createdAt: Date;
}

export interface SoulCredential {
  type: string;
  issuedAt: Date;
  expiresAt?: Date;
  metadata?: Record<string, unknown>;
}

export interface SoulConnection {
  soulId: string;
  connectedAt: Date;
  strength: number;
  interactionCount: number;
}

export interface ResonanceBreakdown {
  total: number;
  categories: {
    listening: number;
    patronage: number;
    collaboration: number;
    community: number;
    presence: number;
  };
}

export interface MycelixSoul {
  id: string;
  address?: string;
  profile: SoulProfile;
  credentials: SoulCredential[];
  connections: SoulConnection[];
  resonance: ResonanceBreakdown;
  onChainId?: bigint;
}

export interface WalletConnection {
  provider: 'metamask' | 'walletconnect' | 'coinbase' | 'smart_wallet';
  address: string;
  chainId: number;
  isConnected: boolean;
}

interface IdentityState {
  soul: MycelixSoul | null;
  wallet: WalletConnection | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

interface IdentityContextValue extends IdentityState {
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
  connectWithEmail: (email: string) => Promise<void>;
  updateProfile: (updates: Partial<SoulProfile>) => Promise<void>;
  refreshSoul: () => Promise<void>;
}

const IdentityContext = createContext<IdentityContextValue | undefined>(undefined);

const SOUL_STORAGE_KEY = 'mycelix_soul';
const WALLET_STORAGE_KEY = 'mycelix_wallet';

// Mock soul data for development
const createMockSoul = (address?: string): MycelixSoul => ({
  id: `soul_${Math.random().toString(36).slice(2, 10)}`,
  address,
  profile: {
    displayName: 'Explorer',
    bio: 'Discovering new sounds through the mycelium',
    genres: ['ambient', 'electronic', 'experimental'],
    createdAt: new Date(),
  },
  credentials: [],
  connections: [],
  resonance: {
    total: 0,
    categories: {
      listening: 0,
      patronage: 0,
      collaboration: 0,
      community: 0,
      presence: 0,
    },
  },
});

export function IdentityProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<IdentityState>({
    soul: null,
    wallet: null,
    isAuthenticated: false,
    isLoading: true,
    error: null,
  });

  // Load persisted state on mount
  useEffect(() => {
    async function loadPersistedState() {
      try {
        const [soulJson, walletJson] = await Promise.all([
          SecureStore.getItemAsync(SOUL_STORAGE_KEY),
          SecureStore.getItemAsync(WALLET_STORAGE_KEY),
        ]);

        const soul = soulJson ? JSON.parse(soulJson) : null;
        const wallet = walletJson ? JSON.parse(walletJson) : null;

        setState({
          soul,
          wallet,
          isAuthenticated: !!soul,
          isLoading: false,
          error: null,
        });
      } catch (error) {
        console.error('Failed to load identity state:', error);
        setState((prev) => ({ ...prev, isLoading: false }));
      }
    }

    loadPersistedState();
  }, []);

  // Persist state changes
  useEffect(() => {
    async function persistState() {
      if (state.isLoading) return;

      try {
        if (state.soul) {
          await SecureStore.setItemAsync(SOUL_STORAGE_KEY, JSON.stringify(state.soul));
        } else {
          await SecureStore.deleteItemAsync(SOUL_STORAGE_KEY);
        }

        if (state.wallet) {
          await SecureStore.setItemAsync(WALLET_STORAGE_KEY, JSON.stringify(state.wallet));
        } else {
          await SecureStore.deleteItemAsync(WALLET_STORAGE_KEY);
        }
      } catch (error) {
        console.error('Failed to persist identity state:', error);
      }
    }

    persistState();
  }, [state.soul, state.wallet, state.isLoading]);

  const connect = useCallback(async () => {
    try {
      setState((prev) => ({ ...prev, isLoading: true, error: null }));
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

      // In production, this would use WalletConnect or a native wallet SDK
      // For now, simulate wallet connection
      await new Promise((resolve) => setTimeout(resolve, 1500));

      const mockAddress = `0x${Math.random().toString(16).slice(2, 42)}`;
      const wallet: WalletConnection = {
        provider: 'walletconnect',
        address: mockAddress,
        chainId: 84532, // Base Sepolia
        isConnected: true,
      };

      const soul = createMockSoul(mockAddress);

      setState({
        soul,
        wallet,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      });

      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (error) {
      console.error('Wallet connection failed:', error);
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: 'Failed to connect wallet',
      }));
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    }
  }, []);

  const connectWithEmail = useCallback(async (email: string) => {
    try {
      setState((prev) => ({ ...prev, isLoading: true, error: null }));
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

      // In production, this would trigger magic link or passkey flow
      await new Promise((resolve) => setTimeout(resolve, 1500));

      const soul = createMockSoul();
      soul.profile.displayName = email.split('@')[0];

      // Smart wallet would be created via account abstraction
      const wallet: WalletConnection = {
        provider: 'smart_wallet',
        address: `0x${Math.random().toString(16).slice(2, 42)}`,
        chainId: 84532,
        isConnected: true,
      };

      setState({
        soul,
        wallet,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      });

      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (error) {
      console.error('Email connection failed:', error);
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: 'Failed to connect with email',
      }));
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    }
  }, []);

  const disconnect = useCallback(async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    setState({
      soul: null,
      wallet: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
    });

    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  }, []);

  const updateProfile = useCallback(async (updates: Partial<SoulProfile>) => {
    if (!state.soul) return;

    setState((prev) => ({
      ...prev,
      soul: prev.soul
        ? {
            ...prev.soul,
            profile: {
              ...prev.soul.profile,
              ...updates,
            },
          }
        : null,
    }));

    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  }, [state.soul]);

  const refreshSoul = useCallback(async () => {
    if (!state.wallet?.address) return;

    setState((prev) => ({ ...prev, isLoading: true }));

    // In production, fetch latest soul state from chain/API
    await new Promise((resolve) => setTimeout(resolve, 500));

    setState((prev) => ({ ...prev, isLoading: false }));
  }, [state.wallet?.address]);

  const value: IdentityContextValue = {
    ...state,
    connect,
    disconnect,
    connectWithEmail,
    updateProfile,
    refreshSoul,
  };

  return (
    <IdentityContext.Provider value={value}>
      {children}
    </IdentityContext.Provider>
  );
}

export function useIdentityContext(): IdentityContextValue {
  const context = useContext(IdentityContext);
  if (!context) {
    throw new Error('useIdentityContext must be used within an IdentityProvider');
  }
  return context;
}
