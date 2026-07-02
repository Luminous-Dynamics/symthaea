// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Authentication Hook
 *
 * Wraps Privy authentication with app-specific logic.
 */

import { usePrivy, useWallets } from '@privy-io/react-auth';
import { useCallback, useEffect } from 'react';
import { api } from '@/lib/api';

export interface AuthUser {
  walletAddress: string;
  email?: string;
  displayName?: string;
  avatar?: string;
  isArtist: boolean;
}

export function useAuth() {
  const {
    ready,
    authenticated,
    user,
    login,
    logout: privyLogout,
    getAccessToken,
  } = usePrivy();

  const { wallets } = useWallets();
  const activeWallet = wallets[0];

  // Sync auth token with API client
  useEffect(() => {
    if (authenticated) {
      getAccessToken().then((token) => {
        api.setToken(token);
      });
    } else {
      api.setToken(null);
    }
  }, [authenticated, getAccessToken]);

  // Get current user
  const currentUser: AuthUser | null = user
    ? {
        walletAddress: activeWallet?.address || '',
        email: user.email?.address,
        displayName: user.email?.address?.split('@')[0] || activeWallet?.address?.slice(0, 8),
        isArtist: false, // Would be fetched from profile
      }
    : null;

  // Logout handler
  const logout = useCallback(async () => {
    api.setToken(null);
    await privyLogout();
  }, [privyLogout]);

  // Sign message for verification
  const signMessage = useCallback(
    async (message: string): Promise<string | null> => {
      if (!activeWallet) return null;
      try {
        const signature = await activeWallet.sign(message);
        return signature;
      } catch (error) {
        console.error('Failed to sign message:', error);
        return null;
      }
    },
    [activeWallet]
  );

  return {
    ready,
    authenticated,
    user: currentUser,
    walletAddress: activeWallet?.address || null,
    login,
    logout,
    signMessage,
    getAccessToken,
  };
}

export default useAuth;
