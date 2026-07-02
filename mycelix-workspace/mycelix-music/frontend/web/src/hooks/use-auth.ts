// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useContext, createContext, useState, useEffect, ReactNode, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useAccount, useDisconnect, useSignMessage } from 'wagmi';
import { api } from '@/lib/api';

interface User {
  id: string;
  email?: string;
  username: string;
  displayName: string;
  avatarUrl?: string;
  walletAddress?: string;
  isArtist: boolean;
  subscription?: {
    tier: 'free' | 'premium' | 'family' | 'creator';
    expiresAt: string;
  };
  preferences: {
    audioQuality: string;
    autoplay: boolean;
    privateProfile: boolean;
  };
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  loginWithWallet: () => Promise<void>;
  signup: (data: SignupData) => Promise<void>;
  logout: () => Promise<void>;
  updateUser: (data: Partial<User>) => Promise<void>;
  refreshUser: () => Promise<void>;
}

interface SignupData {
  email: string;
  username: string;
  password: string;
  displayName: string;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();
  const { address, isConnected } = useAccount();
  const { disconnect } = useDisconnect();
  const { signMessageAsync } = useSignMessage();

  const fetchUser = useCallback(async () => {
    try {
      const response = await api.get('/auth/me');
      setUser(response.data);
    } catch {
      setUser(null);
    }
  }, []);

  useEffect(() => {
    const initAuth = async () => {
      const token = localStorage.getItem('auth_token');
      if (token) {
        await fetchUser();
      }
      setIsLoading(false);
    };
    initAuth();
  }, [fetchUser]);

  const login = async (email: string, password: string) => {
    const response = await api.post('/auth/login', { email, password });
    localStorage.setItem('auth_token', response.data.token);
    setUser(response.data.user);
  };

  const loginWithWallet = async () => {
    if (!isConnected || !address) {
      throw new Error('Wallet not connected');
    }

    // Get nonce
    const { data: { nonce } } = await api.post('/auth/wallet/nonce', { address });

    // Sign message
    const message = `Sign this message to verify your wallet ownership.\n\nNonce: ${nonce}`;
    const signature = await signMessageAsync({ message });

    // Verify and login
    const response = await api.post('/auth/wallet/verify', {
      address,
      signature,
      nonce,
    });

    localStorage.setItem('auth_token', response.data.token);
    setUser(response.data.user);
  };

  const signup = async (data: SignupData) => {
    const response = await api.post('/auth/signup', data);
    localStorage.setItem('auth_token', response.data.token);
    setUser(response.data.user);
  };

  const logout = async () => {
    try {
      await api.post('/auth/logout');
    } finally {
      localStorage.removeItem('auth_token');
      setUser(null);
      if (isConnected) {
        disconnect();
      }
      router.push('/');
    }
  };

  const updateUser = async (data: Partial<User>) => {
    const response = await api.patch('/users/me', data);
    setUser(response.data);
  };

  const refreshUser = async () => {
    await fetchUser();
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        loginWithWallet,
        signup,
        logout,
        updateUser,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
