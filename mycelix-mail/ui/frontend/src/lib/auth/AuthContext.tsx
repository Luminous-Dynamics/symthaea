// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Authentication Context for Mycelix Mail
 *
 * Provides OAuth2/OIDC authentication with PKCE flow
 */

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';

// Types
interface User {
  id: string;
  email: string;
  displayName: string;
  avatarUrl?: string;
  holochainAgentId?: string;
}

interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

interface AuthState {
  user: User | null;
  tokens: AuthTokens | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

interface AuthContextValue extends AuthState {
  login: () => Promise<void>;
  logout: () => Promise<void>;
  refreshSession: () => Promise<void>;
  getAccessToken: () => Promise<string | null>;
}

// Context
const AuthContext = createContext<AuthContextValue | null>(null);

// Constants
const API_URL = import.meta.env.VITE_API_URL || '/api';
const STORAGE_KEY = 'mycelix_auth';
const TOKEN_REFRESH_MARGIN = 5 * 60 * 1000; // 5 minutes before expiry

// PKCE helpers
function generateCodeVerifier(): string {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return btoa(String.fromCharCode(...array))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '');
}

async function generateCodeChallenge(verifier: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(verifier);
  const hash = await crypto.subtle.digest('SHA-256', data);
  return btoa(String.fromCharCode(...new Uint8Array(hash)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '');
}

// Provider component
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null,
    tokens: null,
    isAuthenticated: false,
    isLoading: true,
    error: null,
  });

  // Load stored auth on mount
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const { tokens, user } = JSON.parse(stored);
        if (tokens && tokens.expiresAt > Date.now()) {
          setState({
            user,
            tokens,
            isAuthenticated: true,
            isLoading: false,
            error: null,
          });
          return;
        }
      } catch {
        localStorage.removeItem(STORAGE_KEY);
      }
    }
    setState((s) => ({ ...s, isLoading: false }));
  }, []);

  // Persist auth state
  useEffect(() => {
    if (state.isAuthenticated && state.tokens && state.user) {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ tokens: state.tokens, user: state.user })
      );
    }
  }, [state.isAuthenticated, state.tokens, state.user]);

  // Auto-refresh tokens
  useEffect(() => {
    if (!state.tokens) return;

    const timeUntilExpiry = state.tokens.expiresAt - Date.now();
    const refreshTime = timeUntilExpiry - TOKEN_REFRESH_MARGIN;

    if (refreshTime <= 0) {
      refreshSession();
      return;
    }

    const timeout = setTimeout(refreshSession, refreshTime);
    return () => clearTimeout(timeout);
  }, [state.tokens]);

  const login = useCallback(async () => {
    try {
      setState((s) => ({ ...s, isLoading: true, error: null }));

      // Generate PKCE verifier and challenge
      const codeVerifier = generateCodeVerifier();
      const codeChallenge = await generateCodeChallenge(codeVerifier);

      // Store verifier for callback
      sessionStorage.setItem('pkce_verifier', codeVerifier);

      // Get authorization URL from backend
      const response = await fetch(`${API_URL}/auth/oauth2/authorize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code_challenge: codeChallenge,
          code_challenge_method: 'S256',
          redirect_uri: `${window.location.origin}/auth/callback`,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to initiate login');
      }

      const { authorization_url } = await response.json();

      // Redirect to authorization server
      window.location.href = authorization_url;
    } catch (error) {
      setState((s) => ({
        ...s,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Login failed',
      }));
    }
  }, []);

  const handleCallback = useCallback(async (code: string, state: string) => {
    try {
      setState((s) => ({ ...s, isLoading: true, error: null }));

      const codeVerifier = sessionStorage.getItem('pkce_verifier');
      if (!codeVerifier) {
        throw new Error('PKCE verifier not found');
      }
      sessionStorage.removeItem('pkce_verifier');

      // Exchange code for tokens
      const response = await fetch(`${API_URL}/auth/oauth2/token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code,
          code_verifier: codeVerifier,
          redirect_uri: `${window.location.origin}/auth/callback`,
        }),
      });

      if (!response.ok) {
        throw new Error('Token exchange failed');
      }

      const { access_token, refresh_token, expires_in, user } = await response.json();

      const tokens: AuthTokens = {
        accessToken: access_token,
        refreshToken: refresh_token,
        expiresAt: Date.now() + expires_in * 1000,
      };

      setState({
        user,
        tokens,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      });
    } catch (error) {
      setState((s) => ({
        ...s,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Authentication failed',
      }));
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      if (state.tokens) {
        await fetch(`${API_URL}/auth/logout`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${state.tokens.accessToken}`,
          },
        });
      }
    } catch {
      // Ignore logout errors
    } finally {
      localStorage.removeItem(STORAGE_KEY);
      setState({
        user: null,
        tokens: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,
      });
    }
  }, [state.tokens]);

  const refreshSession = useCallback(async () => {
    if (!state.tokens?.refreshToken) return;

    try {
      const response = await fetch(`${API_URL}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          refresh_token: state.tokens.refreshToken,
        }),
      });

      if (!response.ok) {
        throw new Error('Token refresh failed');
      }

      const { access_token, refresh_token, expires_in } = await response.json();

      setState((s) => ({
        ...s,
        tokens: {
          accessToken: access_token,
          refreshToken: refresh_token,
          expiresAt: Date.now() + expires_in * 1000,
        },
      }));
    } catch {
      // Refresh failed - logout
      await logout();
    }
  }, [state.tokens, logout]);

  const getAccessToken = useCallback(async (): Promise<string | null> => {
    if (!state.tokens) return null;

    // Refresh if expiring soon
    if (state.tokens.expiresAt - Date.now() < TOKEN_REFRESH_MARGIN) {
      await refreshSession();
    }

    return state.tokens?.accessToken || null;
  }, [state.tokens, refreshSession]);

  // Listen for OAuth callback
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code');
    const state = params.get('state');

    if (code && state && window.location.pathname === '/auth/callback') {
      handleCallback(code, state);
      // Clean URL
      window.history.replaceState({}, '', '/');
    }
  }, [handleCallback]);

  const value: AuthContextValue = {
    ...state,
    login,
    logout,
    refreshSession,
    getAccessToken,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// Hook
export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

// Protected route component
export function RequireAuth({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading, login } = useAuth();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      login();
    }
  }, [isLoading, isAuthenticated, login]);

  if (isLoading) {
    return (
      <div
        role="status"
        aria-label="Loading authentication"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
        }}
      >
        <span>Loading...</span>
      </div>
    );
  }

  if (!isAuthenticated) {
    return null;
  }

  return <>{children}</>;
}
