// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Authentication Store
 *
 * Centralized state management for user authentication and profile
 */

import { writable, derived, get } from 'svelte/store';
import { browser } from '$app/environment';
import type { UserProfile, AuthState } from '$types';

/**
 * localStorage key for auth token
 */
const AUTH_TOKEN_KEY = 'mycelix_auth_token';
const AUTH_EXPIRY_KEY = 'mycelix_auth_expiry';

/**
 * Load auth token from localStorage
 */
function loadAuthToken(): { token: string; expiry: number } | null {
  if (!browser) return null;

  try {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    const expiry = localStorage.getItem(AUTH_EXPIRY_KEY);

    if (token && expiry) {
      const expiryTime = parseInt(expiry, 10);
      if (expiryTime > Date.now()) {
        return { token, expiry: expiryTime };
      }
      // Token expired, clear it
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(AUTH_EXPIRY_KEY);
    }
  } catch (e) {
    console.error('Failed to load auth token:', e);
  }

  return null;
}

/**
 * Save auth token to localStorage
 */
function saveAuthToken(token: string, expiry: number): void {
  if (!browser) return;

  try {
    localStorage.setItem(AUTH_TOKEN_KEY, token);
    localStorage.setItem(AUTH_EXPIRY_KEY, expiry.toString());
  } catch (e) {
    console.error('Failed to save auth token:', e);
  }
}

/**
 * Clear auth token from localStorage
 */
function clearAuthToken(): void {
  if (!browser) return;

  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_EXPIRY_KEY);
}

/**
 * Initial auth state
 */
const initialAuthState: AuthState = {
  isAuthenticated: false,
  user: null,
  token: undefined,
  tokenExpiry: undefined,
};

// Try to restore auth from localStorage
const savedAuth = loadAuthToken();
if (savedAuth) {
  initialAuthState.token = savedAuth.token;
  initialAuthState.tokenExpiry = savedAuth.expiry;
  // Note: We'll need to fetch user profile from Holochain
  // This is just the token restoration
}

/**
 * Create authentication store
 */
function createAuthStore() {
  const store = writable<AuthState>(initialAuthState);
  const { subscribe, set, update } = store;

  return {
    subscribe,

    /**
     * Login with user profile and token
     */
    login: (user: UserProfile, token: string, expiryHours: number = 24) => {
      const expiry = Date.now() + expiryHours * 60 * 60 * 1000;

      saveAuthToken(token, expiry);

      set({
        isAuthenticated: true,
        user,
        token,
        tokenExpiry: expiry,
      });
    },

    /**
     * Logout and clear session
     */
    logout: () => {
      clearAuthToken();

      set({
        isAuthenticated: false,
        user: null,
        token: undefined,
        tokenExpiry: undefined,
      });
    },

    /**
     * Update user profile
     */
    updateProfile: (user: UserProfile) => {
      update((state) => ({
        ...state,
        user,
      }));
    },

    /**
     * Check if token is expired
     */
    isTokenExpired: (): boolean => {
      const { tokenExpiry } = get(store);
      if (!tokenExpiry) return true;
      return tokenExpiry < Date.now();
    },

    /**
     * Refresh token
     */
    refreshToken: (newToken: string, expiryHours: number = 24) => {
      const expiry = Date.now() + expiryHours * 60 * 60 * 1000;

      saveAuthToken(newToken, expiry);

      update((state) => ({
        ...state,
        token: newToken,
        tokenExpiry: expiry,
      }));
    },
  };
}

/**
 * Authentication store
 */
export const auth = createAuthStore();

/**
 * Derived store: is authenticated
 */
export const isAuthenticated = derived(auth, ($auth) => $auth.isAuthenticated);

/**
 * Derived store: current user
 */
export const currentUser = derived(auth, ($auth) => $auth.user);

/**
 * Derived store: user agent ID
 */
export const userAgentId = derived(auth, ($auth) => $auth.user?.agent_id || null);

/**
 * Derived store: user roles
 */
export const userRoles = derived(auth, ($auth) => $auth.user?.roles || []);

/**
 * Derived store: is user an arbitrator
 */
export const isArbitrator = derived(userRoles, ($roles) => $roles.includes('arbitrator'));

/**
 * Derived store: is user an admin
 */
export const isAdmin = derived(userRoles, ($roles) => $roles.includes('admin'));
