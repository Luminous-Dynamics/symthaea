// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/identity - Core Store
 *
 * Unified identity state management using Zustand.
 * This store is designed to be shared across all Mycelix applications.
 */

import { createStore } from 'zustand/vanilla';
import { persist, subscribeWithSelector } from 'zustand/middleware';
import type {
  MycelixSoul,
  SoulProfile,
  SoulCredential,
  SoulSession,
  WalletConnection,
  WalletState,
  AuthState,
  AuthMethod,
  MycelixApp,
  IdentityEvent,
  IdentityEventHandler,
  ChainConfig,
} from './types';

// ============================================================================
// Store State Interface
// ============================================================================

export interface IdentityState {
  // Soul State
  soul: MycelixSoul | null;
  isAuthenticated: boolean;
  authMethod: AuthMethod | null;

  // Wallet State
  wallet: WalletState;

  // Loading States
  isInitializing: boolean;
  isConnecting: boolean;
  isSigning: boolean;

  // Current App Context
  currentApp: MycelixApp;

  // Error State
  error: string | null;

  // Event Subscribers
  _eventHandlers: Set<IdentityEventHandler>;
}

export interface IdentityActions {
  // Initialization
  initialize: (appId: MycelixApp) => Promise<void>;

  // Soul Management
  createSoul: (profile: Partial<SoulProfile>) => Promise<MycelixSoul>;
  updateSoul: (updates: Partial<SoulProfile>) => Promise<void>;
  getSoul: () => MycelixSoul | null;

  // Wallet Actions
  connectWallet: (provider: WalletConnection['provider']) => Promise<WalletConnection>;
  disconnectWallet: () => Promise<void>;
  switchChain: (chainId: number) => Promise<void>;
  signMessage: (message: string) => Promise<string>;

  // Authentication
  signIn: (method: AuthMethod) => Promise<MycelixSoul>;
  signOut: () => Promise<void>;

  // Session Management
  startSession: (appId: MycelixApp) => Promise<SoulSession>;
  endSession: (sessionId: string) => Promise<void>;
  getActiveSessions: () => SoulSession[];

  // Credentials
  addCredential: (credential: SoulCredential) => Promise<void>;
  getCredentials: (type?: SoulCredential['type']) => SoulCredential[];
  hasCredential: (type: SoulCredential['type']) => boolean;

  // Resonance
  addResonance: (amount: number, source: string) => Promise<void>;
  getResonance: () => number;

  // Event System
  subscribe: (handler: IdentityEventHandler) => () => void;
  emit: (event: IdentityEvent) => void;

  // Error Handling
  setError: (error: string | null) => void;
  clearError: () => void;
}

export type IdentityStore = IdentityState & IdentityActions;

// ============================================================================
// Default Chain Configurations
// ============================================================================

export const DEFAULT_CHAINS: ChainConfig[] = [
  {
    id: 1,
    name: 'Ethereum',
    network: 'mainnet',
    nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
    rpcUrls: ['https://eth.llamarpc.com'],
    blockExplorers: [{ name: 'Etherscan', url: 'https://etherscan.io' }],
  },
  {
    id: 137,
    name: 'Polygon',
    network: 'matic',
    nativeCurrency: { name: 'MATIC', symbol: 'MATIC', decimals: 18 },
    rpcUrls: ['https://polygon.llamarpc.com'],
    blockExplorers: [{ name: 'PolygonScan', url: 'https://polygonscan.com' }],
  },
  {
    id: 8453,
    name: 'Base',
    network: 'base',
    nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
    rpcUrls: ['https://mainnet.base.org'],
    blockExplorers: [{ name: 'BaseScan', url: 'https://basescan.org' }],
  },
  {
    id: 10,
    name: 'Optimism',
    network: 'optimism',
    nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
    rpcUrls: ['https://mainnet.optimism.io'],
    blockExplorers: [{ name: 'Optimistic Etherscan', url: 'https://optimistic.etherscan.io' }],
  },
];

// ============================================================================
// Store Creation
// ============================================================================

const generateSoulId = (): string => {
  return `soul_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

const generateSessionId = (): string => {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

const generateSoulColor = (): string => {
  const colors = [
    '#8B5CF6', '#EC4899', '#10B981', '#F59E0B', '#3B82F6',
    '#6366F1', '#14B8A6', '#F97316', '#EF4444', '#84CC16',
  ];
  return colors[Math.floor(Math.random() * colors.length)];
};

export const createIdentityStore = (appId: MycelixApp = 'music') => {
  return createStore<IdentityStore>()(
    subscribeWithSelector(
      persist(
        (set, get) => ({
          // Initial State
          soul: null,
          isAuthenticated: false,
          authMethod: null,
          wallet: {
            status: 'disconnected',
            connection: null,
            supportedChains: DEFAULT_CHAINS,
          },
          isInitializing: true,
          isConnecting: false,
          isSigning: false,
          currentApp: appId,
          error: null,
          _eventHandlers: new Set(),

          // ================================================================
          // Initialization
          // ================================================================

          initialize: async (appId: MycelixApp) => {
            set({ isInitializing: true, currentApp: appId });

            try {
              // Check for existing session
              const state = get();
              if (state.soul) {
                // Update session
                await get().startSession(appId);
              }
            } catch (error) {
              console.error('Identity initialization failed:', error);
            } finally {
              set({ isInitializing: false });
            }
          },

          // ================================================================
          // Soul Management
          // ================================================================

          createSoul: async (profile: Partial<SoulProfile>) => {
            const now = new Date();
            const soul: MycelixSoul = {
              id: generateSoulId(),
              profile: {
                name: profile.name || 'Anonymous Soul',
                color: profile.color || generateSoulColor(),
                avatar: profile.avatar,
                bio: profile.bio,
                genres: profile.genres,
                location: profile.location,
                links: profile.links,
              },
              bornAt: now,
              totalResonance: 0,
              connections: 0,
              credentials: [],
              sessions: [],
            };

            set({ soul, isAuthenticated: true });
            get().emit({ type: 'soul:created', soul });

            // Start session in current app
            await get().startSession(get().currentApp);

            return soul;
          },

          updateSoul: async (updates: Partial<SoulProfile>) => {
            const { soul } = get();
            if (!soul) throw new Error('No soul to update');

            const updatedSoul: MycelixSoul = {
              ...soul,
              profile: { ...soul.profile, ...updates },
            };

            set({ soul: updatedSoul });
            get().emit({ type: 'soul:updated', soul: updatedSoul, changes: updates });
          },

          getSoul: () => get().soul,

          // ================================================================
          // Wallet Actions
          // ================================================================

          connectWallet: async (provider) => {
            set({ isConnecting: true, error: null });

            try {
              // In production, this would use wagmi/viem
              // For now, simulate connection
              const connection: WalletConnection = {
                provider,
                address: `0x${Math.random().toString(16).slice(2, 42)}` as `0x${string}`,
                chainId: 1,
                isSmartAccount: provider === 'email' || provider === 'social' || provider === 'passkey',
                connectedAt: new Date(),
              };

              set({
                wallet: {
                  ...get().wallet,
                  status: 'connected',
                  connection,
                },
                isConnecting: false,
              });

              get().emit({ type: 'wallet:connected', connection });

              // If no soul exists, create one linked to this wallet
              const { soul } = get();
              if (!soul) {
                await get().createSoul({});
              } else {
                // Link wallet to existing soul
                set({
                  soul: { ...soul, address: connection.address },
                });
              }

              return connection;
            } catch (error: any) {
              set({
                wallet: { ...get().wallet, status: 'error' },
                error: error.message,
                isConnecting: false,
              });
              get().emit({ type: 'error', code: 'WALLET_CONNECT_FAILED', message: error.message });
              throw error;
            }
          },

          disconnectWallet: async () => {
            const { wallet } = get();
            const address = wallet.connection?.address;

            set({
              wallet: {
                ...wallet,
                status: 'disconnected',
                connection: null,
              },
            });

            if (address) {
              get().emit({ type: 'wallet:disconnected', address });
            }
          },

          switchChain: async (chainId: number) => {
            const { wallet } = get();
            if (!wallet.connection) throw new Error('No wallet connected');

            // In production, use wagmi switchChain
            set({
              wallet: {
                ...wallet,
                connection: { ...wallet.connection, chainId },
              },
            });

            get().emit({ type: 'wallet:chainChanged', chainId });
          },

          signMessage: async (message: string) => {
            set({ isSigning: true });

            try {
              // In production, use viem signMessage
              const signature = `0x${Math.random().toString(16).slice(2)}`;
              set({ isSigning: false });
              return signature;
            } catch (error: any) {
              set({ isSigning: false, error: error.message });
              throw error;
            }
          },

          // ================================================================
          // Authentication
          // ================================================================

          signIn: async (method: AuthMethod) => {
            set({ isConnecting: true, error: null });

            try {
              let soul = get().soul;

              if (method.type === 'wallet') {
                await get().connectWallet(method.provider);
                soul = get().soul;
              } else if (method.type === 'email' || method.type === 'social') {
                // Account abstraction flow
                // In production, integrate with a service like Privy, Dynamic, or Web3Auth
                soul = await get().createSoul({
                  name: method.type === 'email' ? method.email.split('@')[0] : 'Social User',
                });
              } else if (method.type === 'passkey') {
                // WebAuthn passkey flow
                soul = await get().createSoul({});
              }

              if (!soul) {
                throw new Error('Failed to create or retrieve soul');
              }

              set({
                isAuthenticated: true,
                authMethod: method,
                isConnecting: false,
              });

              return soul;
            } catch (error: any) {
              set({
                isConnecting: false,
                error: error.message,
              });
              throw error;
            }
          },

          signOut: async () => {
            const { soul } = get();

            // End all sessions
            if (soul) {
              for (const session of soul.sessions) {
                await get().endSession(session.id);
              }
            }

            // Disconnect wallet
            await get().disconnectWallet();

            set({
              soul: null,
              isAuthenticated: false,
              authMethod: null,
            });
          },

          // ================================================================
          // Session Management
          // ================================================================

          startSession: async (appId: MycelixApp) => {
            const { soul } = get();
            if (!soul) throw new Error('No soul for session');

            const session: SoulSession = {
              id: generateSessionId(),
              appId,
              clientId: typeof window !== 'undefined' ? window.navigator.userAgent : 'server',
              startedAt: new Date(),
              lastActiveAt: new Date(),
            };

            const updatedSoul: MycelixSoul = {
              ...soul,
              sessions: [...soul.sessions.filter(s => s.appId !== appId), session],
            };

            set({ soul: updatedSoul });
            get().emit({ type: 'session:started', session });

            return session;
          },

          endSession: async (sessionId: string) => {
            const { soul } = get();
            if (!soul) return;

            const updatedSoul: MycelixSoul = {
              ...soul,
              sessions: soul.sessions.filter(s => s.id !== sessionId),
            };

            set({ soul: updatedSoul });
            get().emit({ type: 'session:ended', sessionId });
          },

          getActiveSessions: () => {
            return get().soul?.sessions || [];
          },

          // ================================================================
          // Credentials
          // ================================================================

          addCredential: async (credential: SoulCredential) => {
            const { soul } = get();
            if (!soul) throw new Error('No soul for credential');

            const updatedSoul: MycelixSoul = {
              ...soul,
              credentials: [...soul.credentials, credential],
            };

            set({ soul: updatedSoul });
            get().emit({ type: 'credential:added', credential });
          },

          getCredentials: (type) => {
            const credentials = get().soul?.credentials || [];
            return type ? credentials.filter(c => c.type === type) : credentials;
          },

          hasCredential: (type) => {
            return get().getCredentials(type).length > 0;
          },

          // ================================================================
          // Resonance
          // ================================================================

          addResonance: async (amount: number, source: string) => {
            const { soul } = get();
            if (!soul) throw new Error('No soul for resonance');

            const newTotal = soul.totalResonance + amount;
            const updatedSoul: MycelixSoul = {
              ...soul,
              totalResonance: newTotal,
            };

            set({ soul: updatedSoul });
            get().emit({ type: 'resonance:updated', total: newTotal, delta: amount });
          },

          getResonance: () => {
            return get().soul?.totalResonance || 0;
          },

          // ================================================================
          // Event System
          // ================================================================

          subscribe: (handler: IdentityEventHandler) => {
            get()._eventHandlers.add(handler);
            return () => {
              get()._eventHandlers.delete(handler);
            };
          },

          emit: (event: IdentityEvent) => {
            get()._eventHandlers.forEach(handler => {
              try {
                handler(event);
              } catch (error) {
                console.error('Event handler error:', error);
              }
            });
          },

          // ================================================================
          // Error Handling
          // ================================================================

          setError: (error) => set({ error }),
          clearError: () => set({ error: null }),
        }),
        {
          name: 'mycelix-identity',
          partialize: (state) => ({
            soul: state.soul,
            authMethod: state.authMethod,
            wallet: {
              status: state.wallet.status,
              connection: state.wallet.connection,
            },
          }),
        }
      )
    )
  );
};

// Default store instance
export const identityStore = createIdentityStore();
