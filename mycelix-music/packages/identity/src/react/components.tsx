// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/identity/react - React Components
 *
 * Pre-built UI components for Mycelix Identity integration.
 */

'use client';

import React, { useState, useCallback, createContext, useContext } from 'react';
import {
  useIdentity,
  useWallet,
  useAuth,
  useSoul,
  useCredentials,
  useFormattedAddress,
  useSoulAvatar,
  useResonance,
  useIdentityInit,
} from './hooks';
import type { MycelixApp, WalletConnection, SoulProfile } from '../types';

// ============================================================================
// Context Provider
// ============================================================================

interface IdentityContextValue {
  appId: MycelixApp;
}

const IdentityContext = createContext<IdentityContextValue | null>(null);

interface IdentityProviderProps {
  appId: MycelixApp;
  children: React.ReactNode;
}

/**
 * Provider component that initializes identity for an app
 */
export function IdentityProvider({ appId, children }: IdentityProviderProps) {
  const { isInitializing } = useIdentityInit(appId);

  return (
    <IdentityContext.Provider value={{ appId }}>
      {children}
    </IdentityContext.Provider>
  );
}

export function useIdentityContext() {
  const context = useContext(IdentityContext);
  if (!context) {
    throw new Error('useIdentityContext must be used within IdentityProvider');
  }
  return context;
}

// ============================================================================
// Connect Button
// ============================================================================

interface ConnectButtonProps {
  className?: string;
  variant?: 'default' | 'compact' | 'icon';
  showAvatar?: boolean;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

/**
 * Unified connect button that handles wallet and auth states
 */
export function ConnectButton({
  className = '',
  variant = 'default',
  showAvatar = true,
  onConnect,
  onDisconnect,
}: ConnectButtonProps) {
  const { isAuthenticated, soul } = useIdentity();
  const { isConnecting } = useAuth();
  const formattedAddress = useFormattedAddress();
  const avatar = useSoulAvatar();
  const { formatted: resonance } = useResonance();
  const [showModal, setShowModal] = useState(false);
  const [showMenu, setShowMenu] = useState(false);

  const handleConnect = () => {
    setShowModal(true);
    onConnect?.();
  };

  const handleDisconnect = async () => {
    const { signOut } = useAuth();
    await signOut();
    setShowMenu(false);
    onDisconnect?.();
  };

  if (isConnecting) {
    return (
      <button
        disabled
        className={`flex items-center gap-2 px-4 py-2 bg-purple-500/50 rounded-full text-sm font-medium cursor-wait ${className}`}
      >
        <LoadingSpinner />
        Connecting...
      </button>
    );
  }

  if (isAuthenticated && soul) {
    return (
      <div className="relative">
        <button
          onClick={() => setShowMenu(!showMenu)}
          className={`flex items-center gap-2 px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded-full transition-colors ${className}`}
        >
          {showAvatar && avatar && (
            <img
              src={avatar}
              alt=""
              className="w-7 h-7 rounded-full border-2"
              style={{ borderColor: soul.profile.color }}
            />
          )}
          <div className="text-left">
            <p className="text-sm font-medium">{soul.profile.name}</p>
            {variant === 'default' && formattedAddress && (
              <p className="text-xs text-muted-foreground">{formattedAddress}</p>
            )}
          </div>
          {variant === 'default' && (
            <div className="flex items-center gap-1 px-2 py-0.5 bg-purple-500/20 rounded-full">
              <span className="text-xs">✨</span>
              <span className="text-xs font-medium text-purple-300">{resonance}</span>
            </div>
          )}
        </button>

        {showMenu && (
          <SoulMenu onClose={() => setShowMenu(false)} onDisconnect={handleDisconnect} />
        )}
      </div>
    );
  }

  return (
    <>
      <button
        onClick={handleConnect}
        className={`flex items-center gap-2 px-4 py-2 bg-purple-500 hover:bg-purple-600 rounded-full text-sm font-medium transition-colors ${className}`}
      >
        <span>✨</span>
        Connect Soul
      </button>

      {showModal && <ConnectModal onClose={() => setShowModal(false)} />}
    </>
  );
}

// ============================================================================
// Connect Modal
// ============================================================================

interface ConnectModalProps {
  onClose: () => void;
}

/**
 * Modal for connecting wallets or signing in
 */
export function ConnectModal({ onClose }: ConnectModalProps) {
  const {
    signInWithWallet,
    signInWithEmail,
    signInWithSocial,
    signInWithPasskey,
    isLoading,
    error,
    clearError,
  } = useAuth();
  const [view, setView] = useState<'main' | 'email'>('main');
  const [email, setEmail] = useState('');

  const handleWalletConnect = async (provider: WalletConnection['provider']) => {
    try {
      await signInWithWallet(provider);
      onClose();
    } catch (e) {
      // Error handled by store
    }
  };

  const handleEmailConnect = async () => {
    if (!email) return;
    try {
      await signInWithEmail(email);
      onClose();
    } catch (e) {
      // Error handled by store
    }
  };

  const handleSocialConnect = async (provider: 'google' | 'apple' | 'discord') => {
    try {
      await signInWithSocial(provider);
      onClose();
    } catch (e) {
      // Error handled by store
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4">
      <div className="bg-gray-900 rounded-2xl max-w-md w-full p-6 border border-white/10">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              ✨
            </div>
            <div>
              <h2 className="text-xl font-bold">Connect Your Soul</h2>
              <p className="text-sm text-muted-foreground">Join the Mycelix network</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-lg">
            ✕
          </button>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-400">
            {error}
            <button onClick={clearError} className="ml-2 underline">
              Dismiss
            </button>
          </div>
        )}

        {view === 'main' ? (
          <>
            {/* Wallet Options */}
            <div className="space-y-2 mb-6">
              <p className="text-sm text-muted-foreground mb-3">Connect with wallet</p>
              <WalletOption
                name="MetaMask"
                icon="🦊"
                onClick={() => handleWalletConnect('metamask')}
                disabled={isLoading}
              />
              <WalletOption
                name="WalletConnect"
                icon="🔗"
                onClick={() => handleWalletConnect('walletconnect')}
                disabled={isLoading}
              />
              <WalletOption
                name="Coinbase Wallet"
                icon="🔵"
                onClick={() => handleWalletConnect('coinbase')}
                disabled={isLoading}
              />
              <WalletOption
                name="Rainbow"
                icon="🌈"
                onClick={() => handleWalletConnect('rainbow')}
                disabled={isLoading}
              />
            </div>

            <div className="relative my-6">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-white/10" />
              </div>
              <div className="relative flex justify-center text-xs">
                <span className="px-2 bg-gray-900 text-muted-foreground">or continue with</span>
              </div>
            </div>

            {/* Social Options */}
            <div className="grid grid-cols-3 gap-2 mb-4">
              <button
                onClick={() => handleSocialConnect('google')}
                disabled={isLoading}
                className="flex flex-col items-center gap-1 p-3 bg-white/5 hover:bg-white/10 rounded-lg disabled:opacity-50"
              >
                <span className="text-xl">🌐</span>
                <span className="text-xs">Google</span>
              </button>
              <button
                onClick={() => handleSocialConnect('apple')}
                disabled={isLoading}
                className="flex flex-col items-center gap-1 p-3 bg-white/5 hover:bg-white/10 rounded-lg disabled:opacity-50"
              >
                <span className="text-xl">🍎</span>
                <span className="text-xs">Apple</span>
              </button>
              <button
                onClick={() => handleSocialConnect('discord')}
                disabled={isLoading}
                className="flex flex-col items-center gap-1 p-3 bg-white/5 hover:bg-white/10 rounded-lg disabled:opacity-50"
              >
                <span className="text-xl">💬</span>
                <span className="text-xs">Discord</span>
              </button>
            </div>

            {/* Email Option */}
            <button
              onClick={() => setView('email')}
              disabled={isLoading}
              className="w-full py-2.5 bg-white/5 hover:bg-white/10 rounded-lg text-sm disabled:opacity-50"
            >
              Continue with Email
            </button>
          </>
        ) : (
          <>
            <button
              onClick={() => setView('main')}
              className="flex items-center gap-2 text-sm text-muted-foreground hover:text-white mb-4"
            >
              ← Back
            </button>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Email address</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="your@email.com"
                  className="w-full px-4 py-2.5 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:border-purple-500"
                />
              </div>
              <button
                onClick={handleEmailConnect}
                disabled={!email || isLoading}
                className="w-full py-2.5 bg-purple-500 hover:bg-purple-600 rounded-lg font-medium disabled:opacity-50"
              >
                {isLoading ? 'Connecting...' : 'Continue'}
              </button>
            </div>
          </>
        )}

        {/* Footer */}
        <p className="mt-6 text-xs text-center text-muted-foreground">
          By connecting, you agree to the Mycelix Terms of Service and acknowledge
          that your soul is sovereign and non-transferable.
        </p>
      </div>
    </div>
  );
}

interface WalletOptionProps {
  name: string;
  icon: string;
  onClick: () => void;
  disabled?: boolean;
}

function WalletOption({ name, icon, onClick, disabled }: WalletOptionProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="w-full flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
    >
      <span className="text-2xl">{icon}</span>
      <span className="font-medium">{name}</span>
    </button>
  );
}

// ============================================================================
// Soul Menu
// ============================================================================

interface SoulMenuProps {
  onClose: () => void;
  onDisconnect: () => void;
}

function SoulMenu({ onClose, onDisconnect }: SoulMenuProps) {
  const { soul } = useSoul();
  const { isVerifiedArtist, patronTier } = useCredentials();
  const { value: resonance } = useResonance();
  const formattedAddress = useFormattedAddress();

  if (!soul) return null;

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 z-40" onClick={onClose} />

      {/* Menu */}
      <div className="absolute right-0 top-full mt-2 w-72 bg-gray-900 rounded-xl border border-white/10 shadow-xl z-50 overflow-hidden">
        {/* Header */}
        <div className="p-4 bg-gradient-to-br from-purple-500/20 to-pink-500/20">
          <div className="flex items-center gap-3">
            <div
              className="w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold"
              style={{ backgroundColor: soul.profile.color }}
            >
              {soul.profile.name[0].toUpperCase()}
            </div>
            <div>
              <p className="font-semibold">{soul.profile.name}</p>
              {formattedAddress && (
                <p className="text-sm text-muted-foreground">{formattedAddress}</p>
              )}
            </div>
          </div>

          {/* Badges */}
          <div className="flex items-center gap-2 mt-3">
            {isVerifiedArtist && (
              <span className="px-2 py-0.5 bg-purple-500/30 text-purple-300 rounded text-xs">
                ✓ Verified Artist
              </span>
            )}
            {patronTier && (
              <span className="px-2 py-0.5 bg-amber-500/30 text-amber-300 rounded text-xs">
                🌱 {patronTier}
              </span>
            )}
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-px bg-white/10">
          <div className="p-3 bg-gray-900 text-center">
            <p className="text-lg font-bold">{resonance}</p>
            <p className="text-xs text-muted-foreground">Resonance</p>
          </div>
          <div className="p-3 bg-gray-900 text-center">
            <p className="text-lg font-bold">{soul.connections}</p>
            <p className="text-xs text-muted-foreground">Connections</p>
          </div>
        </div>

        {/* Menu Items */}
        <div className="p-2">
          <MenuItem icon="👤" label="View Profile" href="/profile" />
          <MenuItem icon="🎯" label="My Presence" href="/presence" />
          <MenuItem icon="⚙️" label="Settings" href="/settings" />
          <MenuItem icon="🌐" label="Network" href="/mycelium" />
          <div className="border-t border-white/10 my-2" />
          <button
            onClick={onDisconnect}
            className="w-full flex items-center gap-3 px-3 py-2 text-red-400 hover:bg-red-500/10 rounded-lg text-sm"
          >
            <span>🚪</span>
            Disconnect
          </button>
        </div>
      </div>
    </>
  );
}

interface MenuItemProps {
  icon: string;
  label: string;
  href: string;
}

function MenuItem({ icon, label, href }: MenuItemProps) {
  return (
    <a
      href={href}
      className="flex items-center gap-3 px-3 py-2 hover:bg-white/5 rounded-lg text-sm"
    >
      <span>{icon}</span>
      {label}
    </a>
  );
}

// ============================================================================
// Soul Card
// ============================================================================

interface SoulCardProps {
  className?: string;
  showStats?: boolean;
  showCredentials?: boolean;
}

/**
 * Display card for the current soul
 */
export function SoulCard({ className = '', showStats = true, showCredentials = true }: SoulCardProps) {
  const { soul, resonance, connections } = useSoul();
  const { credentials } = useCredentials();
  const avatar = useSoulAvatar();

  if (!soul) {
    return (
      <div className={`p-6 bg-white/5 rounded-xl text-center ${className}`}>
        <p className="text-muted-foreground">No soul connected</p>
        <ConnectButton className="mt-4" variant="default" />
      </div>
    );
  }

  return (
    <div className={`bg-white/5 rounded-xl overflow-hidden ${className}`}>
      {/* Banner */}
      <div
        className="h-20"
        style={{
          background: `linear-gradient(135deg, ${soul.profile.color}40, ${soul.profile.color}10)`,
        }}
      />

      {/* Content */}
      <div className="px-6 pb-6">
        {/* Avatar */}
        <div className="-mt-10 mb-4">
          {avatar ? (
            <img
              src={avatar}
              alt={soul.profile.name}
              className="w-20 h-20 rounded-full border-4 border-gray-900"
              style={{ borderColor: soul.profile.color }}
            />
          ) : (
            <div
              className="w-20 h-20 rounded-full border-4 border-gray-900 flex items-center justify-center text-2xl font-bold"
              style={{ backgroundColor: soul.profile.color }}
            >
              {soul.profile.name[0].toUpperCase()}
            </div>
          )}
        </div>

        {/* Info */}
        <h3 className="text-xl font-bold">{soul.profile.name}</h3>
        {soul.profile.bio && (
          <p className="text-sm text-muted-foreground mt-1">{soul.profile.bio}</p>
        )}

        {/* Stats */}
        {showStats && (
          <div className="grid grid-cols-3 gap-4 mt-4">
            <div className="text-center">
              <p className="text-lg font-bold">{resonance}</p>
              <p className="text-xs text-muted-foreground">Resonance</p>
            </div>
            <div className="text-center">
              <p className="text-lg font-bold">{connections}</p>
              <p className="text-xs text-muted-foreground">Connections</p>
            </div>
            <div className="text-center">
              <p className="text-lg font-bold">{credentials.length}</p>
              <p className="text-xs text-muted-foreground">Credentials</p>
            </div>
          </div>
        )}

        {/* Credentials */}
        {showCredentials && credentials.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-1">
            {credentials.slice(0, 5).map((cred, i) => (
              <span
                key={i}
                className="px-2 py-0.5 bg-purple-500/20 text-purple-300 rounded text-xs"
              >
                {formatCredentialType(cred.type)}
              </span>
            ))}
            {credentials.length > 5 && (
              <span className="px-2 py-0.5 bg-white/10 rounded text-xs">
                +{credentials.length - 5} more
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function formatCredentialType(type: string): string {
  return type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

// ============================================================================
// Utilities
// ============================================================================

function LoadingSpinner() {
  return (
    <svg className="animate-spin h-4 w-4\" viewBox="0 0 24 24">
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
        fill="none"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}
