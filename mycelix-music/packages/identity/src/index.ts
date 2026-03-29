// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/identity
 *
 * Unified identity and wallet management for the Mycelix ecosystem.
 *
 * This package provides:
 * - Soulbound identity (non-transferable, experience-based)
 * - Multi-wallet support (MetaMask, WalletConnect, Coinbase, etc.)
 * - Account abstraction for gasless onboarding
 * - Cross-app session management
 * - Credential and resonance tracking
 *
 * @example
 * ```tsx
 * import { IdentityProvider, ConnectButton } from '@mycelix/identity/react';
 *
 * function App() {
 *   return (
 *     <IdentityProvider appId="music">
 *       <ConnectButton />
 *     </IdentityProvider>
 *   );
 * }
 * ```
 */

// Core types
export * from './types';

// Store
export {
  createIdentityStore,
  identityStore,
  DEFAULT_CHAINS,
  type IdentityState,
  type IdentityActions,
  type IdentityStore,
} from './store';
