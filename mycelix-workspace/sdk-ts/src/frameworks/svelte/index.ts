// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Svelte Framework Integration for Mycelix SDK
 *
 * Provides Svelte-compatible stores and utilities for building
 * Mycelix-powered Svelte/SvelteKit applications.
 *
 * @module frameworks/svelte
 * @packageDocumentation
 *
 * @example Quick Start
 * ```svelte
 * <script>
 *   import {
 *     initMycelix,
 *     mycelix,
 *     identity,
 *     proposals,
 *     walletBalance,
 *     castVote
 *   } from '@mycelix/sdk/svelte';
 *   import { onMount } from 'svelte';
 *
 *   onMount(async () => {
 *     await initMycelix({
 *       conductorUrl: 'ws://localhost:8888',
 *       appId: 'my-app'
 *     });
 *   });
 * </script>
 *
 * {#if $mycelix}
 *   <p>Connected as: {$identity}</p>
 *   <p>Balance: {$walletBalance.toString()} MYC</p>
 *
 *   <h2>Active Proposals</h2>
 *   {#each $proposals as proposal}
 *     <div>
 *       <h3>{proposal.title}</h3>
 *       <button on:click={() => castVote(proposal.id, 'approve')}>
 *         Approve
 *       </button>
 *     </div>
 *   {/each}
 * {:else}
 *   <p>Connecting...</p>
 * {/if}
 * ```
 *
 * @example SvelteKit Integration
 * ```ts
 * // src/hooks.client.ts
 * import { initMycelix } from '@mycelix/sdk/svelte';
 *
 * export async function init() {
 *   await initMycelix({
 *     conductorUrl: import.meta.env.VITE_CONDUCTOR_URL,
 *     appId: import.meta.env.VITE_APP_ID
 *   });
 * }
 * ```
 *
 * @example Creating Derived Stores
 * ```svelte
 * <script>
 *   import { createDerivedStore, createDaoStore } from '@mycelix/sdk/svelte';
 *
 *   // Custom derived store
 *   const myData = createDerivedStore(
 *     async (mycelix) => {
 *       const did = await mycelix.getMyDid();
 *       return mycelix.identity.profiles.getProfile(did);
 *     },
 *     { refetchInterval: 60000 }
 *   );
 *
 *   // Pre-built DAO store
 *   const dao = createDaoStore('my-dao-id');
 * </script>
 *
 * {#if $myData.loading}
 *   <LoadingSpinner />
 * {:else if $myData.error}
 *   <ErrorMessage error={$myData.error} />
 * {:else if $myData.data}
 *   <ProfileCard profile={$myData.data} />
 * {/if}
 * ```
 */

// =============================================================================
// Core Stores
// =============================================================================

export {
  // Main Mycelix store
  mycelix,

  // Connection state
  connectionState,
  connectionError,
  isConnected,
  isConnecting,

  // Identity stores
  identity,

  // Governance stores
  proposals,

  // Wallet stores
  walletBalance,
  walletLoading,

  // Initialization functions
  initMycelix,
  disconnectMycelix,
  reconnectMycelix,

  // Refresh functions
  refreshIdentity,
  refreshProposals,
  refreshWalletBalance,

  // Action functions
  createDid,
  sendTokens,
  castVote,

  // Store utilities
  writable,
  derived,

  // Types
  type DID,
  type Proposal,
  type ProposalStatus,
  type ConnectionState,
  type Readable,
  type Writable,
  type AsyncState,
  type AsyncStore,
} from './stores.js';

// =============================================================================
// Derived Store Utilities
// =============================================================================

export {
  // Main derived store factory
  createDerivedStore,

  // Specialized stores
  createDaoStore,
  createProposalsStore,
  createBalanceStore,
  createProfileStore,

  // Utility functions
  computedStore,
  combineStores,

  // Types
  type DerivedStoreOptions,
  type DerivedAsyncStore,
} from './derived.js';

// =============================================================================
// Re-exports from Core
// =============================================================================

// Re-export core types for convenience
export type { Mycelix, MycelixConfig } from '../../core/index.js';
