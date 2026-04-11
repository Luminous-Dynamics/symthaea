// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail Holochain SDK
 *
 * TypeScript SDK for Holochain-based email with MATL trust algorithm
 */

// Client
export {
  MycelixHolochainClient,
  createHolochainClient,
  type MycelixHolochainConfig,
} from './client';

// Types
export * from './types';

// React Hooks
export {
  MycelixProvider,
  useMycelix,
  useInbox,
  useEmail,
  useSendEmail,
  useEmailState,
  useContacts,
  useContact,
  useTrustScore,
  useCreateAttestation,
  useSearch,
  useSyncState,
  useSignal,
} from './hooks';
