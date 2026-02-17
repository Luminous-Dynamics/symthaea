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
