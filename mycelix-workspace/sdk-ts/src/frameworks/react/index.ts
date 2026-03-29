// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React Framework Integration for Mycelix SDK
 *
 * Provides React-specific bindings including Context, Provider, and hooks
 * for building Mycelix-powered React applications.
 *
 * @module frameworks/react
 * @packageDocumentation
 *
 * @example Quick Start
 * ```tsx
 * import { MycelixProvider, useIdentity, useProposals, useWallet } from '@mycelix/sdk/react';
 *
 * // 1. Wrap your app with the provider
 * function App() {
 *   return (
 *     <MycelixProvider config={{
 *       conductorUrl: 'ws://localhost:8888',
 *       appId: 'my-app'
 *     }}>
 *       <Dashboard />
 *     </MycelixProvider>
 *   );
 * }
 *
 * // 2. Use hooks in your components
 * function Dashboard() {
 *   const { did, loading: identityLoading } = useIdentity();
 *   const { balance, loading: walletLoading } = useWallet();
 *   const { proposals } = useProposals({ status: 'active' });
 *
 *   if (identityLoading || walletLoading) return <LoadingScreen />;
 *
 *   return (
 *     <div>
 *       <h1>Welcome, {did}</h1>
 *       <p>Balance: {balance.toString()} MYC</p>
 *       <h2>Active Proposals ({proposals.length})</h2>
 *       <ProposalList proposals={proposals} />
 *     </div>
 *   );
 * }
 * ```
 *
 * @example MFA (Multi-Factor Authentication) Hooks
 * ```tsx
 * import { useMfaState, useMfaEnroll, useFlEligibility } from '@mycelix/sdk/react';
 *
 * function SecuritySettings() {
 *   const { mfaState, loading, refresh } = useMfaState();
 *   const { enroll, enrolling } = useMfaEnroll();
 *   const { eligible, requirements } = useFlEligibility();
 *
 *   const handleEnrollHardwareKey = async () => {
 *     await enroll({
 *       factorType: 'HardwareKey',
 *       metadata: { keyName: 'YubiKey 5' }
 *     });
 *     refresh();
 *   };
 *
 *   return (
 *     <div>
 *       <h2>Security Level: {mfaState?.assuranceLevel}</h2>
 *       <p>Enrolled Factors: {mfaState?.factors.length}</p>
 *       <p>FL Eligible: {eligible ? 'Yes' : 'No'}</p>
 *       <button onClick={handleEnrollHardwareKey} disabled={enrolling}>
 *         Add Hardware Key
 *       </button>
 *     </div>
 *   );
 * }
 * ```
 *
 * @example Advanced Usage with Mutations
 * ```tsx
 * import { useMycelix, useMutation, useQuery } from '@mycelix/sdk/react';
 *
 * function CreateProposal({ daoId }) {
 *   const mycelix = useMycelix();
 *
 *   // Query hook for DAO details
 *   const { data: dao } = useQuery(
 *     (m) => m.governance.dao.getDao(daoId),
 *     [daoId]
 *   );
 *
 *   // Mutation hook for creating proposals
 *   const { mutate, loading } = useMutation(
 *     (m, input) => m.governance.proposals.createProposal(input),
 *     { onSuccess: () => toast.success('Proposal created!') }
 *   );
 *
 *   return (
 *     <form onSubmit={(e) => {
 *       e.preventDefault();
 *       mutate({ daoId, title: 'New Proposal', ... });
 *     }}>
 *       <button disabled={loading}>Create</button>
 *     </form>
 *   );
 * }
 * ```
 */

// =============================================================================
// Provider & Context
// =============================================================================

export {
  // Provider component
  MycelixProvider,

  // Context hooks
  useMycelix,
  useMycelixContext,
  useConnectionStatus,

  // HOC for class components
  withMycelix,

  // Types
  type MycelixContextValue,
  type MycelixProviderProps,
  type ConnectionStatus,
} from './provider.js';

// =============================================================================
// Domain Hooks
// =============================================================================

export {
  // Identity
  useIdentity,
  type UseIdentityReturn,

  // Proposals
  useProposals,
  type UseProposalsReturn,
  type Proposal,
  type ProposalFilter,
  type ProposalStatus,

  // Wallet
  useWallet,
  type UseWalletReturn,
  type WalletBalance,

  // Voting
  useVote,
  type UseVoteReturn,
  type VoteChoice,

  // MFA (Multi-Factor Authentication)
  useMfaState,
  useMfaEnroll,
  useMfaVerify,
  useFlEligibility,
  useAssuranceLevel,
  useEnrollmentHistory,
  type UseMfaStateReturn,
  type UseMfaEnrollReturn,
  type UseMfaVerifyReturn,
  type UseFlEligibilityReturn,
  type UseAssuranceLevelReturn,
  type UseEnrollmentHistoryReturn,
  type FactorType,
  type FactorCategory,
  type AssuranceLevel,
  type MfaState,
  type EnrolledFactor,
  type FlEligibilityResult,
  type FlRequirement,
  type FactorEnrollment,
  type EnrollFactorInput,
  type AssuranceOutput,
  type VerificationChallenge,
  type MfaVerificationResult,
  type VerificationProof,

  // Generic hooks
  useQuery,
  useMutation,
  type UseQueryOptions,
  type UseMutationOptions,
  type UseMutationReturn,
  type AsyncState,

  // Common types
  type DID,
} from './hooks.js';

// =============================================================================
// MFA UI Components
// =============================================================================

export {
  // Enrollment
  MfaEnrollmentWizard,
  type MfaEnrollmentWizardProps,

  // Verification
  FactorVerificationModal,
  type FactorVerificationModalProps,

  // Status Display
  AssuranceLevelBadge,
  AssuranceLevelIndicator,
  type AssuranceLevelBadgeProps,
  type AssuranceLevelIndicatorProps,
  type BadgeSize,

  FlEligibilityBanner,
  FlEligibilityIndicator,
  type FlEligibilityBannerProps,
  type FlEligibilityIndicatorProps,
  type BannerVariant,

  // Factor Management
  FactorList,
  type FactorListProps,
} from './components/index.js';

// =============================================================================
// Re-exports from Core
// =============================================================================

// Re-export core types for convenience
export type { Mycelix, MycelixConfig } from '../../core/index.js';
