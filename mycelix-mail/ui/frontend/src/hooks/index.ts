// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Hooks Index
 *
 * Central export for all custom React hooks:
 * - Epistemic service hooks (trust, claims, AI)
 * - Real-time WebSocket hooks
 */

// Epistemic service hooks
export {
  // Query keys
  epistemicKeys,
  // Claims hooks
  useCredentials,
  useCredential,
  useAssuranceLevel,
  useVerifyClaims,
  useAttachClaim,
  // AI hooks
  useAIStatus,
  useEmailAnalysis,
  useEmailIntent,
  useReplySuggestions,
  useThreadSummary,
  useExplainTrust,
  // Trust graph hooks
  useTrustGraph,
  useTrustPath,
  useTrustedBy,
  useTrusters,
  useCreateAttestation,
  useRevokeAttestation,
  // Combined hooks
  useEmailEpistemicData,
  useSenderTrust,
} from './useEpistemicServices';

// WebSocket hooks for real-time updates
export {
  default as useTrustWebSocket,
  TrustConnectionStatus,
} from './useTrustWebSocket';

export type {
  TrustEvent,
  TrustEventType,
  UseTrustWebSocketOptions,
  UseTrustWebSocketResult,
} from './useTrustWebSocket';
