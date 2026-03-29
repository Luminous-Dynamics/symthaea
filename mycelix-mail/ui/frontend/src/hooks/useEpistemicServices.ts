// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React Query hooks for Epistemic Mail services
 *
 * Provides hooks for:
 * - Claims verification (0TML integration)
 * - AI insights (Symthaea integration)
 * - Trust graph visualization
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/services/api';
import type {
  Email,
  VerifiableClaim,
  AssuranceLevel,
  ProofType,
  RelationType,
  CreateAttestationRequest,
} from '@/services/api';

// ============================================
// Query Keys
// ============================================

export const epistemicKeys = {
  // Claims
  claims: ['claims'] as const,
  credentials: () => [...epistemicKeys.claims, 'credentials'] as const,
  credential: (id: string) => [...epistemicKeys.credentials(), id] as const,
  assurance: (did: string) => [...epistemicKeys.claims, 'assurance', did] as const,
  verification: (claimIds: string) => [...epistemicKeys.claims, 'verify', claimIds] as const,

  // AI
  ai: ['ai'] as const,
  aiStatus: () => [...epistemicKeys.ai, 'status'] as const,
  emailAnalysis: (emailId: string) => [...epistemicKeys.ai, 'analysis', emailId] as const,
  threadSummary: (threadId: string) => [...epistemicKeys.ai, 'summary', threadId] as const,
  emailIntent: (emailId: string) => [...epistemicKeys.ai, 'intent', emailId] as const,
  replySuggestions: (emailId: string) => [...epistemicKeys.ai, 'replies', emailId] as const,

  // Trust Graph
  trustGraph: ['trust-graph'] as const,
  graph: (did: string, depth: number) => [...epistemicKeys.trustGraph, 'graph', did, depth] as const,
  path: (from: string, to: string) => [...epistemicKeys.trustGraph, 'path', from, to] as const,
  trustedBy: (did: string) => [...epistemicKeys.trustGraph, 'trusted-by', did] as const,
  trusters: (did: string) => [...epistemicKeys.trustGraph, 'trusters', did] as const,
};

// ============================================
// Claims Hooks (0TML Integration)
// ============================================

/**
 * Fetch user's verifiable credentials
 */
export function useCredentials() {
  return useQuery({
    queryKey: epistemicKeys.credentials(),
    queryFn: () => api.listCredentials(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Fetch a specific credential by ID
 */
export function useCredential(id: string) {
  return useQuery({
    queryKey: epistemicKeys.credential(id),
    queryFn: () => api.getCredential(id),
    enabled: !!id,
  });
}

/**
 * Fetch assurance level for a DID
 */
export function useAssuranceLevel(did: string) {
  return useQuery({
    queryKey: epistemicKeys.assurance(did),
    queryFn: () => api.getAssuranceLevel(did),
    enabled: !!did,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}

/**
 * Verify a set of claims
 */
export function useVerifyClaims() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      claims,
      assuranceLevel,
    }: {
      claims: VerifiableClaim[];
      assuranceLevel?: AssuranceLevel;
    }) => api.verifyClaims(claims, assuranceLevel),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: epistemicKeys.claims });
    },
  });
}

/**
 * Attach a claim with proof to an outgoing email
 */
export function useAttachClaim() {
  return useMutation({
    mutationFn: ({
      claim,
      proofType,
    }: {
      claim: VerifiableClaim;
      proofType: ProofType;
    }) => api.attachClaim(claim, proofType),
  });
}

// ============================================
// AI Hooks (Symthaea Integration)
// ============================================

/**
 * Get AI system status
 */
export function useAIStatus() {
  return useQuery({
    queryKey: epistemicKeys.aiStatus(),
    queryFn: () => api.getAIStatus(),
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // Refetch every minute
  });
}

/**
 * Analyze an email with local AI
 */
export function useEmailAnalysis(email: Email | null) {
  return useQuery({
    queryKey: epistemicKeys.emailAnalysis(email?.id ?? ''),
    queryFn: () => api.analyzeEmail(email!),
    enabled: !!email,
    staleTime: 5 * 60 * 1000, // 5 minutes - analysis doesn't change
  });
}

/**
 * Detect intent of an email
 */
export function useEmailIntent(email: Email | null) {
  return useQuery({
    queryKey: epistemicKeys.emailIntent(email?.id ?? ''),
    queryFn: () => api.detectIntent(email!),
    enabled: !!email,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Generate reply suggestions for an email
 */
export function useReplySuggestions(email: Email | null, enabled: boolean = true) {
  return useQuery({
    queryKey: epistemicKeys.replySuggestions(email?.id ?? ''),
    queryFn: () => api.suggestReplies(email!),
    enabled: !!email && enabled,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}

/**
 * Summarize an email thread
 */
export function useThreadSummary(emails: Email[], threadId: string) {
  return useQuery({
    queryKey: epistemicKeys.threadSummary(threadId),
    queryFn: () => api.summarizeThread(emails),
    enabled: emails.length > 0,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Explain trust score with AI
 */
export function useExplainTrust() {
  return useMutation({
    mutationFn: ({
      senderDid,
      trustScore,
      trustPath,
    }: {
      senderDid: string;
      trustScore: number;
      trustPath: [string, string, number][];
    }) => api.explainTrust(senderDid, trustScore, trustPath),
  });
}

// ============================================
// Trust Graph Hooks
// ============================================

/**
 * Fetch trust graph centered on a DID
 */
export function useTrustGraph(did: string, depth: number = 2) {
  return useQuery({
    queryKey: epistemicKeys.graph(did, depth),
    queryFn: () => api.getTrustGraph(did, depth),
    enabled: !!did,
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
}

/**
 * Find trust path between two DIDs
 */
export function useTrustPath(fromDid: string, toDid: string) {
  return useQuery({
    queryKey: epistemicKeys.path(fromDid, toDid),
    queryFn: () => api.findTrustPath(fromDid, toDid),
    enabled: !!fromDid && !!toDid,
    staleTime: 2 * 60 * 1000,
    retry: false, // Don't retry if no path exists
  });
}

/**
 * Get entities trusted by a DID
 */
export function useTrustedBy(did: string) {
  return useQuery({
    queryKey: epistemicKeys.trustedBy(did),
    queryFn: () => api.getTrustedBy(did),
    enabled: !!did,
    staleTime: 2 * 60 * 1000,
  });
}

/**
 * Get entities that trust a DID
 */
export function useTrusters(did: string) {
  return useQuery({
    queryKey: epistemicKeys.trusters(did),
    queryFn: () => api.getTrusters(did),
    enabled: !!did,
    staleTime: 2 * 60 * 1000,
  });
}

/**
 * Create a trust attestation
 */
export function useCreateAttestation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: CreateAttestationRequest) => api.createAttestation(request),
    onSuccess: (data) => {
      // Invalidate trust graph queries
      queryClient.invalidateQueries({ queryKey: epistemicKeys.trustGraph });
      // Invalidate specific DID queries
      queryClient.invalidateQueries({
        queryKey: epistemicKeys.trustedBy(data.attestation.attestor_did),
      });
      queryClient.invalidateQueries({
        queryKey: epistemicKeys.trusters(data.attestation.subject_did),
      });
    },
  });
}

/**
 * Revoke a trust attestation
 */
export function useRevokeAttestation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => api.revokeAttestation(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: epistemicKeys.trustGraph });
    },
  });
}

// ============================================
// Combined Hooks for Email Views
// ============================================

/**
 * Get all epistemic data for an email (analysis + intent + trust)
 * Useful for the enhanced email detail view
 */
export function useEmailEpistemicData(email: Email | null, senderDid?: string) {
  const analysis = useEmailAnalysis(email);
  const intent = useEmailIntent(email);
  const assurance = useAssuranceLevel(senderDid ?? '');

  return {
    analysis: analysis.data,
    intent: intent.data,
    assurance: assurance.data,
    isLoading: analysis.isLoading || intent.isLoading || assurance.isLoading,
    isError: analysis.isError || intent.isError || assurance.isError,
    error: analysis.error || intent.error || assurance.error,
  };
}

/**
 * Combined hook for sender trust information
 */
export function useSenderTrust(senderDid: string, userDid: string) {
  const trustPath = useTrustPath(userDid, senderDid);
  const assurance = useAssuranceLevel(senderDid);
  const trusters = useTrusters(senderDid);

  return {
    path: trustPath.data,
    assurance: assurance.data,
    trusters: trusters.data?.edges ?? [],
    isLoading: trustPath.isLoading || assurance.isLoading || trusters.isLoading,
    hasPath: !!trustPath.data && trustPath.data.hops.length > 0,
    pathLength: trustPath.data?.path_length ?? 0,
    trustWeight: trustPath.data?.total_weight ?? 0,
  };
}
