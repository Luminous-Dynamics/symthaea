// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Vue 3 Composables for Media Module
 *
 * Provides Vue 3 Composition API composables for the Media hApp integration.
 *
 * @packageDocumentation
 * @module vue/media
 */

import type { UseQueryReturn, UseMutationReturn, UseQueryOptions } from './index.js';
import type { HolochainRecord } from '../identity/index.js';

// ============================================================================
// Types
// ============================================================================

export interface Content {
  id: string;
  author_did: string;
  type: 'Article' | 'Opinion' | 'Investigation' | 'Review' | 'Analysis' | 'Interview' | 'Report' | 'Editorial' | 'Other';
  title: string;
  description?: string;
  content_hash: string;
  storage_uri: string;
  license: License;
  tags: string[];
  created_at: number;
  updated_at: number;
  view_count: number;
  endorsement_count: number;
}

export interface License {
  type: 'CC0' | 'CCBY' | 'CCBYSA' | 'CCBYNC' | 'CCBYNCSA' | 'AllRightsReserved' | 'Custom';
  custom_terms?: string;
  commercial_use: boolean;
  derivative_works: boolean;
  attribution_required: boolean;
}

export interface Attribution {
  id: string;
  content_id: string;
  contributor_did: string;
  role: 'author' | 'editor' | 'contributor' | 'source' | 'translator';
  description?: string;
  royalty_percentage?: number;
  verified: boolean;
  created_at: number;
}

export interface FactCheck {
  id: string;
  content_id: string;
  checker_did: string;
  claim_id?: string; // Link to Knowledge hApp
  status: 'pending' | 'verified' | 'disputed' | 'false' | 'misleading' | 'satire';
  summary: string;
  evidence: string[];
  epistemic_score: {
    empirical: number;
    normative: number;
    metaphysical: number;
  };
  created_at: number;
  updated_at: number;
}

export interface Endorsement {
  id: string;
  content_id: string;
  endorser_did: string;
  type: 'quality' | 'accuracy' | 'importance' | 'originality';
  weight: number;
  comment?: string;
  created_at: number;
}

export interface CurationList {
  id: string;
  curator_did: string;
  name: string;
  description: string;
  content_ids: string[];
  visibility: 'public' | 'private' | 'unlisted';
  created_at: number;
  updated_at: number;
}

// ============================================================================
// Publication Composables
// ============================================================================

export interface UseContentOptions extends UseQueryOptions {
  /** Include attributions */
  includeAttributions?: boolean;
  /** Include fact checks */
  includeFactChecks?: boolean;
  /** Include endorsements */
  includeEndorsements?: boolean;
}

/**
 * Composable to publish content
 */
export function usePublishContent(): UseMutationReturn<
  HolochainRecord<Content>,
  {
    type: Content['type'];
    title: string;
    description?: string;
    content_hash: string;
    storage_uri: string;
    license: License;
    tags?: string[];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get content
 */
export function useContent(
  _contentId: string,
  _options?: UseContentOptions
): UseQueryReturn<HolochainRecord<Content> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get content by author
 */
export function useContentByAuthor(
  _authorDid: string,
  _options?: UseContentOptions
): UseQueryReturn<HolochainRecord<Content>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get content by tag
 */
export function useContentByTag(
  _tag: string,
  _options?: UseContentOptions
): UseQueryReturn<HolochainRecord<Content>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to search content
 */
export function useSearchContent(): UseMutationReturn<
  HolochainRecord<Content>[],
  {
    query: string;
    type?: Content['type'];
    tags?: string[];
    author?: string;
    limit?: number;
    offset?: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to update content metadata
 */
export function useUpdateContent(): UseMutationReturn<
  HolochainRecord<Content>,
  {
    content_id: string;
    title?: string;
    description?: string;
    tags?: string[];
    license?: License;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to delete content
 */
export function useDeleteContent(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    content_id: string;
    reason: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to record content view
 */
export function useRecordView(): UseMutationReturn<void, string> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Attribution Composables
// ============================================================================

/**
 * Composable to add attribution
 */
export function useAddAttribution(): UseMutationReturn<
  HolochainRecord<Attribution>,
  {
    content_id: string;
    contributor_did: string;
    role: Attribution['role'];
    description?: string;
    royalty_percentage?: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get attributions for content
 */
export function useAttributionsForContent(
  _contentId: string
): UseQueryReturn<HolochainRecord<Attribution>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get attributions by contributor
 */
export function useAttributionsByContributor(
  _contributorDid: string
): UseQueryReturn<HolochainRecord<Attribution>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to verify attribution
 */
export function useVerifyAttribution(): UseMutationReturn<HolochainRecord<Attribution>, string> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to dispute attribution
 */
export function useDisputeAttribution(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    attribution_id: string;
    reason: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Fact Check Composables
// ============================================================================

/**
 * Composable to request fact check
 */
export function useRequestFactCheck(): UseMutationReturn<
  HolochainRecord<FactCheck>,
  {
    content_id: string;
    specific_claims?: string[];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to submit fact check
 */
export function useSubmitFactCheck(): UseMutationReturn<
  HolochainRecord<FactCheck>,
  {
    content_id: string;
    claim_id?: string;
    status: FactCheck['status'];
    summary: string;
    evidence: string[];
    epistemic_score: FactCheck['epistemic_score'];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get fact checks for content
 */
export function useFactChecksForContent(
  _contentId: string
): UseQueryReturn<HolochainRecord<FactCheck>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get fact checks by checker
 */
export function useFactChecksByChecker(
  _checkerDid: string
): UseQueryReturn<HolochainRecord<FactCheck>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to challenge fact check
 */
export function useChallengeFactCheck(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    factcheck_id: string;
    challenge_summary: string;
    counter_evidence: string[];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get content verification status
 */
export function useContentVerificationStatus(_contentId: string): UseQueryReturn<{
  has_fact_checks: boolean;
  consensus_status: FactCheck['status'] | 'unknown';
  check_count: number;
  latest_check_date?: number;
}> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Endorsement Composables
// ============================================================================

export interface UseEndorsementStats {
  total: number;
  by_type: Record<Endorsement['type'], number>;
  weighted_score: number;
}

/**
 * Composable to endorse content
 */
export function useEndorseContent(): UseMutationReturn<
  HolochainRecord<Endorsement>,
  {
    content_id: string;
    type: Endorsement['type'];
    comment?: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to remove content endorsement
 */
export function useRemoveContentEndorsement(): UseMutationReturn<HolochainRecord<unknown>, string> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get endorsements for content
 */
export function useEndorsementsForContent(
  _contentId: string
): UseQueryReturn<HolochainRecord<Endorsement>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get endorsement stats
 */
export function useEndorsementStats(_contentId: string): UseQueryReturn<UseEndorsementStats> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to check if user endorsed
 */
export function useHasEndorsed(
  _contentId: string,
  _endorserDid: string
): UseQueryReturn<HolochainRecord<Endorsement> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Curation Composables
// ============================================================================

/**
 * Composable to create curation list
 */
export function useCreateCurationList(): UseMutationReturn<
  HolochainRecord<CurationList>,
  {
    name: string;
    description: string;
    visibility: CurationList['visibility'];
    initial_content?: string[];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get curation list
 */
export function useCurationList(
  _listId: string
): UseQueryReturn<HolochainRecord<CurationList> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get curation lists by curator
 */
export function useCurationListsByCurator(
  _curatorDid: string
): UseQueryReturn<HolochainRecord<CurationList>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to add to curation list
 */
export function useAddToCurationList(): UseMutationReturn<
  HolochainRecord<CurationList>,
  {
    list_id: string;
    content_id: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to remove from curation list
 */
export function useRemoveFromCurationList(): UseMutationReturn<
  HolochainRecord<CurationList>,
  {
    list_id: string;
    content_id: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get content from curation list
 */
export function useCurationListContent(
  _listId: string,
  _options?: UseContentOptions
): UseQueryReturn<HolochainRecord<Content>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Feed Composables
// ============================================================================

/**
 * Composable to get personalized feed
 */
export function usePersonalizedFeed(
  _did: string,
  _options?: {
    types?: Content['type'][];
    limit?: number;
    offset?: number;
  }
): UseQueryReturn<HolochainRecord<Content>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get trending content
 */
export function useTrendingContent(_options?: {
  type?: Content['type'];
  timeframe?: 'day' | 'week' | 'month';
  limit?: number;
}): UseQueryReturn<HolochainRecord<Content>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get followed authors' content
 */
export function useFollowedContent(
  _followerDid: string,
  _options?: UseContentOptions
): UseQueryReturn<HolochainRecord<Content>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}
