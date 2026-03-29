// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React Hooks for Mycelix Media Module
 */

import type { QueryState, MutationState } from './index.js';
import type {
  Content,
  PublishContentInput,
  Attribution,
  AddAttributionInput,
  FactCheck,
  SubmitFactCheckInput,
  CurationList,
  CreateCurationListInput,
  Endorsement,
  ContentType,
  VerificationStatus,
  HolochainRecord,
} from '../media/index.js';

// Publication Hooks
export function useContentById(_contentId: string): QueryState<HolochainRecord<Content> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useContentByAuthorDid(_authorDid: string): QueryState<HolochainRecord<Content>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useContentByTagMedia(_tag: string): QueryState<HolochainRecord<Content>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useContentByTypeMedia(_type: ContentType): QueryState<HolochainRecord<Content>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function usePublishContentMutation(): MutationState<
  HolochainRecord<Content>,
  PublishContentInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useUpdateContentMutation(): MutationState<
  HolochainRecord<Content>,
  { contentId: string; updates: Partial<PublishContentInput> }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useRecordViewMutation(): MutationState<void, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

// Attribution Hooks
export function useAttributionsForContent(
  _contentId: string
): QueryState<HolochainRecord<Attribution>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useContributorWorks(
  _contributorDid: string
): QueryState<HolochainRecord<Attribution>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useAddAttributionMutation(): MutationState<
  HolochainRecord<Attribution>,
  AddAttributionInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

// Fact Check Hooks
export function useFactChecksForContent(
  _contentId: string
): QueryState<HolochainRecord<FactCheck>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useFactChecksByChecker(
  _checkerDid: string
): QueryState<HolochainRecord<FactCheck>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useContentVerificationStatus(_contentId: string): QueryState<VerificationStatus> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useSubmitFactCheckMutation(): MutationState<
  HolochainRecord<FactCheck>,
  SubmitFactCheckInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

// Curation Hooks
export function useCurationList(_listId: string): QueryState<HolochainRecord<CurationList> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useCurationListsByCurator(
  _curatorDid: string
): QueryState<HolochainRecord<CurationList>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useCreateCurationListMutation(): MutationState<
  HolochainRecord<CurationList>,
  CreateCurationListInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useAddToListMutation(): MutationState<
  HolochainRecord<CurationList>,
  { listId: string; contentId: string }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useRemoveFromListMutation(): MutationState<
  HolochainRecord<CurationList>,
  { listId: string; contentId: string }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useEndorsementsForContent(
  _contentId: string
): QueryState<HolochainRecord<Endorsement>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

export function useEndorseMutation(): MutationState<
  HolochainRecord<Endorsement>,
  { contentId: string; weight: number; comment?: string }
> {
  throw new Error('React hooks require React. This is a type stub.');
}
