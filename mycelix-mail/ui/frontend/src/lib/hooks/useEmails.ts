// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Email hooks for Mycelix Mail
 *
 * React Query hooks for email operations with GraphQL
 */

import { useQuery, useMutation, useQueryClient, useInfiniteQuery } from '@tanstack/react-query';
import { graphqlClient, queries, mutations } from '../api/graphql-client';
import { useAuth } from '../auth/AuthContext';

// Types
export interface Email {
  id: string;
  subject: string;
  bodyText?: string;
  bodyHtml?: string;
  from: EmailAddress;
  to: EmailAddress[];
  cc: EmailAddress[];
  isRead: boolean;
  isStarred: boolean;
  isArchived: boolean;
  trustScore?: number;
  labels: string[];
  sentAt?: string;
  receivedAt?: string;
  threadId?: string;
  attachments: Attachment[];
}

export interface EmailAddress {
  email: string;
  name?: string;
}

export interface Attachment {
  id: string;
  filename: string;
  contentType: string;
  size: number;
}

export interface EmailFilter {
  isRead?: boolean;
  isStarred?: boolean;
  isArchived?: boolean;
  labels?: string[];
  minTrustScore?: number;
  search?: string;
}

export interface CreateEmailInput {
  subject: string;
  bodyText?: string;
  bodyHtml?: string;
  to: string[];
  cc?: string[];
  bcc?: string[];
  inReplyTo?: string;
}

// Query keys
export const emailKeys = {
  all: ['emails'] as const,
  lists: () => [...emailKeys.all, 'list'] as const,
  list: (filter: EmailFilter) => [...emailKeys.lists(), filter] as const,
  details: () => [...emailKeys.all, 'detail'] as const,
  detail: (id: string) => [...emailKeys.details(), id] as const,
  thread: (id: string) => [...emailKeys.all, 'thread', id] as const,
};

// Hooks

/**
 * Fetch paginated emails with filters
 */
export function useEmails(filter: EmailFilter = {}) {
  const { getAccessToken } = useAuth();

  return useInfiniteQuery({
    queryKey: emailKeys.list(filter),
    queryFn: async ({ pageParam = 1 }) => {
      const token = await getAccessToken();
      if (!token) throw new Error('Not authenticated');

      const response = await graphqlClient.query<{
        emails: {
          items: Email[];
          total: number;
          page: number;
          totalPages: number;
        };
      }>(queries.GET_EMAILS, {
        filter,
        page: pageParam,
        perPage: 50,
      });

      return response.emails;
    },
    getNextPageParam: (lastPage) => {
      if (lastPage.page < lastPage.totalPages) {
        return lastPage.page + 1;
      }
      return undefined;
    },
    initialPageParam: 1,
    staleTime: 30 * 1000, // 30 seconds
    refetchOnWindowFocus: true,
  });
}

/**
 * Fetch single email by ID
 */
export function useEmail(id: string) {
  const { getAccessToken } = useAuth();

  return useQuery({
    queryKey: emailKeys.detail(id),
    queryFn: async () => {
      const token = await getAccessToken();
      if (!token) throw new Error('Not authenticated');

      const response = await graphqlClient.query<{ email: Email }>(
        queries.GET_EMAIL,
        { id }
      );

      return response.email;
    },
    enabled: !!id,
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Fetch email thread
 */
export function useEmailThread(threadId: string) {
  const { getAccessToken } = useAuth();

  return useQuery({
    queryKey: emailKeys.thread(threadId),
    queryFn: async () => {
      const token = await getAccessToken();
      if (!token) throw new Error('Not authenticated');

      const response = await graphqlClient.query<{ thread: { emails: Email[] } }>(
        `query GetThread($id: ID!) {
          thread(id: $id) {
            id
            subject
            emails {
              id
              subject
              bodyText
              bodyHtml
              from { email name }
              to { email name }
              sentAt
              receivedAt
            }
          }
        }`,
        { id: threadId }
      );

      return response.thread;
    },
    enabled: !!threadId,
  });
}

/**
 * Create draft email
 */
export function useCreateEmail() {
  const queryClient = useQueryClient();
  const { getAccessToken } = useAuth();

  return useMutation({
    mutationFn: async (input: CreateEmailInput) => {
      const token = await getAccessToken();
      if (!token) throw new Error('Not authenticated');

      const response = await graphqlClient.mutate<{ createEmail: Email }>(
        mutations.CREATE_EMAIL,
        { input }
      );

      return response.createEmail;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: emailKeys.lists() });
    },
  });
}

/**
 * Send email
 */
export function useSendEmail() {
  const queryClient = useQueryClient();
  const { getAccessToken } = useAuth();

  return useMutation({
    mutationFn: async (id: string) => {
      const token = await getAccessToken();
      if (!token) throw new Error('Not authenticated');

      const response = await graphqlClient.mutate<{ sendEmail: Email }>(
        mutations.SEND_EMAIL,
        { id }
      );

      return response.sendEmail;
    },
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: emailKeys.lists() });
      queryClient.invalidateQueries({ queryKey: emailKeys.detail(id) });
    },
  });
}

/**
 * Mark email as read/unread
 */
export function useMarkEmailRead() {
  const queryClient = useQueryClient();
  const { getAccessToken } = useAuth();

  return useMutation({
    mutationFn: async ({ id, isRead }: { id: string; isRead: boolean }) => {
      const token = await getAccessToken();
      if (!token) throw new Error('Not authenticated');

      const response = await graphqlClient.mutate<{ updateEmail: Email }>(
        mutations.UPDATE_EMAIL,
        { id, input: { isRead } }
      );

      return response.updateEmail;
    },
    onMutate: async ({ id, isRead }) => {
      // Optimistic update
      await queryClient.cancelQueries({ queryKey: emailKeys.detail(id) });

      const previousEmail = queryClient.getQueryData<Email>(emailKeys.detail(id));

      queryClient.setQueryData<Email>(emailKeys.detail(id), (old) =>
        old ? { ...old, isRead } : old
      );

      return { previousEmail };
    },
    onError: (_, { id }, context) => {
      if (context?.previousEmail) {
        queryClient.setQueryData(emailKeys.detail(id), context.previousEmail);
      }
    },
    onSettled: (_, __, { id }) => {
      queryClient.invalidateQueries({ queryKey: emailKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: emailKeys.lists() });
    },
  });
}

/**
 * Star/unstar email
 */
export function useStarEmail() {
  const queryClient = useQueryClient();
  const { getAccessToken } = useAuth();

  return useMutation({
    mutationFn: async ({ id, isStarred }: { id: string; isStarred: boolean }) => {
      const token = await getAccessToken();
      if (!token) throw new Error('Not authenticated');

      const response = await graphqlClient.mutate<{ updateEmail: Email }>(
        mutations.UPDATE_EMAIL,
        { id, input: { isStarred } }
      );

      return response.updateEmail;
    },
    onMutate: async ({ id, isStarred }) => {
      await queryClient.cancelQueries({ queryKey: emailKeys.detail(id) });

      const previousEmail = queryClient.getQueryData<Email>(emailKeys.detail(id));

      queryClient.setQueryData<Email>(emailKeys.detail(id), (old) =>
        old ? { ...old, isStarred } : old
      );

      return { previousEmail };
    },
    onError: (_, { id }, context) => {
      if (context?.previousEmail) {
        queryClient.setQueryData(emailKeys.detail(id), context.previousEmail);
      }
    },
    onSettled: (_, __, { id }) => {
      queryClient.invalidateQueries({ queryKey: emailKeys.detail(id) });
      queryClient.invalidateQueries({ queryKey: emailKeys.lists() });
    },
  });
}

/**
 * Archive/unarchive email
 */
export function useArchiveEmail() {
  const queryClient = useQueryClient();
  const { getAccessToken } = useAuth();

  return useMutation({
    mutationFn: async ({ id, isArchived }: { id: string; isArchived: boolean }) => {
      const token = await getAccessToken();
      if (!token) throw new Error('Not authenticated');

      const response = await graphqlClient.mutate<{ updateEmail: Email }>(
        mutations.UPDATE_EMAIL,
        { id, input: { isArchived } }
      );

      return response.updateEmail;
    },
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: emailKeys.lists() });
      queryClient.invalidateQueries({ queryKey: emailKeys.detail(id) });
    },
  });
}

/**
 * Delete email (soft delete)
 */
export function useDeleteEmail() {
  const queryClient = useQueryClient();
  const { getAccessToken } = useAuth();

  return useMutation({
    mutationFn: async (id: string) => {
      const token = await getAccessToken();
      if (!token) throw new Error('Not authenticated');

      await graphqlClient.mutate(mutations.DELETE_EMAIL, { id });
    },
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: emailKeys.lists() });
      queryClient.removeQueries({ queryKey: emailKeys.detail(id) });
    },
  });
}

/**
 * Bulk mark as read
 */
export function useBulkMarkRead() {
  const queryClient = useQueryClient();
  const { getAccessToken } = useAuth();

  return useMutation({
    mutationFn: async ({ ids, isRead }: { ids: string[]; isRead: boolean }) => {
      const token = await getAccessToken();
      if (!token) throw new Error('Not authenticated');

      await graphqlClient.mutate(
        `mutation BulkMarkRead($ids: [ID!]!, $isRead: Boolean!) {
          bulkUpdateEmails(ids: $ids, input: { isRead: $isRead }) {
            count
          }
        }`,
        { ids, isRead }
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: emailKeys.all });
    },
  });
}
