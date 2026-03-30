// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Virtual Email List
 *
 * High-performance virtualized list for rendering thousands of emails
 * Uses windowing to only render visible items
 */

import React, { useCallback, useRef, useState, useEffect, useMemo } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useInfiniteQuery } from '@tanstack/react-query';
import { graphqlClient } from '../lib/api/graphql-client';
import { useAccessibility } from '../lib/a11y/AccessibilityProvider';
import EmailListItem from './EmailListItem';

// Types
interface Email {
  id: string;
  subject: string;
  bodyPreview: string;
  from: { email: string; name?: string };
  isRead: boolean;
  isStarred: boolean;
  isEncrypted: boolean;
  trustScore?: number;
  labels: string[];
  receivedAt: string;
  hasAttachments: boolean;
}

interface EmailPage {
  emails: Email[];
  nextCursor?: string;
  hasMore: boolean;
  total: number;
}

interface VirtualEmailListProps {
  folderId?: string;
  searchQuery?: string;
  onEmailSelect: (email: Email) => void;
  selectedEmailId?: string;
}

// Row height for consistent virtualization
const ROW_HEIGHT = 72;
const OVERSCAN = 5;

export function VirtualEmailList({
  folderId = 'inbox',
  searchQuery,
  onEmailSelect,
  selectedEmailId,
}: VirtualEmailListProps) {
  const parentRef = useRef<HTMLDivElement>(null);
  const { announce } = useAccessibility();
  const [focusedIndex, setFocusedIndex] = useState(0);

  // Infinite query for paginated emails
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    isError,
  } = useInfiniteQuery({
    queryKey: ['emails', folderId, searchQuery],
    queryFn: async ({ pageParam }) => {
      const response = await graphqlClient.query<{ emails: EmailPage }>(
        `query GetEmails($folderId: ID!, $cursor: String, $search: String) {
          emails(folderId: $folderId, cursor: $cursor, search: $search, limit: 50) {
            emails {
              id subject bodyPreview
              from { email name }
              isRead isStarred isEncrypted trustScore
              labels receivedAt hasAttachments
            }
            nextCursor hasMore total
          }
        }`,
        { folderId, cursor: pageParam, search: searchQuery }
      );
      return response.emails;
    },
    getNextPageParam: (lastPage) => lastPage.nextCursor,
    initialPageParam: undefined as string | undefined,
  });

  // Flatten all pages into single array
  const allEmails = useMemo(
    () => data?.pages.flatMap((page) => page.emails) ?? [],
    [data]
  );

  const totalCount = data?.pages[0]?.total ?? 0;

  // Virtual list configuration
  const virtualizer = useVirtualizer({
    count: hasNextPage ? allEmails.length + 1 : allEmails.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: OVERSCAN,
  });

  const virtualItems = virtualizer.getVirtualItems();

  // Load more when scrolling near bottom
  useEffect(() => {
    const lastItem = virtualItems[virtualItems.length - 1];
    if (!lastItem) return;

    if (
      lastItem.index >= allEmails.length - 1 &&
      hasNextPage &&
      !isFetchingNextPage
    ) {
      fetchNextPage();
    }
  }, [virtualItems, allEmails.length, hasNextPage, isFetchingNextPage, fetchNextPage]);

  // Keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setFocusedIndex((prev) => Math.min(prev + 1, allEmails.length - 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setFocusedIndex((prev) => Math.max(prev - 1, 0));
          break;
        case 'Enter':
        case ' ':
          e.preventDefault();
          if (allEmails[focusedIndex]) {
            onEmailSelect(allEmails[focusedIndex]);
            announce(`Selected email: ${allEmails[focusedIndex].subject}`);
          }
          break;
        case 'Home':
          e.preventDefault();
          setFocusedIndex(0);
          virtualizer.scrollToIndex(0);
          break;
        case 'End':
          e.preventDefault();
          setFocusedIndex(allEmails.length - 1);
          virtualizer.scrollToIndex(allEmails.length - 1);
          break;
      }
    },
    [allEmails, focusedIndex, onEmailSelect, announce, virtualizer]
  );

  // Scroll focused item into view
  useEffect(() => {
    virtualizer.scrollToIndex(focusedIndex, { align: 'auto' });
  }, [focusedIndex, virtualizer]);

  if (isLoading) {
    return (
      <div
        role="status"
        aria-live="polite"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '200px',
          color: '#666',
        }}
      >
        Loading emails...
      </div>
    );
  }

  if (isError) {
    return (
      <div
        role="alert"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '200px',
          color: '#d32f2f',
        }}
      >
        Failed to load emails. Please try again.
      </div>
    );
  }

  if (allEmails.length === 0) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '300px',
          color: '#666',
        }}
      >
        <span style={{ fontSize: '48px', marginBottom: '16px' }}>📭</span>
        <p>No emails found</p>
      </div>
    );
  }

  return (
    <div
      ref={parentRef}
      role="listbox"
      aria-label={`Email list, ${totalCount} emails`}
      aria-activedescendant={allEmails[focusedIndex]?.id}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      style={{
        height: '100%',
        overflow: 'auto',
        contain: 'strict',
      }}
    >
      {/* Status bar */}
      <div
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 10,
          backgroundColor: '#f5f5f5',
          padding: '8px 16px',
          borderBottom: '1px solid #e0e0e0',
          fontSize: '12px',
          color: '#666',
        }}
      >
        {searchQuery ? (
          <span>
            Found {totalCount} emails matching &quot;{searchQuery}&quot;
          </span>
        ) : (
          <span>
            {totalCount} emails in {folderId}
          </span>
        )}
      </div>

      {/* Virtual list container */}
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualItems.map((virtualRow) => {
          const isLoaderRow = virtualRow.index >= allEmails.length;
          const email = allEmails[virtualRow.index];

          if (isLoaderRow) {
            return (
              <div
                key="loader"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: `${virtualRow.size}px`,
                  transform: `translateY(${virtualRow.start}px)`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#666',
                }}
              >
                {isFetchingNextPage ? 'Loading more...' : 'Load more'}
              </div>
            );
          }

          return (
            <div
              key={email.id}
              role="option"
              aria-selected={email.id === selectedEmailId}
              data-index={virtualRow.index}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              <EmailListItem
                email={email}
                isSelected={email.id === selectedEmailId}
                isFocused={virtualRow.index === focusedIndex}
                onClick={() => onEmailSelect(email)}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Skeleton loader for initial load
export function EmailListSkeleton({ count = 10 }: { count?: number }) {
  return (
    <div role="status" aria-label="Loading emails">
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          style={{
            height: `${ROW_HEIGHT}px`,
            padding: '12px 16px',
            borderBottom: '1px solid #e0e0e0',
            display: 'flex',
            gap: '12px',
            animation: 'pulse 1.5s ease-in-out infinite',
          }}
        >
          <div
            style={{
              width: '40px',
              height: '40px',
              borderRadius: '50%',
              backgroundColor: '#e0e0e0',
            }}
          />
          <div style={{ flex: 1 }}>
            <div
              style={{
                width: '30%',
                height: '14px',
                backgroundColor: '#e0e0e0',
                marginBottom: '8px',
                borderRadius: '4px',
              }}
            />
            <div
              style={{
                width: '80%',
                height: '12px',
                backgroundColor: '#e0e0e0',
                borderRadius: '4px',
              }}
            />
          </div>
        </div>
      ))}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}

export default VirtualEmailList;
