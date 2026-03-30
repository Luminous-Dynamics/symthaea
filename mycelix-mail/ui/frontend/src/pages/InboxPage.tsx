// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Inbox Page
 *
 * Main email list view with filtering and actions
 */

import React, { useState, useCallback, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useEmails, useMarkEmailRead, useStarEmail, useArchiveEmail, useDeleteEmail } from '../lib/hooks/useEmails';
import { useI18n } from '../lib/i18n';
import { useAccessibility, useKeyboardShortcut } from '../lib/a11y/AccessibilityProvider';
import { EmailListItem } from '../components/EmailListItem';

type FilterType = 'all' | 'unread' | 'starred' | 'archived';

export default function InboxPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { t } = useI18n();
  const { announce } = useAccessibility();

  // Filter state
  const [filter, setFilter] = useState<FilterType>(() => {
    const f = searchParams.get('filter');
    return (f as FilterType) || 'all';
  });

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [focusedIndex, setFocusedIndex] = useState(0);

  // Email query based on filter
  const emailFilter = {
    isRead: filter === 'unread' ? false : undefined,
    isStarred: filter === 'starred' ? true : undefined,
    isArchived: filter === 'archived' ? true : undefined,
  };

  const {
    data,
    isLoading,
    isError,
    error,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useEmails(emailFilter);

  // Mutations
  const markReadMutation = useMarkEmailRead();
  const starMutation = useStarEmail();
  const archiveMutation = useArchiveEmail();
  const deleteMutation = useDeleteEmail();

  // Flatten paginated data
  const emails = data?.pages.flatMap((page) => page.items) ?? [];
  const totalCount = data?.pages[0]?.total ?? 0;

  // Update URL when filter changes
  useEffect(() => {
    if (filter === 'all') {
      searchParams.delete('filter');
    } else {
      searchParams.set('filter', filter);
    }
    setSearchParams(searchParams, { replace: true });
  }, [filter, searchParams, setSearchParams]);

  // Handlers
  const handleSelectEmail = useCallback(
    (id: string) => {
      navigate(`/email/${id}`);
    },
    [navigate]
  );

  const handleStarEmail = useCallback(
    (id: string, isStarred: boolean) => {
      starMutation.mutate({ id, isStarred });
    },
    [starMutation]
  );

  const handleMarkRead = useCallback(
    (id: string, isRead: boolean) => {
      markReadMutation.mutate({ id, isRead });
    },
    [markReadMutation]
  );

  const handleArchiveEmail = useCallback(
    (id: string) => {
      archiveMutation.mutate({ id, isArchived: true });
      announce(t('email.archive'));
    },
    [archiveMutation, announce, t]
  );

  const handleDeleteEmail = useCallback(
    (id: string) => {
      if (window.confirm(t('email.confirmDelete', { defaultValue: 'Are you sure you want to delete this email?' }))) {
        deleteMutation.mutate(id);
        announce(t('email.delete'));
      }
    },
    [deleteMutation, announce, t]
  );

  const handleFilterChange = useCallback((newFilter: FilterType) => {
    setFilter(newFilter);
    setFocusedIndex(0);
  }, []);

  // Keyboard shortcuts
  useKeyboardShortcut('c', () => navigate('/compose'), { enabled: true });
  useKeyboardShortcut('/', () => {
    const searchInput = document.querySelector<HTMLInputElement>('[data-search-input]');
    searchInput?.focus();
  }, { enabled: true });

  // Keyboard navigation in list
  const handleListKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowDown':
        case 'j':
          e.preventDefault();
          setFocusedIndex((i) => Math.min(i + 1, emails.length - 1));
          break;
        case 'ArrowUp':
        case 'k':
          e.preventDefault();
          setFocusedIndex((i) => Math.max(i - 1, 0));
          break;
        case 'Home':
          e.preventDefault();
          setFocusedIndex(0);
          break;
        case 'End':
          e.preventDefault();
          setFocusedIndex(emails.length - 1);
          break;
      }
    },
    [emails.length]
  );

  // Focus management
  useEffect(() => {
    const listItems = document.querySelectorAll<HTMLElement>('[role="listitem"]');
    listItems[focusedIndex]?.focus();
  }, [focusedIndex]);

  // Loading state
  if (isLoading) {
    return (
      <div
        role="status"
        aria-label={t('common.loading')}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          padding: '2rem',
        }}
      >
        {t('common.loading')}
      </div>
    );
  }

  // Error state
  if (isError) {
    return (
      <div
        role="alert"
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          padding: '2rem',
          textAlign: 'center',
        }}
      >
        <p style={{ color: '#d32f2f', marginBottom: '1rem' }}>
          {error instanceof Error ? error.message : t('common.error')}
        </p>
        <button
          onClick={() => window.location.reload()}
          style={{
            padding: '0.5rem 1rem',
            cursor: 'pointer',
          }}
        >
          {t('common.retry')}
        </button>
      </div>
    );
  }

  return (
    <div className="inbox-page" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <header
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '16px',
          borderBottom: '1px solid #e0e0e0',
        }}
      >
        <h1 style={{ margin: 0, fontSize: '1.5rem' }}>{t('email.inbox')}</h1>

        {/* Filter tabs */}
        <nav role="tablist" aria-label="Email filters">
          {(['all', 'unread', 'starred', 'archived'] as FilterType[]).map((f) => (
            <button
              key={f}
              role="tab"
              aria-selected={filter === f}
              onClick={() => handleFilterChange(f)}
              style={{
                padding: '8px 16px',
                border: 'none',
                background: filter === f ? '#e3f2fd' : 'transparent',
                cursor: 'pointer',
                borderRadius: '4px',
                marginLeft: '4px',
              }}
            >
              {t(`email.${f === 'all' ? 'inbox' : f}`)}
            </button>
          ))}
        </nav>

        {/* Compose button */}
        <button
          onClick={() => navigate('/compose')}
          aria-label={t('email.compose')}
          style={{
            padding: '8px 16px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold',
          }}
        >
          + {t('email.compose')}
        </button>
      </header>

      {/* Email count */}
      <div
        style={{
          padding: '8px 16px',
          backgroundColor: '#f5f5f5',
          fontSize: '14px',
          color: '#666',
        }}
        aria-live="polite"
      >
        {totalCount} {totalCount === 1 ? 'email' : 'emails'}
      </div>

      {/* Email list */}
      <div
        role="list"
        aria-label="Email list"
        onKeyDown={handleListKeyDown}
        style={{
          flex: 1,
          overflow: 'auto',
        }}
      >
        {emails.length === 0 ? (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '200px',
              color: '#666',
            }}
          >
            {t('email.noEmails')}
          </div>
        ) : (
          emails.map((email, index) => (
            <EmailListItem
              key={email.id}
              email={email}
              isSelected={selectedIds.has(email.id)}
              onSelect={handleSelectEmail}
              onStar={handleStarEmail}
              onMarkRead={handleMarkRead}
              onArchive={handleArchiveEmail}
              onDelete={handleDeleteEmail}
            />
          ))
        )}

        {/* Load more */}
        {hasNextPage && (
          <div style={{ padding: '16px', textAlign: 'center' }}>
            <button
              onClick={() => fetchNextPage()}
              disabled={isFetchingNextPage}
              style={{
                padding: '8px 16px',
                cursor: 'pointer',
              }}
            >
              {isFetchingNextPage ? t('common.loading') : 'Load More'}
            </button>
          </div>
        )}
      </div>

      {/* Keyboard shortcuts hint */}
      <footer
        style={{
          padding: '8px 16px',
          backgroundColor: '#f5f5f5',
          fontSize: '12px',
          color: '#666',
          borderTop: '1px solid #e0e0e0',
        }}
      >
        Shortcuts: <kbd>c</kbd> compose | <kbd>j/k</kbd> navigate | <kbd>s</kbd> star | <kbd>e</kbd> archive | <kbd>/</kbd> search
      </footer>
    </div>
  );
}
