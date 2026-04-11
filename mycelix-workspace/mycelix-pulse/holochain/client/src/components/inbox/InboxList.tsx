// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * InboxList Component
 *
 * Paginated, filterable email list with bulk actions.
 */

import React, { useState, useCallback, useMemo } from 'react';
import { EmailListItem, Email } from './EmailListItem';

export interface InboxListProps {
  emails: Email[];
  loading?: boolean;
  error?: string;
  hasMore?: boolean;
  onLoadMore?: () => void;
  onEmailClick?: (email: Email) => void;
  onBulkAction?: (action: BulkAction, hashes: string[]) => void;
  onRefresh?: () => void;
}

export type BulkAction =
  | 'archive'
  | 'delete'
  | 'markRead'
  | 'markUnread'
  | 'star'
  | 'unstar'
  | 'spam'
  | 'notSpam';

export const InboxList: React.FC<InboxListProps> = ({
  emails,
  loading = false,
  error,
  hasMore = false,
  onLoadMore,
  onEmailClick,
  onBulkAction,
  onRefresh,
}) => {
  const [selectedHashes, setSelectedHashes] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState<'all' | 'unread' | 'starred'>('all');

  const filteredEmails = useMemo(() => {
    switch (filter) {
      case 'unread':
        return emails.filter((e) => !e.read);
      case 'starred':
        return emails.filter((e) => e.starred);
      default:
        return emails;
    }
  }, [emails, filter]);

  const allSelected = useMemo(
    () => filteredEmails.length > 0 && filteredEmails.every((e) => selectedHashes.has(e.hash)),
    [filteredEmails, selectedHashes]
  );

  const someSelected = useMemo(
    () => filteredEmails.some((e) => selectedHashes.has(e.hash)),
    [filteredEmails, selectedHashes]
  );

  const handleSelectAll = useCallback(() => {
    if (allSelected) {
      setSelectedHashes(new Set());
    } else {
      setSelectedHashes(new Set(filteredEmails.map((e) => e.hash)));
    }
  }, [allSelected, filteredEmails]);

  const handleSelect = useCallback((hash: string) => {
    setSelectedHashes((prev) => {
      const next = new Set(prev);
      if (next.has(hash)) {
        next.delete(hash);
      } else {
        next.add(hash);
      }
      return next;
    });
  }, []);

  const handleBulkAction = useCallback(
    (action: BulkAction) => {
      if (selectedHashes.size === 0) return;
      onBulkAction?.(action, Array.from(selectedHashes));
      setSelectedHashes(new Set());
    },
    [selectedHashes, onBulkAction]
  );

  const handleStar = useCallback(
    (hash: string, starred: boolean) => {
      onBulkAction?.(starred ? 'star' : 'unstar', [hash]);
    },
    [onBulkAction]
  );

  const handleArchive = useCallback(
    (hash: string) => {
      onBulkAction?.('archive', [hash]);
    },
    [onBulkAction]
  );

  const handleDelete = useCallback(
    (hash: string) => {
      onBulkAction?.('delete', [hash]);
    },
    [onBulkAction]
  );

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-gray-200 bg-gray-50">
        {/* Select all checkbox */}
        <input
          type="checkbox"
          checked={allSelected}
          ref={(input) => {
            if (input) {
              input.indeterminate = someSelected && !allSelected;
            }
          }}
          onChange={handleSelectAll}
          className="w-4 h-4 rounded border-gray-300 text-blue-600"
        />

        {/* Bulk actions (shown when items selected) */}
        {someSelected ? (
          <div className="flex items-center gap-1 ml-2">
            <span className="text-sm text-gray-600 mr-2">
              {selectedHashes.size} selected
            </span>
            <button
              onClick={() => handleBulkAction('archive')}
              className="p-1.5 rounded hover:bg-gray-200"
              title="Archive"
            >
              <ArchiveIcon className="w-4 h-4 text-gray-600" />
            </button>
            <button
              onClick={() => handleBulkAction('delete')}
              className="p-1.5 rounded hover:bg-gray-200"
              title="Delete"
            >
              <TrashIcon className="w-4 h-4 text-gray-600" />
            </button>
            <button
              onClick={() => handleBulkAction('markRead')}
              className="p-1.5 rounded hover:bg-gray-200"
              title="Mark as read"
            >
              <MailOpenIcon className="w-4 h-4 text-gray-600" />
            </button>
            <button
              onClick={() => handleBulkAction('markUnread')}
              className="p-1.5 rounded hover:bg-gray-200"
              title="Mark as unread"
            >
              <MailIcon className="w-4 h-4 text-gray-600" />
            </button>
            <button
              onClick={() => handleBulkAction('spam')}
              className="p-1.5 rounded hover:bg-gray-200"
              title="Report spam"
            >
              <AlertIcon className="w-4 h-4 text-gray-600" />
            </button>
          </div>
        ) : (
          <>
            {/* Refresh button */}
            <button
              onClick={onRefresh}
              disabled={loading}
              className={`p-1.5 rounded hover:bg-gray-200 ${loading ? 'opacity-50' : ''}`}
              title="Refresh"
            >
              <RefreshIcon className={`w-4 h-4 text-gray-600 ${loading ? 'animate-spin' : ''}`} />
            </button>

            {/* Filter tabs */}
            <div className="flex items-center ml-4 border rounded overflow-hidden">
              {(['all', 'unread', 'starred'] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={`px-3 py-1 text-sm capitalize ${
                    filter === f
                      ? 'bg-blue-500 text-white'
                      : 'bg-white text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  {f}
                </button>
              ))}
            </div>
          </>
        )}

        {/* Email count */}
        <div className="ml-auto text-sm text-gray-500">
          {filteredEmails.length} email{filteredEmails.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="px-4 py-3 bg-red-50 text-red-600 border-b border-red-100">
          {error}
        </div>
      )}

      {/* Email list */}
      <div className="flex-1 overflow-auto">
        {filteredEmails.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <InboxIcon className="w-16 h-16 mb-4" />
            <p className="text-lg">
              {filter === 'all'
                ? 'No emails yet'
                : filter === 'unread'
                ? 'No unread emails'
                : 'No starred emails'}
            </p>
          </div>
        ) : (
          <>
            {filteredEmails.map((email) => (
              <EmailListItem
                key={email.hash}
                email={email}
                selected={selectedHashes.has(email.hash)}
                onSelect={handleSelect}
                onClick={onEmailClick}
                onStar={handleStar}
                onArchive={handleArchive}
                onDelete={handleDelete}
              />
            ))}

            {/* Load more button */}
            {hasMore && (
              <div className="p-4 text-center">
                <button
                  onClick={onLoadMore}
                  disabled={loading}
                  className="px-4 py-2 text-blue-600 hover:bg-blue-50 rounded"
                >
                  {loading ? 'Loading...' : 'Load more'}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

// Icon components
const ArchiveIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="21 8 21 21 3 21 3 8" />
    <rect x="1" y="3" width="22" height="5" />
    <line x1="10" y1="12" x2="14" y2="12" />
  </svg>
);

const TrashIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
);

const MailOpenIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21.2 8.4c.5.38.8.97.8 1.6v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V10a2 2 0 0 1 .8-1.6l8-6a2 2 0 0 1 2.4 0l8 6Z" />
    <path d="m22 10-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 10" />
  </svg>
);

const MailIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="2" y="4" width="20" height="16" rx="2" />
    <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7" />
  </svg>
);

const AlertIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

const RefreshIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 2v6h-6" />
    <path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
    <path d="M3 22v-6h6" />
    <path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
  </svg>
);

const InboxIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="22 12 16 12 14 15 10 15 8 12 2 12" />
    <path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" />
  </svg>
);

export default InboxList;
