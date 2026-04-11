// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mobile-Optimized Email List Component
 *
 * Features:
 * - Swipe actions (archive, delete, mark read)
 * - Pull-to-refresh
 * - Infinite scroll
 * - Long press for selection
 * - Haptic feedback
 * - Safe area handling
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  usePullToRefresh,
  useSwipeActions,
  useLongPress,
  useSafeAreaInsets,
  useViewportHeight,
  SwipeAction,
} from './useMobileGestures';

export interface Email {
  hash: string;
  from: {
    name?: string;
    email: string;
  };
  subject: string;
  preview: string;
  timestamp: number;
  isRead: boolean;
  isStarred: boolean;
  hasAttachments: boolean;
  trustLevel?: number;
  labels?: string[];
}

export interface MobileEmailListProps {
  emails: Email[];
  onRefresh: () => Promise<void>;
  onEmailClick: (email: Email) => void;
  onArchive: (hash: string) => Promise<void>;
  onDelete: (hash: string) => Promise<void>;
  onToggleRead: (hash: string) => Promise<void>;
  onToggleStar: (hash: string) => Promise<void>;
  onSelectionChange?: (selectedIds: Set<string>) => void;
  isLoading?: boolean;
  hasMore?: boolean;
  onLoadMore?: () => void;
}

// Swipe action configurations
const LEFT_ACTIONS: SwipeAction[] = [
  {
    id: 'archive',
    label: 'Archive',
    icon: '📥',
    color: '#ffffff',
    backgroundColor: '#22c55e',
  },
];

const RIGHT_ACTIONS: SwipeAction[] = [
  {
    id: 'delete',
    label: 'Delete',
    icon: '🗑️',
    color: '#ffffff',
    backgroundColor: '#ef4444',
  },
];

/**
 * Individual email list item with swipe actions
 */
interface EmailListItemProps {
  email: Email;
  isSelected: boolean;
  onAction: (actionId: string) => void;
  onClick: () => void;
  onLongPress: () => void;
}

const EmailListItem: React.FC<EmailListItemProps> = ({
  email,
  isSelected,
  onAction,
  onClick,
  onLongPress,
}) => {
  const swipeRef = useSwipeActions<HTMLDivElement>(onAction, {
    leftActions: LEFT_ACTIONS,
    rightActions: RIGHT_ACTIONS,
    actionThreshold: 80,
  });

  const longPressRef = useLongPress<HTMLDivElement>(onLongPress, 500);

  // Merge refs
  const mergedRef = useCallback(
    (node: HTMLDivElement | null) => {
      (swipeRef as React.MutableRefObject<HTMLDivElement | null>).current = node;
      (longPressRef as React.MutableRefObject<HTMLDivElement | null>).current = node;
    },
    [swipeRef, longPressRef]
  );

  const formatTime = (timestamp: number): string => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return date.toLocaleDateString([], { weekday: 'short' });
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };

  const getTrustColor = (level?: number): string => {
    if (!level) return '#94a3b8';
    if (level >= 0.8) return '#22c55e';
    if (level >= 0.6) return '#3b82f6';
    if (level >= 0.4) return '#eab308';
    if (level >= 0.2) return '#f97316';
    return '#ef4444';
  };

  return (
    <div
      ref={mergedRef}
      onClick={onClick}
      className={`
        email-list-item
        ${!email.isRead ? 'unread' : ''}
        ${isSelected ? 'selected' : ''}
      `}
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        padding: '12px 16px',
        borderBottom: '1px solid #e2e8f0',
        backgroundColor: isSelected ? '#eff6ff' : email.isRead ? '#ffffff' : '#f8fafc',
        cursor: 'pointer',
        touchAction: 'pan-y',
        userSelect: 'none',
      }}
    >
      {/* Selection indicator / Avatar */}
      <div
        style={{
          width: 40,
          height: 40,
          borderRadius: '50%',
          backgroundColor: isSelected ? '#3b82f6' : getTrustColor(email.trustLevel),
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#ffffff',
          fontSize: 16,
          fontWeight: 500,
          marginRight: 12,
          flexShrink: 0,
        }}
      >
        {isSelected ? '✓' : (email.from.name?.[0] || email.from.email[0]).toUpperCase()}
      </div>

      {/* Content */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Header row */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
          <span
            style={{
              fontWeight: email.isRead ? 400 : 600,
              fontSize: 15,
              color: '#0f172a',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            {email.from.name || email.from.email}
          </span>
          <span
            style={{
              fontSize: 12,
              color: '#64748b',
              marginLeft: 8,
              flexShrink: 0,
            }}
          >
            {formatTime(email.timestamp)}
          </span>
        </div>

        {/* Subject */}
        <div
          style={{
            fontWeight: email.isRead ? 400 : 500,
            fontSize: 14,
            color: '#1e293b',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            marginBottom: 2,
          }}
        >
          {email.subject}
        </div>

        {/* Preview */}
        <div
          style={{
            fontSize: 13,
            color: '#64748b',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {email.preview}
        </div>

        {/* Indicators */}
        <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
          {email.isStarred && <span style={{ fontSize: 12 }}>⭐</span>}
          {email.hasAttachments && <span style={{ fontSize: 12 }}>📎</span>}
          {email.labels?.map((label) => (
            <span
              key={label}
              style={{
                fontSize: 10,
                padding: '2px 6px',
                borderRadius: 4,
                backgroundColor: '#e2e8f0',
                color: '#475569',
              }}
            >
              {label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

/**
 * Main mobile email list component
 */
export const MobileEmailList: React.FC<MobileEmailListProps> = ({
  emails,
  onRefresh,
  onEmailClick,
  onArchive,
  onDelete,
  onToggleRead,
  onToggleStar,
  onSelectionChange,
  isLoading = false,
  hasMore = false,
  onLoadMore,
}) => {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const insets = useSafeAreaInsets();
  const viewportHeight = useViewportHeight();

  const containerRef = usePullToRefresh<HTMLDivElement>(onRefresh);

  const isSelectionMode = selectedIds.size > 0;

  const handleAction = useCallback(
    async (email: Email, actionId: string) => {
      switch (actionId) {
        case 'archive':
          await onArchive(email.hash);
          break;
        case 'delete':
          await onDelete(email.hash);
          break;
        case 'toggleRead':
          await onToggleRead(email.hash);
          break;
        case 'toggleStar':
          await onToggleStar(email.hash);
          break;
      }
    },
    [onArchive, onDelete, onToggleRead, onToggleStar]
  );

  const handleClick = useCallback(
    (email: Email) => {
      if (isSelectionMode) {
        // Toggle selection
        setSelectedIds((prev) => {
          const newSet = new Set(prev);
          if (newSet.has(email.hash)) {
            newSet.delete(email.hash);
          } else {
            newSet.add(email.hash);
          }
          onSelectionChange?.(newSet);
          return newSet;
        });
      } else {
        onEmailClick(email);
      }
    },
    [isSelectionMode, onEmailClick, onSelectionChange]
  );

  const handleLongPress = useCallback(
    (email: Email) => {
      setSelectedIds((prev) => {
        const newSet = new Set(prev);
        newSet.add(email.hash);
        onSelectionChange?.(newSet);
        return newSet;
      });
    },
    [onSelectionChange]
  );

  const handleScroll = useCallback(
    (e: React.UIEvent<HTMLDivElement>) => {
      if (!hasMore || isLoading || !onLoadMore) return;

      const target = e.target as HTMLDivElement;
      const scrolledToBottom =
        target.scrollHeight - target.scrollTop <= target.clientHeight + 100;

      if (scrolledToBottom) {
        onLoadMore();
      }
    },
    [hasMore, isLoading, onLoadMore]
  );

  const clearSelection = useCallback(() => {
    setSelectedIds(new Set());
    onSelectionChange?.(new Set());
  }, [onSelectionChange]);

  const selectAll = useCallback(() => {
    const allIds = new Set(emails.map((e) => e.hash));
    setSelectedIds(allIds);
    onSelectionChange?.(allIds);
  }, [emails, onSelectionChange]);

  // Batch actions for selection mode
  const handleBatchArchive = useCallback(async () => {
    await Promise.all(Array.from(selectedIds).map((id) => onArchive(id)));
    clearSelection();
  }, [selectedIds, onArchive, clearSelection]);

  const handleBatchDelete = useCallback(async () => {
    await Promise.all(Array.from(selectedIds).map((id) => onDelete(id)));
    clearSelection();
  }, [selectedIds, onDelete, clearSelection]);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: `calc(${viewportHeight}px - ${insets.top}px - ${insets.bottom}px)`,
        backgroundColor: '#ffffff',
      }}
    >
      {/* Selection mode header */}
      {isSelectionMode && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '12px 16px',
            backgroundColor: '#3b82f6',
            color: '#ffffff',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <button
              onClick={clearSelection}
              style={{
                background: 'none',
                border: 'none',
                color: '#ffffff',
                fontSize: 20,
                cursor: 'pointer',
              }}
            >
              ✕
            </button>
            <span style={{ fontWeight: 500 }}>{selectedIds.size} selected</span>
          </div>
          <div style={{ display: 'flex', gap: 16 }}>
            <button
              onClick={selectAll}
              style={{
                background: 'none',
                border: 'none',
                color: '#ffffff',
                cursor: 'pointer',
              }}
            >
              Select All
            </button>
            <button
              onClick={handleBatchArchive}
              style={{
                background: 'none',
                border: 'none',
                color: '#ffffff',
                fontSize: 20,
                cursor: 'pointer',
              }}
            >
              📥
            </button>
            <button
              onClick={handleBatchDelete}
              style={{
                background: 'none',
                border: 'none',
                color: '#ffffff',
                fontSize: 20,
                cursor: 'pointer',
              }}
            >
              🗑️
            </button>
          </div>
        </div>
      )}

      {/* Email list */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        style={{
          flex: 1,
          overflowY: 'auto',
          WebkitOverflowScrolling: 'touch',
        }}
      >
        {emails.length === 0 ? (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              padding: 32,
              color: '#64748b',
            }}
          >
            <span style={{ fontSize: 48, marginBottom: 16 }}>📭</span>
            <span style={{ fontSize: 16, fontWeight: 500 }}>No emails</span>
            <span style={{ fontSize: 14, marginTop: 4 }}>
              Pull down to refresh
            </span>
          </div>
        ) : (
          emails.map((email) => (
            <EmailListItem
              key={email.hash}
              email={email}
              isSelected={selectedIds.has(email.hash)}
              onAction={(actionId) => handleAction(email, actionId)}
              onClick={() => handleClick(email)}
              onLongPress={() => handleLongPress(email)}
            />
          ))
        )}

        {/* Loading indicator */}
        {isLoading && (
          <div
            style={{
              display: 'flex',
              justifyContent: 'center',
              padding: 16,
            }}
          >
            <div
              style={{
                width: 24,
                height: 24,
                border: '2px solid #e2e8f0',
                borderTopColor: '#3b82f6',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite',
              }}
            />
          </div>
        )}

        {/* End of list indicator */}
        {!hasMore && emails.length > 0 && (
          <div
            style={{
              textAlign: 'center',
              padding: 16,
              color: '#94a3b8',
              fontSize: 12,
            }}
          >
            End of messages
          </div>
        )}
      </div>

      {/* Floating action button */}
      {!isSelectionMode && (
        <button
          style={{
            position: 'fixed',
            right: 16,
            bottom: 16 + insets.bottom,
            width: 56,
            height: 56,
            borderRadius: '50%',
            backgroundColor: '#3b82f6',
            color: '#ffffff',
            border: 'none',
            fontSize: 24,
            cursor: 'pointer',
            boxShadow: '0 4px 12px rgba(59, 130, 246, 0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onClick={() => {
            // Navigate to compose
            console.log('Compose new email');
          }}
        >
          ✏️
        </button>
      )}

      {/* Styles */}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default MobileEmailList;
