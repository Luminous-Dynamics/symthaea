// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Email List Item Component
 *
 * Displays email preview with trust badge integration
 */

import React, { useCallback } from 'react';
import { useI18n } from '../lib/i18n';
import { useAccessibility } from '../lib/a11y/AccessibilityProvider';
import { useTrustScore } from '../lib/api/holochain-client';
import type { Email } from '../lib/hooks/useEmails';

// Trust Badge Component (inline for simplicity)
function TrustBadge({
  score,
  size = 'small',
  showLabel = false,
}: {
  score: number;
  size?: 'small' | 'medium' | 'large';
  showLabel?: boolean;
}) {
  const getColor = (s: number) => {
    if (s >= 0.8) return '#4CAF50';
    if (s >= 0.5) return '#FFC107';
    if (s >= 0.2) return '#FF9800';
    return '#F44336';
  };

  const getLabel = (s: number) => {
    if (s >= 0.8) return 'Trusted';
    if (s >= 0.5) return 'Known';
    if (s >= 0.2) return 'Caution';
    return 'Unknown';
  };

  const sizes = {
    small: { badge: 16, font: 10 },
    medium: { badge: 24, font: 12 },
    large: { badge: 32, font: 14 },
  };

  const { badge, font } = sizes[size];
  const color = getColor(score);
  const label = getLabel(score);

  return (
    <div
      style={{ display: 'flex', alignItems: 'center', gap: 4 }}
      role="img"
      aria-label={`Trust level: ${label} (${Math.round(score * 100)}%)`}
    >
      <div
        style={{
          width: badge,
          height: badge,
          borderRadius: '50%',
          backgroundColor: color,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
          fontSize: font,
          fontWeight: 'bold',
          flexShrink: 0,
        }}
      >
        {score >= 0.8 ? '✓' : score >= 0.5 ? '~' : score >= 0.2 ? '!' : '?'}
      </div>
      {showLabel && (
        <span style={{ fontSize: font, color: '#666' }}>
          {label}
        </span>
      )}
    </div>
  );
}

interface EmailListItemProps {
  email: Email;
  isSelected?: boolean;
  onSelect?: (id: string) => void;
  onStar?: (id: string, isStarred: boolean) => void;
  onMarkRead?: (id: string, isRead: boolean) => void;
  onArchive?: (id: string) => void;
  onDelete?: (id: string) => void;
}

export function EmailListItem({
  email,
  isSelected = false,
  onSelect,
  onStar,
  onMarkRead,
  onArchive,
  onDelete,
}: EmailListItemProps) {
  const { t, formatRelativeTime } = useI18n();
  const { reduceMotion, announce } = useAccessibility();

  // Get trust score from Holochain (if agent ID available)
  const { data: trustData } = useTrustScore(email.from.email);
  const trustScore = email.trustScore ?? trustData?.score ?? 0.5;

  const handleClick = useCallback(() => {
    onSelect?.(email.id);
  }, [email.id, onSelect]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case 'Enter':
        case ' ':
          e.preventDefault();
          onSelect?.(email.id);
          break;
        case 's':
          e.preventDefault();
          onStar?.(email.id, !email.isStarred);
          announce(email.isStarred ? t('email.unstar') : t('email.star'));
          break;
        case 'r':
          e.preventDefault();
          onMarkRead?.(email.id, !email.isRead);
          announce(email.isRead ? t('email.markUnread') : t('email.markRead'));
          break;
        case 'e':
          e.preventDefault();
          onArchive?.(email.id);
          announce(t('email.archive'));
          break;
        case 'Delete':
        case 'Backspace':
          e.preventDefault();
          onDelete?.(email.id);
          announce(t('email.delete'));
          break;
      }
    },
    [email, onSelect, onStar, onMarkRead, onArchive, onDelete, announce, t]
  );

  const handleStarClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onStar?.(email.id, !email.isStarred);
    },
    [email.id, email.isStarred, onStar]
  );

  return (
    <article
      role="listitem"
      tabIndex={0}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      aria-selected={isSelected}
      aria-label={`${email.isRead ? '' : 'Unread '}Email from ${email.from.name || email.from.email}: ${email.subject}`}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        padding: '12px 16px',
        borderBottom: '1px solid #e0e0e0',
        backgroundColor: isSelected ? '#e3f2fd' : email.isRead ? '#ffffff' : '#f5f5f5',
        cursor: 'pointer',
        transition: reduceMotion ? 'none' : 'background-color 0.15s ease',
        fontWeight: email.isRead ? 'normal' : 'bold',
      }}
    >
      {/* Star button */}
      <button
        onClick={handleStarClick}
        aria-label={email.isStarred ? t('email.unstar') : t('email.star')}
        aria-pressed={email.isStarred}
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: '4px',
          fontSize: '18px',
          color: email.isStarred ? '#FFC107' : '#9e9e9e',
        }}
      >
        {email.isStarred ? '★' : '☆'}
      </button>

      {/* Trust badge */}
      <TrustBadge score={trustScore} size="small" />

      {/* Sender */}
      <div
        style={{
          width: '180px',
          flexShrink: 0,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {email.from.name || email.from.email}
      </div>

      {/* Subject and preview */}
      <div
        style={{
          flex: 1,
          overflow: 'hidden',
          display: 'flex',
          gap: '8px',
        }}
      >
        <span
          style={{
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {email.subject || t('email.noSubject', { defaultValue: '(No subject)' })}
        </span>
        {email.bodyText && (
          <span
            style={{
              color: '#666',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              fontWeight: 'normal',
            }}
          >
            — {email.bodyText.slice(0, 100)}
          </span>
        )}
      </div>

      {/* Attachments indicator */}
      {email.attachments && email.attachments.length > 0 && (
        <span
          aria-label={`${email.attachments.length} ${t('email.attachments')}`}
          style={{ color: '#666' }}
        >
          📎 {email.attachments.length}
        </span>
      )}

      {/* Labels */}
      {email.labels && email.labels.length > 0 && (
        <div style={{ display: 'flex', gap: '4px' }}>
          {email.labels.slice(0, 2).map((label) => (
            <span
              key={label}
              style={{
                backgroundColor: '#e0e0e0',
                padding: '2px 6px',
                borderRadius: '4px',
                fontSize: '11px',
              }}
            >
              {label}
            </span>
          ))}
        </div>
      )}

      {/* Date */}
      <time
        dateTime={email.receivedAt || email.sentAt}
        style={{
          color: '#666',
          fontSize: '13px',
          whiteSpace: 'nowrap',
        }}
      >
        {formatRelativeTime(email.receivedAt || email.sentAt || new Date().toISOString())}
      </time>
    </article>
  );
}

export default EmailListItem;
