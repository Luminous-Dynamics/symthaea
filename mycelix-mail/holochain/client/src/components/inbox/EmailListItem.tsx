// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * EmailListItem Component
 *
 * Single email row in the inbox list.
 */

import React, { useCallback } from 'react';
import { Avatar } from '../common/Avatar';
import { TrustBadge } from '../common/TrustBadge';

export interface Email {
  hash: string;
  from: {
    address: string;
    name?: string;
  };
  to: Array<{ address: string; name?: string }>;
  subject: string;
  preview: string;
  receivedAt: number;
  read: boolean;
  starred: boolean;
  hasAttachments: boolean;
  labels: string[];
  trustLevel?: number;
  encrypted?: boolean;
  threadId?: string;
  threadCount?: number;
}

export interface EmailListItemProps {
  email: Email;
  selected?: boolean;
  onSelect?: (hash: string) => void;
  onClick?: (email: Email) => void;
  onStar?: (hash: string, starred: boolean) => void;
  onArchive?: (hash: string) => void;
  onDelete?: (hash: string) => void;
}

function formatDate(timestamp: number): string {
  const date = new Date(timestamp / 1000); // Convert from microseconds
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  // Today
  if (diff < 24 * 60 * 60 * 1000 && date.getDate() === now.getDate()) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  // This week
  if (diff < 7 * 24 * 60 * 60 * 1000) {
    return date.toLocaleDateString([], { weekday: 'short' });
  }

  // This year
  if (date.getFullYear() === now.getFullYear()) {
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }

  // Older
  return date.toLocaleDateString([], {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

export const EmailListItem: React.FC<EmailListItemProps> = ({
  email,
  selected = false,
  onSelect,
  onClick,
  onStar,
  onArchive,
  onDelete,
}) => {
  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      if ((e.target as HTMLElement).closest('[data-action]')) return;
      onClick?.(email);
    },
    [email, onClick]
  );

  const handleCheckbox = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      e.stopPropagation();
      onSelect?.(email.hash);
    },
    [email.hash, onSelect]
  );

  const handleStar = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onStar?.(email.hash, !email.starred);
    },
    [email.hash, email.starred, onStar]
  );

  const senderName = email.from.name || email.from.address.split('@')[0];

  return (
    <div
      className={`
        flex items-center gap-3 px-4 py-3 cursor-pointer
        border-b border-gray-100 transition-colors
        ${selected ? 'bg-blue-50' : 'hover:bg-gray-50'}
        ${!email.read ? 'bg-white font-semibold' : 'bg-gray-50'}
      `}
      onClick={handleClick}
    >
      {/* Checkbox */}
      <input
        type="checkbox"
        checked={selected}
        onChange={handleCheckbox}
        className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
        onClick={(e) => e.stopPropagation()}
      />

      {/* Star */}
      <button
        data-action="star"
        onClick={handleStar}
        className={`p-1 rounded hover:bg-gray-100 ${
          email.starred ? 'text-yellow-500' : 'text-gray-300 hover:text-gray-400'
        }`}
      >
        <StarIcon filled={email.starred} />
      </button>

      {/* Avatar */}
      <Avatar
        name={senderName}
        email={email.from.address}
        size={36}
      />

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`truncate ${!email.read ? 'text-gray-900' : 'text-gray-600'}`}>
            {senderName}
          </span>
          {email.trustLevel !== undefined && (
            <TrustBadge level={email.trustLevel} size="sm" />
          )}
          {email.encrypted && (
            <LockIcon className="w-4 h-4 text-green-500" />
          )}
          {email.threadCount && email.threadCount > 1 && (
            <span className="text-xs bg-gray-200 text-gray-600 px-1.5 py-0.5 rounded">
              {email.threadCount}
            </span>
          )}
        </div>

        <div className="flex items-center gap-1">
          <span className={`truncate ${!email.read ? 'text-gray-900' : 'text-gray-500'}`}>
            {email.subject || '(no subject)'}
          </span>
          <span className="text-gray-400 mx-1">-</span>
          <span className="text-gray-400 truncate flex-1">
            {email.preview}
          </span>
        </div>
      </div>

      {/* Labels */}
      {email.labels.length > 0 && (
        <div className="flex gap-1">
          {email.labels.slice(0, 2).map((label) => (
            <span
              key={label}
              className="text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-700"
            >
              {label}
            </span>
          ))}
          {email.labels.length > 2 && (
            <span className="text-xs text-gray-400">+{email.labels.length - 2}</span>
          )}
        </div>
      )}

      {/* Attachment indicator */}
      {email.hasAttachments && (
        <AttachmentIcon className="w-4 h-4 text-gray-400" />
      )}

      {/* Date */}
      <span className="text-sm text-gray-500 whitespace-nowrap">
        {formatDate(email.receivedAt)}
      </span>

      {/* Quick actions (shown on hover) */}
      <div className="hidden group-hover:flex items-center gap-1">
        <button
          data-action="archive"
          onClick={(e) => {
            e.stopPropagation();
            onArchive?.(email.hash);
          }}
          className="p-1 rounded hover:bg-gray-200"
          title="Archive"
        >
          <ArchiveIcon className="w-4 h-4 text-gray-500" />
        </button>
        <button
          data-action="delete"
          onClick={(e) => {
            e.stopPropagation();
            onDelete?.(email.hash);
          }}
          className="p-1 rounded hover:bg-gray-200"
          title="Delete"
        >
          <TrashIcon className="w-4 h-4 text-gray-500" />
        </button>
      </div>
    </div>
  );
};

// Icon components
const StarIcon: React.FC<{ filled?: boolean }> = ({ filled }) => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill={filled ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2">
    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
  </svg>
);

const LockIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
  </svg>
);

const AttachmentIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
  </svg>
);

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

export default EmailListItem;
