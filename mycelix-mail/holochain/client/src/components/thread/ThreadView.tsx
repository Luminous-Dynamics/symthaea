// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * ThreadView Component
 *
 * Displays a conversation thread with all messages.
 */

import React, { useState, useCallback, useMemo } from 'react';
import { Avatar } from '../common/Avatar';
import { TrustBadge } from '../common/TrustBadge';

export interface ThreadMessage {
  hash: string;
  from: {
    address: string;
    name?: string;
    trustLevel?: number;
  };
  to: Array<{ address: string; name?: string }>;
  cc?: Array<{ address: string; name?: string }>;
  subject: string;
  body: string;
  bodyHtml?: string;
  sentAt: number;
  read: boolean;
  encrypted?: boolean;
  attachments?: Array<{
    id: string;
    name: string;
    size: number;
    type: string;
  }>;
}

export interface ThreadViewProps {
  /** Thread subject */
  subject: string;
  /** Messages in the thread (ordered oldest first) */
  messages: ThreadMessage[];
  /** Current user's email */
  currentUserEmail: string;
  /** Labels on this thread */
  labels?: string[];
  /** Loading state */
  loading?: boolean;
  /** Reply callback */
  onReply?: (messageHash: string, replyAll: boolean) => void;
  /** Forward callback */
  onForward?: (messageHash: string) => void;
  /** Archive callback */
  onArchive?: () => void;
  /** Delete callback */
  onDelete?: () => void;
  /** Add label callback */
  onAddLabel?: (label: string) => void;
  /** Remove label callback */
  onRemoveLabel?: (label: string) => void;
  /** Mark as spam callback */
  onSpam?: () => void;
  /** Download attachment callback */
  onDownloadAttachment?: (messageHash: string, attachmentId: string) => void;
  /** Back callback */
  onBack?: () => void;
}

export const ThreadView: React.FC<ThreadViewProps> = ({
  subject,
  messages,
  currentUserEmail,
  labels = [],
  loading = false,
  onReply,
  onForward,
  onArchive,
  onDelete,
  onAddLabel,
  onRemoveLabel,
  onSpam,
  onDownloadAttachment,
  onBack,
}) => {
  const [expandedMessages, setExpandedMessages] = useState<Set<string>>(() => {
    // Expand the last message and any unread messages
    const expanded = new Set<string>();
    if (messages.length > 0) {
      expanded.add(messages[messages.length - 1].hash);
      messages.forEach((m) => {
        if (!m.read) expanded.add(m.hash);
      });
    }
    return expanded;
  });

  const toggleExpand = useCallback((hash: string) => {
    setExpandedMessages((prev) => {
      const next = new Set(prev);
      if (next.has(hash)) {
        next.delete(hash);
      } else {
        next.add(hash);
      }
      return next;
    });
  }, []);

  const expandAll = useCallback(() => {
    setExpandedMessages(new Set(messages.map((m) => m.hash)));
  }, [messages]);

  const collapseAll = useCallback(() => {
    // Keep last message expanded
    if (messages.length > 0) {
      setExpandedMessages(new Set([messages[messages.length - 1].hash]));
    } else {
      setExpandedMessages(new Set());
    }
  }, [messages]);

  const participantCount = useMemo(() => {
    const participants = new Set<string>();
    messages.forEach((m) => {
      participants.add(m.from.address);
      m.to.forEach((r) => participants.add(r.address));
      m.cc?.forEach((r) => participants.add(r.address));
    });
    return participants.size;
  }, [messages]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <SpinnerIcon className="w-8 h-8 animate-spin text-gray-400" />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="flex items-start gap-4 px-6 py-4 border-b">
        {/* Back button */}
        {onBack && (
          <button
            onClick={onBack}
            className="p-2 hover:bg-gray-100 rounded mt-1"
          >
            <ArrowLeftIcon className="w-5 h-5" />
          </button>
        )}

        <div className="flex-1">
          <h1 className="text-xl font-semibold text-gray-900">{subject}</h1>
          <div className="flex items-center gap-2 mt-1 text-sm text-gray-500">
            <span>{messages.length} message{messages.length !== 1 ? 's' : ''}</span>
            <span>|</span>
            <span>{participantCount} participant{participantCount !== 1 ? 's' : ''}</span>
          </div>

          {/* Labels */}
          {labels.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {labels.map((label) => (
                <span
                  key={label}
                  className="inline-flex items-center gap-1 px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full text-sm"
                >
                  {label}
                  {onRemoveLabel && (
                    <button
                      onClick={() => onRemoveLabel(label)}
                      className="hover:text-blue-900"
                    >
                      <CloseIcon className="w-3 h-3" />
                    </button>
                  )}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1">
          <button
            onClick={expandAll}
            className="px-2 py-1 text-sm text-gray-600 hover:bg-gray-100 rounded"
          >
            Expand all
          </button>
          <button
            onClick={collapseAll}
            className="px-2 py-1 text-sm text-gray-600 hover:bg-gray-100 rounded"
          >
            Collapse
          </button>
          <div className="w-px h-4 bg-gray-300 mx-2" />
          <button
            onClick={onArchive}
            className="p-2 hover:bg-gray-100 rounded"
            title="Archive"
          >
            <ArchiveIcon className="w-5 h-5 text-gray-600" />
          </button>
          <button
            onClick={onSpam}
            className="p-2 hover:bg-gray-100 rounded"
            title="Report spam"
          >
            <AlertIcon className="w-5 h-5 text-gray-600" />
          </button>
          <button
            onClick={onDelete}
            className="p-2 hover:bg-gray-100 rounded"
            title="Delete"
          >
            <TrashIcon className="w-5 h-5 text-gray-600" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-auto px-6 py-4 space-y-4">
        {messages.map((message, index) => (
          <MessageCard
            key={message.hash}
            message={message}
            expanded={expandedMessages.has(message.hash)}
            onToggle={() => toggleExpand(message.hash)}
            onReply={(replyAll) => onReply?.(message.hash, replyAll)}
            onForward={() => onForward?.(message.hash)}
            onDownloadAttachment={(attachmentId) =>
              onDownloadAttachment?.(message.hash, attachmentId)
            }
            isLast={index === messages.length - 1}
            currentUserEmail={currentUserEmail}
          />
        ))}
      </div>

      {/* Quick reply */}
      <div className="px-6 py-4 border-t bg-gray-50">
        <button
          onClick={() => onReply?.(messages[messages.length - 1]?.hash, false)}
          className="w-full px-4 py-3 border border-gray-300 rounded-lg text-left text-gray-500 hover:bg-white"
        >
          Click here to reply...
        </button>
      </div>
    </div>
  );
};

// Message Card Component
interface MessageCardProps {
  message: ThreadMessage;
  expanded: boolean;
  onToggle: () => void;
  onReply: (replyAll: boolean) => void;
  onForward: () => void;
  onDownloadAttachment: (attachmentId: string) => void;
  isLast: boolean;
  currentUserEmail: string;
}

const MessageCard: React.FC<MessageCardProps> = ({
  message,
  expanded,
  onToggle,
  onReply,
  onForward,
  onDownloadAttachment,
  isLast,
  currentUserEmail,
}) => {
  const isFromMe = message.from.address === currentUserEmail;
  const senderName = message.from.name || message.from.address.split('@')[0];

  const recipientSummary = useMemo(() => {
    const toNames = message.to.map((r) => r.name || r.address.split('@')[0]);
    if (toNames.length === 1) return `to ${toNames[0]}`;
    if (toNames.length === 2) return `to ${toNames[0]} and ${toNames[1]}`;
    return `to ${toNames[0]} and ${toNames.length - 1} others`;
  }, [message.to]);

  return (
    <div className={`border rounded-lg ${expanded ? 'bg-white' : 'bg-gray-50'}`}>
      {/* Header - always visible */}
      <div
        className="flex items-start gap-3 p-4 cursor-pointer hover:bg-gray-50"
        onClick={onToggle}
      >
        <Avatar
          name={senderName}
          email={message.from.address}
          size={40}
        />

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-900">
              {isFromMe ? 'Me' : senderName}
            </span>
            {message.from.trustLevel !== undefined && (
              <TrustBadge level={message.from.trustLevel} size="sm" />
            )}
            {message.encrypted && (
              <LockIcon className="w-4 h-4 text-green-500" />
            )}
          </div>
          <div className="text-sm text-gray-500">{recipientSummary}</div>
        </div>

        <div className="text-sm text-gray-500 whitespace-nowrap">
          {formatDate(message.sentAt)}
        </div>

        <ChevronIcon
          className={`w-5 h-5 text-gray-400 transition-transform ${
            expanded ? 'rotate-180' : ''
          }`}
        />
      </div>

      {/* Body - shown when expanded */}
      {expanded && (
        <div className="px-4 pb-4">
          {/* Full recipient list */}
          <div className="text-sm text-gray-500 mb-4 pl-13">
            <div>
              <span className="text-gray-400">To:</span>{' '}
              {message.to.map((r) => r.name || r.address).join(', ')}
            </div>
            {message.cc && message.cc.length > 0 && (
              <div>
                <span className="text-gray-400">Cc:</span>{' '}
                {message.cc.map((r) => r.name || r.address).join(', ')}
              </div>
            )}
          </div>

          {/* Message body */}
          <div className="prose prose-sm max-w-none pl-13">
            {message.bodyHtml ? (
              <div dangerouslySetInnerHTML={{ __html: message.bodyHtml }} />
            ) : (
              <pre className="whitespace-pre-wrap font-sans">{message.body}</pre>
            )}
          </div>

          {/* Attachments */}
          {message.attachments && message.attachments.length > 0 && (
            <div className="mt-4 pl-13">
              <div className="text-sm font-medium text-gray-700 mb-2">
                Attachments ({message.attachments.length})
              </div>
              <div className="flex flex-wrap gap-2">
                {message.attachments.map((attachment) => (
                  <button
                    key={attachment.id}
                    onClick={() => onDownloadAttachment(attachment.id)}
                    className="flex items-center gap-2 px-3 py-2 border rounded hover:bg-gray-50"
                  >
                    <AttachmentIcon className="w-4 h-4 text-gray-500" />
                    <span className="text-sm">{attachment.name}</span>
                    <span className="text-xs text-gray-400">
                      ({formatSize(attachment.size)})
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 mt-4 pl-13">
            <button
              onClick={() => onReply(false)}
              className="flex items-center gap-1 px-3 py-1.5 border rounded hover:bg-gray-50 text-sm"
            >
              <ReplyIcon className="w-4 h-4" />
              Reply
            </button>
            {message.to.length > 1 && (
              <button
                onClick={() => onReply(true)}
                className="flex items-center gap-1 px-3 py-1.5 border rounded hover:bg-gray-50 text-sm"
              >
                <ReplyAllIcon className="w-4 h-4" />
                Reply all
              </button>
            )}
            <button
              onClick={onForward}
              className="flex items-center gap-1 px-3 py-1.5 border rounded hover:bg-gray-50 text-sm"
            >
              <ForwardIcon className="w-4 h-4" />
              Forward
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper functions
function formatDate(timestamp: number): string {
  const date = new Date(timestamp / 1000);
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  if (diff < 24 * 60 * 60 * 1000) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  if (diff < 7 * 24 * 60 * 60 * 1000) {
    return date.toLocaleDateString([], {
      weekday: 'short',
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  return date.toLocaleDateString([], {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Icon components
const ArrowLeftIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="19" y1="12" x2="5" y2="12" />
    <polyline points="12 19 5 12 12 5" />
  </svg>
);

const ChevronIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

const CloseIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

const ArchiveIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="21 8 21 21 3 21 3 8" />
    <rect x="1" y="3" width="22" height="5" />
  </svg>
);

const AlertIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

const TrashIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
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

const ReplyIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="9 17 4 12 9 7" />
    <path d="M20 18v-2a4 4 0 0 0-4-4H4" />
  </svg>
);

const ReplyAllIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="7 17 2 12 7 7" />
    <polyline points="12 17 7 12 12 7" />
    <path d="M22 18v-2a4 4 0 0 0-4-4H7" />
  </svg>
);

const ForwardIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="15 17 20 12 15 7" />
    <path d="M4 18v-2a4 4 0 0 1 4-4h12" />
  </svg>
);

const SpinnerIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 12a9 9 0 11-6.219-8.56" />
  </svg>
);

export default ThreadView;
