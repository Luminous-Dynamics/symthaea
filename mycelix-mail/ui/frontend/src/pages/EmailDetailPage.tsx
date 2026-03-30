// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Email Detail Page
 *
 * Full email view with thread, actions, and trust info
 */

import React, { useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useI18n } from '../lib/i18n';
import { useAccessibility, useKeyboardShortcut } from '../lib/a11y/AccessibilityProvider';
import { graphqlClient } from '../lib/api/graphql-client';
import { useTrustScore } from '../lib/api/holochain-client';
import { usePQCrypto } from '../lib/crypto/post-quantum';

// Types
interface Email {
  id: string;
  subject: string;
  bodyText?: string;
  bodyHtml?: string;
  from: { email: string; name?: string };
  to: { email: string; name?: string }[];
  cc: { email: string; name?: string }[];
  isRead: boolean;
  isStarred: boolean;
  isArchived: boolean;
  isEncrypted: boolean;
  trustScore?: number;
  labels: string[];
  sentAt?: string;
  receivedAt?: string;
  threadId?: string;
  attachments: Attachment[];
}

interface Attachment {
  id: string;
  filename: string;
  contentType: string;
  size: number;
  isEncrypted: boolean;
}

// Trust Badge
function TrustBadge({ score }: { score: number }) {
  const getColor = (s: number) => {
    if (s >= 0.8) return { bg: '#e8f5e9', border: '#4CAF50', text: 'Trusted' };
    if (s >= 0.5) return { bg: '#fff8e1', border: '#FFC107', text: 'Known' };
    if (s >= 0.2) return { bg: '#fff3e0', border: '#FF9800', text: 'Caution' };
    return { bg: '#ffebee', border: '#F44336', text: 'Unknown' };
  };

  const { bg, border, text } = getColor(score);

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '4px',
        padding: '4px 10px',
        borderRadius: '12px',
        backgroundColor: bg,
        border: `1px solid ${border}`,
        fontSize: '12px',
        fontWeight: 500,
      }}
      role="img"
      aria-label={`Trust level: ${text} (${Math.round(score * 100)}%)`}
    >
      {score >= 0.8 ? '✓' : score >= 0.5 ? '~' : score >= 0.2 ? '!' : '?'} {text}
    </span>
  );
}

// Attachment List
function AttachmentList({ attachments, emailId }: { attachments: Attachment[]; emailId: string }) {
  const { t } = useI18n();
  const { decrypt } = usePQCrypto();

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const handleDownload = async (attachment: Attachment) => {
    const url = `/api/emails/${emailId}/attachments/${attachment.id}`;
    // In real implementation, would decrypt if encrypted
    window.open(url, '_blank');
  };

  if (attachments.length === 0) return null;

  return (
    <div style={{ marginTop: '24px', padding: '16px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
      <h3 style={{ margin: '0 0 12px', fontSize: '14px' }}>
        {t('email.attachments')} ({attachments.length})
      </h3>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        {attachments.map((att) => (
          <button
            key={att.id}
            onClick={() => handleDownload(att)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 12px',
              backgroundColor: 'white',
              border: '1px solid #e0e0e0',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            <span>📎</span>
            <span>{att.filename}</span>
            <span style={{ color: '#666', fontSize: '12px' }}>({formatSize(att.size)})</span>
            {att.isEncrypted && <span title="Encrypted">🔒</span>}
          </button>
        ))}
      </div>
    </div>
  );
}

// Thread View
function ThreadView({ threadId, currentEmailId }: { threadId: string; currentEmailId: string }) {
  const { data: thread } = useQuery({
    queryKey: ['thread', threadId],
    queryFn: async () => {
      const response = await graphqlClient.query<{ thread: { emails: Email[] } }>(
        `query GetThread($id: ID!) {
          thread(id: $id) {
            emails { id subject from { email name } receivedAt }
          }
        }`,
        { id: threadId }
      );
      return response.thread;
    },
    enabled: !!threadId,
  });

  if (!thread || thread.emails.length <= 1) return null;

  return (
    <div style={{ marginBottom: '24px', padding: '16px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
      <h3 style={{ margin: '0 0 12px', fontSize: '14px' }}>Thread ({thread.emails.length} messages)</h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {thread.emails.map((email) => (
          <a
            key={email.id}
            href={`/email/${email.id}`}
            style={{
              padding: '8px',
              backgroundColor: email.id === currentEmailId ? '#e3f2fd' : 'white',
              borderRadius: '4px',
              textDecoration: 'none',
              color: 'inherit',
              display: 'block',
            }}
          >
            <div style={{ fontWeight: email.id === currentEmailId ? 'bold' : 'normal' }}>
              {email.from.name || email.from.email}
            </div>
            <div style={{ fontSize: '12px', color: '#666' }}>{email.subject}</div>
          </a>
        ))}
      </div>
    </div>
  );
}

// Main Component
export default function EmailDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { t, formatDate } = useI18n();
  const { announce } = useAccessibility();
  const queryClient = useQueryClient();
  const { decrypt, isInitialized: cryptoReady } = usePQCrypto();

  // Fetch email
  const { data: email, isLoading, isError } = useQuery({
    queryKey: ['email', id],
    queryFn: async () => {
      const response = await graphqlClient.query<{ email: Email }>(
        `query GetEmail($id: ID!) {
          email(id: $id) {
            id subject bodyText bodyHtml
            from { email name }
            to { email name }
            cc { email name }
            isRead isStarred isArchived isEncrypted
            trustScore labels sentAt receivedAt threadId
            attachments { id filename contentType size isEncrypted }
          }
        }`,
        { id }
      );
      return response.email;
    },
    enabled: !!id,
  });

  // Get live trust score
  const { data: trustData } = useTrustScore(email?.from.email);
  const trustScore = trustData?.score ?? email?.trustScore ?? 0.5;

  // Mutations
  const starMutation = useMutation({
    mutationFn: async (isStarred: boolean) => {
      return graphqlClient.mutate(
        `mutation StarEmail($id: ID!, $isStarred: Boolean!) {
          updateEmail(id: $id, input: { isStarred: $isStarred }) { id }
        }`,
        { id, isStarred }
      );
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['email', id] }),
  });

  const archiveMutation = useMutation({
    mutationFn: async () => {
      return graphqlClient.mutate(
        `mutation ArchiveEmail($id: ID!) {
          updateEmail(id: $id, input: { isArchived: true }) { id }
        }`,
        { id }
      );
    },
    onSuccess: () => {
      announce(t('email.archived'));
      navigate('/inbox');
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async () => {
      return graphqlClient.mutate(
        `mutation DeleteEmail($id: ID!) { deleteEmail(id: $id) }`,
        { id }
      );
    },
    onSuccess: () => {
      announce(t('email.deleted'));
      navigate('/inbox');
    },
  });

  // Mark as read on view
  React.useEffect(() => {
    if (email && !email.isRead) {
      graphqlClient.mutate(
        `mutation MarkRead($id: ID!) {
          updateEmail(id: $id, input: { isRead: true }) { id }
        }`,
        { id }
      );
    }
  }, [email, id]);

  // Actions
  const handleReply = useCallback(() => {
    navigate(`/compose?replyTo=${email?.from.email}&subject=${encodeURIComponent(email?.subject || '')}`);
  }, [navigate, email]);

  const handleReplyAll = useCallback(() => {
    const allRecipients = [email?.from.email, ...(email?.to.map((t) => t.email) || [])].join(',');
    navigate(`/compose?replyTo=${allRecipients}&subject=${encodeURIComponent(email?.subject || '')}`);
  }, [navigate, email]);

  const handleForward = useCallback(() => {
    navigate(`/compose?forward=${id}&subject=${encodeURIComponent(email?.subject || '')}`);
  }, [navigate, id, email]);

  const handleDelete = useCallback(() => {
    if (window.confirm(t('email.confirmDelete', { defaultValue: 'Delete this email?' }))) {
      deleteMutation.mutate();
    }
  }, [deleteMutation, t]);

  // Keyboard shortcuts
  useKeyboardShortcut('r', handleReply);
  useKeyboardShortcut('a', handleReplyAll, { shift: true });
  useKeyboardShortcut('f', handleForward);
  useKeyboardShortcut('e', () => archiveMutation.mutate());
  useKeyboardShortcut('s', () => starMutation.mutate(!email?.isStarred));
  useKeyboardShortcut('Escape', () => navigate('/inbox'));

  if (isLoading) {
    return (
      <div role="status" style={{ padding: '2rem', textAlign: 'center' }}>
        {t('common.loading')}
      </div>
    );
  }

  if (isError || !email) {
    return (
      <div role="alert" style={{ padding: '2rem', textAlign: 'center' }}>
        <p style={{ color: '#d32f2f' }}>{t('errors.notFound')}</p>
        <button onClick={() => navigate('/inbox')} style={{ marginTop: '16px', padding: '8px 16px' }}>
          Back to Inbox
        </button>
      </div>
    );
  }

  return (
    <article style={{ maxWidth: '900px', margin: '0 auto', padding: '24px' }}>
      {/* Toolbar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginBottom: '24px',
          padding: '12px',
          backgroundColor: '#f5f5f5',
          borderRadius: '8px',
        }}
      >
        <button onClick={() => navigate('/inbox')} style={{ padding: '8px 12px' }}>
          ← Back
        </button>
        <div style={{ flex: 1 }} />
        <button onClick={handleReply} style={{ padding: '8px 12px' }}>
          ↩ {t('email.reply')}
        </button>
        <button onClick={handleReplyAll} style={{ padding: '8px 12px' }}>
          ↩↩ {t('email.replyAll')}
        </button>
        <button onClick={handleForward} style={{ padding: '8px 12px' }}>
          ↪ {t('email.forward')}
        </button>
        <button
          onClick={() => starMutation.mutate(!email.isStarred)}
          aria-pressed={email.isStarred}
          style={{ padding: '8px 12px', fontSize: '18px' }}
        >
          {email.isStarred ? '★' : '☆'}
        </button>
        <button onClick={() => archiveMutation.mutate()} style={{ padding: '8px 12px' }}>
          📥 {t('email.archive')}
        </button>
        <button onClick={handleDelete} style={{ padding: '8px 12px', color: '#d32f2f' }}>
          🗑️ {t('email.delete')}
        </button>
      </div>

      {/* Thread */}
      {email.threadId && <ThreadView threadId={email.threadId} currentEmailId={email.id} />}

      {/* Header */}
      <header style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '16px' }}>
          <h1 style={{ margin: 0, fontSize: '24px' }}>
            {email.isEncrypted && <span title="Encrypted">🔒 </span>}
            {email.subject || '(No subject)'}
          </h1>
          <TrustBadge score={trustScore} />
        </div>

        {/* Labels */}
        {email.labels.length > 0 && (
          <div style={{ display: 'flex', gap: '4px', marginBottom: '16px' }}>
            {email.labels.map((label) => (
              <span
                key={label}
                style={{
                  backgroundColor: '#e0e0e0',
                  padding: '2px 8px',
                  borderRadius: '4px',
                  fontSize: '12px',
                }}
              >
                {label}
              </span>
            ))}
          </div>
        )}

        {/* From/To info */}
        <div style={{ display: 'grid', gap: '8px', fontSize: '14px' }}>
          <div>
            <strong>{t('email.from')}:</strong>{' '}
            <span>{email.from.name ? `${email.from.name} <${email.from.email}>` : email.from.email}</span>
          </div>
          <div>
            <strong>{t('email.to')}:</strong>{' '}
            <span>{email.to.map((r) => r.name || r.email).join(', ')}</span>
          </div>
          {email.cc.length > 0 && (
            <div>
              <strong>{t('email.cc')}:</strong>{' '}
              <span>{email.cc.map((r) => r.name || r.email).join(', ')}</span>
            </div>
          )}
          <div>
            <strong>{t('email.date')}:</strong>{' '}
            <time dateTime={email.receivedAt || email.sentAt}>
              {formatDate(email.receivedAt || email.sentAt || new Date().toISOString(), 'long')}
            </time>
          </div>
        </div>
      </header>

      {/* Encryption notice */}
      {email.isEncrypted && (
        <div
          style={{
            padding: '12px',
            backgroundColor: '#e8f5e9',
            borderRadius: '8px',
            marginBottom: '24px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <span>🔒</span>
          <span>This message is end-to-end encrypted with Post-Quantum cryptography</span>
        </div>
      )}

      {/* Body */}
      <div
        style={{
          padding: '24px',
          backgroundColor: 'white',
          border: '1px solid #e0e0e0',
          borderRadius: '8px',
          minHeight: '200px',
        }}
      >
        {email.bodyHtml ? (
          <div
            dangerouslySetInnerHTML={{ __html: email.bodyHtml }}
            style={{ lineHeight: 1.6 }}
          />
        ) : (
          <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit', margin: 0 }}>
            {email.bodyText || '(No content)'}
          </pre>
        )}
      </div>

      {/* Attachments */}
      <AttachmentList attachments={email.attachments} emailId={email.id} />

      {/* Keyboard hints */}
      <div style={{ marginTop: '24px', fontSize: '12px', color: '#666' }}>
        Shortcuts: <kbd>r</kbd> reply | <kbd>Shift+a</kbd> reply all | <kbd>f</kbd> forward | <kbd>e</kbd> archive | <kbd>s</kbd> star | <kbd>Esc</kbd> back
      </div>
    </article>
  );
}
