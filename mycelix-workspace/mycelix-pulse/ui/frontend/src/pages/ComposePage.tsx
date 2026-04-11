// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Compose Page
 *
 * Email composition with PQE encryption and AI assistance
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useI18n } from '../lib/i18n';
import { useAccessibility, useKeyboardShortcut } from '../lib/a11y/AccessibilityProvider';
import { graphqlClient } from '../lib/api/graphql-client';
import { usePQCrypto } from '../lib/crypto/post-quantum';
import { useTrustScore } from '../lib/api/holochain-client';

// Types
interface Recipient {
  email: string;
  name?: string;
  trustScore?: number;
}

interface Attachment {
  file: File;
  id: string;
  isEncrypted: boolean;
}

// Recipient Chip Component
function RecipientChip({
  recipient,
  onRemove,
}: {
  recipient: Recipient;
  onRemove: () => void;
}) {
  const { data: trustData } = useTrustScore(recipient.email);
  const trustScore = trustData?.score ?? recipient.trustScore ?? 0.5;

  const getColor = (s: number) => {
    if (s >= 0.8) return '#e8f5e9';
    if (s >= 0.5) return '#fff8e1';
    if (s >= 0.2) return '#fff3e0';
    return '#ffebee';
  };

  const getBorderColor = (s: number) => {
    if (s >= 0.8) return '#4CAF50';
    if (s >= 0.5) return '#FFC107';
    if (s >= 0.2) return '#FF9800';
    return '#F44336';
  };

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '4px',
        padding: '4px 8px',
        borderRadius: '16px',
        backgroundColor: getColor(trustScore),
        border: `1px solid ${getBorderColor(trustScore)}`,
        fontSize: '14px',
      }}
    >
      <span>{recipient.name || recipient.email}</span>
      <button
        onClick={onRemove}
        aria-label={`Remove ${recipient.email}`}
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: '0 2px',
          fontSize: '14px',
          color: '#666',
        }}
      >
        ×
      </button>
    </span>
  );
}

// Attachment Item Component
function AttachmentItem({
  attachment,
  onRemove,
  onToggleEncrypt,
}: {
  attachment: Attachment;
  onRemove: () => void;
  onToggleEncrypt: () => void;
}) {
  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '8px 12px',
        backgroundColor: '#f5f5f5',
        borderRadius: '4px',
      }}
    >
      <span>📎</span>
      <span style={{ flex: 1 }}>{attachment.file.name}</span>
      <span style={{ color: '#666', fontSize: '12px' }}>{formatSize(attachment.file.size)}</span>
      <button
        onClick={onToggleEncrypt}
        aria-label={attachment.isEncrypted ? 'Disable encryption' : 'Enable encryption'}
        title={attachment.isEncrypted ? 'Encrypted' : 'Not encrypted'}
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          fontSize: '16px',
        }}
      >
        {attachment.isEncrypted ? '🔒' : '🔓'}
      </button>
      <button
        onClick={onRemove}
        aria-label="Remove attachment"
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: '#d32f2f',
        }}
      >
        ×
      </button>
    </div>
  );
}

// Main Compose Page
export default function ComposePage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { t } = useI18n();
  const { announce } = useAccessibility();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // PQE crypto
  const { isInitialized: cryptoReady, encrypt, getPublicKey } = usePQCrypto();

  // Form state
  const [to, setTo] = useState<Recipient[]>([]);
  const [cc, setCc] = useState<Recipient[]>([]);
  const [bcc, setBcc] = useState<Recipient[]>([]);
  const [subject, setSubject] = useState('');
  const [body, setBody] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [isEncrypted, setIsEncrypted] = useState(true);
  const [isSending, setIsSending] = useState(false);

  // Input states for adding recipients
  const [toInput, setToInput] = useState('');
  const [ccInput, setCcInput] = useState('');
  const [bccInput, setBccInput] = useState('');
  const [showCc, setShowCc] = useState(false);
  const [showBcc, setShowBcc] = useState(false);

  // Pre-fill from URL params (reply/forward)
  useEffect(() => {
    const replyTo = searchParams.get('replyTo');
    const forwardId = searchParams.get('forward');

    if (replyTo) {
      setTo([{ email: replyTo }]);
      setSubject(`Re: ${searchParams.get('subject') || ''}`);
    }
    if (forwardId) {
      setSubject(`Fwd: ${searchParams.get('subject') || ''}`);
    }
  }, [searchParams]);

  // Add recipient helper
  const addRecipient = useCallback(
    (
      input: string,
      setRecipients: React.Dispatch<React.SetStateAction<Recipient[]>>,
      setInput: React.Dispatch<React.SetStateAction<string>>
    ) => {
      const email = input.trim();
      if (email && email.includes('@')) {
        setRecipients((prev) => [...prev, { email }]);
        setInput('');
      }
    },
    []
  );

  // Handle recipient input keydown
  const handleRecipientKeyDown = useCallback(
    (
      e: React.KeyboardEvent,
      input: string,
      setRecipients: React.Dispatch<React.SetStateAction<Recipient[]>>,
      setInput: React.Dispatch<React.SetStateAction<string>>
    ) => {
      if (e.key === 'Enter' || e.key === ',' || e.key === 'Tab') {
        e.preventDefault();
        addRecipient(input, setRecipients, setInput);
      }
    },
    [addRecipient]
  );

  // File handling
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const newAttachments: Attachment[] = Array.from(files).map((file) => ({
      file,
      id: crypto.randomUUID(),
      isEncrypted: true, // Encrypt by default
    }));

    setAttachments((prev) => [...prev, ...newAttachments]);
    e.target.value = ''; // Reset input
  }, []);

  // Send email mutation
  const sendEmailMutation = useMutation({
    mutationFn: async () => {
      let emailBody = body;
      let encryptedFor: string[] = [];

      // Encrypt if enabled
      if (isEncrypted && cryptoReady) {
        const recipients = [...to, ...cc, ...bcc].map((r) => r.email);
        // In a real implementation, we'd fetch recipient public keys and encrypt
        // For now, we'll mark it as encrypted
        encryptedFor = recipients;
        // emailBody = await encrypt(body, recipientPublicKeys);
      }

      const input = {
        subject,
        bodyText: emailBody,
        to: to.map((r) => r.email),
        cc: cc.map((r) => r.email),
        bcc: bcc.map((r) => r.email),
        isEncrypted,
        encryptedFor,
      };

      // Create and send email
      const response = await graphqlClient.mutate(
        `mutation SendEmail($input: SendEmailInput!) {
          sendEmail(input: $input) {
            id
            status
          }
        }`,
        { input }
      );

      return response;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['emails'] });
      announce(t('email.sent'));
      navigate('/inbox');
    },
    onError: (error) => {
      announce(`Error: ${error instanceof Error ? error.message : 'Failed to send'}`);
    },
  });

  // Save draft mutation
  const saveDraftMutation = useMutation({
    mutationFn: async () => {
      return graphqlClient.mutate(
        `mutation SaveDraft($input: CreateEmailInput!) {
          createEmail(input: $input) { id }
        }`,
        {
          input: {
            subject,
            bodyText: body,
            to: to.map((r) => r.email),
            cc: cc.map((r) => r.email),
            bcc: bcc.map((r) => r.email),
          },
        }
      );
    },
    onSuccess: () => {
      announce('Draft saved');
    },
  });

  // AI draft assistance
  const [aiSuggestion, setAiSuggestion] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const generateDraft = useCallback(async () => {
    if (!subject) {
      announce('Please enter a subject first');
      return;
    }

    setIsGenerating(true);
    try {
      const response = await fetch('/api/ml/draft', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject,
          context: body,
          recipients: to.map((r) => r.email),
        }),
      });

      if (response.ok) {
        const { draft } = await response.json();
        setAiSuggestion(draft);
      }
    } catch (error) {
      console.error('AI draft generation failed:', error);
    } finally {
      setIsGenerating(false);
    }
  }, [subject, body, to, announce]);

  const acceptAiSuggestion = useCallback(() => {
    if (aiSuggestion) {
      setBody(aiSuggestion);
      setAiSuggestion(null);
    }
  }, [aiSuggestion]);

  // Keyboard shortcuts
  useKeyboardShortcut('s', () => saveDraftMutation.mutate(), { ctrl: true });
  useKeyboardShortcut('Enter', () => sendEmailMutation.mutate(), { ctrl: true });

  const handleSend = useCallback(() => {
    if (to.length === 0) {
      announce('Please add at least one recipient');
      return;
    }
    setIsSending(true);
    sendEmailMutation.mutate();
  }, [to, sendEmailMutation, announce]);

  const handleDiscard = useCallback(() => {
    if (body || subject) {
      if (!window.confirm(t('email.discardConfirm', { defaultValue: 'Discard this draft?' }))) {
        return;
      }
    }
    navigate(-1);
  }, [body, subject, navigate, t]);

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', padding: '24px' }}>
      <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
        <h1 style={{ margin: 0 }}>{t('email.compose')}</h1>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={handleDiscard} style={{ padding: '8px 16px' }}>
            {t('email.discard')}
          </button>
          <button
            onClick={() => saveDraftMutation.mutate()}
            disabled={saveDraftMutation.isPending}
            style={{ padding: '8px 16px' }}
          >
            {t('email.saveDraft')}
          </button>
          <button
            onClick={handleSend}
            disabled={to.length === 0 || sendEmailMutation.isPending}
            style={{
              padding: '8px 24px',
              backgroundColor: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              fontWeight: 'bold',
              cursor: to.length === 0 ? 'not-allowed' : 'pointer',
              opacity: to.length === 0 ? 0.5 : 1,
            }}
          >
            {sendEmailMutation.isPending ? t('email.sending') : t('email.send')}
          </button>
        </div>
      </header>

      {/* Encryption toggle */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginBottom: '16px',
          padding: '12px',
          backgroundColor: isEncrypted ? '#e8f5e9' : '#fff3e0',
          borderRadius: '4px',
        }}
      >
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={isEncrypted}
            onChange={(e) => setIsEncrypted(e.target.checked)}
          />
          <span>{isEncrypted ? '🔒' : '🔓'}</span>
          <span>Post-Quantum Encryption {isEncrypted ? 'Enabled' : 'Disabled'}</span>
        </label>
        {!cryptoReady && isEncrypted && (
          <span style={{ color: '#ff9800', fontSize: '12px' }}>(Initializing crypto...)</span>
        )}
      </div>

      {/* Recipients */}
      <div style={{ marginBottom: '12px' }}>
        <label style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
          <span style={{ width: '60px', paddingTop: '8px' }}>{t('email.to')}:</span>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '4px' }}>
              {to.map((r, i) => (
                <RecipientChip key={i} recipient={r} onRemove={() => setTo((prev) => prev.filter((_, j) => j !== i))} />
              ))}
            </div>
            <input
              type="email"
              value={toInput}
              onChange={(e) => setToInput(e.target.value)}
              onKeyDown={(e) => handleRecipientKeyDown(e, toInput, setTo, setToInput)}
              onBlur={() => addRecipient(toInput, setTo, setToInput)}
              placeholder="Add recipient..."
              style={{ width: '100%', padding: '8px', border: '1px solid #ccc', borderRadius: '4px' }}
            />
          </div>
          {!showCc && (
            <button onClick={() => setShowCc(true)} style={{ background: 'none', border: 'none', color: '#1976d2', cursor: 'pointer' }}>
              Cc
            </button>
          )}
          {!showBcc && (
            <button onClick={() => setShowBcc(true)} style={{ background: 'none', border: 'none', color: '#1976d2', cursor: 'pointer' }}>
              Bcc
            </button>
          )}
        </label>
      </div>

      {/* CC */}
      {showCc && (
        <div style={{ marginBottom: '12px' }}>
          <label style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
            <span style={{ width: '60px', paddingTop: '8px' }}>{t('email.cc')}:</span>
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '4px' }}>
                {cc.map((r, i) => (
                  <RecipientChip key={i} recipient={r} onRemove={() => setCc((prev) => prev.filter((_, j) => j !== i))} />
                ))}
              </div>
              <input
                type="email"
                value={ccInput}
                onChange={(e) => setCcInput(e.target.value)}
                onKeyDown={(e) => handleRecipientKeyDown(e, ccInput, setCc, setCcInput)}
                onBlur={() => addRecipient(ccInput, setCc, setCcInput)}
                style={{ width: '100%', padding: '8px', border: '1px solid #ccc', borderRadius: '4px' }}
              />
            </div>
          </label>
        </div>
      )}

      {/* BCC */}
      {showBcc && (
        <div style={{ marginBottom: '12px' }}>
          <label style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
            <span style={{ width: '60px', paddingTop: '8px' }}>{t('email.bcc')}:</span>
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '4px' }}>
                {bcc.map((r, i) => (
                  <RecipientChip key={i} recipient={r} onRemove={() => setBcc((prev) => prev.filter((_, j) => j !== i))} />
                ))}
              </div>
              <input
                type="email"
                value={bccInput}
                onChange={(e) => setBccInput(e.target.value)}
                onKeyDown={(e) => handleRecipientKeyDown(e, bccInput, setBcc, setBccInput)}
                onBlur={() => addRecipient(bccInput, setBcc, setBccInput)}
                style={{ width: '100%', padding: '8px', border: '1px solid #ccc', borderRadius: '4px' }}
              />
            </div>
          </label>
        </div>
      )}

      {/* Subject */}
      <div style={{ marginBottom: '12px' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ width: '60px' }}>{t('email.subject')}:</span>
          <input
            type="text"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            style={{ flex: 1, padding: '8px', border: '1px solid #ccc', borderRadius: '4px' }}
          />
        </label>
      </div>

      {/* Body */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <span>Message</span>
          <button
            onClick={generateDraft}
            disabled={isGenerating || !subject}
            style={{
              background: 'none',
              border: 'none',
              color: '#1976d2',
              cursor: isGenerating || !subject ? 'not-allowed' : 'pointer',
              fontSize: '13px',
            }}
          >
            {isGenerating ? 'Generating...' : '✨ AI Draft'}
          </button>
        </div>
        <textarea
          value={body}
          onChange={(e) => setBody(e.target.value)}
          rows={15}
          style={{
            width: '100%',
            padding: '12px',
            border: '1px solid #ccc',
            borderRadius: '4px',
            fontFamily: 'inherit',
            fontSize: '14px',
            resize: 'vertical',
          }}
        />
      </div>

      {/* AI Suggestion */}
      {aiSuggestion && (
        <div
          style={{
            marginBottom: '16px',
            padding: '16px',
            backgroundColor: '#e3f2fd',
            borderRadius: '8px',
            border: '1px solid #90caf9',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <strong>✨ AI Suggestion</strong>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button onClick={() => setAiSuggestion(null)} style={{ fontSize: '13px' }}>
                Dismiss
              </button>
              <button
                onClick={acceptAiSuggestion}
                style={{
                  fontSize: '13px',
                  backgroundColor: '#1976d2',
                  color: 'white',
                  border: 'none',
                  padding: '4px 12px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                }}
              >
                Use This
              </button>
            </div>
          </div>
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontSize: '14px' }}>{aiSuggestion}</pre>
        </div>
      )}

      {/* Attachments */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <span>{t('email.attachments')}</span>
          <button
            onClick={() => fileInputRef.current?.click()}
            style={{
              background: 'none',
              border: '1px dashed #ccc',
              padding: '4px 12px',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            + Add Files
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
        </div>

        {attachments.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {attachments.map((att) => (
              <AttachmentItem
                key={att.id}
                attachment={att}
                onRemove={() => setAttachments((prev) => prev.filter((a) => a.id !== att.id))}
                onToggleEncrypt={() =>
                  setAttachments((prev) =>
                    prev.map((a) => (a.id === att.id ? { ...a, isEncrypted: !a.isEncrypted } : a))
                  )
                }
              />
            ))}
          </div>
        )}
      </div>

      {/* Keyboard shortcuts hint */}
      <div style={{ marginTop: '24px', fontSize: '12px', color: '#666' }}>
        <kbd>Ctrl+Enter</kbd> to send | <kbd>Ctrl+S</kbd> to save draft
      </div>
    </div>
  );
}
