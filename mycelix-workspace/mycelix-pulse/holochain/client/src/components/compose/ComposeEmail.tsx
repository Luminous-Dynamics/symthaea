// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * ComposeEmail Component
 *
 * Full-featured email composer with recipients, attachments, and encryption.
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Avatar } from '../common/Avatar';
import { TrustBadge } from '../common/TrustBadge';

export interface Recipient {
  address: string;
  name?: string;
  trustLevel?: number;
}

export interface Attachment {
  id: string;
  name: string;
  size: number;
  type: string;
  data?: ArrayBuffer;
}

export interface ComposedEmail {
  to: Recipient[];
  cc: Recipient[];
  bcc: Recipient[];
  subject: string;
  body: string;
  bodyHtml?: string;
  attachments: Attachment[];
  encrypted: boolean;
  scheduledAt?: number;
  replyTo?: string;
  threadId?: string;
}

export interface ComposeEmailProps {
  /** Initial recipient */
  to?: Recipient[];
  /** Reply-to email hash */
  replyTo?: string;
  /** Thread ID for threading */
  threadId?: string;
  /** Initial subject */
  subject?: string;
  /** Initial body */
  body?: string;
  /** Whether to show CC/BCC by default */
  showCcBcc?: boolean;
  /** Contact suggestions callback */
  onSuggestContacts?: (query: string) => Promise<Recipient[]>;
  /** Send callback */
  onSend?: (email: ComposedEmail) => Promise<void>;
  /** Save draft callback */
  onSaveDraft?: (email: ComposedEmail) => Promise<void>;
  /** Schedule send callback */
  onSchedule?: (email: ComposedEmail, sendAt: Date) => Promise<void>;
  /** Discard callback */
  onDiscard?: () => void;
  /** Minimize callback */
  onMinimize?: () => void;
}

export const ComposeEmail: React.FC<ComposeEmailProps> = ({
  to: initialTo = [],
  replyTo,
  threadId,
  subject: initialSubject = '',
  body: initialBody = '',
  showCcBcc: initialShowCcBcc = false,
  onSuggestContacts,
  onSend,
  onSaveDraft,
  onSchedule,
  onDiscard,
  onMinimize,
}) => {
  const [to, setTo] = useState<Recipient[]>(initialTo);
  const [cc, setCc] = useState<Recipient[]>([]);
  const [bcc, setBcc] = useState<Recipient[]>([]);
  const [subject, setSubject] = useState(initialSubject);
  const [body, setBody] = useState(initialBody);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [encrypted, setEncrypted] = useState(true);
  const [showCcBcc, setShowCcBcc] = useState(initialShowCcBcc);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showScheduler, setShowScheduler] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const bodyRef = useRef<HTMLTextAreaElement>(null);

  // Auto-save draft periodically
  useEffect(() => {
    const interval = setInterval(() => {
      if (subject || body || to.length > 0) {
        onSaveDraft?.({
          to,
          cc,
          bcc,
          subject,
          body,
          attachments,
          encrypted,
          replyTo,
          threadId,
        });
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [to, cc, bcc, subject, body, attachments, encrypted, replyTo, threadId, onSaveDraft]);

  const handleSend = useCallback(async () => {
    if (to.length === 0) {
      setError('Please add at least one recipient');
      return;
    }

    setSending(true);
    setError(null);

    try {
      await onSend?.({
        to,
        cc,
        bcc,
        subject,
        body,
        attachments,
        encrypted,
        replyTo,
        threadId,
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setSending(false);
    }
  }, [to, cc, bcc, subject, body, attachments, encrypted, replyTo, threadId, onSend]);

  const handleScheduleSend = useCallback(
    async (sendAt: Date) => {
      if (to.length === 0) {
        setError('Please add at least one recipient');
        return;
      }

      setSending(true);
      setError(null);

      try {
        await onSchedule?.(
          {
            to,
            cc,
            bcc,
            subject,
            body,
            attachments,
            encrypted,
            replyTo,
            threadId,
            scheduledAt: sendAt.getTime() * 1000,
          },
          sendAt
        );
      } catch (e) {
        setError(String(e));
      } finally {
        setSending(false);
        setShowScheduler(false);
      }
    },
    [to, cc, bcc, subject, body, attachments, encrypted, replyTo, threadId, onSchedule]
  );

  const handleAttachmentSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (!files) return;

      const newAttachments: Attachment[] = [];

      for (const file of files) {
        const data = await file.arrayBuffer();
        newAttachments.push({
          id: `${Date.now()}-${file.name}`,
          name: file.name,
          size: file.size,
          type: file.type,
          data,
        });
      }

      setAttachments((prev) => [...prev, ...newAttachments]);
      e.target.value = '';
    },
    []
  );

  const handleRemoveAttachment = useCallback((id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id));
  }, []);

  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="flex flex-col bg-white rounded-t-lg shadow-xl border border-gray-300 w-full max-w-2xl">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-100 rounded-t-lg border-b">
        <span className="font-medium text-gray-700">
          {replyTo ? 'Reply' : 'New Message'}
        </span>
        <div className="flex items-center gap-1">
          {onMinimize && (
            <button
              onClick={onMinimize}
              className="p-1 hover:bg-gray-200 rounded"
            >
              <MinimizeIcon className="w-4 h-4" />
            </button>
          )}
          <button
            onClick={onDiscard}
            className="p-1 hover:bg-gray-200 rounded"
          >
            <CloseIcon className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="px-4 py-2 bg-red-50 text-red-600 text-sm">
          {error}
        </div>
      )}

      {/* Recipients */}
      <div className="border-b">
        <RecipientField
          label="To"
          recipients={to}
          onChange={setTo}
          onSuggest={onSuggestContacts}
          autoFocus
        />

        {showCcBcc && (
          <>
            <RecipientField
              label="Cc"
              recipients={cc}
              onChange={setCc}
              onSuggest={onSuggestContacts}
            />
            <RecipientField
              label="Bcc"
              recipients={bcc}
              onChange={setBcc}
              onSuggest={onSuggestContacts}
            />
          </>
        )}

        {!showCcBcc && (
          <button
            onClick={() => setShowCcBcc(true)}
            className="text-sm text-gray-500 hover:text-gray-700 px-4 py-1"
          >
            Cc/Bcc
          </button>
        )}
      </div>

      {/* Subject */}
      <input
        type="text"
        value={subject}
        onChange={(e) => setSubject(e.target.value)}
        placeholder="Subject"
        className="px-4 py-2 border-b focus:outline-none"
      />

      {/* Body */}
      <textarea
        ref={bodyRef}
        value={body}
        onChange={(e) => setBody(e.target.value)}
        placeholder="Compose your message..."
        className="flex-1 px-4 py-2 min-h-[200px] resize-none focus:outline-none"
      />

      {/* Attachments */}
      {attachments.length > 0 && (
        <div className="px-4 py-2 border-t flex flex-wrap gap-2">
          {attachments.map((attachment) => (
            <div
              key={attachment.id}
              className="flex items-center gap-2 px-2 py-1 bg-gray-100 rounded text-sm"
            >
              <AttachmentIcon className="w-4 h-4 text-gray-500" />
              <span className="truncate max-w-[150px]">{attachment.name}</span>
              <span className="text-gray-400">({formatSize(attachment.size)})</span>
              <button
                onClick={() => handleRemoveAttachment(attachment.id)}
                className="p-0.5 hover:bg-gray-200 rounded"
              >
                <CloseIcon className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between px-4 py-2 border-t bg-gray-50">
        <div className="flex items-center gap-2">
          {/* Send button */}
          <button
            onClick={handleSend}
            disabled={sending || to.length === 0}
            className={`px-4 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 flex items-center gap-2 ${
              sending || to.length === 0 ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            {sending ? (
              <SpinnerIcon className="w-4 h-4 animate-spin" />
            ) : (
              <SendIcon className="w-4 h-4" />
            )}
            Send
          </button>

          {/* Schedule button */}
          {onSchedule && (
            <button
              onClick={() => setShowScheduler(!showScheduler)}
              className="p-1.5 hover:bg-gray-200 rounded"
              title="Schedule send"
            >
              <ClockIcon className="w-5 h-5 text-gray-500" />
            </button>
          )}

          {/* Attachment button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-1.5 hover:bg-gray-200 rounded"
            title="Attach files"
          >
            <AttachmentIcon className="w-5 h-5 text-gray-500" />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleAttachmentSelect}
            className="hidden"
          />

          {/* Encryption toggle */}
          <button
            onClick={() => setEncrypted(!encrypted)}
            className={`p-1.5 rounded ${
              encrypted ? 'bg-green-100 text-green-600' : 'hover:bg-gray-200 text-gray-500'
            }`}
            title={encrypted ? 'End-to-end encrypted' : 'Enable encryption'}
          >
            <LockIcon className="w-5 h-5" />
          </button>
        </div>

        {/* Discard button */}
        <button
          onClick={onDiscard}
          className="p-1.5 hover:bg-gray-200 rounded"
          title="Discard draft"
        >
          <TrashIcon className="w-5 h-5 text-gray-500" />
        </button>
      </div>

      {/* Schedule picker */}
      {showScheduler && (
        <SchedulePicker
          onSchedule={handleScheduleSend}
          onCancel={() => setShowScheduler(false)}
        />
      )}
    </div>
  );
};

// Recipient Field Component
interface RecipientFieldProps {
  label: string;
  recipients: Recipient[];
  onChange: (recipients: Recipient[]) => void;
  onSuggest?: (query: string) => Promise<Recipient[]>;
  autoFocus?: boolean;
}

const RecipientField: React.FC<RecipientFieldProps> = ({
  label,
  recipients,
  onChange,
  onSuggest,
  autoFocus,
}) => {
  const [input, setInput] = useState('');
  const [suggestions, setSuggestions] = useState<Recipient[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  const handleInputChange = useCallback(
    async (value: string) => {
      setInput(value);

      if (value.length >= 2 && onSuggest) {
        const results = await onSuggest(value);
        setSuggestions(results);
        setShowSuggestions(results.length > 0);
      } else {
        setShowSuggestions(false);
      }
    },
    [onSuggest]
  );

  const handleAddRecipient = useCallback(
    (recipient: Recipient) => {
      onChange([...recipients, recipient]);
      setInput('');
      setShowSuggestions(false);
    },
    [recipients, onChange]
  );

  const handleRemoveRecipient = useCallback(
    (address: string) => {
      onChange(recipients.filter((r) => r.address !== address));
    },
    [recipients, onChange]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ',') {
        e.preventDefault();
        const email = input.trim();
        if (email && email.includes('@')) {
          handleAddRecipient({ address: email });
        }
      } else if (e.key === 'Backspace' && !input && recipients.length > 0) {
        handleRemoveRecipient(recipients[recipients.length - 1].address);
      }
    },
    [input, recipients, handleAddRecipient, handleRemoveRecipient]
  );

  return (
    <div className="flex items-start px-4 py-2 relative">
      <span className="text-gray-500 w-12 pt-1">{label}</span>
      <div className="flex-1 flex flex-wrap gap-1">
        {recipients.map((r) => (
          <span
            key={r.address}
            className="inline-flex items-center gap-1 px-2 py-0.5 bg-gray-100 rounded-full text-sm"
          >
            <Avatar name={r.name} email={r.address} size={20} />
            <span>{r.name || r.address}</span>
            {r.trustLevel !== undefined && (
              <TrustBadge level={r.trustLevel} size="sm" />
            )}
            <button
              onClick={() => handleRemoveRecipient(r.address)}
              className="hover:text-red-500"
            >
              <CloseIcon className="w-3 h-3" />
            </button>
          </span>
        ))}
        <input
          type="text"
          value={input}
          onChange={(e) => handleInputChange(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
          autoFocus={autoFocus}
          placeholder={recipients.length === 0 ? 'Enter email address' : ''}
          className="flex-1 min-w-[200px] focus:outline-none"
        />
      </div>

      {/* Suggestions dropdown */}
      {showSuggestions && (
        <div className="absolute top-full left-12 right-4 bg-white border rounded shadow-lg z-10 max-h-48 overflow-auto">
          {suggestions.map((s) => (
            <button
              key={s.address}
              onClick={() => handleAddRecipient(s)}
              className="w-full flex items-center gap-2 px-3 py-2 hover:bg-gray-100 text-left"
            >
              <Avatar name={s.name} email={s.address} size={32} />
              <div>
                <div className="font-medium">{s.name || s.address}</div>
                {s.name && <div className="text-sm text-gray-500">{s.address}</div>}
              </div>
              {s.trustLevel !== undefined && (
                <TrustBadge level={s.trustLevel} size="sm" className="ml-auto" />
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

// Schedule Picker Component
interface SchedulePickerProps {
  onSchedule: (sendAt: Date) => void;
  onCancel: () => void;
}

const SchedulePicker: React.FC<SchedulePickerProps> = ({ onSchedule, onCancel }) => {
  const [date, setDate] = useState('');
  const [time, setTime] = useState('09:00');

  const presets = [
    { label: 'Tomorrow morning', getDate: () => {
      const d = new Date();
      d.setDate(d.getDate() + 1);
      d.setHours(9, 0, 0, 0);
      return d;
    }},
    { label: 'Tomorrow afternoon', getDate: () => {
      const d = new Date();
      d.setDate(d.getDate() + 1);
      d.setHours(14, 0, 0, 0);
      return d;
    }},
    { label: 'Monday morning', getDate: () => {
      const d = new Date();
      d.setDate(d.getDate() + ((8 - d.getDay()) % 7 || 7));
      d.setHours(9, 0, 0, 0);
      return d;
    }},
  ];

  const handleCustomSchedule = () => {
    if (!date) return;
    const [hours, minutes] = time.split(':').map(Number);
    const sendAt = new Date(date);
    sendAt.setHours(hours, minutes, 0, 0);
    onSchedule(sendAt);
  };

  return (
    <div className="absolute bottom-full left-0 mb-2 bg-white border rounded shadow-lg p-4 w-72">
      <h3 className="font-medium mb-3">Schedule send</h3>

      {/* Presets */}
      <div className="space-y-1 mb-4">
        {presets.map((preset) => (
          <button
            key={preset.label}
            onClick={() => onSchedule(preset.getDate())}
            className="w-full text-left px-3 py-2 hover:bg-gray-100 rounded text-sm"
          >
            {preset.label}
          </button>
        ))}
      </div>

      {/* Custom date/time */}
      <div className="border-t pt-3">
        <div className="flex gap-2 mb-2">
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            min={new Date().toISOString().split('T')[0]}
            className="flex-1 px-2 py-1 border rounded text-sm"
          />
          <input
            type="time"
            value={time}
            onChange={(e) => setTime(e.target.value)}
            className="px-2 py-1 border rounded text-sm"
          />
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleCustomSchedule}
            disabled={!date}
            className="flex-1 px-3 py-1.5 bg-blue-600 text-white rounded text-sm disabled:opacity-50"
          >
            Schedule
          </button>
          <button
            onClick={onCancel}
            className="px-3 py-1.5 border rounded text-sm"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

// Icon components
const MinimizeIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);

const CloseIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

const SendIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const AttachmentIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
  </svg>
);

const LockIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
  </svg>
);

const TrashIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
);

const ClockIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </svg>
);

const SpinnerIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 12a9 9 0 11-6.219-8.56" />
  </svg>
);

export default ComposeEmail;
