// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Real-time Collaboration Components
 *
 * Provides live draft editing, presence indicators, collaborative threads,
 * shared drafts, comment threads, and version history.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';

// ============================================================================
// Types
// ============================================================================

interface CollaborativeDraft {
  id: string;
  ownerId: string;
  subject: string;
  body: string;
  recipients: string[];
  cc: string[];
  bcc: string[];
  collaborators: DraftCollaborator[];
  version: number;
  isLocked: boolean;
  lockedBy?: string;
}

interface DraftCollaborator {
  userId: string;
  userName: string;
  userEmail: string;
  permission: 'view' | 'comment' | 'edit' | 'admin';
  cursor?: CursorPosition;
  lastActive?: string;
  color: string;
}

interface CursorPosition {
  field: string;
  offset: number;
  selectionEnd?: number;
}

interface DraftOperation {
  type: 'insert' | 'delete' | 'replace' | 'set_field';
  field: string;
  position: number;
  content?: string;
  length?: number;
  version: number;
}

interface CollaborativeThread {
  id: string;
  emailId: string;
  createdBy: string;
  messageCount: number;
  participants: string[];
  isResolved: boolean;
  lastMessageAt?: string;
}

interface ThreadMessage {
  id: string;
  threadId: string;
  authorId: string;
  authorName: string;
  content: string;
  createdAt: string;
  reactions: MessageReaction[];
}

interface MessageReaction {
  emoji: string;
  count: number;
  userIds: string[];
}

interface DraftVersion {
  id: string;
  version: number;
  subject: string;
  body: string;
  createdBy: string;
  createdByName: string;
  createdAt: string;
  changeSummary?: string;
}

// ============================================================================
// Collaborative Draft Editor
// ============================================================================

interface CollaborativeEditorProps {
  draftId: string;
  currentUserId: string;
  onSend: () => void;
  onSaveDraft: () => void;
}

export function CollaborativeEditor({
  draftId,
  currentUserId,
  onSend,
  onSaveDraft,
}: CollaborativeEditorProps) {
  const [draft, setDraft] = useState<CollaborativeDraft | null>(null);
  const [localContent, setLocalContent] = useState({ subject: '', body: '' });
  const [cursors, setCursors] = useState<Map<string, CursorPosition>>(new Map());
  const [isConnected, setIsConnected] = useState(false);
  const [pendingOps, setPendingOps] = useState<DraftOperation[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const editorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    connectWebSocket();
    return () => {
      wsRef.current?.close();
    };
  }, [draftId]);

  const connectWebSocket = () => {
    const ws = new WebSocket(`/api/drafts/${draftId}/collaborate`);

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleWebSocketMessage(message);
    };

    ws.onclose = () => {
      setIsConnected(false);
      // Reconnect after delay
      setTimeout(connectWebSocket, 3000);
    };

    wsRef.current = ws;
  };

  const handleWebSocketMessage = (message: any) => {
    switch (message.type) {
      case 'draft_state':
        setDraft(message.draft);
        setLocalContent({
          subject: message.draft.subject,
          body: message.draft.body,
        });
        break;

      case 'operation':
        applyRemoteOperation(message.operation);
        break;

      case 'cursor_update':
        setCursors(prev => {
          const next = new Map(prev);
          next.set(message.userId, message.cursor);
          return next;
        });
        break;

      case 'collaborator_joined':
        setDraft(prev => prev ? {
          ...prev,
          collaborators: [...prev.collaborators, message.collaborator],
        } : null);
        break;

      case 'collaborator_left':
        setDraft(prev => prev ? {
          ...prev,
          collaborators: prev.collaborators.filter(c => c.userId !== message.userId),
        } : null);
        setCursors(prev => {
          const next = new Map(prev);
          next.delete(message.userId);
          return next;
        });
        break;
    }
  };

  const applyRemoteOperation = (op: DraftOperation) => {
    setLocalContent(prev => {
      const field = op.field as 'subject' | 'body';
      let newValue = prev[field];

      switch (op.type) {
        case 'insert':
          newValue = newValue.slice(0, op.position) +
            (op.content || '') +
            newValue.slice(op.position);
          break;
        case 'delete':
          newValue = newValue.slice(0, op.position) +
            newValue.slice(op.position + (op.length || 1));
          break;
        case 'replace':
          newValue = newValue.slice(0, op.position) +
            (op.content || '') +
            newValue.slice(op.position + (op.length || 0));
          break;
        case 'set_field':
          newValue = op.content || '';
          break;
      }

      return { ...prev, [field]: newValue };
    });
  };

  const sendOperation = (op: DraftOperation) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'operation',
        operation: { ...op, version: draft?.version || 0 },
      }));
    }
  };

  const handleSubjectChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    const oldValue = localContent.subject;

    setLocalContent(prev => ({ ...prev, subject: newValue }));

    // Generate operation
    if (newValue.length > oldValue.length) {
      const pos = findDiffPosition(oldValue, newValue);
      const inserted = newValue.slice(pos, pos + (newValue.length - oldValue.length));
      sendOperation({
        type: 'insert',
        field: 'subject',
        position: pos,
        content: inserted,
        version: draft?.version || 0,
      });
    } else if (newValue.length < oldValue.length) {
      const pos = findDiffPosition(oldValue, newValue);
      sendOperation({
        type: 'delete',
        field: 'subject',
        position: pos,
        length: oldValue.length - newValue.length,
        version: draft?.version || 0,
      });
    }
  };

  const handleBodyChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    const oldValue = localContent.body;

    setLocalContent(prev => ({ ...prev, body: newValue }));

    // Generate operation (simplified - production would use OT)
    sendOperation({
      type: 'set_field',
      field: 'body',
      position: 0,
      content: newValue,
      version: draft?.version || 0,
    });
  };

  const handleCursorMove = (field: string, offset: number, selectionEnd?: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'cursor_update',
        cursor: { field, offset, selectionEnd },
      }));
    }
  };

  const findDiffPosition = (oldStr: string, newStr: string): number => {
    for (let i = 0; i < Math.min(oldStr.length, newStr.length); i++) {
      if (oldStr[i] !== newStr[i]) return i;
    }
    return Math.min(oldStr.length, newStr.length);
  };

  const otherCollaborators = draft?.collaborators.filter(c => c.userId !== currentUserId) || [];
  const activeCollaborators = otherCollaborators.filter(c =>
    c.lastActive && (Date.now() - new Date(c.lastActive).getTime()) < 300000
  );

  return (
    <div className="collaborative-editor">
      <div className="editor-header">
        <div className="connection-status">
          <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
          {isConnected ? 'Connected' : 'Reconnecting...'}
        </div>
        <CollaboratorAvatars
          collaborators={activeCollaborators}
          cursors={cursors}
        />
        <div className="editor-actions">
          <button onClick={onSaveDraft}>Save Draft</button>
          <button className="primary" onClick={onSend}>Send</button>
        </div>
      </div>

      <div className="editor-body" ref={editorRef}>
        <div className="field-group">
          <label>To:</label>
          <RecipientInput
            value={draft?.recipients || []}
            onChange={(recipients) => {
              // Update recipients
            }}
          />
        </div>

        <div className="field-group">
          <label>Subject:</label>
          <div className="input-with-cursors">
            <input
              type="text"
              value={localContent.subject}
              onChange={handleSubjectChange}
              onSelect={(e) => {
                const input = e.target as HTMLInputElement;
                handleCursorMove('subject', input.selectionStart || 0, input.selectionEnd || undefined);
              }}
            />
            <RemoteCursors
              cursors={Array.from(cursors.entries())
                .filter(([_, c]) => c.field === 'subject')
                .map(([userId, cursor]) => ({
                  userId,
                  offset: cursor.offset,
                  color: otherCollaborators.find(c => c.userId === userId)?.color || '#ccc',
                  userName: otherCollaborators.find(c => c.userId === userId)?.userName || 'Unknown',
                }))}
              fieldType="input"
              content={localContent.subject}
            />
          </div>
        </div>

        <div className="field-group body-field">
          <div className="textarea-with-cursors">
            <textarea
              value={localContent.body}
              onChange={handleBodyChange}
              onSelect={(e) => {
                const textarea = e.target as HTMLTextAreaElement;
                handleCursorMove('body', textarea.selectionStart, textarea.selectionEnd);
              }}
              placeholder="Compose your email..."
            />
            <RemoteCursors
              cursors={Array.from(cursors.entries())
                .filter(([_, c]) => c.field === 'body')
                .map(([userId, cursor]) => ({
                  userId,
                  offset: cursor.offset,
                  selectionEnd: cursor.selectionEnd,
                  color: otherCollaborators.find(c => c.userId === userId)?.color || '#ccc',
                  userName: otherCollaborators.find(c => c.userId === userId)?.userName || 'Unknown',
                }))}
              fieldType="textarea"
              content={localContent.body}
            />
          </div>
        </div>
      </div>

      <VersionHistoryPanel draftId={draftId} currentVersion={draft?.version || 0} />
    </div>
  );
}

// ============================================================================
// Collaborator Avatars
// ============================================================================

interface CollaboratorAvatarsProps {
  collaborators: DraftCollaborator[];
  cursors: Map<string, CursorPosition>;
}

function CollaboratorAvatars({ collaborators, cursors }: CollaboratorAvatarsProps) {
  return (
    <div className="collaborator-avatars">
      {collaborators.map(collaborator => (
        <div
          key={collaborator.userId}
          className="collaborator-avatar"
          style={{ borderColor: collaborator.color }}
          title={`${collaborator.userName} - ${getActivityDescription(cursors.get(collaborator.userId))}`}
        >
          {collaborator.userName.charAt(0)}
          <span className="typing-indicator" style={{ backgroundColor: collaborator.color }} />
        </div>
      ))}
      {collaborators.length > 0 && (
        <span className="collaborator-count">
          {collaborators.length} editing
        </span>
      )}
    </div>
  );
}

function getActivityDescription(cursor?: CursorPosition): string {
  if (!cursor) return 'viewing';
  return `editing ${cursor.field}`;
}

// ============================================================================
// Remote Cursors Display
// ============================================================================

interface RemoteCursor {
  userId: string;
  offset: number;
  selectionEnd?: number;
  color: string;
  userName: string;
}

interface RemoteCursorsProps {
  cursors: RemoteCursor[];
  fieldType: 'input' | 'textarea';
  content: string;
}

function RemoteCursors({ cursors, fieldType, content }: RemoteCursorsProps) {
  // This is a simplified version - real implementation would
  // calculate pixel positions based on text metrics
  return (
    <div className="remote-cursors">
      {cursors.map(cursor => (
        <div
          key={cursor.userId}
          className="remote-cursor"
          style={{
            left: `${(cursor.offset / Math.max(content.length, 1)) * 100}%`,
            backgroundColor: cursor.color,
          }}
        >
          <span className="cursor-label" style={{ backgroundColor: cursor.color }}>
            {cursor.userName}
          </span>
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// Recipient Input
// ============================================================================

interface RecipientInputProps {
  value: string[];
  onChange: (recipients: string[]) => void;
}

function RecipientInput({ value, onChange }: RecipientInputProps) {
  const [inputValue, setInputValue] = useState('');

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      if (inputValue.trim()) {
        onChange([...value, inputValue.trim()]);
        setInputValue('');
      }
    } else if (e.key === 'Backspace' && !inputValue && value.length > 0) {
      onChange(value.slice(0, -1));
    }
  };

  const removeRecipient = (index: number) => {
    onChange(value.filter((_, i) => i !== index));
  };

  return (
    <div className="recipient-input">
      {value.map((recipient, index) => (
        <span key={index} className="recipient-chip">
          {recipient}
          <button onClick={() => removeRecipient(index)}>&times;</button>
        </span>
      ))}
      <input
        type="text"
        value={inputValue}
        onChange={e => setInputValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={value.length === 0 ? 'Add recipients...' : ''}
      />
    </div>
  );
}

// ============================================================================
// Collaborative Thread
// ============================================================================

interface CollaborativeThreadProps {
  thread: CollaborativeThread;
  messages: ThreadMessage[];
  currentUserId: string;
  onAddMessage: (content: string) => void;
  onAddReaction: (messageId: string, emoji: string) => void;
  onResolve: () => void;
}

export function CollaborativeThreadPanel({
  thread,
  messages,
  currentUserId,
  onAddMessage,
  onAddReaction,
  onResolve,
}: CollaborativeThreadProps) {
  const [newMessage, setNewMessage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (newMessage.trim()) {
      onAddMessage(newMessage);
      setNewMessage('');
    }
  };

  const commonEmojis = ['👍', '👎', '❤️', '😄', '🎉', '👀'];

  return (
    <div className={`collaborative-thread ${thread.isResolved ? 'resolved' : ''}`}>
      <div className="thread-header">
        <h3><ThreadIcon /> Discussion</h3>
        <div className="thread-meta">
          <span>{thread.messageCount} messages</span>
          <span>{thread.participants.length} participants</span>
        </div>
        {!thread.isResolved && (
          <button className="resolve-button" onClick={onResolve}>
            <CheckIcon /> Resolve
          </button>
        )}
      </div>

      <div className="thread-messages">
        {messages.map(message => (
          <div key={message.id} className="thread-message">
            <div className="message-author">
              <span className="author-name">{message.authorName}</span>
              <span className="message-time">{formatTime(message.createdAt)}</span>
            </div>
            <div className="message-content">{message.content}</div>
            <div className="message-reactions">
              {message.reactions.map(reaction => (
                <button
                  key={reaction.emoji}
                  className={`reaction ${reaction.userIds.includes(currentUserId) ? 'active' : ''}`}
                  onClick={() => onAddReaction(message.id, reaction.emoji)}
                >
                  {reaction.emoji} {reaction.count}
                </button>
              ))}
              <div className="add-reaction">
                {commonEmojis.map(emoji => (
                  <button
                    key={emoji}
                    className="emoji-button"
                    onClick={() => onAddReaction(message.id, emoji)}
                  >
                    {emoji}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {!thread.isResolved && (
        <form className="thread-input" onSubmit={handleSubmit}>
          <input
            type="text"
            value={newMessage}
            onChange={e => setNewMessage(e.target.value)}
            placeholder="Add to discussion..."
          />
          <button type="submit" disabled={!newMessage.trim()}>
            <SendIcon />
          </button>
        </form>
      )}
    </div>
  );
}

// ============================================================================
// Presence Indicator
// ============================================================================

interface UserPresence {
  userId: string;
  userName: string;
  status: 'online' | 'away' | 'offline';
  currentView?: {
    type: string;
    resourceId?: string;
  };
  typingIn?: string;
}

interface PresenceIndicatorProps {
  presence: UserPresence[];
  contextId?: string;
}

export function PresenceIndicator({ presence, contextId }: PresenceIndicatorProps) {
  const online = presence.filter(p => p.status === 'online');
  const away = presence.filter(p => p.status === 'away');
  const typing = presence.filter(p => p.typingIn === contextId);

  return (
    <div className="presence-indicator">
      <div className="presence-avatars">
        {online.slice(0, 5).map(user => (
          <div
            key={user.userId}
            className={`presence-avatar ${user.status}`}
            title={`${user.userName} - ${user.status}`}
          >
            {user.userName.charAt(0)}
          </div>
        ))}
        {online.length > 5 && (
          <div className="presence-overflow">+{online.length - 5}</div>
        )}
      </div>

      {typing.length > 0 && (
        <div className="typing-indicator">
          <TypingDots />
          <span>
            {typing.length === 1
              ? `${typing[0].userName} is typing...`
              : `${typing.length} people are typing...`}
          </span>
        </div>
      )}
    </div>
  );
}

function TypingDots() {
  return (
    <span className="typing-dots">
      <span className="dot" />
      <span className="dot" />
      <span className="dot" />
    </span>
  );
}

// ============================================================================
// Version History Panel
// ============================================================================

interface VersionHistoryPanelProps {
  draftId: string;
  currentVersion: number;
  onRestore?: (versionId: string) => void;
}

export function VersionHistoryPanel({ draftId, currentVersion, onRestore }: VersionHistoryPanelProps) {
  const [versions, setVersions] = useState<DraftVersion[]>([]);
  const [expanded, setExpanded] = useState(false);
  const [selectedVersions, setSelectedVersions] = useState<[string?, string?]>([undefined, undefined]);
  const [diffView, setDiffView] = useState(false);

  useEffect(() => {
    if (expanded) {
      fetchVersions();
    }
  }, [expanded, draftId]);

  const fetchVersions = async () => {
    try {
      const response = await fetch(`/api/drafts/${draftId}/versions?limit=20`);
      const data = await response.json();
      setVersions(data.versions);
    } catch (error) {
      console.error('Failed to fetch versions:', error);
    }
  };

  const handleVersionClick = (version: DraftVersion) => {
    if (diffView) {
      if (!selectedVersions[0]) {
        setSelectedVersions([version.id, undefined]);
      } else if (!selectedVersions[1]) {
        setSelectedVersions([selectedVersions[0], version.id]);
      } else {
        setSelectedVersions([version.id, undefined]);
      }
    }
  };

  return (
    <div className={`version-history-panel ${expanded ? 'expanded' : ''}`}>
      <button className="toggle-button" onClick={() => setExpanded(!expanded)}>
        <HistoryIcon />
        <span>Version History</span>
        <span className="version-badge">v{currentVersion}</span>
      </button>

      {expanded && (
        <div className="version-list">
          <div className="version-controls">
            <label>
              <input
                type="checkbox"
                checked={diffView}
                onChange={e => {
                  setDiffView(e.target.checked);
                  setSelectedVersions([undefined, undefined]);
                }}
              />
              Compare versions
            </label>
          </div>

          {versions.map(version => (
            <div
              key={version.id}
              className={`version-item ${
                selectedVersions.includes(version.id) ? 'selected' : ''
              }`}
              onClick={() => handleVersionClick(version)}
            >
              <div className="version-info">
                <span className="version-number">v{version.version}</span>
                <span className="version-author">{version.createdByName}</span>
                <span className="version-time">{formatTime(version.createdAt)}</span>
              </div>
              {version.changeSummary && (
                <div className="version-summary">{version.changeSummary}</div>
              )}
              {onRestore && (
                <button
                  className="restore-button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRestore(version.id);
                  }}
                >
                  Restore
                </button>
              )}
            </div>
          ))}

          {diffView && selectedVersions[0] && selectedVersions[1] && (
            <VersionDiffView
              draftId={draftId}
              versionA={selectedVersions[0]}
              versionB={selectedVersions[1]}
            />
          )}
        </div>
      )}
    </div>
  );
}

interface VersionDiffViewProps {
  draftId: string;
  versionA: string;
  versionB: string;
}

function VersionDiffView({ draftId, versionA, versionB }: VersionDiffViewProps) {
  const [diff, setDiff] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDiff();
  }, [versionA, versionB]);

  const fetchDiff = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `/api/drafts/${draftId}/versions/diff?a=${versionA}&b=${versionB}`
      );
      const data = await response.json();
      setDiff(data);
    } catch (error) {
      console.error('Failed to fetch diff:', error);
    }
    setLoading(false);
  };

  if (loading) return <div className="diff-loading">Loading diff...</div>;
  if (!diff) return null;

  return (
    <div className="version-diff">
      <h4>Changes: v{diff.versionA} → v{diff.versionB}</h4>
      <div className="diff-content">
        {diff.bodyDiff.map((chunk: any, index: number) => (
          <span
            key={index}
            className={`diff-chunk ${chunk.diffType.toLowerCase()}`}
          >
            {chunk.content}
          </span>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Shared Drafts Folder
// ============================================================================

interface SharedDraft {
  id: string;
  subject: string;
  owner: string;
  ownerName: string;
  collaborators: number;
  lastModified: string;
  version: number;
}

interface SharedDraftsFolderProps {
  drafts: SharedDraft[];
  onOpen: (draftId: string) => void;
  onCreate: () => void;
}

export function SharedDraftsFolder({ drafts, onOpen, onCreate }: SharedDraftsFolderProps) {
  return (
    <div className="shared-drafts-folder">
      <div className="folder-header">
        <h2><FolderIcon /> Shared Drafts</h2>
        <button className="primary" onClick={onCreate}>
          <PlusIcon /> New Shared Draft
        </button>
      </div>

      <div className="drafts-list">
        {drafts.map(draft => (
          <div
            key={draft.id}
            className="shared-draft-item"
            onClick={() => onOpen(draft.id)}
          >
            <div className="draft-icon">
              <DocumentIcon />
            </div>
            <div className="draft-info">
              <span className="draft-subject">{draft.subject || 'Untitled'}</span>
              <span className="draft-owner">by {draft.ownerName}</span>
            </div>
            <div className="draft-meta">
              <span className="collaborators">
                <UsersIcon /> {draft.collaborators}
              </span>
              <span className="version">v{draft.version}</span>
              <span className="modified">{formatTime(draft.lastModified)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Helper Functions
// ============================================================================

function formatTime(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
  return date.toLocaleDateString();
}

// ============================================================================
// Icon Components
// ============================================================================

function ThreadIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>;
}

function CheckIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="20 6 9 17 4 12" />
  </svg>;
}

function SendIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>;
}

function HistoryIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M3 3v5h5" />
    <path d="M3.05 13A9 9 0 1 0 6 5.3L3 8" />
    <path d="M12 7v5l4 2" />
  </svg>;
}

function FolderIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
  </svg>;
}

function PlusIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>;
}

function DocumentIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
  </svg>;
}

function UsersIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </svg>;
}

export default {
  CollaborativeEditor,
  CollaborativeThreadPanel,
  PresenceIndicator,
  VersionHistoryPanel,
  SharedDraftsFolder,
};
