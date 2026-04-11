// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Team Collaboration Components
 *
 * Provides shared mailbox management, internal notes, @mentions,
 * collision detection, email assignment, and team activity feed.
 */

import React, { useState, useEffect, useCallback } from 'react';

// ============================================================================
// Types
// ============================================================================

interface SharedMailbox {
  id: string;
  name: string;
  emailAddress: string;
  description?: string;
  memberCount: number;
  unreadCount: number;
  settings: MailboxSettings;
}

interface MailboxSettings {
  autoAssign: boolean;
  roundRobin: boolean;
  notifyAllMembers: boolean;
  requireAssignment: boolean;
  slaHours?: number;
}

interface MailboxMember {
  userId: string;
  userName: string;
  userEmail: string;
  role: 'owner' | 'admin' | 'member' | 'readonly';
  addedAt: string;
}

interface InternalNote {
  id: string;
  emailId: string;
  authorId: string;
  authorName: string;
  content: string;
  createdAt: string;
  mentions: string[];
  isPinned: boolean;
}

interface EmailPresence {
  userId: string;
  userName: string;
  activity: 'viewing' | 'replying' | 'forwarding' | 'editing';
  startedAt: string;
}

interface EmailAssignment {
  emailId: string;
  assignedTo: string;
  assignedToName: string;
  assignedBy: string;
  assignedAt: string;
  status: 'pending' | 'in_progress' | 'on_hold' | 'resolved' | 'escalated';
  priority: 'low' | 'normal' | 'high' | 'urgent';
  dueAt?: string;
}

interface TeamActivity {
  id: string;
  userId: string;
  userName: string;
  activityType: string;
  emailId?: string;
  emailSubject?: string;
  details?: string;
  createdAt: string;
}

interface TeamMember {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  status: 'online' | 'away' | 'offline';
}

// ============================================================================
// Shared Mailbox List
// ============================================================================

interface SharedMailboxListProps {
  mailboxes: SharedMailbox[];
  selectedId?: string;
  onSelect: (mailbox: SharedMailbox) => void;
  onManage: (mailbox: SharedMailbox) => void;
}

export function SharedMailboxList({
  mailboxes,
  selectedId,
  onSelect,
  onManage,
}: SharedMailboxListProps) {
  return (
    <div className="shared-mailbox-list">
      <h3>Shared Mailboxes</h3>
      <ul role="listbox" aria-label="Shared mailboxes">
        {mailboxes.map(mailbox => (
          <li
            key={mailbox.id}
            role="option"
            aria-selected={mailbox.id === selectedId}
            className={`mailbox-item ${mailbox.id === selectedId ? 'selected' : ''}`}
          >
            <button
              className="mailbox-button"
              onClick={() => onSelect(mailbox)}
            >
              <div className="mailbox-icon">
                <MailboxIcon />
              </div>
              <div className="mailbox-info">
                <span className="mailbox-name">{mailbox.name}</span>
                <span className="mailbox-email">{mailbox.emailAddress}</span>
              </div>
              {mailbox.unreadCount > 0 && (
                <span className="unread-badge">{mailbox.unreadCount}</span>
              )}
            </button>
            <button
              className="manage-button"
              onClick={(e) => {
                e.stopPropagation();
                onManage(mailbox);
              }}
              aria-label={`Manage ${mailbox.name}`}
            >
              <SettingsIcon />
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

// ============================================================================
// Mailbox Settings Modal
// ============================================================================

interface MailboxSettingsModalProps {
  mailbox: SharedMailbox;
  members: MailboxMember[];
  onClose: () => void;
  onSave: (settings: MailboxSettings) => void;
  onAddMember: (email: string, role: string) => void;
  onRemoveMember: (userId: string) => void;
}

export function MailboxSettingsModal({
  mailbox,
  members,
  onClose,
  onSave,
  onAddMember,
  onRemoveMember,
}: MailboxSettingsModalProps) {
  const [settings, setSettings] = useState(mailbox.settings);
  const [newMemberEmail, setNewMemberEmail] = useState('');
  const [newMemberRole, setNewMemberRole] = useState('member');
  const [activeTab, setActiveTab] = useState<'settings' | 'members'>('settings');

  const handleSave = () => {
    onSave(settings);
    onClose();
  };

  const handleAddMember = () => {
    if (newMemberEmail) {
      onAddMember(newMemberEmail, newMemberRole);
      setNewMemberEmail('');
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal mailbox-settings-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{mailbox.name} Settings</h2>
          <button className="close-button" onClick={onClose} aria-label="Close">
            <CloseIcon />
          </button>
        </div>

        <div className="modal-tabs">
          <button
            className={activeTab === 'settings' ? 'active' : ''}
            onClick={() => setActiveTab('settings')}
          >
            Settings
          </button>
          <button
            className={activeTab === 'members' ? 'active' : ''}
            onClick={() => setActiveTab('members')}
          >
            Members ({members.length})
          </button>
        </div>

        <div className="modal-body">
          {activeTab === 'settings' ? (
            <div className="settings-tab">
              <div className="setting-group">
                <h4>Assignment</h4>
                <label>
                  <input
                    type="checkbox"
                    checked={settings.autoAssign}
                    onChange={e => setSettings({ ...settings, autoAssign: e.target.checked })}
                  />
                  Auto-assign incoming emails
                </label>
                {settings.autoAssign && (
                  <label>
                    <input
                      type="checkbox"
                      checked={settings.roundRobin}
                      onChange={e => setSettings({ ...settings, roundRobin: e.target.checked })}
                    />
                    Use round-robin assignment
                  </label>
                )}
                <label>
                  <input
                    type="checkbox"
                    checked={settings.requireAssignment}
                    onChange={e => setSettings({ ...settings, requireAssignment: e.target.checked })}
                  />
                  Require assignment before reply
                </label>
              </div>

              <div className="setting-group">
                <h4>Notifications</h4>
                <label>
                  <input
                    type="checkbox"
                    checked={settings.notifyAllMembers}
                    onChange={e => setSettings({ ...settings, notifyAllMembers: e.target.checked })}
                  />
                  Notify all members of new emails
                </label>
              </div>

              <div className="setting-group">
                <h4>SLA</h4>
                <label>
                  Response time goal (hours)
                  <input
                    type="number"
                    value={settings.slaHours || ''}
                    onChange={e => setSettings({
                      ...settings,
                      slaHours: e.target.value ? parseInt(e.target.value) : undefined
                    })}
                    placeholder="No SLA"
                    min="1"
                  />
                </label>
              </div>
            </div>
          ) : (
            <div className="members-tab">
              <div className="add-member-form">
                <input
                  type="email"
                  value={newMemberEmail}
                  onChange={e => setNewMemberEmail(e.target.value)}
                  placeholder="Email address"
                />
                <select
                  value={newMemberRole}
                  onChange={e => setNewMemberRole(e.target.value)}
                >
                  <option value="member">Member</option>
                  <option value="admin">Admin</option>
                  <option value="readonly">Read Only</option>
                </select>
                <button onClick={handleAddMember}>Add</button>
              </div>

              <ul className="members-list">
                {members.map(member => (
                  <li key={member.userId} className="member-item">
                    <div className="member-info">
                      <span className="member-name">{member.userName}</span>
                      <span className="member-email">{member.userEmail}</span>
                      <span className={`member-role ${member.role}`}>{member.role}</span>
                    </div>
                    {member.role !== 'owner' && (
                      <button
                        className="remove-member"
                        onClick={() => onRemoveMember(member.userId)}
                        aria-label={`Remove ${member.userName}`}
                      >
                        <CloseIcon />
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="secondary" onClick={onClose}>Cancel</button>
          <button className="primary" onClick={handleSave}>Save Changes</button>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Internal Notes Panel
// ============================================================================

interface InternalNotesPanelProps {
  emailId: string;
  notes: InternalNote[];
  teamMembers: TeamMember[];
  currentUserId: string;
  onAddNote: (content: string, mentions: string[]) => void;
  onDeleteNote: (noteId: string) => void;
  onPinNote: (noteId: string, pinned: boolean) => void;
}

export function InternalNotesPanel({
  emailId,
  notes,
  teamMembers,
  currentUserId,
  onAddNote,
  onDeleteNote,
  onPinNote,
}: InternalNotesPanelProps) {
  const [newNote, setNewNote] = useState('');
  const [showMentions, setShowMentions] = useState(false);
  const [mentionSearch, setMentionSearch] = useState('');

  const parseMentions = (text: string): string[] => {
    const mentionRegex = /@(\w+)/g;
    const mentions: string[] = [];
    let match;
    while ((match = mentionRegex.exec(text)) !== null) {
      const member = teamMembers.find(m =>
        m.name.toLowerCase().includes(match[1].toLowerCase())
      );
      if (member) mentions.push(member.id);
    }
    return mentions;
  };

  const handleSubmit = () => {
    if (newNote.trim()) {
      const mentions = parseMentions(newNote);
      onAddNote(newNote, mentions);
      setNewNote('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === '@') {
      setShowMentions(true);
      setMentionSearch('');
    } else if (showMentions) {
      if (e.key === 'Escape') {
        setShowMentions(false);
      } else if (e.key === ' ') {
        setShowMentions(false);
      }
    }

    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSubmit();
    }
  };

  const insertMention = (member: TeamMember) => {
    setNewNote(prev => prev + `@${member.name.split(' ')[0]} `);
    setShowMentions(false);
  };

  const filteredMembers = teamMembers.filter(m =>
    m.name.toLowerCase().includes(mentionSearch.toLowerCase())
  );

  const pinnedNotes = notes.filter(n => n.isPinned);
  const unpinnedNotes = notes.filter(n => !n.isPinned);

  return (
    <div className="internal-notes-panel">
      <h3>
        <NoteIcon /> Internal Notes
        <span className="note-count">{notes.length}</span>
      </h3>

      <div className="note-composer">
        <div className="textarea-wrapper">
          <textarea
            value={newNote}
            onChange={e => setNewNote(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Add a note... Use @ to mention teammates"
            rows={3}
          />
          {showMentions && (
            <div className="mention-dropdown">
              {filteredMembers.map(member => (
                <button
                  key={member.id}
                  className="mention-option"
                  onClick={() => insertMention(member)}
                >
                  <span className="mention-name">{member.name}</span>
                  <span className={`status-dot ${member.status}`} />
                </button>
              ))}
            </div>
          )}
        </div>
        <button
          className="submit-note"
          onClick={handleSubmit}
          disabled={!newNote.trim()}
        >
          Add Note
        </button>
      </div>

      <div className="notes-list">
        {pinnedNotes.length > 0 && (
          <div className="pinned-notes">
            <h4><PinIcon /> Pinned</h4>
            {pinnedNotes.map(note => (
              <NoteItem
                key={note.id}
                note={note}
                currentUserId={currentUserId}
                onDelete={onDeleteNote}
                onPin={onPinNote}
              />
            ))}
          </div>
        )}

        {unpinnedNotes.map(note => (
          <NoteItem
            key={note.id}
            note={note}
            currentUserId={currentUserId}
            onDelete={onDeleteNote}
            onPin={onPinNote}
          />
        ))}
      </div>
    </div>
  );
}

interface NoteItemProps {
  note: InternalNote;
  currentUserId: string;
  onDelete: (noteId: string) => void;
  onPin: (noteId: string, pinned: boolean) => void;
}

function NoteItem({ note, currentUserId, onDelete, onPin }: NoteItemProps) {
  const isOwner = note.authorId === currentUserId;

  const formatContent = (content: string) => {
    return content.replace(/@(\w+)/g, '<span class="mention">@$1</span>');
  };

  return (
    <div className={`note-item ${note.isPinned ? 'pinned' : ''}`}>
      <div className="note-header">
        <span className="note-author">{note.authorName}</span>
        <span className="note-time">{formatTime(note.createdAt)}</span>
      </div>
      <div
        className="note-content"
        dangerouslySetInnerHTML={{ __html: formatContent(note.content) }}
      />
      <div className="note-actions">
        <button
          onClick={() => onPin(note.id, !note.isPinned)}
          aria-label={note.isPinned ? 'Unpin' : 'Pin'}
        >
          <PinIcon />
        </button>
        {isOwner && (
          <button
            onClick={() => onDelete(note.id)}
            aria-label="Delete note"
          >
            <TrashIcon />
          </button>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Collision Detection Banner
// ============================================================================

interface CollisionBannerProps {
  presence: EmailPresence[];
  currentUserId: string;
}

export function CollisionBanner({ presence, currentUserId }: CollisionBannerProps) {
  const others = presence.filter(p => p.userId !== currentUserId);

  if (others.length === 0) return null;

  const getActivityText = (activity: EmailPresence['activity']) => {
    switch (activity) {
      case 'viewing': return 'is viewing';
      case 'replying': return 'is replying to';
      case 'forwarding': return 'is forwarding';
      case 'editing': return 'is editing';
    }
  };

  const isConflicting = others.some(p =>
    p.activity === 'replying' || p.activity === 'forwarding'
  );

  return (
    <div className={`collision-banner ${isConflicting ? 'warning' : 'info'}`}>
      <div className="presence-avatars">
        {others.map(p => (
          <div key={p.userId} className="presence-avatar" title={p.userName}>
            {p.userName.charAt(0)}
          </div>
        ))}
      </div>
      <div className="presence-text">
        {others.length === 1 ? (
          <span>
            <strong>{others[0].userName}</strong> {getActivityText(others[0].activity)} this email
          </span>
        ) : (
          <span>
            <strong>{others.length} people</strong> are viewing this email
          </span>
        )}
      </div>
      {isConflicting && (
        <div className="conflict-warning">
          <WarningIcon />
          <span>Someone is already replying - coordinate to avoid duplicate responses</span>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Assignment Panel
// ============================================================================

interface AssignmentPanelProps {
  assignment?: EmailAssignment;
  teamMembers: TeamMember[];
  onAssign: (userId: string, priority: string, dueAt?: string) => void;
  onUpdateStatus: (status: string) => void;
}

export function AssignmentPanel({
  assignment,
  teamMembers,
  onAssign,
  onUpdateStatus,
}: AssignmentPanelProps) {
  const [selectedUser, setSelectedUser] = useState(assignment?.assignedTo || '');
  const [priority, setPriority] = useState(assignment?.priority || 'normal');
  const [dueDate, setDueDate] = useState(assignment?.dueAt?.split('T')[0] || '');
  const [showAssignForm, setShowAssignForm] = useState(!assignment);

  const handleAssign = () => {
    onAssign(selectedUser, priority, dueDate || undefined);
    setShowAssignForm(false);
  };

  const statusOptions = ['pending', 'in_progress', 'on_hold', 'resolved', 'escalated'];

  return (
    <div className="assignment-panel">
      <h3>Assignment</h3>

      {assignment && !showAssignForm ? (
        <div className="current-assignment">
          <div className="assignee">
            <span className="label">Assigned to:</span>
            <span className="value">{assignment.assignedToName}</span>
          </div>
          <div className="status">
            <span className="label">Status:</span>
            <select
              value={assignment.status}
              onChange={e => onUpdateStatus(e.target.value)}
              className={`status-select ${assignment.status}`}
            >
              {statusOptions.map(status => (
                <option key={status} value={status}>
                  {status.replace('_', ' ')}
                </option>
              ))}
            </select>
          </div>
          <div className="priority">
            <span className="label">Priority:</span>
            <span className={`priority-badge ${assignment.priority}`}>
              {assignment.priority}
            </span>
          </div>
          {assignment.dueAt && (
            <div className="due-date">
              <span className="label">Due:</span>
              <span className="value">{formatDate(assignment.dueAt)}</span>
            </div>
          )}
          <button
            className="reassign-button"
            onClick={() => setShowAssignForm(true)}
          >
            Reassign
          </button>
        </div>
      ) : (
        <div className="assign-form">
          <div className="form-group">
            <label>Assign to:</label>
            <select
              value={selectedUser}
              onChange={e => setSelectedUser(e.target.value)}
            >
              <option value="">Select team member...</option>
              {teamMembers.map(member => (
                <option key={member.id} value={member.id}>
                  {member.name}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Priority:</label>
            <select value={priority} onChange={e => setPriority(e.target.value)}>
              <option value="low">Low</option>
              <option value="normal">Normal</option>
              <option value="high">High</option>
              <option value="urgent">Urgent</option>
            </select>
          </div>

          <div className="form-group">
            <label>Due date (optional):</label>
            <input
              type="date"
              value={dueDate}
              onChange={e => setDueDate(e.target.value)}
            />
          </div>

          <div className="form-actions">
            {assignment && (
              <button className="secondary" onClick={() => setShowAssignForm(false)}>
                Cancel
              </button>
            )}
            <button
              className="primary"
              onClick={handleAssign}
              disabled={!selectedUser}
            >
              Assign
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Team Activity Feed
// ============================================================================

interface ActivityFeedProps {
  activities: TeamActivity[];
  onLoadMore: () => void;
  hasMore: boolean;
}

export function ActivityFeed({ activities, onLoadMore, hasMore }: ActivityFeedProps) {
  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'EmailReceived': return <MailIcon />;
      case 'EmailSent': return <SendIcon />;
      case 'EmailAssigned': return <UserPlusIcon />;
      case 'EmailResolved': return <CheckIcon />;
      case 'NoteAdded': return <NoteIcon />;
      case 'MentionCreated': return <AtIcon />;
      default: return <ActivityIcon />;
    }
  };

  const getActivityText = (activity: TeamActivity) => {
    switch (activity.activityType) {
      case 'EmailReceived':
        return `received an email`;
      case 'EmailSent':
        return `sent a reply`;
      case 'EmailAssigned':
        return `was assigned an email`;
      case 'EmailResolved':
        return `resolved an email`;
      case 'NoteAdded':
        return `added a note`;
      case 'MentionCreated':
        return `mentioned someone`;
      default:
        return activity.details || 'performed an action';
    }
  };

  return (
    <div className="activity-feed">
      <h3>Team Activity</h3>
      <ul className="activity-list">
        {activities.map(activity => (
          <li key={activity.id} className="activity-item">
            <div className="activity-icon">
              {getActivityIcon(activity.activityType)}
            </div>
            <div className="activity-content">
              <span className="activity-user">{activity.userName}</span>
              <span className="activity-text">{getActivityText(activity)}</span>
              {activity.emailSubject && (
                <span className="activity-subject">"{activity.emailSubject}"</span>
              )}
              <span className="activity-time">{formatTime(activity.createdAt)}</span>
            </div>
          </li>
        ))}
      </ul>
      {hasMore && (
        <button className="load-more" onClick={onLoadMore}>
          Load more
        </button>
      )}
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

function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleDateString();
}

// ============================================================================
// Icon Components
// ============================================================================

function MailboxIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
      <polyline points="22,6 12,13 2,6" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function NoteIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
    </svg>
  );
}

function PinIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="12" y1="17" x2="12" y2="22" />
      <path d="M5 17h14v-1.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8a2 2 0 0 0 0 4h1v4.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24z" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
  );
}

function WarningIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

function MailIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
      <polyline points="22,6 12,13 2,6" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function UserPlusIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
      <circle cx="8.5" cy="7" r="4" />
      <line x1="20" y1="8" x2="20" y2="14" />
      <line x1="23" y1="11" x2="17" y2="11" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function AtIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="4" />
      <path d="M16 8v5a3 3 0 0 0 6 0v-1a10 10 0 1 0-3.92 7.94" />
    </svg>
  );
}

function ActivityIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}

export default {
  SharedMailboxList,
  MailboxSettingsModal,
  InternalNotesPanel,
  CollisionBanner,
  AssignmentPanel,
  ActivityFeed,
};
