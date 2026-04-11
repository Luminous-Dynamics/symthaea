// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * SharedInbox - Team Email Collaboration
 *
 * This component provides:
 * - Shared inbox management
 * - Email assignment and delegation
 * - Collision detection (multiple users viewing same email)
 * - Team response templates
 * - SLA tracking and escalation
 */

import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  Inbox,
  Users,
  User,
  Clock,
  CheckCircle,
  AlertTriangle,
  MessageSquare,
  Tag,
  Filter,
  MoreHorizontal,
  ArrowRight,
  Eye,
  Edit3,
  Send,
  Trash2,
  Archive,
  Star,
  Flag,
  Bell,
  Zap,
  TrendingUp,
  BarChart2,
  UserPlus,
  RefreshCw,
  Settings,
  FileText,
  Lock,
  Unlock,
} from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

export interface SharedInbox {
  id: string;
  name: string;
  email: string;
  description: string;
  teamId: string;
  members: InboxMember[];
  settings: InboxSettings;
  stats: InboxStats;
  createdAt: Date;
}

export interface InboxMember {
  userId: string;
  name: string;
  email: string;
  avatar?: string;
  role: 'manager' | 'agent' | 'viewer';
  isOnline: boolean;
  lastSeen?: Date;
  assignedCount: number;
}

export interface InboxSettings {
  autoAssign: boolean;
  roundRobinEnabled: boolean;
  slaEnabled: boolean;
  slaFirstResponse: number; // minutes
  slaResolution: number; // minutes
  escalationEnabled: boolean;
  escalationAfter: number; // minutes
  escalateTo: string; // userId
  collisionDetection: boolean;
  requireAssignment: boolean;
}

export interface InboxStats {
  unassigned: number;
  assigned: number;
  pending: number;
  resolved: number;
  avgResponseTime: number; // minutes
  avgResolutionTime: number; // minutes
  slaBreaches: number;
}

export interface SharedEmail {
  id: string;
  inboxId: string;
  from: { name: string; email: string };
  subject: string;
  preview: string;
  receivedAt: Date;
  status: 'unassigned' | 'assigned' | 'in_progress' | 'pending' | 'resolved';
  assignee?: string;
  assignedAt?: Date;
  firstResponseAt?: Date;
  resolvedAt?: Date;
  priority: 'low' | 'normal' | 'high' | 'urgent';
  tags: string[];
  slaStatus: 'ok' | 'warning' | 'breached';
  viewers: string[]; // userIds currently viewing
  notes: InternalNote[];
  isLocked: boolean;
  lockedBy?: string;
}

export interface InternalNote {
  id: string;
  authorId: string;
  authorName: string;
  content: string;
  createdAt: Date;
  isPrivate: boolean;
}

export interface Assignment {
  emailId: string;
  fromUserId?: string;
  toUserId: string;
  reason?: string;
  assignedAt: Date;
}

// ============================================================================
// Store
// ============================================================================

interface SharedInboxState {
  inboxes: SharedInbox[];
  emails: SharedEmail[];
  activeInboxId: string | null;
  activeEmailId: string | null;
  viewingUsers: Map<string, string[]>; // emailId -> userIds

  // Actions
  createInbox: (inbox: Omit<SharedInbox, 'id' | 'createdAt' | 'stats'>) => string;
  updateInbox: (id: string, updates: Partial<SharedInbox>) => void;
  deleteInbox: (id: string) => void;

  setActiveInbox: (id: string | null) => void;
  setActiveEmail: (id: string | null) => void;

  assignEmail: (emailId: string, userId: string, reason?: string) => void;
  unassignEmail: (emailId: string) => void;
  updateEmailStatus: (emailId: string, status: SharedEmail['status']) => void;
  updateEmailPriority: (emailId: string, priority: SharedEmail['priority']) => void;
  addTag: (emailId: string, tag: string) => void;
  removeTag: (emailId: string, tag: string) => void;

  lockEmail: (emailId: string, userId: string) => void;
  unlockEmail: (emailId: string) => void;

  addNote: (emailId: string, note: Omit<InternalNote, 'id' | 'createdAt'>) => void;

  setViewing: (emailId: string, userId: string, viewing: boolean) => void;
}

export const useSharedInboxStore = create<SharedInboxState>()(
  persist(
    (set, get) => ({
      inboxes: [],
      emails: [],
      activeInboxId: null,
      activeEmailId: null,
      viewingUsers: new Map(),

      createInbox: (inbox) => {
        const id = `inbox_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const newInbox: SharedInbox = {
          ...inbox,
          id,
          createdAt: new Date(),
          stats: {
            unassigned: 0,
            assigned: 0,
            pending: 0,
            resolved: 0,
            avgResponseTime: 0,
            avgResolutionTime: 0,
            slaBreaches: 0,
          },
        };
        set((state) => ({ inboxes: [...state.inboxes, newInbox] }));
        return id;
      },

      updateInbox: (id, updates) => {
        set((state) => ({
          inboxes: state.inboxes.map((i) => (i.id === id ? { ...i, ...updates } : i)),
        }));
      },

      deleteInbox: (id) => {
        set((state) => ({
          inboxes: state.inboxes.filter((i) => i.id !== id),
          activeInboxId: state.activeInboxId === id ? null : state.activeInboxId,
        }));
      },

      setActiveInbox: (id) => set({ activeInboxId: id }),
      setActiveEmail: (id) => set({ activeEmailId: id }),

      assignEmail: (emailId, userId, reason) => {
        set((state) => ({
          emails: state.emails.map((e) =>
            e.id === emailId
              ? {
                  ...e,
                  assignee: userId,
                  assignedAt: new Date(),
                  status: 'assigned' as const,
                }
              : e
          ),
        }));
      },

      unassignEmail: (emailId) => {
        set((state) => ({
          emails: state.emails.map((e) =>
            e.id === emailId
              ? { ...e, assignee: undefined, assignedAt: undefined, status: 'unassigned' as const }
              : e
          ),
        }));
      },

      updateEmailStatus: (emailId, status) => {
        set((state) => ({
          emails: state.emails.map((e) => {
            if (e.id !== emailId) return e;
            const updates: Partial<SharedEmail> = { status };
            if (status === 'resolved') updates.resolvedAt = new Date();
            if (status === 'in_progress' && !e.firstResponseAt) {
              updates.firstResponseAt = new Date();
            }
            return { ...e, ...updates };
          }),
        }));
      },

      updateEmailPriority: (emailId, priority) => {
        set((state) => ({
          emails: state.emails.map((e) => (e.id === emailId ? { ...e, priority } : e)),
        }));
      },

      addTag: (emailId, tag) => {
        set((state) => ({
          emails: state.emails.map((e) =>
            e.id === emailId && !e.tags.includes(tag)
              ? { ...e, tags: [...e.tags, tag] }
              : e
          ),
        }));
      },

      removeTag: (emailId, tag) => {
        set((state) => ({
          emails: state.emails.map((e) =>
            e.id === emailId ? { ...e, tags: e.tags.filter((t) => t !== tag) } : e
          ),
        }));
      },

      lockEmail: (emailId, userId) => {
        set((state) => ({
          emails: state.emails.map((e) =>
            e.id === emailId ? { ...e, isLocked: true, lockedBy: userId } : e
          ),
        }));
      },

      unlockEmail: (emailId) => {
        set((state) => ({
          emails: state.emails.map((e) =>
            e.id === emailId ? { ...e, isLocked: false, lockedBy: undefined } : e
          ),
        }));
      },

      addNote: (emailId, note) => {
        const newNote: InternalNote = {
          ...note,
          id: `note_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          createdAt: new Date(),
        };
        set((state) => ({
          emails: state.emails.map((e) =>
            e.id === emailId ? { ...e, notes: [...e.notes, newNote] } : e
          ),
        }));
      },

      setViewing: (emailId, userId, viewing) => {
        set((state) => {
          const viewingUsers = new Map(state.viewingUsers);
          const current = viewingUsers.get(emailId) || [];
          if (viewing && !current.includes(userId)) {
            viewingUsers.set(emailId, [...current, userId]);
          } else if (!viewing) {
            viewingUsers.set(emailId, current.filter((id) => id !== userId));
          }
          return { viewingUsers };
        });
      },
    }),
    {
      name: 'mycelix-shared-inbox',
      partialize: (state) => ({
        inboxes: state.inboxes,
        emails: state.emails,
        activeInboxId: state.activeInboxId,
      }),
    }
  )
);

// ============================================================================
// Hooks
// ============================================================================

export function useActiveInbox() {
  const { inboxes, activeInboxId } = useSharedInboxStore();
  return useMemo(
    () => inboxes.find((i) => i.id === activeInboxId) || null,
    [inboxes, activeInboxId]
  );
}

export function useInboxEmails(inboxId: string | null, filter?: Partial<SharedEmail>) {
  const { emails } = useSharedInboxStore();
  return useMemo(() => {
    if (!inboxId) return [];
    let filtered = emails.filter((e) => e.inboxId === inboxId);
    if (filter?.status) filtered = filtered.filter((e) => e.status === filter.status);
    if (filter?.assignee) filtered = filtered.filter((e) => e.assignee === filter.assignee);
    if (filter?.priority) filtered = filtered.filter((e) => e.priority === filter.priority);
    return filtered.sort((a, b) => b.receivedAt.getTime() - a.receivedAt.getTime());
  }, [emails, inboxId, filter]);
}

export function useCollisionDetection(emailId: string | null) {
  const { viewingUsers } = useSharedInboxStore();
  const currentUserId = 'self'; // Would come from auth

  return useMemo(() => {
    if (!emailId) return { hasCollision: false, viewers: [] };
    const viewers = viewingUsers.get(emailId) || [];
    const otherViewers = viewers.filter((id) => id !== currentUserId);
    return {
      hasCollision: otherViewers.length > 0,
      viewers: otherViewers,
    };
  }, [viewingUsers, emailId]);
}

// ============================================================================
// Components
// ============================================================================

// Inbox Sidebar
export function InboxSidebar() {
  const { inboxes, activeInboxId, setActiveInbox } = useSharedInboxStore();
  const [showCreate, setShowCreate] = useState(false);

  return (
    <div className="w-64 border-r bg-gray-50 dark:bg-gray-900 flex flex-col">
      <div className="p-4 border-b">
        <h2 className="font-semibold flex items-center gap-2">
          <Inbox className="w-5 h-5" />
          Shared Inboxes
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {inboxes.map((inbox) => (
          <button
            key={inbox.id}
            onClick={() => setActiveInbox(inbox.id)}
            className={`w-full p-3 rounded-lg text-left transition-colors ${
              activeInboxId === inbox.id
                ? 'bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-100'
                : 'hover:bg-gray-100 dark:hover:bg-gray-800'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="font-medium">{inbox.name}</span>
              {inbox.stats.unassigned > 0 && (
                <span className="bg-red-500 text-white px-2 py-0.5 rounded-full text-xs">
                  {inbox.stats.unassigned}
                </span>
              )}
            </div>
            <p className="text-xs text-gray-500 truncate mt-1">{inbox.email}</p>
          </button>
        ))}
      </div>

      <div className="p-4 border-t">
        <button
          onClick={() => setShowCreate(true)}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
        >
          <Inbox className="w-4 h-4" />
          New Inbox
        </button>
      </div>
    </div>
  );
}

// Email Queue
export function EmailQueue() {
  const inbox = useActiveInbox();
  const [filter, setFilter] = useState<'all' | 'unassigned' | 'mine' | 'pending'>('all');
  const { setActiveEmail, activeEmailId } = useSharedInboxStore();

  const filterMap: Record<string, Partial<SharedEmail> | undefined> = {
    all: undefined,
    unassigned: { status: 'unassigned' },
    mine: { assignee: 'self' },
    pending: { status: 'pending' },
  };

  const emails = useInboxEmails(inbox?.id || null, filterMap[filter]);

  if (!inbox) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500">
        <div className="text-center">
          <Inbox className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Select an inbox</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-96 border-r flex flex-col">
      {/* Filters */}
      <div className="p-3 border-b flex items-center gap-2">
        {(['all', 'unassigned', 'mine', 'pending'] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
              filter === f
                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-100'
                : 'hover:bg-gray-100 dark:hover:bg-gray-800'
            }`}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {/* Email List */}
      <div className="flex-1 overflow-y-auto">
        {emails.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            <CheckCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No emails in this queue</p>
          </div>
        ) : (
          emails.map((email) => (
            <EmailQueueItem
              key={email.id}
              email={email}
              isActive={activeEmailId === email.id}
              onClick={() => setActiveEmail(email.id)}
              members={inbox.members}
            />
          ))
        )}
      </div>
    </div>
  );
}

// Email Queue Item
function EmailQueueItem({
  email,
  isActive,
  onClick,
  members,
}: {
  email: SharedEmail;
  isActive: boolean;
  onClick: () => void;
  members: InboxMember[];
}) {
  const assignee = members.find((m) => m.userId === email.assignee);
  const { hasCollision, viewers } = useCollisionDetection(email.id);

  const priorityColors = {
    low: 'text-gray-400',
    normal: 'text-blue-400',
    high: 'text-orange-400',
    urgent: 'text-red-500',
  };

  const slaColors = {
    ok: 'bg-green-100 text-green-700',
    warning: 'bg-yellow-100 text-yellow-700',
    breached: 'bg-red-100 text-red-700',
  };

  return (
    <div
      onClick={onClick}
      className={`p-3 border-b cursor-pointer transition-colors ${
        isActive ? 'bg-blue-50 dark:bg-blue-900/20' : 'hover:bg-gray-50 dark:hover:bg-gray-800'
      }`}
    >
      <div className="flex items-start gap-3">
        <Flag className={`w-4 h-4 mt-1 ${priorityColors[email.priority]}`} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium truncate">{email.from.name}</span>
            {hasCollision && (
              <span className="flex items-center gap-1 text-xs text-orange-500">
                <Eye className="w-3 h-3" />
                {viewers.length}
              </span>
            )}
            {email.isLocked && <Lock className="w-3 h-3 text-red-500" />}
          </div>
          <p className="text-sm truncate">{email.subject}</p>
          <p className="text-xs text-gray-500 truncate mt-1">{email.preview}</p>
          <div className="flex items-center gap-2 mt-2">
            {assignee ? (
              <span className="flex items-center gap-1 text-xs bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded">
                <User className="w-3 h-3" />
                {assignee.name}
              </span>
            ) : (
              <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-0.5 rounded">
                Unassigned
              </span>
            )}
            <span className={`text-xs px-2 py-0.5 rounded ${slaColors[email.slaStatus]}`}>
              SLA {email.slaStatus}
            </span>
          </div>
        </div>
        <span className="text-xs text-gray-400 whitespace-nowrap">
          {formatTime(email.receivedAt)}
        </span>
      </div>
    </div>
  );
}

// Email Detail Panel
export function EmailDetailPanel() {
  const { activeEmailId, emails, assignEmail, updateEmailStatus, lockEmail, unlockEmail, addNote } =
    useSharedInboxStore();
  const inbox = useActiveInbox();
  const email = emails.find((e) => e.id === activeEmailId);
  const { hasCollision, viewers } = useCollisionDetection(activeEmailId);

  const [showAssign, setShowAssign] = useState(false);
  const [noteText, setNoteText] = useState('');

  if (!email || !inbox) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500">
        <div className="text-center">
          <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Select an email to view</p>
        </div>
      </div>
    );
  }

  const assignee = inbox.members.find((m) => m.userId === email.assignee);

  const handleAddNote = () => {
    if (!noteText.trim()) return;
    addNote(email.id, {
      authorId: 'self',
      authorName: 'You',
      content: noteText.trim(),
      isPrivate: false,
    });
    setNoteText('');
  };

  return (
    <div className="flex-1 flex flex-col">
      {/* Collision Warning */}
      {hasCollision && (
        <div className="px-4 py-2 bg-orange-100 dark:bg-orange-900/20 flex items-center gap-2 text-orange-700 dark:text-orange-300">
          <AlertTriangle className="w-4 h-4" />
          <span className="text-sm">
            {viewers.length} other {viewers.length === 1 ? 'person is' : 'people are'} viewing this
            email
          </span>
        </div>
      )}

      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold">{email.subject}</h2>
            <p className="text-sm text-gray-500">
              From: {email.from.name} &lt;{email.from.email}&gt;
            </p>
          </div>
          <div className="flex items-center gap-2">
            {email.isLocked ? (
              <button
                onClick={() => unlockEmail(email.id)}
                className="p-2 text-red-500 hover:bg-red-50 rounded-lg"
                title="Unlock email"
              >
                <Lock className="w-5 h-5" />
              </button>
            ) : (
              <button
                onClick={() => lockEmail(email.id, 'self')}
                className="p-2 text-gray-500 hover:bg-gray-100 rounded-lg"
                title="Lock email"
              >
                <Unlock className="w-5 h-5" />
              </button>
            )}
            <button className="p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
              <MoreHorizontal className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Assignment & Status */}
        <div className="flex items-center gap-4 mt-4">
          <div className="relative">
            <button
              onClick={() => setShowAssign(!showAssign)}
              className="flex items-center gap-2 px-3 py-1.5 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800"
            >
              <User className="w-4 h-4" />
              {assignee ? assignee.name : 'Unassigned'}
            </button>
            {showAssign && (
              <AssignmentDropdown
                members={inbox.members}
                currentAssignee={email.assignee}
                onAssign={(userId) => {
                  assignEmail(email.id, userId);
                  setShowAssign(false);
                }}
                onClose={() => setShowAssign(false)}
              />
            )}
          </div>

          <select
            value={email.status}
            onChange={(e) => updateEmailStatus(email.id, e.target.value as SharedEmail['status'])}
            className="px-3 py-1.5 border rounded-lg bg-transparent"
          >
            <option value="unassigned">Unassigned</option>
            <option value="assigned">Assigned</option>
            <option value="in_progress">In Progress</option>
            <option value="pending">Pending</option>
            <option value="resolved">Resolved</option>
          </select>

          <div className="flex items-center gap-1">
            {email.tags.map((tag) => (
              <span key={tag} className="px-2 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-xs">
                {tag}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Email Content */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="prose dark:prose-invert max-w-none">
          <p>{email.preview}</p>
          {/* Full email content would go here */}
        </div>
      </div>

      {/* Internal Notes */}
      <div className="border-t">
        <div className="p-3 border-b bg-gray-50 dark:bg-gray-800">
          <h3 className="font-medium flex items-center gap-2">
            <MessageSquare className="w-4 h-4" />
            Internal Notes ({email.notes.length})
          </h3>
        </div>
        <div className="max-h-48 overflow-y-auto">
          {email.notes.map((note) => (
            <div key={note.id} className="p-3 border-b">
              <div className="flex items-center justify-between">
                <span className="font-medium text-sm">{note.authorName}</span>
                <span className="text-xs text-gray-500">{formatTime(note.createdAt)}</span>
              </div>
              <p className="text-sm mt-1">{note.content}</p>
            </div>
          ))}
        </div>
        <div className="p-3 flex gap-2">
          <input
            type="text"
            value={noteText}
            onChange={(e) => setNoteText(e.target.value)}
            placeholder="Add internal note..."
            className="flex-1 px-3 py-2 border rounded-lg dark:bg-gray-700 dark:border-gray-600"
            onKeyDown={(e) => e.key === 'Enter' && handleAddNote()}
          />
          <button
            onClick={handleAddNote}
            disabled={!noteText.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
          >
            Add
          </button>
        </div>
      </div>

      {/* Actions */}
      <div className="p-4 border-t flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600">
            <Send className="w-4 h-4" />
            Reply
          </button>
          <button className="p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
            <Archive className="w-5 h-5" />
          </button>
          <button className="p-2 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
            <Trash2 className="w-5 h-5" />
          </button>
        </div>
        <button
          onClick={() => updateEmailStatus(email.id, 'resolved')}
          className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600"
        >
          <CheckCircle className="w-4 h-4" />
          Mark Resolved
        </button>
      </div>
    </div>
  );
}

// Assignment Dropdown
function AssignmentDropdown({
  members,
  currentAssignee,
  onAssign,
  onClose,
}: {
  members: InboxMember[];
  currentAssignee?: string;
  onAssign: (userId: string) => void;
  onClose: () => void;
}) {
  return (
    <>
      <div className="fixed inset-0 z-10" onClick={onClose} />
      <div className="absolute top-full left-0 mt-1 w-64 bg-white dark:bg-gray-800 border rounded-lg shadow-lg z-20">
        <div className="p-2">
          {members.map((member) => (
            <button
              key={member.userId}
              onClick={() => onAssign(member.userId)}
              className={`w-full flex items-center gap-3 p-2 rounded-lg text-left hover:bg-gray-50 dark:hover:bg-gray-700 ${
                currentAssignee === member.userId ? 'bg-blue-50 dark:bg-blue-900/20' : ''
              }`}
            >
              <div className="w-8 h-8 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center text-white text-sm">
                {member.name.charAt(0)}
              </div>
              <div className="flex-1">
                <p className="font-medium text-sm">{member.name}</p>
                <p className="text-xs text-gray-500">{member.assignedCount} assigned</p>
              </div>
              {member.isOnline && <span className="w-2 h-2 bg-green-500 rounded-full" />}
            </button>
          ))}
        </div>
      </div>
    </>
  );
}

// Stats Dashboard
export function InboxStatsDashboard() {
  const inbox = useActiveInbox();

  if (!inbox) return null;

  const stats = [
    { label: 'Unassigned', value: inbox.stats.unassigned, icon: Inbox, color: 'text-yellow-500' },
    { label: 'In Progress', value: inbox.stats.assigned, icon: Clock, color: 'text-blue-500' },
    { label: 'Pending', value: inbox.stats.pending, icon: AlertTriangle, color: 'text-orange-500' },
    { label: 'Resolved', value: inbox.stats.resolved, icon: CheckCircle, color: 'text-green-500' },
  ];

  return (
    <div className="p-4 border-b bg-gray-50 dark:bg-gray-800">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">{inbox.name} Overview</h3>
        <button className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded">
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>
      <div className="grid grid-cols-4 gap-4">
        {stats.map((stat) => (
          <div key={stat.label} className="p-3 bg-white dark:bg-gray-900 rounded-lg">
            <div className="flex items-center gap-2">
              <stat.icon className={`w-4 h-4 ${stat.color}`} />
              <span className="text-sm text-gray-500">{stat.label}</span>
            </div>
            <p className="text-2xl font-bold mt-1">{stat.value}</p>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-4 mt-4">
        <div className="p-3 bg-white dark:bg-gray-900 rounded-lg">
          <p className="text-sm text-gray-500">Avg Response Time</p>
          <p className="text-lg font-semibold">{inbox.stats.avgResponseTime} min</p>
        </div>
        <div className="p-3 bg-white dark:bg-gray-900 rounded-lg">
          <p className="text-sm text-gray-500">SLA Breaches</p>
          <p className="text-lg font-semibold text-red-500">{inbox.stats.slaBreaches}</p>
        </div>
      </div>
    </div>
  );
}

// Main Shared Inbox Page
export function SharedInboxPage() {
  return (
    <div className="flex h-full">
      <InboxSidebar />
      <div className="flex-1 flex flex-col">
        <InboxStatsDashboard />
        <div className="flex-1 flex">
          <EmailQueue />
          <EmailDetailPanel />
        </div>
      </div>
    </div>
  );
}

// Utility function
function formatTime(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - new Date(date).getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return new Date(date).toLocaleDateString();
}

export default SharedInboxPage;
