// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Presence System
 *
 * Real-time user presence:
 * - Live cursors
 * - Selection awareness
 * - User status (online, away, busy)
 * - Typing indicators
 * - Activity tracking
 */

import { LWWMap, ORSet, VectorClockImpl } from './crdt';

// ==================== Types ====================

export interface CursorPosition {
  x: number;
  y: number;
  target?: string; // Element ID or path
  viewport?: { x: number; y: number; width: number; height: number };
}

export interface Selection {
  type: 'range' | 'region' | 'element';
  start: number;
  end: number;
  path?: string[];
  color?: string;
}

export type UserStatus = 'online' | 'away' | 'busy' | 'offline';

export interface UserPresence {
  id: string;
  name: string;
  avatar?: string;
  color: string;
  status: UserStatus;
  cursor?: CursorPosition;
  selection?: Selection;
  lastActive: number;
  isTyping?: boolean;
  currentView?: string;
  permissions: ('view' | 'edit' | 'comment' | 'admin')[];
}

export interface PresenceUpdate {
  type: 'cursor' | 'selection' | 'status' | 'typing' | 'view' | 'full';
  userId: string;
  data: Partial<UserPresence>;
  timestamp: number;
}

// ==================== Color Generation ====================

const PRESENCE_COLORS = [
  '#EF4444', // Red
  '#F97316', // Orange
  '#F59E0B', // Amber
  '#84CC16', // Lime
  '#22C55E', // Green
  '#14B8A6', // Teal
  '#06B6D4', // Cyan
  '#3B82F6', // Blue
  '#6366F1', // Indigo
  '#8B5CF6', // Violet
  '#A855F7', // Purple
  '#EC4899', // Pink
];

function generateUserColor(userId: string): string {
  let hash = 0;
  for (let i = 0; i < userId.length; i++) {
    hash = ((hash << 5) - hash) + userId.charCodeAt(i);
    hash = hash & hash;
  }
  return PRESENCE_COLORS[Math.abs(hash) % PRESENCE_COLORS.length];
}

// ==================== Presence Manager ====================

export class PresenceManager {
  private localUser: UserPresence;
  private users: Map<string, UserPresence> = new Map();
  private updateInterval: NodeJS.Timeout | null = null;
  private awayTimeout: NodeJS.Timeout | null = null;
  private onUpdate?: (users: UserPresence[]) => void;
  private onBroadcast?: (update: PresenceUpdate) => void;

  constructor(
    userId: string,
    userName: string,
    callbacks: {
      onUpdate?: (users: UserPresence[]) => void;
      onBroadcast?: (update: PresenceUpdate) => void;
    } = {}
  ) {
    this.onUpdate = callbacks.onUpdate;
    this.onBroadcast = callbacks.onBroadcast;

    this.localUser = {
      id: userId,
      name: userName,
      color: generateUserColor(userId),
      status: 'online',
      lastActive: Date.now(),
      permissions: ['view', 'edit'],
    };

    this.users.set(userId, this.localUser);
    this.startActivityTracking();
  }

  private startActivityTracking(): void {
    // Track mouse/keyboard activity
    if (typeof window !== 'undefined') {
      const resetAway = () => {
        if (this.localUser.status === 'away') {
          this.setStatus('online');
        }
        this.localUser.lastActive = Date.now();

        // Reset away timeout
        if (this.awayTimeout) {
          clearTimeout(this.awayTimeout);
        }
        this.awayTimeout = setTimeout(() => {
          this.setStatus('away');
        }, 5 * 60 * 1000); // 5 minutes
      };

      window.addEventListener('mousemove', resetAway);
      window.addEventListener('keydown', resetAway);
      window.addEventListener('click', resetAway);
      window.addEventListener('scroll', resetAway);
    }

    // Periodic heartbeat
    this.updateInterval = setInterval(() => {
      this.broadcastUpdate('full', this.localUser);
      this.cleanupStaleUsers();
    }, 30000); // Every 30 seconds
  }

  private cleanupStaleUsers(): void {
    const now = Date.now();
    const staleThreshold = 2 * 60 * 1000; // 2 minutes

    for (const [userId, user] of this.users) {
      if (userId !== this.localUser.id && now - user.lastActive > staleThreshold) {
        user.status = 'offline';
      }
    }

    this.notifyUpdate();
  }

  private broadcastUpdate(type: PresenceUpdate['type'], data: Partial<UserPresence>): void {
    const update: PresenceUpdate = {
      type,
      userId: this.localUser.id,
      data,
      timestamp: Date.now(),
    };

    this.onBroadcast?.(update);
  }

  private notifyUpdate(): void {
    this.onUpdate?.(Array.from(this.users.values()));
  }

  // ==================== Local User Updates ====================

  setCursor(position: CursorPosition): void {
    this.localUser.cursor = position;
    this.localUser.lastActive = Date.now();
    this.broadcastUpdate('cursor', { cursor: position });
    this.notifyUpdate();
  }

  setSelection(selection: Selection | undefined): void {
    this.localUser.selection = selection;
    this.localUser.lastActive = Date.now();
    this.broadcastUpdate('selection', { selection });
    this.notifyUpdate();
  }

  setStatus(status: UserStatus): void {
    this.localUser.status = status;
    this.localUser.lastActive = Date.now();
    this.broadcastUpdate('status', { status });
    this.notifyUpdate();
  }

  setTyping(isTyping: boolean): void {
    this.localUser.isTyping = isTyping;
    this.localUser.lastActive = Date.now();
    this.broadcastUpdate('typing', { isTyping });
    this.notifyUpdate();
  }

  setCurrentView(view: string): void {
    this.localUser.currentView = view;
    this.localUser.lastActive = Date.now();
    this.broadcastUpdate('view', { currentView: view });
    this.notifyUpdate();
  }

  updateProfile(updates: { name?: string; avatar?: string }): void {
    if (updates.name) this.localUser.name = updates.name;
    if (updates.avatar) this.localUser.avatar = updates.avatar;
    this.broadcastUpdate('full', updates);
    this.notifyUpdate();
  }

  // ==================== Remote User Updates ====================

  handleRemoteUpdate(update: PresenceUpdate): void {
    if (update.userId === this.localUser.id) return;

    let user = this.users.get(update.userId);
    if (!user) {
      user = {
        id: update.userId,
        name: update.data.name || `User ${update.userId.slice(0, 4)}`,
        color: generateUserColor(update.userId),
        status: 'online',
        lastActive: update.timestamp,
        permissions: ['view'],
      };
      this.users.set(update.userId, user);
    }

    // Apply updates
    Object.assign(user, update.data);
    user.lastActive = update.timestamp;

    // Ensure status is online if we're receiving updates
    if (user.status === 'offline') {
      user.status = 'online';
    }

    this.notifyUpdate();
  }

  removeUser(userId: string): void {
    this.users.delete(userId);
    this.notifyUpdate();
  }

  // ==================== Queries ====================

  getLocalUser(): UserPresence {
    return this.localUser;
  }

  getUser(userId: string): UserPresence | undefined {
    return this.users.get(userId);
  }

  getAllUsers(): UserPresence[] {
    return Array.from(this.users.values());
  }

  getOnlineUsers(): UserPresence[] {
    return Array.from(this.users.values()).filter(u => u.status !== 'offline');
  }

  getUsersInView(view: string): UserPresence[] {
    return Array.from(this.users.values()).filter(u => u.currentView === view);
  }

  getUsersWithSelection(): UserPresence[] {
    return Array.from(this.users.values()).filter(u => u.selection);
  }

  // ==================== Cleanup ====================

  dispose(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
    if (this.awayTimeout) {
      clearTimeout(this.awayTimeout);
    }
    this.users.clear();
  }
}

// ==================== Cursor Renderer ====================

export interface CursorStyle {
  color: string;
  size: number;
  labelOffset: { x: number; y: number };
  showLabel: boolean;
  fadeDelay: number;
}

export class CursorRenderer {
  private container: HTMLElement;
  private cursors: Map<string, HTMLElement> = new Map();
  private defaultStyle: CursorStyle = {
    color: '#8B5CF6',
    size: 20,
    labelOffset: { x: 15, y: -5 },
    showLabel: true,
    fadeDelay: 3000,
  };

  constructor(container: HTMLElement) {
    this.container = container;
    this.setupStyles();
  }

  private setupStyles(): void {
    const style = document.createElement('style');
    style.textContent = `
      .remote-cursor {
        position: absolute;
        pointer-events: none;
        z-index: 9999;
        transition: transform 0.1s ease-out, opacity 0.3s ease;
      }
      .remote-cursor-pointer {
        width: 0;
        height: 0;
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-bottom: 12px solid var(--cursor-color);
        transform: rotate(-45deg);
      }
      .remote-cursor-label {
        position: absolute;
        left: 15px;
        top: -5px;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        white-space: nowrap;
        background: var(--cursor-color);
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
      }
      .remote-cursor.faded {
        opacity: 0.3;
      }
    `;
    document.head.appendChild(style);
  }

  updateCursor(user: UserPresence): void {
    if (!user.cursor) {
      this.removeCursor(user.id);
      return;
    }

    let element = this.cursors.get(user.id);

    if (!element) {
      element = document.createElement('div');
      element.className = 'remote-cursor';
      element.innerHTML = `
        <div class="remote-cursor-pointer"></div>
        <div class="remote-cursor-label">${user.name}</div>
      `;
      element.style.setProperty('--cursor-color', user.color);
      this.container.appendChild(element);
      this.cursors.set(user.id, element);
    }

    element.style.transform = `translate(${user.cursor.x}px, ${user.cursor.y}px)`;
    element.classList.remove('faded');

    // Fade after inactivity
    const existingTimeout = (element as any)._fadeTimeout;
    if (existingTimeout) {
      clearTimeout(existingTimeout);
    }
    (element as any)._fadeTimeout = setTimeout(() => {
      element!.classList.add('faded');
    }, this.defaultStyle.fadeDelay);
  }

  removeCursor(userId: string): void {
    const element = this.cursors.get(userId);
    if (element) {
      element.remove();
      this.cursors.delete(userId);
    }
  }

  updateAllCursors(users: UserPresence[]): void {
    const userIds = new Set(users.map(u => u.id));

    // Remove cursors for users no longer present
    for (const userId of this.cursors.keys()) {
      if (!userIds.has(userId)) {
        this.removeCursor(userId);
      }
    }

    // Update/add cursors
    for (const user of users) {
      this.updateCursor(user);
    }
  }

  dispose(): void {
    for (const element of this.cursors.values()) {
      element.remove();
    }
    this.cursors.clear();
  }
}

// ==================== Selection Renderer ====================

export class SelectionRenderer {
  private container: HTMLElement;
  private selections: Map<string, HTMLElement[]> = new Map();

  constructor(container: HTMLElement) {
    this.container = container;
    this.setupStyles();
  }

  private setupStyles(): void {
    const style = document.createElement('style');
    style.textContent = `
      .remote-selection {
        position: absolute;
        pointer-events: none;
        z-index: 9998;
        background: var(--selection-color);
        opacity: 0.3;
        border-radius: 2px;
      }
      .remote-selection-border {
        position: absolute;
        pointer-events: none;
        z-index: 9998;
        border: 2px solid var(--selection-color);
        border-radius: 2px;
      }
    `;
    document.head.appendChild(style);
  }

  updateSelection(user: UserPresence, getRect: (selection: Selection) => DOMRect[]): void {
    // Clear existing selections
    this.clearSelection(user.id);

    if (!user.selection) return;

    const rects = getRect(user.selection);
    const elements: HTMLElement[] = [];

    for (const rect of rects) {
      const element = document.createElement('div');
      element.className = 'remote-selection';
      element.style.setProperty('--selection-color', user.color);
      element.style.left = `${rect.left}px`;
      element.style.top = `${rect.top}px`;
      element.style.width = `${rect.width}px`;
      element.style.height = `${rect.height}px`;
      this.container.appendChild(element);
      elements.push(element);
    }

    this.selections.set(user.id, elements);
  }

  clearSelection(userId: string): void {
    const elements = this.selections.get(userId);
    if (elements) {
      for (const element of elements) {
        element.remove();
      }
      this.selections.delete(userId);
    }
  }

  dispose(): void {
    for (const elements of this.selections.values()) {
      for (const element of elements) {
        element.remove();
      }
    }
    this.selections.clear();
  }
}

// ==================== Typing Indicator ====================

export class TypingIndicator {
  private container: HTMLElement;
  private typingUsers: Set<string> = new Set();
  private element: HTMLElement | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  update(users: UserPresence[]): void {
    const typing = users.filter(u => u.isTyping);

    if (typing.length === 0) {
      this.hide();
      return;
    }

    if (!this.element) {
      this.element = document.createElement('div');
      this.element.className = 'typing-indicator';
      this.element.innerHTML = `
        <div class="typing-dots">
          <span></span><span></span><span></span>
        </div>
        <span class="typing-text"></span>
      `;
      this.container.appendChild(this.element);
    }

    const names = typing.map(u => u.name);
    let text: string;

    if (names.length === 1) {
      text = `${names[0]} is typing...`;
    } else if (names.length === 2) {
      text = `${names[0]} and ${names[1]} are typing...`;
    } else {
      text = `${names.length} people are typing...`;
    }

    this.element.querySelector('.typing-text')!.textContent = text;
    this.element.style.display = 'flex';
  }

  hide(): void {
    if (this.element) {
      this.element.style.display = 'none';
    }
  }

  dispose(): void {
    this.element?.remove();
  }
}

export default {
  PresenceManager,
  CursorRenderer,
  SelectionRenderer,
  TypingIndicator,
  generateUserColor,
};
