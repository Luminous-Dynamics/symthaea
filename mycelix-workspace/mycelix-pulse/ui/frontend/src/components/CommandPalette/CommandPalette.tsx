// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Command Palette
 *
 * Power user command interface (Cmd+K) with keyboard shortcuts
 */

import React, { useEffect, useState, useRef, useCallback } from 'react';

interface Command {
  id: string;
  name: string;
  description?: string;
  shortcut?: string;
  category: 'navigation' | 'email' | 'compose' | 'search' | 'settings' | 'help';
  action: () => void;
  keywords?: string[];
}

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigate: (path: string) => void;
  onAction: (action: string, payload?: any) => void;
}

export default function CommandPalette({
  isOpen,
  onClose,
  onNavigate,
  onAction,
}: CommandPaletteProps) {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const commands: Command[] = [
    // Navigation
    { id: 'goto-inbox', name: 'Go to Inbox', shortcut: 'g i', category: 'navigation', action: () => onNavigate('/inbox'), keywords: ['mail', 'home'] },
    { id: 'goto-sent', name: 'Go to Sent', shortcut: 'g s', category: 'navigation', action: () => onNavigate('/sent') },
    { id: 'goto-drafts', name: 'Go to Drafts', shortcut: 'g d', category: 'navigation', action: () => onNavigate('/drafts') },
    { id: 'goto-archive', name: 'Go to Archive', shortcut: 'g a', category: 'navigation', action: () => onNavigate('/archive') },
    { id: 'goto-trash', name: 'Go to Trash', shortcut: 'g t', category: 'navigation', action: () => onNavigate('/trash') },
    { id: 'goto-contacts', name: 'Go to Contacts', shortcut: 'g c', category: 'navigation', action: () => onNavigate('/contacts') },
    { id: 'goto-calendar', name: 'Go to Calendar', shortcut: 'g l', category: 'navigation', action: () => onNavigate('/calendar') },
    { id: 'goto-settings', name: 'Go to Settings', shortcut: 'g ,', category: 'navigation', action: () => onNavigate('/settings') },
    { id: 'goto-analytics', name: 'Go to Analytics', category: 'navigation', action: () => onNavigate('/analytics') },

    // Compose
    { id: 'compose-new', name: 'Compose New Email', shortcut: 'c', category: 'compose', action: () => onAction('compose'), keywords: ['new', 'write', 'create'] },
    { id: 'compose-reply', name: 'Reply to Email', shortcut: 'r', category: 'compose', action: () => onAction('reply') },
    { id: 'compose-reply-all', name: 'Reply All', shortcut: 'a', category: 'compose', action: () => onAction('replyAll') },
    { id: 'compose-forward', name: 'Forward Email', shortcut: 'f', category: 'compose', action: () => onAction('forward') },

    // Email Actions
    { id: 'email-archive', name: 'Archive Email', shortcut: 'e', category: 'email', action: () => onAction('archive'), keywords: ['hide', 'done'] },
    { id: 'email-delete', name: 'Delete Email', shortcut: '#', category: 'email', action: () => onAction('delete'), keywords: ['remove', 'trash'] },
    { id: 'email-star', name: 'Star/Unstar Email', shortcut: 's', category: 'email', action: () => onAction('star'), keywords: ['favorite', 'important'] },
    { id: 'email-read', name: 'Mark as Read', shortcut: 'Shift+i', category: 'email', action: () => onAction('markRead') },
    { id: 'email-unread', name: 'Mark as Unread', shortcut: 'Shift+u', category: 'email', action: () => onAction('markUnread') },
    { id: 'email-spam', name: 'Report as Spam', shortcut: '!', category: 'email', action: () => onAction('spam') },
    { id: 'email-snooze', name: 'Snooze Email', shortcut: 'b', category: 'email', action: () => onAction('snooze') },
    { id: 'email-move', name: 'Move to Folder', shortcut: 'v', category: 'email', action: () => onAction('move') },
    { id: 'email-label', name: 'Add Label', shortcut: 'l', category: 'email', action: () => onAction('label') },

    // Search
    { id: 'search-focus', name: 'Focus Search', shortcut: '/', category: 'search', action: () => onAction('focusSearch') },
    { id: 'search-advanced', name: 'Advanced Search', category: 'search', action: () => onAction('advancedSearch') },
    { id: 'search-unread', name: 'Show Unread Only', category: 'search', action: () => onAction('filterUnread') },
    { id: 'search-starred', name: 'Show Starred Only', category: 'search', action: () => onAction('filterStarred') },
    { id: 'search-attachments', name: 'Show with Attachments', category: 'search', action: () => onAction('filterAttachments') },

    // Settings
    { id: 'settings-account', name: 'Account Settings', category: 'settings', action: () => onNavigate('/settings/account') },
    { id: 'settings-notifications', name: 'Notification Settings', category: 'settings', action: () => onNavigate('/settings/notifications') },
    { id: 'settings-shortcuts', name: 'Keyboard Shortcuts', shortcut: '?', category: 'settings', action: () => onAction('showShortcuts') },
    { id: 'settings-theme', name: 'Toggle Dark Mode', category: 'settings', action: () => onAction('toggleTheme'), keywords: ['dark', 'light', 'appearance'] },

    // Help
    { id: 'help-docs', name: 'View Documentation', category: 'help', action: () => window.open('/docs', '_blank') },
    { id: 'help-feedback', name: 'Send Feedback', category: 'help', action: () => onAction('feedback') },
    { id: 'help-whats-new', name: "What's New", category: 'help', action: () => onAction('changelog') },
  ];

  const filteredCommands = commands.filter((cmd) => {
    if (!query) return true;
    const lowerQuery = query.toLowerCase();
    return (
      cmd.name.toLowerCase().includes(lowerQuery) ||
      cmd.description?.toLowerCase().includes(lowerQuery) ||
      cmd.keywords?.some((k) => k.toLowerCase().includes(lowerQuery)) ||
      cmd.category.toLowerCase().includes(lowerQuery)
    );
  });

  const groupedCommands = filteredCommands.reduce((acc, cmd) => {
    if (!acc[cmd.category]) acc[cmd.category] = [];
    acc[cmd.category].push(cmd);
    return acc;
  }, {} as Record<string, Command[]>);

  const categoryOrder = ['navigation', 'compose', 'email', 'search', 'settings', 'help'];
  const categoryLabels: Record<string, string> = {
    navigation: 'Navigation',
    compose: 'Compose',
    email: 'Email Actions',
    search: 'Search',
    settings: 'Settings',
    help: 'Help',
  };

  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
      inputRef.current?.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex((i) => Math.min(i + 1, filteredCommands.length - 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex((i) => Math.max(i - 1, 0));
          break;
        case 'Enter':
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            filteredCommands[selectedIndex].action();
            onClose();
          }
          break;
        case 'Escape':
          e.preventDefault();
          onClose();
          break;
      }
    },
    [filteredCommands, selectedIndex, onClose]
  );

  // Scroll selected item into view
  useEffect(() => {
    const selectedEl = listRef.current?.querySelector(`[data-index="${selectedIndex}"]`);
    selectedEl?.scrollIntoView({ block: 'nearest' });
  }, [selectedIndex]);

  if (!isOpen) return null;

  let commandIndex = 0;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />

      {/* Palette */}
      <div className="relative w-full max-w-xl bg-background rounded-lg shadow-2xl border border-border overflow-hidden">
        {/* Search Input */}
        <div className="flex items-center border-b border-border px-4">
          <span className="text-muted mr-2">{'>'}</span>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a command or search..."
            className="flex-1 py-4 bg-transparent outline-none text-lg"
            autoComplete="off"
            autoCorrect="off"
            spellCheck={false}
          />
          <kbd className="px-2 py-1 bg-muted/30 rounded text-xs text-muted">ESC</kbd>
        </div>

        {/* Command List */}
        <div ref={listRef} className="max-h-[400px] overflow-y-auto">
          {filteredCommands.length === 0 ? (
            <div className="p-8 text-center text-muted">No commands found</div>
          ) : (
            categoryOrder.map((category) => {
              const cmds = groupedCommands[category];
              if (!cmds?.length) return null;

              return (
                <div key={category}>
                  <div className="px-4 py-2 text-xs font-semibold text-muted uppercase bg-muted/10">
                    {categoryLabels[category]}
                  </div>
                  {cmds.map((cmd) => {
                    const idx = commandIndex++;
                    const isSelected = idx === selectedIndex;

                    return (
                      <div
                        key={cmd.id}
                        data-index={idx}
                        className={`flex items-center justify-between px-4 py-3 cursor-pointer ${
                          isSelected ? 'bg-primary/10' : 'hover:bg-muted/30'
                        }`}
                        onClick={() => {
                          cmd.action();
                          onClose();
                        }}
                        onMouseEnter={() => setSelectedIndex(idx)}
                      >
                        <div>
                          <p className="font-medium">{cmd.name}</p>
                          {cmd.description && (
                            <p className="text-sm text-muted">{cmd.description}</p>
                          )}
                        </div>
                        {cmd.shortcut && (
                          <kbd className="px-2 py-1 bg-muted/30 rounded text-xs font-mono">
                            {cmd.shortcut}
                          </kbd>
                        )}
                      </div>
                    );
                  })}
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-border px-4 py-2 flex items-center justify-between text-xs text-muted">
          <div className="flex items-center gap-4">
            <span>
              <kbd className="px-1 bg-muted/30 rounded">Up/Down</kbd> to navigate
            </span>
            <span>
              <kbd className="px-1 bg-muted/30 rounded">Enter</kbd> to select
            </span>
          </div>
          <span>{filteredCommands.length} commands</span>
        </div>
      </div>
    </div>
  );
}

/**
 * Keyboard Shortcuts Hook
 *
 * Global keyboard shortcut handler with Vim-style navigation
 */
export function useKeyboardShortcuts(
  onAction: (action: string, payload?: any) => void,
  onNavigate: (path: string) => void
) {
  const [pendingKey, setPendingKey] = useState<string | null>(null);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);

  useEffect(() => {
    let timeout: NodeJS.Timeout;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
        return;
      }

      // Command palette
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setCommandPaletteOpen(true);
        return;
      }

      // Vim-style navigation (g + key)
      if (pendingKey === 'g') {
        setPendingKey(null);
        clearTimeout(timeout);

        switch (e.key) {
          case 'i': onNavigate('/inbox'); break;
          case 's': onNavigate('/sent'); break;
          case 'd': onNavigate('/drafts'); break;
          case 'a': onNavigate('/archive'); break;
          case 't': onNavigate('/trash'); break;
          case 'c': onNavigate('/contacts'); break;
          case 'l': onNavigate('/calendar'); break;
          case ',': onNavigate('/settings'); break;
        }
        return;
      }

      // Start pending key sequence
      if (e.key === 'g') {
        setPendingKey('g');
        timeout = setTimeout(() => setPendingKey(null), 1000);
        return;
      }

      // Single key shortcuts
      switch (e.key) {
        case 'c': onAction('compose'); break;
        case 'r': onAction('reply'); break;
        case 'a': onAction('replyAll'); break;
        case 'f': onAction('forward'); break;
        case 'e': onAction('archive'); break;
        case '#': onAction('delete'); break;
        case 's': onAction('star'); break;
        case '/': e.preventDefault(); onAction('focusSearch'); break;
        case '?': onAction('showShortcuts'); break;
        case 'j': onAction('nextEmail'); break;
        case 'k': onAction('prevEmail'); break;
        case 'o':
        case 'Enter': onAction('openEmail'); break;
        case 'u': onAction('goBack'); break;
        case 'x': onAction('selectEmail'); break;
        case 'I': if (e.shiftKey) onAction('markRead'); break;
        case 'U': if (e.shiftKey) onAction('markUnread'); break;
        case '!': onAction('spam'); break;
        case 'b': onAction('snooze'); break;
        case 'v': onAction('move'); break;
        case 'l': onAction('label'); break;
        case 'z': onAction('undo'); break;
        case '.': onAction('moreActions'); break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      clearTimeout(timeout);
    };
  }, [pendingKey, onAction, onNavigate]);

  return {
    commandPaletteOpen,
    setCommandPaletteOpen,
    pendingKey,
  };
}

/**
 * Undo Send Manager
 *
 * Delayed email sending with cancellation window
 */
export class UndoSendManager {
  private pendingEmails: Map<string, { email: any; timeout: NodeJS.Timeout; onSend: () => void }> = new Map();
  private defaultDelay = 30000; // 30 seconds

  queue(
    emailId: string,
    email: any,
    onSend: () => Promise<void>,
    onCancel: () => void,
    delay?: number
  ): void {
    const timeout = setTimeout(async () => {
      await onSend();
      this.pendingEmails.delete(emailId);
    }, delay || this.defaultDelay);

    this.pendingEmails.set(emailId, {
      email,
      timeout,
      onSend: async () => {
        await onSend();
        this.pendingEmails.delete(emailId);
      },
    });
  }

  cancel(emailId: string): boolean {
    const pending = this.pendingEmails.get(emailId);
    if (pending) {
      clearTimeout(pending.timeout);
      this.pendingEmails.delete(emailId);
      return true;
    }
    return false;
  }

  sendNow(emailId: string): void {
    const pending = this.pendingEmails.get(emailId);
    if (pending) {
      clearTimeout(pending.timeout);
      pending.onSend();
    }
  }

  getPending(): string[] {
    return Array.from(this.pendingEmails.keys());
  }

  getTimeRemaining(emailId: string): number {
    // Would need to track start time to calculate this
    return 0;
  }

  setDefaultDelay(ms: number): void {
    this.defaultDelay = ms;
  }
}

export const undoSendManager = new UndoSendManager();
