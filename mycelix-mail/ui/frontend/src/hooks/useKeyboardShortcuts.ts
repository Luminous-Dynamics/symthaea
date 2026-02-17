import { useEffect } from 'react';

export interface KeyboardShortcut {
  key: string;
  description: string;
  action: () => void;
  modifiers?: {
    ctrl?: boolean;
    alt?: boolean;
    shift?: boolean;
    meta?: boolean;
  };
}

/**
 * Custom hook for registering keyboard shortcuts
 * @param shortcuts - Array of keyboard shortcuts to register
 * @param enabled - Whether shortcuts are enabled (default: true)
 */
export function useKeyboardShortcuts(
  shortcuts: KeyboardShortcut[],
  enabled: boolean = true
) {
  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in input fields
      const target = event.target as HTMLElement;
      const isInputField =
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable;

      // Allow '/' to work even in input fields (for search focus)
      if (isInputField && event.key !== '/') {
        return;
      }

      for (const shortcut of shortcuts) {
        const modifiersMatch =
          (!shortcut.modifiers?.ctrl || event.ctrlKey) &&
          (!shortcut.modifiers?.alt || event.altKey) &&
          (!shortcut.modifiers?.shift || event.shiftKey) &&
          (!shortcut.modifiers?.meta || event.metaKey);

        if (event.key === shortcut.key && modifiersMatch) {
          event.preventDefault();
          shortcut.action();
          break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [shortcuts, enabled]);
}

/**
 * Common keyboard shortcuts for the email client
 */
export const EMAIL_SHORTCUTS = {
  COMPOSE: { key: 'c', description: 'Compose new email' },
  REPLY: { key: 'r', description: 'Reply to email' },
  REPLY_ALL: { key: 'a', description: 'Reply all' },
  FORWARD: { key: 'f', description: 'Forward email' },
  NEXT_EMAIL: { key: 'j', description: 'Next email' },
  PREV_EMAIL: { key: 'k', description: 'Previous email' },
  SEARCH: { key: '/', description: 'Focus search' },
  ESCAPE: { key: 'Escape', description: 'Close modal/Clear selection' },
  DELETE: { key: 'Delete', description: 'Delete email' },
  STAR: { key: 's', description: 'Star/Unstar email' },
  MARK_READ: { key: 'u', description: 'Mark as read/unread' },
  HELP: { key: '?', description: 'Show keyboard shortcuts' },
} as const;
