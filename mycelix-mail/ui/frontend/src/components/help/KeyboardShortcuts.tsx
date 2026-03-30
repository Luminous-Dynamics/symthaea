// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Keyboard Shortcuts Help Modal
 *
 * Displays all available keyboard shortcuts organized by category.
 * Can be triggered with '?' key or from the help menu.
 */

import { useEffect, useState } from 'react';

interface ShortcutCategory {
  name: string;
  icon: string;
  shortcuts: Array<{
    keys: string[];
    description: string;
    context?: string;
  }>;
}

const shortcutCategories: ShortcutCategory[] = [
  {
    name: 'Navigation',
    icon: '🧭',
    shortcuts: [
      { keys: ['j'], description: 'Move to next email' },
      { keys: ['k'], description: 'Move to previous email' },
      { keys: ['o', 'Enter'], description: 'Open selected email' },
      { keys: ['u'], description: 'Return to email list' },
      { keys: ['g', 'i'], description: 'Go to Inbox' },
      { keys: ['g', 's'], description: 'Go to Starred' },
      { keys: ['g', 'd'], description: 'Go to Drafts' },
      { keys: ['g', 't'], description: 'Go to Sent' },
    ],
  },
  {
    name: 'Email Actions',
    icon: '📧',
    shortcuts: [
      { keys: ['c'], description: 'Compose new email' },
      { keys: ['r'], description: 'Reply to email' },
      { keys: ['a'], description: 'Reply all' },
      { keys: ['f'], description: 'Forward email' },
      { keys: ['#'], description: 'Delete email' },
      { keys: ['s'], description: 'Star/unstar email' },
      { keys: ['x'], description: 'Select/deselect email' },
      { keys: ['Shift', 'i'], description: 'Mark as read' },
      { keys: ['Shift', 'u'], description: 'Mark as unread' },
    ],
  },
  {
    name: 'Epistemic Features',
    icon: '🔐',
    shortcuts: [
      { keys: ['Alt', 'i'], description: 'Toggle AI Insights panel', context: 'Email view' },
      { keys: ['Alt', 'c'], description: 'Toggle Contact Profile panel', context: 'Email view' },
      { keys: ['Alt', 't'], description: 'Toggle Thread Summary panel', context: 'Email view' },
      { keys: ['Alt', 'g'], description: 'Open Trust Graph', context: 'Email view' },
      { keys: ['Alt', 'a'], description: 'Create attestation for sender', context: 'Email view' },
      { keys: ['Alt', 'q'], description: 'Toggle quarantine view', context: 'Email list' },
    ],
  },
  {
    name: 'Search & Filter',
    icon: '🔍',
    shortcuts: [
      { keys: ['/'], description: 'Focus search box' },
      { keys: ['Esc'], description: 'Clear search / Close panel' },
      { keys: ['Ctrl', 'Shift', 'f'], description: 'Advanced search' },
      { keys: ['Alt', 'f'], description: 'Open epistemic filters', context: 'Email list' },
    ],
  },
  {
    name: 'General',
    icon: '⚡',
    shortcuts: [
      { keys: ['?'], description: 'Show keyboard shortcuts' },
      { keys: ['Ctrl', ','], description: 'Open settings' },
      { keys: ['Ctrl', 'z'], description: 'Undo last action' },
      { keys: ['Ctrl', 'Shift', 'z'], description: 'Redo last action' },
      { keys: ['Ctrl', 'Enter'], description: 'Send email', context: 'Compose' },
    ],
  },
];

// Key display component
function KeyBadge({ keyName }: { keyName: string }) {
  // Map key names to display symbols
  const keyDisplayMap: Record<string, string> = {
    'Ctrl': '⌃',
    'Alt': '⌥',
    'Shift': '⇧',
    'Enter': '↵',
    'Esc': '⎋',
    'Tab': '⇥',
    'Backspace': '⌫',
    'Delete': '⌦',
    'ArrowUp': '↑',
    'ArrowDown': '↓',
    'ArrowLeft': '←',
    'ArrowRight': '→',
  };

  const display = keyDisplayMap[keyName] || keyName;

  return (
    <kbd className="inline-flex items-center justify-center min-w-[24px] h-6 px-1.5 text-xs font-mono font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded shadow-sm">
      {display}
    </kbd>
  );
}

// Shortcut row component
function ShortcutRow({
  keys,
  description,
  context,
}: {
  keys: string[];
  description: string;
  context?: string;
}) {
  return (
    <div className="flex items-center justify-between py-2">
      <div className="flex items-center gap-2">
        <span className="text-sm text-gray-700 dark:text-gray-300">{description}</span>
        {context && (
          <span className="text-xs text-gray-400 dark:text-gray-500">({context})</span>
        )}
      </div>
      <div className="flex items-center gap-1">
        {keys.map((key, index) => (
          <span key={index} className="flex items-center gap-1">
            <KeyBadge keyName={key} />
            {index < keys.length - 1 && (
              <span className="text-gray-400 dark:text-gray-500 text-xs">+</span>
            )}
          </span>
        ))}
      </div>
    </div>
  );
}

// Main component
interface KeyboardShortcutsProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function KeyboardShortcuts({ isOpen, onClose }: KeyboardShortcutsProps) {
  const [searchQuery, setSearchQuery] = useState('');

  // Close on Escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Filter shortcuts by search
  const filteredCategories = shortcutCategories
    .map((category) => ({
      ...category,
      shortcuts: category.shortcuts.filter(
        (shortcut) =>
          shortcut.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
          shortcut.keys.some((k) => k.toLowerCase().includes(searchQuery.toLowerCase()))
      ),
    }))
    .filter((category) => category.shortcuts.length > 0);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-50"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="fixed inset-4 md:inset-auto md:top-1/2 md:left-1/2 md:-translate-x-1/2 md:-translate-y-1/2 md:w-[600px] md:max-h-[80vh] bg-white dark:bg-gray-900 rounded-xl shadow-2xl z-50 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3">
            <span className="text-2xl">⌨️</span>
            <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
              Keyboard Shortcuts
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Search */}
        <div className="px-6 py-3 border-b border-gray-200 dark:border-gray-700">
          <div className="relative">
            <svg
              className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            <input
              type="text"
              placeholder="Search shortcuts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 text-sm border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              autoFocus
            />
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {filteredCategories.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-gray-500 dark:text-gray-400">No shortcuts found for "{searchQuery}"</p>
            </div>
          ) : (
            <div className="space-y-6">
              {filteredCategories.map((category) => (
                <div key={category.name}>
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-lg">{category.icon}</span>
                    <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                      {category.name}
                    </h3>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg px-4 divide-y divide-gray-200 dark:divide-gray-700">
                    {category.shortcuts.map((shortcut, index) => (
                      <ShortcutRow
                        key={index}
                        keys={shortcut.keys}
                        description={shortcut.description}
                        context={shortcut.context}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
          <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
            Press <KeyBadge keyName="?" /> anytime to show this help
          </p>
        </div>
      </div>
    </>
  );
}

// Hook to manage keyboard shortcuts modal
export function useKeyboardShortcuts() {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if typing in input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // '?' key opens shortcuts help
      if (e.key === '?' && !e.ctrlKey && !e.altKey && !e.metaKey) {
        e.preventDefault();
        setIsOpen(true);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return {
    isOpen,
    open: () => setIsOpen(true),
    close: () => setIsOpen(false),
    toggle: () => setIsOpen((v) => !v),
  };
}
