// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

/**
 * Keyboard Shortcuts Modal
 *
 * Displays all available keyboard shortcuts organized by category.
 */

import { Fragment } from 'react';
import { X, Keyboard } from 'lucide-react';
import { Shortcut, ShortcutCategory, getShortcutDisplay } from '@/hooks/useKeyboardShortcuts';

interface ShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
  shortcuts: Shortcut[];
}

const categoryLabels: Record<ShortcutCategory, string> = {
  playback: 'Playback',
  navigation: 'Navigation',
  studio: 'Studio',
  general: 'General',
};

const categoryOrder: ShortcutCategory[] = ['playback', 'navigation', 'studio', 'general'];

export function ShortcutsModal({ isOpen, onClose, shortcuts }: ShortcutsModalProps) {
  if (!isOpen) return null;

  // Group shortcuts by category
  const grouped = categoryOrder.reduce(
    (acc, category) => {
      const categoryShortcuts = shortcuts.filter((s) => s.category === category);
      if (categoryShortcuts.length > 0) {
        acc[category] = categoryShortcuts;
      }
      return acc;
    },
    {} as Record<ShortcutCategory, Shortcut[]>
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />

      {/* Modal */}
      <div className="relative bg-zinc-900 rounded-xl border border-white/10 w-full max-w-2xl max-h-[80vh] overflow-hidden shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-purple-500/20">
              <Keyboard className="w-5 h-5 text-purple-400" />
            </div>
            <h2 className="text-xl font-semibold text-white">Keyboard Shortcuts</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5 text-muted-foreground" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(80vh-80px)]">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {categoryOrder.map((category) => {
              const categoryShortcuts = grouped[category];
              if (!categoryShortcuts) return null;

              return (
                <div key={category}>
                  <h3 className="text-sm font-semibold text-purple-400 uppercase tracking-wider mb-4">
                    {categoryLabels[category]}
                  </h3>
                  <div className="space-y-3">
                    {categoryShortcuts.map((shortcut, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between"
                      >
                        <span className="text-sm text-muted-foreground">
                          {shortcut.description}
                        </span>
                        <kbd className="px-2 py-1 text-xs font-mono bg-white/5 border border-white/10 rounded">
                          {getShortcutDisplay(shortcut.key, shortcut.modifiers)}
                        </kbd>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Footer hint */}
          <div className="mt-8 pt-4 border-t border-white/10">
            <p className="text-xs text-muted-foreground text-center">
              Press <kbd className="px-1 py-0.5 bg-white/5 rounded text-xs">?</kbd> to toggle this dialog
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ShortcutsModal;
