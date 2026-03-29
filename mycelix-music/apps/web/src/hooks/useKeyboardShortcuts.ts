// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Keyboard Shortcuts Hook
 *
 * Global keyboard shortcuts for the music app.
 * Provides playback control, navigation, and studio shortcuts.
 */

import { useEffect, useCallback, useRef } from 'react';

// Shortcut categories
export type ShortcutCategory = 'playback' | 'navigation' | 'studio' | 'general';

export interface Shortcut {
  key: string;
  modifiers?: ('ctrl' | 'alt' | 'shift' | 'meta')[];
  description: string;
  category: ShortcutCategory;
  action: () => void;
}

interface ShortcutConfig {
  // Playback
  togglePlay?: () => void;
  nextTrack?: () => void;
  prevTrack?: () => void;
  seekForward?: () => void;
  seekBackward?: () => void;
  volumeUp?: () => void;
  volumeDown?: () => void;
  toggleMute?: () => void;
  toggleShuffle?: () => void;
  toggleRepeat?: () => void;
  toggleLike?: () => void;

  // Navigation
  goHome?: () => void;
  goSearch?: () => void;
  goLibrary?: () => void;
  goStudio?: () => void;
  goSettings?: () => void;

  // Studio
  toggleRecording?: () => void;
  undo?: () => void;
  redo?: () => void;
  save?: () => void;
  zoomIn?: () => void;
  zoomOut?: () => void;
  selectAll?: () => void;
  deleteSelected?: () => void;
  duplicate?: () => void;
  split?: () => void;

  // General
  showShortcuts?: () => void;
  toggleFullscreen?: () => void;
  escape?: () => void;
}

interface UseKeyboardShortcutsOptions {
  enabled?: boolean;
  enableInInputs?: boolean;
}

/**
 * Get shortcut key display string
 */
export function getShortcutDisplay(key: string, modifiers?: string[]): string {
  const isMac = typeof navigator !== 'undefined' && /Mac/.test(navigator.platform);

  const modifierSymbols: Record<string, string> = isMac
    ? { ctrl: '⌃', alt: '⌥', shift: '⇧', meta: '⌘' }
    : { ctrl: 'Ctrl', alt: 'Alt', shift: 'Shift', meta: 'Win' };

  const parts = [
    ...(modifiers?.map((m) => modifierSymbols[m]) || []),
    key.toUpperCase(),
  ];

  return parts.join(isMac ? '' : '+');
}

/**
 * Default shortcuts configuration
 */
export function getDefaultShortcuts(config: ShortcutConfig): Shortcut[] {
  const shortcuts: Shortcut[] = [];

  // Playback shortcuts
  if (config.togglePlay) {
    shortcuts.push({
      key: ' ',
      description: 'Play/Pause',
      category: 'playback',
      action: config.togglePlay,
    });
  }

  if (config.nextTrack) {
    shortcuts.push({
      key: 'ArrowRight',
      modifiers: ['ctrl'],
      description: 'Next Track',
      category: 'playback',
      action: config.nextTrack,
    });
  }

  if (config.prevTrack) {
    shortcuts.push({
      key: 'ArrowLeft',
      modifiers: ['ctrl'],
      description: 'Previous Track',
      category: 'playback',
      action: config.prevTrack,
    });
  }

  if (config.seekForward) {
    shortcuts.push({
      key: 'ArrowRight',
      modifiers: ['shift'],
      description: 'Seek Forward 10s',
      category: 'playback',
      action: config.seekForward,
    });
  }

  if (config.seekBackward) {
    shortcuts.push({
      key: 'ArrowLeft',
      modifiers: ['shift'],
      description: 'Seek Backward 10s',
      category: 'playback',
      action: config.seekBackward,
    });
  }

  if (config.volumeUp) {
    shortcuts.push({
      key: 'ArrowUp',
      description: 'Volume Up',
      category: 'playback',
      action: config.volumeUp,
    });
  }

  if (config.volumeDown) {
    shortcuts.push({
      key: 'ArrowDown',
      description: 'Volume Down',
      category: 'playback',
      action: config.volumeDown,
    });
  }

  if (config.toggleMute) {
    shortcuts.push({
      key: 'm',
      description: 'Toggle Mute',
      category: 'playback',
      action: config.toggleMute,
    });
  }

  if (config.toggleShuffle) {
    shortcuts.push({
      key: 's',
      description: 'Toggle Shuffle',
      category: 'playback',
      action: config.toggleShuffle,
    });
  }

  if (config.toggleRepeat) {
    shortcuts.push({
      key: 'r',
      description: 'Toggle Repeat',
      category: 'playback',
      action: config.toggleRepeat,
    });
  }

  if (config.toggleLike) {
    shortcuts.push({
      key: 'l',
      description: 'Like/Unlike Track',
      category: 'playback',
      action: config.toggleLike,
    });
  }

  // Navigation shortcuts
  if (config.goHome) {
    shortcuts.push({
      key: 'h',
      modifiers: ['ctrl'],
      description: 'Go to Home',
      category: 'navigation',
      action: config.goHome,
    });
  }

  if (config.goSearch) {
    shortcuts.push({
      key: 'k',
      modifiers: ['ctrl'],
      description: 'Search',
      category: 'navigation',
      action: config.goSearch,
    });
  }

  if (config.goLibrary) {
    shortcuts.push({
      key: 'b',
      modifiers: ['ctrl'],
      description: 'Go to Library',
      category: 'navigation',
      action: config.goLibrary,
    });
  }

  if (config.goStudio) {
    shortcuts.push({
      key: 't',
      modifiers: ['ctrl'],
      description: 'Go to Studio',
      category: 'navigation',
      action: config.goStudio,
    });
  }

  // Studio shortcuts
  if (config.undo) {
    shortcuts.push({
      key: 'z',
      modifiers: ['ctrl'],
      description: 'Undo',
      category: 'studio',
      action: config.undo,
    });
  }

  if (config.redo) {
    shortcuts.push({
      key: 'z',
      modifiers: ['ctrl', 'shift'],
      description: 'Redo',
      category: 'studio',
      action: config.redo,
    });

    shortcuts.push({
      key: 'y',
      modifiers: ['ctrl'],
      description: 'Redo',
      category: 'studio',
      action: config.redo,
    });
  }

  if (config.save) {
    shortcuts.push({
      key: 's',
      modifiers: ['ctrl'],
      description: 'Save',
      category: 'studio',
      action: config.save,
    });
  }

  if (config.toggleRecording) {
    shortcuts.push({
      key: 'r',
      modifiers: ['ctrl'],
      description: 'Toggle Recording',
      category: 'studio',
      action: config.toggleRecording,
    });
  }

  if (config.zoomIn) {
    shortcuts.push({
      key: '=',
      modifiers: ['ctrl'],
      description: 'Zoom In',
      category: 'studio',
      action: config.zoomIn,
    });
  }

  if (config.zoomOut) {
    shortcuts.push({
      key: '-',
      modifiers: ['ctrl'],
      description: 'Zoom Out',
      category: 'studio',
      action: config.zoomOut,
    });
  }

  if (config.selectAll) {
    shortcuts.push({
      key: 'a',
      modifiers: ['ctrl'],
      description: 'Select All',
      category: 'studio',
      action: config.selectAll,
    });
  }

  if (config.deleteSelected) {
    shortcuts.push({
      key: 'Delete',
      description: 'Delete Selected',
      category: 'studio',
      action: config.deleteSelected,
    });

    shortcuts.push({
      key: 'Backspace',
      description: 'Delete Selected',
      category: 'studio',
      action: config.deleteSelected,
    });
  }

  if (config.duplicate) {
    shortcuts.push({
      key: 'd',
      modifiers: ['ctrl'],
      description: 'Duplicate',
      category: 'studio',
      action: config.duplicate,
    });
  }

  if (config.split) {
    shortcuts.push({
      key: 'x',
      modifiers: ['ctrl'],
      description: 'Split at Playhead',
      category: 'studio',
      action: config.split,
    });
  }

  // General shortcuts
  if (config.showShortcuts) {
    shortcuts.push({
      key: '?',
      modifiers: ['shift'],
      description: 'Show Shortcuts',
      category: 'general',
      action: config.showShortcuts,
    });
  }

  if (config.toggleFullscreen) {
    shortcuts.push({
      key: 'f',
      description: 'Toggle Fullscreen',
      category: 'general',
      action: config.toggleFullscreen,
    });
  }

  if (config.escape) {
    shortcuts.push({
      key: 'Escape',
      description: 'Close/Cancel',
      category: 'general',
      action: config.escape,
    });
  }

  return shortcuts;
}

/**
 * Hook for registering keyboard shortcuts
 */
export function useKeyboardShortcuts(
  config: ShortcutConfig,
  options: UseKeyboardShortcutsOptions = {}
) {
  const { enabled = true, enableInInputs = false } = options;
  const shortcuts = useRef<Shortcut[]>([]);

  // Build shortcuts from config
  useEffect(() => {
    shortcuts.current = getDefaultShortcuts(config);
  }, [config]);

  // Check if event matches shortcut
  const matchesShortcut = useCallback((event: KeyboardEvent, shortcut: Shortcut): boolean => {
    // Check key
    if (event.key !== shortcut.key && event.code !== shortcut.key) {
      return false;
    }

    const modifiers = shortcut.modifiers || [];

    // Check modifiers
    const hasCtrl = modifiers.includes('ctrl') || modifiers.includes('meta');
    const hasAlt = modifiers.includes('alt');
    const hasShift = modifiers.includes('shift');

    const ctrlPressed = event.ctrlKey || event.metaKey;

    if (hasCtrl !== ctrlPressed) return false;
    if (hasAlt !== event.altKey) return false;
    if (hasShift !== event.shiftKey) return false;

    return true;
  }, []);

  // Handle keydown
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled) return;

      // Skip if in input element (unless explicitly enabled)
      const target = event.target as HTMLElement;
      const isInput =
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable;

      if (isInput && !enableInInputs) {
        // Allow certain shortcuts even in inputs
        const allowedInInputs = ['Escape'];
        if (!allowedInInputs.includes(event.key)) {
          return;
        }
      }

      // Find matching shortcut
      for (const shortcut of shortcuts.current) {
        if (matchesShortcut(event, shortcut)) {
          event.preventDefault();
          shortcut.action();
          return;
        }
      }
    },
    [enabled, enableInInputs, matchesShortcut]
  );

  // Register event listener
  useEffect(() => {
    if (!enabled) return;

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [enabled, handleKeyDown]);

  // Return shortcuts for display
  return {
    shortcuts: shortcuts.current,
    getShortcutsByCategory: (category: ShortcutCategory) =>
      shortcuts.current.filter((s) => s.category === category),
  };
}

export default useKeyboardShortcuts;
