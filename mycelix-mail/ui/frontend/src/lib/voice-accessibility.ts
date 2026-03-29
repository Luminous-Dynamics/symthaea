// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Voice & Accessibility Service
 *
 * Provides:
 * - Voice commands for hands-free operation
 * - Speech-to-text email dictation
 * - Screen reader optimizations
 * - High contrast and dyslexia-friendly modes
 * - Keyboard navigation enhancements
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// ============================================================================
// Types
// ============================================================================

export interface VoiceCommand {
  id: string;
  phrases: string[];
  action: string;
  description: string;
  category: 'navigation' | 'email' | 'search' | 'compose' | 'system';
  parameters?: VoiceCommandParameter[];
}

export interface VoiceCommandParameter {
  name: string;
  type: 'text' | 'number' | 'email' | 'folder';
  required: boolean;
  extractPattern?: RegExp;
}

export interface VoiceRecognitionResult {
  transcript: string;
  confidence: number;
  isFinal: boolean;
  matchedCommand?: VoiceCommand;
  extractedParams?: Record<string, string>;
}

export interface DictationSession {
  id: string;
  startedAt: Date;
  endedAt?: Date;
  transcript: string;
  segments: DictationSegment[];
  status: 'active' | 'paused' | 'completed' | 'cancelled';
}

export interface DictationSegment {
  text: string;
  startTime: number;
  endTime: number;
  confidence: number;
  isFinal: boolean;
}

export interface AccessibilitySettings {
  // Visual
  highContrastMode: boolean;
  dyslexiaFont: boolean;
  fontSize: 'small' | 'medium' | 'large' | 'extra-large';
  reduceMotion: boolean;
  colorBlindMode: 'none' | 'protanopia' | 'deuteranopia' | 'tritanopia';

  // Audio
  voiceCommandsEnabled: boolean;
  dictationEnabled: boolean;
  audioFeedback: boolean;
  speechRate: number;

  // Navigation
  keyboardShortcutsEnabled: boolean;
  focusIndicatorEnhanced: boolean;
  skipNavigation: boolean;

  // Screen Reader
  screenReaderOptimized: boolean;
  verbosityLevel: 'minimal' | 'normal' | 'verbose';
  announceNotifications: boolean;
}

export interface KeyboardShortcut {
  id: string;
  keys: string[];
  action: string;
  description: string;
  category: 'navigation' | 'email' | 'compose' | 'search' | 'system';
  isCustomizable: boolean;
}

export interface FocusableElement {
  id: string;
  element: HTMLElement;
  label: string;
  role: string;
  tabIndex: number;
  region: string;
}

// ============================================================================
// Voice Recognition Service
// ============================================================================

class VoiceRecognitionService {
  private recognition: SpeechRecognition | null = null;
  private isListening = false;
  private commands: VoiceCommand[] = [];
  private onResultCallback: ((result: VoiceRecognitionResult) => void) | null = null;
  private onErrorCallback: ((error: string) => void) | null = null;

  constructor() {
    if (typeof window !== 'undefined') {
      const SpeechRecognition =
        (window as Window & { SpeechRecognition?: typeof window.SpeechRecognition; webkitSpeechRecognition?: typeof window.SpeechRecognition }).SpeechRecognition ||
        (window as Window & { webkitSpeechRecognition?: typeof window.SpeechRecognition }).webkitSpeechRecognition;

      if (SpeechRecognition) {
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';

        this.recognition.onresult = (event) => {
          const result = event.results[event.results.length - 1];
          const transcript = result[0].transcript.trim().toLowerCase();
          const confidence = result[0].confidence;
          const isFinal = result.isFinal;

          const matchedCommand = this.matchCommand(transcript);

          this.onResultCallback?.({
            transcript,
            confidence,
            isFinal,
            matchedCommand: matchedCommand?.command,
            extractedParams: matchedCommand?.params,
          });
        };

        this.recognition.onerror = (event) => {
          this.onErrorCallback?.(event.error);
        };

        this.recognition.onend = () => {
          if (this.isListening) {
            this.recognition?.start();
          }
        };
      }
    }

    this.initializeCommands();
  }

  private initializeCommands(): void {
    this.commands = [
      // Navigation commands
      {
        id: 'nav_inbox',
        phrases: ['go to inbox', 'open inbox', 'show inbox'],
        action: 'navigate:inbox',
        description: 'Navigate to inbox',
        category: 'navigation',
      },
      {
        id: 'nav_sent',
        phrases: ['go to sent', 'open sent', 'show sent'],
        action: 'navigate:sent',
        description: 'Navigate to sent folder',
        category: 'navigation',
      },
      {
        id: 'nav_drafts',
        phrases: ['go to drafts', 'open drafts', 'show drafts'],
        action: 'navigate:drafts',
        description: 'Navigate to drafts',
        category: 'navigation',
      },
      {
        id: 'nav_settings',
        phrases: ['go to settings', 'open settings'],
        action: 'navigate:settings',
        description: 'Navigate to settings',
        category: 'navigation',
      },

      // Email commands
      {
        id: 'email_compose',
        phrases: ['compose email', 'new email', 'write email', 'start new email'],
        action: 'compose:new',
        description: 'Start composing a new email',
        category: 'compose',
      },
      {
        id: 'email_reply',
        phrases: ['reply', 'reply to this', 'respond'],
        action: 'email:reply',
        description: 'Reply to current email',
        category: 'email',
      },
      {
        id: 'email_forward',
        phrases: ['forward', 'forward this'],
        action: 'email:forward',
        description: 'Forward current email',
        category: 'email',
      },
      {
        id: 'email_delete',
        phrases: ['delete', 'delete this', 'trash this'],
        action: 'email:delete',
        description: 'Delete current email',
        category: 'email',
      },
      {
        id: 'email_archive',
        phrases: ['archive', 'archive this'],
        action: 'email:archive',
        description: 'Archive current email',
        category: 'email',
      },
      {
        id: 'email_star',
        phrases: ['star', 'star this', 'mark important'],
        action: 'email:star',
        description: 'Star current email',
        category: 'email',
      },
      {
        id: 'email_mark_read',
        phrases: ['mark as read', 'mark read'],
        action: 'email:markRead',
        description: 'Mark email as read',
        category: 'email',
      },
      {
        id: 'email_mark_unread',
        phrases: ['mark as unread', 'mark unread'],
        action: 'email:markUnread',
        description: 'Mark email as unread',
        category: 'email',
      },
      {
        id: 'email_next',
        phrases: ['next email', 'next', 'go next'],
        action: 'email:next',
        description: 'Go to next email',
        category: 'email',
      },
      {
        id: 'email_previous',
        phrases: ['previous email', 'previous', 'go back'],
        action: 'email:previous',
        description: 'Go to previous email',
        category: 'email',
      },

      // Search commands
      {
        id: 'search_open',
        phrases: ['search', 'open search', 'find'],
        action: 'search:open',
        description: 'Open search',
        category: 'search',
      },
      {
        id: 'search_from',
        phrases: ['search from', 'emails from'],
        action: 'search:from',
        description: 'Search emails from specific sender',
        category: 'search',
        parameters: [{ name: 'sender', type: 'text', required: true }],
      },

      // System commands
      {
        id: 'system_refresh',
        phrases: ['refresh', 'check for new mail', 'sync'],
        action: 'system:refresh',
        description: 'Refresh and sync emails',
        category: 'system',
      },
      {
        id: 'system_help',
        phrases: ['help', 'what can i say', 'show commands'],
        action: 'system:help',
        description: 'Show available voice commands',
        category: 'system',
      },
      {
        id: 'system_stop',
        phrases: ['stop listening', 'stop', 'disable voice'],
        action: 'system:stopListening',
        description: 'Stop voice recognition',
        category: 'system',
      },
    ];
  }

  private matchCommand(transcript: string): { command: VoiceCommand; params: Record<string, string> } | null {
    for (const command of this.commands) {
      for (const phrase of command.phrases) {
        if (transcript.includes(phrase)) {
          const params: Record<string, string> = {};

          // Extract parameters if any
          if (command.parameters) {
            for (const param of command.parameters) {
              if (param.extractPattern) {
                const match = transcript.match(param.extractPattern);
                if (match) {
                  params[param.name] = match[1];
                }
              } else {
                // Simple extraction: take text after the phrase
                const afterPhrase = transcript.slice(transcript.indexOf(phrase) + phrase.length).trim();
                if (afterPhrase) {
                  params[param.name] = afterPhrase;
                }
              }
            }
          }

          return { command, params };
        }
      }
    }
    return null;
  }

  start(): boolean {
    if (!this.recognition) return false;

    try {
      this.recognition.start();
      this.isListening = true;
      return true;
    } catch (error) {
      console.error('Failed to start voice recognition:', error);
      return false;
    }
  }

  stop(): void {
    this.isListening = false;
    this.recognition?.stop();
  }

  onResult(callback: (result: VoiceRecognitionResult) => void): void {
    this.onResultCallback = callback;
  }

  onError(callback: (error: string) => void): void {
    this.onErrorCallback = callback;
  }

  isSupported(): boolean {
    return this.recognition !== null;
  }

  getCommands(): VoiceCommand[] {
    return this.commands;
  }
}

// ============================================================================
// Dictation Service
// ============================================================================

class DictationService {
  private recognition: SpeechRecognition | null = null;
  private currentSession: DictationSession | null = null;
  private onSegmentCallback: ((segment: DictationSegment) => void) | null = null;

  constructor() {
    if (typeof window !== 'undefined') {
      const SpeechRecognition =
        (window as Window & { SpeechRecognition?: typeof window.SpeechRecognition; webkitSpeechRecognition?: typeof window.SpeechRecognition }).SpeechRecognition ||
        (window as Window & { webkitSpeechRecognition?: typeof window.SpeechRecognition }).webkitSpeechRecognition;

      if (SpeechRecognition) {
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';

        this.recognition.onresult = (event) => {
          if (!this.currentSession) return;

          const result = event.results[event.results.length - 1];
          const segment: DictationSegment = {
            text: result[0].transcript,
            startTime: Date.now() - this.currentSession.startedAt.getTime(),
            endTime: Date.now() - this.currentSession.startedAt.getTime(),
            confidence: result[0].confidence,
            isFinal: result.isFinal,
          };

          if (result.isFinal) {
            this.currentSession.segments.push(segment);
            this.currentSession.transcript += ' ' + segment.text;
          }

          this.onSegmentCallback?.(segment);
        };
      }
    }
  }

  startSession(): DictationSession | null {
    if (!this.recognition) return null;

    this.currentSession = {
      id: `dictation_${Date.now()}`,
      startedAt: new Date(),
      transcript: '',
      segments: [],
      status: 'active',
    };

    this.recognition.start();
    return this.currentSession;
  }

  pauseSession(): void {
    if (this.currentSession) {
      this.currentSession.status = 'paused';
      this.recognition?.stop();
    }
  }

  resumeSession(): void {
    if (this.currentSession?.status === 'paused') {
      this.currentSession.status = 'active';
      this.recognition?.start();
    }
  }

  endSession(): DictationSession | null {
    if (!this.currentSession) return null;

    this.recognition?.stop();
    this.currentSession.status = 'completed';
    this.currentSession.endedAt = new Date();

    const session = this.currentSession;
    this.currentSession = null;
    return session;
  }

  cancelSession(): void {
    if (this.currentSession) {
      this.currentSession.status = 'cancelled';
      this.currentSession = null;
    }
    this.recognition?.stop();
  }

  onSegment(callback: (segment: DictationSegment) => void): void {
    this.onSegmentCallback = callback;
  }

  getCurrentSession(): DictationSession | null {
    return this.currentSession;
  }

  isSupported(): boolean {
    return this.recognition !== null;
  }
}

// ============================================================================
// Text-to-Speech Service
// ============================================================================

class TextToSpeechService {
  private synth: SpeechSynthesis | null = null;
  private currentUtterance: SpeechSynthesisUtterance | null = null;

  constructor() {
    if (typeof window !== 'undefined') {
      this.synth = window.speechSynthesis;
    }
  }

  speak(text: string, options?: { rate?: number; pitch?: number; volume?: number }): void {
    if (!this.synth) return;

    // Cancel any ongoing speech
    this.synth.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = options?.rate ?? 1;
    utterance.pitch = options?.pitch ?? 1;
    utterance.volume = options?.volume ?? 1;

    this.currentUtterance = utterance;
    this.synth.speak(utterance);
  }

  stop(): void {
    this.synth?.cancel();
  }

  pause(): void {
    this.synth?.pause();
  }

  resume(): void {
    this.synth?.resume();
  }

  isSpeaking(): boolean {
    return this.synth?.speaking ?? false;
  }

  getVoices(): SpeechSynthesisVoice[] {
    return this.synth?.getVoices() ?? [];
  }

  isSupported(): boolean {
    return this.synth !== null;
  }
}

// ============================================================================
// Accessibility Manager
// ============================================================================

class AccessibilityManager {
  private announcer: HTMLElement | null = null;

  constructor() {
    this.createAnnouncer();
  }

  private createAnnouncer(): void {
    if (typeof document === 'undefined') return;

    this.announcer = document.createElement('div');
    this.announcer.setAttribute('role', 'status');
    this.announcer.setAttribute('aria-live', 'polite');
    this.announcer.setAttribute('aria-atomic', 'true');
    this.announcer.className = 'sr-only';
    this.announcer.style.cssText =
      'position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;';

    document.body.appendChild(this.announcer);
  }

  announce(message: string, priority: 'polite' | 'assertive' = 'polite'): void {
    if (!this.announcer) return;

    this.announcer.setAttribute('aria-live', priority);
    this.announcer.textContent = '';

    // Small delay to ensure screen readers pick up the change
    setTimeout(() => {
      if (this.announcer) {
        this.announcer.textContent = message;
      }
    }, 100);
  }

  setFocus(element: HTMLElement): void {
    element.focus();
    this.announce(`Focused on ${element.getAttribute('aria-label') || element.textContent}`);
  }

  trapFocus(container: HTMLElement): () => void {
    const focusableElements = container.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    const firstFocusable = focusableElements[0];
    const lastFocusable = focusableElements[focusableElements.length - 1];

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstFocusable) {
          e.preventDefault();
          lastFocusable?.focus();
        }
      } else {
        if (document.activeElement === lastFocusable) {
          e.preventDefault();
          firstFocusable?.focus();
        }
      }
    };

    container.addEventListener('keydown', handleKeyDown);
    firstFocusable?.focus();

    return () => {
      container.removeEventListener('keydown', handleKeyDown);
    };
  }

  applyHighContrast(enabled: boolean): void {
    if (typeof document === 'undefined') return;
    document.documentElement.classList.toggle('high-contrast', enabled);
  }

  applyDyslexiaFont(enabled: boolean): void {
    if (typeof document === 'undefined') return;
    document.documentElement.classList.toggle('dyslexia-font', enabled);
  }

  applyFontSize(size: AccessibilitySettings['fontSize']): void {
    if (typeof document === 'undefined') return;

    const sizes = {
      small: '14px',
      medium: '16px',
      large: '18px',
      'extra-large': '20px',
    };

    document.documentElement.style.setProperty('--base-font-size', sizes[size]);
  }

  applyReduceMotion(enabled: boolean): void {
    if (typeof document === 'undefined') return;
    document.documentElement.classList.toggle('reduce-motion', enabled);
  }

  applyColorBlindMode(mode: AccessibilitySettings['colorBlindMode']): void {
    if (typeof document === 'undefined') return;

    document.documentElement.classList.remove(
      'colorblind-protanopia',
      'colorblind-deuteranopia',
      'colorblind-tritanopia'
    );

    if (mode !== 'none') {
      document.documentElement.classList.add(`colorblind-${mode}`);
    }
  }
}

// ============================================================================
// Keyboard Shortcuts Manager
// ============================================================================

class KeyboardShortcutsManager {
  private shortcuts: Map<string, KeyboardShortcut> = new Map();
  private handlers: Map<string, () => void> = new Map();
  private isEnabled = true;

  constructor() {
    this.initializeDefaultShortcuts();

    if (typeof document !== 'undefined') {
      document.addEventListener('keydown', this.handleKeyDown.bind(this));
    }
  }

  private initializeDefaultShortcuts(): void {
    const defaults: KeyboardShortcut[] = [
      // Navigation
      { id: 'goto_inbox', keys: ['g', 'i'], action: 'navigate:inbox', description: 'Go to Inbox', category: 'navigation', isCustomizable: true },
      { id: 'goto_sent', keys: ['g', 's'], action: 'navigate:sent', description: 'Go to Sent', category: 'navigation', isCustomizable: true },
      { id: 'goto_drafts', keys: ['g', 'd'], action: 'navigate:drafts', description: 'Go to Drafts', category: 'navigation', isCustomizable: true },
      { id: 'goto_settings', keys: ['g', ','], action: 'navigate:settings', description: 'Go to Settings', category: 'navigation', isCustomizable: true },

      // Email actions
      { id: 'compose', keys: ['c'], action: 'compose:new', description: 'Compose new email', category: 'compose', isCustomizable: true },
      { id: 'reply', keys: ['r'], action: 'email:reply', description: 'Reply', category: 'email', isCustomizable: true },
      { id: 'reply_all', keys: ['a'], action: 'email:replyAll', description: 'Reply all', category: 'email', isCustomizable: true },
      { id: 'forward', keys: ['f'], action: 'email:forward', description: 'Forward', category: 'email', isCustomizable: true },
      { id: 'archive', keys: ['e'], action: 'email:archive', description: 'Archive', category: 'email', isCustomizable: true },
      { id: 'delete', keys: ['#'], action: 'email:delete', description: 'Delete', category: 'email', isCustomizable: true },
      { id: 'star', keys: ['s'], action: 'email:star', description: 'Star/Unstar', category: 'email', isCustomizable: true },
      { id: 'mark_read', keys: ['Shift', 'i'], action: 'email:markRead', description: 'Mark as read', category: 'email', isCustomizable: true },
      { id: 'mark_unread', keys: ['Shift', 'u'], action: 'email:markUnread', description: 'Mark as unread', category: 'email', isCustomizable: true },

      // Navigation within list
      { id: 'next_email', keys: ['j'], action: 'email:next', description: 'Next email', category: 'email', isCustomizable: true },
      { id: 'prev_email', keys: ['k'], action: 'email:previous', description: 'Previous email', category: 'email', isCustomizable: true },
      { id: 'open_email', keys: ['o'], action: 'email:open', description: 'Open email', category: 'email', isCustomizable: true },
      { id: 'back', keys: ['u'], action: 'navigate:back', description: 'Go back to list', category: 'navigation', isCustomizable: true },

      // Search
      { id: 'search', keys: ['/'], action: 'search:open', description: 'Search', category: 'search', isCustomizable: true },
      { id: 'search_advanced', keys: ['Shift', '/'], action: 'search:advanced', description: 'Advanced search', category: 'search', isCustomizable: true },

      // System
      { id: 'refresh', keys: ['Shift', 'n'], action: 'system:refresh', description: 'Refresh', category: 'system', isCustomizable: true },
      { id: 'help', keys: ['?'], action: 'system:help', description: 'Show shortcuts', category: 'system', isCustomizable: false },
      { id: 'escape', keys: ['Escape'], action: 'system:escape', description: 'Close/Cancel', category: 'system', isCustomizable: false },
    ];

    defaults.forEach((shortcut) => {
      this.shortcuts.set(shortcut.id, shortcut);
    });
  }

  private handleKeyDown(event: KeyboardEvent): void {
    if (!this.isEnabled) return;

    // Ignore if typing in input
    const target = event.target as HTMLElement;
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
      return;
    }

    const pressedKeys: string[] = [];
    if (event.shiftKey) pressedKeys.push('Shift');
    if (event.ctrlKey) pressedKeys.push('Ctrl');
    if (event.altKey) pressedKeys.push('Alt');
    if (event.metaKey) pressedKeys.push('Meta');
    pressedKeys.push(event.key);

    // Find matching shortcut
    for (const shortcut of this.shortcuts.values()) {
      if (this.keysMatch(pressedKeys, shortcut.keys)) {
        event.preventDefault();
        const handler = this.handlers.get(shortcut.action);
        handler?.();
        return;
      }
    }
  }

  private keysMatch(pressed: string[], shortcut: string[]): boolean {
    if (pressed.length !== shortcut.length) return false;
    return shortcut.every((key) => pressed.includes(key));
  }

  registerHandler(action: string, handler: () => void): void {
    this.handlers.set(action, handler);
  }

  unregisterHandler(action: string): void {
    this.handlers.delete(action);
  }

  setEnabled(enabled: boolean): void {
    this.isEnabled = enabled;
  }

  getShortcuts(): KeyboardShortcut[] {
    return Array.from(this.shortcuts.values());
  }

  updateShortcut(id: string, keys: string[]): void {
    const shortcut = this.shortcuts.get(id);
    if (shortcut && shortcut.isCustomizable) {
      this.shortcuts.set(id, { ...shortcut, keys });
    }
  }

  resetToDefaults(): void {
    this.shortcuts.clear();
    this.initializeDefaultShortcuts();
  }
}

// ============================================================================
// Store
// ============================================================================

interface VoiceAccessibilityState {
  settings: AccessibilitySettings;
  isVoiceListening: boolean;
  isDictating: boolean;
  currentTranscript: string;
  recentCommands: VoiceRecognitionResult[];

  updateSettings: (updates: Partial<AccessibilitySettings>) => void;
  setVoiceListening: (listening: boolean) => void;
  setDictating: (dictating: boolean) => void;
  setCurrentTranscript: (transcript: string) => void;
  addRecentCommand: (result: VoiceRecognitionResult) => void;
}

export const useVoiceAccessibilityStore = create<VoiceAccessibilityState>()(
  persist(
    (set) => ({
      settings: {
        highContrastMode: false,
        dyslexiaFont: false,
        fontSize: 'medium',
        reduceMotion: false,
        colorBlindMode: 'none',
        voiceCommandsEnabled: false,
        dictationEnabled: false,
        audioFeedback: true,
        speechRate: 1,
        keyboardShortcutsEnabled: true,
        focusIndicatorEnhanced: false,
        skipNavigation: true,
        screenReaderOptimized: false,
        verbosityLevel: 'normal',
        announceNotifications: true,
      },
      isVoiceListening: false,
      isDictating: false,
      currentTranscript: '',
      recentCommands: [],

      updateSettings: (updates) =>
        set((state) => ({
          settings: { ...state.settings, ...updates },
        })),

      setVoiceListening: (listening) => set({ isVoiceListening: listening }),
      setDictating: (dictating) => set({ isDictating: dictating }),
      setCurrentTranscript: (transcript) => set({ currentTranscript: transcript }),
      addRecentCommand: (result) =>
        set((state) => ({
          recentCommands: [result, ...state.recentCommands.slice(0, 9)],
        })),
    }),
    {
      name: 'mycelix-accessibility',
    }
  )
);

// ============================================================================
// Singleton Services
// ============================================================================

const voiceRecognition = new VoiceRecognitionService();
const dictation = new DictationService();
const textToSpeech = new TextToSpeechService();
const accessibilityManager = new AccessibilityManager();
const keyboardShortcuts = new KeyboardShortcutsManager();

// ============================================================================
// React Hooks
// ============================================================================

import { useCallback, useEffect } from 'react';

export function useVoiceCommands(onCommand?: (action: string, params?: Record<string, string>) => void) {
  const { settings, setVoiceListening, isVoiceListening, addRecentCommand } =
    useVoiceAccessibilityStore();

  useEffect(() => {
    if (!settings.voiceCommandsEnabled) return;

    voiceRecognition.onResult((result) => {
      addRecentCommand(result);

      if (result.isFinal && result.matchedCommand && onCommand) {
        onCommand(result.matchedCommand.action, result.extractedParams);
      }
    });
  }, [settings.voiceCommandsEnabled, onCommand, addRecentCommand]);

  const startListening = useCallback(() => {
    if (voiceRecognition.start()) {
      setVoiceListening(true);
    }
  }, [setVoiceListening]);

  const stopListening = useCallback(() => {
    voiceRecognition.stop();
    setVoiceListening(false);
  }, [setVoiceListening]);

  return {
    isListening: isVoiceListening,
    isSupported: voiceRecognition.isSupported(),
    commands: voiceRecognition.getCommands(),
    startListening,
    stopListening,
  };
}

export function useDictation() {
  const { setDictating, isDictating, setCurrentTranscript, currentTranscript } =
    useVoiceAccessibilityStore();

  useEffect(() => {
    dictation.onSegment((segment) => {
      setCurrentTranscript(segment.text);
    });
  }, [setCurrentTranscript]);

  const startDictation = useCallback(() => {
    const session = dictation.startSession();
    if (session) {
      setDictating(true);
    }
    return session;
  }, [setDictating]);

  const stopDictation = useCallback(() => {
    const session = dictation.endSession();
    setDictating(false);
    setCurrentTranscript('');
    return session;
  }, [setDictating, setCurrentTranscript]);

  const pauseDictation = useCallback(() => {
    dictation.pauseSession();
  }, []);

  const resumeDictation = useCallback(() => {
    dictation.resumeSession();
  }, []);

  return {
    isDictating,
    currentTranscript,
    isSupported: dictation.isSupported(),
    startDictation,
    stopDictation,
    pauseDictation,
    resumeDictation,
    getSession: () => dictation.getCurrentSession(),
  };
}

export function useTextToSpeech() {
  const { settings } = useVoiceAccessibilityStore();

  const speak = useCallback(
    (text: string) => {
      if (settings.audioFeedback) {
        textToSpeech.speak(text, { rate: settings.speechRate });
      }
    },
    [settings.audioFeedback, settings.speechRate]
  );

  return {
    speak,
    stop: () => textToSpeech.stop(),
    pause: () => textToSpeech.pause(),
    resume: () => textToSpeech.resume(),
    isSpeaking: () => textToSpeech.isSpeaking(),
    isSupported: textToSpeech.isSupported(),
    voices: textToSpeech.getVoices(),
  };
}

export function useAccessibility() {
  const { settings, updateSettings } = useVoiceAccessibilityStore();

  // Apply settings when they change
  useEffect(() => {
    accessibilityManager.applyHighContrast(settings.highContrastMode);
    accessibilityManager.applyDyslexiaFont(settings.dyslexiaFont);
    accessibilityManager.applyFontSize(settings.fontSize);
    accessibilityManager.applyReduceMotion(settings.reduceMotion);
    accessibilityManager.applyColorBlindMode(settings.colorBlindMode);
  }, [settings]);

  const announce = useCallback((message: string, priority?: 'polite' | 'assertive') => {
    accessibilityManager.announce(message, priority);
  }, []);

  const setFocus = useCallback((element: HTMLElement) => {
    accessibilityManager.setFocus(element);
  }, []);

  const trapFocus = useCallback((container: HTMLElement) => {
    return accessibilityManager.trapFocus(container);
  }, []);

  return {
    settings,
    updateSettings,
    announce,
    setFocus,
    trapFocus,
  };
}

export function useKeyboardShortcuts(handlers: Record<string, () => void>) {
  const { settings } = useVoiceAccessibilityStore();

  useEffect(() => {
    if (!settings.keyboardShortcutsEnabled) {
      keyboardShortcuts.setEnabled(false);
      return;
    }

    keyboardShortcuts.setEnabled(true);

    Object.entries(handlers).forEach(([action, handler]) => {
      keyboardShortcuts.registerHandler(action, handler);
    });

    return () => {
      Object.keys(handlers).forEach((action) => {
        keyboardShortcuts.unregisterHandler(action);
      });
    };
  }, [handlers, settings.keyboardShortcutsEnabled]);

  return {
    shortcuts: keyboardShortcuts.getShortcuts(),
    updateShortcut: keyboardShortcuts.updateShortcut.bind(keyboardShortcuts),
    resetToDefaults: keyboardShortcuts.resetToDefaults.bind(keyboardShortcuts),
    isEnabled: settings.keyboardShortcutsEnabled,
  };
}
