// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Voice & Accessibility Components
 *
 * Provides voice dictation, voice commands, text-to-speech,
 * and comprehensive accessibility settings.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';

// ============================================================================
// Types
// ============================================================================

interface DictationResult {
  text: string;
  confidence: number;
  isFinal: boolean;
}

interface VoiceCommand {
  command: string;
  action: string;
  description: string;
}

interface TtsSettings {
  voiceId: string;
  speed: number;
  pitch: number;
  autoReadNew: boolean;
  skipSignatures: boolean;
  skipQuotedText: boolean;
}

interface AccessibilitySettings {
  highContrast: boolean;
  fontSize: 'small' | 'medium' | 'large' | 'xlarge';
  reducedMotion: boolean;
  screenReaderOptimized: boolean;
  keyboardOnlyMode: boolean;
  dyslexiaFriendlyFont: boolean;
  colorBlindMode: 'none' | 'protanopia' | 'deuteranopia' | 'tritanopia';
}

interface Voice {
  id: string;
  name: string;
  language: string;
  gender: string;
}

// ============================================================================
// Dictation Hook
// ============================================================================

export function useDictation(onResult: (result: DictationResult) => void) {
  const [isListening, setIsListening] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (SpeechRecognition) {
      setIsSupported(true);
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;

      recognitionRef.current.onresult = (event: any) => {
        const result = event.results[event.results.length - 1];
        onResult({
          text: result[0].transcript,
          confidence: result[0].confidence,
          isFinal: result.isFinal,
        });
      };

      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, [onResult]);

  const startListening = useCallback(() => {
    if (recognitionRef.current && !isListening) {
      recognitionRef.current.start();
      setIsListening(true);
    }
  }, [isListening]);

  const stopListening = useCallback(() => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  }, [isListening]);

  const toggleListening = useCallback(() => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  }, [isListening, startListening, stopListening]);

  return {
    isListening,
    isSupported,
    startListening,
    stopListening,
    toggleListening,
  };
}

// ============================================================================
// Dictation Button Component
// ============================================================================

interface DictationButtonProps {
  onTranscript: (text: string, isFinal: boolean) => void;
  className?: string;
}

export function DictationButton({ onTranscript, className = '' }: DictationButtonProps) {
  const [interimText, setInterimText] = useState('');

  const handleResult = useCallback((result: DictationResult) => {
    if (result.isFinal) {
      onTranscript(result.text, true);
      setInterimText('');
    } else {
      setInterimText(result.text);
    }
  }, [onTranscript]);

  const { isListening, isSupported, toggleListening } = useDictation(handleResult);

  if (!isSupported) {
    return (
      <button
        className={`dictation-button unsupported ${className}`}
        disabled
        title="Voice dictation not supported in this browser"
      >
        <MicOffIcon />
      </button>
    );
  }

  return (
    <div className="dictation-container">
      <button
        className={`dictation-button ${isListening ? 'listening' : ''} ${className}`}
        onClick={toggleListening}
        aria-label={isListening ? 'Stop dictation' : 'Start dictation'}
        aria-pressed={isListening}
      >
        {isListening ? <MicActiveIcon /> : <MicIcon />}
      </button>
      {interimText && (
        <div className="interim-text" aria-live="polite">
          {interimText}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Voice Command Listener
// ============================================================================

interface VoiceCommandListenerProps {
  onCommand: (command: string) => void;
  wakeWord?: string;
  enabled?: boolean;
}

export function VoiceCommandListener({
  onCommand,
  wakeWord = 'Hey Mail',
  enabled = true,
}: VoiceCommandListenerProps) {
  const [isWaiting, setIsWaiting] = useState(false);
  const [lastCommand, setLastCommand] = useState<string | null>(null);

  const handleResult = useCallback((result: DictationResult) => {
    if (!result.isFinal) return;

    const text = result.text.toLowerCase();

    if (!isWaiting && text.includes(wakeWord.toLowerCase())) {
      setIsWaiting(true);
      return;
    }

    if (isWaiting) {
      setLastCommand(result.text);
      onCommand(result.text);
      setIsWaiting(false);
    }
  }, [isWaiting, wakeWord, onCommand]);

  const { isListening, startListening, stopListening } = useDictation(handleResult);

  useEffect(() => {
    if (enabled && !isListening) {
      startListening();
    } else if (!enabled && isListening) {
      stopListening();
    }
  }, [enabled, isListening, startListening, stopListening]);

  return (
    <div className="voice-command-status" aria-live="polite">
      {enabled && (
        <>
          <span className={`status-indicator ${isWaiting ? 'waiting' : 'listening'}`} />
          <span className="status-text">
            {isWaiting ? 'Listening for command...' : `Say "${wakeWord}" to activate`}
          </span>
          {lastCommand && (
            <span className="last-command">Last: {lastCommand}</span>
          )}
        </>
      )}
    </div>
  );
}

// ============================================================================
// Text-to-Speech Hook
// ============================================================================

export function useTextToSpeech() {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voices, setVoices] = useState<Voice[]>([]);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  useEffect(() => {
    const loadVoices = () => {
      const availableVoices = speechSynthesis.getVoices();
      setVoices(availableVoices.map(v => ({
        id: v.voiceURI,
        name: v.name,
        language: v.lang,
        gender: v.name.toLowerCase().includes('female') ? 'female' : 'male',
      })));
    };

    loadVoices();
    speechSynthesis.onvoiceschanged = loadVoices;

    return () => {
      speechSynthesis.cancel();
    };
  }, []);

  const speak = useCallback((text: string, settings?: Partial<TtsSettings>) => {
    speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utteranceRef.current = utterance;

    if (settings?.voiceId) {
      const voice = speechSynthesis.getVoices().find(v => v.voiceURI === settings.voiceId);
      if (voice) utterance.voice = voice;
    }

    utterance.rate = settings?.speed ?? 1;
    utterance.pitch = settings?.pitch ?? 1;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    speechSynthesis.speak(utterance);
  }, []);

  const stop = useCallback(() => {
    speechSynthesis.cancel();
    setIsSpeaking(false);
  }, []);

  const pause = useCallback(() => {
    speechSynthesis.pause();
  }, []);

  const resume = useCallback(() => {
    speechSynthesis.resume();
  }, []);

  return {
    speak,
    stop,
    pause,
    resume,
    isSpeaking,
    voices,
  };
}

// ============================================================================
// Read Aloud Button
// ============================================================================

interface ReadAloudButtonProps {
  text: string;
  label?: string;
  className?: string;
}

export function ReadAloudButton({ text, label = 'Read aloud', className = '' }: ReadAloudButtonProps) {
  const { speak, stop, isSpeaking } = useTextToSpeech();

  const handleClick = () => {
    if (isSpeaking) {
      stop();
    } else {
      speak(text);
    }
  };

  return (
    <button
      className={`read-aloud-button ${isSpeaking ? 'speaking' : ''} ${className}`}
      onClick={handleClick}
      aria-label={isSpeaking ? 'Stop reading' : label}
      aria-pressed={isSpeaking}
    >
      {isSpeaking ? <StopIcon /> : <SpeakerIcon />}
      <span>{isSpeaking ? 'Stop' : label}</span>
    </button>
  );
}

// ============================================================================
// Email Reader Component
// ============================================================================

interface EmailReaderProps {
  emailId: string;
  subject: string;
  sender: string;
  body: string;
}

export function EmailReader({ emailId, subject, sender, body }: EmailReaderProps) {
  const { speak, stop, pause, resume, isSpeaking } = useTextToSpeech();
  const [isPaused, setIsPaused] = useState(false);

  const prepareText = () => {
    // Remove signatures and quoted text for cleaner reading
    let cleanBody = body
      .replace(/--\s*\n[\s\S]*$/, '') // Remove signature
      .replace(/^>.*$/gm, '') // Remove quoted lines
      .replace(/On .* wrote:/g, ''); // Remove quote headers

    return `Email from ${sender}. Subject: ${subject}. ${cleanBody}`;
  };

  const handlePlayPause = () => {
    if (isSpeaking) {
      if (isPaused) {
        resume();
        setIsPaused(false);
      } else {
        pause();
        setIsPaused(true);
      }
    } else {
      speak(prepareText());
      setIsPaused(false);
    }
  };

  return (
    <div className="email-reader" role="region" aria-label="Email audio reader">
      <div className="reader-controls">
        <button
          onClick={handlePlayPause}
          aria-label={isSpeaking ? (isPaused ? 'Resume' : 'Pause') : 'Read email'}
        >
          {isSpeaking ? (isPaused ? <PlayIcon /> : <PauseIcon />) : <PlayIcon />}
        </button>
        {isSpeaking && (
          <button onClick={stop} aria-label="Stop reading">
            <StopIcon />
          </button>
        )}
      </div>
      {isSpeaking && (
        <div className="reader-status" aria-live="polite">
          {isPaused ? 'Paused' : 'Reading...'}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Accessibility Settings Panel
// ============================================================================

interface AccessibilitySettingsProps {
  settings: AccessibilitySettings;
  onChange: (settings: AccessibilitySettings) => void;
}

export function AccessibilitySettingsPanel({ settings, onChange }: AccessibilitySettingsProps) {
  const updateSetting = <K extends keyof AccessibilitySettings>(
    key: K,
    value: AccessibilitySettings[K]
  ) => {
    onChange({ ...settings, [key]: value });
  };

  return (
    <div className="accessibility-settings" role="region" aria-label="Accessibility settings">
      <h2>Accessibility Settings</h2>

      <section className="settings-section">
        <h3>Display</h3>

        <div className="setting-item">
          <label htmlFor="high-contrast">
            <input
              type="checkbox"
              id="high-contrast"
              checked={settings.highContrast}
              onChange={(e) => updateSetting('highContrast', e.target.checked)}
            />
            High contrast mode
          </label>
          <p className="setting-description">
            Increase contrast for better visibility
          </p>
        </div>

        <div className="setting-item">
          <label htmlFor="font-size">Font size</label>
          <select
            id="font-size"
            value={settings.fontSize}
            onChange={(e) => updateSetting('fontSize', e.target.value as any)}
          >
            <option value="small">Small</option>
            <option value="medium">Medium</option>
            <option value="large">Large</option>
            <option value="xlarge">Extra Large</option>
          </select>
        </div>

        <div className="setting-item">
          <label htmlFor="dyslexia-font">
            <input
              type="checkbox"
              id="dyslexia-font"
              checked={settings.dyslexiaFriendlyFont}
              onChange={(e) => updateSetting('dyslexiaFriendlyFont', e.target.checked)}
            />
            Dyslexia-friendly font
          </label>
          <p className="setting-description">
            Use OpenDyslexic font for improved readability
          </p>
        </div>

        <div className="setting-item">
          <label htmlFor="color-blind-mode">Color blind mode</label>
          <select
            id="color-blind-mode"
            value={settings.colorBlindMode}
            onChange={(e) => updateSetting('colorBlindMode', e.target.value as any)}
          >
            <option value="none">None</option>
            <option value="protanopia">Protanopia (Red-blind)</option>
            <option value="deuteranopia">Deuteranopia (Green-blind)</option>
            <option value="tritanopia">Tritanopia (Blue-blind)</option>
          </select>
        </div>
      </section>

      <section className="settings-section">
        <h3>Motion & Animation</h3>

        <div className="setting-item">
          <label htmlFor="reduced-motion">
            <input
              type="checkbox"
              id="reduced-motion"
              checked={settings.reducedMotion}
              onChange={(e) => updateSetting('reducedMotion', e.target.checked)}
            />
            Reduce motion
          </label>
          <p className="setting-description">
            Minimize animations and transitions
          </p>
        </div>
      </section>

      <section className="settings-section">
        <h3>Navigation</h3>

        <div className="setting-item">
          <label htmlFor="keyboard-only">
            <input
              type="checkbox"
              id="keyboard-only"
              checked={settings.keyboardOnlyMode}
              onChange={(e) => updateSetting('keyboardOnlyMode', e.target.checked)}
            />
            Keyboard-only mode
          </label>
          <p className="setting-description">
            Enhanced focus indicators and keyboard navigation
          </p>
        </div>

        <div className="setting-item">
          <label htmlFor="screen-reader">
            <input
              type="checkbox"
              id="screen-reader"
              checked={settings.screenReaderOptimized}
              onChange={(e) => updateSetting('screenReaderOptimized', e.target.checked)}
            />
            Screen reader optimization
          </label>
          <p className="setting-description">
            Enhanced ARIA labels and semantic markup
          </p>
        </div>
      </section>
    </div>
  );
}

// ============================================================================
// Keyboard Shortcuts Panel
// ============================================================================

interface KeyboardShortcut {
  key: string;
  modifiers: string[];
  action: string;
  description: string;
  category: string;
}

interface KeyboardShortcutsPanelProps {
  shortcuts: KeyboardShortcut[];
  onCustomize?: (action: string, newKeyCombo: string) => void;
}

export function KeyboardShortcutsPanel({ shortcuts, onCustomize }: KeyboardShortcutsPanelProps) {
  const categories = [...new Set(shortcuts.map(s => s.category))];

  const formatKeyCombo = (shortcut: KeyboardShortcut) => {
    const parts = [...shortcut.modifiers, shortcut.key];
    return parts.join(' + ');
  };

  return (
    <div className="keyboard-shortcuts-panel" role="region" aria-label="Keyboard shortcuts">
      <h2>Keyboard Shortcuts</h2>

      {categories.map(category => (
        <section key={category} className="shortcut-category">
          <h3>{category}</h3>
          <table>
            <thead>
              <tr>
                <th>Action</th>
                <th>Shortcut</th>
              </tr>
            </thead>
            <tbody>
              {shortcuts
                .filter(s => s.category === category)
                .map(shortcut => (
                  <tr key={shortcut.action}>
                    <td>{shortcut.description}</td>
                    <td>
                      <kbd>{formatKeyCombo(shortcut)}</kbd>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </section>
      ))}

      <p className="shortcut-help">
        Press <kbd>?</kbd> to show this panel at any time
      </p>
    </div>
  );
}

// ============================================================================
// Voice Settings Panel
// ============================================================================

interface VoiceSettingsProps {
  ttsSettings: TtsSettings;
  onTtsChange: (settings: TtsSettings) => void;
  voiceCommandsEnabled: boolean;
  onVoiceCommandsChange: (enabled: boolean) => void;
  wakeWord: string;
  onWakeWordChange: (word: string) => void;
}

export function VoiceSettingsPanel({
  ttsSettings,
  onTtsChange,
  voiceCommandsEnabled,
  onVoiceCommandsChange,
  wakeWord,
  onWakeWordChange,
}: VoiceSettingsProps) {
  const { voices } = useTextToSpeech();

  const updateTts = <K extends keyof TtsSettings>(key: K, value: TtsSettings[K]) => {
    onTtsChange({ ...ttsSettings, [key]: value });
  };

  return (
    <div className="voice-settings" role="region" aria-label="Voice settings">
      <h2>Voice Settings</h2>

      <section className="settings-section">
        <h3>Text-to-Speech</h3>

        <div className="setting-item">
          <label htmlFor="tts-voice">Voice</label>
          <select
            id="tts-voice"
            value={ttsSettings.voiceId}
            onChange={(e) => updateTts('voiceId', e.target.value)}
          >
            {voices.map(voice => (
              <option key={voice.id} value={voice.id}>
                {voice.name} ({voice.language})
              </option>
            ))}
          </select>
        </div>

        <div className="setting-item">
          <label htmlFor="tts-speed">Speed: {ttsSettings.speed.toFixed(1)}x</label>
          <input
            type="range"
            id="tts-speed"
            min="0.5"
            max="2"
            step="0.1"
            value={ttsSettings.speed}
            onChange={(e) => updateTts('speed', parseFloat(e.target.value))}
          />
        </div>

        <div className="setting-item">
          <label htmlFor="tts-pitch">Pitch: {ttsSettings.pitch.toFixed(1)}</label>
          <input
            type="range"
            id="tts-pitch"
            min="0.5"
            max="2"
            step="0.1"
            value={ttsSettings.pitch}
            onChange={(e) => updateTts('pitch', parseFloat(e.target.value))}
          />
        </div>

        <div className="setting-item">
          <label>
            <input
              type="checkbox"
              checked={ttsSettings.autoReadNew}
              onChange={(e) => updateTts('autoReadNew', e.target.checked)}
            />
            Auto-read new emails
          </label>
        </div>

        <div className="setting-item">
          <label>
            <input
              type="checkbox"
              checked={ttsSettings.skipSignatures}
              onChange={(e) => updateTts('skipSignatures', e.target.checked)}
            />
            Skip signatures when reading
          </label>
        </div>

        <div className="setting-item">
          <label>
            <input
              type="checkbox"
              checked={ttsSettings.skipQuotedText}
              onChange={(e) => updateTts('skipQuotedText', e.target.checked)}
            />
            Skip quoted text when reading
          </label>
        </div>
      </section>

      <section className="settings-section">
        <h3>Voice Commands</h3>

        <div className="setting-item">
          <label>
            <input
              type="checkbox"
              checked={voiceCommandsEnabled}
              onChange={(e) => onVoiceCommandsChange(e.target.checked)}
            />
            Enable voice commands
          </label>
        </div>

        {voiceCommandsEnabled && (
          <div className="setting-item">
            <label htmlFor="wake-word">Wake word</label>
            <input
              type="text"
              id="wake-word"
              value={wakeWord}
              onChange={(e) => onWakeWordChange(e.target.value)}
              placeholder="e.g., Hey Mail"
            />
          </div>
        )}
      </section>
    </div>
  );
}

// ============================================================================
// Icon Components
// ============================================================================

function MicIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

function MicActiveIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

function MicOffIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="1" y1="1" x2="23" y2="23" />
      <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6" />
      <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

function SpeakerIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <polygon points="5 3 19 12 5 21 5 3" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="4" width="4" height="16" />
      <rect x="14" y="4" width="4" height="16" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <rect x="4" y="4" width="16" height="16" rx="2" />
    </svg>
  );
}

export default {
  useDictation,
  useTextToSpeech,
  DictationButton,
  VoiceCommandListener,
  ReadAloudButton,
  EmailReader,
  AccessibilitySettingsPanel,
  KeyboardShortcutsPanel,
  VoiceSettingsPanel,
};
