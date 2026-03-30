// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Voice & Accessibility Settings Component
 *
 * Comprehensive settings UI for voice commands and accessibility features.
 */

import React, { useState, useEffect } from 'react';
import {
  useVoiceCommands,
  useAccessibility,
  useKeyboardShortcuts,
  useTextToSpeech,
} from '../../lib/voice-accessibility';

// ============================================================================
// Types
// ============================================================================

interface ShortcutConfig {
  id: string;
  label: string;
  description: string;
  keys: string[];
  category: 'navigation' | 'actions' | 'compose' | 'search';
}

// ============================================================================
// Voice Settings Section
// ============================================================================

const VoiceSettingsSection: React.FC = () => {
  const {
    isListening,
    isSupported,
    start,
    stop,
    confidence,
  } = useVoiceCommands();

  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [sensitivity, setSensitivity] = useState(0.7);
  const [wakeWord, setWakeWord] = useState('hey mycelix');
  const [showConfidence, setShowConfidence] = useState(true);

  const handleToggleVoice = () => {
    if (voiceEnabled) {
      stop();
      setVoiceEnabled(false);
    } else {
      start();
      setVoiceEnabled(true);
    }
  };

  if (!isSupported) {
    return (
      <div className="settings-section">
        <h3>Voice Commands</h3>
        <div className="warning-banner">
          <span className="icon">⚠️</span>
          <p>Voice commands are not supported in your browser. Try using Chrome or Edge.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="settings-section">
      <h3>Voice Commands</h3>

      <div className="setting-row">
        <div className="setting-info">
          <label>Enable Voice Commands</label>
          <p className="description">Control Mycelix Mail with your voice</p>
        </div>
        <label className="toggle-switch">
          <input
            type="checkbox"
            checked={voiceEnabled}
            onChange={handleToggleVoice}
          />
          <span className="slider"></span>
        </label>
      </div>

      {voiceEnabled && (
        <>
          <div className="setting-row">
            <div className="setting-info">
              <label>Listening Status</label>
              <p className="description">
                {isListening ? 'Actively listening for commands' : 'Voice recognition paused'}
              </p>
            </div>
            <span className={`status-indicator ${isListening ? 'active' : 'inactive'}`}>
              {isListening ? '🎤 Listening' : '⏸️ Paused'}
            </span>
          </div>

          <div className="setting-row">
            <div className="setting-info">
              <label>Wake Word</label>
              <p className="description">Say this phrase to activate voice commands</p>
            </div>
            <input
              type="text"
              value={wakeWord}
              onChange={(e) => setWakeWord(e.target.value)}
              placeholder="hey mycelix"
              className="text-input"
            />
          </div>

          <div className="setting-row">
            <div className="setting-info">
              <label>Recognition Sensitivity</label>
              <p className="description">Higher values require clearer speech</p>
            </div>
            <div className="slider-container">
              <input
                type="range"
                min="0.3"
                max="1"
                step="0.1"
                value={sensitivity}
                onChange={(e) => setSensitivity(parseFloat(e.target.value))}
              />
              <span className="slider-value">{Math.round(sensitivity * 100)}%</span>
            </div>
          </div>

          <div className="setting-row">
            <div className="setting-info">
              <label>Show Confidence Score</label>
              <p className="description">Display how confident the system is in recognized commands</p>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={showConfidence}
                onChange={(e) => setShowConfidence(e.target.checked)}
              />
              <span className="slider"></span>
            </label>
          </div>

          {showConfidence && confidence !== null && (
            <div className="confidence-display">
              <span>Last recognition confidence:</span>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{ width: `${confidence * 100}%` }}
                />
              </div>
              <span>{Math.round(confidence * 100)}%</span>
            </div>
          )}

          <div className="voice-commands-list">
            <h4>Available Commands</h4>
            <div className="commands-grid">
              <div className="command-group">
                <h5>Navigation</h5>
                <ul>
                  <li><code>go to inbox</code></li>
                  <li><code>go to sent</code></li>
                  <li><code>go to drafts</code></li>
                  <li><code>open settings</code></li>
                </ul>
              </div>
              <div className="command-group">
                <h5>Actions</h5>
                <ul>
                  <li><code>compose email</code></li>
                  <li><code>reply</code></li>
                  <li><code>archive</code></li>
                  <li><code>delete</code></li>
                </ul>
              </div>
              <div className="command-group">
                <h5>Reading</h5>
                <ul>
                  <li><code>read email</code></li>
                  <li><code>next email</code></li>
                  <li><code>previous email</code></li>
                  <li><code>stop reading</code></li>
                </ul>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

// ============================================================================
// Text-to-Speech Settings Section
// ============================================================================

const TextToSpeechSection: React.FC = () => {
  const { speak, stop, isSpeaking, voices } = useTextToSpeech();
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [rate, setRate] = useState(1);
  const [pitch, setPitch] = useState(1);
  const [volume, setVolume] = useState(1);

  const handleTestVoice = () => {
    if (isSpeaking) {
      stop();
    } else {
      speak('Hello! This is a test of the Mycelix Mail text to speech system.', {
        voice: selectedVoice,
        rate,
        pitch,
        volume,
      });
    }
  };

  return (
    <div className="settings-section">
      <h3>Text-to-Speech</h3>

      <div className="setting-row">
        <div className="setting-info">
          <label>Enable Text-to-Speech</label>
          <p className="description">Have emails and notifications read aloud</p>
        </div>
        <label className="toggle-switch">
          <input
            type="checkbox"
            checked={ttsEnabled}
            onChange={(e) => setTtsEnabled(e.target.checked)}
          />
          <span className="slider"></span>
        </label>
      </div>

      {ttsEnabled && (
        <>
          <div className="setting-row">
            <div className="setting-info">
              <label>Voice</label>
              <p className="description">Select the voice for reading</p>
            </div>
            <select
              value={selectedVoice}
              onChange={(e) => setSelectedVoice(e.target.value)}
              className="select-input"
            >
              <option value="">System Default</option>
              {voices.map((voice) => (
                <option key={voice.voiceURI} value={voice.voiceURI}>
                  {voice.name} ({voice.lang})
                </option>
              ))}
            </select>
          </div>

          <div className="setting-row">
            <div className="setting-info">
              <label>Speaking Rate</label>
              <p className="description">How fast the voice speaks</p>
            </div>
            <div className="slider-container">
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={rate}
                onChange={(e) => setRate(parseFloat(e.target.value))}
              />
              <span className="slider-value">{rate}x</span>
            </div>
          </div>

          <div className="setting-row">
            <div className="setting-info">
              <label>Pitch</label>
              <p className="description">Voice pitch level</p>
            </div>
            <div className="slider-container">
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={pitch}
                onChange={(e) => setPitch(parseFloat(e.target.value))}
              />
              <span className="slider-value">{pitch}</span>
            </div>
          </div>

          <div className="setting-row">
            <div className="setting-info">
              <label>Volume</label>
              <p className="description">Speech volume level</p>
            </div>
            <div className="slider-container">
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={volume}
                onChange={(e) => setVolume(parseFloat(e.target.value))}
              />
              <span className="slider-value">{Math.round(volume * 100)}%</span>
            </div>
          </div>

          <button
            onClick={handleTestVoice}
            className={`test-button ${isSpeaking ? 'speaking' : ''}`}
          >
            {isSpeaking ? '⏹️ Stop' : '▶️ Test Voice'}
          </button>
        </>
      )}
    </div>
  );
};

// ============================================================================
// Accessibility Settings Section
// ============================================================================

const AccessibilitySection: React.FC = () => {
  const { settings, updateSettings } = useAccessibility();

  return (
    <div className="settings-section">
      <h3>Visual Accessibility</h3>

      <div className="setting-row">
        <div className="setting-info">
          <label>High Contrast Mode</label>
          <p className="description">Increase contrast for better visibility</p>
        </div>
        <label className="toggle-switch">
          <input
            type="checkbox"
            checked={settings.highContrast}
            onChange={(e) => updateSettings({ highContrast: e.target.checked })}
          />
          <span className="slider"></span>
        </label>
      </div>

      <div className="setting-row">
        <div className="setting-info">
          <label>Reduce Motion</label>
          <p className="description">Minimize animations and transitions</p>
        </div>
        <label className="toggle-switch">
          <input
            type="checkbox"
            checked={settings.reduceMotion}
            onChange={(e) => updateSettings({ reduceMotion: e.target.checked })}
          />
          <span className="slider"></span>
        </label>
      </div>

      <div className="setting-row">
        <div className="setting-info">
          <label>Large Text</label>
          <p className="description">Increase text size throughout the application</p>
        </div>
        <label className="toggle-switch">
          <input
            type="checkbox"
            checked={settings.largeText}
            onChange={(e) => updateSettings({ largeText: e.target.checked })}
          />
          <span className="slider"></span>
        </label>
      </div>

      <div className="setting-row">
        <div className="setting-info">
          <label>Font Size</label>
          <p className="description">Adjust base font size</p>
        </div>
        <div className="slider-container">
          <input
            type="range"
            min="12"
            max="24"
            step="1"
            value={settings.fontSize}
            onChange={(e) => updateSettings({ fontSize: parseInt(e.target.value) })}
          />
          <span className="slider-value">{settings.fontSize}px</span>
        </div>
      </div>

      <div className="setting-row">
        <div className="setting-info">
          <label>Dyslexia-Friendly Font</label>
          <p className="description">Use OpenDyslexic font for easier reading</p>
        </div>
        <label className="toggle-switch">
          <input
            type="checkbox"
            checked={settings.dyslexiaFont}
            onChange={(e) => updateSettings({ dyslexiaFont: e.target.checked })}
          />
          <span className="slider"></span>
        </label>
      </div>

      <div className="setting-row">
        <div className="setting-info">
          <label>Focus Indicators</label>
          <p className="description">Enhanced focus outlines for keyboard navigation</p>
        </div>
        <label className="toggle-switch">
          <input
            type="checkbox"
            checked={settings.focusIndicators}
            onChange={(e) => updateSettings({ focusIndicators: e.target.checked })}
          />
          <span className="slider"></span>
        </label>
      </div>

      <div className="setting-row">
        <div className="setting-info">
          <label>Screen Reader Optimization</label>
          <p className="description">Optimize for screen reader compatibility</p>
        </div>
        <label className="toggle-switch">
          <input
            type="checkbox"
            checked={settings.screenReaderOptimized}
            onChange={(e) => updateSettings({ screenReaderOptimized: e.target.checked })}
          />
          <span className="slider"></span>
        </label>
      </div>

      <div className="setting-row">
        <div className="setting-info">
          <label>Color Theme</label>
          <p className="description">Choose a color scheme</p>
        </div>
        <div className="color-theme-options">
          {['default', 'dark', 'sepia', 'blue-light'].map((theme) => (
            <button
              key={theme}
              className={`theme-button ${settings.colorTheme === theme ? 'active' : ''}`}
              onClick={() => updateSettings({ colorTheme: theme })}
            >
              <span className={`theme-preview theme-${theme}`}></span>
              <span className="theme-label">
                {theme.charAt(0).toUpperCase() + theme.slice(1).replace('-', ' ')}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Keyboard Shortcuts Section
// ============================================================================

const KeyboardShortcutsSection: React.FC = () => {
  const { shortcuts, updateShortcut, resetToDefaults } = useKeyboardShortcuts();
  const [editingShortcut, setEditingShortcut] = useState<string | null>(null);
  const [recordedKeys, setRecordedKeys] = useState<string[]>([]);

  const defaultShortcuts: ShortcutConfig[] = [
    // Navigation
    { id: 'go-inbox', label: 'Go to Inbox', description: 'Navigate to inbox', keys: ['g', 'i'], category: 'navigation' },
    { id: 'go-sent', label: 'Go to Sent', description: 'Navigate to sent', keys: ['g', 's'], category: 'navigation' },
    { id: 'go-drafts', label: 'Go to Drafts', description: 'Navigate to drafts', keys: ['g', 'd'], category: 'navigation' },
    { id: 'go-starred', label: 'Go to Starred', description: 'Navigate to starred', keys: ['g', 't'], category: 'navigation' },
    // Actions
    { id: 'compose', label: 'Compose', description: 'Start new email', keys: ['c'], category: 'actions' },
    { id: 'reply', label: 'Reply', description: 'Reply to email', keys: ['r'], category: 'actions' },
    { id: 'reply-all', label: 'Reply All', description: 'Reply to all', keys: ['a'], category: 'actions' },
    { id: 'forward', label: 'Forward', description: 'Forward email', keys: ['f'], category: 'actions' },
    { id: 'archive', label: 'Archive', description: 'Archive email', keys: ['e'], category: 'actions' },
    { id: 'delete', label: 'Delete', description: 'Delete email', keys: ['#'], category: 'actions' },
    { id: 'mark-read', label: 'Mark Read', description: 'Mark as read', keys: ['Shift', 'i'], category: 'actions' },
    { id: 'mark-unread', label: 'Mark Unread', description: 'Mark as unread', keys: ['Shift', 'u'], category: 'actions' },
    // Search
    { id: 'search', label: 'Search', description: 'Focus search bar', keys: ['/'], category: 'search' },
    { id: 'search-from', label: 'Search From', description: 'Search by sender', keys: ['Shift', '/'], category: 'search' },
  ];

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!editingShortcut) return;

    e.preventDefault();
    const key = e.key;

    if (key === 'Escape') {
      setEditingShortcut(null);
      setRecordedKeys([]);
      return;
    }

    if (key === 'Enter' && recordedKeys.length > 0) {
      updateShortcut(editingShortcut, recordedKeys);
      setEditingShortcut(null);
      setRecordedKeys([]);
      return;
    }

    const newKeys = [...recordedKeys];
    if (e.ctrlKey && !newKeys.includes('Ctrl')) newKeys.push('Ctrl');
    if (e.altKey && !newKeys.includes('Alt')) newKeys.push('Alt');
    if (e.shiftKey && !newKeys.includes('Shift')) newKeys.push('Shift');
    if (e.metaKey && !newKeys.includes('Cmd')) newKeys.push('Cmd');

    if (!['Control', 'Alt', 'Shift', 'Meta'].includes(key)) {
      newKeys.push(key.length === 1 ? key.toUpperCase() : key);
    }

    setRecordedKeys(newKeys);
  };

  const groupedShortcuts = defaultShortcuts.reduce((acc, shortcut) => {
    if (!acc[shortcut.category]) acc[shortcut.category] = [];
    acc[shortcut.category].push(shortcut);
    return acc;
  }, {} as Record<string, ShortcutConfig[]>);

  return (
    <div className="settings-section" onKeyDown={handleKeyDown}>
      <div className="section-header">
        <h3>Keyboard Shortcuts</h3>
        <button onClick={resetToDefaults} className="reset-button">
          Reset to Defaults
        </button>
      </div>

      {Object.entries(groupedShortcuts).map(([category, categoryShortcuts]) => (
        <div key={category} className="shortcut-category">
          <h4>{category.charAt(0).toUpperCase() + category.slice(1)}</h4>
          <div className="shortcuts-list">
            {categoryShortcuts.map((shortcut) => {
              const currentKeys = shortcuts[shortcut.id] || shortcut.keys;
              const isEditing = editingShortcut === shortcut.id;

              return (
                <div
                  key={shortcut.id}
                  className={`shortcut-row ${isEditing ? 'editing' : ''}`}
                >
                  <div className="shortcut-info">
                    <span className="shortcut-label">{shortcut.label}</span>
                    <span className="shortcut-description">{shortcut.description}</span>
                  </div>
                  <div className="shortcut-keys">
                    {isEditing ? (
                      <div className="key-recorder">
                        {recordedKeys.length > 0 ? (
                          recordedKeys.map((key, i) => (
                            <kbd key={i}>{key}</kbd>
                          ))
                        ) : (
                          <span className="recording-hint">Press keys...</span>
                        )}
                      </div>
                    ) : (
                      <>
                        {currentKeys.map((key, i) => (
                          <React.Fragment key={i}>
                            <kbd>{key}</kbd>
                            {i < currentKeys.length - 1 && <span>+</span>}
                          </React.Fragment>
                        ))}
                      </>
                    )}
                    <button
                      className="edit-shortcut-btn"
                      onClick={() => {
                        if (isEditing) {
                          setEditingShortcut(null);
                          setRecordedKeys([]);
                        } else {
                          setEditingShortcut(shortcut.id);
                          setRecordedKeys([]);
                        }
                      }}
                    >
                      {isEditing ? 'Cancel' : 'Edit'}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}

      {editingShortcut && (
        <div className="editing-instructions">
          Press the key combination you want to use, then press <kbd>Enter</kbd> to save or <kbd>Escape</kbd> to cancel.
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const VoiceAccessibilitySettings: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'voice' | 'tts' | 'visual' | 'keyboard'>('voice');

  return (
    <div className="voice-accessibility-settings">
      <div className="settings-header">
        <h2>Voice & Accessibility</h2>
        <p>Customize how you interact with Mycelix Mail</p>
      </div>

      <div className="settings-tabs">
        <button
          className={`tab ${activeTab === 'voice' ? 'active' : ''}`}
          onClick={() => setActiveTab('voice')}
        >
          🎤 Voice Commands
        </button>
        <button
          className={`tab ${activeTab === 'tts' ? 'active' : ''}`}
          onClick={() => setActiveTab('tts')}
        >
          🔊 Text-to-Speech
        </button>
        <button
          className={`tab ${activeTab === 'visual' ? 'active' : ''}`}
          onClick={() => setActiveTab('visual')}
        >
          👁️ Visual
        </button>
        <button
          className={`tab ${activeTab === 'keyboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('keyboard')}
        >
          ⌨️ Keyboard
        </button>
      </div>

      <div className="settings-content">
        {activeTab === 'voice' && <VoiceSettingsSection />}
        {activeTab === 'tts' && <TextToSpeechSection />}
        {activeTab === 'visual' && <AccessibilitySection />}
        {activeTab === 'keyboard' && <KeyboardShortcutsSection />}
      </div>

      <style>{`
        .voice-accessibility-settings {
          max-width: 800px;
          margin: 0 auto;
          padding: 24px;
        }

        .settings-header {
          margin-bottom: 24px;
        }

        .settings-header h2 {
          margin: 0 0 8px 0;
          font-size: 24px;
        }

        .settings-header p {
          margin: 0;
          color: #666;
        }

        .settings-tabs {
          display: flex;
          gap: 4px;
          border-bottom: 1px solid #e0e0e0;
          margin-bottom: 24px;
        }

        .settings-tabs .tab {
          padding: 12px 20px;
          border: none;
          background: none;
          cursor: pointer;
          font-size: 14px;
          color: #666;
          border-bottom: 2px solid transparent;
          transition: all 0.2s;
        }

        .settings-tabs .tab:hover {
          color: #333;
          background: #f5f5f5;
        }

        .settings-tabs .tab.active {
          color: #2196F3;
          border-bottom-color: #2196F3;
        }

        .settings-section {
          background: white;
          border-radius: 8px;
          padding: 24px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .settings-section h3 {
          margin: 0 0 20px 0;
          font-size: 18px;
        }

        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .section-header h3 {
          margin: 0;
        }

        .setting-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px 0;
          border-bottom: 1px solid #f0f0f0;
        }

        .setting-row:last-child {
          border-bottom: none;
        }

        .setting-info {
          flex: 1;
        }

        .setting-info label {
          display: block;
          font-weight: 500;
          margin-bottom: 4px;
        }

        .setting-info .description {
          margin: 0;
          font-size: 13px;
          color: #666;
        }

        .toggle-switch {
          position: relative;
          display: inline-block;
          width: 48px;
          height: 24px;
        }

        .toggle-switch input {
          opacity: 0;
          width: 0;
          height: 0;
        }

        .toggle-switch .slider {
          position: absolute;
          cursor: pointer;
          inset: 0;
          background-color: #ccc;
          border-radius: 24px;
          transition: 0.3s;
        }

        .toggle-switch .slider::before {
          position: absolute;
          content: '';
          height: 18px;
          width: 18px;
          left: 3px;
          bottom: 3px;
          background-color: white;
          border-radius: 50%;
          transition: 0.3s;
        }

        .toggle-switch input:checked + .slider {
          background-color: #2196F3;
        }

        .toggle-switch input:checked + .slider::before {
          transform: translateX(24px);
        }

        .slider-container {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .slider-container input[type="range"] {
          width: 150px;
        }

        .slider-value {
          min-width: 50px;
          text-align: right;
          font-size: 14px;
          color: #666;
        }

        .text-input, .select-input {
          padding: 8px 12px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 14px;
          min-width: 200px;
        }

        .status-indicator {
          padding: 6px 12px;
          border-radius: 16px;
          font-size: 13px;
        }

        .status-indicator.active {
          background: #e8f5e9;
          color: #2e7d32;
        }

        .status-indicator.inactive {
          background: #f5f5f5;
          color: #666;
        }

        .test-button {
          margin-top: 16px;
          padding: 10px 20px;
          background: #2196F3;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
        }

        .test-button:hover {
          background: #1976D2;
        }

        .test-button.speaking {
          background: #f44336;
        }

        .reset-button {
          padding: 8px 16px;
          background: #f5f5f5;
          border: 1px solid #ddd;
          border-radius: 4px;
          cursor: pointer;
          font-size: 13px;
        }

        .reset-button:hover {
          background: #e0e0e0;
        }

        .voice-commands-list {
          margin-top: 24px;
          padding-top: 24px;
          border-top: 1px solid #e0e0e0;
        }

        .voice-commands-list h4 {
          margin: 0 0 16px 0;
        }

        .commands-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 20px;
        }

        .command-group h5 {
          margin: 0 0 8px 0;
          font-size: 13px;
          color: #666;
          text-transform: uppercase;
        }

        .command-group ul {
          list-style: none;
          padding: 0;
          margin: 0;
        }

        .command-group li {
          padding: 4px 0;
        }

        .command-group code {
          background: #f5f5f5;
          padding: 2px 8px;
          border-radius: 4px;
          font-size: 13px;
        }

        .color-theme-options {
          display: flex;
          gap: 12px;
        }

        .theme-button {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 8px;
          padding: 12px;
          border: 2px solid transparent;
          border-radius: 8px;
          background: none;
          cursor: pointer;
        }

        .theme-button:hover {
          background: #f5f5f5;
        }

        .theme-button.active {
          border-color: #2196F3;
        }

        .theme-preview {
          width: 48px;
          height: 32px;
          border-radius: 4px;
          border: 1px solid #ddd;
        }

        .theme-default { background: linear-gradient(to bottom, #fff 50%, #f5f5f5 50%); }
        .theme-dark { background: linear-gradient(to bottom, #1e1e1e 50%, #2d2d2d 50%); }
        .theme-sepia { background: linear-gradient(to bottom, #f4ecd8 50%, #e8dcc8 50%); }
        .theme-blue-light { background: linear-gradient(to bottom, #e3f2fd 50%, #bbdefb 50%); }

        .theme-label {
          font-size: 12px;
          color: #666;
        }

        .shortcut-category {
          margin-bottom: 24px;
        }

        .shortcut-category h4 {
          margin: 0 0 12px 0;
          font-size: 14px;
          color: #666;
          text-transform: uppercase;
        }

        .shortcuts-list {
          background: #f9f9f9;
          border-radius: 8px;
          overflow: hidden;
        }

        .shortcut-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 16px;
          border-bottom: 1px solid #eee;
        }

        .shortcut-row:last-child {
          border-bottom: none;
        }

        .shortcut-row.editing {
          background: #e3f2fd;
        }

        .shortcut-info {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .shortcut-label {
          font-weight: 500;
        }

        .shortcut-description {
          font-size: 12px;
          color: #666;
        }

        .shortcut-keys {
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .shortcut-keys kbd {
          display: inline-block;
          padding: 4px 8px;
          background: white;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 12px;
          font-family: monospace;
          box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }

        .edit-shortcut-btn {
          margin-left: 12px;
          padding: 4px 12px;
          background: none;
          border: 1px solid #ddd;
          border-radius: 4px;
          cursor: pointer;
          font-size: 12px;
        }

        .edit-shortcut-btn:hover {
          background: #f5f5f5;
        }

        .key-recorder {
          display: flex;
          align-items: center;
          gap: 4px;
          min-width: 120px;
        }

        .recording-hint {
          color: #666;
          font-style: italic;
          font-size: 13px;
        }

        .editing-instructions {
          margin-top: 16px;
          padding: 12px;
          background: #fff3e0;
          border-radius: 6px;
          font-size: 13px;
          color: #e65100;
        }

        .editing-instructions kbd {
          display: inline-block;
          padding: 2px 6px;
          background: white;
          border: 1px solid #ddd;
          border-radius: 3px;
          font-size: 11px;
        }

        .warning-banner {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 16px;
          background: #fff3e0;
          border-radius: 8px;
        }

        .warning-banner .icon {
          font-size: 24px;
        }

        .warning-banner p {
          margin: 0;
          color: #e65100;
        }

        .confidence-display {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px;
          background: #f5f5f5;
          border-radius: 6px;
          margin-top: 12px;
        }

        .confidence-bar {
          flex: 1;
          height: 8px;
          background: #e0e0e0;
          border-radius: 4px;
          overflow: hidden;
        }

        .confidence-fill {
          height: 100%;
          background: #4caf50;
          transition: width 0.3s;
        }
      `}</style>
    </div>
  );
};

export default VoiceAccessibilitySettings;
