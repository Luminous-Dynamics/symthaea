// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * VoiceControl Component
 *
 * Voice command interface with visual feedback
 * Shows listening state, transcripts, and command help
 */

import React, { useEffect, useState } from 'react';
import { useVoiceCommands, VoiceCommand, CommandParams } from '../../hooks/useVoiceCommands';

interface VoiceControlProps {
  className?: string;
  onCommand?: (action: string, params: CommandParams) => void;
  compact?: boolean;
}

const CATEGORY_LABELS: Record<string, string> = {
  playback: 'Playback',
  volume: 'Volume',
  navigation: 'Navigation',
  search: 'Search',
  playlist: 'Playlist',
  dj: 'DJ',
  system: 'System',
};

const CATEGORY_ICONS: Record<string, string> = {
  playback: '▶',
  volume: '🔊',
  navigation: '🧭',
  search: '🔍',
  playlist: '📋',
  dj: '🎧',
  system: '⚙',
};

export function VoiceControl({ className = '', onCommand, compact = false }: VoiceControlProps) {
  const {
    isListening,
    isSupported,
    isAwake,
    transcript,
    interimTranscript,
    lastResult,
    history,
    error,
    confidence,
    wakeWord,
    startListening,
    stopListening,
    toggleListening,
    setWakeWord,
    registerHandlers,
    getAllCommands,
  } = useVoiceCommands();

  const [showHelp, setShowHelp] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  // Register command handler
  useEffect(() => {
    return registerHandlers({
      help: () => setShowHelp(true),
      // Forward all commands to parent
      ...Object.fromEntries(
        getAllCommands().map(cmd => [
          cmd.action,
          (params: CommandParams) => onCommand?.(cmd.action, params),
        ])
      ),
    });
  }, [registerHandlers, getAllCommands, onCommand]);

  if (!isSupported) {
    return (
      <div className={className} style={styles.unsupported}>
        <span style={styles.unsupportedIcon}>🎤</span>
        <p>Voice commands not supported in this browser</p>
        <p style={styles.unsupportedHint}>Try Chrome or Edge for voice control</p>
      </div>
    );
  }

  if (compact) {
    return (
      <button
        onClick={toggleListening}
        style={{
          ...styles.compactBtn,
          backgroundColor: isListening ? (isAwake ? '#10b981' : '#818cf8') : '#374151',
        }}
        title={isListening ? 'Stop listening' : 'Start voice control'}
      >
        <span style={styles.micIcon}>{isListening ? '🎤' : '🎙️'}</span>
        {isListening && (
          <span style={styles.pulseRing} />
        )}
      </button>
    );
  }

  const categories = ['playback', 'volume', 'navigation', 'search', 'playlist', 'dj', 'system'];
  const allCommands = getAllCommands();

  return (
    <div className={className} style={styles.container}>
      {/* Main Control */}
      <div style={styles.mainControl}>
        <button
          onClick={toggleListening}
          style={{
            ...styles.micButton,
            backgroundColor: isListening ? (isAwake ? '#10b981' : '#818cf8') : '#374151',
            boxShadow: isListening ? '0 0 30px rgba(129, 140, 248, 0.5)' : 'none',
          }}
        >
          <span style={styles.micLarge}>{isListening ? '🎤' : '🎙️'}</span>
          {isListening && (
            <>
              <span style={{ ...styles.pulseRingLarge, animationDelay: '0s' }} />
              <span style={{ ...styles.pulseRingLarge, animationDelay: '0.5s' }} />
            </>
          )}
        </button>

        <div style={styles.statusText}>
          {isListening ? (
            isAwake || !wakeWord ? (
              <span style={{ color: '#10b981' }}>Listening...</span>
            ) : (
              <span style={{ color: '#818cf8' }}>Say "{wakeWord}" to activate</span>
            )
          ) : (
            <span style={{ color: '#6b7280' }}>Click to start voice control</span>
          )}
        </div>

        {error && (
          <div style={styles.error}>
            <span>⚠️ {error}</span>
          </div>
        )}
      </div>

      {/* Transcript Display */}
      {isListening && (
        <div style={styles.transcriptBox}>
          {interimTranscript && (
            <p style={styles.interimText}>{interimTranscript}</p>
          )}
          {transcript && (
            <p style={styles.finalText}>{transcript}</p>
          )}
          {!interimTranscript && !transcript && (
            <p style={styles.placeholderText}>Speak a command...</p>
          )}
        </div>
      )}

      {/* Last Result */}
      {lastResult && (
        <div
          style={{
            ...styles.resultBox,
            borderColor: lastResult.success ? '#10b981' : '#ef4444',
          }}
        >
          <span style={styles.resultIcon}>
            {lastResult.success ? '✓' : '✗'}
          </span>
          <div style={styles.resultContent}>
            <span style={styles.resultTranscript}>"{lastResult.transcript}"</span>
            {lastResult.success && (
              <span style={styles.resultAction}>{lastResult.action}</span>
            )}
          </div>
          {confidence > 0 && (
            <span style={styles.confidence}>
              {Math.round(confidence * 100)}%
            </span>
          )}
        </div>
      )}

      {/* Wake Word Setting */}
      <div style={styles.wakeWordSection}>
        <label style={styles.wakeWordLabel}>Wake word:</label>
        <input
          type="text"
          value={wakeWord || ''}
          onChange={(e) => setWakeWord(e.target.value || null)}
          placeholder="None (always listening)"
          style={styles.wakeWordInput}
        />
      </div>

      {/* Help Toggle */}
      <button
        onClick={() => setShowHelp(!showHelp)}
        style={styles.helpToggle}
      >
        {showHelp ? 'Hide Commands' : 'Show Available Commands'}
      </button>

      {/* Command Help */}
      {showHelp && (
        <div style={styles.helpContainer}>
          {/* Category Tabs */}
          <div style={styles.categoryTabs}>
            {categories.map(cat => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(selectedCategory === cat ? null : cat)}
                style={{
                  ...styles.categoryTab,
                  backgroundColor: selectedCategory === cat ? '#374151' : 'transparent',
                }}
              >
                <span>{CATEGORY_ICONS[cat]}</span>
                <span>{CATEGORY_LABELS[cat]}</span>
              </button>
            ))}
          </div>

          {/* Commands List */}
          <div style={styles.commandsList}>
            {allCommands
              .filter(cmd => !selectedCategory || cmd.category === selectedCategory)
              .map((cmd, idx) => (
                <div key={idx} style={styles.commandItem}>
                  <span style={styles.commandPattern}>"{cmd.patterns[0]}"</span>
                  <span style={styles.commandDesc}>{cmd.description}</span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* History */}
      {history.length > 0 && (
        <div style={styles.historySection}>
          <h4 style={styles.historyTitle}>Recent Commands</h4>
          <div style={styles.historyList}>
            {history.slice(-5).reverse().map((result, idx) => (
              <div
                key={idx}
                style={{
                  ...styles.historyItem,
                  opacity: 1 - idx * 0.15,
                }}
              >
                <span style={{ color: result.success ? '#10b981' : '#ef4444' }}>
                  {result.success ? '✓' : '✗'}
                </span>
                <span>{result.transcript}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0% { transform: scale(1); opacity: 0.7; }
          100% { transform: scale(2); opacity: 0; }
        }
      `}</style>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: '#1f2937',
    borderRadius: '12px',
    padding: '24px',
    color: '#f9fafb',
    fontFamily: 'system-ui, -apple-system, sans-serif',
  },
  unsupported: {
    backgroundColor: '#1f2937',
    borderRadius: '12px',
    padding: '24px',
    textAlign: 'center',
    color: '#9ca3af',
  },
  unsupportedIcon: {
    fontSize: '48px',
    opacity: 0.5,
    display: 'block',
    marginBottom: '12px',
  },
  unsupportedHint: {
    fontSize: '12px',
    color: '#6b7280',
  },
  mainControl: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    marginBottom: '24px',
  },
  micButton: {
    width: '80px',
    height: '80px',
    borderRadius: '50%',
    border: 'none',
    cursor: 'pointer',
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.3s',
  },
  micLarge: {
    fontSize: '32px',
  },
  pulseRingLarge: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    borderRadius: '50%',
    border: '2px solid #818cf8',
    animation: 'pulse 1.5s ease-out infinite',
  },
  statusText: {
    marginTop: '16px',
    fontSize: '14px',
  },
  error: {
    marginTop: '12px',
    padding: '8px 16px',
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderRadius: '8px',
    color: '#ef4444',
    fontSize: '12px',
  },
  transcriptBox: {
    backgroundColor: '#111827',
    borderRadius: '8px',
    padding: '16px',
    marginBottom: '16px',
    minHeight: '60px',
  },
  interimText: {
    color: '#9ca3af',
    fontStyle: 'italic',
    margin: 0,
  },
  finalText: {
    color: '#f9fafb',
    margin: 0,
    fontWeight: 500,
  },
  placeholderText: {
    color: '#4b5563',
    margin: 0,
    fontStyle: 'italic',
  },
  resultBox: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '12px 16px',
    backgroundColor: '#111827',
    borderRadius: '8px',
    border: '1px solid',
    marginBottom: '16px',
  },
  resultIcon: {
    fontSize: '20px',
  },
  resultContent: {
    flex: 1,
  },
  resultTranscript: {
    display: 'block',
    fontSize: '14px',
    color: '#d1d5db',
  },
  resultAction: {
    display: 'block',
    fontSize: '12px',
    color: '#818cf8',
    marginTop: '4px',
  },
  confidence: {
    fontSize: '12px',
    color: '#6b7280',
    backgroundColor: '#374151',
    padding: '2px 8px',
    borderRadius: '12px',
  },
  wakeWordSection: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '16px',
  },
  wakeWordLabel: {
    fontSize: '12px',
    color: '#9ca3af',
  },
  wakeWordInput: {
    flex: 1,
    padding: '8px 12px',
    backgroundColor: '#111827',
    border: '1px solid #374151',
    borderRadius: '6px',
    color: '#f9fafb',
    fontSize: '14px',
  },
  helpToggle: {
    width: '100%',
    padding: '12px',
    backgroundColor: '#374151',
    border: 'none',
    borderRadius: '8px',
    color: '#d1d5db',
    fontSize: '14px',
    cursor: 'pointer',
    marginBottom: '16px',
  },
  helpContainer: {
    backgroundColor: '#111827',
    borderRadius: '8px',
    padding: '16px',
    marginBottom: '16px',
  },
  categoryTabs: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
    marginBottom: '16px',
  },
  categoryTab: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    padding: '6px 12px',
    borderRadius: '16px',
    border: '1px solid #374151',
    fontSize: '12px',
    cursor: 'pointer',
    color: '#d1d5db',
  },
  commandsList: {
    maxHeight: '300px',
    overflowY: 'auto',
  },
  commandItem: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '8px 0',
    borderBottom: '1px solid #1f2937',
  },
  commandPattern: {
    color: '#818cf8',
    fontSize: '13px',
    fontFamily: 'monospace',
  },
  commandDesc: {
    color: '#9ca3af',
    fontSize: '12px',
  },
  historySection: {
    marginTop: '16px',
  },
  historyTitle: {
    fontSize: '12px',
    color: '#6b7280',
    margin: '0 0 8px 0',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  historyList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  historyItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '12px',
    color: '#9ca3af',
  },
  compactBtn: {
    width: '40px',
    height: '40px',
    borderRadius: '50%',
    border: 'none',
    cursor: 'pointer',
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.3s',
  },
  micIcon: {
    fontSize: '16px',
  },
  pulseRing: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    borderRadius: '50%',
    border: '2px solid currentColor',
    animation: 'pulse 1.5s ease-out infinite',
  },
};

export default VoiceControl;
