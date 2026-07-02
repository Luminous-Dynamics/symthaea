// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * BroadcastStudio Component
 *
 * Full broadcasting interface for live streaming DJ sets
 * Includes audio monitoring, chat, and listener management
 */

import React, { useState, useRef, useCallback } from 'react';
import { useBroadcast, BroadcastConfig } from '../../hooks/useBroadcast';

interface BroadcastStudioProps {
  className?: string;
  audioSource?: MediaStream | AudioNode;
  onBroadcastEnd?: (recording: Blob | null) => void;
}

export function BroadcastStudio({ className = '', audioSource, onBroadcastEnd }: BroadcastStudioProps) {
  const {
    state,
    listeners,
    stats,
    messages,
    audioLevel,
    error,
    startBroadcast,
    stopBroadcast,
    sendMessage,
    sendReaction,
    kickListener,
    updateConfig,
    getShareUrl,
    isLive,
  } = useBroadcast();

  const [config, setConfig] = useState<BroadcastConfig>({
    title: '',
    description: '',
    genre: '',
    isPrivate: false,
    allowChat: true,
    recordBroadcast: false,
  });
  const [chatInput, setChatInput] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [copied, setCopied] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle start broadcast
  const handleStart = async () => {
    try {
      let source: MediaStream;

      if (audioSource) {
        if (audioSource instanceof MediaStream) {
          source = audioSource;
        } else {
          const ctx = new AudioContext();
          const dest = ctx.createMediaStreamDestination();
          audioSource.connect(dest);
          source = dest.stream;
        }
      } else {
        // Request microphone access as fallback
        source = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
      }

      await startBroadcast(source, config);
    } catch (err) {
      console.error('Failed to start broadcast:', err);
    }
  };

  // Handle stop broadcast
  const handleStop = async () => {
    const recording = await stopBroadcast();
    onBroadcastEnd?.(recording);
  };

  // Handle send message
  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (chatInput.trim()) {
      sendMessage(chatInput.trim());
      setChatInput('');
    }
  };

  // Handle copy share link
  const handleCopyLink = () => {
    const url = getShareUrl();
    if (url) {
      navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  // Format duration
  const formatDuration = (seconds: number): string => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hrs > 0) {
      return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Format bytes
  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className={className} style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <h2 style={styles.title}>
          {isLive ? (
            <>
              <span style={styles.liveIndicator} />
              Live Broadcast
            </>
          ) : (
            'Broadcast Studio'
          )}
        </h2>
        {isLive && (
          <div style={styles.liveStats}>
            <span style={styles.statItem}>
              <span style={styles.statIcon}>👥</span>
              {stats.listeners}
            </span>
            <span style={styles.statItem}>
              <span style={styles.statIcon}>⏱</span>
              {formatDuration(stats.duration)}
            </span>
          </div>
        )}
      </div>

      {error && (
        <div style={styles.error}>
          <span>⚠️ {error}</span>
        </div>
      )}

      {/* Pre-broadcast Setup */}
      {!isLive && state.status !== 'preparing' && (
        <div style={styles.setupSection}>
          <div style={styles.inputGroup}>
            <label style={styles.label}>Stream Title</label>
            <input
              type="text"
              value={config.title}
              onChange={(e) => setConfig(prev => ({ ...prev, title: e.target.value }))}
              placeholder="My Live DJ Set"
              style={styles.input}
            />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Description</label>
            <textarea
              value={config.description}
              onChange={(e) => setConfig(prev => ({ ...prev, description: e.target.value }))}
              placeholder="What's this stream about?"
              style={{ ...styles.input, minHeight: '80px', resize: 'vertical' }}
            />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Genre</label>
            <select
              value={config.genre}
              onChange={(e) => setConfig(prev => ({ ...prev, genre: e.target.value }))}
              style={styles.input}
            >
              <option value="">Select genre...</option>
              <option value="house">House</option>
              <option value="techno">Techno</option>
              <option value="drum-and-bass">Drum & Bass</option>
              <option value="trance">Trance</option>
              <option value="hip-hop">Hip-Hop</option>
              <option value="ambient">Ambient</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div style={styles.toggleGroup}>
            <label style={styles.toggleLabel}>
              <input
                type="checkbox"
                checked={config.isPrivate}
                onChange={(e) => setConfig(prev => ({ ...prev, isPrivate: e.target.checked }))}
                style={styles.checkbox}
              />
              Private stream (invite only)
            </label>

            <label style={styles.toggleLabel}>
              <input
                type="checkbox"
                checked={config.allowChat}
                onChange={(e) => setConfig(prev => ({ ...prev, allowChat: e.target.checked }))}
                style={styles.checkbox}
              />
              Allow chat
            </label>

            <label style={styles.toggleLabel}>
              <input
                type="checkbox"
                checked={config.recordBroadcast}
                onChange={(e) => setConfig(prev => ({ ...prev, recordBroadcast: e.target.checked }))}
                style={styles.checkbox}
              />
              Record broadcast
            </label>
          </div>

          <button onClick={handleStart} style={styles.goLiveBtn}>
            <span style={styles.goLiveIcon}>📡</span>
            Go Live
          </button>
        </div>
      )}

      {/* Preparing State */}
      {state.status === 'preparing' && (
        <div style={styles.preparingSection}>
          <div style={styles.spinner} />
          <p>Preparing broadcast...</p>
        </div>
      )}

      {/* Live Broadcast Controls */}
      {isLive && (
        <>
          {/* Audio Level Meter */}
          <div style={styles.meterSection}>
            <div style={styles.meterLabel}>Audio Level</div>
            <div style={styles.meterTrack}>
              <div
                style={{
                  ...styles.meterFill,
                  width: `${audioLevel * 100}%`,
                  backgroundColor: audioLevel > 0.9 ? '#ef4444' : audioLevel > 0.7 ? '#f59e0b' : '#10b981',
                }}
              />
            </div>
          </div>

          {/* Share Section */}
          <div style={styles.shareSection}>
            <div style={styles.shareUrl}>
              <input
                type="text"
                value={getShareUrl() || ''}
                readOnly
                style={styles.shareInput}
              />
              <button onClick={handleCopyLink} style={styles.copyBtn}>
                {copied ? '✓ Copied' : 'Copy'}
              </button>
            </div>
          </div>

          {/* Stats Grid */}
          <div style={styles.statsGrid}>
            <div style={styles.statCard}>
              <span style={styles.statValue}>{stats.listeners}</span>
              <span style={styles.statLabel}>Listeners</span>
            </div>
            <div style={styles.statCard}>
              <span style={styles.statValue}>{stats.peakListeners}</span>
              <span style={styles.statLabel}>Peak</span>
            </div>
            <div style={styles.statCard}>
              <span style={styles.statValue}>{formatDuration(stats.duration)}</span>
              <span style={styles.statLabel}>Duration</span>
            </div>
            <div style={styles.statCard}>
              <span style={styles.statValue}>{formatBytes(stats.bytesTransferred)}</span>
              <span style={styles.statLabel}>Data</span>
            </div>
          </div>

          {/* Reactions */}
          <div style={styles.reactionsSection}>
            <div style={styles.reactionButtons}>
              {['🔥', '❤️', '🎵', '🙌', '🎉'].map(reaction => (
                <button
                  key={reaction}
                  onClick={() => sendReaction(reaction)}
                  style={styles.reactionBtn}
                >
                  {reaction}
                  {stats.reactions[reaction] ? (
                    <span style={styles.reactionCount}>{stats.reactions[reaction]}</span>
                  ) : null}
                </button>
              ))}
            </div>
          </div>

          {/* Listeners List */}
          {listeners.length > 0 && (
            <div style={styles.listenersSection}>
              <h4 style={styles.sectionTitle}>Listeners ({listeners.length})</h4>
              <div style={styles.listenersList}>
                {listeners.map(listener => (
                  <div key={listener.id} style={styles.listenerItem}>
                    <span style={styles.listenerName}>{listener.name}</span>
                    <button
                      onClick={() => kickListener(listener.id)}
                      style={styles.kickBtn}
                      title="Remove listener"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Chat */}
          {config.allowChat && (
            <div style={styles.chatSection}>
              <h4 style={styles.sectionTitle}>Chat</h4>
              <div style={styles.chatMessages}>
                {messages.map(msg => (
                  <div
                    key={msg.id}
                    style={{
                      ...styles.chatMessage,
                      backgroundColor: msg.type === 'system' ? '#374151' : 'transparent',
                    }}
                  >
                    <span style={styles.chatUser}>{msg.userName}:</span>
                    <span style={styles.chatText}>{msg.message}</span>
                  </div>
                ))}
                {messages.length === 0 && (
                  <p style={styles.noMessages}>No messages yet</p>
                )}
              </div>
              <form onSubmit={handleSendMessage} style={styles.chatForm}>
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Send a message..."
                  style={styles.chatInput}
                />
                <button type="submit" style={styles.chatSendBtn}>
                  Send
                </button>
              </form>
            </div>
          )}

          {/* End Broadcast */}
          <button onClick={handleStop} style={styles.endBtn}>
            End Broadcast
          </button>
        </>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
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
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '24px',
  },
  title: {
    fontSize: '24px',
    fontWeight: 700,
    margin: 0,
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  liveIndicator: {
    width: '12px',
    height: '12px',
    borderRadius: '50%',
    backgroundColor: '#ef4444',
    animation: 'pulse 1s ease-in-out infinite',
  },
  liveStats: {
    display: 'flex',
    gap: '16px',
  },
  statItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    fontSize: '14px',
    color: '#d1d5db',
  },
  statIcon: {
    fontSize: '16px',
  },
  error: {
    padding: '12px 16px',
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderRadius: '8px',
    color: '#ef4444',
    marginBottom: '16px',
  },
  setupSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  inputGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  label: {
    fontSize: '12px',
    color: '#9ca3af',
    fontWeight: 500,
  },
  input: {
    padding: '12px 16px',
    backgroundColor: '#111827',
    border: '1px solid #374151',
    borderRadius: '8px',
    color: '#f9fafb',
    fontSize: '14px',
  },
  toggleGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    padding: '16px',
    backgroundColor: '#111827',
    borderRadius: '8px',
  },
  toggleLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    fontSize: '14px',
    color: '#d1d5db',
    cursor: 'pointer',
  },
  checkbox: {
    width: '18px',
    height: '18px',
    accentColor: '#818cf8',
  },
  goLiveBtn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    padding: '16px 24px',
    backgroundColor: '#ef4444',
    border: 'none',
    borderRadius: '8px',
    color: '#fff',
    fontSize: '18px',
    fontWeight: 600,
    cursor: 'pointer',
    marginTop: '8px',
  },
  goLiveIcon: {
    fontSize: '24px',
  },
  preparingSection: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '48px',
    gap: '16px',
  },
  spinner: {
    width: '40px',
    height: '40px',
    border: '4px solid #374151',
    borderTopColor: '#818cf8',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
  meterSection: {
    marginBottom: '20px',
  },
  meterLabel: {
    fontSize: '12px',
    color: '#9ca3af',
    marginBottom: '8px',
  },
  meterTrack: {
    height: '8px',
    backgroundColor: '#374151',
    borderRadius: '4px',
    overflow: 'hidden',
  },
  meterFill: {
    height: '100%',
    transition: 'width 0.1s, background-color 0.2s',
  },
  shareSection: {
    marginBottom: '20px',
  },
  shareUrl: {
    display: 'flex',
    gap: '8px',
  },
  shareInput: {
    flex: 1,
    padding: '10px 14px',
    backgroundColor: '#111827',
    border: '1px solid #374151',
    borderRadius: '8px',
    color: '#9ca3af',
    fontSize: '13px',
  },
  copyBtn: {
    padding: '10px 16px',
    backgroundColor: '#374151',
    border: 'none',
    borderRadius: '8px',
    color: '#fff',
    fontSize: '13px',
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '12px',
    marginBottom: '20px',
  },
  statCard: {
    backgroundColor: '#111827',
    borderRadius: '8px',
    padding: '16px',
    textAlign: 'center',
  },
  statValue: {
    display: 'block',
    fontSize: '24px',
    fontWeight: 700,
    color: '#818cf8',
  },
  statLabel: {
    fontSize: '11px',
    color: '#6b7280',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  reactionsSection: {
    marginBottom: '20px',
  },
  reactionButtons: {
    display: 'flex',
    gap: '8px',
    justifyContent: 'center',
  },
  reactionBtn: {
    position: 'relative',
    padding: '12px 16px',
    backgroundColor: '#111827',
    border: '1px solid #374151',
    borderRadius: '8px',
    fontSize: '24px',
    cursor: 'pointer',
    transition: 'transform 0.1s',
  },
  reactionCount: {
    position: 'absolute',
    top: '-8px',
    right: '-8px',
    backgroundColor: '#ef4444',
    color: '#fff',
    fontSize: '10px',
    padding: '2px 6px',
    borderRadius: '10px',
    fontWeight: 600,
  },
  listenersSection: {
    marginBottom: '20px',
  },
  sectionTitle: {
    fontSize: '12px',
    color: '#9ca3af',
    margin: '0 0 12px 0',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  listenersList: {
    backgroundColor: '#111827',
    borderRadius: '8px',
    padding: '8px',
    maxHeight: '150px',
    overflowY: 'auto',
  },
  listenerItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '8px 12px',
    borderRadius: '6px',
  },
  listenerName: {
    fontSize: '13px',
  },
  kickBtn: {
    width: '24px',
    height: '24px',
    borderRadius: '50%',
    border: 'none',
    backgroundColor: 'transparent',
    color: '#6b7280',
    cursor: 'pointer',
    fontSize: '12px',
  },
  chatSection: {
    marginBottom: '20px',
  },
  chatMessages: {
    backgroundColor: '#111827',
    borderRadius: '8px',
    padding: '12px',
    height: '200px',
    overflowY: 'auto',
    marginBottom: '12px',
  },
  chatMessage: {
    padding: '6px 8px',
    borderRadius: '4px',
    marginBottom: '4px',
  },
  chatUser: {
    fontWeight: 600,
    marginRight: '8px',
    color: '#818cf8',
    fontSize: '13px',
  },
  chatText: {
    fontSize: '13px',
    color: '#d1d5db',
  },
  noMessages: {
    color: '#6b7280',
    textAlign: 'center',
    fontStyle: 'italic',
    margin: 0,
    fontSize: '13px',
  },
  chatForm: {
    display: 'flex',
    gap: '8px',
  },
  chatInput: {
    flex: 1,
    padding: '10px 14px',
    backgroundColor: '#111827',
    border: '1px solid #374151',
    borderRadius: '8px',
    color: '#f9fafb',
    fontSize: '13px',
  },
  chatSendBtn: {
    padding: '10px 20px',
    backgroundColor: '#818cf8',
    border: 'none',
    borderRadius: '8px',
    color: '#fff',
    fontSize: '13px',
    fontWeight: 500,
    cursor: 'pointer',
  },
  endBtn: {
    width: '100%',
    padding: '14px',
    backgroundColor: 'transparent',
    border: '2px solid #ef4444',
    borderRadius: '8px',
    color: '#ef4444',
    fontSize: '16px',
    fontWeight: 600,
    cursor: 'pointer',
  },
};

export default BroadcastStudio;
