// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * P2P Audio Stream Component
 *
 * Manages peer-to-peer audio streaming for listening parties.
 * Allows hosts to share their audio directly with participants.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useWebRTC, getConnectionStats, isWebRTCSupported } from '@/hooks/useWebRTC';
import type { PeerConnection, SignalingMessage } from '@/hooks/useWebRTC';

// === Types ===

interface P2PAudioStreamProps {
  partyId: string;
  isHost: boolean;
  audioElement?: HTMLAudioElement | null;
  participants: Array<{ id: string; displayName: string }>;
  onPeerConnected?: (peerId: string) => void;
  onPeerDisconnected?: (peerId: string) => void;
  onError?: (error: string) => void;
}

interface PeerStats {
  peerId: string;
  rtt: number;
  packetsLost: number;
  jitter: number;
  quality: 'excellent' | 'good' | 'fair' | 'poor';
}

// === Helper Functions ===

function getQualityFromStats(rtt: number, packetsLost: number): PeerStats['quality'] {
  if (rtt < 50 && packetsLost < 10) return 'excellent';
  if (rtt < 150 && packetsLost < 50) return 'good';
  if (rtt < 300 && packetsLost < 100) return 'fair';
  return 'poor';
}

function getQualityColor(quality: PeerStats['quality']): string {
  switch (quality) {
    case 'excellent': return '#22c55e';
    case 'good': return '#84cc16';
    case 'fair': return '#eab308';
    case 'poor': return '#ef4444';
  }
}

// === Component ===

export function P2PAudioStream({
  partyId,
  isHost,
  audioElement,
  participants,
  onPeerConnected,
  onPeerDisconnected,
  onError,
}: P2PAudioStreamProps) {
  const webrtc = useWebRTC();
  const audioRefs = useRef<Map<string, HTMLAudioElement>>(new Map());
  const [peerStats, setPeerStats] = useState<Map<string, PeerStats>>(new Map());
  const [isStreaming, setIsStreaming] = useState(false);
  const statsIntervalRef = useRef<ReturnType<typeof setInterval>>();

  // Check WebRTC support
  const supported = isWebRTCSupported();

  // Initialize WebRTC when component mounts
  useEffect(() => {
    if (!supported) {
      onError?.('WebRTC is not supported in this browser');
      return;
    }

    // Initialize with audio element if host
    if (isHost && audioElement) {
      webrtc.initialize(audioElement).catch((err) => {
        onError?.(err.message);
      });
    } else {
      webrtc.initialize().catch((err) => {
        onError?.(err.message);
      });
    }

    return () => {
      webrtc.cleanup();
    };
  }, [supported, isHost]);

  // Update audio source when audioElement changes (host only)
  useEffect(() => {
    if (isHost && audioElement && webrtc.isReady) {
      webrtc.setLocalAudioSource(audioElement);
    }
  }, [audioElement, isHost, webrtc.isReady]);

  // Connect to participants when they join (host initiates connections)
  useEffect(() => {
    if (!isHost || !webrtc.isReady) return;

    participants.forEach((participant) => {
      if (!webrtc.peers.has(participant.id)) {
        webrtc.connectToPeer(participant.id, participant.displayName);
      }
    });
  }, [participants, isHost, webrtc.isReady]);

  // Track peer connection/disconnection
  useEffect(() => {
    webrtc.peers.forEach((peer) => {
      if (peer.state === 'connected') {
        onPeerConnected?.(peer.id);
        setIsStreaming(true);
      } else if (peer.state === 'disconnected' || peer.state === 'failed') {
        onPeerDisconnected?.(peer.id);
      }
    });
  }, [webrtc.peers]);

  // Play remote audio streams
  useEffect(() => {
    webrtc.peers.forEach((peer) => {
      if (peer.audioStream && !audioRefs.current.has(peer.id)) {
        const audio = new Audio();
        audio.srcObject = peer.audioStream;
        audio.autoplay = true;
        audioRefs.current.set(peer.id, audio);
      }
    });

    // Cleanup removed peers
    audioRefs.current.forEach((audio, peerId) => {
      if (!webrtc.peers.has(peerId)) {
        audio.pause();
        audio.srcObject = null;
        audioRefs.current.delete(peerId);
      }
    });
  }, [webrtc.peers]);

  // Collect connection stats periodically
  useEffect(() => {
    if (webrtc.peers.size === 0) return;

    const collectStats = async () => {
      const newStats = new Map<string, PeerStats>();

      for (const [peerId, peer] of webrtc.peers) {
        const stats = await getConnectionStats(peer.connection);
        if (stats) {
          newStats.set(peerId, {
            peerId,
            ...stats,
            quality: getQualityFromStats(stats.rtt * 1000, stats.packetsLost),
          });
        }
      }

      setPeerStats(newStats);
    };

    collectStats();
    statsIntervalRef.current = setInterval(collectStats, 2000);

    return () => {
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current);
      }
    };
  }, [webrtc.peers.size]);

  // Handle incoming signaling messages (would be called by WebSocket handler)
  const handleSignaling = useCallback(
    (message: SignalingMessage) => {
      webrtc.handleSignalingMessage(message);
    },
    [webrtc.handleSignalingMessage]
  );

  // Mute control for host
  const toggleMute = useCallback(() => {
    if (isHost) {
      const isMuted = webrtc.localStream?.getAudioTracks()[0]?.enabled === false;
      webrtc.muteLocalAudio(!isMuted);
    }
  }, [isHost, webrtc]);

  if (!supported) {
    return (
      <div style={styles.container}>
        <div style={styles.unsupported}>
          <span style={styles.warningIcon}>!</span>
          <span>P2P audio streaming is not supported in this browser</span>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.title}>
          <span style={styles.icon}>
            {isStreaming ? '\u{1F4E1}' : '\u{1F50A}'}
          </span>
          <span>P2P Audio</span>
        </div>
        <div style={styles.status}>
          {webrtc.isConnecting && <span style={styles.connecting}>Connecting...</span>}
          {webrtc.error && <span style={styles.error}>{webrtc.error}</span>}
          {isStreaming && (
            <span style={styles.streaming}>
              {webrtc.peers.size} peer{webrtc.peers.size !== 1 ? 's' : ''} connected
            </span>
          )}
        </div>
      </div>

      {/* Host Controls */}
      {isHost && (
        <div style={styles.hostControls}>
          <button
            style={styles.muteButton}
            onClick={toggleMute}
            disabled={!webrtc.localStream}
          >
            {webrtc.localStream?.getAudioTracks()[0]?.enabled ? 'Mute Stream' : 'Unmute Stream'}
          </button>
        </div>
      )}

      {/* Peer List with Quality Indicators */}
      {webrtc.peers.size > 0 && (
        <div style={styles.peerList}>
          {Array.from(webrtc.peers.values()).map((peer) => {
            const stats = peerStats.get(peer.id);
            return (
              <div key={peer.id} style={styles.peerItem}>
                <div style={styles.peerInfo}>
                  <span style={styles.peerName}>{peer.displayName}</span>
                  <span style={{ ...styles.peerState, color: getStateColor(peer.state) }}>
                    {peer.state}
                  </span>
                </div>
                {stats && (
                  <div style={styles.peerStats}>
                    <span
                      style={{
                        ...styles.qualityIndicator,
                        backgroundColor: getQualityColor(stats.quality),
                      }}
                    />
                    <span style={styles.statValue}>{Math.round(stats.rtt * 1000)}ms</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Empty State */}
      {webrtc.peers.size === 0 && webrtc.isReady && !webrtc.isConnecting && (
        <div style={styles.emptyState}>
          {isHost
            ? 'Waiting for participants to connect...'
            : 'Waiting for host to share audio...'}
        </div>
      )}
    </div>
  );
}

// Helper for connection state color
function getStateColor(state: RTCPeerConnectionState): string {
  switch (state) {
    case 'connected': return '#22c55e';
    case 'connecting': return '#eab308';
    case 'disconnected': return '#f97316';
    case 'failed': return '#ef4444';
    case 'closed': return '#6b7280';
    default: return '#9ca3af';
  }
}

// === Styles ===

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '12px',
    padding: '16px',
    fontFamily: 'system-ui, -apple-system, sans-serif',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px',
  },
  title: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontWeight: 600,
    color: '#ffffff',
  },
  icon: {
    fontSize: '18px',
  },
  status: {
    fontSize: '12px',
  },
  connecting: {
    color: '#eab308',
  },
  error: {
    color: '#ef4444',
  },
  streaming: {
    color: '#22c55e',
  },
  hostControls: {
    marginBottom: '12px',
  },
  muteButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    border: 'none',
    borderRadius: '6px',
    padding: '8px 16px',
    color: '#ffffff',
    cursor: 'pointer',
    fontSize: '14px',
  },
  peerList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  peerItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '8px 12px',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: '8px',
  },
  peerInfo: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  peerName: {
    color: '#ffffff',
    fontSize: '14px',
    fontWeight: 500,
  },
  peerState: {
    fontSize: '11px',
    textTransform: 'capitalize' as const,
  },
  peerStats: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  qualityIndicator: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
  },
  statValue: {
    color: '#9ca3af',
    fontSize: '12px',
  },
  emptyState: {
    textAlign: 'center' as const,
    color: '#9ca3af',
    fontSize: '14px',
    padding: '16px',
  },
  unsupported: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    color: '#f97316',
    fontSize: '14px',
  },
  warningIcon: {
    width: '20px',
    height: '20px',
    borderRadius: '50%',
    backgroundColor: '#f97316',
    color: '#ffffff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 'bold',
    fontSize: '12px',
  },
};

export default P2PAudioStream;
