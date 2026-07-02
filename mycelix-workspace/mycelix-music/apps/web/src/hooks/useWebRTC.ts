// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useWebRTC Hook
 *
 * Manages WebRTC peer connections for P2P audio streaming.
 * Supports direct peer-to-peer audio sharing for listening parties.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { getWebSocket, type MycelixWebSocket } from '@/lib/websocket';

// === Types ===

export interface PeerConnection {
  id: string;
  displayName: string;
  connection: RTCPeerConnection;
  state: RTCPeerConnectionState;
  audioStream?: MediaStream;
  isRemote: boolean;
}

export interface IceCandidate {
  candidate: string;
  sdpMid: string | null;
  sdpMLineIndex: number | null;
}

export interface SignalingMessage {
  type: 'offer' | 'answer' | 'ice-candidate';
  peerId: string;
  sdp?: string;
  candidate?: IceCandidate;
}

export interface WebRTCState {
  isReady: boolean;
  localStream: MediaStream | null;
  peers: Map<string, PeerConnection>;
  isConnecting: boolean;
  error: string | null;
}

// ICE server configuration
const DEFAULT_ICE_SERVERS: RTCIceServer[] = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
  { urls: 'stun:stun2.l.google.com:19302' },
];

// === Hook ===

export interface UseWebRTCReturn extends WebRTCState {
  // Connection management
  initialize: (localAudioSource?: HTMLAudioElement | MediaStream) => Promise<void>;
  cleanup: () => void;

  // Peer operations
  connectToPeer: (peerId: string, displayName: string) => Promise<void>;
  disconnectFromPeer: (peerId: string) => void;

  // Stream management
  setLocalAudioSource: (source: HTMLAudioElement | MediaStream) => void;
  muteLocalAudio: (muted: boolean) => void;

  // Signaling (for integration with WebSocket)
  handleSignalingMessage: (message: SignalingMessage) => Promise<void>;
}

export function useWebRTC(iceServers: RTCIceServer[] = DEFAULT_ICE_SERVERS): UseWebRTCReturn {
  const wsRef = useRef<MycelixWebSocket | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const peersRef = useRef<Map<string, PeerConnection>>(new Map());
  const audioContextRef = useRef<AudioContext | null>(null);

  const [state, setState] = useState<WebRTCState>({
    isReady: false,
    localStream: null,
    peers: new Map(),
    isConnecting: false,
    error: null,
  });

  // Update peers in state
  const updatePeers = useCallback(() => {
    setState((s) => ({
      ...s,
      peers: new Map(peersRef.current),
    }));
  }, []);

  // Create RTCPeerConnection with handlers
  const createPeerConnection = useCallback(
    (peerId: string, displayName: string): RTCPeerConnection => {
      const pc = new RTCPeerConnection({ iceServers });

      // Track connection state
      pc.onconnectionstatechange = () => {
        const peer = peersRef.current.get(peerId);
        if (peer) {
          peer.state = pc.connectionState;
          updatePeers();
        }

        if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
          console.log(`[WebRTC] Peer ${peerId} disconnected`);
        }
      };

      // Handle ICE candidates
      pc.onicecandidate = (event) => {
        if (event.candidate && wsRef.current?.isConnected) {
          sendSignalingMessage({
            type: 'ice-candidate',
            peerId,
            candidate: {
              candidate: event.candidate.candidate,
              sdpMid: event.candidate.sdpMid,
              sdpMLineIndex: event.candidate.sdpMLineIndex,
            },
          });
        }
      };

      // Handle incoming tracks
      pc.ontrack = (event) => {
        console.log(`[WebRTC] Received track from ${peerId}`);
        const peer = peersRef.current.get(peerId);
        if (peer && event.streams[0]) {
          peer.audioStream = event.streams[0];
          updatePeers();
        }
      };

      // Add local stream tracks
      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach((track) => {
          pc.addTrack(track, localStreamRef.current!);
        });
      }

      // Store peer connection
      peersRef.current.set(peerId, {
        id: peerId,
        displayName,
        connection: pc,
        state: pc.connectionState,
        isRemote: true,
      });

      updatePeers();
      return pc;
    },
    [iceServers, updatePeers]
  );

  // Send signaling message via WebSocket
  const sendSignalingMessage = useCallback((message: SignalingMessage) => {
    if (!wsRef.current?.isConnected) {
      console.error('[WebRTC] WebSocket not connected for signaling');
      return;
    }

    // In production, this would send via the WebSocket connection
    console.log('[WebRTC] Sending signaling message:', message.type, message.peerId);
  }, []);

  // Initialize WebRTC with local audio source
  const initialize = useCallback(
    async (localAudioSource?: HTMLAudioElement | MediaStream) => {
      setState((s) => ({ ...s, isConnecting: true, error: null }));

      try {
        // Connect WebSocket for signaling
        wsRef.current = getWebSocket();
        await wsRef.current.connect();

        // Set up audio context
        audioContextRef.current = new AudioContext();

        // Get local audio stream
        if (localAudioSource instanceof MediaStream) {
          localStreamRef.current = localAudioSource;
        } else if (localAudioSource instanceof HTMLAudioElement) {
          // Capture audio from HTML audio element
          const stream = await captureAudioElement(localAudioSource, audioContextRef.current);
          localStreamRef.current = stream;
        }

        setState((s) => ({
          ...s,
          isReady: true,
          isConnecting: false,
          localStream: localStreamRef.current,
        }));
      } catch (error) {
        setState((s) => ({
          ...s,
          isConnecting: false,
          error: error instanceof Error ? error.message : 'Failed to initialize WebRTC',
        }));
      }
    },
    []
  );

  // Capture audio from HTMLAudioElement
  const captureAudioElement = async (
    audioElement: HTMLAudioElement,
    audioContext: AudioContext
  ): Promise<MediaStream> => {
    const source = audioContext.createMediaElementSource(audioElement);
    const destination = audioContext.createMediaStreamDestination();
    source.connect(destination);
    source.connect(audioContext.destination); // Also play locally
    return destination.stream;
  };

  // Connect to a peer (initiate offer)
  const connectToPeer = useCallback(
    async (peerId: string, displayName: string) => {
      const pc = createPeerConnection(peerId, displayName);

      try {
        const offer = await pc.createOffer({
          offerToReceiveAudio: true,
          offerToReceiveVideo: false,
        });
        await pc.setLocalDescription(offer);

        sendSignalingMessage({
          type: 'offer',
          peerId,
          sdp: offer.sdp,
        });
      } catch (error) {
        console.error(`[WebRTC] Failed to create offer for ${peerId}:`, error);
        peersRef.current.delete(peerId);
        updatePeers();
      }
    },
    [createPeerConnection, sendSignalingMessage, updatePeers]
  );

  // Disconnect from a peer
  const disconnectFromPeer = useCallback(
    (peerId: string) => {
      const peer = peersRef.current.get(peerId);
      if (peer) {
        peer.connection.close();
        peersRef.current.delete(peerId);
        updatePeers();
      }
    },
    [updatePeers]
  );

  // Handle incoming signaling messages
  const handleSignalingMessage = useCallback(
    async (message: SignalingMessage) => {
      const { type, peerId, sdp, candidate } = message;

      let pc = peersRef.current.get(peerId)?.connection;

      switch (type) {
        case 'offer':
          // Create peer connection if doesn't exist
          if (!pc) {
            pc = createPeerConnection(peerId, 'Remote Peer');
          }

          await pc.setRemoteDescription(new RTCSessionDescription({ type: 'offer', sdp }));
          const answer = await pc.createAnswer();
          await pc.setLocalDescription(answer);

          sendSignalingMessage({
            type: 'answer',
            peerId,
            sdp: answer.sdp,
          });
          break;

        case 'answer':
          if (pc) {
            await pc.setRemoteDescription(new RTCSessionDescription({ type: 'answer', sdp }));
          }
          break;

        case 'ice-candidate':
          if (pc && candidate) {
            await pc.addIceCandidate(
              new RTCIceCandidate({
                candidate: candidate.candidate,
                sdpMid: candidate.sdpMid ?? undefined,
                sdpMLineIndex: candidate.sdpMLineIndex ?? undefined,
              })
            );
          }
          break;
      }

      updatePeers();
    },
    [createPeerConnection, sendSignalingMessage, updatePeers]
  );

  // Set local audio source
  const setLocalAudioSource = useCallback(
    (source: HTMLAudioElement | MediaStream) => {
      if (source instanceof MediaStream) {
        localStreamRef.current = source;
      } else if (audioContextRef.current) {
        captureAudioElement(source, audioContextRef.current).then((stream) => {
          localStreamRef.current = stream;
          setState((s) => ({ ...s, localStream: stream }));
        });
      }
    },
    []
  );

  // Mute/unmute local audio
  const muteLocalAudio = useCallback((muted: boolean) => {
    if (localStreamRef.current) {
      localStreamRef.current.getAudioTracks().forEach((track) => {
        track.enabled = !muted;
      });
    }
  }, []);

  // Cleanup
  const cleanup = useCallback(() => {
    // Close all peer connections
    peersRef.current.forEach((peer) => {
      peer.connection.close();
    });
    peersRef.current.clear();

    // Stop local stream
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => track.stop());
      localStreamRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Disconnect WebSocket
    if (wsRef.current) {
      wsRef.current.disconnect();
      wsRef.current = null;
    }

    setState({
      isReady: false,
      localStream: null,
      peers: new Map(),
      isConnecting: false,
      error: null,
    });
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => cleanup();
  }, [cleanup]);

  return {
    ...state,
    initialize,
    cleanup,
    connectToPeer,
    disconnectFromPeer,
    setLocalAudioSource,
    muteLocalAudio,
    handleSignalingMessage,
  };
}

// === Utility Functions ===

/**
 * Get audio stream from microphone
 */
export async function getMicrophoneStream(): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
    video: false,
  });
}

/**
 * Check WebRTC support
 */
export function isWebRTCSupported(): boolean {
  return !!(
    typeof RTCPeerConnection !== 'undefined' &&
    navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia
  );
}

/**
 * Get connection quality metrics
 */
export async function getConnectionStats(
  pc: RTCPeerConnection
): Promise<{ rtt: number; packetsLost: number; jitter: number } | null> {
  try {
    const stats = await pc.getStats();
    let rtt = 0;
    let packetsLost = 0;
    let jitter = 0;

    stats.forEach((report) => {
      if (report.type === 'candidate-pair' && report.state === 'succeeded') {
        rtt = report.currentRoundTripTime || 0;
      }
      if (report.type === 'inbound-rtp' && report.kind === 'audio') {
        packetsLost = report.packetsLost || 0;
        jitter = report.jitter || 0;
      }
    });

    return { rtt, packetsLost, jitter };
  } catch {
    return null;
  }
}

export default useWebRTC;
