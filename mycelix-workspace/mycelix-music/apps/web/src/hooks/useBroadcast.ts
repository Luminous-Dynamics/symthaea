// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Live Broadcasting Hook
 *
 * Stream DJ sets and live audio to listeners
 * Uses WebRTC for peer-to-peer and WebSocket for signaling
 */

import { useState, useCallback, useRef, useEffect } from 'react';

export interface BroadcastConfig {
  title: string;
  description?: string;
  genre?: string;
  isPrivate: boolean;
  maxListeners?: number;
  allowChat: boolean;
  recordBroadcast: boolean;
}

export interface Listener {
  id: string;
  name: string;
  joinedAt: number;
  location?: string;
}

export interface ChatMessage {
  id: string;
  userId: string;
  userName: string;
  message: string;
  timestamp: number;
  type: 'chat' | 'system' | 'reaction';
}

export interface BroadcastStats {
  listeners: number;
  peakListeners: number;
  duration: number;
  bytesTransferred: number;
  avgBitrate: number;
  reactions: Record<string, number>;
}

export interface BroadcastState {
  status: 'idle' | 'preparing' | 'live' | 'ending' | 'error';
  streamId: string | null;
  startTime: number | null;
  config: BroadcastConfig | null;
}

const DEFAULT_CONFIG: BroadcastConfig = {
  title: 'Live DJ Set',
  isPrivate: false,
  allowChat: true,
  recordBroadcast: false,
};

export function useBroadcast(signalingUrl?: string) {
  const [state, setState] = useState<BroadcastState>({
    status: 'idle',
    streamId: null,
    startTime: null,
    config: null,
  });
  const [listeners, setListeners] = useState<Listener[]>([]);
  const [stats, setStats] = useState<BroadcastStats>({
    listeners: 0,
    peakListeners: 0,
    duration: 0,
    bytesTransferred: 0,
    avgBitrate: 0,
    reactions: {},
  });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [audioLevel, setAudioLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const peerConnectionsRef = useRef<Map<string, RTCPeerConnection>>(new Map());
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const animationFrameRef = useRef<number | null>(null);
  const durationIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // ICE servers for WebRTC
  const iceServers: RTCIceServer[] = [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
  ];

  // Generate stream ID
  const generateStreamId = (): string => {
    return `live-${Date.now().toString(36)}-${Math.random().toString(36).substr(2, 9)}`;
  };

  // Connect to signaling server
  const connectSignaling = useCallback((streamId: string, isBroadcaster: boolean) => {
    const url = signalingUrl || `wss://api.mycelix.local/broadcast/signal`;

    // For development, simulate signaling
    if (!signalingUrl) {
      console.log('Broadcast signaling: Using simulated mode');
      return;
    }

    const ws = new WebSocket(`${url}?stream=${streamId}&role=${isBroadcaster ? 'broadcaster' : 'listener'}`);

    ws.onopen = () => {
      console.log('Signaling connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleSignalingMessage(data);
    };

    ws.onerror = (err) => {
      console.error('Signaling error:', err);
      setError('Connection error');
    };

    ws.onclose = () => {
      console.log('Signaling disconnected');
    };

    wsRef.current = ws;
  }, [signalingUrl]);

  // Handle signaling messages
  const handleSignalingMessage = useCallback(async (data: any) => {
    switch (data.type) {
      case 'listener-joined':
        setListeners(prev => [...prev, data.listener]);
        setStats(prev => ({
          ...prev,
          listeners: prev.listeners + 1,
          peakListeners: Math.max(prev.peakListeners, prev.listeners + 1),
        }));
        // Create peer connection for new listener
        await createPeerConnection(data.listener.id);
        break;

      case 'listener-left':
        setListeners(prev => prev.filter(l => l.id !== data.listenerId));
        setStats(prev => ({ ...prev, listeners: prev.listeners - 1 }));
        closePeerConnection(data.listenerId);
        break;

      case 'offer':
        await handleOffer(data.from, data.offer);
        break;

      case 'answer':
        await handleAnswer(data.from, data.answer);
        break;

      case 'ice-candidate':
        await handleIceCandidate(data.from, data.candidate);
        break;

      case 'chat':
        setMessages(prev => [...prev, data.message]);
        break;

      case 'reaction':
        setStats(prev => ({
          ...prev,
          reactions: {
            ...prev.reactions,
            [data.reaction]: (prev.reactions[data.reaction] || 0) + 1,
          },
        }));
        break;
    }
  }, []);

  // Create peer connection for a listener
  const createPeerConnection = useCallback(async (listenerId: string) => {
    if (!mediaStreamRef.current) return;

    const pc = new RTCPeerConnection({ iceServers });

    // Add audio tracks
    mediaStreamRef.current.getAudioTracks().forEach(track => {
      pc.addTrack(track, mediaStreamRef.current!);
    });

    // Handle ICE candidates
    pc.onicecandidate = (event) => {
      if (event.candidate && wsRef.current) {
        wsRef.current.send(JSON.stringify({
          type: 'ice-candidate',
          to: listenerId,
          candidate: event.candidate,
        }));
      }
    };

    // Create and send offer
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({
        type: 'offer',
        to: listenerId,
        offer,
      }));
    }

    peerConnectionsRef.current.set(listenerId, pc);
  }, []);

  // Handle incoming offer (for listeners)
  const handleOffer = useCallback(async (fromId: string, offer: RTCSessionDescriptionInit) => {
    const pc = new RTCPeerConnection({ iceServers });

    pc.ontrack = (event) => {
      // Handle incoming audio
      const audio = document.createElement('audio');
      audio.srcObject = event.streams[0];
      audio.autoplay = true;
    };

    pc.onicecandidate = (event) => {
      if (event.candidate && wsRef.current) {
        wsRef.current.send(JSON.stringify({
          type: 'ice-candidate',
          to: fromId,
          candidate: event.candidate,
        }));
      }
    };

    await pc.setRemoteDescription(offer);
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);

    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({
        type: 'answer',
        to: fromId,
        answer,
      }));
    }

    peerConnectionsRef.current.set(fromId, pc);
  }, []);

  // Handle answer
  const handleAnswer = useCallback(async (fromId: string, answer: RTCSessionDescriptionInit) => {
    const pc = peerConnectionsRef.current.get(fromId);
    if (pc) {
      await pc.setRemoteDescription(answer);
    }
  }, []);

  // Handle ICE candidate
  const handleIceCandidate = useCallback(async (fromId: string, candidate: RTCIceCandidateInit) => {
    const pc = peerConnectionsRef.current.get(fromId);
    if (pc) {
      await pc.addIceCandidate(candidate);
    }
  }, []);

  // Close peer connection
  const closePeerConnection = useCallback((peerId: string) => {
    const pc = peerConnectionsRef.current.get(peerId);
    if (pc) {
      pc.close();
      peerConnectionsRef.current.delete(peerId);
    }
  }, []);

  // Start broadcasting
  const startBroadcast = useCallback(async (
    audioSource: MediaStream | AudioNode,
    config: Partial<BroadcastConfig> = {}
  ) => {
    try {
      setState(prev => ({ ...prev, status: 'preparing' }));
      setError(null);

      const broadcastConfig = { ...DEFAULT_CONFIG, ...config };
      const streamId = generateStreamId();

      // Set up audio context
      audioContextRef.current = new AudioContext();

      // Get media stream
      let stream: MediaStream;
      if (audioSource instanceof MediaStream) {
        stream = audioSource;
      } else {
        // Create stream from AudioNode
        const dest = audioContextRef.current.createMediaStreamDestination();
        audioSource.connect(dest);
        stream = dest.stream;
      }
      mediaStreamRef.current = stream;

      // Set up analyser for level metering
      const analyser = audioContextRef.current.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;

      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyser);

      // Start level metering
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const updateLevel = () => {
        if (analyserRef.current && state.status === 'live') {
          analyserRef.current.getByteFrequencyData(dataArray);
          const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
          setAudioLevel(avg / 255);
          animationFrameRef.current = requestAnimationFrame(updateLevel);
        }
      };

      // Set up recording if enabled
      if (broadcastConfig.recordBroadcast) {
        const recorder = new MediaRecorder(stream, {
          mimeType: 'audio/webm;codecs=opus',
        });

        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            recordedChunksRef.current.push(event.data);
            setStats(prev => ({
              ...prev,
              bytesTransferred: prev.bytesTransferred + event.data.size,
            }));
          }
        };

        recorder.start(1000); // Capture every second
        recorderRef.current = recorder;
      }

      // Connect to signaling server
      connectSignaling(streamId, true);

      // Update state
      setState({
        status: 'live',
        streamId,
        startTime: Date.now(),
        config: broadcastConfig,
      });

      // Start duration counter
      durationIntervalRef.current = setInterval(() => {
        setStats(prev => ({ ...prev, duration: prev.duration + 1 }));
      }, 1000);

      // Start level updates
      animationFrameRef.current = requestAnimationFrame(updateLevel);

      return streamId;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start broadcast');
      setState(prev => ({ ...prev, status: 'error' }));
      throw err;
    }
  }, [connectSignaling, state.status]);

  // Stop broadcasting
  const stopBroadcast = useCallback(async (): Promise<Blob | null> => {
    setState(prev => ({ ...prev, status: 'ending' }));

    // Stop animations
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current);
    }

    // Stop recording
    let recordingBlob: Blob | null = null;
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop();
      await new Promise(resolve => setTimeout(resolve, 100));
      recordingBlob = new Blob(recordedChunksRef.current, { type: 'audio/webm' });
      recordedChunksRef.current = [];
    }

    // Close all peer connections
    peerConnectionsRef.current.forEach((pc, id) => {
      pc.close();
    });
    peerConnectionsRef.current.clear();

    // Close signaling connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    // Stop media tracks
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Reset state
    setState({
      status: 'idle',
      streamId: null,
      startTime: null,
      config: null,
    });
    setListeners([]);
    setMessages([]);
    setAudioLevel(0);

    return recordingBlob;
  }, []);

  // Send chat message
  const sendMessage = useCallback((message: string) => {
    if (!wsRef.current || state.status !== 'live') return;

    const chatMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      userId: 'broadcaster',
      userName: 'DJ',
      message,
      timestamp: Date.now(),
      type: 'chat',
    };

    wsRef.current.send(JSON.stringify({
      type: 'chat',
      message: chatMessage,
    }));

    setMessages(prev => [...prev, chatMessage]);
  }, [state.status]);

  // Send reaction
  const sendReaction = useCallback((reaction: string) => {
    if (!wsRef.current) return;

    wsRef.current.send(JSON.stringify({
      type: 'reaction',
      reaction,
    }));

    setStats(prev => ({
      ...prev,
      reactions: {
        ...prev.reactions,
        [reaction]: (prev.reactions[reaction] || 0) + 1,
      },
    }));
  }, []);

  // Kick listener
  const kickListener = useCallback((listenerId: string) => {
    if (!wsRef.current) return;

    wsRef.current.send(JSON.stringify({
      type: 'kick',
      listenerId,
    }));

    closePeerConnection(listenerId);
    setListeners(prev => prev.filter(l => l.id !== listenerId));
  }, [closePeerConnection]);

  // Update broadcast config
  const updateConfig = useCallback((updates: Partial<BroadcastConfig>) => {
    setState(prev => ({
      ...prev,
      config: prev.config ? { ...prev.config, ...updates } : null,
    }));

    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({
        type: 'config-update',
        config: updates,
      }));
    }
  }, []);

  // Get share URL
  const getShareUrl = useCallback(() => {
    if (!state.streamId) return null;
    return `${window.location.origin}/live/${state.streamId}`;
  }, [state.streamId]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (state.status === 'live') {
        stopBroadcast();
      }
    };
  }, []);

  return {
    // State
    state,
    listeners,
    stats,
    messages,
    audioLevel,
    error,

    // Controls
    startBroadcast,
    stopBroadcast,
    sendMessage,
    sendReaction,
    kickListener,
    updateConfig,

    // Utils
    getShareUrl,
    isLive: state.status === 'live',
  };
}

// Listener hook for joining broadcasts
export function useBroadcastListener(streamId: string, signalingUrl?: string) {
  const [isConnected, setIsConnected] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [broadcasterInfo, setBroadcasterInfo] = useState<BroadcastConfig | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [listenerCount, setListenerCount] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Connect to broadcast
  const connect = useCallback(async () => {
    try {
      const url = signalingUrl || `wss://api.mycelix.local/broadcast/signal`;

      // For development, simulate connection
      if (!signalingUrl) {
        console.log('Listener: Using simulated mode');
        setIsConnected(true);
        setBroadcasterInfo({
          title: 'Simulated Live Stream',
          isPrivate: false,
          allowChat: true,
          recordBroadcast: false,
        });
        return;
      }

      const ws = new WebSocket(`${url}?stream=${streamId}&role=listener`);

      ws.onopen = () => {
        setIsConnected(true);
      };

      ws.onmessage = async (event) => {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case 'broadcast-info':
            setBroadcasterInfo(data.config);
            setListenerCount(data.listenerCount);
            break;

          case 'offer':
            await handleOffer(data.offer);
            break;

          case 'ice-candidate':
            if (pcRef.current) {
              await pcRef.current.addIceCandidate(data.candidate);
            }
            break;

          case 'chat':
            setMessages(prev => [...prev, data.message]);
            break;

          case 'listener-count':
            setListenerCount(data.count);
            break;

          case 'broadcast-ended':
            disconnect();
            break;
        }
      };

      ws.onerror = () => {
        setError('Connection error');
      };

      ws.onclose = () => {
        setIsConnected(false);
      };

      wsRef.current = ws;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect');
    }
  }, [streamId, signalingUrl]);

  // Handle offer from broadcaster
  const handleOffer = async (offer: RTCSessionDescriptionInit) => {
    const pc = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
      ],
    });

    pc.ontrack = (event) => {
      if (!audioRef.current) {
        audioRef.current = document.createElement('audio');
        audioRef.current.autoplay = true;
      }
      audioRef.current.srcObject = event.streams[0];
      setIsPlaying(true);
    };

    pc.onicecandidate = (event) => {
      if (event.candidate && wsRef.current) {
        wsRef.current.send(JSON.stringify({
          type: 'ice-candidate',
          candidate: event.candidate,
        }));
      }
    };

    await pc.setRemoteDescription(offer);
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);

    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({
        type: 'answer',
        answer,
      }));
    }

    pcRef.current = pc;
  };

  // Disconnect
  const disconnect = useCallback(() => {
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.srcObject = null;
    }
    setIsConnected(false);
    setIsPlaying(false);
  }, []);

  // Send chat message
  const sendMessage = useCallback((message: string, userName: string) => {
    if (!wsRef.current) return;

    const chatMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      userId: 'listener',
      userName,
      message,
      timestamp: Date.now(),
      type: 'chat',
    };

    wsRef.current.send(JSON.stringify({
      type: 'chat',
      message: chatMessage,
    }));

    setMessages(prev => [...prev, chatMessage]);
  }, []);

  // Send reaction
  const sendReaction = useCallback((reaction: string) => {
    if (!wsRef.current) return;

    wsRef.current.send(JSON.stringify({
      type: 'reaction',
      reaction,
    }));
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected,
    isPlaying,
    broadcasterInfo,
    messages,
    listenerCount,
    error,
    connect,
    disconnect,
    sendMessage,
    sendReaction,
  };
}
