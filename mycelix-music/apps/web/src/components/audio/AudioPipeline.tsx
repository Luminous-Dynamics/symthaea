// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { createContext, useContext, useRef, useState, useCallback, useEffect, ReactNode } from 'react';

/**
 * Unified Audio Pipeline
 *
 * Central audio routing system that connects all audio features:
 * - Main player output
 * - DJ Mixer decks
 * - Effects chain
 * - Stem separation
 * - Broadcasting
 * - Voice control audio feedback
 */

interface AudioNode {
  id: string;
  type: 'source' | 'effect' | 'analyzer' | 'destination';
  node: AudioNode | GainNode | AnalyserNode | MediaStreamAudioDestinationNode;
}

interface AudioRoute {
  from: string;
  to: string;
  gain: number;
}

interface AudioPipelineState {
  isInitialized: boolean;
  masterVolume: number;
  isMuted: boolean;
  activeRoutes: AudioRoute[];
  audioLevel: number;
}

interface AudioPipelineContextType {
  // State
  state: AudioPipelineState;

  // Audio Context
  audioContext: AudioContext | null;
  masterGain: GainNode | null;
  masterAnalyser: AnalyserNode | null;

  // Node Management
  registerNode: (id: string, node: AudioNode) => void;
  unregisterNode: (id: string) => void;
  getNode: (id: string) => AudioNode | undefined;

  // Routing
  connect: (fromId: string, toId: string, gain?: number) => void;
  disconnect: (fromId: string, toId: string) => void;
  setRouteGain: (fromId: string, toId: string, gain: number) => void;

  // Master Controls
  setMasterVolume: (volume: number) => void;
  setMuted: (muted: boolean) => void;

  // Destinations
  getStreamDestination: () => MediaStreamAudioDestinationNode | null;
  getBroadcastStream: () => MediaStream | null;

  // Analysis
  getAudioLevel: () => number;
  getFrequencyData: () => Uint8Array | null;
  getWaveformData: () => Uint8Array | null;
}

const AudioPipelineContext = createContext<AudioPipelineContextType | null>(null);

export function useAudioPipeline() {
  const context = useContext(AudioPipelineContext);
  if (!context) {
    throw new Error('useAudioPipeline must be used within AudioPipelineProvider');
  }
  return context;
}

interface AudioPipelineProviderProps {
  children: ReactNode;
}

export function AudioPipelineProvider({ children }: AudioPipelineProviderProps) {
  const [state, setState] = useState<AudioPipelineState>({
    isInitialized: false,
    masterVolume: 1.0,
    isMuted: false,
    activeRoutes: [],
    audioLevel: 0,
  });

  const audioContextRef = useRef<AudioContext | null>(null);
  const masterGainRef = useRef<GainNode | null>(null);
  const masterAnalyserRef = useRef<AnalyserNode | null>(null);
  const streamDestinationRef = useRef<MediaStreamAudioDestinationNode | null>(null);
  const nodesRef = useRef<Map<string, AudioNode>>(new Map());
  const routesRef = useRef<Map<string, GainNode>>(new Map());
  const animationFrameRef = useRef<number | null>(null);

  // Initialize audio context on first user interaction
  const initializeAudio = useCallback(() => {
    if (audioContextRef.current) return;

    const ctx = new AudioContext();
    audioContextRef.current = ctx;

    // Create master gain
    const masterGain = ctx.createGain();
    masterGain.gain.value = state.masterVolume;
    masterGainRef.current = masterGain;

    // Create master analyser
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.8;
    masterAnalyserRef.current = analyser;

    // Create stream destination for broadcasting
    const streamDest = ctx.createMediaStreamDestination();
    streamDestinationRef.current = streamDest;

    // Connect: masterGain -> analyser -> destination
    masterGain.connect(analyser);
    analyser.connect(ctx.destination);

    // Also connect to stream destination for broadcasting
    masterGain.connect(streamDest);

    // Register built-in nodes
    nodesRef.current.set('master', {
      id: 'master',
      type: 'destination',
      node: masterGain,
    } as any);

    setState(prev => ({ ...prev, isInitialized: true }));

    // Start level monitoring
    const updateLevel = () => {
      if (masterAnalyserRef.current) {
        const data = new Uint8Array(masterAnalyserRef.current.frequencyBinCount);
        masterAnalyserRef.current.getByteFrequencyData(data);
        const avg = data.reduce((a, b) => a + b, 0) / data.length;
        setState(prev => ({ ...prev, audioLevel: avg / 255 }));
      }
      animationFrameRef.current = requestAnimationFrame(updateLevel);
    };
    animationFrameRef.current = requestAnimationFrame(updateLevel);
  }, [state.masterVolume]);

  // Auto-initialize on user interaction
  useEffect(() => {
    const handleInteraction = () => {
      initializeAudio();
      window.removeEventListener('click', handleInteraction);
      window.removeEventListener('keydown', handleInteraction);
    };

    window.addEventListener('click', handleInteraction);
    window.addEventListener('keydown', handleInteraction);

    return () => {
      window.removeEventListener('click', handleInteraction);
      window.removeEventListener('keydown', handleInteraction);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [initializeAudio]);

  // Register a node
  const registerNode = useCallback((id: string, node: AudioNode) => {
    nodesRef.current.set(id, node);
  }, []);

  // Unregister a node
  const unregisterNode = useCallback((id: string) => {
    // Disconnect all routes involving this node
    routesRef.current.forEach((gainNode, routeKey) => {
      if (routeKey.includes(id)) {
        try {
          gainNode.disconnect();
        } catch (e) {}
        routesRef.current.delete(routeKey);
      }
    });
    nodesRef.current.delete(id);
  }, []);

  // Get a node
  const getNode = useCallback((id: string) => {
    return nodesRef.current.get(id);
  }, []);

  // Connect two nodes
  const connect = useCallback((fromId: string, toId: string, gain = 1.0) => {
    if (!audioContextRef.current) return;

    const fromNode = nodesRef.current.get(fromId);
    const toNode = nodesRef.current.get(toId);

    if (!fromNode || !toNode) return;

    const routeKey = `${fromId}->${toId}`;

    // Create a gain node for this route
    const routeGain = audioContextRef.current.createGain();
    routeGain.gain.value = gain;

    try {
      (fromNode.node as any).connect(routeGain);
      routeGain.connect(toNode.node as any);
      routesRef.current.set(routeKey, routeGain);

      setState(prev => ({
        ...prev,
        activeRoutes: [...prev.activeRoutes, { from: fromId, to: toId, gain }],
      }));
    } catch (e) {
      console.error('Failed to connect nodes:', e);
    }
  }, []);

  // Disconnect two nodes
  const disconnect = useCallback((fromId: string, toId: string) => {
    const routeKey = `${fromId}->${toId}`;
    const routeGain = routesRef.current.get(routeKey);

    if (routeGain) {
      try {
        routeGain.disconnect();
      } catch (e) {}
      routesRef.current.delete(routeKey);

      setState(prev => ({
        ...prev,
        activeRoutes: prev.activeRoutes.filter(
          r => !(r.from === fromId && r.to === toId)
        ),
      }));
    }
  }, []);

  // Set route gain
  const setRouteGain = useCallback((fromId: string, toId: string, gain: number) => {
    const routeKey = `${fromId}->${toId}`;
    const routeGain = routesRef.current.get(routeKey);

    if (routeGain) {
      routeGain.gain.setValueAtTime(gain, audioContextRef.current?.currentTime || 0);

      setState(prev => ({
        ...prev,
        activeRoutes: prev.activeRoutes.map(r =>
          r.from === fromId && r.to === toId ? { ...r, gain } : r
        ),
      }));
    }
  }, []);

  // Set master volume
  const setMasterVolume = useCallback((volume: number) => {
    if (masterGainRef.current) {
      masterGainRef.current.gain.setValueAtTime(
        state.isMuted ? 0 : volume,
        audioContextRef.current?.currentTime || 0
      );
    }
    setState(prev => ({ ...prev, masterVolume: volume }));
  }, [state.isMuted]);

  // Set muted
  const setMuted = useCallback((muted: boolean) => {
    if (masterGainRef.current) {
      masterGainRef.current.gain.setValueAtTime(
        muted ? 0 : state.masterVolume,
        audioContextRef.current?.currentTime || 0
      );
    }
    setState(prev => ({ ...prev, isMuted: muted }));
  }, [state.masterVolume]);

  // Get stream destination for broadcasting
  const getStreamDestination = useCallback(() => {
    return streamDestinationRef.current;
  }, []);

  // Get broadcast stream
  const getBroadcastStream = useCallback(() => {
    return streamDestinationRef.current?.stream || null;
  }, []);

  // Get current audio level
  const getAudioLevel = useCallback(() => {
    return state.audioLevel;
  }, [state.audioLevel]);

  // Get frequency data
  const getFrequencyData = useCallback(() => {
    if (!masterAnalyserRef.current) return null;
    const data = new Uint8Array(masterAnalyserRef.current.frequencyBinCount);
    masterAnalyserRef.current.getByteFrequencyData(data);
    return data;
  }, []);

  // Get waveform data
  const getWaveformData = useCallback(() => {
    if (!masterAnalyserRef.current) return null;
    const data = new Uint8Array(masterAnalyserRef.current.frequencyBinCount);
    masterAnalyserRef.current.getByteTimeDomainData(data);
    return data;
  }, []);

  const value: AudioPipelineContextType = {
    state,
    audioContext: audioContextRef.current,
    masterGain: masterGainRef.current,
    masterAnalyser: masterAnalyserRef.current,
    registerNode,
    unregisterNode,
    getNode,
    connect,
    disconnect,
    setRouteGain,
    setMasterVolume,
    setMuted,
    getStreamDestination,
    getBroadcastStream,
    getAudioLevel,
    getFrequencyData,
    getWaveformData,
  };

  return (
    <AudioPipelineContext.Provider value={value}>
      {children}
    </AudioPipelineContext.Provider>
  );
}

// Export a simple audio level indicator component
export function AudioLevelMeter({ className = '' }: { className?: string }) {
  const { state } = useAudioPipeline();

  return (
    <div className={`h-2 bg-gray-800 rounded-full overflow-hidden ${className}`}>
      <div
        className="h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 transition-all duration-75"
        style={{ width: `${state.audioLevel * 100}%` }}
      />
    </div>
  );
}
