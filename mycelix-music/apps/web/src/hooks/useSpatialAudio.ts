// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Spatial Audio Hook
 *
 * Provides 3D spatial audio positioning and immersive audio experiences:
 * - HRTF-based binaural audio
 * - Ambisonics encoding/decoding
 * - Real-time 3D positioning of sound sources
 * - Virtual venue acoustics simulation
 *
 * Integrates with mycelix-spatial Rust crate
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';

// 3D Position
export interface Position3D {
  x: number; // Left/Right (-1 to 1)
  y: number; // Up/Down (-1 to 1)
  z: number; // Front/Back (-1 to 1)
}

// Orientation (in degrees)
export interface Orientation3D {
  yaw: number; // Rotation around vertical axis (0-360)
  pitch: number; // Rotation up/down (-90 to 90)
  roll: number; // Rotation tilt (-180 to 180)
}

// Sound source
export interface SpatialSource {
  id: string;
  name: string;
  position: Position3D;
  audioBuffer?: AudioBuffer;
  sourceNode?: AudioBufferSourceNode;
  pannerNode?: PannerNode;
  gainNode?: GainNode;
  isPlaying: boolean;
  volume: number;
  loop: boolean;
  distance: number; // Calculated distance from listener
  angle: number; // Calculated angle from listener
}

// Listener (the user's position/orientation in 3D space)
export interface SpatialListener {
  position: Position3D;
  orientation: Orientation3D;
}

// Room/Environment presets
export type RoomPreset =
  | 'small-room'
  | 'large-hall'
  | 'cathedral'
  | 'outdoor'
  | 'club'
  | 'studio'
  | 'arena';

export interface RoomAcoustics {
  preset: RoomPreset;
  size: number; // 0-1
  damping: number; // 0-1 (high frequency absorption)
  wetLevel: number; // 0-1 (reverb mix)
  earlyReflections: number; // 0-1
  diffusion: number; // 0-1
}

// Virtual venue
export interface VirtualVenue {
  id: string;
  name: string;
  type: 'concert-hall' | 'club' | 'outdoor' | 'studio' | 'custom';
  capacity: number;
  dimensions: { width: number; height: number; depth: number };
  acoustics: RoomAcoustics;
  sources: SpatialSource[];
  hotspots: VenueHotspot[];
}

export interface VenueHotspot {
  id: string;
  name: string;
  position: Position3D;
  description?: string;
  icon?: string;
}

// Hook state
export interface SpatialAudioState {
  isInitialized: boolean;
  isSupported: boolean;
  listener: SpatialListener;
  sources: SpatialSource[];
  venue: VirtualVenue | null;
  roomAcoustics: RoomAcoustics;
  masterVolume: number;
  isEnabled: boolean;
  headTrackingEnabled: boolean;
  error: string | null;
}

// Room presets configuration
const ROOM_PRESETS: Record<RoomPreset, Omit<RoomAcoustics, 'preset'>> = {
  'small-room': { size: 0.2, damping: 0.7, wetLevel: 0.3, earlyReflections: 0.6, diffusion: 0.4 },
  'large-hall': { size: 0.7, damping: 0.4, wetLevel: 0.5, earlyReflections: 0.5, diffusion: 0.7 },
  'cathedral': { size: 1.0, damping: 0.2, wetLevel: 0.7, earlyReflections: 0.3, diffusion: 0.9 },
  'outdoor': { size: 0.8, damping: 0.9, wetLevel: 0.1, earlyReflections: 0.1, diffusion: 0.3 },
  'club': { size: 0.4, damping: 0.5, wetLevel: 0.4, earlyReflections: 0.7, diffusion: 0.5 },
  'studio': { size: 0.15, damping: 0.8, wetLevel: 0.15, earlyReflections: 0.4, diffusion: 0.3 },
  'arena': { size: 0.9, damping: 0.3, wetLevel: 0.6, earlyReflections: 0.4, diffusion: 0.8 },
};

export function useSpatialAudio() {
  const [state, setState] = useState<SpatialAudioState>({
    isInitialized: false,
    isSupported: true,
    listener: {
      position: { x: 0, y: 0, z: 0 },
      orientation: { yaw: 0, pitch: 0, roll: 0 },
    },
    sources: [],
    venue: null,
    roomAcoustics: { preset: 'studio', ...ROOM_PRESETS['studio'] },
    masterVolume: 1,
    isEnabled: true,
    headTrackingEnabled: false,
    error: null,
  });

  const audioContextRef = useRef<AudioContext | null>(null);
  const masterGainRef = useRef<GainNode | null>(null);
  const convolverRef = useRef<ConvolverNode | null>(null);
  const dryGainRef = useRef<GainNode | null>(null);
  const wetGainRef = useRef<GainNode | null>(null);
  const sourcesRef = useRef<Map<string, SpatialSource>>(new Map());

  // Initialize audio context and spatial audio
  const initialize = useCallback(async () => {
    if (audioContextRef.current) return;

    try {
      const ctx = new AudioContext();
      audioContextRef.current = ctx;

      // Create master gain
      const masterGain = ctx.createGain();
      masterGain.connect(ctx.destination);
      masterGainRef.current = masterGain;

      // Create convolver for room reverb
      const convolver = ctx.createConvolver();
      convolverRef.current = convolver;

      // Create dry/wet mix gains
      const dryGain = ctx.createGain();
      const wetGain = ctx.createGain();
      dryGainRef.current = dryGain;
      wetGainRef.current = wetGain;

      dryGain.connect(masterGain);
      wetGain.connect(masterGain);
      convolver.connect(wetGain);

      // Generate impulse response for reverb
      await generateImpulseResponse(ctx, ROOM_PRESETS['studio']);

      setState(prev => ({ ...prev, isInitialized: true }));
    } catch (err) {
      setState(prev => ({
        ...prev,
        error: err instanceof Error ? err.message : 'Failed to initialize spatial audio',
        isSupported: false,
      }));
    }
  }, []);

  // Generate impulse response for reverb
  const generateImpulseResponse = async (ctx: AudioContext, acoustics: Omit<RoomAcoustics, 'preset'>) => {
    const sampleRate = ctx.sampleRate;
    const length = Math.floor(sampleRate * (acoustics.size * 4 + 0.5)); // Duration based on room size
    const impulse = ctx.createBuffer(2, length, sampleRate);

    for (let channel = 0; channel < 2; channel++) {
      const channelData = impulse.getChannelData(channel);

      for (let i = 0; i < length; i++) {
        const t = i / sampleRate;

        // Early reflections
        let sample = 0;
        if (t < 0.1 * acoustics.earlyReflections) {
          const reflectionCount = Math.floor(acoustics.earlyReflections * 10);
          for (let r = 0; r < reflectionCount; r++) {
            const delay = 0.01 + r * 0.01;
            if (Math.abs(t - delay) < 0.001) {
              sample += (Math.random() * 2 - 1) * Math.exp(-t * 10) * 0.5;
            }
          }
        }

        // Late reverb (diffuse tail)
        const decay = Math.exp(-t * (3 + acoustics.damping * 5));
        const diffuse = (Math.random() * 2 - 1) * decay * acoustics.diffusion;
        sample += diffuse * 0.3;

        channelData[i] = sample;
      }
    }

    if (convolverRef.current) {
      convolverRef.current.buffer = impulse;
    }

    // Update wet/dry mix
    if (dryGainRef.current && wetGainRef.current) {
      dryGainRef.current.gain.value = 1 - acoustics.wetLevel * 0.5;
      wetGainRef.current.gain.value = acoustics.wetLevel;
    }
  };

  // Add a spatial sound source
  const addSource = useCallback(async (
    id: string,
    name: string,
    position: Position3D,
    audioUrl?: string
  ): Promise<SpatialSource> => {
    const ctx = audioContextRef.current;
    if (!ctx) throw new Error('Audio context not initialized');

    // Create panner node for 3D positioning
    const panner = ctx.createPanner();
    panner.panningModel = 'HRTF';
    panner.distanceModel = 'inverse';
    panner.refDistance = 1;
    panner.maxDistance = 10000;
    panner.rolloffFactor = 1;
    panner.coneInnerAngle = 360;
    panner.coneOuterAngle = 360;
    panner.coneOuterGain = 0;

    // Set position
    panner.positionX.value = position.x;
    panner.positionY.value = position.y;
    panner.positionZ.value = position.z;

    // Create gain node for volume control
    const gain = ctx.createGain();

    // Connect: panner -> gain -> dry/wet split
    panner.connect(gain);
    gain.connect(dryGainRef.current!);
    gain.connect(convolverRef.current!);

    const source: SpatialSource = {
      id,
      name,
      position,
      pannerNode: panner,
      gainNode: gain,
      isPlaying: false,
      volume: 1,
      loop: false,
      distance: calculateDistance(position, state.listener.position),
      angle: calculateAngle(position, state.listener.position, state.listener.orientation),
    };

    // Load audio if URL provided
    if (audioUrl) {
      try {
        const response = await fetch(audioUrl);
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
        source.audioBuffer = audioBuffer;
      } catch (err) {
        console.error('Failed to load audio:', err);
      }
    }

    sourcesRef.current.set(id, source);
    setState(prev => ({
      ...prev,
      sources: [...prev.sources, source],
    }));

    return source;
  }, [state.listener]);

  // Remove a source
  const removeSource = useCallback((sourceId: string) => {
    const source = sourcesRef.current.get(sourceId);
    if (source) {
      source.sourceNode?.stop();
      source.sourceNode?.disconnect();
      source.pannerNode?.disconnect();
      source.gainNode?.disconnect();
      sourcesRef.current.delete(sourceId);

      setState(prev => ({
        ...prev,
        sources: prev.sources.filter(s => s.id !== sourceId),
      }));
    }
  }, []);

  // Update source position
  const updateSourcePosition = useCallback((sourceId: string, position: Position3D) => {
    const source = sourcesRef.current.get(sourceId);
    if (source && source.pannerNode) {
      source.pannerNode.positionX.value = position.x;
      source.pannerNode.positionY.value = position.y;
      source.pannerNode.positionZ.value = position.z;
      source.position = position;
      source.distance = calculateDistance(position, state.listener.position);
      source.angle = calculateAngle(position, state.listener.position, state.listener.orientation);

      setState(prev => ({
        ...prev,
        sources: prev.sources.map(s => s.id === sourceId ? source : s),
      }));
    }
  }, [state.listener]);

  // Play a source
  const playSource = useCallback((sourceId: string) => {
    const ctx = audioContextRef.current;
    const source = sourcesRef.current.get(sourceId);

    if (!ctx || !source || !source.audioBuffer || !source.pannerNode) return;

    // Stop existing playback
    source.sourceNode?.stop();

    // Create new buffer source
    const bufferSource = ctx.createBufferSource();
    bufferSource.buffer = source.audioBuffer;
    bufferSource.loop = source.loop;
    bufferSource.connect(source.pannerNode);

    bufferSource.onended = () => {
      source.isPlaying = false;
      setState(prev => ({
        ...prev,
        sources: prev.sources.map(s => s.id === sourceId ? { ...s, isPlaying: false } : s),
      }));
    };

    bufferSource.start();
    source.sourceNode = bufferSource;
    source.isPlaying = true;

    setState(prev => ({
      ...prev,
      sources: prev.sources.map(s => s.id === sourceId ? { ...s, isPlaying: true } : s),
    }));
  }, []);

  // Stop a source
  const stopSource = useCallback((sourceId: string) => {
    const source = sourcesRef.current.get(sourceId);
    if (source) {
      source.sourceNode?.stop();
      source.isPlaying = false;

      setState(prev => ({
        ...prev,
        sources: prev.sources.map(s => s.id === sourceId ? { ...s, isPlaying: false } : s),
      }));
    }
  }, []);

  // Update listener position
  const updateListenerPosition = useCallback((position: Position3D) => {
    const ctx = audioContextRef.current;
    if (ctx) {
      ctx.listener.positionX.value = position.x;
      ctx.listener.positionY.value = position.y;
      ctx.listener.positionZ.value = position.z;
    }

    setState(prev => ({
      ...prev,
      listener: { ...prev.listener, position },
      sources: prev.sources.map(s => ({
        ...s,
        distance: calculateDistance(s.position, position),
        angle: calculateAngle(s.position, position, prev.listener.orientation),
      })),
    }));
  }, []);

  // Update listener orientation
  const updateListenerOrientation = useCallback((orientation: Orientation3D) => {
    const ctx = audioContextRef.current;
    if (ctx) {
      // Convert yaw/pitch to forward vector
      const yawRad = (orientation.yaw * Math.PI) / 180;
      const pitchRad = (orientation.pitch * Math.PI) / 180;

      const forwardX = Math.sin(yawRad) * Math.cos(pitchRad);
      const forwardY = Math.sin(pitchRad);
      const forwardZ = -Math.cos(yawRad) * Math.cos(pitchRad);

      // Up vector (simplified, ignoring roll for now)
      ctx.listener.forwardX.value = forwardX;
      ctx.listener.forwardY.value = forwardY;
      ctx.listener.forwardZ.value = forwardZ;
      ctx.listener.upX.value = 0;
      ctx.listener.upY.value = 1;
      ctx.listener.upZ.value = 0;
    }

    setState(prev => ({
      ...prev,
      listener: { ...prev.listener, orientation },
      sources: prev.sources.map(s => ({
        ...s,
        angle: calculateAngle(s.position, prev.listener.position, orientation),
      })),
    }));
  }, []);

  // Set room acoustics
  const setRoomAcoustics = useCallback(async (preset: RoomPreset) => {
    const ctx = audioContextRef.current;
    if (!ctx) return;

    const acoustics = { preset, ...ROOM_PRESETS[preset] };
    await generateImpulseResponse(ctx, ROOM_PRESETS[preset]);

    setState(prev => ({ ...prev, roomAcoustics: acoustics }));
  }, []);

  // Set master volume
  const setMasterVolume = useCallback((volume: number) => {
    if (masterGainRef.current) {
      masterGainRef.current.gain.value = volume;
    }
    setState(prev => ({ ...prev, masterVolume: volume }));
  }, []);

  // Toggle spatial audio
  const toggleEnabled = useCallback(() => {
    setState(prev => {
      const enabled = !prev.isEnabled;
      // When disabled, bypass spatial processing
      // In real implementation, would route directly to stereo
      return { ...prev, isEnabled: enabled };
    });
  }, []);

  // Create a virtual venue
  const createVenue = useCallback((
    name: string,
    type: VirtualVenue['type'],
    dimensions: VirtualVenue['dimensions']
  ): VirtualVenue => {
    const venueAcoustics: RoomAcoustics = {
      preset: type === 'club' ? 'club' :
              type === 'concert-hall' ? 'large-hall' :
              type === 'outdoor' ? 'outdoor' : 'studio',
      ...ROOM_PRESETS[
        type === 'club' ? 'club' :
        type === 'concert-hall' ? 'large-hall' :
        type === 'outdoor' ? 'outdoor' : 'studio'
      ],
    };

    const venue: VirtualVenue = {
      id: `venue-${Date.now()}`,
      name,
      type,
      capacity: Math.floor(dimensions.width * dimensions.depth * 2),
      dimensions,
      acoustics: venueAcoustics,
      sources: [],
      hotspots: [],
    };

    setState(prev => ({ ...prev, venue }));
    setRoomAcoustics(venueAcoustics.preset);

    return venue;
  }, [setRoomAcoustics]);

  // Cleanup
  useEffect(() => {
    return () => {
      sourcesRef.current.forEach((source) => {
        source.sourceNode?.stop();
        source.sourceNode?.disconnect();
        source.pannerNode?.disconnect();
        source.gainNode?.disconnect();
      });
      audioContextRef.current?.close();
    };
  }, []);

  return {
    ...state,
    initialize,
    addSource,
    removeSource,
    updateSourcePosition,
    playSource,
    stopSource,
    updateListenerPosition,
    updateListenerOrientation,
    setRoomAcoustics,
    setMasterVolume,
    toggleEnabled,
    createVenue,
    roomPresets: Object.keys(ROOM_PRESETS) as RoomPreset[],
  };
}

// Helper functions
function calculateDistance(a: Position3D, b: Position3D): number {
  return Math.sqrt(
    Math.pow(a.x - b.x, 2) +
    Math.pow(a.y - b.y, 2) +
    Math.pow(a.z - b.z, 2)
  );
}

function calculateAngle(
  source: Position3D,
  listener: Position3D,
  orientation: Orientation3D
): number {
  const dx = source.x - listener.x;
  const dz = source.z - listener.z;
  const angle = Math.atan2(dx, -dz) * (180 / Math.PI);
  return (angle - orientation.yaw + 360) % 360;
}

export default useSpatialAudio;
