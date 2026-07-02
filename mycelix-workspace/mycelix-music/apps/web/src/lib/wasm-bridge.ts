// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WASM Bridge
 *
 * Connects frontend hooks to Rust/WASM backend services.
 * Provides a unified interface for audio processing, ML inference,
 * and music analysis.
 */

import type { MusicStyle, GenerationParams } from '@/hooks/useAIGeneration';
import type { RoomPreset, Position3D } from '@/hooks/useSpatialAudio';
import type { MasteringChain, TargetPlatform } from '@/hooks/useAIMastering';
import type { MusicalKey, Chord } from '@/hooks/useMusicTheory';

// WASM Module Types
interface MycelixWasm {
  // Audio Processing
  process_audio: (buffer: Float32Array, sampleRate: number) => Float32Array;
  apply_effects: (buffer: Float32Array, effects: string) => Float32Array;
  separate_stems: (buffer: Float32Array, stemType: string) => Float32Array;

  // ML/AI
  detect_genre: (buffer: Float32Array) => string;
  detect_mood: (buffer: Float32Array) => string;
  detect_key: (buffer: Float32Array) => { root: string; mode: string; confidence: number };
  detect_bpm: (buffer: Float32Array) => number;
  detect_chords: (buffer: Float32Array) => Array<{ root: string; quality: string; time: number }>;
  generate_embedding: (buffer: Float32Array) => Float32Array;

  // Spatial Audio
  apply_hrtf: (buffer: Float32Array, position: string) => Float32Array;
  apply_reverb: (buffer: Float32Array, roomPreset: string) => Float32Array;
  encode_ambisonics: (buffer: Float32Array, position: string) => Float32Array;

  // Mastering
  analyze_loudness: (buffer: Float32Array) => { lufs: number; peak: number; range: number };
  apply_mastering_chain: (buffer: Float32Array, chain: string) => Float32Array;
  match_reference: (source: Float32Array, reference: Float32Array) => Float32Array;

  // Fingerprinting
  generate_fingerprint: (buffer: Float32Array) => Uint8Array;
  match_fingerprint: (fingerprint: Uint8Array) => string | null;

  // Memory Management
  alloc: (size: number) => number;
  free: (ptr: number) => void;
}

// Singleton WASM instance
let wasmModule: MycelixWasm | null = null;
let isLoading = false;
let loadPromise: Promise<MycelixWasm> | null = null;

/**
 * Load the WASM module
 */
export async function loadWasm(): Promise<MycelixWasm> {
  if (wasmModule) return wasmModule;

  if (loadPromise) return loadPromise;

  isLoading = true;
  loadPromise = new Promise(async (resolve, reject) => {
    try {
      // Dynamic import of WASM module
      const wasm = await import('@mycelix/wasm');
      await wasm.default();

      wasmModule = wasm as unknown as MycelixWasm;
      isLoading = false;
      resolve(wasmModule);
    } catch (error) {
      isLoading = false;
      loadPromise = null;

      // Fallback to JS implementation if WASM fails
      console.warn('WASM loading failed, using JS fallback:', error);
      wasmModule = createJsFallback();
      resolve(wasmModule);
    }
  });

  return loadPromise;
}

/**
 * Check if WASM is loaded
 */
export function isWasmLoaded(): boolean {
  return wasmModule !== null;
}

/**
 * Check if WASM is currently loading
 */
export function isWasmLoading(): boolean {
  return isLoading;
}

/**
 * Create JS fallback implementations for when WASM is not available
 */
function createJsFallback(): MycelixWasm {
  return {
    process_audio: (buffer) => buffer,
    apply_effects: (buffer) => buffer,
    separate_stems: (buffer) => buffer,

    detect_genre: () => 'electronic',
    detect_mood: () => 'energetic',
    detect_key: () => ({ root: 'C', mode: 'major', confidence: 0.8 }),
    detect_bpm: () => 120,
    detect_chords: () => [],
    generate_embedding: () => new Float32Array(256),

    apply_hrtf: (buffer) => buffer,
    apply_reverb: (buffer) => buffer,
    encode_ambisonics: (buffer) => buffer,

    analyze_loudness: () => ({ lufs: -14, peak: -1, range: 8 }),
    apply_mastering_chain: (buffer) => buffer,
    match_reference: (source) => source,

    generate_fingerprint: () => new Uint8Array(32),
    match_fingerprint: () => null,

    alloc: () => 0,
    free: () => {},
  };
}

// ============================================================================
// High-Level API Functions
// ============================================================================

/**
 * Audio Analysis API
 */
export const AudioAnalysis = {
  async detectKey(audioData: Float32Array): Promise<MusicalKey> {
    const wasm = await loadWasm();
    const result = wasm.detect_key(audioData);

    return {
      root: result.root as any,
      mode: result.mode as any,
      confidence: result.confidence,
      alternates: [],
    };
  },

  async detectBPM(audioData: Float32Array): Promise<number> {
    const wasm = await loadWasm();
    return wasm.detect_bpm(audioData);
  },

  async detectChords(audioData: Float32Array): Promise<Chord[]> {
    const wasm = await loadWasm();
    const chords = wasm.detect_chords(audioData);

    return chords.map((c, i) => ({
      root: c.root as any,
      quality: c.quality as any,
      time: c.time,
      duration: i < chords.length - 1 ? chords[i + 1].time - c.time : 1,
      confidence: 0.8,
    }));
  },

  async detectGenre(audioData: Float32Array): Promise<string> {
    const wasm = await loadWasm();
    return wasm.detect_genre(audioData);
  },

  async detectMood(audioData: Float32Array): Promise<string> {
    const wasm = await loadWasm();
    return wasm.detect_mood(audioData);
  },

  async generateEmbedding(audioData: Float32Array): Promise<Float32Array> {
    const wasm = await loadWasm();
    return wasm.generate_embedding(audioData);
  },

  async analyzeLoudness(audioData: Float32Array): Promise<{
    lufs: number;
    peak: number;
    range: number;
  }> {
    const wasm = await loadWasm();
    return wasm.analyze_loudness(audioData);
  },
};

/**
 * Stem Separation API
 */
export const StemSeparation = {
  async separate(
    audioData: Float32Array,
    stemType: 'vocals' | 'drums' | 'bass' | 'other'
  ): Promise<Float32Array> {
    const wasm = await loadWasm();
    return wasm.separate_stems(audioData, stemType);
  },

  async separateAll(audioData: Float32Array): Promise<{
    vocals: Float32Array;
    drums: Float32Array;
    bass: Float32Array;
    other: Float32Array;
  }> {
    const wasm = await loadWasm();

    const [vocals, drums, bass, other] = await Promise.all([
      wasm.separate_stems(audioData, 'vocals'),
      wasm.separate_stems(audioData, 'drums'),
      wasm.separate_stems(audioData, 'bass'),
      wasm.separate_stems(audioData, 'other'),
    ]);

    return { vocals, drums, bass, other };
  },
};

/**
 * Spatial Audio API
 */
export const SpatialAudio = {
  async applyHRTF(
    audioData: Float32Array,
    position: Position3D
  ): Promise<Float32Array> {
    const wasm = await loadWasm();
    return wasm.apply_hrtf(audioData, JSON.stringify(position));
  },

  async applyReverb(
    audioData: Float32Array,
    roomPreset: RoomPreset
  ): Promise<Float32Array> {
    const wasm = await loadWasm();
    return wasm.apply_reverb(audioData, roomPreset);
  },

  async encodeAmbisonics(
    audioData: Float32Array,
    position: Position3D
  ): Promise<Float32Array> {
    const wasm = await loadWasm();
    return wasm.encode_ambisonics(audioData, JSON.stringify(position));
  },
};

/**
 * Mastering API
 */
export const Mastering = {
  async applyChain(
    audioData: Float32Array,
    chain: MasteringChain
  ): Promise<Float32Array> {
    const wasm = await loadWasm();
    return wasm.apply_mastering_chain(audioData, JSON.stringify(chain));
  },

  async matchReference(
    source: Float32Array,
    reference: Float32Array
  ): Promise<Float32Array> {
    const wasm = await loadWasm();
    return wasm.match_reference(source, reference);
  },

  async analyzeLoudness(audioData: Float32Array): Promise<{
    lufs: number;
    peak: number;
    range: number;
  }> {
    const wasm = await loadWasm();
    return wasm.analyze_loudness(audioData);
  },
};

/**
 * Effects Processing API
 */
export const Effects = {
  async apply(
    audioData: Float32Array,
    effects: Array<{ type: string; params: Record<string, number> }>
  ): Promise<Float32Array> {
    const wasm = await loadWasm();
    return wasm.apply_effects(audioData, JSON.stringify(effects));
  },
};

/**
 * Fingerprinting API
 */
export const Fingerprinting = {
  async generate(audioData: Float32Array): Promise<Uint8Array> {
    const wasm = await loadWasm();
    return wasm.generate_fingerprint(audioData);
  },

  async match(fingerprint: Uint8Array): Promise<string | null> {
    const wasm = await loadWasm();
    return wasm.match_fingerprint(fingerprint);
  },
};

// ============================================================================
// WebSocket API for Real-time Features
// ============================================================================

type MessageHandler = (data: any) => void;

class RealtimeConnection {
  private ws: WebSocket | null = null;
  private handlers: Map<string, Set<MessageHandler>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor(private url: string) {}

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        resolve();
      };

      this.ws.onerror = (error) => {
        reject(error);
      };

      this.ws.onclose = () => {
        this.attemptReconnect();
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          const handlers = this.handlers.get(message.type);
          handlers?.forEach((handler) => handler(message.data));
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      };
    });
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    setTimeout(() => {
      this.connect().catch(console.error);
    }, delay);
  }

  send(type: string, data: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, data }));
    }
  }

  on(type: string, handler: MessageHandler) {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);

    return () => {
      this.handlers.get(type)?.delete(handler);
    };
  }

  disconnect() {
    this.ws?.close();
    this.ws = null;
  }
}

// Connection instances
let collaborationConnection: RealtimeConnection | null = null;
let broadcastConnection: RealtimeConnection | null = null;

export const Realtime = {
  async connectCollaboration(sessionId: string): Promise<RealtimeConnection> {
    collaborationConnection = new RealtimeConnection(
      `wss://api.mycelix.music/collab/${sessionId}`
    );
    await collaborationConnection.connect();
    return collaborationConnection;
  },

  async connectBroadcast(channelId: string): Promise<RealtimeConnection> {
    broadcastConnection = new RealtimeConnection(
      `wss://api.mycelix.music/broadcast/${channelId}`
    );
    await broadcastConnection.connect();
    return broadcastConnection;
  },

  getCollaborationConnection(): RealtimeConnection | null {
    return collaborationConnection;
  },

  getBroadcastConnection(): RealtimeConnection | null {
    return broadcastConnection;
  },

  disconnectAll() {
    collaborationConnection?.disconnect();
    broadcastConnection?.disconnect();
    collaborationConnection = null;
    broadcastConnection = null;
  },
};

// Export default
export default {
  loadWasm,
  isWasmLoaded,
  isWasmLoading,
  AudioAnalysis,
  StemSeparation,
  SpatialAudio,
  Mastering,
  Effects,
  Fingerprinting,
  Realtime,
};
