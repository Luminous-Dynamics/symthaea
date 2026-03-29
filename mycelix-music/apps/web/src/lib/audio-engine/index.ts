// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Advanced Audio Engine
 *
 * Professional audio processing system:
 * - Node-based processing graph
 * - Audio Worklet low-latency DSP
 * - MIDI input/output
 * - Plugin architecture (VST-like)
 * - Real-time parameter automation
 */

// Audio Graph
export {
  AudioGraph,
  AudioNodeWrapper,
  type AudioNodeType,
  type AudioNodeConfig,
  type AudioConnection,
  type AudioGraphState,
} from './graph';

// MIDI
export {
  MIDIManager,
  MIDIClock,
  VirtualKeyboard,
  getMIDIManager,
  parseMIDIMessage,
  type MIDIMessage,
  type MIDINoteMessage,
  type MIDIControlMessage,
  type MIDIDeviceInfo,
  type MIDILearnBinding,
  type MIDIMessageType,
} from './midi';

// Plugins
export {
  AudioPlugin,
  ParametricEQ,
  CompressorPlugin,
  ReverbPlugin,
  DelayPlugin,
  PluginHost,
  PluginRegistry,
  createDefaultRegistry,
  type PluginCategory,
  type PluginParameter,
  type PluginPreset,
  type PluginMetadata,
} from './plugins';

// ==================== Engine Class ====================

import { AudioGraph } from './graph';
import { MIDIManager, getMIDIManager, VirtualKeyboard } from './midi';
import { PluginHost, createDefaultRegistry, PluginRegistry } from './plugins';

export interface AudioEngineConfig {
  sampleRate?: number;
  bufferSize?: number;
  enableMIDI?: boolean;
  enableVirtualKeyboard?: boolean;
}

export class AudioEngine {
  public readonly graph: AudioGraph;
  public readonly midi: MIDIManager;
  public readonly plugins: PluginHost;
  public readonly registry: PluginRegistry;
  public readonly virtualKeyboard: VirtualKeyboard | null;

  private isInitialized = false;

  constructor(config: AudioEngineConfig = {}) {
    this.graph = new AudioGraph(config.sampleRate);
    this.midi = getMIDIManager();
    this.plugins = new PluginHost(this.graph.getContext());
    this.registry = createDefaultRegistry();
    this.virtualKeyboard = config.enableVirtualKeyboard ? new VirtualKeyboard() : null;
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    // Load audio worklets
    try {
      await this.graph.getContext().audioWorklet.addModule('/audio-worklets/processor.js');
    } catch (e) {
      console.warn('Audio worklets not available:', e);
    }

    // Initialize MIDI
    try {
      await this.midi.initialize();
    } catch (e) {
      console.warn('MIDI not available:', e);
    }

    this.isInitialized = true;
  }

  async start(): Promise<void> {
    await this.graph.start();
  }

  stop(): void {
    this.graph.stop();
  }

  dispose(): void {
    this.graph.dispose();
    this.midi.dispose();
    this.plugins.dispose();
  }
}

// ==================== Singleton ====================

let engineInstance: AudioEngine | null = null;

export function getAudioEngine(config?: AudioEngineConfig): AudioEngine {
  if (!engineInstance) {
    engineInstance = new AudioEngine(config);
  }
  return engineInstance;
}

export default {
  AudioEngine,
  getAudioEngine,
};
