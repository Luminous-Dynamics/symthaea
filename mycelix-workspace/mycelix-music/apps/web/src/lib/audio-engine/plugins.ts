// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Audio Plugin System
 *
 * VST-like plugin architecture:
 * - Plugin base class
 * - Built-in effects
 * - Instrument plugins
 * - Plugin hosting
 * - Preset management
 * - Parameter automation
 */

// ==================== Types ====================

export type PluginCategory = 'effect' | 'instrument' | 'analyzer' | 'utility';

export interface PluginParameter {
  id: string;
  name: string;
  type: 'float' | 'int' | 'bool' | 'enum';
  defaultValue: number;
  minValue?: number;
  maxValue?: number;
  step?: number;
  unit?: string;
  options?: { value: number; label: string }[];
}

export interface PluginPreset {
  id: string;
  name: string;
  author?: string;
  parameters: Record<string, number>;
  tags?: string[];
}

export interface PluginMetadata {
  id: string;
  name: string;
  version: string;
  author: string;
  category: PluginCategory;
  description: string;
  parameters: PluginParameter[];
  presets: PluginPreset[];
  inputs: number;
  outputs: number;
  sidechain?: boolean;
}

// ==================== Base Plugin Class ====================

export abstract class AudioPlugin {
  public readonly metadata: PluginMetadata;
  protected context: AudioContext | null = null;
  protected inputNode: GainNode | null = null;
  protected outputNode: GainNode | null = null;
  protected parameters: Map<string, number> = new Map();
  protected bypassed = false;

  public onParameterChange?: (paramId: string, value: number) => void;

  constructor(metadata: PluginMetadata) {
    this.metadata = metadata;

    // Initialize parameters with defaults
    for (const param of metadata.parameters) {
      this.parameters.set(param.id, param.defaultValue);
    }
  }

  async initialize(context: AudioContext): Promise<void> {
    this.context = context;

    // Create I/O nodes
    this.inputNode = context.createGain();
    this.outputNode = context.createGain();

    await this.setup();
  }

  protected abstract setup(): Promise<void>;

  get input(): AudioNode {
    return this.inputNode!;
  }

  get output(): AudioNode {
    return this.outputNode!;
  }

  getParameter(id: string): number {
    return this.parameters.get(id) ?? 0;
  }

  setParameter(id: string, value: number, rampTime = 0): void {
    const param = this.metadata.parameters.find(p => p.id === id);
    if (!param) return;

    // Clamp value
    if (param.minValue !== undefined) {
      value = Math.max(param.minValue, value);
    }
    if (param.maxValue !== undefined) {
      value = Math.min(param.maxValue, value);
    }

    this.parameters.set(id, value);
    this.applyParameter(id, value, rampTime);
    this.onParameterChange?.(id, value);
  }

  protected abstract applyParameter(id: string, value: number, rampTime: number): void;

  getAllParameters(): Record<string, number> {
    return Object.fromEntries(this.parameters);
  }

  loadPreset(preset: PluginPreset): void {
    for (const [id, value] of Object.entries(preset.parameters)) {
      this.setParameter(id, value);
    }
  }

  savePreset(name: string): PluginPreset {
    return {
      id: `preset-${Date.now()}`,
      name,
      parameters: this.getAllParameters(),
    };
  }

  setBypass(bypass: boolean): void {
    this.bypassed = bypass;
  }

  isActive(): boolean {
    return !this.bypassed && this.context !== null;
  }

  dispose(): void {
    this.inputNode?.disconnect();
    this.outputNode?.disconnect();
  }
}

// ==================== Built-in Effects ====================

// Parametric EQ
export class ParametricEQ extends AudioPlugin {
  private bands: BiquadFilterNode[] = [];

  static createMetadata(): PluginMetadata {
    const bandParams: PluginParameter[] = [];

    for (let i = 1; i <= 4; i++) {
      bandParams.push(
        { id: `band${i}_freq`, name: `Band ${i} Freq`, type: 'float', defaultValue: 250 * Math.pow(4, i - 1), minValue: 20, maxValue: 20000, unit: 'Hz' },
        { id: `band${i}_gain`, name: `Band ${i} Gain`, type: 'float', defaultValue: 0, minValue: -18, maxValue: 18, unit: 'dB' },
        { id: `band${i}_q`, name: `Band ${i} Q`, type: 'float', defaultValue: 1, minValue: 0.1, maxValue: 18 },
        { id: `band${i}_type`, name: `Band ${i} Type`, type: 'enum', defaultValue: 0, options: [
          { value: 0, label: 'Peak' },
          { value: 1, label: 'Low Shelf' },
          { value: 2, label: 'High Shelf' },
          { value: 3, label: 'Low Pass' },
          { value: 4, label: 'High Pass' },
        ]}
      );
    }

    return {
      id: 'parametric-eq',
      name: 'Parametric EQ',
      version: '1.0.0',
      author: 'Mycelix',
      category: 'effect',
      description: '4-band parametric equalizer',
      parameters: bandParams,
      presets: [
        { id: 'flat', name: 'Flat', parameters: {} },
        { id: 'vocal-presence', name: 'Vocal Presence', parameters: { band2_freq: 3000, band2_gain: 3, band3_freq: 200, band3_gain: -2 } },
      ],
      inputs: 2,
      outputs: 2,
    };
  }

  constructor() {
    super(ParametricEQ.createMetadata());
  }

  protected async setup(): Promise<void> {
    // Create 4 filter bands
    for (let i = 0; i < 4; i++) {
      const filter = this.context!.createBiquadFilter();
      filter.type = 'peaking';
      this.bands.push(filter);
    }

    // Chain filters
    this.inputNode!.connect(this.bands[0]);
    for (let i = 0; i < this.bands.length - 1; i++) {
      this.bands[i].connect(this.bands[i + 1]);
    }
    this.bands[this.bands.length - 1].connect(this.outputNode!);

    // Apply initial parameters
    for (const param of this.metadata.parameters) {
      this.applyParameter(param.id, this.getParameter(param.id), 0);
    }
  }

  protected applyParameter(id: string, value: number, rampTime: number): void {
    const match = id.match(/band(\d)_(\w+)/);
    if (!match) return;

    const bandIndex = parseInt(match[1]) - 1;
    const paramType = match[2];
    const band = this.bands[bandIndex];
    if (!band) return;

    const now = this.context!.currentTime;

    switch (paramType) {
      case 'freq':
        if (rampTime > 0) {
          band.frequency.linearRampToValueAtTime(value, now + rampTime);
        } else {
          band.frequency.setValueAtTime(value, now);
        }
        break;
      case 'gain':
        if (rampTime > 0) {
          band.gain.linearRampToValueAtTime(value, now + rampTime);
        } else {
          band.gain.setValueAtTime(value, now);
        }
        break;
      case 'q':
        if (rampTime > 0) {
          band.Q.linearRampToValueAtTime(value, now + rampTime);
        } else {
          band.Q.setValueAtTime(value, now);
        }
        break;
      case 'type':
        const types: BiquadFilterType[] = ['peaking', 'lowshelf', 'highshelf', 'lowpass', 'highpass'];
        band.type = types[Math.floor(value)] || 'peaking';
        break;
    }
  }
}

// Compressor
export class CompressorPlugin extends AudioPlugin {
  private compressor: DynamicsCompressorNode | null = null;
  private makeupGain: GainNode | null = null;

  static createMetadata(): PluginMetadata {
    return {
      id: 'compressor',
      name: 'Compressor',
      version: '1.0.0',
      author: 'Mycelix',
      category: 'effect',
      description: 'Dynamic range compressor',
      parameters: [
        { id: 'threshold', name: 'Threshold', type: 'float', defaultValue: -24, minValue: -60, maxValue: 0, unit: 'dB' },
        { id: 'ratio', name: 'Ratio', type: 'float', defaultValue: 4, minValue: 1, maxValue: 20, unit: ':1' },
        { id: 'attack', name: 'Attack', type: 'float', defaultValue: 0.003, minValue: 0, maxValue: 1, unit: 's' },
        { id: 'release', name: 'Release', type: 'float', defaultValue: 0.25, minValue: 0.01, maxValue: 1, unit: 's' },
        { id: 'knee', name: 'Knee', type: 'float', defaultValue: 6, minValue: 0, maxValue: 40, unit: 'dB' },
        { id: 'makeup', name: 'Makeup Gain', type: 'float', defaultValue: 0, minValue: 0, maxValue: 24, unit: 'dB' },
      ],
      presets: [
        { id: 'gentle', name: 'Gentle', parameters: { threshold: -20, ratio: 2, attack: 0.01, release: 0.3 } },
        { id: 'punchy', name: 'Punchy', parameters: { threshold: -18, ratio: 4, attack: 0.001, release: 0.1 } },
        { id: 'limiting', name: 'Limiting', parameters: { threshold: -6, ratio: 20, attack: 0, release: 0.05 } },
      ],
      inputs: 2,
      outputs: 2,
    };
  }

  constructor() {
    super(CompressorPlugin.createMetadata());
  }

  protected async setup(): Promise<void> {
    this.compressor = this.context!.createDynamicsCompressor();
    this.makeupGain = this.context!.createGain();

    this.inputNode!.connect(this.compressor);
    this.compressor.connect(this.makeupGain);
    this.makeupGain.connect(this.outputNode!);
  }

  protected applyParameter(id: string, value: number, rampTime: number): void {
    const now = this.context!.currentTime;

    switch (id) {
      case 'threshold':
        this.compressor!.threshold.setValueAtTime(value, now);
        break;
      case 'ratio':
        this.compressor!.ratio.setValueAtTime(value, now);
        break;
      case 'attack':
        this.compressor!.attack.setValueAtTime(value, now);
        break;
      case 'release':
        this.compressor!.release.setValueAtTime(value, now);
        break;
      case 'knee':
        this.compressor!.knee.setValueAtTime(value, now);
        break;
      case 'makeup':
        const gain = Math.pow(10, value / 20);
        this.makeupGain!.gain.setValueAtTime(gain, now);
        break;
    }
  }

  getGainReduction(): number {
    return this.compressor?.reduction || 0;
  }
}

// Reverb
export class ReverbPlugin extends AudioPlugin {
  private convolver: ConvolverNode | null = null;
  private dryGain: GainNode | null = null;
  private wetGain: GainNode | null = null;
  private preDelay: DelayNode | null = null;

  static createMetadata(): PluginMetadata {
    return {
      id: 'reverb',
      name: 'Reverb',
      version: '1.0.0',
      author: 'Mycelix',
      category: 'effect',
      description: 'Convolution reverb',
      parameters: [
        { id: 'decay', name: 'Decay', type: 'float', defaultValue: 2, minValue: 0.1, maxValue: 10, unit: 's' },
        { id: 'predelay', name: 'Pre-delay', type: 'float', defaultValue: 0.01, minValue: 0, maxValue: 0.1, unit: 's' },
        { id: 'mix', name: 'Mix', type: 'float', defaultValue: 0.3, minValue: 0, maxValue: 1 },
        { id: 'damping', name: 'Damping', type: 'float', defaultValue: 0.5, minValue: 0, maxValue: 1 },
      ],
      presets: [
        { id: 'room', name: 'Room', parameters: { decay: 0.5, predelay: 0.005, mix: 0.2 } },
        { id: 'hall', name: 'Hall', parameters: { decay: 3, predelay: 0.02, mix: 0.35 } },
        { id: 'cathedral', name: 'Cathedral', parameters: { decay: 6, predelay: 0.05, mix: 0.5 } },
        { id: 'plate', name: 'Plate', parameters: { decay: 1.5, predelay: 0, mix: 0.4, damping: 0.3 } },
      ],
      inputs: 2,
      outputs: 2,
    };
  }

  constructor() {
    super(ReverbPlugin.createMetadata());
  }

  protected async setup(): Promise<void> {
    this.convolver = this.context!.createConvolver();
    this.dryGain = this.context!.createGain();
    this.wetGain = this.context!.createGain();
    this.preDelay = this.context!.createDelay(0.1);

    // Generate initial impulse response
    this.generateImpulseResponse(this.getParameter('decay'), this.getParameter('damping'));

    // Routing
    this.inputNode!.connect(this.dryGain!);
    this.inputNode!.connect(this.preDelay!);
    this.preDelay!.connect(this.convolver!);
    this.convolver!.connect(this.wetGain!);
    this.dryGain!.connect(this.outputNode!);
    this.wetGain!.connect(this.outputNode!);

    // Apply initial mix
    this.applyParameter('mix', this.getParameter('mix'), 0);
  }

  private generateImpulseResponse(decay: number, damping: number): void {
    const sampleRate = this.context!.sampleRate;
    const length = Math.ceil(sampleRate * decay);
    const impulse = this.context!.createBuffer(2, length, sampleRate);

    for (let channel = 0; channel < 2; channel++) {
      const data = impulse.getChannelData(channel);
      for (let i = 0; i < length; i++) {
        const t = i / sampleRate;
        const envelope = Math.exp(-3 * t / decay);
        const highFreqDamping = Math.exp(-damping * t * 10);
        data[i] = (Math.random() * 2 - 1) * envelope * highFreqDamping;
      }
    }

    this.convolver!.buffer = impulse;
  }

  protected applyParameter(id: string, value: number, rampTime: number): void {
    const now = this.context!.currentTime;

    switch (id) {
      case 'decay':
      case 'damping':
        this.generateImpulseResponse(
          this.getParameter('decay'),
          this.getParameter('damping')
        );
        break;
      case 'predelay':
        this.preDelay!.delayTime.setValueAtTime(value, now);
        break;
      case 'mix':
        this.dryGain!.gain.setValueAtTime(1 - value, now);
        this.wetGain!.gain.setValueAtTime(value, now);
        break;
    }
  }
}

// Delay
export class DelayPlugin extends AudioPlugin {
  private delayL: DelayNode | null = null;
  private delayR: DelayNode | null = null;
  private feedbackL: GainNode | null = null;
  private feedbackR: GainNode | null = null;
  private dryGain: GainNode | null = null;
  private wetGain: GainNode | null = null;
  private filter: BiquadFilterNode | null = null;

  static createMetadata(): PluginMetadata {
    return {
      id: 'delay',
      name: 'Stereo Delay',
      version: '1.0.0',
      author: 'Mycelix',
      category: 'effect',
      description: 'Stereo delay with ping-pong',
      parameters: [
        { id: 'timeL', name: 'Time L', type: 'float', defaultValue: 0.25, minValue: 0.001, maxValue: 2, unit: 's' },
        { id: 'timeR', name: 'Time R', type: 'float', defaultValue: 0.375, minValue: 0.001, maxValue: 2, unit: 's' },
        { id: 'feedback', name: 'Feedback', type: 'float', defaultValue: 0.3, minValue: 0, maxValue: 0.95 },
        { id: 'mix', name: 'Mix', type: 'float', defaultValue: 0.5, minValue: 0, maxValue: 1 },
        { id: 'highcut', name: 'High Cut', type: 'float', defaultValue: 8000, minValue: 200, maxValue: 20000, unit: 'Hz' },
        { id: 'pingpong', name: 'Ping Pong', type: 'bool', defaultValue: 0 },
      ],
      presets: [
        { id: 'slapback', name: 'Slapback', parameters: { timeL: 0.08, timeR: 0.08, feedback: 0.1, mix: 0.4 } },
        { id: 'quarter', name: '1/4 Note', parameters: { timeL: 0.5, timeR: 0.5, feedback: 0.4, mix: 0.35 } },
        { id: 'dotted', name: 'Dotted 8th', parameters: { timeL: 0.375, timeR: 0.25, feedback: 0.45, mix: 0.4 } },
      ],
      inputs: 2,
      outputs: 2,
    };
  }

  constructor() {
    super(DelayPlugin.createMetadata());
  }

  protected async setup(): Promise<void> {
    this.delayL = this.context!.createDelay(2);
    this.delayR = this.context!.createDelay(2);
    this.feedbackL = this.context!.createGain();
    this.feedbackR = this.context!.createGain();
    this.dryGain = this.context!.createGain();
    this.wetGain = this.context!.createGain();
    this.filter = this.context!.createBiquadFilter();
    this.filter.type = 'lowpass';

    const splitter = this.context!.createChannelSplitter(2);
    const merger = this.context!.createChannelMerger(2);

    // Dry path
    this.inputNode!.connect(this.dryGain!);
    this.dryGain!.connect(this.outputNode!);

    // Wet path
    this.inputNode!.connect(splitter);

    // Left channel
    splitter.connect(this.delayL!, 0);
    this.delayL!.connect(this.filter!);
    this.filter!.connect(this.feedbackL!);
    this.feedbackL!.connect(this.delayL!);
    this.delayL!.connect(merger, 0, 0);

    // Right channel
    splitter.connect(this.delayR!, 1);
    this.delayR!.connect(this.feedbackR!);
    this.feedbackR!.connect(this.delayR!);
    this.delayR!.connect(merger, 0, 1);

    merger.connect(this.wetGain!);
    this.wetGain!.connect(this.outputNode!);
  }

  protected applyParameter(id: string, value: number, rampTime: number): void {
    const now = this.context!.currentTime;

    switch (id) {
      case 'timeL':
        this.delayL!.delayTime.setValueAtTime(value, now);
        break;
      case 'timeR':
        this.delayR!.delayTime.setValueAtTime(value, now);
        break;
      case 'feedback':
        this.feedbackL!.gain.setValueAtTime(value, now);
        this.feedbackR!.gain.setValueAtTime(value, now);
        break;
      case 'mix':
        this.dryGain!.gain.setValueAtTime(1 - value, now);
        this.wetGain!.gain.setValueAtTime(value, now);
        break;
      case 'highcut':
        this.filter!.frequency.setValueAtTime(value, now);
        break;
    }
  }
}

// ==================== Plugin Host ====================

export class PluginHost {
  private context: AudioContext;
  private plugins: Map<string, AudioPlugin> = new Map();
  private pluginOrder: string[] = [];

  constructor(context: AudioContext) {
    this.context = context;
  }

  async addPlugin(plugin: AudioPlugin, id?: string): Promise<string> {
    const pluginId = id || `${plugin.metadata.id}-${Date.now()}`;
    await plugin.initialize(this.context);
    this.plugins.set(pluginId, plugin);
    this.pluginOrder.push(pluginId);
    this.rebuildChain();
    return pluginId;
  }

  removePlugin(pluginId: string): void {
    const plugin = this.plugins.get(pluginId);
    if (plugin) {
      plugin.dispose();
      this.plugins.delete(pluginId);
      this.pluginOrder = this.pluginOrder.filter(id => id !== pluginId);
      this.rebuildChain();
    }
  }

  movePlugin(pluginId: string, newIndex: number): void {
    const currentIndex = this.pluginOrder.indexOf(pluginId);
    if (currentIndex === -1) return;

    this.pluginOrder.splice(currentIndex, 1);
    this.pluginOrder.splice(newIndex, 0, pluginId);
    this.rebuildChain();
  }

  private rebuildChain(): void {
    // Disconnect all
    for (const plugin of this.plugins.values()) {
      plugin.input.disconnect();
      plugin.output.disconnect();
    }

    // Reconnect in order
    for (let i = 0; i < this.pluginOrder.length - 1; i++) {
      const current = this.plugins.get(this.pluginOrder[i])!;
      const next = this.plugins.get(this.pluginOrder[i + 1])!;
      current.output.connect(next.input);
    }
  }

  getPlugin(pluginId: string): AudioPlugin | undefined {
    return this.plugins.get(pluginId);
  }

  getPluginOrder(): string[] {
    return [...this.pluginOrder];
  }

  getFirstInput(): AudioNode | null {
    if (this.pluginOrder.length === 0) return null;
    return this.plugins.get(this.pluginOrder[0])!.input;
  }

  getLastOutput(): AudioNode | null {
    if (this.pluginOrder.length === 0) return null;
    return this.plugins.get(this.pluginOrder[this.pluginOrder.length - 1])!.output;
  }

  dispose(): void {
    for (const plugin of this.plugins.values()) {
      plugin.dispose();
    }
    this.plugins.clear();
    this.pluginOrder = [];
  }
}

// ==================== Plugin Registry ====================

export class PluginRegistry {
  private factories: Map<string, () => AudioPlugin> = new Map();

  register(id: string, factory: () => AudioPlugin): void {
    this.factories.set(id, factory);
  }

  create(id: string): AudioPlugin | null {
    const factory = this.factories.get(id);
    return factory ? factory() : null;
  }

  getAvailablePlugins(): PluginMetadata[] {
    const plugins: PluginMetadata[] = [];
    for (const factory of this.factories.values()) {
      const plugin = factory();
      plugins.push(plugin.metadata);
    }
    return plugins;
  }
}

// ==================== Default Registry ====================

export function createDefaultRegistry(): PluginRegistry {
  const registry = new PluginRegistry();

  registry.register('parametric-eq', () => new ParametricEQ());
  registry.register('compressor', () => new CompressorPlugin());
  registry.register('reverb', () => new ReverbPlugin());
  registry.register('delay', () => new DelayPlugin());

  return registry;
}

export default {
  AudioPlugin,
  ParametricEQ,
  CompressorPlugin,
  ReverbPlugin,
  DelayPlugin,
  PluginHost,
  PluginRegistry,
  createDefaultRegistry,
};
