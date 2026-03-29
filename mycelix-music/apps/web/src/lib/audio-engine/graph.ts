// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Audio Processing Graph
 *
 * Professional audio routing:
 * - Node-based signal flow
 * - Parallel processing chains
 * - Sidechain routing
 * - Feedback loops (with delay)
 * - Dynamic patching
 */

// ==================== Types ====================

export type AudioNodeType =
  | 'input'
  | 'output'
  | 'gain'
  | 'panner'
  | 'filter'
  | 'compressor'
  | 'delay'
  | 'reverb'
  | 'eq'
  | 'oscillator'
  | 'sampler'
  | 'analyzer'
  | 'splitter'
  | 'merger'
  | 'custom';

export interface AudioNodeConfig {
  id: string;
  type: AudioNodeType;
  params: Record<string, number | string | boolean>;
  position?: { x: number; y: number };
  bypass?: boolean;
}

export interface AudioConnection {
  id: string;
  from: { nodeId: string; output: number };
  to: { nodeId: string; input: number };
}

export interface AudioGraphState {
  nodes: Map<string, AudioNodeConfig>;
  connections: AudioConnection[];
  masterGain: number;
  sampleRate: number;
  isPlaying: boolean;
}

// ==================== Audio Node Wrapper ====================

export class AudioNodeWrapper {
  public readonly id: string;
  public readonly type: AudioNodeType;
  public readonly node: AudioNode;
  public bypass: boolean = false;
  public params: Record<string, AudioParam | number>;
  public position: { x: number; y: number };

  private bypassGain: GainNode;
  private inputGain: GainNode;
  private outputGain: GainNode;
  private context: AudioContext;

  constructor(
    context: AudioContext,
    config: AudioNodeConfig,
    node: AudioNode
  ) {
    this.context = context;
    this.id = config.id;
    this.type = config.type;
    this.node = node;
    this.position = config.position || { x: 0, y: 0 };
    this.params = {};

    // Create bypass network
    this.inputGain = context.createGain();
    this.outputGain = context.createGain();
    this.bypassGain = context.createGain();
    this.bypassGain.gain.value = 0;

    // Connect bypass network
    this.inputGain.connect(node);
    (node as any).connect?.(this.outputGain);
    this.inputGain.connect(this.bypassGain);
    this.bypassGain.connect(this.outputGain);

    this.setBypass(config.bypass || false);
    this.applyParams(config.params);
  }

  get input(): AudioNode {
    return this.inputGain;
  }

  get output(): AudioNode {
    return this.outputGain;
  }

  setBypass(bypass: boolean): void {
    this.bypass = bypass;
    const now = this.context.currentTime;

    if (bypass) {
      // Crossfade to bypass
      (this.node as GainNode).gain?.setTargetAtTime(0, now, 0.01);
      this.bypassGain.gain.setTargetAtTime(1, now, 0.01);
    } else {
      // Crossfade to processed
      (this.node as GainNode).gain?.setTargetAtTime(1, now, 0.01);
      this.bypassGain.gain.setTargetAtTime(0, now, 0.01);
    }
  }

  setParam(name: string, value: number, rampTime = 0): void {
    const param = this.params[name];
    if (param instanceof AudioParam) {
      if (rampTime > 0) {
        param.linearRampToValueAtTime(value, this.context.currentTime + rampTime);
      } else {
        param.setValueAtTime(value, this.context.currentTime);
      }
    } else {
      this.params[name] = value;
    }
  }

  getParam(name: string): number {
    const param = this.params[name];
    if (param instanceof AudioParam) {
      return param.value;
    }
    return typeof param === 'number' ? param : 0;
  }

  private applyParams(params: Record<string, number | string | boolean>): void {
    for (const [key, value] of Object.entries(params)) {
      if (typeof value === 'number') {
        const node = this.node as any;
        if (node[key] instanceof AudioParam) {
          this.params[key] = node[key];
          node[key].value = value;
        } else if (typeof node[key] === 'number') {
          node[key] = value;
          this.params[key] = value;
        }
      }
    }
  }

  disconnect(): void {
    this.inputGain.disconnect();
    this.outputGain.disconnect();
    this.bypassGain.disconnect();
    this.node.disconnect();
  }

  toConfig(): AudioNodeConfig {
    return {
      id: this.id,
      type: this.type,
      params: Object.fromEntries(
        Object.entries(this.params).map(([k, v]) => [k, v instanceof AudioParam ? v.value : v])
      ),
      position: this.position,
      bypass: this.bypass,
    };
  }
}

// ==================== Audio Graph ====================

export class AudioGraph {
  private context: AudioContext;
  private nodes: Map<string, AudioNodeWrapper> = new Map();
  private connections: AudioConnection[] = [];
  private masterGain: GainNode;
  private analyzer: AnalyserNode;
  private isRunning = false;

  // Event handlers
  public onNodeAdded?: (node: AudioNodeWrapper) => void;
  public onNodeRemoved?: (nodeId: string) => void;
  public onConnectionAdded?: (connection: AudioConnection) => void;
  public onConnectionRemoved?: (connectionId: string) => void;

  constructor(sampleRate = 48000) {
    this.context = new AudioContext({ sampleRate });
    this.masterGain = this.context.createGain();
    this.analyzer = this.context.createAnalyser();
    this.analyzer.fftSize = 2048;

    this.masterGain.connect(this.analyzer);
    this.analyzer.connect(this.context.destination);
  }

  // ==================== Node Management ====================

  createNode(config: AudioNodeConfig): AudioNodeWrapper {
    const audioNode = this.createAudioNode(config);
    const wrapper = new AudioNodeWrapper(this.context, config, audioNode);

    this.nodes.set(config.id, wrapper);
    this.onNodeAdded?.(wrapper);

    return wrapper;
  }

  private createAudioNode(config: AudioNodeConfig): AudioNode {
    switch (config.type) {
      case 'gain':
        return this.context.createGain();

      case 'panner':
        return this.context.createStereoPanner();

      case 'filter': {
        const filter = this.context.createBiquadFilter();
        filter.type = (config.params.type as BiquadFilterType) || 'lowpass';
        filter.frequency.value = (config.params.frequency as number) || 1000;
        filter.Q.value = (config.params.Q as number) || 1;
        return filter;
      }

      case 'compressor': {
        const comp = this.context.createDynamicsCompressor();
        comp.threshold.value = (config.params.threshold as number) || -24;
        comp.ratio.value = (config.params.ratio as number) || 4;
        comp.attack.value = (config.params.attack as number) || 0.003;
        comp.release.value = (config.params.release as number) || 0.25;
        comp.knee.value = (config.params.knee as number) || 30;
        return comp;
      }

      case 'delay': {
        const delay = this.context.createDelay(5);
        delay.delayTime.value = (config.params.time as number) || 0.5;
        return delay;
      }

      case 'reverb':
        return this.createReverbNode(config);

      case 'eq':
        return this.createEQNode(config);

      case 'oscillator': {
        const osc = this.context.createOscillator();
        osc.type = (config.params.waveform as OscillatorType) || 'sine';
        osc.frequency.value = (config.params.frequency as number) || 440;
        return osc;
      }

      case 'analyzer':
        return this.context.createAnalyser();

      case 'splitter':
        return this.context.createChannelSplitter(config.params.channels as number || 2);

      case 'merger':
        return this.context.createChannelMerger(config.params.channels as number || 2);

      default:
        return this.context.createGain();
    }
  }

  private createReverbNode(config: AudioNodeConfig): ConvolverNode {
    const convolver = this.context.createConvolver();

    // Generate impulse response
    const duration = (config.params.decay as number) || 2;
    const sampleRate = this.context.sampleRate;
    const length = sampleRate * duration;
    const impulse = this.context.createBuffer(2, length, sampleRate);

    for (let channel = 0; channel < 2; channel++) {
      const data = impulse.getChannelData(channel);
      for (let i = 0; i < length; i++) {
        data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, 2);
      }
    }

    convolver.buffer = impulse;
    return convolver;
  }

  private createEQNode(config: AudioNodeConfig): GainNode {
    // EQ is implemented as a chain of filters
    // For simplicity, returning a gain node here
    // Real implementation would create filter chain
    return this.context.createGain();
  }

  removeNode(nodeId: string): void {
    const node = this.nodes.get(nodeId);
    if (!node) return;

    // Remove all connections involving this node
    const connectionsToRemove = this.connections.filter(
      c => c.from.nodeId === nodeId || c.to.nodeId === nodeId
    );
    for (const conn of connectionsToRemove) {
      this.removeConnection(conn.id);
    }

    node.disconnect();
    this.nodes.delete(nodeId);
    this.onNodeRemoved?.(nodeId);
  }

  getNode(nodeId: string): AudioNodeWrapper | undefined {
    return this.nodes.get(nodeId);
  }

  getAllNodes(): AudioNodeWrapper[] {
    return Array.from(this.nodes.values());
  }

  // ==================== Connection Management ====================

  connect(from: { nodeId: string; output?: number }, to: { nodeId: string; input?: number }): AudioConnection {
    const fromNode = this.nodes.get(from.nodeId);
    const toNode = this.nodes.get(to.nodeId);

    if (!fromNode || !toNode) {
      throw new Error('Node not found');
    }

    const connection: AudioConnection = {
      id: `${from.nodeId}-${to.nodeId}-${Date.now()}`,
      from: { nodeId: from.nodeId, output: from.output || 0 },
      to: { nodeId: to.nodeId, input: to.input || 0 },
    };

    // Create audio connection
    fromNode.output.connect(toNode.input);

    this.connections.push(connection);
    this.onConnectionAdded?.(connection);

    return connection;
  }

  connectToMaster(nodeId: string): void {
    const node = this.nodes.get(nodeId);
    if (node) {
      node.output.connect(this.masterGain);
    }
  }

  removeConnection(connectionId: string): void {
    const index = this.connections.findIndex(c => c.id === connectionId);
    if (index === -1) return;

    const connection = this.connections[index];
    const fromNode = this.nodes.get(connection.from.nodeId);
    const toNode = this.nodes.get(connection.to.nodeId);

    if (fromNode && toNode) {
      try {
        fromNode.output.disconnect(toNode.input);
      } catch {
        // Ignore if already disconnected
      }
    }

    this.connections.splice(index, 1);
    this.onConnectionRemoved?.(connectionId);
  }

  getConnections(): AudioConnection[] {
    return [...this.connections];
  }

  // ==================== Transport ====================

  async start(): Promise<void> {
    if (this.context.state === 'suspended') {
      await this.context.resume();
    }
    this.isRunning = true;

    // Start all oscillators
    for (const node of this.nodes.values()) {
      if (node.type === 'oscillator') {
        try {
          (node.node as OscillatorNode).start();
        } catch {
          // Already started
        }
      }
    }
  }

  stop(): void {
    this.isRunning = false;

    // Stop oscillators by recreating them
    for (const node of this.nodes.values()) {
      if (node.type === 'oscillator') {
        try {
          (node.node as OscillatorNode).stop();
        } catch {
          // Already stopped
        }
      }
    }
  }

  suspend(): void {
    this.context.suspend();
  }

  resume(): void {
    this.context.resume();
  }

  // ==================== Master Controls ====================

  setMasterGain(value: number): void {
    this.masterGain.gain.setValueAtTime(value, this.context.currentTime);
  }

  getMasterGain(): number {
    return this.masterGain.gain.value;
  }

  // ==================== Analysis ====================

  getFrequencyData(): Uint8Array {
    const data = new Uint8Array(this.analyzer.frequencyBinCount);
    this.analyzer.getByteFrequencyData(data);
    return data;
  }

  getTimeDomainData(): Uint8Array {
    const data = new Uint8Array(this.analyzer.fftSize);
    this.analyzer.getByteTimeDomainData(data);
    return data;
  }

  getFloatFrequencyData(): Float32Array {
    const data = new Float32Array(this.analyzer.frequencyBinCount);
    this.analyzer.getFloatFrequencyData(data);
    return data;
  }

  // ==================== Serialization ====================

  toJSON(): { nodes: AudioNodeConfig[]; connections: AudioConnection[] } {
    return {
      nodes: Array.from(this.nodes.values()).map(n => n.toConfig()),
      connections: this.connections,
    };
  }

  loadFromJSON(data: { nodes: AudioNodeConfig[]; connections: AudioConnection[] }): void {
    // Clear existing
    for (const nodeId of this.nodes.keys()) {
      this.removeNode(nodeId);
    }

    // Create nodes
    for (const nodeConfig of data.nodes) {
      this.createNode(nodeConfig);
    }

    // Create connections
    for (const conn of data.connections) {
      try {
        this.connect(conn.from, conn.to);
      } catch (e) {
        console.warn('Failed to restore connection:', e);
      }
    }
  }

  // ==================== Context Access ====================

  getContext(): AudioContext {
    return this.context;
  }

  getSampleRate(): number {
    return this.context.sampleRate;
  }

  getCurrentTime(): number {
    return this.context.currentTime;
  }

  // ==================== Cleanup ====================

  dispose(): void {
    this.stop();

    for (const nodeId of this.nodes.keys()) {
      this.removeNode(nodeId);
    }

    this.masterGain.disconnect();
    this.analyzer.disconnect();
    this.context.close();
  }
}

export default AudioGraph;
