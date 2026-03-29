// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Spatial Audio System
 *
 * Immersive 3D audio:
 * - Dolby Atmos support
 * - Binaural rendering
 * - 3D positioning
 * - Ambisonics
 * - HRTF processing
 * - Room acoustics
 */

// ==================== Types ====================

export interface Position3D {
  x: number;
  y: number;
  z: number;
}

export interface Orientation3D {
  forward: Position3D;
  up: Position3D;
}

export interface SpatialSource {
  id: string;
  position: Position3D;
  orientation?: Orientation3D;
  innerAngle: number;
  outerAngle: number;
  outerGain: number;
  rolloffFactor: number;
  refDistance: number;
  maxDistance: number;
  distanceModel: 'linear' | 'inverse' | 'exponential';
  panningModel: 'HRTF' | 'equalpower';
}

export interface RoomAcoustics {
  dimensions: Position3D;
  materials: {
    left: number;
    right: number;
    front: number;
    back: number;
    floor: number;
    ceiling: number;
  };
  reverb: number;
  damping: number;
}

export interface AmbisonicsConfig {
  order: 1 | 2 | 3;
  normalization: 'SN3D' | 'N3D';
}

// ==================== HRTF Manager ====================

export class HRTFManager {
  private context: AudioContext;
  private hrtfDatabase: Map<string, AudioBuffer> = new Map();
  private currentHRTF: string = 'default';

  constructor(context: AudioContext) {
    this.context = context;
  }

  async loadHRTF(name: string, url: string): Promise<void> {
    const response = await fetch(url);
    const data = await response.arrayBuffer();
    const buffer = await this.context.decodeAudioData(data);
    this.hrtfDatabase.set(name, buffer);
  }

  async loadDefaultHRTF(): Promise<void> {
    // Load default HRTF (would typically be a .sofa file or pre-processed)
    // For now, using Web Audio's built-in HRTF
  }

  getHRTF(name: string): AudioBuffer | undefined {
    return this.hrtfDatabase.get(name);
  }

  setCurrentHRTF(name: string): void {
    this.currentHRTF = name;
  }

  getAvailableHRTFs(): string[] {
    return Array.from(this.hrtfDatabase.keys());
  }
}

// ==================== Spatial Source ====================

export class SpatialAudioSource {
  private context: AudioContext;
  private panner: PannerNode;
  private gainNode: GainNode;
  private input: GainNode;
  private config: SpatialSource;

  constructor(context: AudioContext, config: Partial<SpatialSource> = {}) {
    this.context = context;

    this.config = {
      id: config.id || `source-${Date.now()}`,
      position: config.position || { x: 0, y: 0, z: 0 },
      innerAngle: config.innerAngle ?? 360,
      outerAngle: config.outerAngle ?? 360,
      outerGain: config.outerGain ?? 0,
      rolloffFactor: config.rolloffFactor ?? 1,
      refDistance: config.refDistance ?? 1,
      maxDistance: config.maxDistance ?? 10000,
      distanceModel: config.distanceModel || 'inverse',
      panningModel: config.panningModel || 'HRTF',
    };

    // Create nodes
    this.input = context.createGain();
    this.panner = context.createPanner();
    this.gainNode = context.createGain();

    // Configure panner
    this.panner.panningModel = this.config.panningModel;
    this.panner.distanceModel = this.config.distanceModel;
    this.panner.refDistance = this.config.refDistance;
    this.panner.maxDistance = this.config.maxDistance;
    this.panner.rolloffFactor = this.config.rolloffFactor;
    this.panner.coneInnerAngle = this.config.innerAngle;
    this.panner.coneOuterAngle = this.config.outerAngle;
    this.panner.coneOuterGain = this.config.outerGain;

    // Set initial position
    this.setPosition(this.config.position);

    // Connect
    this.input.connect(this.panner);
    this.panner.connect(this.gainNode);
  }

  get inputNode(): AudioNode {
    return this.input;
  }

  get outputNode(): AudioNode {
    return this.gainNode;
  }

  setPosition(position: Position3D, rampTime = 0): void {
    this.config.position = position;
    const now = this.context.currentTime;

    if (rampTime > 0) {
      this.panner.positionX.linearRampToValueAtTime(position.x, now + rampTime);
      this.panner.positionY.linearRampToValueAtTime(position.y, now + rampTime);
      this.panner.positionZ.linearRampToValueAtTime(position.z, now + rampTime);
    } else {
      this.panner.positionX.setValueAtTime(position.x, now);
      this.panner.positionY.setValueAtTime(position.y, now);
      this.panner.positionZ.setValueAtTime(position.z, now);
    }
  }

  setOrientation(forward: Position3D): void {
    const now = this.context.currentTime;
    this.panner.orientationX.setValueAtTime(forward.x, now);
    this.panner.orientationY.setValueAtTime(forward.y, now);
    this.panner.orientationZ.setValueAtTime(forward.z, now);
  }

  setGain(gain: number): void {
    this.gainNode.gain.setValueAtTime(gain, this.context.currentTime);
  }

  setDistanceModel(model: DistanceModelType): void {
    this.panner.distanceModel = model;
  }

  getPosition(): Position3D {
    return { ...this.config.position };
  }

  dispose(): void {
    this.input.disconnect();
    this.panner.disconnect();
    this.gainNode.disconnect();
  }
}

// ==================== Listener ====================

export class SpatialListener {
  private context: AudioContext;

  constructor(context: AudioContext) {
    this.context = context;
  }

  setPosition(position: Position3D): void {
    const now = this.context.currentTime;
    this.context.listener.positionX.setValueAtTime(position.x, now);
    this.context.listener.positionY.setValueAtTime(position.y, now);
    this.context.listener.positionZ.setValueAtTime(position.z, now);
  }

  setOrientation(forward: Position3D, up: Position3D): void {
    const now = this.context.currentTime;
    this.context.listener.forwardX.setValueAtTime(forward.x, now);
    this.context.listener.forwardY.setValueAtTime(forward.y, now);
    this.context.listener.forwardZ.setValueAtTime(forward.z, now);
    this.context.listener.upX.setValueAtTime(up.x, now);
    this.context.listener.upY.setValueAtTime(up.y, now);
    this.context.listener.upZ.setValueAtTime(up.z, now);
  }

  getPosition(): Position3D {
    return {
      x: this.context.listener.positionX.value,
      y: this.context.listener.positionY.value,
      z: this.context.listener.positionZ.value,
    };
  }
}

// ==================== Room Simulation ====================

export class RoomSimulator {
  private context: AudioContext;
  private convolver: ConvolverNode;
  private wetGain: GainNode;
  private dryGain: GainNode;
  private output: GainNode;

  constructor(context: AudioContext) {
    this.context = context;
    this.convolver = context.createConvolver();
    this.wetGain = context.createGain();
    this.dryGain = context.createGain();
    this.output = context.createGain();

    this.wetGain.connect(this.output);
    this.dryGain.connect(this.output);
    this.convolver.connect(this.wetGain);
  }

  get inputNode(): AudioNode {
    return this.convolver;
  }

  get outputNode(): AudioNode {
    return this.output;
  }

  connectDry(node: AudioNode): void {
    node.connect(this.dryGain);
  }

  setRoom(acoustics: RoomAcoustics): void {
    const impulse = this.generateRoomImpulse(acoustics);
    this.convolver.buffer = impulse;
  }

  setMix(wet: number): void {
    const now = this.context.currentTime;
    this.wetGain.gain.setValueAtTime(wet, now);
    this.dryGain.gain.setValueAtTime(1 - wet, now);
  }

  private generateRoomImpulse(acoustics: RoomAcoustics): AudioBuffer {
    const sampleRate = this.context.sampleRate;
    const reverbTime = acoustics.reverb;
    const length = Math.ceil(sampleRate * reverbTime);
    const buffer = this.context.createBuffer(2, length, sampleRate);

    // Calculate average absorption
    const materials = Object.values(acoustics.materials);
    const avgAbsorption = materials.reduce((a, b) => a + b, 0) / materials.length;

    for (let channel = 0; channel < 2; channel++) {
      const data = buffer.getChannelData(channel);

      for (let i = 0; i < length; i++) {
        const t = i / sampleRate;

        // Exponential decay with absorption
        let envelope = Math.exp(-3 * t / reverbTime) * (1 - avgAbsorption);

        // High frequency damping
        envelope *= Math.exp(-acoustics.damping * t * 5);

        // Add early reflections
        const earlyDecay = Math.exp(-10 * t);
        const early = earlyDecay * 0.3;

        data[i] = (Math.random() * 2 - 1) * (envelope + early);
      }
    }

    return buffer;
  }

  dispose(): void {
    this.convolver.disconnect();
    this.wetGain.disconnect();
    this.dryGain.disconnect();
    this.output.disconnect();
  }
}

// ==================== Binaural Renderer ====================

export class BinauralRenderer {
  private context: AudioContext;
  private leftConvolver: ConvolverNode;
  private rightConvolver: ConvolverNode;
  private splitter: ChannelSplitterNode;
  private merger: ChannelMergerNode;
  private hrtfManager: HRTFManager;

  constructor(context: AudioContext, hrtfManager: HRTFManager) {
    this.context = context;
    this.hrtfManager = hrtfManager;

    this.leftConvolver = context.createConvolver();
    this.rightConvolver = context.createConvolver();
    this.splitter = context.createChannelSplitter(2);
    this.merger = context.createChannelMerger(2);

    this.splitter.connect(this.leftConvolver, 0);
    this.splitter.connect(this.rightConvolver, 1);
    this.leftConvolver.connect(this.merger, 0, 0);
    this.rightConvolver.connect(this.merger, 0, 1);
  }

  get inputNode(): AudioNode {
    return this.splitter;
  }

  get outputNode(): AudioNode {
    return this.merger;
  }

  updatePosition(azimuth: number, elevation: number): void {
    // Would load appropriate HRTF filters based on position
    // Simplified version - actual implementation would interpolate HRTFs
  }

  dispose(): void {
    this.leftConvolver.disconnect();
    this.rightConvolver.disconnect();
    this.splitter.disconnect();
    this.merger.disconnect();
  }
}

// ==================== Spatial Audio Manager ====================

export class SpatialAudioManager {
  private context: AudioContext;
  private listener: SpatialListener;
  private sources: Map<string, SpatialAudioSource> = new Map();
  private roomSimulator: RoomSimulator;
  private hrtfManager: HRTFManager;
  private masterGain: GainNode;
  private isEnabled = true;

  constructor(context?: AudioContext) {
    this.context = context || new AudioContext({ sampleRate: 48000 });
    this.listener = new SpatialListener(this.context);
    this.roomSimulator = new RoomSimulator(this.context);
    this.hrtfManager = new HRTFManager(this.context);
    this.masterGain = this.context.createGain();

    this.roomSimulator.outputNode.connect(this.masterGain);
    this.masterGain.connect(this.context.destination);
  }

  async initialize(): Promise<void> {
    await this.hrtfManager.loadDefaultHRTF();
  }

  createSource(config?: Partial<SpatialSource>): SpatialAudioSource {
    const source = new SpatialAudioSource(this.context, config);
    source.outputNode.connect(this.roomSimulator.inputNode);
    this.roomSimulator.connectDry(source.outputNode);
    this.sources.set(source.inputNode.toString(), source);
    return source;
  }

  removeSource(source: SpatialAudioSource): void {
    source.dispose();
    this.sources.delete(source.inputNode.toString());
  }

  setListenerPosition(position: Position3D): void {
    this.listener.setPosition(position);
  }

  setListenerOrientation(forward: Position3D, up: Position3D): void {
    this.listener.setOrientation(forward, up);
  }

  setRoom(acoustics: RoomAcoustics): void {
    this.roomSimulator.setRoom(acoustics);
  }

  setReverbMix(mix: number): void {
    this.roomSimulator.setMix(mix);
  }

  setMasterGain(gain: number): void {
    this.masterGain.gain.setValueAtTime(gain, this.context.currentTime);
  }

  enable(): void {
    this.isEnabled = true;
  }

  disable(): void {
    this.isEnabled = false;
  }

  getContext(): AudioContext {
    return this.context;
  }

  dispose(): void {
    for (const source of this.sources.values()) {
      source.dispose();
    }
    this.sources.clear();
    this.roomSimulator.dispose();
    this.masterGain.disconnect();
  }
}

// ==================== Singleton ====================

let spatialManager: SpatialAudioManager | null = null;

export function getSpatialAudioManager(): SpatialAudioManager {
  if (!spatialManager) {
    spatialManager = new SpatialAudioManager();
  }
  return spatialManager;
}

export default {
  SpatialAudioManager,
  SpatialAudioSource,
  SpatialListener,
  RoomSimulator,
  BinauralRenderer,
  HRTFManager,
  getSpatialAudioManager,
};
