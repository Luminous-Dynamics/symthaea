// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Audio Engine
 *
 * Web Audio API-based audio engine for the in-browser music studio.
 * Handles audio routing, effects processing, recording, and playback.
 */

// ============================================================================
// Types
// ============================================================================

export interface AudioTrack {
  id: string;
  name: string;
  type: 'audio' | 'midi' | 'sample';
  clips: AudioClip[];
  volume: number;          // 0-1
  pan: number;             // -1 to 1
  muted: boolean;
  solo: boolean;
  armed: boolean;          // Ready to record
  effects: Effect[];
  color: string;
}

export interface AudioClip {
  id: string;
  trackId: string;
  startTime: number;       // In beats
  duration: number;        // In beats
  offset: number;          // Start offset within source
  sourceId: string;        // Reference to audio buffer
  gain: number;
  fadeIn: number;
  fadeOut: number;
}

export interface Effect {
  id: string;
  type: EffectType;
  enabled: boolean;
  params: Record<string, number>;
}

export type EffectType =
  | 'eq'
  | 'compressor'
  | 'reverb'
  | 'delay'
  | 'distortion'
  | 'chorus'
  | 'filter'
  | 'gain';

export interface TransportState {
  isPlaying: boolean;
  isRecording: boolean;
  position: number;        // In beats
  bpm: number;
  timeSignature: [number, number];
  loopStart: number;
  loopEnd: number;
  loopEnabled: boolean;
}

export interface MeterLevels {
  left: number;
  right: number;
  peak: number;
}

// ============================================================================
// Audio Engine
// ============================================================================

export class AudioEngine {
  private audioContext: AudioContext;
  private masterGain: GainNode;
  private masterAnalyser: AnalyserNode;
  private masterCompressor: DynamicsCompressorNode;

  private tracks: Map<string, TrackNode> = new Map();
  private audioBuffers: Map<string, AudioBuffer> = new Map();
  private activeNodes: Map<string, AudioBufferSourceNode> = new Map();

  private transport: TransportState = {
    isPlaying: false,
    isRecording: false,
    position: 0,
    bpm: 120,
    timeSignature: [4, 4],
    loopStart: 0,
    loopEnd: 16,
    loopEnabled: false,
  };

  private playStartTime: number = 0;
  private playStartPosition: number = 0;
  private recordedChunks: Float32Array[] = [];
  private mediaRecorder: MediaRecorder | null = null;

  private onTransportChange?: (state: TransportState) => void;
  private onMeterUpdate?: (trackId: string, levels: MeterLevels) => void;

  constructor() {
    this.audioContext = new AudioContext({ sampleRate: 48000 });

    // Master chain
    this.masterCompressor = this.audioContext.createDynamicsCompressor();
    this.masterGain = this.audioContext.createGain();
    this.masterAnalyser = this.audioContext.createAnalyser();

    this.masterCompressor.connect(this.masterGain);
    this.masterGain.connect(this.masterAnalyser);
    this.masterAnalyser.connect(this.audioContext.destination);

    // Start meter updates
    this.startMeterUpdates();
  }

  // ============================================================================
  // Transport Controls
  // ============================================================================

  play(): void {
    if (this.transport.isPlaying) return;

    this.audioContext.resume();
    this.transport.isPlaying = true;
    this.playStartTime = this.audioContext.currentTime;
    this.playStartPosition = this.transport.position;

    // Schedule all clips
    this.scheduleAllClips();

    this.notifyTransportChange();
  }

  pause(): void {
    if (!this.transport.isPlaying) return;

    this.transport.isPlaying = false;
    this.transport.position = this.getCurrentPosition();

    // Stop all active sources
    this.stopAllSources();

    this.notifyTransportChange();
  }

  stop(): void {
    this.transport.isPlaying = false;
    this.transport.position = 0;

    this.stopAllSources();
    this.notifyTransportChange();
  }

  setPosition(beats: number): void {
    const wasPlaying = this.transport.isPlaying;

    if (wasPlaying) {
      this.stopAllSources();
    }

    this.transport.position = Math.max(0, beats);
    this.playStartPosition = this.transport.position;
    this.playStartTime = this.audioContext.currentTime;

    if (wasPlaying) {
      this.scheduleAllClips();
    }

    this.notifyTransportChange();
  }

  setBpm(bpm: number): void {
    const currentPosition = this.getCurrentPosition();
    this.transport.bpm = Math.max(20, Math.min(300, bpm));

    if (this.transport.isPlaying) {
      this.playStartPosition = currentPosition;
      this.playStartTime = this.audioContext.currentTime;
      this.stopAllSources();
      this.scheduleAllClips();
    }

    this.notifyTransportChange();
  }

  setLoop(start: number, end: number, enabled: boolean): void {
    this.transport.loopStart = start;
    this.transport.loopEnd = end;
    this.transport.loopEnabled = enabled;
    this.notifyTransportChange();
  }

  getCurrentPosition(): number {
    if (!this.transport.isPlaying) {
      return this.transport.position;
    }

    const elapsedSeconds = this.audioContext.currentTime - this.playStartTime;
    const elapsedBeats = (elapsedSeconds / 60) * this.transport.bpm;
    let position = this.playStartPosition + elapsedBeats;

    // Handle looping
    if (this.transport.loopEnabled && position >= this.transport.loopEnd) {
      const loopLength = this.transport.loopEnd - this.transport.loopStart;
      position = this.transport.loopStart + ((position - this.transport.loopStart) % loopLength);
    }

    return position;
  }

  // ============================================================================
  // Track Management
  // ============================================================================

  createTrack(track: AudioTrack): void {
    const trackNode = new TrackNode(this.audioContext, track);
    trackNode.connect(this.masterCompressor);
    this.tracks.set(track.id, trackNode);
  }

  updateTrack(trackId: string, updates: Partial<AudioTrack>): void {
    const trackNode = this.tracks.get(trackId);
    if (!trackNode) return;

    if (updates.volume !== undefined) {
      trackNode.setVolume(updates.volume);
    }
    if (updates.pan !== undefined) {
      trackNode.setPan(updates.pan);
    }
    if (updates.muted !== undefined) {
      trackNode.setMuted(updates.muted);
    }
    if (updates.effects !== undefined) {
      trackNode.updateEffects(updates.effects);
    }
  }

  deleteTrack(trackId: string): void {
    const trackNode = this.tracks.get(trackId);
    if (trackNode) {
      trackNode.disconnect();
      this.tracks.delete(trackId);
    }
  }

  // ============================================================================
  // Audio Loading
  // ============================================================================

  async loadAudioFile(file: File): Promise<string> {
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

    const id = `audio_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    this.audioBuffers.set(id, audioBuffer);

    return id;
  }

  async loadAudioFromUrl(url: string): Promise<string> {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

    const id = `audio_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    this.audioBuffers.set(id, audioBuffer);

    return id;
  }

  getAudioBuffer(id: string): AudioBuffer | undefined {
    return this.audioBuffers.get(id);
  }

  // ============================================================================
  // Clip Scheduling
  // ============================================================================

  private scheduleAllClips(): void {
    const currentPosition = this.getCurrentPosition();

    for (const [trackId, trackNode] of this.tracks) {
      const track = trackNode.getTrack();
      if (track.muted) continue;

      for (const clip of track.clips) {
        this.scheduleClip(trackId, clip, currentPosition);
      }
    }
  }

  private scheduleClip(trackId: string, clip: AudioClip, currentPosition: number): void {
    const buffer = this.audioBuffers.get(clip.sourceId);
    if (!buffer) return;

    const trackNode = this.tracks.get(trackId);
    if (!trackNode) return;

    // Calculate timing
    const clipEndTime = clip.startTime + clip.duration;

    // Skip if clip has already passed
    if (clipEndTime <= currentPosition) return;

    // Calculate when to start playback
    const beatsUntilStart = clip.startTime - currentPosition;
    const secondsUntilStart = (beatsUntilStart / this.transport.bpm) * 60;

    // Calculate playback offset within the clip
    let sourceOffset = clip.offset;
    let duration = clip.duration;

    if (beatsUntilStart < 0) {
      // Clip already started, adjust offset
      const beatsIntoClip = -beatsUntilStart;
      sourceOffset += (beatsIntoClip / this.transport.bpm) * 60;
      duration -= beatsIntoClip;
    }

    // Create and schedule source
    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;

    // Create gain for clip-level volume and fades
    const clipGain = this.audioContext.createGain();
    clipGain.gain.value = clip.gain;

    // Apply fades
    if (clip.fadeIn > 0) {
      const fadeInSeconds = (clip.fadeIn / this.transport.bpm) * 60;
      clipGain.gain.setValueAtTime(0, this.audioContext.currentTime + Math.max(0, secondsUntilStart));
      clipGain.gain.linearRampToValueAtTime(
        clip.gain,
        this.audioContext.currentTime + Math.max(0, secondsUntilStart) + fadeInSeconds
      );
    }

    if (clip.fadeOut > 0) {
      const durationSeconds = (duration / this.transport.bpm) * 60;
      const fadeOutSeconds = (clip.fadeOut / this.transport.bpm) * 60;
      clipGain.gain.setValueAtTime(
        clip.gain,
        this.audioContext.currentTime + Math.max(0, secondsUntilStart) + durationSeconds - fadeOutSeconds
      );
      clipGain.gain.linearRampToValueAtTime(
        0,
        this.audioContext.currentTime + Math.max(0, secondsUntilStart) + durationSeconds
      );
    }

    source.connect(clipGain);
    clipGain.connect(trackNode.getInputNode());

    const durationSeconds = (duration / this.transport.bpm) * 60;

    if (secondsUntilStart > 0) {
      source.start(this.audioContext.currentTime + secondsUntilStart, sourceOffset, durationSeconds);
    } else {
      source.start(0, sourceOffset, durationSeconds);
    }

    // Track active nodes for stopping
    const nodeId = `${trackId}_${clip.id}`;
    this.activeNodes.set(nodeId, source);

    source.onended = () => {
      this.activeNodes.delete(nodeId);
    };
  }

  private stopAllSources(): void {
    for (const source of this.activeNodes.values()) {
      try {
        source.stop();
      } catch (e) {
        // Already stopped
      }
    }
    this.activeNodes.clear();
  }

  // ============================================================================
  // Recording
  // ============================================================================

  async startRecording(trackId: string): Promise<void> {
    const trackNode = this.tracks.get(trackId);
    if (!trackNode) return;

    // Get microphone access
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Create media stream source
    const mediaSource = this.audioContext.createMediaStreamSource(stream);
    mediaSource.connect(trackNode.getInputNode());

    // Create recorder
    const dest = this.audioContext.createMediaStreamDestination();
    trackNode.getOutputNode().connect(dest);

    this.mediaRecorder = new MediaRecorder(dest.stream);
    this.recordedChunks = [];

    this.mediaRecorder.ondataavailable = async (e) => {
      if (e.data.size > 0) {
        const arrayBuffer = await e.data.arrayBuffer();
        const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
        const channelData = audioBuffer.getChannelData(0);
        this.recordedChunks.push(channelData);
      }
    };

    this.mediaRecorder.start(100); // Capture every 100ms
    this.transport.isRecording = true;
    this.notifyTransportChange();
  }

  stopRecording(): { audioId: string; startPosition: number; duration: number } | null {
    if (!this.mediaRecorder || !this.transport.isRecording) return null;

    this.mediaRecorder.stop();
    this.transport.isRecording = false;

    // Combine chunks into single buffer
    const totalLength = this.recordedChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const combined = new Float32Array(totalLength);
    let offset = 0;

    for (const chunk of this.recordedChunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    // Create audio buffer
    const buffer = this.audioContext.createBuffer(1, totalLength, this.audioContext.sampleRate);
    buffer.getChannelData(0).set(combined);

    const audioId = `recorded_${Date.now()}`;
    this.audioBuffers.set(audioId, buffer);

    const durationSeconds = totalLength / this.audioContext.sampleRate;
    const durationBeats = (durationSeconds / 60) * this.transport.bpm;

    this.notifyTransportChange();

    return {
      audioId,
      startPosition: this.playStartPosition,
      duration: durationBeats,
    };
  }

  // ============================================================================
  // Export
  // ============================================================================

  async exportMix(startBeat: number, endBeat: number): Promise<Blob> {
    const durationBeats = endBeat - startBeat;
    const durationSeconds = (durationBeats / this.transport.bpm) * 60;
    const sampleRate = this.audioContext.sampleRate;
    const length = Math.ceil(durationSeconds * sampleRate);

    // Create offline context
    const offlineContext = new OfflineAudioContext(2, length, sampleRate);

    // Recreate track nodes in offline context
    const offlineMaster = offlineContext.createGain();
    offlineMaster.connect(offlineContext.destination);

    for (const [trackId, trackNode] of this.tracks) {
      const track = trackNode.getTrack();
      if (track.muted) continue;

      // Create offline track gain
      const offlineGain = offlineContext.createGain();
      offlineGain.gain.value = track.volume;
      offlineGain.connect(offlineMaster);

      // Schedule clips
      for (const clip of track.clips) {
        if (clip.startTime >= endBeat || clip.startTime + clip.duration <= startBeat) {
          continue;
        }

        const buffer = this.audioBuffers.get(clip.sourceId);
        if (!buffer) continue;

        const source = offlineContext.createBufferSource();
        source.buffer = buffer;

        const clipGain = offlineContext.createGain();
        clipGain.gain.value = clip.gain;

        source.connect(clipGain);
        clipGain.connect(offlineGain);

        const clipStartSeconds = ((clip.startTime - startBeat) / this.transport.bpm) * 60;
        const clipDurationSeconds = (clip.duration / this.transport.bpm) * 60;

        source.start(Math.max(0, clipStartSeconds), clip.offset, clipDurationSeconds);
      }
    }

    // Render
    const renderedBuffer = await offlineContext.startRendering();

    // Convert to WAV
    return this.audioBufferToWav(renderedBuffer);
  }

  private audioBufferToWav(buffer: AudioBuffer): Blob {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    const dataLength = buffer.length * blockAlign;
    const bufferLength = 44 + dataLength;

    const arrayBuffer = new ArrayBuffer(bufferLength);
    const view = new DataView(arrayBuffer);

    // WAV header
    this.writeString(view, 0, 'RIFF');
    view.setUint32(4, bufferLength - 8, true);
    this.writeString(view, 8, 'WAVE');
    this.writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    this.writeString(view, 36, 'data');
    view.setUint32(40, dataLength, true);

    // Write audio data
    const channels: Float32Array[] = [];
    for (let i = 0; i < numChannels; i++) {
      channels.push(buffer.getChannelData(i));
    }

    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
      for (let ch = 0; ch < numChannels; ch++) {
        const sample = Math.max(-1, Math.min(1, channels[ch][i]));
        const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        view.setInt16(offset, intSample, true);
        offset += 2;
      }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
  }

  private writeString(view: DataView, offset: number, string: string): void {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // ============================================================================
  // Metering
  // ============================================================================

  private startMeterUpdates(): void {
    const updateMeters = () => {
      if (this.onMeterUpdate) {
        // Master meters
        const masterLevels = this.getAnalyserLevels(this.masterAnalyser);
        this.onMeterUpdate('master', masterLevels);

        // Track meters
        for (const [trackId, trackNode] of this.tracks) {
          const levels = trackNode.getMeterLevels();
          this.onMeterUpdate(trackId, levels);
        }
      }

      requestAnimationFrame(updateMeters);
    };

    updateMeters();
  }

  private getAnalyserLevels(analyser: AnalyserNode): MeterLevels {
    const dataArray = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(dataArray);

    let sum = 0;
    let peak = 0;

    for (const sample of dataArray) {
      const abs = Math.abs(sample);
      sum += abs * abs;
      if (abs > peak) peak = abs;
    }

    const rms = Math.sqrt(sum / dataArray.length);

    return {
      left: rms,
      right: rms,
      peak,
    };
  }

  // ============================================================================
  // Event Handlers
  // ============================================================================

  onTransport(callback: (state: TransportState) => void): void {
    this.onTransportChange = callback;
  }

  onMeters(callback: (trackId: string, levels: MeterLevels) => void): void {
    this.onMeterUpdate = callback;
  }

  private notifyTransportChange(): void {
    if (this.onTransportChange) {
      this.onTransportChange({ ...this.transport });
    }
  }

  // ============================================================================
  // Cleanup
  // ============================================================================

  dispose(): void {
    this.stopAllSources();
    this.tracks.clear();
    this.audioBuffers.clear();
    this.audioContext.close();
  }
}

// ============================================================================
// Track Node (Internal)
// ============================================================================

class TrackNode {
  private track: AudioTrack;
  private inputGain: GainNode;
  private panNode: StereoPannerNode;
  private outputGain: GainNode;
  private analyser: AnalyserNode;
  private effectNodes: AudioNode[] = [];

  constructor(private context: AudioContext, track: AudioTrack) {
    this.track = track;

    this.inputGain = context.createGain();
    this.panNode = context.createStereoPanner();
    this.outputGain = context.createGain();
    this.analyser = context.createAnalyser();

    // Initial chain
    this.inputGain.connect(this.panNode);
    this.panNode.connect(this.outputGain);
    this.outputGain.connect(this.analyser);

    // Apply initial settings
    this.outputGain.gain.value = track.volume;
    this.panNode.pan.value = track.pan;

    // Build effect chain
    this.updateEffects(track.effects);
  }

  connect(destination: AudioNode): void {
    this.analyser.connect(destination);
  }

  disconnect(): void {
    this.analyser.disconnect();
  }

  getTrack(): AudioTrack {
    return this.track;
  }

  getInputNode(): AudioNode {
    return this.inputGain;
  }

  getOutputNode(): AudioNode {
    return this.outputGain;
  }

  setVolume(value: number): void {
    this.track.volume = value;
    this.outputGain.gain.setTargetAtTime(value, this.context.currentTime, 0.01);
  }

  setPan(value: number): void {
    this.track.pan = value;
    this.panNode.pan.setTargetAtTime(value, this.context.currentTime, 0.01);
  }

  setMuted(muted: boolean): void {
    this.track.muted = muted;
    this.outputGain.gain.setTargetAtTime(muted ? 0 : this.track.volume, this.context.currentTime, 0.01);
  }

  updateEffects(effects: Effect[]): void {
    // Disconnect existing effects
    this.inputGain.disconnect();
    for (const node of this.effectNodes) {
      node.disconnect();
    }
    this.effectNodes = [];

    // Build new chain
    let lastNode: AudioNode = this.inputGain;

    for (const effect of effects) {
      if (!effect.enabled) continue;

      const effectNode = this.createEffectNode(effect);
      if (effectNode) {
        lastNode.connect(effectNode);
        lastNode = effectNode;
        this.effectNodes.push(effectNode);
      }
    }

    lastNode.connect(this.panNode);
    this.track.effects = effects;
  }

  private createEffectNode(effect: Effect): AudioNode | null {
    switch (effect.type) {
      case 'eq':
        return this.createEQ(effect.params);
      case 'compressor':
        return this.createCompressor(effect.params);
      case 'reverb':
        return this.createReverb(effect.params);
      case 'delay':
        return this.createDelay(effect.params);
      case 'filter':
        return this.createFilter(effect.params);
      case 'gain':
        return this.createGain(effect.params);
      default:
        return null;
    }
  }

  private createEQ(params: Record<string, number>): AudioNode {
    const low = this.context.createBiquadFilter();
    low.type = 'lowshelf';
    low.frequency.value = 320;
    low.gain.value = params.low || 0;

    const mid = this.context.createBiquadFilter();
    mid.type = 'peaking';
    mid.frequency.value = 1000;
    mid.Q.value = 0.5;
    mid.gain.value = params.mid || 0;

    const high = this.context.createBiquadFilter();
    high.type = 'highshelf';
    high.frequency.value = 3200;
    high.gain.value = params.high || 0;

    low.connect(mid);
    mid.connect(high);

    return low;
  }

  private createCompressor(params: Record<string, number>): AudioNode {
    const compressor = this.context.createDynamicsCompressor();
    compressor.threshold.value = params.threshold || -24;
    compressor.knee.value = params.knee || 30;
    compressor.ratio.value = params.ratio || 12;
    compressor.attack.value = params.attack || 0.003;
    compressor.release.value = params.release || 0.25;
    return compressor;
  }

  private createReverb(params: Record<string, number>): AudioNode {
    // Simplified convolution reverb would use impulse responses
    // For now, use a simple feedback delay network
    const wet = this.context.createGain();
    wet.gain.value = params.mix || 0.3;
    return wet;
  }

  private createDelay(params: Record<string, number>): AudioNode {
    const delay = this.context.createDelay(5);
    delay.delayTime.value = params.time || 0.5;

    const feedback = this.context.createGain();
    feedback.gain.value = params.feedback || 0.3;

    const wet = this.context.createGain();
    wet.gain.value = params.mix || 0.3;

    delay.connect(feedback);
    feedback.connect(delay);
    delay.connect(wet);

    return delay;
  }

  private createFilter(params: Record<string, number>): AudioNode {
    const filter = this.context.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = params.frequency || 1000;
    filter.Q.value = params.resonance || 1;
    return filter;
  }

  private createGain(params: Record<string, number>): AudioNode {
    const gain = this.context.createGain();
    gain.gain.value = params.gain || 1;
    return gain;
  }

  getMeterLevels(): MeterLevels {
    const dataArray = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(dataArray);

    let sum = 0;
    let peak = 0;

    for (const sample of dataArray) {
      const abs = Math.abs(sample);
      sum += abs * abs;
      if (abs > peak) peak = abs;
    }

    const rms = Math.sqrt(sum / dataArray.length);

    return { left: rms, right: rms, peak };
  }
}

export default AudioEngine;
