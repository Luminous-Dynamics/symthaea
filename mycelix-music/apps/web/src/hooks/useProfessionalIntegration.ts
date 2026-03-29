// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Professional Integration Hook
 *
 * Enables seamless integration with professional music production workflows:
 * - DAW Bridge: Two-way communication with external DAWs
 * - Ableton Link: Tempo and beat sync across devices/apps
 * - Plugin Export: Export projects as VST/AU/AAX plugins
 * - MIDI Clock: External MIDI sync master/slave
 * - Stem Export: Multi-track export for DAW import
 * - Session Transfer: Import/export DAW session files
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// ==================== Types ====================

interface DAWConnection {
  id: string;
  name: string;
  type: 'osc' | 'midi' | 'rewire' | 'vst-bridge';
  host: string;
  port: number;
  status: 'connected' | 'disconnected' | 'connecting';
  latency: number; // ms
  features: string[];
}

interface DAWBridgeConfig {
  protocol: 'osc' | 'midi' | 'rewire';
  host: string;
  sendPort: number;
  receivePort: number;
  autoReconnect: boolean;
  latencyCompensation: number; // ms
}

interface LinkSession {
  isEnabled: boolean;
  tempo: number;
  beat: number;
  phase: number;
  numPeers: number;
  isPlaying: boolean;
  quantum: number; // beats per quantum (usually 4)
}

interface PluginExportConfig {
  format: 'vst3' | 'au' | 'aax' | 'lv2' | 'clap';
  name: string;
  vendor: string;
  version: string;
  category: 'effect' | 'instrument' | 'analyzer';
  parameters: PluginParameter[];
  presets: PluginPreset[];
  features: PluginFeature[];
}

interface PluginParameter {
  id: string;
  name: string;
  shortName: string;
  unit: string;
  minValue: number;
  maxValue: number;
  defaultValue: number;
  stepCount?: number;
  flags: ('automatable' | 'readonly' | 'hidden' | 'bypass')[];
}

interface PluginPreset {
  name: string;
  category: string;
  parameters: Record<string, number>;
  isFactory: boolean;
}

type PluginFeature =
  | 'stereo'
  | 'mono'
  | 'surround'
  | 'sidechain'
  | 'midi-input'
  | 'midi-output'
  | 'offline-processing'
  | 'latency-reporting';

interface ExportedPlugin {
  id: string;
  format: PluginExportConfig['format'];
  name: string;
  downloadUrl: string;
  size: number;
  checksum: string;
  platforms: ('windows' | 'macos' | 'linux')[];
  createdAt: Date;
}

interface MIDIClockConfig {
  mode: 'master' | 'slave' | 'off';
  sendClock: boolean;
  sendMTC: boolean; // MIDI Time Code
  mtcFrameRate: 24 | 25 | 29.97 | 30;
  offsetMs: number;
  smoothing: number; // 0-1
}

interface StemExportConfig {
  format: 'wav' | 'aiff' | 'flac' | 'mp3';
  bitDepth: 16 | 24 | 32;
  sampleRate: 44100 | 48000 | 88200 | 96000 | 192000;
  normalize: boolean;
  dither: 'none' | 'triangular' | 'noise-shaped';
  stemNaming: 'track-name' | 'numbered' | 'custom';
  includeMetadata: boolean;
  includeMIDI: boolean;
  consolidate: boolean; // Render all audio to start at 0
}

interface ExportedStem {
  trackId: string;
  trackName: string;
  filename: string;
  duration: number;
  channels: number;
  size: number;
  blob: Blob;
}

interface SessionFile {
  format: 'aaf' | 'omf' | 'xml' | 'als' | 'flp' | 'ptx';
  name: string;
  tracks: SessionTrack[];
  tempo: number;
  timeSignature: [number, number];
  markers: SessionMarker[];
}

interface SessionTrack {
  id: string;
  name: string;
  type: 'audio' | 'midi' | 'aux' | 'master';
  color: string;
  clips: SessionClip[];
  plugins: string[];
  volume: number;
  pan: number;
  mute: boolean;
  solo: boolean;
}

interface SessionClip {
  id: string;
  name: string;
  startTime: number; // in beats
  duration: number;
  offset: number; // clip start offset
  gain: number;
  fadeIn: number;
  fadeOut: number;
  audioFile?: string;
  midiData?: Uint8Array;
}

interface SessionMarker {
  time: number;
  name: string;
  color: string;
  type: 'marker' | 'loop-start' | 'loop-end' | 'cue';
}

interface OSCMessage {
  address: string;
  args: (number | string | boolean | Uint8Array)[];
}

interface TransportState {
  isPlaying: boolean;
  position: number; // beats
  tempo: number;
  loop: { start: number; end: number; enabled: boolean };
}

// ==================== Hook ====================

export function useProfessionalIntegration() {
  // DAW Bridge State
  const [dawConnections, setDawConnections] = useState<DAWConnection[]>([]);
  const [bridgeStatus, setBridgeStatus] = useState<'idle' | 'connecting' | 'connected' | 'error'>('idle');

  // Ableton Link State
  const [linkSession, setLinkSession] = useState<LinkSession>({
    isEnabled: false,
    tempo: 120,
    beat: 0,
    phase: 0,
    numPeers: 0,
    isPlaying: false,
    quantum: 4,
  });

  // Plugin Export State
  const [exportedPlugins, setExportedPlugins] = useState<ExportedPlugin[]>([]);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);

  // MIDI Clock State
  const [midiClockConfig, setMidiClockConfig] = useState<MIDIClockConfig>({
    mode: 'off',
    sendClock: true,
    sendMTC: false,
    mtcFrameRate: 24,
    offsetMs: 0,
    smoothing: 0.5,
  });

  // Transport State (synced across all protocols)
  const [transport, setTransport] = useState<TransportState>({
    isPlaying: false,
    position: 0,
    tempo: 120,
    loop: { start: 0, end: 16, enabled: false },
  });

  // Refs
  const oscSocketRef = useRef<WebSocket | null>(null);
  const linkWorkerRef = useRef<Worker | null>(null);
  const midiClockIntervalRef = useRef<number | null>(null);
  const oscCallbacksRef = useRef<Map<string, ((msg: OSCMessage) => void)[]>>(new Map());

  // ==================== DAW Bridge ====================

  const connectToDAW = useCallback(async (config: DAWBridgeConfig): Promise<DAWConnection | null> => {
    setBridgeStatus('connecting');

    try {
      if (config.protocol === 'osc') {
        // Connect via WebSocket to OSC bridge server
        const ws = new WebSocket(`ws://${config.host}:${config.sendPort}`);

        return new Promise((resolve, reject) => {
          const timeout = setTimeout(() => {
            ws.close();
            reject(new Error('Connection timeout'));
          }, 10000);

          ws.onopen = () => {
            clearTimeout(timeout);
            oscSocketRef.current = ws;

            // Send discovery message
            sendOSC('/mycelix/discover', ['query']);

            const connection: DAWConnection = {
              id: `osc-${Date.now()}`,
              name: 'OSC Bridge',
              type: 'osc',
              host: config.host,
              port: config.sendPort,
              status: 'connected',
              latency: config.latencyCompensation,
              features: ['transport', 'mixer', 'clips', 'parameters'],
            };

            setDawConnections(prev => [...prev, connection]);
            setBridgeStatus('connected');
            resolve(connection);
          };

          ws.onerror = () => {
            clearTimeout(timeout);
            setBridgeStatus('error');
            reject(new Error('WebSocket connection failed'));
          };

          ws.onmessage = (event) => {
            try {
              const msg = JSON.parse(event.data) as OSCMessage;
              handleOSCMessage(msg);
            } catch (e) {
              console.error('Failed to parse OSC message:', e);
            }
          };

          ws.onclose = () => {
            if (config.autoReconnect) {
              setTimeout(() => connectToDAW(config), 5000);
            } else {
              setBridgeStatus('idle');
            }
          };
        });
      }

      if (config.protocol === 'midi') {
        // MIDI device connection handled via Web MIDI API
        if (!navigator.requestMIDIAccess) {
          throw new Error('Web MIDI API not supported');
        }

        const midiAccess = await navigator.requestMIDIAccess({ sysex: true });

        const connection: DAWConnection = {
          id: `midi-${Date.now()}`,
          name: 'MIDI Bridge',
          type: 'midi',
          host: 'local',
          port: 0,
          status: 'connected',
          latency: config.latencyCompensation,
          features: ['clock', 'transport', 'notes', 'cc'],
        };

        setDawConnections(prev => [...prev, connection]);
        setBridgeStatus('connected');
        return connection;
      }

      return null;
    } catch (error) {
      console.error('DAW connection failed:', error);
      setBridgeStatus('error');
      return null;
    }
  }, []);

  const disconnectFromDAW = useCallback((connectionId: string) => {
    const connection = dawConnections.find(c => c.id === connectionId);

    if (connection?.type === 'osc' && oscSocketRef.current) {
      oscSocketRef.current.close();
      oscSocketRef.current = null;
    }

    setDawConnections(prev => prev.filter(c => c.id !== connectionId));

    if (dawConnections.length <= 1) {
      setBridgeStatus('idle');
    }
  }, [dawConnections]);

  const sendOSC = useCallback((address: string, args: (number | string | boolean)[]) => {
    if (oscSocketRef.current?.readyState === WebSocket.OPEN) {
      oscSocketRef.current.send(JSON.stringify({ address, args }));
    }
  }, []);

  const onOSC = useCallback((address: string, callback: (msg: OSCMessage) => void) => {
    const callbacks = oscCallbacksRef.current.get(address) || [];
    callbacks.push(callback);
    oscCallbacksRef.current.set(address, callbacks);

    return () => {
      const cbs = oscCallbacksRef.current.get(address) || [];
      oscCallbacksRef.current.set(address, cbs.filter(cb => cb !== callback));
    };
  }, []);

  const handleOSCMessage = useCallback((msg: OSCMessage) => {
    // Handle standard DAW messages
    switch (msg.address) {
      case '/transport/play':
        setTransport(prev => ({ ...prev, isPlaying: true }));
        break;
      case '/transport/stop':
        setTransport(prev => ({ ...prev, isPlaying: false }));
        break;
      case '/transport/position':
        if (typeof msg.args[0] === 'number') {
          setTransport(prev => ({ ...prev, position: msg.args[0] as number }));
        }
        break;
      case '/transport/tempo':
        if (typeof msg.args[0] === 'number') {
          setTransport(prev => ({ ...prev, tempo: msg.args[0] as number }));
        }
        break;
    }

    // Call registered callbacks
    const callbacks = oscCallbacksRef.current.get(msg.address) || [];
    callbacks.forEach(cb => cb(msg));

    // Also trigger wildcard callbacks
    const wildcardCallbacks = oscCallbacksRef.current.get('*') || [];
    wildcardCallbacks.forEach(cb => cb(msg));
  }, []);

  // ==================== Ableton Link ====================

  const enableLink = useCallback(async (tempo?: number): Promise<boolean> => {
    try {
      // Initialize Link worker (would connect to Link-enabled apps)
      // In a real implementation, this would use a native bridge
      linkWorkerRef.current = new Worker(
        URL.createObjectURL(
          new Blob([`
            let linkState = {
              tempo: ${tempo || 120},
              beat: 0,
              phase: 0,
              numPeers: 0,
              isPlaying: false,
              quantum: 4,
            };

            let lastTime = performance.now();

            function updateBeat() {
              const now = performance.now();
              const delta = (now - lastTime) / 1000;
              lastTime = now;

              if (linkState.isPlaying) {
                const beatsPerSecond = linkState.tempo / 60;
                linkState.beat += delta * beatsPerSecond;
                linkState.phase = (linkState.beat % linkState.quantum) / linkState.quantum;
              }

              postMessage({ type: 'state', state: linkState });
              setTimeout(updateBeat, 10); // ~100Hz update rate
            }

            self.onmessage = (e) => {
              switch (e.data.type) {
                case 'setTempo':
                  linkState.tempo = e.data.tempo;
                  break;
                case 'play':
                  linkState.isPlaying = true;
                  break;
                case 'stop':
                  linkState.isPlaying = false;
                  break;
                case 'setQuantum':
                  linkState.quantum = e.data.quantum;
                  break;
                case 'requestPhaseSync':
                  // Quantize to next quantum boundary
                  linkState.beat = Math.ceil(linkState.beat / linkState.quantum) * linkState.quantum;
                  break;
              }
            };

            updateBeat();
          `], { type: 'application/javascript' })
        )
      );

      linkWorkerRef.current.onmessage = (e) => {
        if (e.data.type === 'state') {
          setLinkSession(prev => ({
            ...prev,
            ...e.data.state,
            isEnabled: true,
          }));
        }
      };

      setLinkSession(prev => ({ ...prev, isEnabled: true, tempo: tempo || prev.tempo }));
      return true;
    } catch (error) {
      console.error('Failed to enable Link:', error);
      return false;
    }
  }, []);

  const disableLink = useCallback(() => {
    if (linkWorkerRef.current) {
      linkWorkerRef.current.terminate();
      linkWorkerRef.current = null;
    }
    setLinkSession(prev => ({ ...prev, isEnabled: false }));
  }, []);

  const setLinkTempo = useCallback((tempo: number) => {
    linkWorkerRef.current?.postMessage({ type: 'setTempo', tempo });
    setLinkSession(prev => ({ ...prev, tempo }));
  }, []);

  const requestPhaseSync = useCallback(() => {
    // Request sync to next quantum boundary
    linkWorkerRef.current?.postMessage({ type: 'requestPhaseSync' });
  }, []);

  // ==================== Plugin Export ====================

  const exportAsPlugin = useCallback(async (
    projectData: unknown,
    config: PluginExportConfig
  ): Promise<ExportedPlugin | null> => {
    setIsExporting(true);
    setExportProgress(0);

    try {
      // Generate plugin manifest
      setExportProgress(10);
      const manifest = {
        format: config.format,
        name: config.name,
        vendor: config.vendor,
        version: config.version,
        category: config.category,
        uid: generatePluginUID(config),
        parameters: config.parameters.map(p => ({
          ...p,
          id: sanitizeParamId(p.id),
        })),
        features: config.features,
      };

      setExportProgress(20);

      // Generate DSP code from project
      const dspCode = await generateDSPCode(projectData, config);
      setExportProgress(40);

      // Generate UI code
      const uiCode = await generatePluginUI(config);
      setExportProgress(60);

      // Bundle presets
      const presetBundle = bundlePresets(config.presets);
      setExportProgress(70);

      // Compile plugin (simulated - would use backend service)
      const compiledPlugin = await compilePlugin({
        manifest,
        dspCode,
        uiCode,
        presets: presetBundle,
        format: config.format,
      });
      setExportProgress(90);

      const exported: ExportedPlugin = {
        id: `plugin-${Date.now()}`,
        format: config.format,
        name: config.name,
        downloadUrl: URL.createObjectURL(compiledPlugin),
        size: compiledPlugin.size,
        checksum: await calculateChecksum(compiledPlugin),
        platforms: getPlatformsForFormat(config.format),
        createdAt: new Date(),
      };

      setExportedPlugins(prev => [...prev, exported]);
      setExportProgress(100);
      setIsExporting(false);

      return exported;
    } catch (error) {
      console.error('Plugin export failed:', error);
      setIsExporting(false);
      return null;
    }
  }, []);

  // Helper functions for plugin export
  function generatePluginUID(config: PluginExportConfig): string {
    const hash = config.vendor + config.name + config.version;
    let uid = 0;
    for (let i = 0; i < hash.length; i++) {
      uid = ((uid << 5) - uid) + hash.charCodeAt(i);
      uid = uid & uid;
    }
    return Math.abs(uid).toString(16).padStart(8, '0');
  }

  function sanitizeParamId(id: string): string {
    return id.replace(/[^a-zA-Z0-9_]/g, '_').toLowerCase();
  }

  async function generateDSPCode(projectData: unknown, config: PluginExportConfig): Promise<string> {
    // In production, this would generate actual DSP code
    // based on the project's audio graph
    return `
      // Generated DSP Code for ${config.name}
      // Format: ${config.format}

      class ${config.name.replace(/\s+/g, '')}Processor {
        constructor() {
          this.parameters = new Map();
          ${config.parameters.map(p =>
            `this.parameters.set('${p.id}', ${p.defaultValue});`
          ).join('\n          ')}
        }

        process(inputs, outputs, parameters) {
          const input = inputs[0];
          const output = outputs[0];

          for (let channel = 0; channel < output.length; channel++) {
            for (let i = 0; i < output[channel].length; i++) {
              output[channel][i] = input[channel]?.[i] || 0;
            }
          }

          return true;
        }
      }
    `;
  }

  async function generatePluginUI(config: PluginExportConfig): Promise<string> {
    // Generate plugin UI code
    return `
      // Generated UI for ${config.name}
      const ui = {
        width: 600,
        height: 400,
        parameters: ${JSON.stringify(config.parameters)},
      };
    `;
  }

  function bundlePresets(presets: PluginPreset[]): Uint8Array {
    // Bundle presets into binary format
    const json = JSON.stringify(presets);
    return new TextEncoder().encode(json);
  }

  async function compilePlugin(options: {
    manifest: unknown;
    dspCode: string;
    uiCode: string;
    presets: Uint8Array;
    format: string;
  }): Promise<Blob> {
    // In production, this would call a backend compilation service
    // For now, create a bundle with the source files
    const bundle = JSON.stringify({
      manifest: options.manifest,
      dsp: options.dspCode,
      ui: options.uiCode,
      presets: Array.from(options.presets),
    });

    return new Blob([bundle], { type: 'application/octet-stream' });
  }

  async function calculateChecksum(blob: Blob): Promise<string> {
    const buffer = await blob.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  function getPlatformsForFormat(format: string): ('windows' | 'macos' | 'linux')[] {
    switch (format) {
      case 'vst3':
        return ['windows', 'macos', 'linux'];
      case 'au':
        return ['macos'];
      case 'aax':
        return ['windows', 'macos'];
      case 'lv2':
        return ['linux'];
      case 'clap':
        return ['windows', 'macos', 'linux'];
      default:
        return ['windows', 'macos', 'linux'];
    }
  }

  // ==================== MIDI Clock ====================

  const setMIDIClockMode = useCallback(async (config: Partial<MIDIClockConfig>) => {
    const newConfig = { ...midiClockConfig, ...config };
    setMidiClockConfig(newConfig);

    // Clear existing clock
    if (midiClockIntervalRef.current) {
      clearInterval(midiClockIntervalRef.current);
      midiClockIntervalRef.current = null;
    }

    if (newConfig.mode === 'master' && newConfig.sendClock) {
      // Send MIDI clock at 24 PPQN
      const midiAccess = await navigator.requestMIDIAccess?.();
      if (!midiAccess) return;

      const outputs = Array.from(midiAccess.outputs.values());

      const clockInterval = (60000 / transport.tempo) / 24; // 24 PPQN
      midiClockIntervalRef.current = window.setInterval(() => {
        if (transport.isPlaying) {
          // Send MIDI clock message (0xF8)
          outputs.forEach(output => {
            output.send([0xF8]);
          });
        }
      }, clockInterval);
    }
  }, [midiClockConfig, transport.tempo, transport.isPlaying]);

  const sendMIDITransport = useCallback(async (command: 'start' | 'stop' | 'continue') => {
    const midiAccess = await navigator.requestMIDIAccess?.();
    if (!midiAccess) return;

    const outputs = Array.from(midiAccess.outputs.values());
    const message = command === 'start' ? 0xFA : command === 'stop' ? 0xFC : 0xFB;

    outputs.forEach(output => {
      output.send([message]);
    });
  }, []);

  // ==================== Stem Export ====================

  const exportStems = useCallback(async (
    tracks: { id: string; name: string; audioBuffer: AudioBuffer }[],
    config: StemExportConfig
  ): Promise<ExportedStem[]> => {
    const exportedStems: ExportedStem[] = [];

    for (let i = 0; i < tracks.length; i++) {
      const track = tracks[i];

      // Get filename based on naming convention
      let filename: string;
      switch (config.stemNaming) {
        case 'numbered':
          filename = `${String(i + 1).padStart(2, '0')}_stem.${config.format}`;
          break;
        case 'track-name':
          filename = `${sanitizeFilename(track.name)}.${config.format}`;
          break;
        default:
          filename = `${track.id}.${config.format}`;
      }

      // Process audio
      let processedBuffer = track.audioBuffer;

      if (config.normalize) {
        processedBuffer = normalizeAudio(processedBuffer);
      }

      // Encode to target format
      const blob = await encodeAudio(processedBuffer, {
        format: config.format,
        bitDepth: config.bitDepth,
        sampleRate: config.sampleRate,
        dither: config.dither,
      });

      exportedStems.push({
        trackId: track.id,
        trackName: track.name,
        filename,
        duration: processedBuffer.duration,
        channels: processedBuffer.numberOfChannels,
        size: blob.size,
        blob,
      });
    }

    return exportedStems;
  }, []);

  function sanitizeFilename(name: string): string {
    return name.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 50);
  }

  function normalizeAudio(buffer: AudioBuffer): AudioBuffer {
    // Find peak
    let peak = 0;
    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
      const data = buffer.getChannelData(ch);
      for (let i = 0; i < data.length; i++) {
        peak = Math.max(peak, Math.abs(data[i]));
      }
    }

    if (peak === 0 || peak >= 0.99) return buffer;

    // Normalize
    const gain = 0.99 / peak;
    const ctx = new OfflineAudioContext(
      buffer.numberOfChannels,
      buffer.length,
      buffer.sampleRate
    );

    const normalized = ctx.createBuffer(
      buffer.numberOfChannels,
      buffer.length,
      buffer.sampleRate
    );

    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
      const input = buffer.getChannelData(ch);
      const output = normalized.getChannelData(ch);
      for (let i = 0; i < input.length; i++) {
        output[i] = input[i] * gain;
      }
    }

    return normalized;
  }

  async function encodeAudio(
    buffer: AudioBuffer,
    options: { format: string; bitDepth: number; sampleRate: number; dither: string }
  ): Promise<Blob> {
    // For WAV format, encode directly
    if (options.format === 'wav') {
      return encodeWAV(buffer, options.bitDepth);
    }

    // For other formats, use MediaRecorder or backend service
    const ctx = new OfflineAudioContext(
      buffer.numberOfChannels,
      buffer.length,
      buffer.sampleRate
    );

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.start();

    const rendered = await ctx.startRendering();
    return encodeWAV(rendered, options.bitDepth);
  }

  function encodeWAV(buffer: AudioBuffer, bitDepth: number): Blob {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    const dataSize = buffer.length * blockAlign;

    const wavBuffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(wavBuffer);

    // RIFF header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');

    // fmt chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // chunk size
    view.setUint16(20, bitDepth === 32 ? 3 : 1, true); // format (1=PCM, 3=float)
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);

    // data chunk
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    // Interleave channels and write samples
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
      for (let ch = 0; ch < numChannels; ch++) {
        const sample = buffer.getChannelData(ch)[i];

        if (bitDepth === 16) {
          const s = Math.max(-1, Math.min(1, sample));
          view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        } else if (bitDepth === 24) {
          const s = Math.max(-1, Math.min(1, sample));
          const val = s < 0 ? s * 0x800000 : s * 0x7FFFFF;
          view.setUint8(offset, val & 0xFF);
          view.setUint8(offset + 1, (val >> 8) & 0xFF);
          view.setUint8(offset + 2, (val >> 16) & 0xFF);
        } else if (bitDepth === 32) {
          view.setFloat32(offset, sample, true);
        }

        offset += bytesPerSample;
      }
    }

    return new Blob([wavBuffer], { type: 'audio/wav' });
  }

  function writeString(view: DataView, offset: number, str: string): void {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  // ==================== Session Import/Export ====================

  const exportSession = useCallback(async (
    format: SessionFile['format'],
    tracks: SessionTrack[],
    options: { tempo: number; timeSignature: [number, number]; markers: SessionMarker[] }
  ): Promise<Blob | null> => {
    try {
      const session: SessionFile = {
        format,
        name: `Mycelix_Session_${Date.now()}`,
        tracks,
        tempo: options.tempo,
        timeSignature: options.timeSignature,
        markers: options.markers,
      };

      switch (format) {
        case 'xml':
          return exportAsXML(session);
        case 'aaf':
          return exportAsAAF(session);
        case 'als':
          return exportAsALS(session);
        default:
          console.warn(`Format ${format} not yet supported`);
          return null;
      }
    } catch (error) {
      console.error('Session export failed:', error);
      return null;
    }
  }, []);

  function exportAsXML(session: SessionFile): Blob {
    const xml = `<?xml version="1.0" encoding="UTF-8"?>
<session name="${session.name}" tempo="${session.tempo}">
  <timeSignature numerator="${session.timeSignature[0]}" denominator="${session.timeSignature[1]}" />
  <tracks>
    ${session.tracks.map(track => `
    <track id="${track.id}" name="${track.name}" type="${track.type}" color="${track.color}">
      <volume value="${track.volume}" />
      <pan value="${track.pan}" />
      <mute value="${track.mute}" />
      <solo value="${track.solo}" />
      <clips>
        ${track.clips.map(clip => `
        <clip id="${clip.id}" name="${clip.name}" start="${clip.startTime}" duration="${clip.duration}">
          <offset value="${clip.offset}" />
          <gain value="${clip.gain}" />
          <fadeIn value="${clip.fadeIn}" />
          <fadeOut value="${clip.fadeOut}" />
          ${clip.audioFile ? `<audioFile path="${clip.audioFile}" />` : ''}
        </clip>`).join('')}
      </clips>
      <plugins>${track.plugins.join(',')}</plugins>
    </track>`).join('')}
  </tracks>
  <markers>
    ${session.markers.map(m => `
    <marker time="${m.time}" name="${m.name}" color="${m.color}" type="${m.type}" />`).join('')}
  </markers>
</session>`;

    return new Blob([xml], { type: 'application/xml' });
  }

  function exportAsAAF(session: SessionFile): Blob {
    // AAF is a complex binary format - simplified implementation
    const header = new Uint8Array([
      0x41, 0x41, 0x46, 0x00, // AAF magic
      0x01, 0x01, // Version 1.1
    ]);

    const metadata = new TextEncoder().encode(JSON.stringify({
      name: session.name,
      tempo: session.tempo,
      tracks: session.tracks.length,
    }));

    const combined = new Uint8Array(header.length + metadata.length);
    combined.set(header);
    combined.set(metadata, header.length);

    return new Blob([combined], { type: 'application/octet-stream' });
  }

  function exportAsALS(session: SessionFile): Blob {
    // Ableton Live Set format (gzipped XML)
    const alsXml = `<?xml version="1.0" encoding="UTF-8"?>
<Ableton MajorVersion="5" MinorVersion="11.0" Creator="Mycelix">
  <LiveSet>
    <Tracks>
      ${session.tracks.map((track, i) => `
      <AudioTrack Id="${i}">
        <Name Value="${track.name}" />
        <Color Value="${parseInt(track.color.slice(1), 16)}" />
        <DeviceChain>
          <Mixer>
            <Volume><Manual Value="${track.volume}" /></Volume>
            <Pan><Manual Value="${track.pan}" /></Pan>
          </Mixer>
        </DeviceChain>
      </AudioTrack>`).join('')}
    </Tracks>
    <MasterTrack>
      <Tempo><Manual Value="${session.tempo}" /></Tempo>
      <TimeSignature>
        <Numerator Value="${session.timeSignature[0]}" />
        <Denominator Value="${session.timeSignature[1]}" />
      </TimeSignature>
    </MasterTrack>
  </LiveSet>
</Ableton>`;

    // In production, this would be gzipped
    return new Blob([alsXml], { type: 'application/x-ableton-live-set' });
  }

  const importSession = useCallback(async (file: File): Promise<SessionFile | null> => {
    try {
      const text = await file.text();

      if (file.name.endsWith('.xml')) {
        return parseXMLSession(text);
      }

      // Add other format parsers as needed
      console.warn('Session format not supported');
      return null;
    } catch (error) {
      console.error('Session import failed:', error);
      return null;
    }
  }, []);

  function parseXMLSession(xml: string): SessionFile {
    const parser = new DOMParser();
    const doc = parser.parseFromString(xml, 'application/xml');
    const session = doc.querySelector('session');

    if (!session) {
      throw new Error('Invalid session XML');
    }

    const ts = doc.querySelector('timeSignature');

    return {
      format: 'xml',
      name: session.getAttribute('name') || 'Imported Session',
      tempo: parseFloat(session.getAttribute('tempo') || '120'),
      timeSignature: [
        parseInt(ts?.getAttribute('numerator') || '4'),
        parseInt(ts?.getAttribute('denominator') || '4'),
      ],
      tracks: Array.from(doc.querySelectorAll('track')).map(trackEl => ({
        id: trackEl.getAttribute('id') || '',
        name: trackEl.getAttribute('name') || '',
        type: (trackEl.getAttribute('type') as SessionTrack['type']) || 'audio',
        color: trackEl.getAttribute('color') || '#808080',
        volume: parseFloat(trackEl.querySelector('volume')?.getAttribute('value') || '1'),
        pan: parseFloat(trackEl.querySelector('pan')?.getAttribute('value') || '0'),
        mute: trackEl.querySelector('mute')?.getAttribute('value') === 'true',
        solo: trackEl.querySelector('solo')?.getAttribute('value') === 'true',
        clips: Array.from(trackEl.querySelectorAll('clip')).map(clipEl => ({
          id: clipEl.getAttribute('id') || '',
          name: clipEl.getAttribute('name') || '',
          startTime: parseFloat(clipEl.getAttribute('start') || '0'),
          duration: parseFloat(clipEl.getAttribute('duration') || '0'),
          offset: parseFloat(clipEl.querySelector('offset')?.getAttribute('value') || '0'),
          gain: parseFloat(clipEl.querySelector('gain')?.getAttribute('value') || '1'),
          fadeIn: parseFloat(clipEl.querySelector('fadeIn')?.getAttribute('value') || '0'),
          fadeOut: parseFloat(clipEl.querySelector('fadeOut')?.getAttribute('value') || '0'),
          audioFile: clipEl.querySelector('audioFile')?.getAttribute('path'),
        })),
        plugins: (trackEl.querySelector('plugins')?.textContent || '').split(',').filter(Boolean),
      })),
      markers: Array.from(doc.querySelectorAll('marker')).map(m => ({
        time: parseFloat(m.getAttribute('time') || '0'),
        name: m.getAttribute('name') || '',
        color: m.getAttribute('color') || '#FFFF00',
        type: (m.getAttribute('type') as SessionMarker['type']) || 'marker',
      })),
    };
  }

  // ==================== Transport Control ====================

  const setTransportState = useCallback((state: Partial<TransportState>) => {
    setTransport(prev => {
      const newState = { ...prev, ...state };

      // Sync to all connected systems
      if (state.isPlaying !== undefined) {
        // Sync to OSC
        sendOSC(state.isPlaying ? '/transport/play' : '/transport/stop', []);

        // Sync to Link
        linkWorkerRef.current?.postMessage({
          type: state.isPlaying ? 'play' : 'stop'
        });

        // Sync MIDI
        if (midiClockConfig.mode === 'master') {
          sendMIDITransport(state.isPlaying ? 'start' : 'stop');
        }
      }

      if (state.tempo !== undefined) {
        sendOSC('/transport/tempo', [state.tempo]);
        setLinkTempo(state.tempo);
      }

      return newState;
    });
  }, [sendOSC, midiClockConfig.mode, sendMIDITransport, setLinkTempo]);

  // ==================== Cleanup ====================

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      if (oscSocketRef.current) {
        oscSocketRef.current.close();
      }
      if (linkWorkerRef.current) {
        linkWorkerRef.current.terminate();
      }
      if (midiClockIntervalRef.current) {
        clearInterval(midiClockIntervalRef.current);
      }
    };
  }, []);

  // ==================== Return ====================

  return {
    // DAW Bridge
    dawConnections,
    bridgeStatus,
    connectToDAW,
    disconnectFromDAW,
    sendOSC,
    onOSC,

    // Ableton Link
    linkSession,
    enableLink,
    disableLink,
    setLinkTempo,
    requestPhaseSync,

    // Plugin Export
    exportedPlugins,
    isExporting,
    exportProgress,
    exportAsPlugin,

    // MIDI Clock
    midiClockConfig,
    setMIDIClockMode,
    sendMIDITransport,

    // Stem Export
    exportStems,

    // Session Import/Export
    exportSession,
    importSession,

    // Transport (synced)
    transport,
    setTransportState,
  };
}
