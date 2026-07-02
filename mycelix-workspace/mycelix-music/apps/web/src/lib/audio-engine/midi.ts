// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * MIDI System
 *
 * Complete MIDI implementation:
 * - Web MIDI API integration
 * - MIDI input/output handling
 * - MIDI message parsing
 * - MIDI clock/sync
 * - Virtual MIDI keyboard
 * - MIDI learn
 * - MIDI file playback
 */

// ==================== Types ====================

export type MIDIMessageType =
  | 'noteOn'
  | 'noteOff'
  | 'controlChange'
  | 'programChange'
  | 'pitchBend'
  | 'aftertouch'
  | 'channelPressure'
  | 'clock'
  | 'start'
  | 'stop'
  | 'continue'
  | 'sysex';

export interface MIDIMessage {
  type: MIDIMessageType;
  channel: number;
  data: number[];
  timestamp: number;
  raw: Uint8Array;
}

export interface MIDINoteMessage extends MIDIMessage {
  type: 'noteOn' | 'noteOff';
  note: number;
  velocity: number;
  noteName: string;
}

export interface MIDIControlMessage extends MIDIMessage {
  type: 'controlChange';
  controller: number;
  value: number;
  controllerName: string;
}

export interface MIDIDeviceInfo {
  id: string;
  name: string;
  manufacturer: string;
  type: 'input' | 'output';
  state: 'connected' | 'disconnected';
}

export interface MIDILearnBinding {
  id: string;
  targetParam: string;
  channel: number;
  controller: number;
  min: number;
  max: number;
  curve: 'linear' | 'logarithmic';
}

// ==================== Constants ====================

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

const CC_NAMES: Record<number, string> = {
  0: 'Bank Select MSB',
  1: 'Modulation',
  2: 'Breath Controller',
  4: 'Foot Controller',
  5: 'Portamento Time',
  6: 'Data Entry MSB',
  7: 'Volume',
  8: 'Balance',
  10: 'Pan',
  11: 'Expression',
  64: 'Sustain Pedal',
  65: 'Portamento',
  66: 'Sostenuto',
  67: 'Soft Pedal',
  68: 'Legato',
  69: 'Hold 2',
  71: 'Resonance',
  72: 'Release',
  73: 'Attack',
  74: 'Cutoff',
  91: 'Reverb',
  93: 'Chorus',
  94: 'Detune',
  95: 'Phaser',
};

// ==================== MIDI Parser ====================

export function parseMIDIMessage(data: Uint8Array, timestamp: number): MIDIMessage | null {
  if (data.length === 0) return null;

  const status = data[0];
  const type = status >> 4;
  const channel = (status & 0x0f) + 1;

  const raw = data;

  switch (type) {
    case 0x9: // Note On
      if (data[2] === 0) {
        // Velocity 0 = Note Off
        return {
          type: 'noteOff',
          channel,
          data: [data[1], data[2]],
          timestamp,
          raw,
          note: data[1],
          velocity: data[2],
          noteName: `${NOTE_NAMES[data[1] % 12]}${Math.floor(data[1] / 12) - 1}`,
        } as MIDINoteMessage;
      }
      return {
        type: 'noteOn',
        channel,
        data: [data[1], data[2]],
        timestamp,
        raw,
        note: data[1],
        velocity: data[2],
        noteName: `${NOTE_NAMES[data[1] % 12]}${Math.floor(data[1] / 12) - 1}`,
      } as MIDINoteMessage;

    case 0x8: // Note Off
      return {
        type: 'noteOff',
        channel,
        data: [data[1], data[2]],
        timestamp,
        raw,
        note: data[1],
        velocity: data[2],
        noteName: `${NOTE_NAMES[data[1] % 12]}${Math.floor(data[1] / 12) - 1}`,
      } as MIDINoteMessage;

    case 0xb: // Control Change
      return {
        type: 'controlChange',
        channel,
        data: [data[1], data[2]],
        timestamp,
        raw,
        controller: data[1],
        value: data[2],
        controllerName: CC_NAMES[data[1]] || `CC ${data[1]}`,
      } as MIDIControlMessage;

    case 0xc: // Program Change
      return {
        type: 'programChange',
        channel,
        data: [data[1]],
        timestamp,
        raw,
      };

    case 0xe: // Pitch Bend
      const bend = ((data[2] << 7) | data[1]) - 8192;
      return {
        type: 'pitchBend',
        channel,
        data: [bend],
        timestamp,
        raw,
      };

    case 0xa: // Polyphonic Aftertouch
      return {
        type: 'aftertouch',
        channel,
        data: [data[1], data[2]],
        timestamp,
        raw,
      };

    case 0xd: // Channel Pressure
      return {
        type: 'channelPressure',
        channel,
        data: [data[1]],
        timestamp,
        raw,
      };

    case 0xf: // System Messages
      switch (status) {
        case 0xf8:
          return { type: 'clock', channel: 0, data: [], timestamp, raw };
        case 0xfa:
          return { type: 'start', channel: 0, data: [], timestamp, raw };
        case 0xfb:
          return { type: 'continue', channel: 0, data: [], timestamp, raw };
        case 0xfc:
          return { type: 'stop', channel: 0, data: [], timestamp, raw };
        case 0xf0:
          return { type: 'sysex', channel: 0, data: Array.from(data), timestamp, raw };
      }
      break;
  }

  return null;
}

// ==================== MIDI Clock ====================

export class MIDIClock {
  private ppq = 24; // Pulses per quarter note
  private bpm = 120;
  private isRunning = false;
  private lastTickTime = 0;
  private tickCount = 0;

  public onTick?: (tick: number) => void;
  public onBeat?: (beat: number) => void;
  public onBar?: (bar: number) => void;

  constructor(bpm = 120) {
    this.bpm = bpm;
  }

  setBPM(bpm: number): void {
    this.bpm = bpm;
  }

  getBPM(): number {
    return this.bpm;
  }

  getTickInterval(): number {
    return 60000 / (this.bpm * this.ppq);
  }

  processClock(timestamp: number): void {
    if (!this.isRunning) return;

    this.tickCount++;
    this.lastTickTime = timestamp;

    this.onTick?.(this.tickCount);

    if (this.tickCount % this.ppq === 0) {
      const beat = this.tickCount / this.ppq;
      this.onBeat?.(beat);

      if (beat % 4 === 0) {
        this.onBar?.(beat / 4);
      }
    }
  }

  start(): void {
    this.isRunning = true;
    this.tickCount = 0;
  }

  stop(): void {
    this.isRunning = false;
  }

  continue(): void {
    this.isRunning = true;
  }

  reset(): void {
    this.tickCount = 0;
  }

  getCurrentPosition(): { tick: number; beat: number; bar: number } {
    return {
      tick: this.tickCount,
      beat: Math.floor(this.tickCount / this.ppq),
      bar: Math.floor(this.tickCount / (this.ppq * 4)),
    };
  }
}

// ==================== MIDI Manager ====================

export class MIDIManager {
  private access: MIDIAccess | null = null;
  private inputs: Map<string, MIDIInput> = new Map();
  private outputs: Map<string, MIDIOutput> = new Map();
  private clock: MIDIClock;
  private bindings: Map<string, MIDILearnBinding> = new Map();
  private isLearning = false;
  private learnCallback?: (binding: Partial<MIDILearnBinding>) => void;

  public onMessage?: (message: MIDIMessage, inputId: string) => void;
  public onDeviceChange?: (devices: MIDIDeviceInfo[]) => void;
  public onNoteOn?: (note: MIDINoteMessage) => void;
  public onNoteOff?: (note: MIDINoteMessage) => void;
  public onControlChange?: (cc: MIDIControlMessage) => void;

  constructor() {
    this.clock = new MIDIClock();
  }

  async initialize(): Promise<void> {
    if (!navigator.requestMIDIAccess) {
      throw new Error('Web MIDI API not supported');
    }

    this.access = await navigator.requestMIDIAccess({ sysex: true });

    // Set up device change listener
    this.access.onstatechange = () => {
      this.updateDevices();
    };

    this.updateDevices();
  }

  private updateDevices(): void {
    if (!this.access) return;

    // Clear old listeners
    for (const input of this.inputs.values()) {
      input.onmidimessage = null;
    }

    this.inputs.clear();
    this.outputs.clear();

    // Add inputs
    for (const input of this.access.inputs.values()) {
      this.inputs.set(input.id, input);
      input.onmidimessage = (event) => this.handleMIDIMessage(event, input.id);
    }

    // Add outputs
    for (const output of this.access.outputs.values()) {
      this.outputs.set(output.id, output);
    }

    this.onDeviceChange?.(this.getDevices());
  }

  private handleMIDIMessage(event: MIDIMessageEvent, inputId: string): void {
    const message = parseMIDIMessage(new Uint8Array(event.data!), event.timeStamp);
    if (!message) return;

    // Handle clock messages
    if (message.type === 'clock') {
      this.clock.processClock(event.timeStamp);
    } else if (message.type === 'start') {
      this.clock.start();
    } else if (message.type === 'stop') {
      this.clock.stop();
    } else if (message.type === 'continue') {
      this.clock.continue();
    }

    // Handle MIDI learn
    if (this.isLearning && message.type === 'controlChange') {
      const cc = message as MIDIControlMessage;
      this.learnCallback?.({
        channel: cc.channel,
        controller: cc.controller,
      });
      return;
    }

    // Process bindings
    if (message.type === 'controlChange') {
      this.processBinding(message as MIDIControlMessage);
    }

    // Emit events
    this.onMessage?.(message, inputId);

    if (message.type === 'noteOn') {
      this.onNoteOn?.(message as MIDINoteMessage);
    } else if (message.type === 'noteOff') {
      this.onNoteOff?.(message as MIDINoteMessage);
    } else if (message.type === 'controlChange') {
      this.onControlChange?.(message as MIDIControlMessage);
    }
  }

  private processBinding(cc: MIDIControlMessage): void {
    for (const binding of this.bindings.values()) {
      if (binding.channel === cc.channel && binding.controller === cc.controller) {
        let value = cc.value / 127;

        if (binding.curve === 'logarithmic') {
          value = Math.pow(value, 2);
        }

        value = binding.min + value * (binding.max - binding.min);

        // Emit binding update (would be connected to parameter system)
        window.dispatchEvent(new CustomEvent('midi-binding', {
          detail: { param: binding.targetParam, value },
        }));
      }
    }
  }

  getDevices(): MIDIDeviceInfo[] {
    const devices: MIDIDeviceInfo[] = [];

    for (const input of this.inputs.values()) {
      devices.push({
        id: input.id,
        name: input.name || 'Unknown Input',
        manufacturer: input.manufacturer || 'Unknown',
        type: 'input',
        state: input.state as 'connected' | 'disconnected',
      });
    }

    for (const output of this.outputs.values()) {
      devices.push({
        id: output.id,
        name: output.name || 'Unknown Output',
        manufacturer: output.manufacturer || 'Unknown',
        type: 'output',
        state: output.state as 'connected' | 'disconnected',
      });
    }

    return devices;
  }

  getInputs(): MIDIDeviceInfo[] {
    return this.getDevices().filter(d => d.type === 'input');
  }

  getOutputs(): MIDIDeviceInfo[] {
    return this.getDevices().filter(d => d.type === 'output');
  }

  // ==================== MIDI Output ====================

  sendNoteOn(outputId: string, note: number, velocity: number, channel = 1): void {
    const output = this.outputs.get(outputId);
    if (output) {
      output.send([0x90 | (channel - 1), note, velocity]);
    }
  }

  sendNoteOff(outputId: string, note: number, channel = 1): void {
    const output = this.outputs.get(outputId);
    if (output) {
      output.send([0x80 | (channel - 1), note, 0]);
    }
  }

  sendCC(outputId: string, controller: number, value: number, channel = 1): void {
    const output = this.outputs.get(outputId);
    if (output) {
      output.send([0xb0 | (channel - 1), controller, value]);
    }
  }

  sendProgramChange(outputId: string, program: number, channel = 1): void {
    const output = this.outputs.get(outputId);
    if (output) {
      output.send([0xc0 | (channel - 1), program]);
    }
  }

  sendPitchBend(outputId: string, value: number, channel = 1): void {
    // value should be -8192 to 8191
    const output = this.outputs.get(outputId);
    if (output) {
      const bendValue = value + 8192;
      const lsb = bendValue & 0x7f;
      const msb = (bendValue >> 7) & 0x7f;
      output.send([0xe0 | (channel - 1), lsb, msb]);
    }
  }

  sendClock(outputId: string): void {
    const output = this.outputs.get(outputId);
    if (output) {
      output.send([0xf8]);
    }
  }

  sendStart(outputId: string): void {
    const output = this.outputs.get(outputId);
    if (output) {
      output.send([0xfa]);
    }
  }

  sendStop(outputId: string): void {
    const output = this.outputs.get(outputId);
    if (output) {
      output.send([0xfc]);
    }
  }

  // ==================== MIDI Learn ====================

  startLearn(callback: (binding: Partial<MIDILearnBinding>) => void): void {
    this.isLearning = true;
    this.learnCallback = callback;
  }

  stopLearn(): void {
    this.isLearning = false;
    this.learnCallback = undefined;
  }

  addBinding(binding: MIDILearnBinding): void {
    this.bindings.set(binding.id, binding);
  }

  removeBinding(id: string): void {
    this.bindings.delete(id);
  }

  getBindings(): MIDILearnBinding[] {
    return Array.from(this.bindings.values());
  }

  // ==================== Clock ====================

  getClock(): MIDIClock {
    return this.clock;
  }

  // ==================== Cleanup ====================

  dispose(): void {
    for (const input of this.inputs.values()) {
      input.onmidimessage = null;
    }
    this.inputs.clear();
    this.outputs.clear();
  }
}

// ==================== Virtual Keyboard ====================

export class VirtualKeyboard {
  private activeNotes: Set<number> = new Set();
  private velocity = 100;
  private octave = 4;

  public onNoteOn?: (note: number, velocity: number) => void;
  public onNoteOff?: (note: number) => void;

  // Key to note mapping (QWERTY keyboard)
  private keyMap: Record<string, number> = {
    'a': 0,  // C
    'w': 1,  // C#
    's': 2,  // D
    'e': 3,  // D#
    'd': 4,  // E
    'f': 5,  // F
    't': 6,  // F#
    'g': 7,  // G
    'y': 8,  // G#
    'h': 9,  // A
    'u': 10, // A#
    'j': 11, // B
    'k': 12, // C (next octave)
    'o': 13, // C#
    'l': 14, // D
    'p': 15, // D#
  };

  constructor() {
    this.setupKeyboardListeners();
  }

  private setupKeyboardListeners(): void {
    if (typeof window === 'undefined') return;

    window.addEventListener('keydown', (e) => {
      if (e.repeat) return;

      const note = this.keyToNote(e.key.toLowerCase());
      if (note !== null && !this.activeNotes.has(note)) {
        this.activeNotes.add(note);
        this.onNoteOn?.(note, this.velocity);
      }

      // Octave controls
      if (e.key === 'z') {
        this.octave = Math.max(0, this.octave - 1);
      } else if (e.key === 'x') {
        this.octave = Math.min(8, this.octave + 1);
      }

      // Velocity controls
      if (e.key === 'c') {
        this.velocity = Math.max(1, this.velocity - 10);
      } else if (e.key === 'v') {
        this.velocity = Math.min(127, this.velocity + 10);
      }
    });

    window.addEventListener('keyup', (e) => {
      const note = this.keyToNote(e.key.toLowerCase());
      if (note !== null && this.activeNotes.has(note)) {
        this.activeNotes.delete(note);
        this.onNoteOff?.(note);
      }
    });
  }

  private keyToNote(key: string): number | null {
    const offset = this.keyMap[key];
    if (offset === undefined) return null;
    return this.octave * 12 + offset;
  }

  triggerNote(note: number, velocity: number): void {
    this.activeNotes.add(note);
    this.onNoteOn?.(note, velocity);
  }

  releaseNote(note: number): void {
    this.activeNotes.delete(note);
    this.onNoteOff?.(note);
  }

  releaseAll(): void {
    for (const note of this.activeNotes) {
      this.onNoteOff?.(note);
    }
    this.activeNotes.clear();
  }

  setVelocity(velocity: number): void {
    this.velocity = Math.max(1, Math.min(127, velocity));
  }

  setOctave(octave: number): void {
    this.octave = Math.max(0, Math.min(8, octave));
  }

  getActiveNotes(): number[] {
    return Array.from(this.activeNotes);
  }
}

// ==================== Singleton ====================

let midiManager: MIDIManager | null = null;

export function getMIDIManager(): MIDIManager {
  if (!midiManager) {
    midiManager = new MIDIManager();
  }
  return midiManager;
}

export default {
  MIDIManager,
  MIDIClock,
  VirtualKeyboard,
  getMIDIManager,
  parseMIDIMessage,
};
