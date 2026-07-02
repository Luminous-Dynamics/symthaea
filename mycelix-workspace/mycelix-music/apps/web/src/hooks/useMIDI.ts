// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * MIDI Controller Hook
 *
 * Web MIDI API integration for hardware controllers:
 * - MIDI input/output device management
 * - Controller mapping and learning
 * - DJ controller support
 * - Keyboard/pad input
 */

import { useState, useCallback, useEffect, useRef } from 'react';

// MIDI message types
export type MIDIMessageType =
  | 'noteon'
  | 'noteoff'
  | 'controlchange'
  | 'pitchbend'
  | 'programchange'
  | 'aftertouch'
  | 'channelaftertouch';

export interface MIDIMessage {
  type: MIDIMessageType;
  channel: number;
  note?: number;
  velocity?: number;
  controller?: number;
  value?: number;
  pitchBend?: number;
  timestamp: number;
}

export interface MIDIDevice {
  id: string;
  name: string;
  manufacturer: string;
  type: 'input' | 'output';
  state: 'connected' | 'disconnected';
  connection: 'open' | 'closed' | 'pending';
}

export interface MIDIMapping {
  id: string;
  name: string;
  type: 'note' | 'cc' | 'pitchbend';
  channel: number;
  noteOrCC: number;
  action: string;           // Action identifier
  targetId?: string;        // Target element/parameter
  min?: number;             // Min value for CC
  max?: number;             // Max value for CC
  toggle?: boolean;         // Toggle mode for buttons
  invert?: boolean;         // Invert value
}

export interface MIDIControllerProfile {
  id: string;
  name: string;
  manufacturer: string;
  mappings: MIDIMapping[];
}

// Common controller actions
export const MIDI_ACTIONS = {
  // Transport
  PLAY_PAUSE: 'transport.playPause',
  STOP: 'transport.stop',
  RECORD: 'transport.record',
  REWIND: 'transport.rewind',
  FORWARD: 'transport.forward',

  // Mixer
  VOLUME: 'mixer.volume',
  PAN: 'mixer.pan',
  MUTE: 'mixer.mute',
  SOLO: 'mixer.solo',
  CROSSFADER: 'mixer.crossfader',

  // DJ
  JOG_WHEEL: 'dj.jogWheel',
  PITCH_FADER: 'dj.pitchFader',
  CUE: 'dj.cue',
  SYNC: 'dj.sync',
  HOT_CUE: 'dj.hotCue',
  LOOP_IN: 'dj.loopIn',
  LOOP_OUT: 'dj.loopOut',

  // Effects
  EFFECT_KNOB: 'effects.knob',
  EFFECT_TOGGLE: 'effects.toggle',
  EFFECT_WET_DRY: 'effects.wetDry',

  // Navigation
  TRACK_SELECT: 'nav.trackSelect',
  BANK_LEFT: 'nav.bankLeft',
  BANK_RIGHT: 'nav.bankRight',

  // Pads
  PAD_TRIGGER: 'pads.trigger',
  PAD_VELOCITY: 'pads.velocity',
} as const;

// Built-in controller profiles
const CONTROLLER_PROFILES: MIDIControllerProfile[] = [
  {
    id: 'pioneer-ddj-400',
    name: 'Pioneer DDJ-400',
    manufacturer: 'Pioneer DJ',
    mappings: [
      { id: '1', name: 'Play/Pause L', type: 'note', channel: 1, noteOrCC: 11, action: MIDI_ACTIONS.PLAY_PAUSE, targetId: 'deck-a' },
      { id: '2', name: 'Play/Pause R', type: 'note', channel: 1, noteOrCC: 71, action: MIDI_ACTIONS.PLAY_PAUSE, targetId: 'deck-b' },
      { id: '3', name: 'Crossfader', type: 'cc', channel: 1, noteOrCC: 31, action: MIDI_ACTIONS.CROSSFADER, min: 0, max: 127 },
      { id: '4', name: 'Volume L', type: 'cc', channel: 1, noteOrCC: 19, action: MIDI_ACTIONS.VOLUME, targetId: 'deck-a' },
      { id: '5', name: 'Volume R', type: 'cc', channel: 1, noteOrCC: 23, action: MIDI_ACTIONS.VOLUME, targetId: 'deck-b' },
      { id: '6', name: 'Jog L', type: 'cc', channel: 1, noteOrCC: 33, action: MIDI_ACTIONS.JOG_WHEEL, targetId: 'deck-a' },
      { id: '7', name: 'Jog R', type: 'cc', channel: 1, noteOrCC: 35, action: MIDI_ACTIONS.JOG_WHEEL, targetId: 'deck-b' },
    ],
  },
  {
    id: 'akai-apc-mini',
    name: 'Akai APC Mini',
    manufacturer: 'Akai',
    mappings: [
      // 8x8 pad grid
      ...Array.from({ length: 64 }, (_, i) => ({
        id: `pad-${i}`,
        name: `Pad ${i + 1}`,
        type: 'note' as const,
        channel: 1,
        noteOrCC: i,
        action: MIDI_ACTIONS.PAD_TRIGGER,
        targetId: `pad-${i}`,
      })),
      // Faders
      ...Array.from({ length: 9 }, (_, i) => ({
        id: `fader-${i}`,
        name: `Fader ${i + 1}`,
        type: 'cc' as const,
        channel: 1,
        noteOrCC: 48 + i,
        action: MIDI_ACTIONS.VOLUME,
        targetId: `channel-${i}`,
      })),
    ],
  },
  {
    id: 'novation-launchpad',
    name: 'Novation Launchpad',
    manufacturer: 'Novation',
    mappings: [
      ...Array.from({ length: 64 }, (_, i) => ({
        id: `pad-${i}`,
        name: `Pad ${Math.floor(i / 8) + 1}-${(i % 8) + 1}`,
        type: 'note' as const,
        channel: 1,
        noteOrCC: i,
        action: MIDI_ACTIONS.PAD_TRIGGER,
        targetId: `pad-${i}`,
      })),
    ],
  },
];

export interface MIDIState {
  isSupported: boolean;
  isEnabled: boolean;
  inputs: MIDIDevice[];
  outputs: MIDIDevice[];
  activeProfile: MIDIControllerProfile | null;
  customMappings: MIDIMapping[];
  isLearning: boolean;
  learningTarget: string | null;
  lastMessage: MIDIMessage | null;
}

type MIDIHandler = (message: MIDIMessage) => void;

export function useMIDI() {
  const [state, setState] = useState<MIDIState>({
    isSupported: false,
    isEnabled: false,
    inputs: [],
    outputs: [],
    activeProfile: null,
    customMappings: [],
    isLearning: false,
    learningTarget: null,
    lastMessage: null,
  });

  const midiAccessRef = useRef<MIDIAccess | null>(null);
  const handlersRef = useRef<Map<string, Set<MIDIHandler>>>(new Map());
  const inputListenersRef = useRef<Map<string, (e: MIDIMessageEvent) => void>>(new Map());

  // Check MIDI support
  useEffect(() => {
    const isSupported = typeof navigator !== 'undefined' && 'requestMIDIAccess' in navigator;
    setState(prev => ({ ...prev, isSupported }));
  }, []);

  /**
   * Enable MIDI access
   */
  const enable = useCallback(async () => {
    if (!state.isSupported) {
      throw new Error('Web MIDI API not supported');
    }

    try {
      const access = await navigator.requestMIDIAccess({ sysex: false });
      midiAccessRef.current = access;

      // Get devices
      const inputs = Array.from(access.inputs.values()).map(deviceToMIDIDevice);
      const outputs = Array.from(access.outputs.values()).map(deviceToMIDIDevice);

      // Listen for device changes
      access.onstatechange = (e) => {
        const port = e.port;
        const device = deviceToMIDIDevice(port);

        setState(prev => {
          if (port.type === 'input') {
            const existing = prev.inputs.findIndex(d => d.id === device.id);
            const newInputs = [...prev.inputs];
            if (existing >= 0) {
              newInputs[existing] = device;
            } else if (port.state === 'connected') {
              newInputs.push(device);
            }
            return { ...prev, inputs: newInputs.filter(d => d.state === 'connected') };
          } else {
            const existing = prev.outputs.findIndex(d => d.id === device.id);
            const newOutputs = [...prev.outputs];
            if (existing >= 0) {
              newOutputs[existing] = device;
            } else if (port.state === 'connected') {
              newOutputs.push(device);
            }
            return { ...prev, outputs: newOutputs.filter(d => d.state === 'connected') };
          }
        });
      };

      setState(prev => ({
        ...prev,
        isEnabled: true,
        inputs,
        outputs,
      }));

      // Start listening to all inputs
      for (const input of access.inputs.values()) {
        listenToInput(input);
      }

      return true;
    } catch (error) {
      console.error('Failed to enable MIDI:', error);
      return false;
    }
  }, [state.isSupported]);

  /**
   * Listen to MIDI input
   */
  const listenToInput = useCallback((input: MIDIInput) => {
    const handleMessage = (e: MIDIMessageEvent) => {
      const message = parseMIDIMessage(e);
      if (!message) return;

      setState(prev => ({ ...prev, lastMessage: message }));

      // If learning, capture this message
      if (state.isLearning && state.learningTarget) {
        handleLearnedMessage(message);
        return;
      }

      // Find matching mapping and trigger handlers
      const mapping = findMapping(message, state.activeProfile, state.customMappings);
      if (mapping) {
        const handlers = handlersRef.current.get(mapping.action);
        handlers?.forEach(handler => {
          handler({
            ...message,
            value: transformValue(message, mapping),
          });
        });
      }

      // Also trigger any wildcard handlers
      const wildcardHandlers = handlersRef.current.get('*');
      wildcardHandlers?.forEach(handler => handler(message));
    };

    input.onmidimessage = handleMessage;
    inputListenersRef.current.set(input.id, handleMessage);
  }, [state.isLearning, state.learningTarget, state.activeProfile, state.customMappings]);

  /**
   * Handle learned MIDI message
   */
  const handleLearnedMessage = useCallback((message: MIDIMessage) => {
    if (!state.learningTarget) return;

    const mapping: MIDIMapping = {
      id: `custom-${Date.now()}`,
      name: `Learned: ${state.learningTarget}`,
      type: message.type === 'controlchange' ? 'cc' : message.type === 'pitchbend' ? 'pitchbend' : 'note',
      channel: message.channel,
      noteOrCC: message.note || message.controller || 0,
      action: state.learningTarget,
    };

    setState(prev => ({
      ...prev,
      customMappings: [...prev.customMappings, mapping],
      isLearning: false,
      learningTarget: null,
    }));
  }, [state.learningTarget]);

  /**
   * Start MIDI learn mode
   */
  const startLearn = useCallback((targetAction: string) => {
    setState(prev => ({
      ...prev,
      isLearning: true,
      learningTarget: targetAction,
    }));
  }, []);

  /**
   * Cancel MIDI learn mode
   */
  const cancelLearn = useCallback(() => {
    setState(prev => ({
      ...prev,
      isLearning: false,
      learningTarget: null,
    }));
  }, []);

  /**
   * Set controller profile
   */
  const setProfile = useCallback((profileId: string) => {
    const profile = CONTROLLER_PROFILES.find(p => p.id === profileId);
    setState(prev => ({ ...prev, activeProfile: profile || null }));
  }, []);

  /**
   * Register action handler
   */
  const onAction = useCallback((action: string, handler: MIDIHandler) => {
    if (!handlersRef.current.has(action)) {
      handlersRef.current.set(action, new Set());
    }
    handlersRef.current.get(action)!.add(handler);

    return () => {
      handlersRef.current.get(action)?.delete(handler);
    };
  }, []);

  /**
   * Send MIDI message to output
   */
  const sendMessage = useCallback((
    outputId: string,
    type: MIDIMessageType,
    channel: number,
    data: { note?: number; velocity?: number; controller?: number; value?: number }
  ) => {
    if (!midiAccessRef.current) return;

    const output = midiAccessRef.current.outputs.get(outputId);
    if (!output) return;

    const message = createMIDIMessage(type, channel, data);
    if (message) {
      output.send(message);
    }
  }, []);

  /**
   * Set LED/button state on controller
   */
  const setLED = useCallback((outputId: string, note: number, color: number, channel: number = 1) => {
    sendMessage(outputId, 'noteon', channel, { note, velocity: color });
  }, [sendMessage]);

  /**
   * Clear a custom mapping
   */
  const clearMapping = useCallback((mappingId: string) => {
    setState(prev => ({
      ...prev,
      customMappings: prev.customMappings.filter(m => m.id !== mappingId),
    }));
  }, []);

  /**
   * Disable MIDI
   */
  const disable = useCallback(() => {
    if (midiAccessRef.current) {
      for (const input of midiAccessRef.current.inputs.values()) {
        input.onmidimessage = null;
      }
    }
    midiAccessRef.current = null;
    inputListenersRef.current.clear();
    setState(prev => ({
      ...prev,
      isEnabled: false,
      inputs: [],
      outputs: [],
    }));
  }, []);

  return {
    ...state,
    profiles: CONTROLLER_PROFILES,
    actions: MIDI_ACTIONS,
    enable,
    disable,
    setProfile,
    startLearn,
    cancelLearn,
    onAction,
    sendMessage,
    setLED,
    clearMapping,
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

function deviceToMIDIDevice(device: MIDIInput | MIDIOutput): MIDIDevice {
  return {
    id: device.id,
    name: device.name || 'Unknown Device',
    manufacturer: device.manufacturer || 'Unknown',
    type: device.type as 'input' | 'output',
    state: device.state as 'connected' | 'disconnected',
    connection: device.connection as 'open' | 'closed' | 'pending',
  };
}

function parseMIDIMessage(event: MIDIMessageEvent): MIDIMessage | null {
  const data = event.data;
  if (!data || data.length < 1) return null;

  const status = data[0];
  const channel = (status & 0x0f) + 1;
  const type = status & 0xf0;

  switch (type) {
    case 0x90: // Note On
      return {
        type: data[2] > 0 ? 'noteon' : 'noteoff',
        channel,
        note: data[1],
        velocity: data[2] / 127,
        timestamp: event.timeStamp,
      };

    case 0x80: // Note Off
      return {
        type: 'noteoff',
        channel,
        note: data[1],
        velocity: 0,
        timestamp: event.timeStamp,
      };

    case 0xb0: // Control Change
      return {
        type: 'controlchange',
        channel,
        controller: data[1],
        value: data[2] / 127,
        timestamp: event.timeStamp,
      };

    case 0xe0: // Pitch Bend
      const pitchBend = ((data[2] << 7) | data[1]) / 16383 * 2 - 1;
      return {
        type: 'pitchbend',
        channel,
        pitchBend,
        timestamp: event.timeStamp,
      };

    case 0xc0: // Program Change
      return {
        type: 'programchange',
        channel,
        value: data[1] / 127,
        timestamp: event.timeStamp,
      };

    default:
      return null;
  }
}

function createMIDIMessage(
  type: MIDIMessageType,
  channel: number,
  data: { note?: number; velocity?: number; controller?: number; value?: number }
): Uint8Array | null {
  const ch = (channel - 1) & 0x0f;

  switch (type) {
    case 'noteon':
      return new Uint8Array([0x90 | ch, data.note || 0, Math.round((data.velocity || 1) * 127)]);
    case 'noteoff':
      return new Uint8Array([0x80 | ch, data.note || 0, 0]);
    case 'controlchange':
      return new Uint8Array([0xb0 | ch, data.controller || 0, Math.round((data.value || 0) * 127)]);
    default:
      return null;
  }
}

function findMapping(
  message: MIDIMessage,
  profile: MIDIControllerProfile | null,
  customMappings: MIDIMapping[]
): MIDIMapping | null {
  const allMappings = [...customMappings, ...(profile?.mappings || [])];

  return allMappings.find(m => {
    if (m.channel !== message.channel) return false;

    if (m.type === 'note' && (message.type === 'noteon' || message.type === 'noteoff')) {
      return m.noteOrCC === message.note;
    }
    if (m.type === 'cc' && message.type === 'controlchange') {
      return m.noteOrCC === message.controller;
    }
    if (m.type === 'pitchbend' && message.type === 'pitchbend') {
      return true;
    }

    return false;
  }) || null;
}

function transformValue(message: MIDIMessage, mapping: MIDIMapping): number {
  let value = message.value ?? message.velocity ?? message.pitchBend ?? 0;

  if (mapping.invert) {
    value = 1 - value;
  }

  if (mapping.min !== undefined && mapping.max !== undefined) {
    value = mapping.min + value * (mapping.max - mapping.min);
  }

  return value;
}

export default useMIDI;
