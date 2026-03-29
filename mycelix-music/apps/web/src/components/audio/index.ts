// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Audio Components Index
 *
 * Professional audio UI components for music production.
 */

// Audio Pipeline
export { AudioPipelineProvider, useAudioPipeline, AudioLevelMeter } from './AudioPipeline';

// Waveform visualization
export { WaveformDisplay } from './WaveformDisplay';
export type {
  WaveformData,
  Region,
  Marker,
  DisplayMode,
  WaveformDisplayProps,
  WaveformDisplayRef,
} from './WaveformDisplay';

// Level metering
export { VUMeter } from './VUMeter';
export type { VUMeterProps } from './VUMeter';

// Rotary controls
export { Knob } from './Knob';
export type { KnobProps } from './Knob';

// Linear controls
export { Fader } from './Fader';
export type { FaderProps } from './Fader';

// Mixer components
export { MixerChannel, MixerConsole } from './MixerChannel';
export type {
  ChannelState,
  SendState,
  InsertSlot,
  MixerChannelProps,
  MixerConsoleProps,
} from './MixerChannel';

// MIDI editing
export { PianoRoll } from './PianoRoll';
export type {
  MIDINote,
  PianoRollProps,
  PianoRollRef,
} from './PianoRoll';

// Arrangement
export { Timeline } from './Timeline';
export type {
  TimelineClip,
  TimelineTrack,
  TimelineMarker,
  LoopRegion,
  TimelineProps,
  TimelineRef,
} from './Timeline';
