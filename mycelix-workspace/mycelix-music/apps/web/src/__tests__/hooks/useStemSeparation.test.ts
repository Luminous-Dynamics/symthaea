// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tests for useStemSeparation hook
 */

import { renderHook, act } from '@testing-library/react';

// Mock AudioContext
const mockAudioContext = {
  createGain: jest.fn(() => ({
    connect: jest.fn(),
    gain: { value: 1, setValueAtTime: jest.fn() },
  })),
  createStereoPanner: jest.fn(() => ({
    connect: jest.fn(),
    pan: { value: 0, setValueAtTime: jest.fn() },
  })),
  createBufferSource: jest.fn(() => ({
    connect: jest.fn(),
    start: jest.fn(),
    stop: jest.fn(),
    buffer: null,
  })),
  createBuffer: jest.fn(() => ({
    numberOfChannels: 2,
    length: 44100,
    sampleRate: 44100,
    getChannelData: jest.fn(() => new Float32Array(44100)),
  })),
  createMediaStreamSource: jest.fn(() => ({
    connect: jest.fn(),
  })),
  currentTime: 0,
  destination: {},
  state: 'running',
  resume: jest.fn(),
};

// Mock global AudioContext
(global as any).AudioContext = jest.fn(() => mockAudioContext);

// Import after mocking
import { useStemSeparation } from '../../hooks/useStemSeparation';

describe('useStemSeparation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('initializes with empty stems', () => {
    const { result } = renderHook(() => useStemSeparation());

    expect(result.current.stems.size).toBe(0);
    expect(result.current.isProcessing).toBe(false);
    expect(result.current.isPlaying).toBe(false);
  });

  it('provides stem color and label constants', () => {
    const { result } = renderHook(() => useStemSeparation());

    expect(result.current.STEM_COLORS).toBeDefined();
    expect(result.current.STEM_COLORS.vocals).toBe('#ef4444');
    expect(result.current.STEM_COLORS.drums).toBe('#f59e0b');
    expect(result.current.STEM_COLORS.bass).toBe('#8b5cf6');

    expect(result.current.STEM_LABELS).toBeDefined();
    expect(result.current.STEM_LABELS.vocals).toBe('Vocals');
  });

  it('provides control methods', () => {
    const { result } = renderHook(() => useStemSeparation());

    expect(typeof result.current.controls.setLevel).toBe('function');
    expect(typeof result.current.controls.setMuted).toBe('function');
    expect(typeof result.current.controls.setSolo).toBe('function');
    expect(typeof result.current.controls.setPan).toBe('function');
    expect(typeof result.current.controls.resetAll).toBe('function');
  });

  it('provides playback methods', () => {
    const { result } = renderHook(() => useStemSeparation());

    expect(typeof result.current.play).toBe('function');
    expect(typeof result.current.pause).toBe('function');
    expect(typeof result.current.stop).toBe('function');
    expect(typeof result.current.seek).toBe('function');
  });

  it('provides export methods', () => {
    const { result } = renderHook(() => useStemSeparation());

    expect(typeof result.current.exportStem).toBe('function');
    expect(typeof result.current.exportMix).toBe('function');
  });

  it('tracks progress during separation', () => {
    const { result } = renderHook(() => useStemSeparation());

    expect(result.current.progress.stage).toBe('loading');
    expect(result.current.progress.progress).toBe(0);
  });
});
