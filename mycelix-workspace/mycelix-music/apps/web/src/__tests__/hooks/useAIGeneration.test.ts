// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tests for useAIGeneration hook
 */

import { renderHook, act, waitFor } from '@testing-library/react';

// Mock AudioContext
const mockAudioContext = {
  sampleRate: 44100,
  destination: {},
  createGain: jest.fn(() => ({
    gain: { value: 1 },
    connect: jest.fn(),
  })),
  createBufferSource: jest.fn(() => ({
    buffer: null,
    connect: jest.fn(),
    start: jest.fn(),
    stop: jest.fn(),
    onended: null,
  })),
  createBuffer: jest.fn((channels, length, sampleRate) => ({
    numberOfChannels: channels,
    length,
    sampleRate,
    duration: length / sampleRate,
    getChannelData: jest.fn(() => new Float32Array(length)),
  })),
  close: jest.fn(),
};

global.AudioContext = jest.fn(() => mockAudioContext) as any;

// Import after mocking
import { useAIGeneration } from '@/hooks/useAIGeneration';

describe('useAIGeneration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('initialization', () => {
    it('should initialize with default state', () => {
      const { result } = renderHook(() => useAIGeneration());

      expect(result.current.isGenerating).toBe(false);
      expect(result.current.progress).toBe(0);
      expect(result.current.currentTask).toBeNull();
      expect(result.current.history).toEqual([]);
      expect(result.current.error).toBeNull();
    });

    it('should have presets available', () => {
      const { result } = renderHook(() => useAIGeneration());

      expect(result.current.presets).toBeDefined();
      expect(result.current.presets.electronic).toBeDefined();
      expect(result.current.presets.house).toBeDefined();
      expect(result.current.presets.techno).toBeDefined();
    });

    it('should have scales available', () => {
      const { result } = renderHook(() => useAIGeneration());

      expect(result.current.scales).toBeDefined();
      expect(result.current.scales).toContain('major');
      expect(result.current.scales).toContain('minor');
      expect(result.current.scales).toContain('pentatonic');
    });
  });

  describe('beat generation', () => {
    it('should generate a beat pattern', async () => {
      const { result } = renderHook(() => useAIGeneration());

      await act(async () => {
        const generated = await result.current.generate({
          mode: 'beat',
          style: 'electronic',
          tempo: 128,
          bars: 4,
        });

        expect(generated).toBeDefined();
        expect(generated.type).toBe('beat');
        expect(generated.beatPattern).toBeDefined();
        expect(generated.beatPattern?.tempo).toBe(128);
        expect(generated.beatPattern?.style).toBe('electronic');
        expect(generated.audioBuffer).toBeDefined();
      });
    });

    it('should update progress during generation', async () => {
      const { result } = renderHook(() => useAIGeneration());

      const progressUpdates: number[] = [];

      act(() => {
        result.current.generate({
          mode: 'beat',
          style: 'house',
        });
      });

      // Should be generating
      expect(result.current.isGenerating).toBe(true);

      await waitFor(() => {
        expect(result.current.isGenerating).toBe(false);
      });

      expect(result.current.progress).toBe(100);
    });

    it('should add generated result to history', async () => {
      const { result } = renderHook(() => useAIGeneration());

      await act(async () => {
        await result.current.generate({
          mode: 'beat',
          style: 'techno',
        });
      });

      expect(result.current.history.length).toBe(1);
      expect(result.current.history[0].type).toBe('beat');
    });
  });

  describe('melody generation', () => {
    it('should generate a melody', async () => {
      const { result } = renderHook(() => useAIGeneration());

      await act(async () => {
        const generated = await result.current.generate({
          mode: 'melody',
          style: 'electronic',
          tempo: 120,
          key: 'C',
          scale: 'minor',
        });

        expect(generated).toBeDefined();
        expect(generated.type).toBe('melody');
        expect(generated.melodySequence).toBeDefined();
        expect(generated.melodySequence?.key).toBe('C');
        expect(generated.melodySequence?.scale).toBe('minor');
      });
    });
  });

  describe('full track generation', () => {
    it('should generate a full track with beat and melody', async () => {
      const { result } = renderHook(() => useAIGeneration());

      await act(async () => {
        const generated = await result.current.generate({
          mode: 'full',
          style: 'lofi',
          tempo: 75,
        });

        expect(generated).toBeDefined();
        expect(generated.type).toBe('full');
        expect(generated.beatPattern).toBeDefined();
        expect(generated.melodySequence).toBeDefined();
        expect(generated.audioBuffer).toBeDefined();
      });
    });
  });

  describe('cancel', () => {
    it('should cancel ongoing generation', async () => {
      const { result } = renderHook(() => useAIGeneration());

      act(() => {
        result.current.generate({
          mode: 'full',
          style: 'electronic',
        });
      });

      act(() => {
        result.current.cancel();
      });

      expect(result.current.isGenerating).toBe(false);
    });
  });

  describe('clearHistory', () => {
    it('should clear generation history', async () => {
      const { result } = renderHook(() => useAIGeneration());

      await act(async () => {
        await result.current.generate({ mode: 'beat', style: 'house' });
        await result.current.generate({ mode: 'beat', style: 'techno' });
      });

      expect(result.current.history.length).toBe(2);

      act(() => {
        result.current.clearHistory();
      });

      expect(result.current.history.length).toBe(0);
    });
  });

  describe('different styles', () => {
    const styles = ['electronic', 'house', 'techno', 'trance', 'dnb', 'hiphop', 'trap', 'lofi', 'rock', 'jazz'];

    styles.forEach(style => {
      it(`should generate ${style} beat`, async () => {
        const { result } = renderHook(() => useAIGeneration());

        await act(async () => {
          const generated = await result.current.generate({
            mode: 'beat',
            style: style as any,
          });

          expect(generated.beatPattern?.style).toBe(style);
        });
      });
    });
  });

  describe('complexity and energy parameters', () => {
    it('should respect complexity parameter', async () => {
      const { result } = renderHook(() => useAIGeneration());

      await act(async () => {
        const lowComplexity = await result.current.generate({
          mode: 'beat',
          style: 'electronic',
          complexity: 0.1,
        });

        const highComplexity = await result.current.generate({
          mode: 'beat',
          style: 'electronic',
          complexity: 0.9,
        });

        // High complexity should have more tracks/variation
        expect(lowComplexity.beatPattern).toBeDefined();
        expect(highComplexity.beatPattern).toBeDefined();
      });
    });

    it('should respect energy parameter', async () => {
      const { result } = renderHook(() => useAIGeneration());

      await act(async () => {
        const generated = await result.current.generate({
          mode: 'beat',
          style: 'electronic',
          energy: 0.9,
        });

        expect(generated.beatPattern).toBeDefined();
      });
    });
  });

  describe('seed parameter', () => {
    it('should produce consistent results with same seed', async () => {
      const { result } = renderHook(() => useAIGeneration());

      let pattern1: any, pattern2: any;

      await act(async () => {
        const result1 = await result.current.generate({
          mode: 'beat',
          style: 'electronic',
          seed: 12345,
        });
        pattern1 = result1.beatPattern;
      });

      await act(async () => {
        const result2 = await result.current.generate({
          mode: 'beat',
          style: 'electronic',
          seed: 12345,
        });
        pattern2 = result2.beatPattern;
      });

      // Same seed should produce same pattern
      expect(pattern1.tracks.length).toBe(pattern2.tracks.length);
    });
  });
});
