// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tests for useVoiceCommands hook
 */

import { renderHook, act } from '@testing-library/react';

// Mock SpeechRecognition
const mockSpeechRecognition = {
  start: jest.fn(),
  stop: jest.fn(),
  continuous: false,
  interimResults: false,
  lang: 'en-US',
  maxAlternatives: 1,
  onresult: null as any,
  onerror: null as any,
  onend: null as any,
};

(global as any).SpeechRecognition = jest.fn(() => mockSpeechRecognition);
(global as any).webkitSpeechRecognition = jest.fn(() => mockSpeechRecognition);

// Mock speechSynthesis
(global as any).speechSynthesis = {
  speak: jest.fn(),
  cancel: jest.fn(),
  getVoices: jest.fn(() => []),
};

(global as any).SpeechSynthesisUtterance = jest.fn();

// Import after mocking
import { useVoiceCommands } from '../../hooks/useVoiceCommands';

describe('useVoiceCommands', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('detects browser support', () => {
    const { result } = renderHook(() => useVoiceCommands());

    expect(result.current.isSupported).toBe(true);
  });

  it('initializes in non-listening state', () => {
    const { result } = renderHook(() => useVoiceCommands());

    expect(result.current.isListening).toBe(false);
    expect(result.current.isAwake).toBe(false);
  });

  it('provides default commands', () => {
    const { result } = renderHook(() => useVoiceCommands());

    const commands = result.current.getAllCommands();
    expect(commands.length).toBeGreaterThan(0);

    // Check for playback commands
    const playCommand = commands.find(c => c.action === 'play');
    expect(playCommand).toBeDefined();
    expect(playCommand?.patterns).toContain('play');
  });

  it('provides commands by category', () => {
    const { result } = renderHook(() => useVoiceCommands());

    const playbackCommands = result.current.getCommandsByCategory('playback');
    expect(playbackCommands.length).toBeGreaterThan(0);

    const volumeCommands = result.current.getCommandsByCategory('volume');
    expect(volumeCommands.length).toBeGreaterThan(0);
  });

  it('can register custom handlers', () => {
    const { result } = renderHook(() => useVoiceCommands());

    const mockHandler = jest.fn();
    const unregister = result.current.registerHandler('play', mockHandler);

    expect(typeof unregister).toBe('function');
  });

  it('supports wake word configuration', () => {
    const { result } = renderHook(() => useVoiceCommands());

    expect(result.current.wakeWord).toBe('hey mycelix');

    act(() => {
      result.current.setWakeWord('hey dj');
    });

    expect(result.current.wakeWord).toBe('hey dj');
  });

  it('can start and stop listening', () => {
    const { result } = renderHook(() => useVoiceCommands());

    act(() => {
      result.current.startListening();
    });

    expect(result.current.isListening).toBe(true);
    expect(mockSpeechRecognition.start).toHaveBeenCalled();

    act(() => {
      result.current.stopListening();
    });

    expect(result.current.isListening).toBe(false);
    expect(mockSpeechRecognition.stop).toHaveBeenCalled();
  });

  it('provides text-to-speech methods', () => {
    const { result } = renderHook(() => useVoiceCommands());

    expect(typeof result.current.speak).toBe('function');
    expect(typeof result.current.cancelSpeak).toBe('function');
  });
});
