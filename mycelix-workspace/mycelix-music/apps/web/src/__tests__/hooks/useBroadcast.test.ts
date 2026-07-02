// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tests for useBroadcast hook
 */

import { renderHook, act } from '@testing-library/react';

// Mock WebSocket
class MockWebSocket {
  onopen: (() => void) | null = null;
  onmessage: ((event: any) => void) | null = null;
  onerror: ((event: any) => void) | null = null;
  onclose: (() => void) | null = null;
  readyState = 1;

  send = jest.fn();
  close = jest.fn();
}

(global as any).WebSocket = MockWebSocket;

// Mock RTCPeerConnection
class MockRTCPeerConnection {
  onicecandidate: ((event: any) => void) | null = null;
  ontrack: ((event: any) => void) | null = null;

  addTrack = jest.fn();
  createOffer = jest.fn(() => Promise.resolve({ type: 'offer', sdp: 'mock-sdp' }));
  createAnswer = jest.fn(() => Promise.resolve({ type: 'answer', sdp: 'mock-sdp' }));
  setLocalDescription = jest.fn(() => Promise.resolve());
  setRemoteDescription = jest.fn(() => Promise.resolve());
  addIceCandidate = jest.fn(() => Promise.resolve());
  close = jest.fn();
}

(global as any).RTCPeerConnection = MockRTCPeerConnection;

// Mock MediaRecorder
class MockMediaRecorder {
  ondataavailable: ((event: any) => void) | null = null;
  state = 'inactive';

  start = jest.fn(() => { this.state = 'recording'; });
  stop = jest.fn(() => { this.state = 'inactive'; });
}

(global as any).MediaRecorder = MockMediaRecorder;

// Mock AudioContext
const mockAudioContext = {
  createMediaStreamSource: jest.fn(() => ({ connect: jest.fn() })),
  createMediaStreamDestination: jest.fn(() => ({ stream: new MediaStream() })),
  createAnalyser: jest.fn(() => ({
    connect: jest.fn(),
    fftSize: 256,
    frequencyBinCount: 128,
    getByteFrequencyData: jest.fn(),
  })),
  createGain: jest.fn(() => ({
    connect: jest.fn(),
    gain: { value: 1 },
  })),
  close: jest.fn(),
};

(global as any).AudioContext = jest.fn(() => mockAudioContext);

// Mock MediaStream
class MockMediaStream {
  getTracks = jest.fn(() => [{ stop: jest.fn() }]);
  getAudioTracks = jest.fn(() => [{ stop: jest.fn() }]);
}

(global as any).MediaStream = MockMediaStream;

// Import after mocking
import { useBroadcast, useBroadcastListener } from '../../hooks/useBroadcast';

describe('useBroadcast', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('initializes in idle state', () => {
    const { result } = renderHook(() => useBroadcast());

    expect(result.current.state.status).toBe('idle');
    expect(result.current.state.streamId).toBeNull();
    expect(result.current.isLive).toBe(false);
  });

  it('has empty listeners initially', () => {
    const { result } = renderHook(() => useBroadcast());

    expect(result.current.listeners).toEqual([]);
  });

  it('initializes stats correctly', () => {
    const { result } = renderHook(() => useBroadcast());

    expect(result.current.stats.listeners).toBe(0);
    expect(result.current.stats.peakListeners).toBe(0);
    expect(result.current.stats.duration).toBe(0);
    expect(result.current.stats.bytesTransferred).toBe(0);
  });

  it('provides broadcast control methods', () => {
    const { result } = renderHook(() => useBroadcast());

    expect(typeof result.current.startBroadcast).toBe('function');
    expect(typeof result.current.stopBroadcast).toBe('function');
    expect(typeof result.current.sendMessage).toBe('function');
    expect(typeof result.current.sendReaction).toBe('function');
    expect(typeof result.current.kickListener).toBe('function');
    expect(typeof result.current.updateConfig).toBe('function');
  });

  it('generates share URL when live', async () => {
    const { result } = renderHook(() => useBroadcast());

    // Initially no share URL
    expect(result.current.getShareUrl()).toBeNull();
  });
});

describe('useBroadcastListener', () => {
  it('initializes in disconnected state', () => {
    const { result } = renderHook(() => useBroadcastListener('test-stream-id'));

    expect(result.current.isConnected).toBe(false);
    expect(result.current.isPlaying).toBe(false);
  });

  it('provides connection methods', () => {
    const { result } = renderHook(() => useBroadcastListener('test-stream-id'));

    expect(typeof result.current.connect).toBe('function');
    expect(typeof result.current.disconnect).toBe('function');
    expect(typeof result.current.sendMessage).toBe('function');
    expect(typeof result.current.sendReaction).toBe('function');
  });

  it('has empty messages initially', () => {
    const { result } = renderHook(() => useBroadcastListener('test-stream-id'));

    expect(result.current.messages).toEqual([]);
  });
});
