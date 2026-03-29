// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Audio Worker Hook
 *
 * Provides access to the audio processing Web Worker
 * for offloading heavy computations from the main thread.
 */

import { useState, useCallback, useRef, useEffect } from 'react';

interface WorkerMessage {
  id: string;
  type: string;
  data: any;
}

interface PendingRequest {
  resolve: (data: any) => void;
  reject: (error: Error) => void;
}

export function useAudioWorker() {
  const [isReady, setIsReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const workerRef = useRef<Worker | null>(null);
  const pendingRef = useRef<Map<string, PendingRequest>>(new Map());
  const messageIdRef = useRef(0);

  // Initialize worker
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const worker = new Worker(
      new URL('../workers/audio.worker.ts', import.meta.url),
      { type: 'module' }
    );

    worker.onmessage = (event) => {
      const { id, type, success, data, error } = event.data;

      if (type === 'ready') {
        setIsReady(true);
        return;
      }

      const pending = pendingRef.current.get(id);
      if (pending) {
        pendingRef.current.delete(id);
        setIsProcessing(pendingRef.current.size > 0);

        if (success) {
          pending.resolve(data);
        } else {
          pending.reject(new Error(error || 'Worker error'));
        }
      }
    };

    worker.onerror = (error) => {
      console.error('Worker error:', error);
      // Reject all pending requests
      pendingRef.current.forEach((pending) => {
        pending.reject(new Error('Worker crashed'));
      });
      pendingRef.current.clear();
      setIsProcessing(false);
    };

    workerRef.current = worker;

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  // Send message to worker
  const sendMessage = useCallback(<T>(type: string, data: any): Promise<T> => {
    return new Promise((resolve, reject) => {
      if (!workerRef.current) {
        reject(new Error('Worker not initialized'));
        return;
      }

      const id = `msg-${++messageIdRef.current}`;

      pendingRef.current.set(id, { resolve, reject });
      setIsProcessing(true);

      const message: WorkerMessage = { id, type, data };
      workerRef.current.postMessage(message);
    });
  }, []);

  // Compute spectrum
  const computeSpectrum = useCallback(
    async (audioData: Float32Array, fftSize: number = 2048): Promise<Float32Array> => {
      const result = await sendMessage<{ spectrum: Float32Array }>('computeSpectrum', {
        audioData,
        fftSize,
      });
      return result.spectrum;
    },
    [sendMessage]
  );

  // Generate waveform
  const generateWaveform = useCallback(
    async (audioData: Float32Array, targetLength: number = 1000): Promise<Float32Array> => {
      const result = await sendMessage<{ waveform: Float32Array }>('generateWaveform', {
        audioData,
        targetLength,
      });
      return result.waveform;
    },
    [sendMessage]
  );

  // Analyze audio
  const analyzeAudio = useCallback(
    async (
      audioData: Float32Array,
      sampleRate: number
    ): Promise<{
      rms: number;
      peak: number;
      rmsDb: number;
      peakDb: number;
      spectrum: Float32Array;
      beats: number[];
      bpm: number;
    }> => {
      return sendMessage('analyzeAudio', { audioData, sampleRate });
    },
    [sendMessage]
  );

  // Apply gain
  const applyGain = useCallback(
    async (audioData: Float32Array, gain: number): Promise<Float32Array> => {
      const result = await sendMessage<{ audioData: Float32Array }>('applyGain', {
        audioData,
        gain,
      });
      return result.audioData;
    },
    [sendMessage]
  );

  // Mix audio
  const mixAudio = useCallback(
    async (buffers: Float32Array[], gains: number[]): Promise<Float32Array> => {
      const result = await sendMessage<{ audioData: Float32Array }>('mixAudio', {
        buffers,
        gains,
      });
      return result.audioData;
    },
    [sendMessage]
  );

  // Low-pass filter
  const lowPassFilter = useCallback(
    async (audioData: Float32Array, cutoff: number, sampleRate: number): Promise<Float32Array> => {
      const result = await sendMessage<{ audioData: Float32Array }>('lowPassFilter', {
        audioData,
        cutoff,
        sampleRate,
      });
      return result.audioData;
    },
    [sendMessage]
  );

  // High-pass filter
  const highPassFilter = useCallback(
    async (audioData: Float32Array, cutoff: number, sampleRate: number): Promise<Float32Array> => {
      const result = await sendMessage<{ audioData: Float32Array }>('highPassFilter', {
        audioData,
        cutoff,
        sampleRate,
      });
      return result.audioData;
    },
    [sendMessage]
  );

  // Compress
  const compress = useCallback(
    async (
      audioData: Float32Array,
      threshold: number,
      ratio: number,
      attack: number,
      release: number,
      sampleRate: number
    ): Promise<Float32Array> => {
      const result = await sendMessage<{ audioData: Float32Array }>('compress', {
        audioData,
        threshold,
        ratio,
        attack,
        release,
        sampleRate,
      });
      return result.audioData;
    },
    [sendMessage]
  );

  // Limit
  const limit = useCallback(
    async (audioData: Float32Array, ceiling: number): Promise<Float32Array> => {
      const result = await sendMessage<{ audioData: Float32Array }>('limit', {
        audioData,
        ceiling,
      });
      return result.audioData;
    },
    [sendMessage]
  );

  // Normalize
  const normalize = useCallback(
    async (audioData: Float32Array, targetPeak: number = 1): Promise<Float32Array> => {
      const result = await sendMessage<{ audioData: Float32Array }>('normalize', {
        audioData,
        targetPeak,
      });
      return result.audioData;
    },
    [sendMessage]
  );

  // Resample
  const resample = useCallback(
    async (
      audioData: Float32Array,
      sourceSampleRate: number,
      targetSampleRate: number
    ): Promise<Float32Array> => {
      const result = await sendMessage<{ audioData: Float32Array }>('resample', {
        audioData,
        sourceSampleRate,
        targetSampleRate,
      });
      return result.audioData;
    },
    [sendMessage]
  );

  // Detect beats
  const detectBeats = useCallback(
    async (audioData: Float32Array, sampleRate: number): Promise<{ beats: number[]; bpm: number }> => {
      return sendMessage('detectBeats', { audioData, sampleRate });
    },
    [sendMessage]
  );

  return {
    isReady,
    isProcessing,
    computeSpectrum,
    generateWaveform,
    analyzeAudio,
    applyGain,
    mixAudio,
    lowPassFilter,
    highPassFilter,
    compress,
    limit,
    normalize,
    resample,
    detectBeats,
  };
}

export default useAudioWorker;
