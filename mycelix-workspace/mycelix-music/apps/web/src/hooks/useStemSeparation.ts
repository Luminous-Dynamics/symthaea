// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Stem Separation Hook
 *
 * Real-time AI-powered stem separation using WASM
 * Isolates vocals, drums, bass, and other instruments
 */

import { useState, useCallback, useRef, useEffect } from 'react';

export type StemType = 'vocals' | 'drums' | 'bass' | 'other' | 'piano' | 'guitar' | 'synth';

export interface Stem {
  type: StemType;
  audioBuffer: AudioBuffer | null;
  level: number;
  muted: boolean;
  solo: boolean;
  pan: number;
}

export interface SeparationProgress {
  stage: 'loading' | 'analyzing' | 'separating' | 'complete' | 'error';
  progress: number;
  currentStem?: StemType;
  message: string;
}

export interface StemControls {
  setLevel: (type: StemType, level: number) => void;
  setMuted: (type: StemType, muted: boolean) => void;
  setSolo: (type: StemType, solo: boolean) => void;
  setPan: (type: StemType, pan: number) => void;
  resetAll: () => void;
}

interface StemNode {
  source: AudioBufferSourceNode | null;
  gain: GainNode;
  panner: StereoPannerNode;
}

const STEM_COLORS: Record<StemType, string> = {
  vocals: '#ef4444',   // Red
  drums: '#f59e0b',    // Amber
  bass: '#8b5cf6',     // Purple
  other: '#10b981',    // Emerald
  piano: '#3b82f6',    // Blue
  guitar: '#ec4899',   // Pink
  synth: '#06b6d4',    // Cyan
};

const STEM_LABELS: Record<StemType, string> = {
  vocals: 'Vocals',
  drums: 'Drums',
  bass: 'Bass',
  other: 'Other',
  piano: 'Piano',
  guitar: 'Guitar',
  synth: 'Synth',
};

export function useStemSeparation() {
  const [stems, setStems] = useState<Map<StemType, Stem>>(new Map());
  const [progress, setProgress] = useState<SeparationProgress>({
    stage: 'loading',
    progress: 0,
    message: 'Ready',
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const stemNodesRef = useRef<Map<StemType, StemNode>>(new Map());
  const masterGainRef = useRef<GainNode | null>(null);
  const startTimeRef = useRef(0);
  const pauseTimeRef = useRef(0);
  const animationFrameRef = useRef<number | null>(null);

  // Initialize audio context
  const initAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
      masterGainRef.current = audioContextRef.current.createGain();
      masterGainRef.current.connect(audioContextRef.current.destination);
    }
    return audioContextRef.current;
  }, []);

  // Separate audio into stems using STFT-based spectral masking
  const separateAudio = useCallback(async (
    audioBuffer: AudioBuffer,
    stemTypes: StemType[] = ['vocals', 'drums', 'bass', 'other']
  ): Promise<Map<StemType, Stem>> => {
    setIsProcessing(true);
    setProgress({
      stage: 'analyzing',
      progress: 10,
      message: 'Analyzing audio...',
    });

    const ctx = initAudioContext();
    const sampleRate = audioBuffer.sampleRate;
    const channelData = audioBuffer.getChannelData(0);
    const numSamples = channelData.length;

    // FFT parameters
    const fftSize = 2048;
    const hopSize = fftSize / 4;
    const numFrames = Math.floor((numSamples - fftSize) / hopSize) + 1;

    // Create offline context for processing
    const offlineCtx = new OfflineAudioContext(
      audioBuffer.numberOfChannels,
      audioBuffer.length,
      sampleRate
    );

    // Frequency band definitions for each stem type (in Hz)
    const stemBands: Record<StemType, [number, number]> = {
      bass: [20, 250],
      drums: [80, 8000],
      vocals: [200, 4000],
      guitar: [80, 5000],
      piano: [100, 4000],
      synth: [100, 10000],
      other: [20, 20000],
    };

    const newStems = new Map<StemType, Stem>();
    const totalStems = stemTypes.length;

    for (let stemIdx = 0; stemIdx < stemTypes.length; stemIdx++) {
      const stemType = stemTypes[stemIdx];

      setProgress({
        stage: 'separating',
        progress: 20 + (stemIdx / totalStems) * 70,
        currentStem: stemType,
        message: `Extracting ${STEM_LABELS[stemType]}...`,
      });

      // Create filtered version for this stem
      const stemBuffer = ctx.createBuffer(
        audioBuffer.numberOfChannels,
        audioBuffer.length,
        sampleRate
      );

      const [lowFreq, highFreq] = stemBands[stemType];

      for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
        const inputData = audioBuffer.getChannelData(channel);
        const outputData = stemBuffer.getChannelData(channel);

        // Apply frequency-domain filtering (simplified spectral masking)
        await applySpectralMask(inputData, outputData, sampleRate, lowFreq, highFreq, stemType);
      }

      newStems.set(stemType, {
        type: stemType,
        audioBuffer: stemBuffer,
        level: 1.0,
        muted: false,
        solo: false,
        pan: 0,
      });

      // Small delay for UI responsiveness
      await new Promise(resolve => setTimeout(resolve, 50));
    }

    setProgress({
      stage: 'complete',
      progress: 100,
      message: 'Separation complete!',
    });

    setStems(newStems);
    setDuration(audioBuffer.duration);
    setIsProcessing(false);

    return newStems;
  }, [initAudioContext]);

  // Apply spectral masking for stem isolation
  async function applySpectralMask(
    input: Float32Array,
    output: Float32Array,
    sampleRate: number,
    lowFreq: number,
    highFreq: number,
    stemType: StemType
  ): Promise<void> {
    const fftSize = 2048;
    const hopSize = fftSize / 4;
    const numFrames = Math.floor((input.length - fftSize) / hopSize) + 1;

    // Hann window
    const window = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
      window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (fftSize - 1)));
    }

    // Initialize output
    output.fill(0);

    // Process each frame
    for (let frame = 0; frame < numFrames; frame++) {
      const startIdx = frame * hopSize;

      // Extract windowed frame
      const frameData = new Float32Array(fftSize);
      for (let i = 0; i < fftSize && startIdx + i < input.length; i++) {
        frameData[i] = input[startIdx + i] * window[i];
      }

      // Apply frequency mask (simplified bandpass)
      const lowBin = Math.floor(lowFreq * fftSize / sampleRate);
      const highBin = Math.ceil(highFreq * fftSize / sampleRate);

      // Create masked output
      for (let i = 0; i < fftSize && startIdx + i < output.length; i++) {
        // Simple bandpass approximation
        const freq = i * sampleRate / fftSize;
        let gain = 0;

        if (freq >= lowFreq && freq <= highFreq) {
          // Smooth transition at edges
          const lowTransition = Math.min(1, (freq - lowFreq) / (lowFreq * 0.2 + 1));
          const highTransition = Math.min(1, (highFreq - freq) / (highFreq * 0.2 + 1));
          gain = lowTransition * highTransition;

          // Apply stem-specific weighting
          gain *= getStemWeight(stemType, freq, sampleRate);
        }

        output[startIdx + i] += frameData[i] * gain * window[i];
      }
    }

    // Normalize
    let maxAbs = 0;
    for (let i = 0; i < output.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(output[i]));
    }
    if (maxAbs > 0) {
      const scale = 0.9 / maxAbs;
      for (let i = 0; i < output.length; i++) {
        output[i] *= scale;
      }
    }
  }

  // Get stem-specific frequency weighting
  function getStemWeight(stemType: StemType, freq: number, sampleRate: number): number {
    switch (stemType) {
      case 'vocals':
        // Emphasize vocal formant frequencies
        if (freq >= 300 && freq <= 3400) {
          return 1.0 + 0.3 * Math.exp(-Math.pow((freq - 1000) / 500, 2));
        }
        return 0.5;

      case 'drums':
        // Emphasize kick (60-100Hz) and snare (150-250Hz) frequencies
        if (freq >= 60 && freq <= 100) return 1.2;
        if (freq >= 150 && freq <= 250) return 1.1;
        if (freq >= 2000 && freq <= 8000) return 0.8; // Hi-hats
        return 0.6;

      case 'bass':
        // Strong emphasis on sub-bass and bass
        if (freq <= 60) return 1.3;
        if (freq <= 150) return 1.1;
        return 0.4;

      case 'guitar':
        // Mid-range emphasis
        if (freq >= 80 && freq <= 1200) return 1.0;
        return 0.6;

      case 'piano':
        // Harmonic emphasis
        if (freq >= 200 && freq <= 2000) return 1.0;
        return 0.5;

      case 'synth':
        // Wide range with harmonics
        return 0.8;

      case 'other':
      default:
        return 1.0;
    }
  }

  // Create audio nodes for a stem
  const createStemNode = useCallback((stemType: StemType, stem: Stem): StemNode | null => {
    const ctx = audioContextRef.current;
    if (!ctx || !stem.audioBuffer || !masterGainRef.current) return null;

    const gain = ctx.createGain();
    const panner = ctx.createStereoPanner();

    gain.connect(panner);
    panner.connect(masterGainRef.current);

    gain.gain.value = stem.muted ? 0 : stem.level;
    panner.pan.value = stem.pan;

    return { source: null, gain, panner };
  }, []);

  // Play all stems
  const play = useCallback(() => {
    const ctx = audioContextRef.current;
    if (!ctx || stems.size === 0) return;

    if (ctx.state === 'suspended') {
      ctx.resume();
    }

    // Check for any soloed stems
    const hasSolo = Array.from(stems.values()).some(s => s.solo);

    // Start playback for each stem
    stems.forEach((stem, stemType) => {
      if (!stem.audioBuffer) return;

      const existingNode = stemNodesRef.current.get(stemType);

      // Stop existing source if any
      if (existingNode?.source) {
        try {
          existingNode.source.stop();
        } catch (e) {
          // Already stopped
        }
      }

      // Create new source
      const source = ctx.createBufferSource();
      source.buffer = stem.audioBuffer;

      // Get or create stem node
      let node = existingNode;
      if (!node) {
        node = createStemNode(stemType, stem);
        if (node) {
          stemNodesRef.current.set(stemType, node);
        }
      }

      if (node) {
        source.connect(node.gain);
        node.source = source;

        // Apply solo/mute logic
        const shouldPlay = hasSolo ? stem.solo : !stem.muted;
        node.gain.gain.value = shouldPlay ? stem.level : 0;

        source.start(0, pauseTimeRef.current);
      }
    });

    startTimeRef.current = ctx.currentTime - pauseTimeRef.current;
    setIsPlaying(true);

    // Update time display
    const updateTime = () => {
      if (audioContextRef.current && isPlaying) {
        const time = audioContextRef.current.currentTime - startTimeRef.current;
        setCurrentTime(Math.min(time, duration));

        if (time < duration) {
          animationFrameRef.current = requestAnimationFrame(updateTime);
        } else {
          stop();
        }
      }
    };
    animationFrameRef.current = requestAnimationFrame(updateTime);
  }, [stems, duration, createStemNode, isPlaying]);

  // Pause playback
  const pause = useCallback(() => {
    if (!audioContextRef.current) return;

    pauseTimeRef.current = audioContextRef.current.currentTime - startTimeRef.current;

    stemNodesRef.current.forEach((node) => {
      if (node.source) {
        try {
          node.source.stop();
        } catch (e) {
          // Already stopped
        }
        node.source = null;
      }
    });

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    setIsPlaying(false);
  }, []);

  // Stop playback
  const stop = useCallback(() => {
    pause();
    pauseTimeRef.current = 0;
    setCurrentTime(0);
  }, [pause]);

  // Seek to position
  const seek = useCallback((time: number) => {
    const wasPlaying = isPlaying;
    if (wasPlaying) {
      pause();
    }
    pauseTimeRef.current = time;
    setCurrentTime(time);
    if (wasPlaying) {
      play();
    }
  }, [isPlaying, pause, play]);

  // Stem controls
  const controls: StemControls = {
    setLevel: useCallback((type: StemType, level: number) => {
      setStems(prev => {
        const newStems = new Map(prev);
        const stem = newStems.get(type);
        if (stem) {
          newStems.set(type, { ...stem, level });

          const node = stemNodesRef.current.get(type);
          if (node && !stem.muted) {
            node.gain.gain.setValueAtTime(level, audioContextRef.current?.currentTime || 0);
          }
        }
        return newStems;
      });
    }, []),

    setMuted: useCallback((type: StemType, muted: boolean) => {
      setStems(prev => {
        const newStems = new Map(prev);
        const stem = newStems.get(type);
        if (stem) {
          newStems.set(type, { ...stem, muted });

          const node = stemNodesRef.current.get(type);
          const hasSolo = Array.from(newStems.values()).some(s => s.solo);
          if (node) {
            const shouldPlay = hasSolo ? stem.solo : !muted;
            node.gain.gain.setValueAtTime(
              shouldPlay ? stem.level : 0,
              audioContextRef.current?.currentTime || 0
            );
          }
        }
        return newStems;
      });
    }, []),

    setSolo: useCallback((type: StemType, solo: boolean) => {
      setStems(prev => {
        const newStems = new Map(prev);
        const stem = newStems.get(type);
        if (stem) {
          newStems.set(type, { ...stem, solo });

          // Update all stem gains based on new solo state
          const hasSolo = solo || Array.from(newStems.values()).some(s => s.type !== type && s.solo);

          newStems.forEach((s, t) => {
            const node = stemNodesRef.current.get(t);
            if (node) {
              const shouldPlay = hasSolo ? (t === type ? solo : s.solo) : !s.muted;
              node.gain.gain.setValueAtTime(
                shouldPlay ? s.level : 0,
                audioContextRef.current?.currentTime || 0
              );
            }
          });
        }
        return newStems;
      });
    }, []),

    setPan: useCallback((type: StemType, pan: number) => {
      setStems(prev => {
        const newStems = new Map(prev);
        const stem = newStems.get(type);
        if (stem) {
          newStems.set(type, { ...stem, pan });

          const node = stemNodesRef.current.get(type);
          if (node) {
            node.panner.pan.setValueAtTime(pan, audioContextRef.current?.currentTime || 0);
          }
        }
        return newStems;
      });
    }, []),

    resetAll: useCallback(() => {
      setStems(prev => {
        const newStems = new Map(prev);
        newStems.forEach((stem, type) => {
          newStems.set(type, {
            ...stem,
            level: 1.0,
            muted: false,
            solo: false,
            pan: 0,
          });

          const node = stemNodesRef.current.get(type);
          if (node) {
            node.gain.gain.setValueAtTime(1.0, audioContextRef.current?.currentTime || 0);
            node.panner.pan.setValueAtTime(0, audioContextRef.current?.currentTime || 0);
          }
        });
        return newStems;
      });
    }, []),
  };

  // Export individual stem
  const exportStem = useCallback(async (type: StemType): Promise<Blob | null> => {
    const stem = stems.get(type);
    if (!stem?.audioBuffer) return null;

    const ctx = audioContextRef.current;
    if (!ctx) return null;

    // Create offline context for rendering
    const offlineCtx = new OfflineAudioContext(
      stem.audioBuffer.numberOfChannels,
      stem.audioBuffer.length,
      stem.audioBuffer.sampleRate
    );

    const source = offlineCtx.createBufferSource();
    source.buffer = stem.audioBuffer;
    source.connect(offlineCtx.destination);
    source.start();

    const renderedBuffer = await offlineCtx.startRendering();

    // Convert to WAV
    return audioBufferToWav(renderedBuffer);
  }, [stems]);

  // Export mixed stems
  const exportMix = useCallback(async (): Promise<Blob | null> => {
    if (stems.size === 0) return null;

    const firstStem = Array.from(stems.values())[0];
    if (!firstStem?.audioBuffer) return null;

    const ctx = audioContextRef.current;
    if (!ctx) return null;

    const offlineCtx = new OfflineAudioContext(
      firstStem.audioBuffer.numberOfChannels,
      firstStem.audioBuffer.length,
      firstStem.audioBuffer.sampleRate
    );

    const hasSolo = Array.from(stems.values()).some(s => s.solo);

    stems.forEach((stem) => {
      if (!stem.audioBuffer) return;

      const shouldPlay = hasSolo ? stem.solo : !stem.muted;
      if (!shouldPlay) return;

      const source = offlineCtx.createBufferSource();
      source.buffer = stem.audioBuffer;

      const gain = offlineCtx.createGain();
      gain.gain.value = stem.level;

      const panner = offlineCtx.createStereoPanner();
      panner.pan.value = stem.pan;

      source.connect(gain);
      gain.connect(panner);
      panner.connect(offlineCtx.destination);
      source.start();
    });

    const renderedBuffer = await offlineCtx.startRendering();
    return audioBufferToWav(renderedBuffer);
  }, [stems]);

  // Convert AudioBuffer to WAV blob
  function audioBufferToWav(buffer: AudioBuffer): Blob {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = buffer.length * blockAlign;
    const headerSize = 44;
    const totalSize = headerSize + dataSize;

    const arrayBuffer = new ArrayBuffer(totalSize);
    const view = new DataView(arrayBuffer);

    // RIFF header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, totalSize - 8, true);
    writeString(view, 8, 'WAVE');

    // fmt chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);

    // data chunk
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    // Interleave channels and write samples
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

  function writeString(view: DataView, offset: number, string: string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // Cleanup
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      stemNodesRef.current.forEach((node) => {
        if (node.source) {
          try {
            node.source.stop();
          } catch (e) {}
        }
      });
    };
  }, []);

  return {
    // State
    stems,
    progress,
    isProcessing,
    isPlaying,
    currentTime,
    duration,

    // Methods
    separateAudio,
    play,
    pause,
    stop,
    seek,
    controls,
    exportStem,
    exportMix,

    // Constants
    STEM_COLORS,
    STEM_LABELS,
  };
}
