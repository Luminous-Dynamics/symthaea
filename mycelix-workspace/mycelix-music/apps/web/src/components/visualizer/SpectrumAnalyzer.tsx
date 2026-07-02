// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

/**
 * SpectrumAnalyzer Component
 *
 * Real-time audio visualization using Web Audio API.
 * Supports multiple visualization types powered by WASM.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useWasmVisualizer, type VisualizationType } from '@/hooks/useWasmAudio';

export interface SpectrumAnalyzerProps {
  /** Audio element or MediaStream to analyze */
  audioSource?: HTMLAudioElement | MediaStream | null;
  /** Visualization type */
  visualizationType?: VisualizationType;
  /** Width of the component */
  width?: number | string;
  /** Height of the component */
  height?: number;
  /** FFT size (power of 2, 32-32768) */
  fftSize?: number;
  /** Smoothing time constant (0-1) */
  smoothingTimeConstant?: number;
  /** Primary color */
  primaryColor?: string;
  /** Secondary color */
  secondaryColor?: string;
  /** Background color */
  backgroundColor?: string;
  /** Bar width for frequency bars */
  barWidth?: number;
  /** Gap between bars */
  barGap?: number;
  /** Enable gradient colors */
  gradient?: boolean;
  /** Enable mirror effect */
  mirror?: boolean;
  /** Whether to auto-start when source is available */
  autoStart?: boolean;
  /** Additional CSS class */
  className?: string;
}

export function SpectrumAnalyzer({
  audioSource,
  visualizationType = 'FrequencyBars',
  width = '100%',
  height = 200,
  fftSize = 2048,
  smoothingTimeConstant = 0.8,
  primaryColor = '#818cf8',
  secondaryColor = '#c084fc',
  backgroundColor = '#0a0a0a',
  barWidth = 3,
  barGap = 1,
  gradient = true,
  mirror = false,
  autoStart = true,
  className = '',
}: SpectrumAnalyzerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceNodeRef = useRef<MediaElementAudioSourceNode | MediaStreamAudioSourceNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 200 });

  const { isReady, setColors, drawWaveform, drawFrequencyBars, drawCircular, drawSpectrogram } = useWasmVisualizer({
    viz_type: visualizationType,
    fft_size: fftSize,
    smoothing: smoothingTimeConstant,
    bar_width: barWidth,
    bar_gap: barGap,
    mirror,
    gradient,
  });

  // Update colors when props change
  useEffect(() => {
    if (isReady) {
      setColors(primaryColor, secondaryColor, backgroundColor);
    }
  }, [isReady, setColors, primaryColor, secondaryColor, backgroundColor]);

  // Measure container
  useEffect(() => {
    if (!containerRef.current) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setCanvasSize({
          width: entry.contentRect.width,
          height: typeof height === 'number' ? height : entry.contentRect.height,
        });
      }
    });

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [height]);

  // Initialize audio context and analyser
  useEffect(() => {
    if (!audioSource) return;

    // Create audio context
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }

    const ctx = audioContextRef.current;

    // Create analyser
    const analyser = ctx.createAnalyser();
    analyser.fftSize = fftSize;
    analyser.smoothingTimeConstant = smoothingTimeConstant;
    analyserRef.current = analyser;

    // Connect source
    try {
      if (audioSource instanceof HTMLAudioElement) {
        // Check if already connected
        if (sourceNodeRef.current) {
          sourceNodeRef.current.disconnect();
        }
        sourceNodeRef.current = ctx.createMediaElementSource(audioSource);
        sourceNodeRef.current.connect(analyser);
        analyser.connect(ctx.destination);
      } else if (audioSource instanceof MediaStream) {
        sourceNodeRef.current = ctx.createMediaStreamSource(audioSource);
        sourceNodeRef.current.connect(analyser);
        // Don't connect to destination for microphone input
      }

      if (autoStart) {
        setIsActive(true);
      }
    } catch (error) {
      console.error('Failed to connect audio source:', error);
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [audioSource, fftSize, smoothingTimeConstant, autoStart]);

  // Animation loop
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser || !isReady) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const displayWidth = canvasSize.width;
    const displayHeight = canvasSize.height;

    // Set canvas size
    if (canvas.width !== displayWidth * dpr || canvas.height !== displayHeight * dpr) {
      canvas.width = displayWidth * dpr;
      canvas.height = displayHeight * dpr;
      canvas.style.width = `${displayWidth}px`;
      canvas.style.height = `${displayHeight}px`;
      ctx.scale(dpr, dpr);
    }

    // Clear canvas
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, displayWidth, displayHeight);

    // Get data from analyser
    const frequencyData = new Uint8Array(analyser.frequencyBinCount);
    const timeData = new Float32Array(analyser.fftSize);

    analyser.getByteFrequencyData(frequencyData);
    analyser.getFloatTimeDomainData(timeData);

    // Draw based on visualization type
    switch (visualizationType) {
      case 'Waveform':
        drawWaveform(ctx, timeData, displayWidth, displayHeight);
        break;
      case 'FrequencyBars':
        drawFrequencyBars(ctx, frequencyData, displayWidth, displayHeight);
        break;
      case 'FrequencyLine':
        // Convert frequency data to float for waveform-style rendering
        const floatFreq = new Float32Array(frequencyData.length);
        for (let i = 0; i < frequencyData.length; i++) {
          floatFreq[i] = frequencyData[i] / 255;
        }
        drawWaveform(ctx, floatFreq, displayWidth, displayHeight);
        break;
      case 'Circular':
        drawCircular(ctx, frequencyData, displayWidth, displayHeight);
        break;
      case 'Spectrogram':
        drawSpectrogram(ctx, frequencyData, displayWidth, displayHeight);
        break;
      default:
        drawFallbackBars(ctx, frequencyData, displayWidth, displayHeight);
    }

    if (isActive) {
      animationFrameRef.current = requestAnimationFrame(draw);
    }
  }, [
    isReady,
    isActive,
    canvasSize,
    visualizationType,
    backgroundColor,
    drawWaveform,
    drawFrequencyBars,
    drawCircular,
    drawSpectrogram,
  ]);

  // Fallback frequency bars (pure JS, no WASM)
  const drawFallbackBars = useCallback(
    (ctx: CanvasRenderingContext2D, data: Uint8Array, displayWidth: number, displayHeight: number) => {
      const barCount = Math.floor(displayWidth / (barWidth + barGap));
      const barsToUse = Math.min(barCount, data.length);
      const samplesPerBar = Math.floor(data.length / barsToUse);

      for (let i = 0; i < barsToUse; i++) {
        // Average frequency values for this bar
        let sum = 0;
        for (let j = 0; j < samplesPerBar; j++) {
          sum += data[i * samplesPerBar + j];
        }
        const avg = sum / samplesPerBar;

        const barHeight = (avg / 255) * displayHeight * 0.9;
        const x = i * (barWidth + barGap);
        const y = displayHeight - barHeight;

        if (gradient) {
          const hue = 250 + (avg / 255) * 60;
          ctx.fillStyle = `hsl(${hue}, 80%, ${50 + (avg / 255) * 30}%)`;
        } else {
          ctx.fillStyle = primaryColor;
        }

        ctx.fillRect(x, y, barWidth, barHeight);
      }
    },
    [barWidth, barGap, gradient, primaryColor]
  );

  // Start animation when active
  useEffect(() => {
    if (isActive) {
      animationFrameRef.current = requestAnimationFrame(draw);
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isActive, draw]);

  // Control methods
  const start = useCallback(() => setIsActive(true), []);
  const stop = useCallback(() => setIsActive(false), []);

  return (
    <div
      ref={containerRef}
      className={`spectrum-analyzer relative overflow-hidden ${className}`}
      style={{ width, height }}
    >
      <canvas
        ref={canvasRef}
        className="w-full h-full"
      />

      {!audioSource && (
        <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm">
          No audio source connected
        </div>
      )}
    </div>
  );
}

export default SpectrumAnalyzer;
