// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

/**
 * WaveformViewer Component
 *
 * Displays a static waveform visualization for audio tracks.
 * Supports seeking by clicking on the waveform.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useWasmAudio, type WaveformData } from '@/hooks/useWasmAudio';

export interface WaveformViewerProps {
  /** Audio URL to generate waveform from */
  audioUrl?: string;
  /** Pre-generated waveform data */
  waveformData?: WaveformData;
  /** Current playback position (0-1) */
  progress?: number;
  /** Callback when user seeks by clicking */
  onSeek?: (position: number) => void;
  /** Width of the component */
  width?: number | string;
  /** Height of the component */
  height?: number;
  /** Primary waveform color */
  primaryColor?: string;
  /** Progress overlay color */
  progressColor?: string;
  /** Background color */
  backgroundColor?: string;
  /** Show RMS overlay */
  showRms?: boolean;
  /** Number of peaks to generate */
  targetPeaks?: number;
  /** Additional CSS class */
  className?: string;
}

export function WaveformViewer({
  audioUrl,
  waveformData: externalData,
  progress = 0,
  onSeek,
  width = '100%',
  height = 80,
  primaryColor = '#818cf8',
  progressColor = '#c084fc',
  backgroundColor = 'transparent',
  showRms = true,
  targetPeaks = 800,
  className = '',
}: WaveformViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [waveformData, setWaveformData] = useState<WaveformData | null>(externalData || null);
  const [isLoading, setIsLoading] = useState(false);
  const [canvasWidth, setCanvasWidth] = useState(800);
  const { isLoaded, generateWaveform } = useWasmAudio();

  // Measure container width
  useEffect(() => {
    if (!containerRef.current) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setCanvasWidth(entry.contentRect.width);
      }
    });

    observer.observe(containerRef.current);
    setCanvasWidth(containerRef.current.clientWidth);

    return () => observer.disconnect();
  }, []);

  // Load and generate waveform from URL
  useEffect(() => {
    if (!audioUrl || !isLoaded || externalData) return;

    setIsLoading(true);

    const loadAudio = async () => {
      try {
        const audioContext = new AudioContext();
        const response = await fetch(audioUrl);
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        const data = generateWaveform(audioBuffer, targetPeaks);
        if (data) {
          setWaveformData(data);
        }

        await audioContext.close();
      } catch (error) {
        console.error('Failed to load waveform:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadAudio();
  }, [audioUrl, isLoaded, generateWaveform, targetPeaks, externalData]);

  // Update from external data
  useEffect(() => {
    if (externalData) {
      setWaveformData(externalData);
    }
  }, [externalData]);

  // Draw waveform
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !waveformData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const displayWidth = canvasWidth;
    const displayHeight = height;

    // Set canvas size with DPR
    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    canvas.style.width = `${displayWidth}px`;
    canvas.style.height = `${displayHeight}px`;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, displayWidth, displayHeight);

    const peaks = Array.from(waveformData.peaks);
    const rms = Array.from(waveformData.rms);
    const barWidth = displayWidth / peaks.length;
    const midY = displayHeight / 2;
    const maxHeight = displayHeight * 0.9;

    // Draw RMS (quieter, background)
    if (showRms && rms.length > 0) {
      ctx.fillStyle = `${primaryColor}30`;
      for (let i = 0; i < rms.length; i++) {
        const h = rms[i] * maxHeight;
        const x = i * barWidth;
        ctx.fillRect(x, midY - h / 2, Math.max(1, barWidth - 0.5), h);
      }
    }

    // Draw peaks (full amplitude)
    ctx.fillStyle = primaryColor;
    for (let i = 0; i < peaks.length; i++) {
      const h = peaks[i] * maxHeight;
      const x = i * barWidth;
      ctx.fillRect(x, midY - h / 2, Math.max(1, barWidth - 0.5), h);
    }

    // Draw progress overlay
    if (progress > 0) {
      const progressX = displayWidth * progress;

      // Progress fill
      ctx.fillStyle = progressColor;
      ctx.globalCompositeOperation = 'source-atop';
      ctx.fillRect(0, 0, progressX, displayHeight);
      ctx.globalCompositeOperation = 'source-over';

      // Progress line
      ctx.strokeStyle = progressColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(progressX, 0);
      ctx.lineTo(progressX, displayHeight);
      ctx.stroke();
    }
  }, [waveformData, canvasWidth, height, progress, primaryColor, progressColor, backgroundColor, showRms]);

  // Handle seek click
  const handleClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onSeek || !canvasRef.current) return;

      const rect = canvasRef.current.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const position = x / rect.width;
      onSeek(Math.max(0, Math.min(1, position)));
    },
    [onSeek]
  );

  // Handle hover
  const [hoverPosition, setHoverPosition] = useState<number | null>(null);

  const handleMouseMove = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    setHoverPosition(x / rect.width);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setHoverPosition(null);
  }, []);

  return (
    <div
      ref={containerRef}
      className={`waveform-viewer relative ${className}`}
      style={{ width, height }}
    >
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/20">
          <div className="animate-pulse text-sm text-gray-400">Loading waveform...</div>
        </div>
      )}

      <canvas
        ref={canvasRef}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className={`w-full h-full ${onSeek ? 'cursor-pointer' : ''}`}
      />

      {/* Hover indicator */}
      {hoverPosition !== null && onSeek && (
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white/50 pointer-events-none"
          style={{ left: `${hoverPosition * 100}%` }}
        />
      )}
    </div>
  );
}

export default WaveformViewer;
