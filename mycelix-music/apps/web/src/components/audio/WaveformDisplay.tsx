// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Waveform Display Component
 *
 * High-performance waveform visualization with:
 * - Canvas-based rendering for 60fps performance
 * - Zoom and scroll support
 * - Selection and region marking
 * - Playhead tracking
 * - Multiple display modes (waveform, spectrogram, bars)
 */

import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
  forwardRef,
  useImperativeHandle,
  memo,
} from 'react';

// ==================== Types ====================

export interface WaveformData {
  peaks: Float32Array;
  duration: number;
  sampleRate: number;
  channels: number;
}

export interface Region {
  id: string;
  start: number;
  end: number;
  color: string;
  label?: string;
  draggable?: boolean;
  resizable?: boolean;
}

export interface Marker {
  id: string;
  time: number;
  color: string;
  label?: string;
}

export type DisplayMode = 'waveform' | 'bars' | 'mirror' | 'line';

export interface WaveformDisplayProps {
  /** Audio data to display */
  audioBuffer?: AudioBuffer;
  /** Pre-computed waveform peaks */
  peaks?: Float32Array;
  /** Total duration in seconds */
  duration: number;
  /** Current playback position in seconds */
  currentTime?: number;
  /** Callback when user seeks */
  onSeek?: (time: number) => void;
  /** Callback when selection changes */
  onSelectionChange?: (start: number, end: number) => void;
  /** Callback when region is updated */
  onRegionUpdate?: (region: Region) => void;
  /** Display mode */
  mode?: DisplayMode;
  /** Zoom level (1 = fit to width) */
  zoom?: number;
  /** Scroll offset in seconds */
  scrollOffset?: number;
  /** Regions to display */
  regions?: Region[];
  /** Markers to display */
  markers?: Marker[];
  /** Waveform color */
  waveformColor?: string;
  /** Progress color */
  progressColor?: string;
  /** Background color */
  backgroundColor?: string;
  /** Playhead color */
  playheadColor?: string;
  /** Height in pixels */
  height?: number;
  /** Show time ruler */
  showRuler?: boolean;
  /** Enable selection */
  selectable?: boolean;
  /** Responsive width */
  responsive?: boolean;
  /** Class name */
  className?: string;
}

export interface WaveformDisplayRef {
  redraw: () => void;
  scrollTo: (time: number) => void;
  zoomToRegion: (start: number, end: number) => void;
  exportImage: (format?: 'png' | 'jpeg') => string;
}

// ==================== Helper Functions ====================

function extractPeaks(audioBuffer: AudioBuffer, targetLength: number): Float32Array {
  const channelData = audioBuffer.getChannelData(0);
  const peaks = new Float32Array(targetLength * 2);
  const blockSize = Math.floor(channelData.length / targetLength);

  for (let i = 0; i < targetLength; i++) {
    const start = i * blockSize;
    const end = Math.min(start + blockSize, channelData.length);

    let min = 0;
    let max = 0;

    for (let j = start; j < end; j++) {
      const sample = channelData[j];
      if (sample < min) min = sample;
      if (sample > max) max = sample;
    }

    peaks[i * 2] = min;
    peaks[i * 2 + 1] = max;
  }

  return peaks;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
}

// ==================== Component ====================

export const WaveformDisplay = memo(forwardRef<WaveformDisplayRef, WaveformDisplayProps>(
  function WaveformDisplay(
    {
      audioBuffer,
      peaks: externalPeaks,
      duration,
      currentTime = 0,
      onSeek,
      onSelectionChange,
      onRegionUpdate,
      mode = 'waveform',
      zoom = 1,
      scrollOffset = 0,
      regions = [],
      markers = [],
      waveformColor = 'var(--color-waveform-primary, #8B5CF6)',
      progressColor = 'var(--color-waveform-progress, #A78BFA)',
      backgroundColor = 'var(--color-background-sunken, #080808)',
      playheadColor = 'var(--color-primary, #8B5CF6)',
      height = 128,
      showRuler = true,
      selectable = true,
      responsive = true,
      className,
    },
    ref
  ) {
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const rulerCanvasRef = useRef<HTMLCanvasElement>(null);
    const animationRef = useRef<number>();

    const [peaks, setPeaks] = useState<Float32Array | null>(externalPeaks || null);
    const [canvasWidth, setCanvasWidth] = useState(800);
    const [isDragging, setIsDragging] = useState(false);
    const [selectionStart, setSelectionStart] = useState<number | null>(null);
    const [selectionEnd, setSelectionEnd] = useState<number | null>(null);
    const [hoverTime, setHoverTime] = useState<number | null>(null);

    // Extract peaks from audio buffer
    useEffect(() => {
      if (audioBuffer && !externalPeaks) {
        const targetLength = Math.min(canvasWidth * zoom * 2, audioBuffer.length);
        const extracted = extractPeaks(audioBuffer, Math.floor(targetLength));
        setPeaks(extracted);
      } else if (externalPeaks) {
        setPeaks(externalPeaks);
      }
    }, [audioBuffer, externalPeaks, canvasWidth, zoom]);

    // Handle resize
    useEffect(() => {
      if (!responsive || !containerRef.current) return;

      const observer = new ResizeObserver((entries) => {
        const width = entries[0]?.contentRect.width || 800;
        setCanvasWidth(width);
      });

      observer.observe(containerRef.current);
      return () => observer.disconnect();
    }, [responsive]);

    // Convert pixel position to time
    const pixelToTime = useCallback((x: number): number => {
      const visibleDuration = duration / zoom;
      return scrollOffset + (x / canvasWidth) * visibleDuration;
    }, [duration, zoom, scrollOffset, canvasWidth]);

    // Convert time to pixel position
    const timeToPixel = useCallback((time: number): number => {
      const visibleDuration = duration / zoom;
      return ((time - scrollOffset) / visibleDuration) * canvasWidth;
    }, [duration, zoom, scrollOffset, canvasWidth]);

    // Draw waveform
    const draw = useCallback(() => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (!canvas || !ctx || !peaks) return;

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

      // Calculate visible range
      const visibleDuration = duration / zoom;
      const startTime = scrollOffset;
      const endTime = scrollOffset + visibleDuration;

      const peaksPerSecond = peaks.length / 2 / duration;
      const startPeak = Math.floor(startTime * peaksPerSecond);
      const endPeak = Math.ceil(endTime * peaksPerSecond);
      const visiblePeaks = endPeak - startPeak;

      const centerY = displayHeight / 2;
      const amplitude = (displayHeight - 4) / 2;

      // Draw regions
      regions.forEach(region => {
        const startX = timeToPixel(region.start);
        const endX = timeToPixel(region.end);
        ctx.fillStyle = region.color + '40'; // 25% opacity
        ctx.fillRect(startX, 0, endX - startX, displayHeight);
        ctx.strokeStyle = region.color;
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, 0, endX - startX, displayHeight);

        if (region.label) {
          ctx.fillStyle = region.color;
          ctx.font = '11px sans-serif';
          ctx.fillText(region.label, startX + 4, 14);
        }
      });

      // Draw selection
      if (selectionStart !== null && selectionEnd !== null) {
        const startX = timeToPixel(Math.min(selectionStart, selectionEnd));
        const endX = timeToPixel(Math.max(selectionStart, selectionEnd));
        ctx.fillStyle = 'rgba(139, 92, 246, 0.3)';
        ctx.fillRect(startX, 0, endX - startX, displayHeight);
      }

      // Draw waveform
      ctx.beginPath();

      if (mode === 'waveform' || mode === 'mirror') {
        for (let i = 0; i < visiblePeaks && (startPeak + i) * 2 + 1 < peaks.length; i++) {
          const peakIndex = startPeak + i;
          const x = (i / visiblePeaks) * displayWidth;
          const min = peaks[peakIndex * 2];
          const max = peaks[peakIndex * 2 + 1];

          if (mode === 'waveform') {
            const y1 = centerY - max * amplitude;
            const y2 = centerY - min * amplitude;
            ctx.moveTo(x, y1);
            ctx.lineTo(x, y2);
          } else {
            // Mirror mode - full height
            const y1 = centerY - Math.abs(max) * amplitude;
            const y2 = centerY + Math.abs(min) * amplitude;
            ctx.moveTo(x, y1);
            ctx.lineTo(x, y2);
          }
        }
      } else if (mode === 'bars') {
        const barWidth = Math.max(2, displayWidth / visiblePeaks - 1);
        for (let i = 0; i < visiblePeaks && (startPeak + i) * 2 + 1 < peaks.length; i++) {
          const peakIndex = startPeak + i;
          const x = (i / visiblePeaks) * displayWidth;
          const max = Math.max(Math.abs(peaks[peakIndex * 2]), Math.abs(peaks[peakIndex * 2 + 1]));
          const barHeight = max * amplitude * 2;
          ctx.rect(x, centerY - barHeight / 2, barWidth, barHeight);
        }
      } else if (mode === 'line') {
        ctx.moveTo(0, centerY);
        for (let i = 0; i < visiblePeaks && (startPeak + i) * 2 + 1 < peaks.length; i++) {
          const peakIndex = startPeak + i;
          const x = (i / visiblePeaks) * displayWidth;
          const avg = (peaks[peakIndex * 2] + peaks[peakIndex * 2 + 1]) / 2;
          ctx.lineTo(x, centerY - avg * amplitude);
        }
      }

      // Calculate progress position
      const progressX = timeToPixel(currentTime);

      // Draw played portion
      ctx.save();
      ctx.clip();
      ctx.strokeStyle = waveformColor;
      ctx.lineWidth = mode === 'line' ? 2 : 1;
      ctx.stroke();
      ctx.restore();

      // Draw progress overlay
      if (progressX > 0) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, progressX, displayHeight);
        ctx.clip();

        ctx.beginPath();
        if (mode === 'waveform' || mode === 'mirror') {
          for (let i = 0; i < visiblePeaks && (startPeak + i) * 2 + 1 < peaks.length; i++) {
            const peakIndex = startPeak + i;
            const x = (i / visiblePeaks) * displayWidth;
            const min = peaks[peakIndex * 2];
            const max = peaks[peakIndex * 2 + 1];
            const y1 = centerY - max * amplitude;
            const y2 = centerY - min * amplitude;
            ctx.moveTo(x, y1);
            ctx.lineTo(x, y2);
          }
        }
        ctx.strokeStyle = progressColor;
        ctx.stroke();
        ctx.restore();
      }

      // Draw markers
      markers.forEach(marker => {
        const x = timeToPixel(marker.time);
        if (x >= 0 && x <= displayWidth) {
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, displayHeight);
          ctx.strokeStyle = marker.color;
          ctx.lineWidth = 2;
          ctx.stroke();

          if (marker.label) {
            ctx.fillStyle = marker.color;
            ctx.font = 'bold 10px sans-serif';
            ctx.fillText(marker.label, x + 4, displayHeight - 4);
          }
        }
      });

      // Draw playhead
      if (progressX >= 0 && progressX <= displayWidth) {
        ctx.beginPath();
        ctx.moveTo(progressX, 0);
        ctx.lineTo(progressX, displayHeight);
        ctx.strokeStyle = playheadColor;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Playhead triangle
        ctx.beginPath();
        ctx.moveTo(progressX - 6, 0);
        ctx.lineTo(progressX + 6, 0);
        ctx.lineTo(progressX, 8);
        ctx.closePath();
        ctx.fillStyle = playheadColor;
        ctx.fill();
      }

      // Draw hover time
      if (hoverTime !== null) {
        const hoverX = timeToPixel(hoverTime);
        ctx.beginPath();
        ctx.moveTo(hoverX, 0);
        ctx.lineTo(hoverX, displayHeight);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Time tooltip
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(hoverX + 4, 4, 60, 20);
        ctx.fillStyle = '#fff';
        ctx.font = '11px monospace';
        ctx.fillText(formatTime(hoverTime), hoverX + 8, 18);
      }

      // Draw center line
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(displayWidth, centerY);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }, [
      peaks, canvasWidth, height, duration, zoom, scrollOffset, currentTime,
      mode, waveformColor, progressColor, backgroundColor, playheadColor,
      regions, markers, selectionStart, selectionEnd, hoverTime, timeToPixel
    ]);

    // Draw ruler
    const drawRuler = useCallback(() => {
      if (!showRuler) return;

      const canvas = rulerCanvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (!canvas || !ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const displayWidth = canvasWidth;
      const rulerHeight = 24;

      canvas.width = displayWidth * dpr;
      canvas.height = rulerHeight * dpr;
      canvas.style.width = `${displayWidth}px`;
      canvas.style.height = `${rulerHeight}px`;
      ctx.scale(dpr, dpr);

      ctx.fillStyle = 'var(--color-surface, #1F1F1F)';
      ctx.fillRect(0, 0, displayWidth, rulerHeight);

      const visibleDuration = duration / zoom;
      const startTime = scrollOffset;

      // Calculate tick interval based on zoom
      let tickInterval = 1; // seconds
      const pixelsPerSecond = displayWidth / visibleDuration;

      if (pixelsPerSecond < 10) tickInterval = 30;
      else if (pixelsPerSecond < 20) tickInterval = 10;
      else if (pixelsPerSecond < 50) tickInterval = 5;
      else if (pixelsPerSecond < 100) tickInterval = 1;
      else if (pixelsPerSecond < 200) tickInterval = 0.5;
      else tickInterval = 0.1;

      const firstTick = Math.ceil(startTime / tickInterval) * tickInterval;

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.fillStyle = 'var(--color-text-muted, #A1A1A1)';
      ctx.font = '10px monospace';

      for (let time = firstTick; time < startTime + visibleDuration; time += tickInterval) {
        const x = timeToPixel(time);
        const isMainTick = time % (tickInterval * 5) < 0.001;

        ctx.beginPath();
        ctx.moveTo(x, isMainTick ? 0 : rulerHeight - 8);
        ctx.lineTo(x, rulerHeight);
        ctx.stroke();

        if (isMainTick) {
          ctx.fillText(formatTime(time), x + 2, 12);
        }
      }
    }, [showRuler, canvasWidth, duration, zoom, scrollOffset, timeToPixel]);

    // Animation loop
    useEffect(() => {
      const animate = () => {
        draw();
        drawRuler();
        animationRef.current = requestAnimationFrame(animate);
      };

      animate();

      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
      };
    }, [draw, drawRuler]);

    // Mouse handlers
    const handleMouseDown = useCallback((e: React.MouseEvent) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left;
      const time = pixelToTime(x);

      if (selectable && e.shiftKey) {
        setSelectionStart(time);
        setSelectionEnd(time);
        setIsDragging(true);
      } else {
        onSeek?.(time);
      }
    }, [pixelToTime, selectable, onSeek]);

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left;
      const time = pixelToTime(x);

      setHoverTime(time);

      if (isDragging && selectionStart !== null) {
        setSelectionEnd(time);
      }
    }, [pixelToTime, isDragging, selectionStart]);

    const handleMouseUp = useCallback(() => {
      if (isDragging && selectionStart !== null && selectionEnd !== null) {
        onSelectionChange?.(
          Math.min(selectionStart, selectionEnd),
          Math.max(selectionStart, selectionEnd)
        );
      }
      setIsDragging(false);
    }, [isDragging, selectionStart, selectionEnd, onSelectionChange]);

    const handleMouseLeave = useCallback(() => {
      setHoverTime(null);
      if (isDragging) {
        handleMouseUp();
      }
    }, [isDragging, handleMouseUp]);

    // Expose methods via ref
    useImperativeHandle(ref, () => ({
      redraw: draw,
      scrollTo: (time: number) => {
        // Would update scrollOffset prop via parent
        console.log('Scroll to:', time);
      },
      zoomToRegion: (start: number, end: number) => {
        console.log('Zoom to region:', start, end);
      },
      exportImage: (format = 'png') => {
        return canvasRef.current?.toDataURL(`image/${format}`) || '';
      },
    }), [draw]);

    return (
      <div
        ref={containerRef}
        className={className}
        style={{
          display: 'flex',
          flexDirection: 'column',
          width: responsive ? '100%' : canvasWidth,
          userSelect: 'none',
        }}
      >
        {showRuler && (
          <canvas
            ref={rulerCanvasRef}
            style={{
              display: 'block',
              borderBottom: '1px solid var(--color-surface-border, #333)',
            }}
          />
        )}
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          style={{
            display: 'block',
            cursor: isDragging ? 'col-resize' : 'pointer',
          }}
        />
      </div>
    );
  }
));

export default WaveformDisplay;
