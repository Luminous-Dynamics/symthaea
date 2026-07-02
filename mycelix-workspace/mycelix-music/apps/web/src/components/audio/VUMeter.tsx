// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * VU Meter Component
 *
 * Professional audio level metering with:
 * - Peak and RMS display
 * - Peak hold
 * - Clip indicators
 * - Stereo/Multi-channel support
 * - Multiple visual styles
 */

import React, { useRef, useEffect, memo, useCallback } from 'react';

// ==================== Types ====================

export interface VUMeterProps {
  /** Current level(s) in dB (-Infinity to 0+) */
  level: number | number[];
  /** Peak level(s) for peak hold */
  peak?: number | number[];
  /** Minimum dB value displayed */
  minDb?: number;
  /** Maximum dB value displayed (usually 0 or +6) */
  maxDb?: number;
  /** Orientation */
  orientation?: 'vertical' | 'horizontal';
  /** Visual style */
  variant?: 'gradient' | 'segmented' | 'classic' | 'minimal';
  /** Number of segments for segmented style */
  segments?: number;
  /** Show peak hold indicator */
  showPeak?: boolean;
  /** Peak hold time in ms */
  peakHoldTime?: number;
  /** Show dB scale */
  showScale?: boolean;
  /** Show clip indicator */
  showClip?: boolean;
  /** Width in pixels */
  width?: number;
  /** Height in pixels */
  height?: number;
  /** Gap between stereo channels */
  channelGap?: number;
  /** Low level color (green) */
  colorLow?: string;
  /** Mid level color (yellow) */
  colorMid?: string;
  /** High level color (orange/red) */
  colorHigh?: string;
  /** Clip color */
  colorClip?: string;
  /** Background color */
  colorBackground?: string;
  /** Class name */
  className?: string;
  /** Click handler (for reset peak) */
  onClick?: () => void;
}

// ==================== Constants ====================

const DEFAULT_THRESHOLDS = {
  mid: -12,  // dB where color changes to yellow
  high: -6,  // dB where color changes to red
  clip: 0,   // dB where clip indicator lights
};

// ==================== Helper Functions ====================

function dbToLinear(db: number, minDb: number, maxDb: number): number {
  if (db <= minDb) return 0;
  if (db >= maxDb) return 1;
  return (db - minDb) / (maxDb - minDb);
}

function getColorForLevel(
  db: number,
  colorLow: string,
  colorMid: string,
  colorHigh: string
): string {
  if (db >= DEFAULT_THRESHOLDS.high) return colorHigh;
  if (db >= DEFAULT_THRESHOLDS.mid) return colorMid;
  return colorLow;
}

// ==================== Component ====================

export const VUMeter = memo(function VUMeter({
  level,
  peak,
  minDb = -60,
  maxDb = 6,
  orientation = 'vertical',
  variant = 'gradient',
  segments = 30,
  showPeak = true,
  peakHoldTime = 2000,
  showScale = true,
  showClip = true,
  width,
  height,
  channelGap = 2,
  colorLow = 'var(--color-meter-low, #10B981)',
  colorMid = 'var(--color-meter-mid, #F59E0B)',
  colorHigh = 'var(--color-meter-high, #EF4444)',
  colorClip = 'var(--color-meter-clip, #FF0000)',
  colorBackground = 'var(--color-background-sunken, #080808)',
  className,
  onClick,
}: VUMeterProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const peakHoldRef = useRef<number[]>([]);
  const peakTimeRef = useRef<number[]>([]);
  const clipRef = useRef<boolean[]>([]);
  const animationRef = useRef<number>();

  // Normalize levels to array
  const levels = Array.isArray(level) ? level : [level];
  const peaks = peak ? (Array.isArray(peak) ? peak : [peak]) : levels;
  const channelCount = levels.length;

  // Calculate dimensions
  const isVertical = orientation === 'vertical';
  const defaultWidth = isVertical ? (channelCount * 12 + (channelCount - 1) * channelGap + (showScale ? 30 : 0)) : 200;
  const defaultHeight = isVertical ? 200 : (channelCount * 12 + (channelCount - 1) * channelGap);
  const actualWidth = width || defaultWidth;
  const actualHeight = height || defaultHeight;

  // Update peak hold
  useEffect(() => {
    const now = Date.now();

    levels.forEach((lvl, i) => {
      if (peakHoldRef.current[i] === undefined || lvl > peakHoldRef.current[i]) {
        peakHoldRef.current[i] = lvl;
        peakTimeRef.current[i] = now;
      } else if (now - peakTimeRef.current[i] > peakHoldTime) {
        peakHoldRef.current[i] = lvl;
      }

      // Track clip state
      if (lvl >= DEFAULT_THRESHOLDS.clip) {
        clipRef.current[i] = true;
      }
    });
  }, [levels, peakHoldTime]);

  // Draw meter
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = actualWidth * dpr;
    canvas.height = actualHeight * dpr;
    canvas.style.width = `${actualWidth}px`;
    canvas.style.height = `${actualHeight}px`;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.fillStyle = colorBackground;
    ctx.fillRect(0, 0, actualWidth, actualHeight);

    const scaleWidth = showScale ? 30 : 0;
    const meterAreaWidth = actualWidth - scaleWidth;
    const channelWidth = isVertical
      ? (meterAreaWidth - (channelCount - 1) * channelGap) / channelCount
      : actualWidth;
    const channelHeight = isVertical
      ? actualHeight
      : (actualHeight - (channelCount - 1) * channelGap) / channelCount;

    // Draw scale
    if (showScale) {
      ctx.fillStyle = 'var(--color-text-subtle, #6B6B6B)';
      ctx.font = '9px monospace';
      ctx.textAlign = 'right';

      const scaleMarks = [0, -6, -12, -24, -48];
      scaleMarks.forEach(db => {
        const ratio = dbToLinear(db, minDb, maxDb);
        const y = isVertical
          ? actualHeight * (1 - ratio)
          : actualWidth * ratio;

        if (isVertical) {
          ctx.fillText(`${db}`, scaleWidth - 4, y + 3);
          ctx.beginPath();
          ctx.moveTo(scaleWidth - 2, y);
          ctx.lineTo(scaleWidth, y);
          ctx.strokeStyle = 'var(--color-text-subtle, #6B6B6B)';
          ctx.stroke();
        }
      });
    }

    // Draw each channel
    levels.forEach((lvl, channelIndex) => {
      const linearLevel = dbToLinear(lvl, minDb, maxDb);
      const peakLevel = dbToLinear(peakHoldRef.current[channelIndex] || lvl, minDb, maxDb);

      let x: number, y: number, w: number, h: number;

      if (isVertical) {
        x = scaleWidth + channelIndex * (channelWidth + channelGap);
        y = 0;
        w = channelWidth;
        h = actualHeight;
      } else {
        x = 0;
        y = channelIndex * (channelHeight + channelGap);
        w = actualWidth;
        h = channelHeight;
      }

      // Draw background track
      ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.fillRect(x, y, w, h);

      if (variant === 'segmented') {
        // Segmented style
        const segmentSize = isVertical ? h / segments : w / segments;
        const gap = 1;
        const litSegments = Math.floor(linearLevel * segments);

        for (let i = 0; i < segments; i++) {
          const segmentDb = minDb + (i / segments) * (maxDb - minDb);
          const isLit = i < litSegments;

          let sx: number, sy: number, sw: number, sh: number;

          if (isVertical) {
            sx = x;
            sy = y + h - (i + 1) * segmentSize + gap;
            sw = w;
            sh = segmentSize - gap;
          } else {
            sx = x + i * segmentSize;
            sy = y;
            sw = segmentSize - gap;
            sh = h;
          }

          if (isLit) {
            ctx.fillStyle = getColorForLevel(segmentDb, colorLow, colorMid, colorHigh);
          } else {
            ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
          }

          ctx.fillRect(sx, sy, sw, sh);
        }
      } else if (variant === 'gradient') {
        // Gradient style
        const gradient = isVertical
          ? ctx.createLinearGradient(x, y + h, x, y)
          : ctx.createLinearGradient(x, y, x + w, y);

        const lowPos = dbToLinear(DEFAULT_THRESHOLDS.mid, minDb, maxDb);
        const midPos = dbToLinear(DEFAULT_THRESHOLDS.high, minDb, maxDb);

        gradient.addColorStop(0, colorLow);
        gradient.addColorStop(lowPos, colorLow);
        gradient.addColorStop(lowPos, colorMid);
        gradient.addColorStop(midPos, colorMid);
        gradient.addColorStop(midPos, colorHigh);
        gradient.addColorStop(1, colorHigh);

        ctx.fillStyle = gradient;

        if (isVertical) {
          ctx.fillRect(x, y + h * (1 - linearLevel), w, h * linearLevel);
        } else {
          ctx.fillRect(x, y, w * linearLevel, h);
        }
      } else if (variant === 'classic') {
        // Classic VU needle style (simplified as bar)
        ctx.fillStyle = getColorForLevel(lvl, colorLow, colorMid, colorHigh);

        if (isVertical) {
          ctx.fillRect(x, y + h * (1 - linearLevel), w, h * linearLevel);
        } else {
          ctx.fillRect(x, y, w * linearLevel, h);
        }

        // Add glow effect
        ctx.shadowColor = getColorForLevel(lvl, colorLow, colorMid, colorHigh);
        ctx.shadowBlur = 10;
        if (isVertical) {
          ctx.fillRect(x, y + h * (1 - linearLevel), w, 2);
        } else {
          ctx.fillRect(x + w * linearLevel - 2, y, 2, h);
        }
        ctx.shadowBlur = 0;
      } else {
        // Minimal style
        ctx.fillStyle = colorLow;
        if (isVertical) {
          ctx.fillRect(x, y + h * (1 - linearLevel), w, h * linearLevel);
        } else {
          ctx.fillRect(x, y, w * linearLevel, h);
        }
      }

      // Draw peak hold
      if (showPeak && peakLevel > 0) {
        const peakColor = getColorForLevel(
          peakHoldRef.current[channelIndex] || 0,
          colorLow,
          colorMid,
          colorHigh
        );

        ctx.fillStyle = peakColor;

        if (isVertical) {
          const peakY = y + h * (1 - peakLevel);
          ctx.fillRect(x, peakY, w, 2);
        } else {
          const peakX = x + w * peakLevel - 2;
          ctx.fillRect(peakX, y, 2, h);
        }
      }

      // Draw clip indicator
      if (showClip) {
        const clipX = isVertical ? x : x + w - 8;
        const clipY = isVertical ? y : y;
        const clipW = isVertical ? w : 8;
        const clipH = isVertical ? 8 : h;

        if (clipRef.current[channelIndex]) {
          ctx.fillStyle = colorClip;
        } else {
          ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
        }

        ctx.fillRect(clipX, clipY, clipW, clipH);
      }
    });

    animationRef.current = requestAnimationFrame(draw);
  }, [
    levels, actualWidth, actualHeight, isVertical, channelCount, channelGap,
    minDb, maxDb, variant, segments, showPeak, showScale, showClip,
    colorLow, colorMid, colorHigh, colorClip, colorBackground
  ]);

  // Animation loop
  useEffect(() => {
    draw();
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [draw]);

  // Reset clip on click
  const handleClick = useCallback(() => {
    clipRef.current = clipRef.current.map(() => false);
    onClick?.();
  }, [onClick]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      onClick={handleClick}
      style={{
        display: 'block',
        cursor: onClick ? 'pointer' : 'default',
      }}
      title="Click to reset clip indicator"
    />
  );
});

export default VUMeter;
