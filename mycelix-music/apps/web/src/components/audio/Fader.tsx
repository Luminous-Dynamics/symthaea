// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Fader Component
 *
 * Professional audio fader/slider with:
 * - Smooth dragging with inertia
 * - Fine-tune with shift key
 * - dB scale markings
 * - Integrated level meter
 * - Touch support
 */

import React, {
  useRef,
  useState,
  useCallback,
  useEffect,
  memo,
  forwardRef,
} from 'react';

// ==================== Types ====================

export interface FaderProps {
  /** Current value */
  value: number;
  /** Minimum value */
  min?: number;
  /** Maximum value */
  max?: number;
  /** Step size */
  step?: number;
  /** Default value (for reset) */
  defaultValue?: number;
  /** Value changed callback */
  onChange: (value: number) => void;
  /** Change complete callback */
  onChangeEnd?: (value: number) => void;
  /** Orientation */
  orientation?: 'vertical' | 'horizontal';
  /** Fader length in pixels */
  length?: number;
  /** Track width in pixels */
  trackWidth?: number;
  /** Visual style */
  variant?: 'default' | 'minimal' | 'studio';
  /** Label text */
  label?: string;
  /** Unit display */
  unit?: string;
  /** Value formatter */
  formatValue?: (value: number) => string;
  /** Active/handle color */
  color?: string;
  /** Track color */
  trackColor?: string;
  /** Show scale markings */
  showScale?: boolean;
  /** Scale markings (array of values) */
  scaleMarks?: number[];
  /** Disabled state */
  disabled?: boolean;
  /** Integrated meter level (optional) */
  meterLevel?: number;
  /** Class name */
  className?: string;
}

// ==================== Helper Functions ====================

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function snapToStep(value: number, step: number, min: number): number {
  return Math.round((value - min) / step) * step + min;
}

// ==================== Component ====================

export const Fader = memo(forwardRef<HTMLDivElement, FaderProps>(
  function Fader(
    {
      value,
      min = -70,
      max = 6,
      step = 0.1,
      defaultValue = 0,
      onChange,
      onChangeEnd,
      orientation = 'vertical',
      length = 200,
      trackWidth = 8,
      variant = 'default',
      label,
      unit = 'dB',
      formatValue,
      color = 'var(--color-primary, #8B5CF6)',
      trackColor = 'var(--color-surface-border, #333)',
      showScale = true,
      scaleMarks = [6, 0, -6, -12, -24, -48, -70],
      disabled = false,
      meterLevel,
      className,
    },
    ref
  ) {
    const containerRef = useRef<HTMLDivElement>(null);
    const trackRef = useRef<HTMLDivElement>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [isHovering, setIsHovering] = useState(false);

    const isVertical = orientation === 'vertical';
    const handleSize = variant === 'minimal' ? 16 : 24;
    const scaleWidth = showScale ? 32 : 0;

    // Calculate handle position
    const normalizedValue = (value - min) / (max - min);
    const handlePosition = isVertical
      ? (1 - normalizedValue) * (length - handleSize)
      : normalizedValue * (length - handleSize);

    // Format display value
    const displayValue = formatValue
      ? formatValue(value)
      : `${value.toFixed(1)}${unit}`;

    // Convert position to value
    const positionToValue = useCallback((clientX: number, clientY: number): number => {
      const track = trackRef.current;
      if (!track) return value;

      const rect = track.getBoundingClientRect();
      let ratio: number;

      if (isVertical) {
        ratio = 1 - (clientY - rect.top - handleSize / 2) / (length - handleSize);
      } else {
        ratio = (clientX - rect.left - handleSize / 2) / (length - handleSize);
      }

      ratio = clamp(ratio, 0, 1);
      return snapToStep(min + ratio * (max - min), step, min);
    }, [isVertical, length, handleSize, min, max, step, value]);

    // Handle start
    const handleStart = useCallback((clientX: number, clientY: number) => {
      if (disabled) return;
      setIsDragging(true);
      const newValue = positionToValue(clientX, clientY);
      if (newValue !== value) {
        onChange(newValue);
      }
    }, [disabled, positionToValue, value, onChange]);

    // Handle move
    const handleMove = useCallback((clientX: number, clientY: number, shiftKey: boolean) => {
      if (!isDragging || disabled) return;

      let newValue = positionToValue(clientX, clientY);

      // Fine-tune with shift
      if (shiftKey) {
        const delta = newValue - value;
        newValue = value + delta * 0.1;
        newValue = clamp(newValue, min, max);
      }

      newValue = snapToStep(newValue, step, min);

      if (newValue !== value) {
        onChange(newValue);
      }
    }, [isDragging, disabled, positionToValue, value, min, max, step, onChange]);

    // Handle end
    const handleEnd = useCallback(() => {
      if (isDragging) {
        setIsDragging(false);
        onChangeEnd?.(value);
      }
    }, [isDragging, value, onChangeEnd]);

    // Double-click to reset
    const handleDoubleClick = useCallback(() => {
      if (disabled) return;
      onChange(defaultValue);
      onChangeEnd?.(defaultValue);
    }, [disabled, defaultValue, onChange, onChangeEnd]);

    // Mouse handlers
    const handleMouseDown = useCallback((e: React.MouseEvent) => {
      e.preventDefault();
      handleStart(e.clientX, e.clientY);
    }, [handleStart]);

    // Touch handlers
    const handleTouchStart = useCallback((e: React.TouchEvent) => {
      const touch = e.touches[0];
      handleStart(touch.clientX, touch.clientY);
    }, [handleStart]);

    // Global move/end handlers
    useEffect(() => {
      if (!isDragging) return;

      const handleMouseMove = (e: MouseEvent) => {
        handleMove(e.clientX, e.clientY, e.shiftKey);
      };

      const handleTouchMove = (e: TouchEvent) => {
        const touch = e.touches[0];
        handleMove(touch.clientX, touch.clientY, false);
      };

      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleEnd);
      window.addEventListener('touchmove', handleTouchMove);
      window.addEventListener('touchend', handleEnd);

      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleEnd);
        window.removeEventListener('touchmove', handleTouchMove);
        window.removeEventListener('touchend', handleEnd);
      };
    }, [isDragging, handleMove, handleEnd]);

    // Wheel handler
    const handleWheel = useCallback((e: React.WheelEvent) => {
      if (disabled) return;
      e.preventDefault();

      const direction = e.deltaY > 0 ? -1 : 1;
      const multiplier = e.shiftKey ? 0.1 : 1;
      const newValue = clamp(value + direction * step * multiplier * 10, min, max);

      onChange(snapToStep(newValue, step, min));
    }, [disabled, value, min, max, step, onChange]);

    const containerStyle: React.CSSProperties = {
      display: 'inline-flex',
      flexDirection: isVertical ? 'column' : 'row',
      alignItems: 'center',
      gap: '8px',
      opacity: disabled ? 0.5 : 1,
    };

    const trackContainerStyle: React.CSSProperties = {
      display: 'flex',
      flexDirection: isVertical ? 'row' : 'column',
      alignItems: 'center',
      gap: '4px',
    };

    const trackStyle: React.CSSProperties = {
      position: 'relative',
      width: isVertical ? trackWidth : length,
      height: isVertical ? length : trackWidth,
      background: trackColor,
      borderRadius: trackWidth / 2,
      cursor: disabled ? 'not-allowed' : isVertical ? 'ns-resize' : 'ew-resize',
      touchAction: 'none',
    };

    const fillStyle: React.CSSProperties = {
      position: 'absolute',
      background: color,
      borderRadius: trackWidth / 2,
      transition: isDragging ? 'none' : 'all 50ms ease',
      ...(isVertical
        ? {
            bottom: 0,
            left: 0,
            right: 0,
            height: `${normalizedValue * 100}%`,
          }
        : {
            top: 0,
            left: 0,
            bottom: 0,
            width: `${normalizedValue * 100}%`,
          }),
    };

    const handleStyle: React.CSSProperties = {
      position: 'absolute',
      width: isVertical ? handleSize + 8 : handleSize,
      height: isVertical ? handleSize : handleSize + 8,
      background: variant === 'studio'
        ? 'linear-gradient(to bottom, #666, #333)'
        : 'var(--color-surface, #1F1F1F)',
      border: `2px solid ${isDragging || isHovering ? color : 'var(--color-surface-border, #333)'}`,
      borderRadius: 4,
      transform: 'translate(-50%, -50%)',
      cursor: disabled ? 'not-allowed' : 'grab',
      boxShadow: isDragging
        ? `0 0 10px ${color}`
        : 'var(--shadow-md, 0 4px 6px rgba(0,0,0,0.5))',
      transition: isDragging ? 'none' : 'border-color 150ms, box-shadow 150ms',
      ...(isVertical
        ? {
            left: '50%',
            top: handlePosition + handleSize / 2,
          }
        : {
            top: '50%',
            left: handlePosition + handleSize / 2,
          }),
    };

    // Handle grip lines
    const gripLines = variant === 'studio' && (
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          display: 'flex',
          flexDirection: isVertical ? 'row' : 'column',
          gap: 2,
        }}
      >
        {[0, 1, 2].map(i => (
          <div
            key={i}
            style={{
              width: isVertical ? 2 : 12,
              height: isVertical ? 12 : 2,
              background: 'rgba(255,255,255,0.3)',
              borderRadius: 1,
            }}
          />
        ))}
      </div>
    );

    // Meter overlay
    const meterOverlay = meterLevel !== undefined && (
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: isVertical ? -trackWidth - 2 : 0,
          width: isVertical ? 3 : `${Math.min(100, Math.max(0, (meterLevel + 70) / 76 * 100))}%`,
          height: isVertical ? `${Math.min(100, Math.max(0, (meterLevel + 70) / 76 * 100))}%` : 3,
          background: meterLevel > -6
            ? 'var(--color-meter-high, #EF4444)'
            : meterLevel > -18
            ? 'var(--color-meter-mid, #F59E0B)'
            : 'var(--color-meter-low, #10B981)',
          borderRadius: 2,
          transition: 'height 50ms, width 50ms',
        }}
      />
    );

    return (
      <div
        ref={(el) => {
          (containerRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
          if (typeof ref === 'function') ref(el);
          else if (ref) ref.current = el;
        }}
        className={className}
        style={containerStyle}
        onMouseEnter={() => setIsHovering(true)}
        onMouseLeave={() => setIsHovering(false)}
      >
        {label && (
          <span
            style={{
              fontSize: '11px',
              color: 'var(--color-text-muted, #A1A1A1)',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              writingMode: isVertical ? 'vertical-rl' : 'horizontal-tb',
              transform: isVertical ? 'rotate(180deg)' : 'none',
            }}
          >
            {label}
          </span>
        )}

        <div style={trackContainerStyle}>
          {/* Scale markings */}
          {showScale && isVertical && (
            <div
              style={{
                position: 'relative',
                height: length,
                width: scaleWidth,
                fontSize: '9px',
                color: 'var(--color-text-subtle, #6B6B6B)',
                fontFamily: 'var(--font-family-mono, monospace)',
              }}
            >
              {scaleMarks.map(mark => {
                const pos = (1 - (mark - min) / (max - min)) * (length - 8);
                return (
                  <div
                    key={mark}
                    style={{
                      position: 'absolute',
                      top: pos,
                      right: 4,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 2,
                    }}
                  >
                    <span>{mark}</span>
                    <div
                      style={{
                        width: 4,
                        height: 1,
                        background: 'var(--color-surface-border, #333)',
                      }}
                    />
                  </div>
                );
              })}
            </div>
          )}

          {/* Track */}
          <div
            ref={trackRef}
            style={trackStyle}
            onMouseDown={handleMouseDown}
            onTouchStart={handleTouchStart}
            onDoubleClick={handleDoubleClick}
            onWheel={handleWheel}
          >
            {meterOverlay}
            <div style={fillStyle} />
            <div style={handleStyle}>
              {gripLines}
            </div>
          </div>
        </div>

        {/* Value display */}
        {(isHovering || isDragging) && (
          <span
            style={{
              fontSize: '10px',
              color: 'var(--color-text, #FAFAFA)',
              fontFamily: 'var(--font-family-mono, monospace)',
              background: 'var(--color-surface, #1F1F1F)',
              padding: '2px 6px',
              borderRadius: '4px',
              whiteSpace: 'nowrap',
            }}
          >
            {displayValue}
          </span>
        )}
      </div>
    );
  }
));

export default Fader;
