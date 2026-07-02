// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Knob Component
 *
 * Rotary control for audio parameters with:
 * - Smooth dragging (vertical or circular)
 * - Fine-tune with shift key
 * - Double-click to reset
 * - Visual feedback and tooltips
 * - Multiple visual styles
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

export interface KnobProps {
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
  /** Size in pixels */
  size?: number;
  /** Visual style */
  variant?: 'default' | 'minimal' | 'vintage' | 'modern';
  /** Label text */
  label?: string;
  /** Unit display (e.g., "dB", "%", "Hz") */
  unit?: string;
  /** Value formatter */
  formatValue?: (value: number) => string;
  /** Rotation range in degrees */
  arcRange?: number;
  /** Start angle in degrees (0 = top) */
  startAngle?: number;
  /** Active/primary color */
  color?: string;
  /** Track color */
  trackColor?: string;
  /** Pointer color */
  pointerColor?: string;
  /** Disabled state */
  disabled?: boolean;
  /** Bipolar mode (center is zero) */
  bipolar?: boolean;
  /** Show value on hover */
  showValue?: boolean;
  /** Class name */
  className?: string;
}

// ==================== Helper Functions ====================

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function mapRange(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number
): number {
  return ((value - inMin) / (inMax - inMin)) * (outMax - outMin) + outMin;
}

function snapToStep(value: number, step: number, min: number): number {
  return Math.round((value - min) / step) * step + min;
}

// ==================== Component ====================

export const Knob = memo(forwardRef<HTMLDivElement, KnobProps>(
  function Knob(
    {
      value,
      min = 0,
      max = 100,
      step = 1,
      defaultValue,
      onChange,
      onChangeEnd,
      size = 48,
      variant = 'default',
      label,
      unit = '',
      formatValue,
      arcRange = 270,
      startAngle = 135,
      color = 'var(--color-primary, #8B5CF6)',
      trackColor = 'var(--color-surface-border, #333)',
      pointerColor = 'var(--color-text, #FAFAFA)',
      disabled = false,
      bipolar = false,
      showValue = true,
      className,
    },
    ref
  ) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [isHovering, setIsHovering] = useState(false);
    const startYRef = useRef(0);
    const startValueRef = useRef(0);

    // Calculate rotation angle from value
    const normalizedValue = (value - min) / (max - min);
    const rotationAngle = startAngle + normalizedValue * arcRange;

    // Format display value
    const displayValue = formatValue
      ? formatValue(value)
      : `${Math.round(value * 100) / 100}${unit}`;

    // Handle mouse/touch start
    const handleStart = useCallback((clientY: number) => {
      if (disabled) return;
      setIsDragging(true);
      startYRef.current = clientY;
      startValueRef.current = value;
    }, [disabled, value]);

    // Handle mouse/touch move
    const handleMove = useCallback((clientY: number, shiftKey: boolean) => {
      if (!isDragging || disabled) return;

      const sensitivity = shiftKey ? 0.1 : 0.5; // Fine-tune with shift
      const deltaY = startYRef.current - clientY;
      const deltaValue = (deltaY * sensitivity / 100) * (max - min);

      let newValue = startValueRef.current + deltaValue;
      newValue = clamp(newValue, min, max);
      newValue = snapToStep(newValue, step, min);

      if (newValue !== value) {
        onChange(newValue);
      }
    }, [isDragging, disabled, value, min, max, step, onChange]);

    // Handle mouse/touch end
    const handleEnd = useCallback(() => {
      if (isDragging) {
        setIsDragging(false);
        onChangeEnd?.(value);
      }
    }, [isDragging, value, onChangeEnd]);

    // Double-click to reset
    const handleDoubleClick = useCallback(() => {
      if (disabled) return;
      const resetValue = defaultValue ?? (bipolar ? (min + max) / 2 : min);
      onChange(resetValue);
      onChangeEnd?.(resetValue);
    }, [disabled, defaultValue, bipolar, min, max, onChange, onChangeEnd]);

    // Wheel handler
    const handleWheel = useCallback((e: React.WheelEvent) => {
      if (disabled) return;
      e.preventDefault();

      const direction = e.deltaY > 0 ? -1 : 1;
      const multiplier = e.shiftKey ? 0.1 : 1;
      const newValue = clamp(value + direction * step * multiplier, min, max);

      onChange(snapToStep(newValue, step, min));
    }, [disabled, value, min, max, step, onChange]);

    // Mouse event handlers
    const handleMouseDown = useCallback((e: React.MouseEvent) => {
      e.preventDefault();
      handleStart(e.clientY);
    }, [handleStart]);

    // Touch event handlers
    const handleTouchStart = useCallback((e: React.TouchEvent) => {
      handleStart(e.touches[0].clientY);
    }, [handleStart]);

    // Global mouse/touch move and end
    useEffect(() => {
      if (!isDragging) return;

      const handleMouseMove = (e: MouseEvent) => {
        handleMove(e.clientY, e.shiftKey);
      };

      const handleTouchMove = (e: TouchEvent) => {
        handleMove(e.touches[0].clientY, false);
      };

      const handleMouseUp = () => handleEnd();
      const handleTouchEnd = () => handleEnd();

      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove);
      window.addEventListener('touchend', handleTouchEnd);

      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
        window.removeEventListener('touchmove', handleTouchMove);
        window.removeEventListener('touchend', handleTouchEnd);
      };
    }, [isDragging, handleMove, handleEnd]);

    // SVG arc path
    const createArc = (
      cx: number,
      cy: number,
      radius: number,
      startDeg: number,
      endDeg: number
    ): string => {
      const startRad = (startDeg - 90) * (Math.PI / 180);
      const endRad = (endDeg - 90) * (Math.PI / 180);

      const x1 = cx + radius * Math.cos(startRad);
      const y1 = cy + radius * Math.sin(startRad);
      const x2 = cx + radius * Math.cos(endRad);
      const y2 = cy + radius * Math.sin(endRad);

      const largeArc = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;

      return `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`;
    };

    const center = size / 2;
    const radius = size / 2 - 6;
    const trackStroke = variant === 'minimal' ? 2 : 4;

    // Calculate active arc
    const activeEndAngle = bipolar
      ? mapRange(value, min, max, startAngle, startAngle + arcRange)
      : rotationAngle;
    const activeStartAngle = bipolar
      ? startAngle + arcRange / 2
      : startAngle;

    return (
      <div
        ref={(el) => {
          (containerRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
          if (typeof ref === 'function') ref(el);
          else if (ref) ref.current = el;
        }}
        className={className}
        style={{
          display: 'inline-flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '4px',
          opacity: disabled ? 0.5 : 1,
          cursor: disabled ? 'not-allowed' : 'ns-resize',
          userSelect: 'none',
          touchAction: 'none',
        }}
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
            }}
          >
            {label}
          </span>
        )}

        <svg
          width={size}
          height={size}
          onMouseDown={handleMouseDown}
          onTouchStart={handleTouchStart}
          onDoubleClick={handleDoubleClick}
          onWheel={handleWheel}
          style={{ overflow: 'visible' }}
        >
          {/* Background */}
          {variant !== 'minimal' && (
            <circle
              cx={center}
              cy={center}
              r={radius + 2}
              fill={variant === 'vintage' ? '#2A2A2A' : 'var(--color-surface, #1F1F1F)'}
              stroke={variant === 'modern' ? 'var(--color-surface-border, #333)' : 'none'}
              strokeWidth={1}
            />
          )}

          {/* Track arc */}
          <path
            d={createArc(center, center, radius, startAngle, startAngle + arcRange)}
            fill="none"
            stroke={trackColor}
            strokeWidth={trackStroke}
            strokeLinecap="round"
          />

          {/* Active arc */}
          {value !== (bipolar ? (min + max) / 2 : min) && (
            <path
              d={createArc(
                center,
                center,
                radius,
                Math.min(activeStartAngle, activeEndAngle),
                Math.max(activeStartAngle, activeEndAngle)
              )}
              fill="none"
              stroke={color}
              strokeWidth={trackStroke}
              strokeLinecap="round"
              style={{
                filter: isDragging || isHovering ? `drop-shadow(0 0 4px ${color})` : 'none',
              }}
            />
          )}

          {/* Pointer/indicator */}
          {variant !== 'minimal' && (
            <line
              x1={center}
              y1={center}
              x2={center + (radius - 8) * Math.cos((rotationAngle - 90) * Math.PI / 180)}
              y2={center + (radius - 8) * Math.sin((rotationAngle - 90) * Math.PI / 180)}
              stroke={pointerColor}
              strokeWidth={2}
              strokeLinecap="round"
            />
          )}

          {/* Center dot */}
          {variant === 'modern' && (
            <circle
              cx={center}
              cy={center}
              r={3}
              fill={color}
            />
          )}

          {/* Tick marks for vintage style */}
          {variant === 'vintage' && (
            <>
              {[0, 0.25, 0.5, 0.75, 1].map((tick, i) => {
                const angle = startAngle + tick * arcRange;
                const rad = (angle - 90) * Math.PI / 180;
                const inner = radius + 4;
                const outer = radius + 8;
                return (
                  <line
                    key={i}
                    x1={center + inner * Math.cos(rad)}
                    y1={center + inner * Math.sin(rad)}
                    x2={center + outer * Math.cos(rad)}
                    y2={center + outer * Math.sin(rad)}
                    stroke="var(--color-text-subtle, #6B6B6B)"
                    strokeWidth={1}
                  />
                );
              })}
            </>
          )}
        </svg>

        {/* Value display */}
        {showValue && (isHovering || isDragging) && (
          <span
            style={{
              fontSize: '10px',
              color: 'var(--color-text, #FAFAFA)',
              fontFamily: 'var(--font-family-mono, monospace)',
              background: 'var(--color-surface, #1F1F1F)',
              padding: '2px 6px',
              borderRadius: '4px',
              position: 'absolute',
              transform: 'translateY(100%)',
              marginTop: '4px',
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

export default Knob;
