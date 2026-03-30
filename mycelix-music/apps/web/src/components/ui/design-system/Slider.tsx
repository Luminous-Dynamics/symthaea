// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { cn } from '@/lib/utils';

export interface SliderProps {
  value?: number;
  defaultValue?: number;
  min?: number;
  max?: number;
  step?: number;
  onChange?: (value: number) => void;
  onChangeEnd?: (value: number) => void;
  disabled?: boolean;
  orientation?: 'horizontal' | 'vertical';
  size?: 'sm' | 'md' | 'lg';
  showValue?: boolean;
  formatValue?: (value: number) => string;
  marks?: Array<{ value: number; label?: string }>;
  gradient?: boolean;
  trackColor?: string;
  thumbColor?: string;
  className?: string;
  label?: string;
}

export function Slider({
  value: controlledValue,
  defaultValue = 0,
  min = 0,
  max = 100,
  step = 1,
  onChange,
  onChangeEnd,
  disabled = false,
  orientation = 'horizontal',
  size = 'md',
  showValue = false,
  formatValue = (v) => v.toString(),
  marks,
  gradient = false,
  trackColor,
  thumbColor,
  className,
  label,
}: SliderProps) {
  const [internalValue, setInternalValue] = useState(defaultValue);
  const [isDragging, setIsDragging] = useState(false);
  const trackRef = useRef<HTMLDivElement>(null);

  const value = controlledValue !== undefined ? controlledValue : internalValue;
  const percentage = ((value - min) / (max - min)) * 100;

  const isVertical = orientation === 'vertical';

  const sizeConfig = {
    sm: { track: 'h-1', thumb: 'w-3 h-3', thumbOffset: '-6px' },
    md: { track: 'h-2', thumb: 'w-4 h-4', thumbOffset: '-8px' },
    lg: { track: 'h-3', thumb: 'w-5 h-5', thumbOffset: '-10px' },
  };

  const updateValue = useCallback(
    (clientX: number, clientY: number) => {
      if (!trackRef.current || disabled) return;

      const rect = trackRef.current.getBoundingClientRect();
      let newPercentage: number;

      if (isVertical) {
        newPercentage = ((rect.bottom - clientY) / rect.height) * 100;
      } else {
        newPercentage = ((clientX - rect.left) / rect.width) * 100;
      }

      newPercentage = Math.max(0, Math.min(100, newPercentage));
      let newValue = min + (newPercentage / 100) * (max - min);

      // Snap to step
      newValue = Math.round(newValue / step) * step;
      newValue = Math.max(min, Math.min(max, newValue));

      if (controlledValue === undefined) {
        setInternalValue(newValue);
      }
      onChange?.(newValue);
    },
    [controlledValue, disabled, isVertical, max, min, onChange, step]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (disabled) return;
      e.preventDefault();
      setIsDragging(true);
      updateValue(e.clientX, e.clientY);
    },
    [disabled, updateValue]
  );

  const handleTouchStart = useCallback(
    (e: React.TouchEvent) => {
      if (disabled) return;
      setIsDragging(true);
      const touch = e.touches[0];
      updateValue(touch.clientX, touch.clientY);
    },
    [disabled, updateValue]
  );

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      updateValue(e.clientX, e.clientY);
    };

    const handleTouchMove = (e: TouchEvent) => {
      const touch = e.touches[0];
      updateValue(touch.clientX, touch.clientY);
    };

    const handleEnd = () => {
      setIsDragging(false);
      onChangeEnd?.(value);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleEnd);
    document.addEventListener('touchmove', handleTouchMove);
    document.addEventListener('touchend', handleEnd);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleEnd);
      document.removeEventListener('touchmove', handleTouchMove);
      document.removeEventListener('touchend', handleEnd);
    };
  }, [isDragging, onChangeEnd, updateValue, value]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (disabled) return;

    let newValue = value;
    const largeStep = (max - min) / 10;

    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowUp':
        newValue = Math.min(max, value + step);
        break;
      case 'ArrowLeft':
      case 'ArrowDown':
        newValue = Math.max(min, value - step);
        break;
      case 'PageUp':
        newValue = Math.min(max, value + largeStep);
        break;
      case 'PageDown':
        newValue = Math.max(min, value - largeStep);
        break;
      case 'Home':
        newValue = min;
        break;
      case 'End':
        newValue = max;
        break;
      default:
        return;
    }

    e.preventDefault();
    newValue = Math.round(newValue / step) * step;

    if (controlledValue === undefined) {
      setInternalValue(newValue);
    }
    onChange?.(newValue);
  };

  return (
    <div
      className={cn(
        'relative',
        isVertical ? 'h-full w-fit' : 'w-full',
        disabled && 'opacity-50',
        className
      )}
    >
      {label && (
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {label}
          </label>
          {showValue && (
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {formatValue(value)}
            </span>
          )}
        </div>
      )}

      <div
        ref={trackRef}
        className={cn(
          'relative cursor-pointer select-none',
          isVertical
            ? cn('w-fit h-full min-h-32', sizeConfig[size].track.replace('h-', 'w-'))
            : cn('w-full', sizeConfig[size].track),
          disabled && 'cursor-not-allowed'
        )}
        onMouseDown={handleMouseDown}
        onTouchStart={handleTouchStart}
      >
        {/* Track background */}
        <div
          className={cn(
            'absolute rounded-full bg-gray-200 dark:bg-gray-700',
            isVertical ? 'w-full h-full' : 'w-full h-full'
          )}
        />

        {/* Track fill */}
        <div
          className={cn(
            'absolute rounded-full transition-all',
            gradient
              ? 'bg-gradient-to-r from-primary-400 to-primary-600'
              : 'bg-primary-600'
          )}
          style={{
            ...(isVertical
              ? { width: '100%', height: `${percentage}%`, bottom: 0 }
              : { height: '100%', width: `${percentage}%` }),
            ...(trackColor && { backgroundColor: trackColor }),
          }}
        />

        {/* Marks */}
        {marks?.map((mark) => {
          const markPercentage = ((mark.value - min) / (max - min)) * 100;
          return (
            <div
              key={mark.value}
              className={cn(
                'absolute',
                isVertical ? '-right-6' : '-bottom-6'
              )}
              style={
                isVertical
                  ? { bottom: `${markPercentage}%` }
                  : { left: `${markPercentage}%`, transform: 'translateX(-50%)' }
              }
            >
              <div className="w-1 h-1 rounded-full bg-gray-400 dark:bg-gray-500" />
              {mark.label && (
                <span className="absolute text-xs text-gray-500 whitespace-nowrap mt-1">
                  {mark.label}
                </span>
              )}
            </div>
          );
        })}

        {/* Thumb */}
        <div
          role="slider"
          tabIndex={disabled ? -1 : 0}
          aria-valuemin={min}
          aria-valuemax={max}
          aria-valuenow={value}
          aria-disabled={disabled}
          onKeyDown={handleKeyDown}
          className={cn(
            'absolute rounded-full bg-white border-2 border-primary-600 shadow-md transition-transform',
            'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
            isDragging && 'scale-110',
            sizeConfig[size].thumb
          )}
          style={{
            ...(isVertical
              ? {
                  bottom: `${percentage}%`,
                  left: '50%',
                  transform: `translate(-50%, 50%)`,
                }
              : {
                  left: `${percentage}%`,
                  top: '50%',
                  transform: `translate(-50%, -50%)`,
                }),
            ...(thumbColor && { borderColor: thumbColor }),
          }}
        />
      </div>

      {showValue && !label && (
        <div className="mt-1 text-center text-sm text-gray-500 dark:text-gray-400">
          {formatValue(value)}
        </div>
      )}
    </div>
  );
}

// Range Slider (dual thumb)
export interface RangeSliderProps extends Omit<SliderProps, 'value' | 'onChange'> {
  value?: [number, number];
  onChange?: (value: [number, number]) => void;
  onChangeEnd?: (value: [number, number]) => void;
  minGap?: number;
}

export function RangeSlider({
  value: controlledValue,
  defaultValue,
  min = 0,
  max = 100,
  step = 1,
  onChange,
  onChangeEnd,
  minGap = 0,
  disabled = false,
  size = 'md',
  className,
  label,
  formatValue = (v) => v.toString(),
}: RangeSliderProps) {
  const [internalValue, setInternalValue] = useState<[number, number]>(
    (defaultValue as [number, number]) || [min, max]
  );
  const [activeThumb, setActiveThumb] = useState<'min' | 'max' | null>(null);
  const trackRef = useRef<HTMLDivElement>(null);

  const value = controlledValue || internalValue;
  const minPercentage = ((value[0] - min) / (max - min)) * 100;
  const maxPercentage = ((value[1] - min) / (max - min)) * 100;

  const sizeConfig = {
    sm: { track: 'h-1', thumb: 'w-3 h-3' },
    md: { track: 'h-2', thumb: 'w-4 h-4' },
    lg: { track: 'h-3', thumb: 'w-5 h-5' },
  };

  const updateValue = useCallback(
    (clientX: number, thumb: 'min' | 'max') => {
      if (!trackRef.current || disabled) return;

      const rect = trackRef.current.getBoundingClientRect();
      let percentage = ((clientX - rect.left) / rect.width) * 100;
      percentage = Math.max(0, Math.min(100, percentage));

      let newValue = min + (percentage / 100) * (max - min);
      newValue = Math.round(newValue / step) * step;

      let newRange: [number, number];
      if (thumb === 'min') {
        newValue = Math.min(newValue, value[1] - minGap);
        newRange = [Math.max(min, newValue), value[1]];
      } else {
        newValue = Math.max(newValue, value[0] + minGap);
        newRange = [value[0], Math.min(max, newValue)];
      }

      if (!controlledValue) {
        setInternalValue(newRange);
      }
      onChange?.(newRange);
    },
    [controlledValue, disabled, max, min, minGap, onChange, step, value]
  );

  const handleMouseDown = (e: React.MouseEvent, thumb: 'min' | 'max') => {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();
    setActiveThumb(thumb);
    updateValue(e.clientX, thumb);
  };

  const handleTrackClick = (e: React.MouseEvent) => {
    if (disabled) return;
    const rect = trackRef.current?.getBoundingClientRect();
    if (!rect) return;

    const clickPercentage = ((e.clientX - rect.left) / rect.width) * 100;
    const clickValue = min + (clickPercentage / 100) * (max - min);

    // Determine which thumb is closer
    const distToMin = Math.abs(clickValue - value[0]);
    const distToMax = Math.abs(clickValue - value[1]);
    const thumb = distToMin < distToMax ? 'min' : 'max';

    updateValue(e.clientX, thumb);
  };

  useEffect(() => {
    if (!activeThumb) return;

    const handleMouseMove = (e: MouseEvent) => {
      updateValue(e.clientX, activeThumb);
    };

    const handleMouseUp = () => {
      setActiveThumb(null);
      onChangeEnd?.(value);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [activeThumb, onChangeEnd, updateValue, value]);

  return (
    <div className={cn('w-full', disabled && 'opacity-50', className)}>
      {label && (
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {label}
          </label>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {formatValue(value[0])} - {formatValue(value[1])}
          </span>
        </div>
      )}

      <div
        ref={trackRef}
        className={cn(
          'relative w-full cursor-pointer select-none',
          sizeConfig[size].track,
          disabled && 'cursor-not-allowed'
        )}
        onClick={handleTrackClick}
      >
        <div className="absolute w-full h-full rounded-full bg-gray-200 dark:bg-gray-700" />
        <div
          className="absolute h-full rounded-full bg-primary-600"
          style={{
            left: `${minPercentage}%`,
            width: `${maxPercentage - minPercentage}%`,
          }}
        />

        {/* Min thumb */}
        <div
          role="slider"
          tabIndex={disabled ? -1 : 0}
          aria-valuemin={min}
          aria-valuemax={value[1]}
          aria-valuenow={value[0]}
          className={cn(
            'absolute top-1/2 -translate-y-1/2 -translate-x-1/2 rounded-full bg-white border-2 border-primary-600 shadow-md cursor-grab',
            'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
            activeThumb === 'min' && 'scale-110 cursor-grabbing',
            sizeConfig[size].thumb
          )}
          style={{ left: `${minPercentage}%` }}
          onMouseDown={(e) => handleMouseDown(e, 'min')}
        />

        {/* Max thumb */}
        <div
          role="slider"
          tabIndex={disabled ? -1 : 0}
          aria-valuemin={value[0]}
          aria-valuemax={max}
          aria-valuenow={value[1]}
          className={cn(
            'absolute top-1/2 -translate-y-1/2 -translate-x-1/2 rounded-full bg-white border-2 border-primary-600 shadow-md cursor-grab',
            'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
            activeThumb === 'max' && 'scale-110 cursor-grabbing',
            sizeConfig[size].thumb
          )}
          style={{ left: `${maxPercentage}%` }}
          onMouseDown={(e) => handleMouseDown(e, 'max')}
        />
      </div>
    </div>
  );
}
