// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * CredibilityBadge - Visual indicator for credibility scores
 *
 * @example
 * ```tsx
 * <CredibilityBadge score={0.85} size="md" showLabel />
 * ```
 */

import React from 'react';

export interface CredibilityBadgeProps {
  /** Credibility score (0-1) */
  score: number;
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Show numeric label */
  showLabel?: boolean;
  /** Show descriptive text */
  showDescription?: boolean;
  /** Animate the ring */
  animated?: boolean;
  /** Custom className */
  className?: string;
  /** Inline styles */
  style?: React.CSSProperties;
}

const SIZES = {
  sm: { ring: 24, stroke: 3, fontSize: 10 },
  md: { ring: 40, stroke: 4, fontSize: 12 },
  lg: { ring: 64, stroke: 5, fontSize: 16 },
};

function getCredibilityColor(score: number): string {
  if (score >= 0.8) return '#22c55e';  // green
  if (score >= 0.6) return '#84cc16';  // lime
  if (score >= 0.4) return '#f59e0b';  // amber
  if (score >= 0.2) return '#f97316';  // orange
  return '#ef4444';                     // red
}

function getCredibilityLabel(score: number): string {
  if (score >= 0.9) return 'Excellent';
  if (score >= 0.8) return 'Very High';
  if (score >= 0.7) return 'High';
  if (score >= 0.6) return 'Good';
  if (score >= 0.5) return 'Moderate';
  if (score >= 0.4) return 'Fair';
  if (score >= 0.3) return 'Low';
  if (score >= 0.2) return 'Very Low';
  return 'Poor';
}

export function CredibilityBadge({
  score,
  size = 'md',
  showLabel = true,
  showDescription = false,
  animated = true,
  className,
  style,
}: CredibilityBadgeProps): JSX.Element {
  const { ring, stroke, fontSize } = SIZES[size];
  const radius = (ring - stroke) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (score * circumference);
  const color = getCredibilityColor(score);

  return (
    <div
      className={className}
      style={{
        display: 'inline-flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '4px',
        ...style,
      }}
    >
      <div style={{ position: 'relative', width: ring, height: ring }}>
        <svg
          width={ring}
          height={ring}
          style={{ transform: 'rotate(-90deg)' }}
        >
          {/* Background circle */}
          <circle
            cx={ring / 2}
            cy={ring / 2}
            r={radius}
            fill="none"
            stroke="#e5e7eb"
            strokeWidth={stroke}
          />
          {/* Progress circle */}
          <circle
            cx={ring / 2}
            cy={ring / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={stroke}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            style={{
              transition: animated ? 'stroke-dashoffset 0.5s ease-out' : 'none',
            }}
          />
        </svg>
        {showLabel && (
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              fontSize,
              fontWeight: 'bold',
              color: '#374151',
            }}
          >
            {Math.round(score * 100)}
          </div>
        )}
      </div>
      {showDescription && (
        <span
          style={{
            fontSize: fontSize - 2,
            color: '#6b7280',
            fontWeight: 500,
          }}
        >
          {getCredibilityLabel(score)}
        </span>
      )}
    </div>
  );
}

export default CredibilityBadge;
