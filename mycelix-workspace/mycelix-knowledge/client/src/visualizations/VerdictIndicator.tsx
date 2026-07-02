// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * VerdictIndicator - Visual display for fact-check verdicts
 *
 * @example
 * ```tsx
 * <VerdictIndicator verdict="MostlyTrue" confidence={0.85} />
 * ```
 */

import React from 'react';

export type Verdict =
  | 'True'
  | 'MostlyTrue'
  | 'Mixed'
  | 'MostlyFalse'
  | 'False'
  | 'Unverifiable'
  | 'InsufficientEvidence';

export interface VerdictIndicatorProps {
  /** The fact-check verdict */
  verdict: Verdict;
  /** Confidence level (0-1) */
  confidence?: number;
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Show confidence bar */
  showConfidence?: boolean;
  /** Compact mode (icon only) */
  compact?: boolean;
  /** Custom className */
  className?: string;
  /** Inline styles */
  style?: React.CSSProperties;
}

const VERDICT_CONFIG: Record<Verdict, { color: string; icon: string; label: string; bg: string }> = {
  True: {
    color: '#22c55e',
    icon: '✓',
    label: 'True',
    bg: '#dcfce7',
  },
  MostlyTrue: {
    color: '#84cc16',
    icon: '◐',
    label: 'Mostly True',
    bg: '#ecfccb',
  },
  Mixed: {
    color: '#f59e0b',
    icon: '◑',
    label: 'Mixed',
    bg: '#fef3c7',
  },
  MostlyFalse: {
    color: '#f97316',
    icon: '◔',
    label: 'Mostly False',
    bg: '#ffedd5',
  },
  False: {
    color: '#ef4444',
    icon: '✗',
    label: 'False',
    bg: '#fef2f2',
  },
  Unverifiable: {
    color: '#6b7280',
    icon: '?',
    label: 'Unverifiable',
    bg: '#f3f4f6',
  },
  InsufficientEvidence: {
    color: '#94a3b8',
    icon: '…',
    label: 'Insufficient Evidence',
    bg: '#f8fafc',
  },
};

const SIZES = {
  sm: { padding: '4px 8px', fontSize: 12, iconSize: 14, gap: 4 },
  md: { padding: '6px 12px', fontSize: 14, iconSize: 18, gap: 6 },
  lg: { padding: '8px 16px', fontSize: 16, iconSize: 22, gap: 8 },
};

export function VerdictIndicator({
  verdict,
  confidence,
  size = 'md',
  showConfidence = true,
  compact = false,
  className,
  style,
}: VerdictIndicatorProps): JSX.Element {
  const config = VERDICT_CONFIG[verdict];
  const sizeConfig = SIZES[size];

  if (compact) {
    return (
      <div
        className={className}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: sizeConfig.iconSize * 1.8,
          height: sizeConfig.iconSize * 1.8,
          borderRadius: '50%',
          backgroundColor: config.bg,
          color: config.color,
          fontSize: sizeConfig.iconSize,
          fontWeight: 'bold',
          ...style,
        }}
        title={`${config.label}${confidence ? ` (${Math.round(confidence * 100)}% confidence)` : ''}`}
      >
        {config.icon}
      </div>
    );
  }

  return (
    <div
      className={className}
      style={{
        display: 'inline-flex',
        flexDirection: 'column',
        gap: sizeConfig.gap,
        ...style,
      }}
    >
      <div
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: sizeConfig.gap,
          padding: sizeConfig.padding,
          borderRadius: '6px',
          backgroundColor: config.bg,
          border: `1px solid ${config.color}`,
        }}
      >
        <span
          style={{
            fontSize: sizeConfig.iconSize,
            color: config.color,
            fontWeight: 'bold',
          }}
        >
          {config.icon}
        </span>
        <span
          style={{
            fontSize: sizeConfig.fontSize,
            fontWeight: 600,
            color: config.color,
          }}
        >
          {config.label}
        </span>
      </div>

      {showConfidence && confidence !== undefined && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: sizeConfig.gap,
          }}
        >
          <div
            style={{
              flex: 1,
              height: 4,
              backgroundColor: '#e5e7eb',
              borderRadius: 2,
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${confidence * 100}%`,
                height: '100%',
                backgroundColor: config.color,
                transition: 'width 0.3s ease-out',
              }}
            />
          </div>
          <span
            style={{
              fontSize: sizeConfig.fontSize - 2,
              color: '#6b7280',
              minWidth: '36px',
            }}
          >
            {Math.round(confidence * 100)}%
          </span>
        </div>
      )}
    </div>
  );
}

export default VerdictIndicator;
