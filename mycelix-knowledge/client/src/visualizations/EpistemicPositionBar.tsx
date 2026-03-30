// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * EpistemicPositionBar - Horizontal bar showing E-N-M distribution
 *
 * @example
 * ```tsx
 * <EpistemicPositionBar
 *   empirical={0.7}
 *   normative={0.2}
 *   mythic={0.1}
 *   showLabels
 * />
 * ```
 */

import React from 'react';

export interface EpistemicPositionBarProps {
  /** Empirical component (0-1) */
  empirical: number;
  /** Normative component (0-1) */
  normative: number;
  /** Mythic component (0-1) */
  mythic: number;
  /** Height of the bar */
  height?: number;
  /** Show axis labels */
  showLabels?: boolean;
  /** Show percentage values */
  showValues?: boolean;
  /** Animated transitions */
  animated?: boolean;
  /** Orientation */
  orientation?: 'horizontal' | 'vertical';
  /** Custom className */
  className?: string;
  /** Inline styles */
  style?: React.CSSProperties;
}

const COLORS = {
  empirical: '#ef4444',
  normative: '#22c55e',
  mythic: '#3b82f6',
};

export function EpistemicPositionBar({
  empirical,
  normative,
  mythic,
  height = 24,
  showLabels = false,
  showValues = false,
  animated = true,
  orientation = 'horizontal',
  className,
  style,
}: EpistemicPositionBarProps): JSX.Element {
  // Normalize to ensure they sum to 1
  const total = empirical + normative + mythic;
  const e = total > 0 ? empirical / total : 0.33;
  const n = total > 0 ? normative / total : 0.33;
  const m = total > 0 ? mythic / total : 0.34;

  const transition = animated ? 'all 0.3s ease-out' : 'none';

  if (orientation === 'vertical') {
    return (
      <div
        className={className}
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          ...style,
        }}
      >
        {/* Empirical */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {showLabels && (
            <span style={{ width: '60px', fontSize: '12px', color: COLORS.empirical, fontWeight: 500 }}>
              Empirical
            </span>
          )}
          <div
            style={{
              flex: 1,
              height,
              backgroundColor: '#fee2e2',
              borderRadius: '4px',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${e * 100}%`,
                height: '100%',
                backgroundColor: COLORS.empirical,
                transition,
              }}
            />
          </div>
          {showValues && (
            <span style={{ minWidth: '36px', fontSize: '12px', color: '#6b7280', textAlign: 'right' }}>
              {Math.round(e * 100)}%
            </span>
          )}
        </div>

        {/* Normative */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {showLabels && (
            <span style={{ width: '60px', fontSize: '12px', color: COLORS.normative, fontWeight: 500 }}>
              Normative
            </span>
          )}
          <div
            style={{
              flex: 1,
              height,
              backgroundColor: '#dcfce7',
              borderRadius: '4px',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${n * 100}%`,
                height: '100%',
                backgroundColor: COLORS.normative,
                transition,
              }}
            />
          </div>
          {showValues && (
            <span style={{ minWidth: '36px', fontSize: '12px', color: '#6b7280', textAlign: 'right' }}>
              {Math.round(n * 100)}%
            </span>
          )}
        </div>

        {/* Mythic */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {showLabels && (
            <span style={{ width: '60px', fontSize: '12px', color: COLORS.mythic, fontWeight: 500 }}>
              Mythic
            </span>
          )}
          <div
            style={{
              flex: 1,
              height,
              backgroundColor: '#dbeafe',
              borderRadius: '4px',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${m * 100}%`,
                height: '100%',
                backgroundColor: COLORS.mythic,
                transition,
              }}
            />
          </div>
          {showValues && (
            <span style={{ minWidth: '36px', fontSize: '12px', color: '#6b7280', textAlign: 'right' }}>
              {Math.round(m * 100)}%
            </span>
          )}
        </div>
      </div>
    );
  }

  // Horizontal stacked bar
  return (
    <div
      className={className}
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
        ...style,
      }}
    >
      {showLabels && (
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px' }}>
          <span style={{ color: COLORS.empirical }}>E</span>
          <span style={{ color: COLORS.normative }}>N</span>
          <span style={{ color: COLORS.mythic }}>M</span>
        </div>
      )}
      <div
        style={{
          display: 'flex',
          height,
          borderRadius: '4px',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${e * 100}%`,
            backgroundColor: COLORS.empirical,
            transition,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {showValues && e > 0.15 && (
            <span style={{ color: '#fff', fontSize: '10px', fontWeight: 'bold' }}>
              {Math.round(e * 100)}%
            </span>
          )}
        </div>
        <div
          style={{
            width: `${n * 100}%`,
            backgroundColor: COLORS.normative,
            transition,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {showValues && n > 0.15 && (
            <span style={{ color: '#fff', fontSize: '10px', fontWeight: 'bold' }}>
              {Math.round(n * 100)}%
            </span>
          )}
        </div>
        <div
          style={{
            width: `${m * 100}%`,
            backgroundColor: COLORS.mythic,
            transition,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {showValues && m > 0.15 && (
            <span style={{ color: '#fff', fontSize: '10px', fontWeight: 'bold' }}>
              {Math.round(m * 100)}%
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default EpistemicPositionBar;
