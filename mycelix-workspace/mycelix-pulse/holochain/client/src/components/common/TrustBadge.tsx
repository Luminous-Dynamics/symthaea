// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * TrustBadge Component
 *
 * Displays trust level with visual indicator and tooltip.
 */

import React, { useMemo } from 'react';

export interface TrustBadgeProps {
  /** Trust level from 0.0 to 1.0 */
  level: number;
  /** Number of attestations */
  attestationCount?: number;
  /** Whether to show the numeric value */
  showValue?: boolean;
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Custom className */
  className?: string;
  /** Click handler */
  onClick?: () => void;
}

interface TrustTier {
  name: string;
  color: string;
  bgColor: string;
  minLevel: number;
}

const TRUST_TIERS: TrustTier[] = [
  { name: 'Unknown', color: '#6b7280', bgColor: '#f3f4f6', minLevel: 0 },
  { name: 'Low', color: '#ef4444', bgColor: '#fef2f2', minLevel: 0.2 },
  { name: 'Medium', color: '#f59e0b', bgColor: '#fffbeb', minLevel: 0.4 },
  { name: 'Good', color: '#10b981', bgColor: '#ecfdf5', minLevel: 0.6 },
  { name: 'High', color: '#3b82f6', bgColor: '#eff6ff', minLevel: 0.8 },
  { name: 'Verified', color: '#8b5cf6', bgColor: '#f5f3ff', minLevel: 0.95 },
];

const SIZE_CLASSES = {
  sm: 'text-xs px-1.5 py-0.5',
  md: 'text-sm px-2 py-1',
  lg: 'text-base px-3 py-1.5',
};

export const TrustBadge: React.FC<TrustBadgeProps> = ({
  level,
  attestationCount,
  showValue = false,
  size = 'md',
  className = '',
  onClick,
}) => {
  const tier = useMemo(() => {
    const normalizedLevel = Math.max(0, Math.min(1, level));
    let currentTier = TRUST_TIERS[0];

    for (const t of TRUST_TIERS) {
      if (normalizedLevel >= t.minLevel) {
        currentTier = t;
      }
    }

    return currentTier;
  }, [level]);

  const percentage = Math.round(level * 100);

  return (
    <span
      className={`
        inline-flex items-center gap-1 rounded-full font-medium
        ${SIZE_CLASSES[size]}
        ${onClick ? 'cursor-pointer hover:opacity-80' : ''}
        ${className}
      `}
      style={{
        color: tier.color,
        backgroundColor: tier.bgColor,
      }}
      onClick={onClick}
      title={`Trust: ${tier.name} (${percentage}%)${
        attestationCount !== undefined ? ` - ${attestationCount} attestations` : ''
      }`}
    >
      <TrustIcon tier={tier} size={size} />
      <span>{tier.name}</span>
      {showValue && <span className="opacity-70">({percentage}%)</span>}
      {attestationCount !== undefined && (
        <span className="opacity-50 text-xs">
          ({attestationCount})
        </span>
      )}
    </span>
  );
};

const TrustIcon: React.FC<{ tier: TrustTier; size: string }> = ({ tier, size }) => {
  const iconSize = size === 'sm' ? 12 : size === 'md' ? 14 : 16;

  // Shield icon with fill based on trust level
  return (
    <svg
      width={iconSize}
      height={iconSize}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      {tier.minLevel >= 0.4 && (
        <path d="M9 12l2 2 4-4" stroke={tier.color} />
      )}
    </svg>
  );
};

export default TrustBadge;
