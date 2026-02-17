/**
 * AssuranceLevelBadge - Visual badge showing identity assurance level
 *
 * Displays the current assurance level with color coding and optional
 * score percentage. Aligned with the MFDI Epistemic E-Axis levels.
 *
 * @module frameworks/react/components/AssuranceLevelBadge
 *
 * @example Basic usage
 * ```tsx
 * import { AssuranceLevelBadge } from '@mycelix/sdk/react';
 *
 * function Profile() {
 *   return (
 *     <div>
 *       <h1>My Profile</h1>
 *       <AssuranceLevelBadge />
 *     </div>
 *   );
 * }
 * ```
 *
 * @example With explicit DID
 * ```tsx
 * <AssuranceLevelBadge did="did:mycelix:abc123" showScore />
 * ```
 *
 * @example Custom styling
 * ```tsx
 * <AssuranceLevelBadge
 *   size="large"
 *   showLabel
 *   showScore
 *   className="custom-badge"
 * />
 * ```
 */

import * as React from 'react';

import { useAssuranceLevel } from '../hooks.js';

import type { AssuranceLevel, DID } from '../hooks.js';

// =============================================================================
// Types
// =============================================================================

/**
 * Badge size variants
 */
export type BadgeSize = 'small' | 'medium' | 'large';

/**
 * Props for AssuranceLevelBadge component
 */
export interface AssuranceLevelBadgeProps {
  /** DID to show assurance for (uses current user if not provided) */
  did?: DID;
  /** Whether to show the score percentage */
  showScore?: boolean;
  /** Whether to show the level label text */
  showLabel?: boolean;
  /** Badge size variant */
  size?: BadgeSize;
  /** CSS class for the badge container */
  className?: string;
  /** Whether to show a loading skeleton while fetching */
  showSkeleton?: boolean;
  /** Callback when badge is clicked */
  onClick?: () => void;
  /** Whether the badge should be interactive (button) */
  interactive?: boolean;
}

// =============================================================================
// Constants
// =============================================================================

/**
 * Level configuration for display
 */
interface LevelConfig {
  label: string;
  shortLabel: string;
  description: string;
  colorClass: string;
}

const LEVEL_CONFIGS: Record<AssuranceLevel, LevelConfig> = {
  Anonymous: {
    label: 'Anonymous',
    shortLabel: 'Anon',
    description: 'No verified identity factors',
    colorClass: 'mfa-badge-anonymous',
  },
  Basic: {
    label: 'Basic',
    shortLabel: 'Basic',
    description: 'Primary key pair established',
    colorClass: 'mfa-badge-basic',
  },
  Verified: {
    label: 'Verified',
    shortLabel: 'Verified',
    description: 'Multiple factors with external verification',
    colorClass: 'mfa-badge-verified',
  },
  HighlyAssured: {
    label: 'Highly Assured',
    shortLabel: 'High',
    description: 'Strong multi-factor with biometrics or hardware',
    colorClass: 'mfa-badge-highly-assured',
  },
  ConstitutionallyCritical: {
    label: 'Constitutionally Critical',
    shortLabel: 'Critical',
    description: 'Maximum assurance for critical operations',
    colorClass: 'mfa-badge-critical',
  },
};

/**
 * Get score color based on percentage
 */
function getScoreColorClass(score: number): string {
  if (score >= 0.8) return 'mfa-score-excellent';
  if (score >= 0.6) return 'mfa-score-good';
  if (score >= 0.4) return 'mfa-score-fair';
  if (score >= 0.2) return 'mfa-score-low';
  return 'mfa-score-minimal';
}

// =============================================================================
// Sub-components
// =============================================================================

interface ScoreRingProps {
  score: number;
  size: BadgeSize;
}

function ScoreRing({ score, size }: ScoreRingProps) {
  const strokeWidth = size === 'small' ? 2 : size === 'large' ? 4 : 3;
  const radius = size === 'small' ? 10 : size === 'large' ? 20 : 14;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score * circumference);

  const svgSize = (radius + strokeWidth) * 2;

  return (
    <svg
      className="mfa-score-ring"
      width={svgSize}
      height={svgSize}
      viewBox={`0 0 ${svgSize} ${svgSize}`}
      aria-hidden="true"
    >
      {/* Background circle */}
      <circle
        className="mfa-score-ring-bg"
        cx={svgSize / 2}
        cy={svgSize / 2}
        r={radius}
        strokeWidth={strokeWidth}
        fill="none"
      />
      {/* Progress circle */}
      <circle
        className={`mfa-score-ring-progress ${getScoreColorClass(score)}`}
        cx={svgSize / 2}
        cy={svgSize / 2}
        r={radius}
        strokeWidth={strokeWidth}
        fill="none"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        transform={`rotate(-90 ${svgSize / 2} ${svgSize / 2})`}
      />
    </svg>
  );
}

interface BadgeSkeletonProps {
  size: BadgeSize;
  showLabel: boolean;
  showScore: boolean;
}

function BadgeSkeleton({ size, showLabel, showScore }: BadgeSkeletonProps) {
  return (
    <div
      className={`mfa-badge mfa-badge-skeleton mfa-badge-${size}`}
      aria-busy="true"
      aria-label="Loading assurance level"
    >
      <span className="mfa-badge-skeleton-icon" />
      {showLabel && <span className="mfa-badge-skeleton-label" />}
      {showScore && <span className="mfa-badge-skeleton-score" />}
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * AssuranceLevelBadge - Visual badge showing identity assurance level
 *
 * Displays the current assurance level color-coded by level:
 * - Anonymous: Gray
 * - Basic: Blue
 * - Verified: Green
 * - Highly Assured: Gold
 * - Constitutionally Critical: Purple
 */
export function AssuranceLevelBadge({
  did,
  showScore = false,
  showLabel = true,
  size = 'medium',
  className = '',
  showSkeleton = true,
  onClick,
  interactive = false,
}: AssuranceLevelBadgeProps): React.ReactElement {
  const { level, score, loading, error } = useAssuranceLevel(did);

  // Determine which element to render
  const Element = interactive || onClick ? 'button' : 'div';

  // Handle loading state
  if (loading && showSkeleton) {
    return <BadgeSkeleton size={size} showLabel={showLabel} showScore={showScore} />;
  }

  // Handle error or no level
  if (error || !level) {
    const config = LEVEL_CONFIGS.Anonymous;

    return (
      <Element
        className={`mfa-badge mfa-badge-${size} ${config.colorClass} ${className}`}
        onClick={onClick}
        type={Element === 'button' ? 'button' : undefined}
        aria-label={error ? 'Error loading assurance level' : 'Anonymous assurance level'}
      >
        <span className="mfa-badge-icon" aria-hidden="true">
          [?]
        </span>
        {showLabel && (
          <span className="mfa-badge-label">
            {error ? 'Error' : config.shortLabel}
          </span>
        )}
        {showScore && (
          <span className="mfa-badge-score">0%</span>
        )}
      </Element>
    );
  }

  const config = LEVEL_CONFIGS[level];
  const percentage = Math.round(score * 100);

  return (
    <Element
      className={`mfa-badge mfa-badge-${size} ${config.colorClass} ${className}`}
      onClick={onClick}
      type={Element === 'button' ? 'button' : undefined}
      title={config.description}
      aria-label={`Assurance level: ${config.label}${showScore ? `, ${percentage}% score` : ''}`}
    >
      {showScore && (
        <ScoreRing score={score} size={size} />
      )}

      <span className="mfa-badge-icon" aria-hidden="true">
        {level === 'Anonymous' && '[?]'}
        {level === 'Basic' && '[key]'}
        {level === 'Verified' && '[check]'}
        {level === 'HighlyAssured' && '[shield]'}
        {level === 'ConstitutionallyCritical' && '[star]'}
      </span>

      {showLabel && (
        <span className="mfa-badge-label">
          {size === 'small' ? config.shortLabel : config.label}
        </span>
      )}

      {showScore && (
        <span className="mfa-badge-score" aria-hidden="true">
          {percentage}%
        </span>
      )}
    </Element>
  );
}

/**
 * AssuranceLevelIndicator - Inline indicator for assurance level
 *
 * A more compact version that just shows the icon and color.
 */
export interface AssuranceLevelIndicatorProps {
  /** The assurance level to display */
  level: AssuranceLevel;
  /** Optional score (0-1) */
  score?: number;
  /** CSS class for the indicator */
  className?: string;
}

export function AssuranceLevelIndicator({
  level,
  score,
  className = '',
}: AssuranceLevelIndicatorProps): React.ReactElement {
  const config = LEVEL_CONFIGS[level];
  const percentage = score !== undefined ? Math.round(score * 100) : null;

  return (
    <span
      className={`mfa-indicator ${config.colorClass} ${className}`}
      title={`${config.label}${percentage !== null ? ` (${percentage}%)` : ''}`}
      role="img"
      aria-label={config.label}
    >
      <span className="mfa-indicator-dot" aria-hidden="true" />
      {percentage !== null && (
        <span className="mfa-indicator-score">{percentage}%</span>
      )}
    </span>
  );
}

export default AssuranceLevelBadge;
