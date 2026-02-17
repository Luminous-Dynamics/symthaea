/**
 * FlEligibilityBanner - Banner showing Federated Learning eligibility status
 *
 * Displays whether the user is eligible for FL participation, with details
 * on met and unmet requirements.
 *
 * @module frameworks/react/components/FlEligibilityBanner
 *
 * @example
 * ```tsx
 * import { FlEligibilityBanner } from '@mycelix/sdk/react';
 *
 * function Dashboard() {
 *   return (
 *     <div>
 *       <FlEligibilityBanner
 *         onEnrollClick={() => navigate('/security/enroll')}
 *       />
 *       <RestOfDashboard />
 *     </div>
 *   );
 * }
 * ```
 */

import * as React from 'react';

import { useFlEligibility } from '../hooks.js';

import type { DID, FlRequirement } from '../hooks.js';

// =============================================================================
// Types
// =============================================================================

/**
 * Banner variant styles
 */
export type BannerVariant = 'inline' | 'card' | 'full-width';

/**
 * Props for FlEligibilityBanner component
 */
export interface FlEligibilityBannerProps {
  /** DID to check eligibility for (uses current user if not provided) */
  did?: DID;
  /** Banner display variant */
  variant?: BannerVariant;
  /** Whether to show requirement details */
  showDetails?: boolean;
  /** Whether to show a progress indicator */
  showProgress?: boolean;
  /** Callback when user clicks to enroll more factors */
  onEnrollClick?: () => void;
  /** Callback when user clicks to learn more */
  onLearnMoreClick?: () => void;
  /** Whether the banner can be dismissed */
  dismissible?: boolean;
  /** Callback when banner is dismissed */
  onDismiss?: () => void;
  /** CSS class for the banner container */
  className?: string;
  /** Custom title for eligible state */
  eligibleTitle?: string;
  /** Custom title for ineligible state */
  ineligibleTitle?: string;
}

// =============================================================================
// Helper Functions
// =============================================================================

function getRequirementIcon(name: string, met: boolean): string {
  if (met) {
    return '[check]';
  }

  if (name.toLowerCase().includes('crypto')) return '[key]';
  if (name.toLowerCase().includes('external') || name.toLowerCase().includes('gitcoin')) return '[passport]';
  if (name.toLowerCase().includes('assurance')) return '[shield]';
  if (name.toLowerCase().includes('biometric')) return '[fingerprint]';

  return '[circle]';
}

// =============================================================================
// Sub-components
// =============================================================================

interface ProgressBarProps {
  currentScore: number;
  requiredScore: number;
}

function ProgressBar({ currentScore, requiredScore }: ProgressBarProps) {
  const percentage = Math.min(100, Math.round((currentScore / requiredScore) * 100));
  const remaining = Math.max(0, requiredScore - currentScore);

  return (
    <div className="fl-progress">
      <div className="fl-progress-bar" role="progressbar" aria-valuenow={percentage} aria-valuemin={0} aria-valuemax={100}>
        <div
          className="fl-progress-fill"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <div className="fl-progress-labels">
        <span className="fl-progress-current">
          Score: {(currentScore * 100).toFixed(0)}%
        </span>
        <span className="fl-progress-required">
          {percentage >= 100 ? 'Eligible!' : `${(remaining * 100).toFixed(0)}% more needed`}
        </span>
      </div>
    </div>
  );
}

interface RequirementListProps {
  requirements: FlRequirement[];
  compact?: boolean;
}

function RequirementList({ requirements, compact = false }: RequirementListProps) {
  const metRequirements = requirements.filter((r) => r.met);
  const unmetRequirements = requirements.filter((r) => !r.met);

  if (compact) {
    return (
      <ul className="fl-requirements-compact" role="list">
        {unmetRequirements.map((req, index) => (
          <li key={index} className="fl-requirement-item fl-requirement-unmet">
            <span className="fl-requirement-icon" aria-hidden="true">
              {getRequirementIcon(req.name, false)}
            </span>
            <span className="fl-requirement-name">{req.name}</span>
          </li>
        ))}
      </ul>
    );
  }

  return (
    <div className="fl-requirements">
      {unmetRequirements.length > 0 && (
        <div className="fl-requirements-section">
          <h4 className="fl-requirements-heading">Required:</h4>
          <ul className="fl-requirements-list" role="list">
            {unmetRequirements.map((req, index) => (
              <li key={index} className="fl-requirement-item fl-requirement-unmet">
                <span className="fl-requirement-icon" aria-hidden="true">
                  {getRequirementIcon(req.name, false)}
                </span>
                <div className="fl-requirement-content">
                  <span className="fl-requirement-name">{req.name}</span>
                  <span className="fl-requirement-description">{req.description}</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {metRequirements.length > 0 && (
        <div className="fl-requirements-section">
          <h4 className="fl-requirements-heading">Completed:</h4>
          <ul className="fl-requirements-list" role="list">
            {metRequirements.map((req, index) => (
              <li key={index} className="fl-requirement-item fl-requirement-met">
                <span className="fl-requirement-icon" aria-hidden="true">
                  [check]
                </span>
                <div className="fl-requirement-content">
                  <span className="fl-requirement-name">{req.name}</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

interface BannerSkeletonProps {
  variant: BannerVariant;
}

function BannerSkeleton({ variant }: BannerSkeletonProps) {
  return (
    <div
      className={`fl-banner fl-banner-${variant} fl-banner-skeleton`}
      aria-busy="true"
      aria-label="Loading FL eligibility status"
    >
      <div className="fl-banner-skeleton-icon" />
      <div className="fl-banner-skeleton-content">
        <div className="fl-banner-skeleton-title" />
        <div className="fl-banner-skeleton-text" />
      </div>
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * FlEligibilityBanner - Banner showing Federated Learning eligibility status
 *
 * Shows green if eligible, yellow/red if not, with unmet requirements listed.
 */
export function FlEligibilityBanner({
  did,
  variant = 'card',
  showDetails = true,
  showProgress = true,
  onEnrollClick,
  onLearnMoreClick,
  dismissible = false,
  onDismiss,
  className = '',
  eligibleTitle = 'Eligible for Federated Learning',
  ineligibleTitle = 'Not Yet Eligible for Federated Learning',
}: FlEligibilityBannerProps): React.ReactElement | null {
  const { eligibility, loading, error, isEligible, requirements, unmetRequirements } = useFlEligibility(did);
  const [isDismissed, setIsDismissed] = React.useState(false);

  const handleDismiss = () => {
    setIsDismissed(true);
    onDismiss?.();
  };

  // Don't render if dismissed
  if (isDismissed) {
    return null;
  }

  // Loading state
  if (loading) {
    return <BannerSkeleton variant={variant} />;
  }

  // Error state
  if (error) {
    return (
      <div
        className={`fl-banner fl-banner-${variant} fl-banner-error ${className}`}
        role="alert"
      >
        <span className="fl-banner-icon" aria-hidden="true">
          [error]
        </span>
        <div className="fl-banner-content">
          <h3 className="fl-banner-title">Unable to check FL eligibility</h3>
          <p className="fl-banner-message">{error.message}</p>
        </div>
        {dismissible && (
          <button
            type="button"
            className="fl-banner-dismiss"
            onClick={handleDismiss}
            aria-label="Dismiss"
          >
            &times;
          </button>
        )}
      </div>
    );
  }

  // Eligible state
  if (isEligible) {
    return (
      <div
        className={`fl-banner fl-banner-${variant} fl-banner-eligible ${className}`}
        role="status"
      >
        <span className="fl-banner-icon" aria-hidden="true">
          [check-circle]
        </span>
        <div className="fl-banner-content">
          <h3 className="fl-banner-title">{eligibleTitle}</h3>
          <p className="fl-banner-message">
            You meet all requirements to participate in federated learning and contribute to collective intelligence.
          </p>
          {showDetails && requirements.length > 0 && (
            <details className="fl-banner-details">
              <summary>View requirements ({requirements.length} met)</summary>
              <RequirementList requirements={requirements} />
            </details>
          )}
        </div>
        {onLearnMoreClick && (
          <button
            type="button"
            className="fl-banner-action mfa-button mfa-button-secondary"
            onClick={onLearnMoreClick}
          >
            Learn More
          </button>
        )}
        {dismissible && (
          <button
            type="button"
            className="fl-banner-dismiss"
            onClick={handleDismiss}
            aria-label="Dismiss"
          >
            &times;
          </button>
        )}
      </div>
    );
  }

  // Ineligible state
  const severity = unmetRequirements.length <= 1 ? 'warning' : 'error';

  return (
    <div
      className={`fl-banner fl-banner-${variant} fl-banner-ineligible fl-banner-${severity} ${className}`}
      role="alert"
    >
      <span className="fl-banner-icon" aria-hidden="true">
        {severity === 'warning' ? '[warning]' : '[x-circle]'}
      </span>
      <div className="fl-banner-content">
        <h3 className="fl-banner-title">{ineligibleTitle}</h3>
        <p className="fl-banner-message">
          {unmetRequirements.length === 1
            ? 'You need to complete one more requirement to participate in federated learning.'
            : `You need to complete ${unmetRequirements.length} requirements to participate in federated learning.`}
        </p>

        {showProgress && eligibility && (
          <ProgressBar
            currentScore={eligibility.currentScore}
            requiredScore={eligibility.requiredScore}
          />
        )}

        {showDetails && (
          <RequirementList requirements={requirements} compact={variant === 'inline'} />
        )}
      </div>

      <div className="fl-banner-actions">
        {onEnrollClick && (
          <button
            type="button"
            className="fl-banner-action mfa-button mfa-button-primary"
            onClick={onEnrollClick}
          >
            Enroll Factors
          </button>
        )}
        {onLearnMoreClick && (
          <button
            type="button"
            className="fl-banner-action mfa-button mfa-button-text"
            onClick={onLearnMoreClick}
          >
            Learn More
          </button>
        )}
      </div>

      {dismissible && (
        <button
          type="button"
          className="fl-banner-dismiss"
          onClick={handleDismiss}
          aria-label="Dismiss"
        >
          &times;
        </button>
      )}
    </div>
  );
}

/**
 * FlEligibilityIndicator - Compact indicator for FL eligibility
 *
 * A minimal version that just shows eligible/ineligible status.
 */
export interface FlEligibilityIndicatorProps {
  /** DID to check eligibility for */
  did?: DID;
  /** CSS class for the indicator */
  className?: string;
  /** Callback when clicked */
  onClick?: () => void;
}

export function FlEligibilityIndicator({
  did,
  className = '',
  onClick,
}: FlEligibilityIndicatorProps): React.ReactElement {
  const { isEligible, loading, unmetRequirements } = useFlEligibility(did);

  const Element = onClick ? 'button' : 'span';

  if (loading) {
    return (
      <Element
        className={`fl-indicator fl-indicator-loading ${className}`}
        aria-busy="true"
        aria-label="Checking FL eligibility"
        onClick={onClick}
        type={Element === 'button' ? 'button' : undefined}
      >
        <span className="fl-indicator-dot" />
        <span className="fl-indicator-text">...</span>
      </Element>
    );
  }

  return (
    <Element
      className={`fl-indicator ${isEligible ? 'fl-indicator-eligible' : 'fl-indicator-ineligible'} ${className}`}
      title={
        isEligible
          ? 'Eligible for Federated Learning'
          : `Not eligible: ${unmetRequirements.length} requirement(s) remaining`
      }
      aria-label={
        isEligible
          ? 'Eligible for Federated Learning'
          : `Not eligible for Federated Learning, ${unmetRequirements.length} requirements remaining`
      }
      onClick={onClick}
      type={Element === 'button' ? 'button' : undefined}
    >
      <span className="fl-indicator-dot" aria-hidden="true" />
      <span className="fl-indicator-text">
        {isEligible ? 'FL Ready' : 'FL Pending'}
      </span>
    </Element>
  );
}

export default FlEligibilityBanner;
