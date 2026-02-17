/**
 * FactorList - List of enrolled MFA factors with actions
 *
 * Displays all enrolled factors with their type, status, last verification time,
 * and provides actions to verify or remove factors.
 *
 * @module frameworks/react/components/FactorList
 *
 * @example
 * ```tsx
 * import { FactorList } from '@mycelix/sdk/react';
 *
 * function SecuritySettings() {
 *   return (
 *     <div>
 *       <h2>Your Security Factors</h2>
 *       <FactorList
 *         onVerify={(factor) => console.log('Verify', factor)}
 *         onRemove={(factor) => console.log('Remove', factor)}
 *         onEnrollNew={() => navigate('/enroll')}
 *       />
 *     </div>
 *   );
 * }
 * ```
 */

import * as React from 'react';

import { useMfaState, useMfaEnroll } from '../hooks.js';

import type { DID, EnrolledFactor, FactorType } from '../hooks.js';

// =============================================================================
// Types
// =============================================================================

/**
 * Props for FactorList component
 */
export interface FactorListProps {
  /** DID to show factors for (uses current user if not provided) */
  did?: DID;
  /** Callback when user clicks verify on a factor */
  onVerify?: (factor: EnrolledFactor) => void;
  /** Callback when user clicks remove on a factor */
  onRemove?: (factor: EnrolledFactor) => void;
  /** Callback when user clicks to enroll a new factor */
  onEnrollNew?: () => void;
  /** Whether to show the "Add Factor" button */
  showAddButton?: boolean;
  /** Whether factors can be removed */
  allowRemove?: boolean;
  /** Whether factors can be verified */
  allowVerify?: boolean;
  /** Factor types to show (all if not specified) */
  filterTypes?: FactorType[];
  /** CSS class for the list container */
  className?: string;
  /** Empty state message */
  emptyMessage?: string;
  /** Whether to group factors by category */
  groupByCategory?: boolean;
}

// =============================================================================
// Constants
// =============================================================================

interface FactorTypeInfo {
  label: string;
  icon: string;
  category: string;
}

const FACTOR_TYPE_INFO: Record<FactorType, FactorTypeInfo> = {
  PrimaryKeyPair: { label: 'Primary Key Pair', icon: '[key]', category: 'Cryptographic' },
  HardwareKey: { label: 'Hardware Key', icon: '[usb]', category: 'Cryptographic' },
  Biometric: { label: 'Biometric', icon: '[fingerprint]', category: 'Biometric' },
  SocialRecovery: { label: 'Social Recovery', icon: '[users]', category: 'Social Proof' },
  ReputationAttestation: { label: 'Reputation', icon: '[star]', category: 'Social Proof' },
  GitcoinPassport: { label: 'Gitcoin Passport', icon: '[passport]', category: 'External Verification' },
  VerifiableCredential: { label: 'Verifiable Credential', icon: '[certificate]', category: 'External Verification' },
  RecoveryPhrase: { label: 'Recovery Phrase', icon: '[note]', category: 'Knowledge' },
  SecurityQuestions: { label: 'Security Questions', icon: '[question]', category: 'Knowledge' },
};

// =============================================================================
// Helper Functions
// =============================================================================

function formatTimeAgo(timestamp: number): string {
  const now = Date.now();
  const seconds = Math.floor((now - timestamp / 1000) / 1000);

  if (seconds < 60) return 'Just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;

  return new Date(timestamp / 1000).toLocaleDateString();
}

function getStrengthLevel(strength: number): 'low' | 'medium' | 'high' | 'excellent' {
  if (strength >= 0.9) return 'excellent';
  if (strength >= 0.7) return 'high';
  if (strength >= 0.4) return 'medium';
  return 'low';
}

function groupFactorsByCategory(factors: EnrolledFactor[]): Map<string, EnrolledFactor[]> {
  const grouped = new Map<string, EnrolledFactor[]>();

  for (const factor of factors) {
    const info = FACTOR_TYPE_INFO[factor.factorType];
    const category = info?.category || 'Other';

    if (!grouped.has(category)) {
      grouped.set(category, []);
    }
    grouped.get(category)!.push(factor);
  }

  return grouped;
}

// =============================================================================
// Sub-components
// =============================================================================

interface FactorCardProps {
  factor: EnrolledFactor;
  onVerify?: () => void;
  onRemove?: () => void;
  allowVerify: boolean;
  allowRemove: boolean;
  isRemoving: boolean;
}

function FactorCard({
  factor,
  onVerify,
  onRemove,
  allowVerify,
  allowRemove,
  isRemoving,
}: FactorCardProps) {
  const info = FACTOR_TYPE_INFO[factor.factorType] || {
    label: factor.factorType,
    icon: '[?]',
    category: 'Unknown',
  };

  const strengthLevel = getStrengthLevel(factor.effectiveStrength);
  const strengthPercent = Math.round(factor.effectiveStrength * 100);
  const isActive = factor.active;
  const needsVerification = factor.effectiveStrength < 0.5;

  return (
    <article
      className={`factor-card ${isActive ? 'factor-card-active' : 'factor-card-inactive'}`}
      aria-labelledby={`factor-${factor.factorId}-label`}
    >
      <div className="factor-card-header">
        <span className="factor-icon" aria-hidden="true">
          {info.icon}
        </span>
        <div className="factor-info">
          <h4 id={`factor-${factor.factorId}-label`} className="factor-label">
            {info.label}
          </h4>
          <span className="factor-id" title={factor.factorId}>
            {factor.factorId.slice(0, 8)}...
          </span>
        </div>
        <span className={`factor-status ${isActive ? 'factor-status-active' : 'factor-status-inactive'}`}>
          {isActive ? 'Active' : 'Inactive'}
        </span>
      </div>

      <div className="factor-card-body">
        <div className="factor-strength">
          <span className="factor-strength-label">Strength:</span>
          <div className="factor-strength-bar" role="meter" aria-valuenow={strengthPercent} aria-valuemin={0} aria-valuemax={100}>
            <div
              className={`factor-strength-fill factor-strength-${strengthLevel}`}
              style={{ width: `${strengthPercent}%` }}
            />
          </div>
          <span className="factor-strength-value">{strengthPercent}%</span>
        </div>

        <div className="factor-meta">
          <div className="factor-meta-item">
            <span className="factor-meta-label">Enrolled:</span>
            <span className="factor-meta-value">{formatTimeAgo(factor.enrolledAt)}</span>
          </div>
          <div className="factor-meta-item">
            <span className="factor-meta-label">Last verified:</span>
            <span className="factor-meta-value">{formatTimeAgo(factor.lastVerified)}</span>
          </div>
        </div>

        {needsVerification && (
          <div className="factor-warning" role="alert">
            <span className="factor-warning-icon" aria-hidden="true">[warning]</span>
            <span className="factor-warning-text">
              Strength is low. Verify to restore full strength.
            </span>
          </div>
        )}
      </div>

      <div className="factor-card-actions">
        {allowVerify && onVerify && (
          <button
            type="button"
            className="mfa-button mfa-button-secondary factor-action-verify"
            onClick={onVerify}
            aria-describedby={`factor-${factor.factorId}-label`}
          >
            Verify
          </button>
        )}
        {allowRemove && onRemove && (
          <button
            type="button"
            className="mfa-button mfa-button-text mfa-button-danger factor-action-remove"
            onClick={onRemove}
            disabled={isRemoving}
            aria-describedby={`factor-${factor.factorId}-label`}
          >
            {isRemoving ? 'Removing...' : 'Remove'}
          </button>
        )}
      </div>
    </article>
  );
}

interface FactorCategoryGroupProps {
  category: string;
  factors: EnrolledFactor[];
  onVerify?: (factor: EnrolledFactor) => void;
  onRemove?: (factor: EnrolledFactor) => void;
  allowVerify: boolean;
  allowRemove: boolean;
  removingFactorId: string | null;
}

function FactorCategoryGroup({
  category,
  factors,
  onVerify,
  onRemove,
  allowVerify,
  allowRemove,
  removingFactorId,
}: FactorCategoryGroupProps) {
  return (
    <section className="factor-category" aria-labelledby={`category-${category}`}>
      <h3 id={`category-${category}`} className="factor-category-title">
        {category}
      </h3>
      <div className="factor-category-list">
        {factors.map((factor) => (
          <FactorCard
            key={factor.factorId}
            factor={factor}
            onVerify={onVerify ? () => onVerify(factor) : undefined}
            onRemove={onRemove ? () => onRemove(factor) : undefined}
            allowVerify={allowVerify}
            allowRemove={allowRemove}
            isRemoving={removingFactorId === factor.factorId}
          />
        ))}
      </div>
    </section>
  );
}

interface EmptyStateProps {
  message: string;
  onEnrollNew?: () => void;
}

function EmptyState({ message, onEnrollNew }: EmptyStateProps) {
  return (
    <div className="factor-list-empty" role="status">
      <span className="factor-list-empty-icon" aria-hidden="true">[shield]</span>
      <p className="factor-list-empty-message">{message}</p>
      {onEnrollNew && (
        <button
          type="button"
          className="mfa-button mfa-button-primary"
          onClick={onEnrollNew}
        >
          Enroll Your First Factor
        </button>
      )}
    </div>
  );
}

interface ListSkeletonProps {
  count?: number;
}

function ListSkeleton({ count = 3 }: ListSkeletonProps) {
  return (
    <div className="factor-list-skeleton" aria-busy="true" aria-label="Loading factors">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="factor-card-skeleton">
          <div className="factor-card-skeleton-header">
            <div className="factor-card-skeleton-icon" />
            <div className="factor-card-skeleton-info">
              <div className="factor-card-skeleton-label" />
              <div className="factor-card-skeleton-id" />
            </div>
          </div>
          <div className="factor-card-skeleton-body">
            <div className="factor-card-skeleton-bar" />
            <div className="factor-card-skeleton-meta" />
          </div>
        </div>
      ))}
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * FactorList - List of enrolled MFA factors with actions
 *
 * Shows each enrolled factor with:
 * - Type and icon
 * - Factor ID (truncated)
 * - Status (active/inactive)
 * - Effective strength bar
 * - Enrollment and last verification times
 * - Verify and Remove actions
 */
export function FactorList({
  did,
  onVerify,
  onRemove,
  onEnrollNew,
  showAddButton = true,
  allowRemove = true,
  allowVerify = true,
  filterTypes,
  className = '',
  emptyMessage = 'No authentication factors enrolled yet.',
  groupByCategory = true,
}: FactorListProps): React.ReactElement {
  const { mfaState, loading, error, refetch, factorCount } = useMfaState(did);
  const { removeFactor, error: removeError } = useMfaEnroll(did);
  const [removingFactorId, setRemovingFactorId] = React.useState<string | null>(null);
  const [confirmRemove, setConfirmRemove] = React.useState<EnrolledFactor | null>(null);

  // Filter factors if filterTypes is provided
  const factors = React.useMemo(() => {
    if (!mfaState?.factors) return [];
    if (!filterTypes) return mfaState.factors;
    return mfaState.factors.filter((f) => filterTypes.includes(f.factorType));
  }, [mfaState, filterTypes]);

  // Handle factor removal
  const handleRemove = async (factor: EnrolledFactor) => {
    setRemovingFactorId(factor.factorId);
    setConfirmRemove(null);

    try {
      await removeFactor(factor.factorId);
      await refetch();
    } catch {
      // Error is handled by the hook
    } finally {
      setRemovingFactorId(null);
    }
  };

  // Handle remove confirmation
  const handleRemoveClick = (factor: EnrolledFactor) => {
    if (onRemove) {
      // If parent wants to handle remove, delegate to them
      onRemove(factor);
    } else {
      // Otherwise, show confirmation
      setConfirmRemove(factor);
    }
  };

  // Loading state
  if (loading) {
    return (
      <div className={`factor-list ${className}`}>
        <ListSkeleton />
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className={`factor-list factor-list-error ${className}`} role="alert">
        <span className="factor-list-error-icon" aria-hidden="true">[error]</span>
        <p className="factor-list-error-message">{error.message}</p>
        <button
          type="button"
          className="mfa-button mfa-button-secondary"
          onClick={() => refetch()}
        >
          Try Again
        </button>
      </div>
    );
  }

  // Empty state
  if (factors.length === 0) {
    return (
      <div className={`factor-list factor-list-empty-container ${className}`}>
        <EmptyState message={emptyMessage} onEnrollNew={onEnrollNew} />
      </div>
    );
  }

  // Group factors if requested
  const groupedFactors = groupByCategory ? groupFactorsByCategory(factors) : null;

  return (
    <div className={`factor-list ${className}`}>
      {/* Summary header */}
      <header className="factor-list-header">
        <div className="factor-list-summary">
          <span className="factor-list-count">
            {factorCount} factor{factorCount !== 1 ? 's' : ''} enrolled
          </span>
          {mfaState?.assuranceLevel && (
            <span className="factor-list-assurance">
              Assurance: {mfaState.assuranceLevel}
            </span>
          )}
        </div>
        {showAddButton && onEnrollNew && (
          <button
            type="button"
            className="mfa-button mfa-button-primary factor-list-add"
            onClick={onEnrollNew}
          >
            Add Factor
          </button>
        )}
      </header>

      {/* Remove error alert */}
      {removeError && (
        <div className="factor-list-alert" role="alert">
          <span className="factor-list-alert-icon" aria-hidden="true">[error]</span>
          <span className="factor-list-alert-message">
            Failed to remove factor: {removeError.message}
          </span>
        </div>
      )}

      {/* Factor list */}
      <div className="factor-list-content">
        {groupedFactors ? (
          Array.from(groupedFactors.entries()).map(([category, categoryFactors]) => (
            <FactorCategoryGroup
              key={category}
              category={category}
              factors={categoryFactors}
              onVerify={onVerify}
              onRemove={handleRemoveClick}
              allowVerify={allowVerify}
              allowRemove={allowRemove}
              removingFactorId={removingFactorId}
            />
          ))
        ) : (
          <div className="factor-list-flat">
            {factors.map((factor) => (
              <FactorCard
                key={factor.factorId}
                factor={factor}
                onVerify={onVerify ? () => onVerify(factor) : undefined}
                onRemove={() => handleRemoveClick(factor)}
                allowVerify={allowVerify}
                allowRemove={allowRemove}
                isRemoving={removingFactorId === factor.factorId}
              />
            ))}
          </div>
        )}
      </div>

      {/* Remove confirmation dialog */}
      {confirmRemove && (
        <div className="factor-confirm-overlay" role="presentation" onClick={() => setConfirmRemove(null)}>
          <div
            className="factor-confirm-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="confirm-remove-title"
            aria-describedby="confirm-remove-desc"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 id="confirm-remove-title" className="factor-confirm-title">
              Remove Factor?
            </h3>
            <p id="confirm-remove-desc" className="factor-confirm-message">
              Are you sure you want to remove your{' '}
              <strong>{FACTOR_TYPE_INFO[confirmRemove.factorType]?.label || confirmRemove.factorType}</strong>?
              This may reduce your assurance level.
            </p>
            <div className="factor-confirm-actions">
              <button
                type="button"
                className="mfa-button mfa-button-text"
                onClick={() => setConfirmRemove(null)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="mfa-button mfa-button-danger"
                onClick={() => handleRemove(confirmRemove)}
              >
                Remove
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default FactorList;
