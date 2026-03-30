// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * FactorVerificationModal - Modal for verifying an MFA factor
 *
 * Displays a challenge and accepts proof input for factor verification.
 * Supports various proof types including signatures, WebAuthn, and more.
 *
 * @module frameworks/react/components/FactorVerificationModal
 *
 * @example
 * ```tsx
 * import { FactorVerificationModal } from '@mycelix/sdk/react';
 *
 * function FactorCard({ factor }) {
 *   const [showVerify, setShowVerify] = useState(false);
 *
 *   return (
 *     <>
 *       <button onClick={() => setShowVerify(true)}>Verify</button>
 *       {showVerify && (
 *         <FactorVerificationModal
 *           factorId={factor.factorId}
 *           factorType={factor.factorType}
 *           onSuccess={(result) => {
 *             console.log('Verified!', result);
 *             setShowVerify(false);
 *           }}
 *           onClose={() => setShowVerify(false)}
 *         />
 *       )}
 *     </>
 *   );
 * }
 * ```
 */

import * as React from 'react';

import { useMfaVerify } from '../hooks.js';

import type { FactorType, MfaVerificationResult, VerificationChallenge, VerificationProof } from '../hooks.js';

// =============================================================================
// Types
// =============================================================================

/**
 * Props for FactorVerificationModal component
 */
export interface FactorVerificationModalProps {
  /** The factor ID to verify */
  factorId: string;
  /** The type of factor being verified */
  factorType: FactorType;
  /** Callback when verification succeeds */
  onSuccess?: (result: MfaVerificationResult) => void;
  /** Callback when verification fails */
  onError?: (error: Error) => void;
  /** Callback when modal is closed */
  onClose: () => void;
  /** CSS class for the modal container */
  className?: string;
  /** Custom title for the modal */
  title?: string;
}

/**
 * Verification state machine
 */
type VerificationState = 'generating-challenge' | 'awaiting-proof' | 'verifying' | 'success' | 'error';

// =============================================================================
// Helper Functions
// =============================================================================

function getProofInputLabel(factorType: FactorType): string {
  switch (factorType) {
    case 'PrimaryKeyPair':
    case 'HardwareKey':
      return 'Signature';
    case 'Biometric':
      return 'Biometric Data';
    case 'RecoveryPhrase':
      return 'Recovery Phrase';
    case 'SecurityQuestions':
      return 'Answer';
    case 'GitcoinPassport':
      return 'Passport Score';
    case 'VerifiableCredential':
      return 'Credential Proof';
    case 'SocialRecovery':
      return 'Guardian Signatures';
    case 'ReputationAttestation':
      return 'Attestation';
    default:
      return 'Proof';
  }
}

function getProofInputType(factorType: FactorType): 'text' | 'password' | 'textarea' {
  switch (factorType) {
    case 'RecoveryPhrase':
    case 'SecurityQuestions':
      return 'password';
    case 'VerifiableCredential':
    case 'SocialRecovery':
      return 'textarea';
    default:
      return 'text';
  }
}

function getProofInstructions(factorType: FactorType): string {
  switch (factorType) {
    case 'PrimaryKeyPair':
      return 'Sign the challenge with your primary key to verify your identity.';
    case 'HardwareKey':
      return 'Connect your hardware key and approve the verification request.';
    case 'Biometric':
      return 'Use your biometric sensor to verify your identity.';
    case 'RecoveryPhrase':
      return 'Enter your recovery phrase to verify access.';
    case 'SecurityQuestions':
      return 'Answer your security question correctly.';
    case 'GitcoinPassport':
      return 'Connect your Gitcoin Passport to verify your stamps.';
    case 'VerifiableCredential':
      return 'Provide a valid presentation of your credential.';
    case 'SocialRecovery':
      return 'Collect signatures from your guardians.';
    case 'ReputationAttestation':
      return 'Provide proof of your reputation attestation.';
    default:
      return 'Provide proof to verify this factor.';
  }
}

// =============================================================================
// Sub-components
// =============================================================================

interface ChallengeDisplayProps {
  challenge: VerificationChallenge;
  factorType: FactorType;
}

function ChallengeDisplay({ challenge, factorType: _factorType }: ChallengeDisplayProps) {
  const expiresAt = new Date(challenge.expiresAt / 1000);
  const [timeLeft, setTimeLeft] = React.useState<number>(
    Math.max(0, Math.floor((expiresAt.getTime() - Date.now()) / 1000))
  );

  React.useEffect(() => {
    const timer = setInterval(() => {
      const remaining = Math.max(0, Math.floor((expiresAt.getTime() - Date.now()) / 1000));
      setTimeLeft(remaining);
    }, 1000);

    return () => clearInterval(timer);
  }, [expiresAt]);

  const isExpired = timeLeft <= 0;

  return (
    <div className="mfa-challenge-display">
      <div className="mfa-challenge-info">
        <span className="mfa-challenge-label">Challenge:</span>
        <code className="mfa-challenge-value">{challenge.challenge}</code>
      </div>

      <div className={`mfa-challenge-timer ${isExpired ? 'mfa-challenge-expired' : ''}`} aria-live="polite">
        {isExpired ? (
          <span className="mfa-challenge-expired-text">Challenge expired</span>
        ) : (
          <span className="mfa-challenge-remaining">
            Expires in: {Math.floor(timeLeft / 60)}:{(timeLeft % 60).toString().padStart(2, '0')}
          </span>
        )}
      </div>
    </div>
  );
}

interface ProofInputProps {
  factorType: FactorType;
  value: string;
  onChange: (value: string) => void;
  disabled: boolean;
}

function ProofInput({ factorType, value, onChange, disabled }: ProofInputProps) {
  const inputType = getProofInputType(factorType);
  const label = getProofInputLabel(factorType);
  const inputId = 'proof-input';

  if (inputType === 'textarea') {
    return (
      <label className="mfa-proof-field">
        <span className="mfa-proof-label">{label}</span>
        <textarea
          id={inputId}
          className="mfa-proof-textarea"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          rows={4}
          placeholder={`Enter your ${label.toLowerCase()}`}
          aria-describedby="proof-help"
        />
      </label>
    );
  }

  return (
    <label className="mfa-proof-field">
      <span className="mfa-proof-label">{label}</span>
      <input
        id={inputId}
        type={inputType}
        className="mfa-proof-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        placeholder={`Enter your ${label.toLowerCase()}`}
        aria-describedby="proof-help"
      />
    </label>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * FactorVerificationModal - Modal for verifying an MFA factor
 *
 * Displays a challenge and accepts proof input for verification.
 */
export function FactorVerificationModal({
  factorId,
  factorType,
  onSuccess,
  onError,
  onClose,
  className = '',
  title,
}: FactorVerificationModalProps): React.ReactElement {
  const { generateChallenge, verifyFactor, loading, error, reset } = useMfaVerify();

  const [state, setState] = React.useState<VerificationState>('generating-challenge');
  const [challenge, setChallenge] = React.useState<VerificationChallenge | null>(null);
  const [proofValue, setProofValue] = React.useState('');
  const [result, setResult] = React.useState<MfaVerificationResult | null>(null);

  const modalRef = React.useRef<HTMLDivElement>(null);
  const previousFocusRef = React.useRef<HTMLElement | null>(null);

  // Focus management for accessibility
  React.useEffect(() => {
    previousFocusRef.current = document.activeElement as HTMLElement;

    if (modalRef.current) {
      const firstFocusable = modalRef.current.querySelector<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      firstFocusable?.focus();
    }

    return () => {
      previousFocusRef.current?.focus();
    };
  }, []);

  // Handle escape key
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  // Generate challenge on mount
  React.useEffect(() => {
    const fetchChallenge = async () => {
      try {
        const challengeData = await generateChallenge(factorId);
        setChallenge(challengeData);
        setState('awaiting-proof');
      } catch (err) {
        setState('error');
        onError?.(err instanceof Error ? err : new Error(String(err)));
      }
    };

    fetchChallenge();
  }, [factorId, generateChallenge, onError]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!proofValue.trim()) {
      return;
    }

    setState('verifying');
    reset();

    try {
      const proof: VerificationProof = {
        type: factorType,
        value: proofValue,
        challengeId: challenge?.challengeId,
      };

      const verificationResult = await verifyFactor(factorId, proof);
      setResult(verificationResult);
      setState('success');
      onSuccess?.(verificationResult);
    } catch (err) {
      setState('error');
      onError?.(err instanceof Error ? err : new Error(String(err)));
    }
  };

  const handleRetry = async () => {
    setState('generating-challenge');
    reset();
    setProofValue('');
    setChallenge(null);

    try {
      const challengeData = await generateChallenge(factorId);
      setChallenge(challengeData);
      setState('awaiting-proof');
    } catch (err) {
      setState('error');
    }
  };

  const modalTitle = title || `Verify ${getProofInputLabel(factorType)}`;

  return (
    <div className="mfa-modal-overlay" onClick={onClose} role="presentation">
      <div
        ref={modalRef}
        className={`mfa-modal mfa-verification-modal ${className}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="verification-modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="mfa-modal-header">
          <h2 id="verification-modal-title" className="mfa-modal-title">
            {modalTitle}
          </h2>
          <button
            type="button"
            className="mfa-modal-close"
            onClick={onClose}
            aria-label="Close modal"
          >
            &times;
          </button>
        </header>

        <main className="mfa-modal-content">
          {state === 'generating-challenge' && (
            <div className="mfa-verification-loading" aria-live="polite">
              <p>Generating verification challenge...</p>
            </div>
          )}

          {state === 'awaiting-proof' && challenge && (
            <form onSubmit={handleSubmit} className="mfa-verification-form">
              <p id="proof-help" className="mfa-verification-instructions">
                {getProofInstructions(factorType)}
              </p>

              <ChallengeDisplay challenge={challenge} factorType={factorType} />

              <ProofInput
                factorType={factorType}
                value={proofValue}
                onChange={setProofValue}
                disabled={loading}
              />

              {/* Hardware key and biometric have auto-trigger */}
              {(factorType === 'HardwareKey' || factorType === 'Biometric') && (
                <button
                  type="button"
                  className="mfa-button mfa-button-secondary mfa-button-trigger"
                  onClick={() => {
                    // Simulate hardware key / biometric trigger
                    setProofValue(`${factorType.toLowerCase()}_proof_${Date.now()}`);
                  }}
                  disabled={loading}
                >
                  {factorType === 'HardwareKey' ? 'Use Hardware Key' : 'Scan Now'}
                </button>
              )}

              <div className="mfa-modal-actions">
                <button
                  type="button"
                  className="mfa-button mfa-button-text"
                  onClick={onClose}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="mfa-button mfa-button-primary"
                  disabled={loading || !proofValue.trim()}
                >
                  {loading ? 'Verifying...' : 'Verify'}
                </button>
              </div>
            </form>
          )}

          {state === 'verifying' && (
            <div className="mfa-verification-loading" aria-live="polite">
              <p>Verifying...</p>
            </div>
          )}

          {state === 'success' && result && (
            <div className="mfa-verification-success" role="status">
              <div className="mfa-verification-success-icon" aria-hidden="true">
                [check]
              </div>
              <h3>Verification Successful</h3>
              <p>
                Factor verified at {new Date(result.verifiedAt / 1000).toLocaleString()}
              </p>
              {result.newAssuranceLevel && (
                <p>
                  Your assurance level is now: <strong>{result.newAssuranceLevel}</strong>
                </p>
              )}
              <button
                type="button"
                className="mfa-button mfa-button-primary"
                onClick={onClose}
              >
                Done
              </button>
            </div>
          )}

          {state === 'error' && (
            <div className="mfa-verification-error" role="alert">
              <div className="mfa-verification-error-icon" aria-hidden="true">
                [error]
              </div>
              <h3>Verification Failed</h3>
              <p>{error?.message || 'An error occurred during verification.'}</p>
              <div className="mfa-modal-actions">
                <button
                  type="button"
                  className="mfa-button mfa-button-text"
                  onClick={onClose}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="mfa-button mfa-button-primary"
                  onClick={handleRetry}
                >
                  Try Again
                </button>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default FactorVerificationModal;
