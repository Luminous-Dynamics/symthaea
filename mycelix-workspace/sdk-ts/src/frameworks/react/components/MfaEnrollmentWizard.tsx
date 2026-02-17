/**
 * MfaEnrollmentWizard - Multi-step wizard for enrolling MFA factors
 *
 * Guides users through the process of adding new authentication factors
 * to their identity, supporting various factor types from the MFDI spec.
 *
 * @module frameworks/react/components/MfaEnrollmentWizard
 *
 * @example
 * ```tsx
 * import { MfaEnrollmentWizard } from '@mycelix/sdk/react';
 *
 * function SecuritySettings() {
 *   return (
 *     <MfaEnrollmentWizard
 *       onComplete={(factor) => {
 *         console.log('Factor enrolled:', factor);
 *       }}
 *       onCancel={() => {
 *         console.log('Enrollment cancelled');
 *       }}
 *     />
 *   );
 * }
 * ```
 */

import * as React from 'react';

import { useMfaEnroll, useMfaState } from '../hooks.js';

import type { FactorType, EnrolledFactor, EnrollFactorInput } from '../hooks.js';

// =============================================================================
// Types
// =============================================================================

/**
 * Wizard step identifiers
 */
type WizardStep = 'choose-type' | 'setup' | 'verification' | 'confirmation';

/**
 * Factor type configuration for display and setup
 */
interface FactorTypeConfig {
  type: FactorType;
  label: string;
  description: string;
  icon: string;
  category: 'Cryptographic' | 'Biometric' | 'SocialProof' | 'ExternalVerification' | 'Knowledge';
  setupInstructions: string;
}

/**
 * Props for MfaEnrollmentWizard component
 */
export interface MfaEnrollmentWizardProps {
  /** Callback when enrollment completes successfully */
  onComplete?: (factor: EnrolledFactor) => void;
  /** Callback when user cancels enrollment */
  onCancel?: () => void;
  /** Allowed factor types (all if not specified) */
  allowedTypes?: FactorType[];
  /** CSS class for the wizard container */
  className?: string;
  /** Initial factor type to select */
  initialType?: FactorType;
}

// =============================================================================
// Constants
// =============================================================================

const FACTOR_CONFIGS: FactorTypeConfig[] = [
  {
    type: 'PrimaryKeyPair',
    label: 'Primary Key Pair',
    description: 'Your main cryptographic identity key',
    icon: 'key',
    category: 'Cryptographic',
    setupInstructions: 'Generate a new cryptographic key pair to serve as your primary identity.',
  },
  {
    type: 'HardwareKey',
    label: 'Hardware Security Key',
    description: 'YubiKey, Ledger, or other hardware authenticator',
    icon: 'usb',
    category: 'Cryptographic',
    setupInstructions: 'Connect your hardware security key and follow the prompts to register it.',
  },
  {
    type: 'Biometric',
    label: 'Biometric Authentication',
    description: 'Fingerprint, face recognition, or voice',
    icon: 'fingerprint',
    category: 'Biometric',
    setupInstructions: 'Use your device biometric sensor to register your biometric data.',
  },
  {
    type: 'SocialRecovery',
    label: 'Social Recovery Guardians',
    description: 'Trusted contacts who can help recover your identity',
    icon: 'users',
    category: 'SocialProof',
    setupInstructions: 'Select trusted guardians who can help recover your identity if needed.',
  },
  {
    type: 'ReputationAttestation',
    label: 'Reputation Attestation',
    description: 'Verified reputation from network participation',
    icon: 'star',
    category: 'SocialProof',
    setupInstructions: 'Your reputation will be calculated from your network participation history.',
  },
  {
    type: 'GitcoinPassport',
    label: 'Gitcoin Passport',
    description: 'Proof of personhood via Gitcoin Passport stamps',
    icon: 'passport',
    category: 'ExternalVerification',
    setupInstructions: 'Connect your Gitcoin Passport to verify your stamps and personhood score.',
  },
  {
    type: 'VerifiableCredential',
    label: 'Verifiable Credential',
    description: 'Third-party verified credentials (KYC, education, etc.)',
    icon: 'certificate',
    category: 'ExternalVerification',
    setupInstructions: 'Import a verifiable credential from a trusted issuer.',
  },
  {
    type: 'RecoveryPhrase',
    label: 'Recovery Phrase',
    description: 'Secure backup phrase for account recovery',
    icon: 'note',
    category: 'Knowledge',
    setupInstructions: 'Generate and securely store a recovery phrase.',
  },
  {
    type: 'SecurityQuestions',
    label: 'Security Questions',
    description: 'Personal questions for identity verification',
    icon: 'question',
    category: 'Knowledge',
    setupInstructions: 'Set up security questions that only you can answer.',
  },
];

// =============================================================================
// Sub-components
// =============================================================================

interface StepIndicatorProps {
  currentStep: WizardStep;
  steps: WizardStep[];
}

function StepIndicator({ currentStep, steps }: StepIndicatorProps) {
  const stepLabels: Record<WizardStep, string> = {
    'choose-type': 'Choose Type',
    setup: 'Setup',
    verification: 'Verify',
    confirmation: 'Complete',
  };

  const currentIndex = steps.indexOf(currentStep);

  return (
    <nav className="mfa-wizard-steps" aria-label="Enrollment progress">
      <ol className="mfa-wizard-steps-list" role="list">
        {steps.map((step, index) => {
          const isComplete = index < currentIndex;
          const isCurrent = index === currentIndex;

          return (
            <li
              key={step}
              className={`mfa-wizard-step ${isComplete ? 'mfa-wizard-step-complete' : ''} ${isCurrent ? 'mfa-wizard-step-current' : ''}`}
              aria-current={isCurrent ? 'step' : undefined}
            >
              <span className="mfa-wizard-step-number">{index + 1}</span>
              <span className="mfa-wizard-step-label">{stepLabels[step]}</span>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

interface FactorTypeSelectorProps {
  allowedTypes?: FactorType[];
  selectedType: FactorType | null;
  onSelect: (type: FactorType) => void;
}

function FactorTypeSelector({ allowedTypes, selectedType, onSelect }: FactorTypeSelectorProps) {
  const availableConfigs = allowedTypes
    ? FACTOR_CONFIGS.filter((c) => allowedTypes.includes(c.type))
    : FACTOR_CONFIGS;

  // Group by category
  const groupedConfigs = availableConfigs.reduce(
    (acc, config) => {
      if (!acc[config.category]) {
        acc[config.category] = [];
      }
      acc[config.category].push(config);
      return acc;
    },
    {} as Record<string, FactorTypeConfig[]>
  );

  return (
    <div className="mfa-factor-selector" role="listbox" aria-label="Select factor type">
      {Object.entries(groupedConfigs).map(([category, configs]) => (
        <div key={category} className="mfa-factor-category">
          <h3 className="mfa-factor-category-title">{category}</h3>
          <div className="mfa-factor-options">
            {configs.map((config) => (
              <button
                key={config.type}
                type="button"
                role="option"
                aria-selected={selectedType === config.type}
                className={`mfa-factor-option ${selectedType === config.type ? 'mfa-factor-option-selected' : ''}`}
                onClick={() => onSelect(config.type)}
              >
                <span className="mfa-factor-icon" aria-hidden="true">
                  [{config.icon}]
                </span>
                <span className="mfa-factor-info">
                  <span className="mfa-factor-label">{config.label}</span>
                  <span className="mfa-factor-description">{config.description}</span>
                </span>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

interface FactorSetupFormProps {
  factorType: FactorType;
  onSubmit: (metadata: Record<string, unknown>) => void;
  loading: boolean;
}

function FactorSetupForm({ factorType, onSubmit, loading }: FactorSetupFormProps) {
  const config = FACTOR_CONFIGS.find((c) => c.type === factorType);
  const [metadata, setMetadata] = React.useState<Record<string, string>>({});
  const [setupState, setSetupState] = React.useState<'idle' | 'generating' | 'ready'>('idle');

  const handleInputChange = (key: string, value: string) => {
    setMetadata((prev) => ({ ...prev, [key]: value }));
  };

  const handleGenerate = async () => {
    setSetupState('generating');

    // Simulate factor-specific generation
    await new Promise((resolve) => setTimeout(resolve, 1500));

    // Generate appropriate metadata based on factor type
    switch (factorType) {
      case 'PrimaryKeyPair':
        setMetadata({
          publicKeyHash: `pkh_${Math.random().toString(36).substring(2, 15)}`,
          algorithm: 'Ed25519',
        });
        break;
      case 'HardwareKey':
        setMetadata({
          credentialId: `cred_${Math.random().toString(36).substring(2, 15)}`,
          authenticatorName: 'Hardware Key',
        });
        break;
      case 'RecoveryPhrase':
        setMetadata({
          phraseHash: `rph_${Math.random().toString(36).substring(2, 15)}`,
          wordCount: '24',
        });
        break;
      case 'SocialRecovery':
        setMetadata({
          threshold: '3',
          guardianCount: '5',
        });
        break;
      default:
        setMetadata({ configured: 'true' });
    }

    setSetupState('ready');
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(metadata);
  };

  if (!config) {
    return <p className="mfa-setup-error">Unknown factor type</p>;
  }

  return (
    <form className="mfa-setup-form" onSubmit={handleSubmit}>
      <div className="mfa-setup-instructions">
        <p>{config.setupInstructions}</p>
      </div>

      {/* Factor-specific setup UI */}
      {factorType === 'SocialRecovery' && (
        <div className="mfa-setup-fields">
          <label className="mfa-setup-field">
            <span className="mfa-setup-field-label">Recovery Threshold</span>
            <input
              type="number"
              min="2"
              max="10"
              value={metadata.threshold || '3'}
              onChange={(e) => handleInputChange('threshold', e.target.value)}
              className="mfa-setup-input"
              aria-describedby="threshold-help"
            />
            <span id="threshold-help" className="mfa-setup-field-help">
              Number of guardians required for recovery
            </span>
          </label>
          <label className="mfa-setup-field">
            <span className="mfa-setup-field-label">Total Guardians</span>
            <input
              type="number"
              min="3"
              max="20"
              value={metadata.guardianCount || '5'}
              onChange={(e) => handleInputChange('guardianCount', e.target.value)}
              className="mfa-setup-input"
              aria-describedby="guardians-help"
            />
            <span id="guardians-help" className="mfa-setup-field-help">
              Total number of trusted guardians to add
            </span>
          </label>
        </div>
      )}

      {factorType === 'SecurityQuestions' && (
        <div className="mfa-setup-fields">
          <label className="mfa-setup-field">
            <span className="mfa-setup-field-label">Question 1</span>
            <input
              type="text"
              value={metadata.question1 || ''}
              onChange={(e) => handleInputChange('question1', e.target.value)}
              placeholder="Enter your security question"
              className="mfa-setup-input"
            />
          </label>
          <label className="mfa-setup-field">
            <span className="mfa-setup-field-label">Answer 1</span>
            <input
              type="password"
              value={metadata.answer1 || ''}
              onChange={(e) => handleInputChange('answer1', e.target.value)}
              placeholder="Enter your answer"
              className="mfa-setup-input"
            />
          </label>
        </div>
      )}

      {factorType === 'VerifiableCredential' && (
        <div className="mfa-setup-fields">
          <label className="mfa-setup-field">
            <span className="mfa-setup-field-label">Credential URL or Data</span>
            <textarea
              value={metadata.credentialData || ''}
              onChange={(e) => handleInputChange('credentialData', e.target.value)}
              placeholder="Paste your verifiable credential JSON"
              className="mfa-setup-textarea"
              rows={4}
            />
          </label>
        </div>
      )}

      {/* Generate button for cryptographic factors */}
      {['PrimaryKeyPair', 'HardwareKey', 'RecoveryPhrase', 'Biometric'].includes(factorType) && (
        <div className="mfa-setup-generate">
          {setupState === 'idle' && (
            <button
              type="button"
              className="mfa-button mfa-button-secondary"
              onClick={handleGenerate}
            >
              {factorType === 'HardwareKey'
                ? 'Connect Device'
                : factorType === 'Biometric'
                  ? 'Scan Biometric'
                  : 'Generate'}
            </button>
          )}
          {setupState === 'generating' && (
            <p className="mfa-setup-generating" aria-live="polite">
              Setting up...
            </p>
          )}
          {setupState === 'ready' && (
            <p className="mfa-setup-ready" aria-live="polite">
              Ready to proceed
            </p>
          )}
        </div>
      )}

      <div className="mfa-setup-actions">
        <button
          type="submit"
          className="mfa-button mfa-button-primary"
          disabled={loading || (setupState !== 'ready' && ['PrimaryKeyPair', 'HardwareKey', 'RecoveryPhrase', 'Biometric'].includes(factorType))}
        >
          {loading ? 'Enrolling...' : 'Continue to Verification'}
        </button>
      </div>
    </form>
  );
}

interface VerificationStepProps {
  factorType: FactorType;
  onVerify: () => Promise<void>;
  loading: boolean;
}

function VerificationStep({ factorType, onVerify, loading }: VerificationStepProps) {
  const [verificationState, setVerificationState] = React.useState<'pending' | 'verifying' | 'success' | 'error'>('pending');
  const [errorMessage, setErrorMessage] = React.useState<string>('');

  const handleVerify = async () => {
    setVerificationState('verifying');
    setErrorMessage('');

    try {
      await onVerify();
      setVerificationState('success');
    } catch (err) {
      setVerificationState('error');
      setErrorMessage(err instanceof Error ? err.message : 'Verification failed');
    }
  };

  return (
    <div className="mfa-verification">
      <div className="mfa-verification-content">
        {verificationState === 'pending' && (
          <>
            <p className="mfa-verification-message">
              Please verify your {FACTOR_CONFIGS.find((c) => c.type === factorType)?.label || 'factor'} to complete enrollment.
            </p>
            <button
              type="button"
              className="mfa-button mfa-button-primary"
              onClick={handleVerify}
              disabled={loading}
            >
              Verify Now
            </button>
          </>
        )}

        {verificationState === 'verifying' && (
          <p className="mfa-verification-status" aria-live="polite">
            Verifying...
          </p>
        )}

        {verificationState === 'success' && (
          <p className="mfa-verification-success" role="status">
            Verification successful!
          </p>
        )}

        {verificationState === 'error' && (
          <div className="mfa-verification-error" role="alert">
            <p>{errorMessage}</p>
            <button
              type="button"
              className="mfa-button mfa-button-secondary"
              onClick={handleVerify}
              disabled={loading}
            >
              Try Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

interface ConfirmationStepProps {
  factor: EnrolledFactor;
  onComplete: () => void;
}

function ConfirmationStep({ factor, onComplete }: ConfirmationStepProps) {
  const config = FACTOR_CONFIGS.find((c) => c.type === factor.factorType);

  return (
    <div className="mfa-confirmation">
      <div className="mfa-confirmation-icon" aria-hidden="true">
        [check]
      </div>

      <h3 className="mfa-confirmation-title">Factor Enrolled Successfully</h3>

      <dl className="mfa-confirmation-details">
        <dt>Factor Type</dt>
        <dd>{config?.label || factor.factorType}</dd>

        <dt>Factor ID</dt>
        <dd className="mfa-confirmation-id">{factor.factorId}</dd>

        <dt>Enrolled At</dt>
        <dd>{new Date(factor.enrolledAt / 1000).toLocaleString()}</dd>

        <dt>Effective Strength</dt>
        <dd>{(factor.effectiveStrength * 100).toFixed(0)}%</dd>
      </dl>

      <button type="button" className="mfa-button mfa-button-primary" onClick={onComplete}>
        Done
      </button>
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * MfaEnrollmentWizard - Multi-step wizard for enrolling MFA factors
 *
 * Guides users through:
 * 1. Choosing a factor type
 * 2. Factor-specific setup
 * 3. Verification
 * 4. Confirmation
 */
export function MfaEnrollmentWizard({
  onComplete,
  onCancel,
  allowedTypes,
  className = '',
  initialType,
}: MfaEnrollmentWizardProps): React.ReactElement {
  const { enrollFactor, createMfaState, loading, error, reset } = useMfaEnroll();
  const { hasMfa, refetch: refetchMfaState } = useMfaState();

  const [currentStep, setCurrentStep] = React.useState<WizardStep>('choose-type');
  const [selectedType, setSelectedType] = React.useState<FactorType | null>(initialType || null);
  const [enrolledFactor, setEnrolledFactor] = React.useState<EnrolledFactor | null>(null);
  const [_setupMetadata, setSetupMetadata] = React.useState<Record<string, unknown>>({});

  const steps: WizardStep[] = ['choose-type', 'setup', 'verification', 'confirmation'];

  // Handle step navigation
  const goToStep = (step: WizardStep) => {
    reset();
    setCurrentStep(step);
  };

  const handleTypeSelect = (type: FactorType) => {
    setSelectedType(type);
  };

  const handleTypeConfirm = () => {
    if (selectedType) {
      goToStep('setup');
    }
  };

  const handleSetupSubmit = async (metadata: Record<string, unknown>) => {
    setSetupMetadata(metadata);

    try {
      // Create MFA state if it doesn't exist
      if (!hasMfa && selectedType === 'PrimaryKeyPair') {
        const primaryKeyHash = (metadata.publicKeyHash as string) || `pkh_${Date.now()}`;
        await createMfaState(primaryKeyHash);
      }

      // Enroll the factor
      const input: EnrollFactorInput = {
        factorType: selectedType!,
        metadata,
      };

      const factor = await enrollFactor(input);
      setEnrolledFactor(factor);
      goToStep('verification');
    } catch {
      // Error is handled by the hook and displayed
    }
  };

  const handleVerification = async () => {
    // In a real implementation, this would call useMfaVerify
    // For now, we simulate verification success
    await new Promise((resolve) => setTimeout(resolve, 1000));
    goToStep('confirmation');
  };

  const handleComplete = () => {
    refetchMfaState();
    if (enrolledFactor && onComplete) {
      onComplete(enrolledFactor);
    }
  };

  const handleCancel = () => {
    reset();
    if (onCancel) {
      onCancel();
    }
  };

  const handleBack = () => {
    const currentIndex = steps.indexOf(currentStep);
    if (currentIndex > 0) {
      goToStep(steps[currentIndex - 1]);
    }
  };

  return (
    <div className={`mfa-enrollment-wizard ${className}`} role="dialog" aria-labelledby="mfa-wizard-title">
      <header className="mfa-wizard-header">
        <h2 id="mfa-wizard-title" className="mfa-wizard-title">
          Enroll Authentication Factor
        </h2>
        <StepIndicator currentStep={currentStep} steps={steps} />
      </header>

      <main className="mfa-wizard-content">
        {error && (
          <div className="mfa-wizard-error" role="alert">
            <p>{error.message}</p>
            <button type="button" className="mfa-button mfa-button-text" onClick={reset}>
              Dismiss
            </button>
          </div>
        )}

        {currentStep === 'choose-type' && (
          <section aria-labelledby="step-choose-type">
            <h3 id="step-choose-type" className="mfa-step-title">
              Choose Factor Type
            </h3>
            <FactorTypeSelector
              allowedTypes={allowedTypes}
              selectedType={selectedType}
              onSelect={handleTypeSelect}
            />
            <div className="mfa-wizard-actions">
              <button type="button" className="mfa-button mfa-button-text" onClick={handleCancel}>
                Cancel
              </button>
              <button
                type="button"
                className="mfa-button mfa-button-primary"
                onClick={handleTypeConfirm}
                disabled={!selectedType}
              >
                Continue
              </button>
            </div>
          </section>
        )}

        {currentStep === 'setup' && selectedType && (
          <section aria-labelledby="step-setup">
            <h3 id="step-setup" className="mfa-step-title">
              Setup {FACTOR_CONFIGS.find((c) => c.type === selectedType)?.label}
            </h3>
            <FactorSetupForm factorType={selectedType} onSubmit={handleSetupSubmit} loading={loading} />
            <div className="mfa-wizard-actions">
              <button type="button" className="mfa-button mfa-button-text" onClick={handleBack}>
                Back
              </button>
            </div>
          </section>
        )}

        {currentStep === 'verification' && selectedType && (
          <section aria-labelledby="step-verification">
            <h3 id="step-verification" className="mfa-step-title">
              Verify Factor
            </h3>
            <VerificationStep factorType={selectedType} onVerify={handleVerification} loading={loading} />
            <div className="mfa-wizard-actions">
              <button type="button" className="mfa-button mfa-button-text" onClick={handleBack}>
                Back
              </button>
            </div>
          </section>
        )}

        {currentStep === 'confirmation' && enrolledFactor && (
          <section aria-labelledby="step-confirmation">
            <h3 id="step-confirmation" className="visually-hidden">
              Enrollment Complete
            </h3>
            <ConfirmationStep factor={enrolledFactor} onComplete={handleComplete} />
          </section>
        )}
      </main>
    </div>
  );
}

export default MfaEnrollmentWizard;
