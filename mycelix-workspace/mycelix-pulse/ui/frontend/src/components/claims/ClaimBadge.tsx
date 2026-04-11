// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useState } from 'react';

// Types matching backend
type ProofType = 'risc0_zk_vm' | 'winterfell_air' | 'simplified_hash' | 'ed25519_signature' | 'none';
type CredentialType =
  | 'verified_human'
  | 'gitcoin_passport'
  | 'reputation_score'
  | 'employment'
  | 'education'
  | 'certification'
  | 'membership'
  | 'attestation';
type AssuranceLevel = 'e0_anonymous' | 'e1_verified_email' | 'e2_gitcoin_passport' | 'e3_multi_factor' | 'e4_constitutional';

interface VerifiableCredential {
  id: string;
  credential_type: CredentialType;
  issuer_did: string;
  subject_did: string;
  issued_at: string;
  expires_at?: string;
  claims: Record<string, unknown>;
}

interface VerifiableClaim {
  claim_type: 'identity' | 'affiliation' | 'credential' | 'cryptographic_proof';
  claim?: string;
  organization?: string;
  role?: string;
  statement?: string;
  proof_type?: ProofType;
  credential?: VerifiableCredential;
}

interface ClaimVerification {
  verified: boolean;
  epistemic_tier: number;
  proof_type: ProofType;
  verified_at: string;
}

interface ClaimBadgeProps {
  claim: VerifiableClaim;
  verification?: ClaimVerification;
  compact?: boolean;
  onClick?: () => void;
  className?: string;
}

// Epistemic tier configuration
const epistemicTierConfig: Record<number, { label: string; color: string; description: string }> = {
  0: {
    label: 'Null',
    color: 'bg-gray-100 text-gray-600 border-gray-300 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-600',
    description: 'No epistemic claim',
  },
  1: {
    label: 'Testimonial',
    color: 'bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-900/30 dark:text-blue-300 dark:border-blue-800',
    description: 'Based on personal testimony',
  },
  2: {
    label: 'Privately Verifiable',
    color: 'bg-indigo-50 text-indigo-700 border-indigo-200 dark:bg-indigo-900/30 dark:text-indigo-300 dark:border-indigo-800',
    description: 'Can be verified by recipient',
  },
  3: {
    label: 'Cryptographically Proven',
    color: 'bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-900/30 dark:text-purple-300 dark:border-purple-800',
    description: 'Verified via cryptographic proof',
  },
  4: {
    label: 'Publicly Reproducible',
    color: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-900/30 dark:text-emerald-300 dark:border-emerald-800',
    description: 'Independently verifiable by anyone',
  },
};

// Proof type icons and labels
const proofTypeConfig: Record<ProofType, { icon: string; label: string }> = {
  risc0_zk_vm: { icon: '🔐', label: 'RISC0 zkSTARK' },
  winterfell_air: { icon: '❄️', label: 'Winterfell AIR' },
  simplified_hash: { icon: '🔗', label: 'Hash Commitment' },
  ed25519_signature: { icon: '✍️', label: 'Ed25519 Signature' },
  none: { icon: '❓', label: 'No Proof' },
};

// Credential type icons
const credentialTypeConfig: Record<CredentialType, { icon: string; label: string }> = {
  verified_human: { icon: '👤', label: 'Verified Human' },
  gitcoin_passport: { icon: '🛂', label: 'Gitcoin Passport' },
  reputation_score: { icon: '⭐', label: 'Reputation Score' },
  employment: { icon: '💼', label: 'Employment' },
  education: { icon: '🎓', label: 'Education' },
  certification: { icon: '📜', label: 'Certification' },
  membership: { icon: '🏛️', label: 'Membership' },
  attestation: { icon: '📋', label: 'Attestation' },
};

// Assurance level configuration
const assuranceLevelConfig: Record<AssuranceLevel, { label: string; color: string; cost: string }> = {
  e0_anonymous: {
    label: 'E0: Anonymous',
    color: 'text-gray-500',
    cost: '$0',
  },
  e1_verified_email: {
    label: 'E1: Email Verified',
    color: 'text-blue-500',
    cost: '$100',
  },
  e2_gitcoin_passport: {
    label: 'E2: Gitcoin Passport',
    color: 'text-indigo-500',
    cost: '$1,000',
  },
  e3_multi_factor: {
    label: 'E3: Multi-Factor',
    color: 'text-purple-500',
    cost: '$100,000',
  },
  e4_constitutional: {
    label: 'E4: Constitutional',
    color: 'text-emerald-500',
    cost: '$10M',
  },
};

// Verification status icon
function VerificationIcon({ verified, className = '' }: { verified: boolean; className?: string }) {
  if (verified) {
    return (
      <svg className={`w-4 h-4 text-emerald-500 ${className}`} fill="currentColor" viewBox="0 0 20 20">
        <path
          fillRule="evenodd"
          d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
          clipRule="evenodd"
        />
      </svg>
    );
  }
  return (
    <svg className={`w-4 h-4 text-gray-400 ${className}`} fill="currentColor" viewBox="0 0 20 20">
      <path
        fillRule="evenodd"
        d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z"
        clipRule="evenodd"
      />
    </svg>
  );
}

// Get claim display text
function getClaimText(claim: VerifiableClaim): string {
  switch (claim.claim_type) {
    case 'identity':
      return claim.claim || 'Identity claim';
    case 'affiliation':
      return claim.role
        ? `${claim.role} at ${claim.organization}`
        : `Member of ${claim.organization}`;
    case 'credential':
      return claim.credential
        ? credentialTypeConfig[claim.credential.credential_type]?.label || 'Credential'
        : 'Credential';
    case 'cryptographic_proof':
      return claim.statement || 'Cryptographic proof';
    default:
      return 'Unknown claim';
  }
}

// Compact badge version
function CompactClaimBadge({
  claim,
  verification,
  onClick,
  className = '',
}: ClaimBadgeProps) {
  const isVerified = verification?.verified ?? false;
  const tier = verification?.epistemic_tier ?? 1;
  const tierConfig = epistemicTierConfig[tier];

  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center px-2 py-1 rounded-full border text-xs font-medium transition-colors ${tierConfig.color} ${
        onClick ? 'cursor-pointer hover:opacity-80' : 'cursor-default'
      } ${className}`}
      title={`${getClaimText(claim)} - ${tierConfig.description}`}
    >
      <VerificationIcon verified={isVerified} className="mr-1" />
      <span className="truncate max-w-[120px]">{getClaimText(claim)}</span>
    </button>
  );
}

// Full claim badge with details
export default function ClaimBadge({
  claim,
  verification,
  compact = false,
  onClick,
  className = '',
}: ClaimBadgeProps) {
  const [showDetails, setShowDetails] = useState(false);

  if (compact) {
    return <CompactClaimBadge claim={claim} verification={verification} onClick={onClick} className={className} />;
  }

  const isVerified = verification?.verified ?? false;
  const tier = verification?.epistemic_tier ?? 1;
  const tierConfig = epistemicTierConfig[tier];
  const proofConfig = verification?.proof_type
    ? proofTypeConfig[verification.proof_type]
    : proofTypeConfig.none;

  return (
    <div className={`relative ${className}`}>
      <button
        onClick={() => setShowDetails(!showDetails)}
        className={`flex items-center px-3 py-2 rounded-lg border text-sm font-medium transition-all ${tierConfig.color} ${
          showDetails ? 'ring-2 ring-offset-2 ring-blue-500' : ''
        }`}
      >
        <VerificationIcon verified={isVerified} className="mr-2" />
        <span>{getClaimText(claim)}</span>
        <span className="ml-2 text-xs opacity-70">{tierConfig.label}</span>
        <svg
          className={`w-4 h-4 ml-2 transition-transform ${showDetails ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Details dropdown */}
      {showDetails && (
        <div className="absolute z-10 mt-2 w-72 bg-white dark:bg-gray-900 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="space-y-3">
            {/* Verification status */}
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                Verification Status
              </span>
              <span className={`text-sm font-medium ${isVerified ? 'text-emerald-600' : 'text-gray-500'}`}>
                {isVerified ? 'Verified' : 'Unverified'}
              </span>
            </div>

            {/* Epistemic tier */}
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                Epistemic Tier
              </span>
              <span className="text-sm">
                <span className="font-medium">Tier {tier}</span>
                <span className="text-gray-500 ml-1">({tierConfig.label})</span>
              </span>
            </div>

            {/* Proof type */}
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                Proof Type
              </span>
              <span className="text-sm">
                {proofConfig.icon} {proofConfig.label}
              </span>
            </div>

            {/* Credential details */}
            {claim.credential && (
              <>
                <hr className="border-gray-100 dark:border-gray-800" />
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                      Credential Type
                    </span>
                    <span className="text-sm">
                      {credentialTypeConfig[claim.credential.credential_type]?.icon}{' '}
                      {credentialTypeConfig[claim.credential.credential_type]?.label}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                      Issuer
                    </span>
                    <span className="text-sm font-mono text-gray-600 dark:text-gray-400 truncate max-w-[150px]">
                      {claim.credential.issuer_did.slice(-12)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                      Issued
                    </span>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {new Date(claim.credential.issued_at).toLocaleDateString()}
                    </span>
                  </div>
                  {claim.credential.expires_at && (
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                        Expires
                      </span>
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {new Date(claim.credential.expires_at).toLocaleDateString()}
                      </span>
                    </div>
                  )}
                </div>
              </>
            )}

            {/* Tier explanation */}
            <div className="pt-2 border-t border-gray-100 dark:border-gray-800">
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {tierConfig.description}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Assurance level badge
export function AssuranceLevelBadge({
  level,
  showCost = false,
  className = '',
}: {
  level: AssuranceLevel;
  showCost?: boolean;
  className?: string;
}) {
  const config = assuranceLevelConfig[level];

  return (
    <span className={`inline-flex items-center text-xs font-medium ${config.color} ${className}`}>
      <span className="w-2 h-2 rounded-full bg-current mr-1.5" />
      {config.label}
      {showCost && (
        <span className="ml-1 text-gray-400">
          (attack cost: {config.cost})
        </span>
      )}
    </span>
  );
}

// Claims list for email view
export function EmailClaimsList({
  claims,
  verifications,
  className = '',
}: {
  claims: VerifiableClaim[];
  verifications?: ClaimVerification[];
  className?: string;
}) {
  if (claims.length === 0) {
    return null;
  }

  return (
    <div className={`space-y-2 ${className}`}>
      <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
        Sender Claims
      </h4>
      <div className="flex flex-wrap gap-2">
        {claims.map((claim, i) => (
          <ClaimBadge
            key={i}
            claim={claim}
            verification={verifications?.[i]}
            compact
          />
        ))}
      </div>
    </div>
  );
}
