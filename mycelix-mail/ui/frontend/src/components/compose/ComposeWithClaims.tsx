// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Compose With Claims Component
 *
 * Enhances email composition with the ability to attach verifiable claims:
 * - Select from user's available credentials
 * - Attach identity proofs to establish trust
 * - Include organization memberships
 * - Sign messages with verifiable signatures
 *
 * This component can be embedded in the compose flow or used standalone.
 */

import { useState, useMemo } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '@/services/api';
import type { VerifiableClaim, AssuranceLevel, ProofType } from '@/services/api';

interface ComposeWithClaimsProps {
  recipientEmail?: string;
  recipientDid?: string;
  onClaimsChange?: (claims: AttachedClaim[]) => void;
  compact?: boolean;
}

export interface AttachedClaim {
  credentialId: string;
  claimType: string;
  displayName: string;
  assuranceLevel: AssuranceLevel;
  includeProof: boolean;
}

// Proof type configuration
const proofTypeConfig: Record<ProofType, {
  label: string;
  icon: string;
  description: string;
}> = {
  email_verification: {
    label: 'Email Verified',
    icon: '📧',
    description: 'Email address ownership verified',
  },
  domain_verification: {
    label: 'Domain Verified',
    icon: '🌐',
    description: 'Domain ownership via DNS/TXT record',
  },
  gitcoin_passport: {
    label: 'Gitcoin Passport',
    icon: '🛂',
    description: 'Sybil-resistant identity proof',
  },
  github_verification: {
    label: 'GitHub Verified',
    icon: '🐙',
    description: 'GitHub account ownership',
  },
  social_verification: {
    label: 'Social Verified',
    icon: '👥',
    description: 'Social media account verified',
  },
  organization_membership: {
    label: 'Organization',
    icon: '🏢',
    description: 'Verified organization membership',
  },
  professional_credential: {
    label: 'Professional',
    icon: '🎓',
    description: 'Professional certification or credential',
  },
  government_id: {
    label: 'Government ID',
    icon: '🪪',
    description: 'Government-issued identity document',
  },
  biometric: {
    label: 'Biometric',
    icon: '🔐',
    description: 'Biometric verification (face/fingerprint)',
  },
  cryptographic_signature: {
    label: 'Crypto Signature',
    icon: '🔏',
    description: 'Cryptographic key ownership proof',
  },
};

// Assurance level styling
const assuranceLevelConfig: Record<AssuranceLevel, {
  label: string;
  shortLabel: string;
  color: string;
  bgColor: string;
}> = {
  e0_anonymous: {
    label: 'Anonymous',
    shortLabel: 'E0',
    color: 'text-gray-600',
    bgColor: 'bg-gray-100 dark:bg-gray-800',
  },
  e1_verified_email: {
    label: 'Email Verified',
    shortLabel: 'E1',
    color: 'text-blue-600',
    bgColor: 'bg-blue-50 dark:bg-blue-900/30',
  },
  e2_gitcoin_passport: {
    label: 'Gitcoin Passport',
    shortLabel: 'E2',
    color: 'text-indigo-600',
    bgColor: 'bg-indigo-50 dark:bg-indigo-900/30',
  },
  e3_multi_factor: {
    label: 'Multi-Factor',
    shortLabel: 'E3',
    color: 'text-purple-600',
    bgColor: 'bg-purple-50 dark:bg-purple-900/30',
  },
  e4_constitutional: {
    label: 'Constitutional',
    shortLabel: 'E4',
    color: 'text-emerald-600',
    bgColor: 'bg-emerald-50 dark:bg-emerald-900/30',
  },
};

// Credential card component
function CredentialCard({
  credential,
  isSelected,
  onToggle,
  onToggleProof,
  includeProof,
}: {
  credential: {
    id: string;
    type: ProofType;
    display_name: string;
    assurance_level: AssuranceLevel;
    issued_at: string;
    expires_at?: string;
    verified: boolean;
  };
  isSelected: boolean;
  onToggle: () => void;
  onToggleProof: () => void;
  includeProof: boolean;
}) {
  const proofConfig = proofTypeConfig[credential.type];
  const assuranceConfig = assuranceLevelConfig[credential.assurance_level];
  const isExpired = credential.expires_at && new Date(credential.expires_at) < new Date();

  return (
    <div
      className={`relative p-3 rounded-lg border-2 transition-all cursor-pointer ${
        isSelected
          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
          : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
      } ${isExpired ? 'opacity-50' : ''}`}
      onClick={onToggle}
    >
      {/* Selection indicator */}
      <div className={`absolute top-2 right-2 w-5 h-5 rounded-full border-2 flex items-center justify-center ${
        isSelected
          ? 'border-blue-500 bg-blue-500'
          : 'border-gray-300 dark:border-gray-600'
      }`}>
        {isSelected && (
          <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
        )}
      </div>

      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className="text-2xl">{proofConfig.icon}</div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-900 dark:text-gray-100 truncate">
              {credential.display_name}
            </span>
            {credential.verified && (
              <span className="text-emerald-500" title="Verified">✓</span>
            )}
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            {proofConfig.label}
          </p>
          <div className="flex items-center gap-2 mt-1">
            <span className={`text-xs px-1.5 py-0.5 rounded ${assuranceConfig.bgColor} ${assuranceConfig.color}`}>
              {assuranceConfig.shortLabel}
            </span>
            {isExpired && (
              <span className="text-xs text-red-500">Expired</span>
            )}
          </div>
        </div>
      </div>

      {/* Include proof toggle */}
      {isSelected && (
        <div
          className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between"
          onClick={(e) => e.stopPropagation()}
        >
          <span className="text-xs text-gray-600 dark:text-gray-400">
            Include cryptographic proof
          </span>
          <button
            type="button"
            onClick={onToggleProof}
            className={`relative w-10 h-5 rounded-full transition-colors ${
              includeProof ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'
            }`}
          >
            <span
              className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${
                includeProof ? 'translate-x-5' : 'translate-x-0.5'
              }`}
            />
          </button>
        </div>
      )}
    </div>
  );
}

// Quick attach badges
function QuickAttachBadges({
  credentials,
  selectedIds,
  onToggle,
}: {
  credentials: Array<{ id: string; type: ProofType; display_name: string }>;
  selectedIds: Set<string>;
  onToggle: (id: string) => void;
}) {
  // Group by type for quick selection
  const groupedByType = useMemo(() => {
    const groups: Record<string, typeof credentials> = {};
    credentials.forEach((cred) => {
      if (!groups[cred.type]) groups[cred.type] = [];
      groups[cred.type].push(cred);
    });
    return groups;
  }, [credentials]);

  return (
    <div className="flex flex-wrap gap-2">
      {Object.entries(groupedByType).map(([type, creds]) => {
        const config = proofTypeConfig[type as ProofType];
        const anySelected = creds.some((c) => selectedIds.has(c.id));

        return (
          <button
            key={type}
            type="button"
            onClick={() => {
              // Toggle first credential of this type
              if (creds.length > 0) {
                onToggle(creds[0].id);
              }
            }}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-xs font-medium transition-colors ${
              anySelected
                ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 ring-1 ring-blue-500'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <span>{config.icon}</span>
            <span>{config.label}</span>
            {creds.length > 1 && (
              <span className="text-gray-400">({creds.length})</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

// Attached claims summary
function AttachedClaimsSummary({
  claims,
  onRemove,
}: {
  claims: AttachedClaim[];
  onRemove: (credentialId: string) => void;
}) {
  if (claims.length === 0) return null;

  // Calculate combined assurance level
  const highestAssurance = claims.reduce((highest, claim) => {
    const levels: AssuranceLevel[] = [
      'e0_anonymous',
      'e1_verified_email',
      'e2_gitcoin_passport',
      'e3_multi_factor',
      'e4_constitutional',
    ];
    const currentIndex = levels.indexOf(claim.assuranceLevel);
    const highestIndex = levels.indexOf(highest);
    return currentIndex > highestIndex ? claim.assuranceLevel : highest;
  }, 'e0_anonymous' as AssuranceLevel);

  const assuranceConfig = assuranceLevelConfig[highestAssurance];

  return (
    <div className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-emerald-700 dark:text-emerald-300">
          {claims.length} claim{claims.length !== 1 ? 's' : ''} attached
        </span>
        <span className={`text-xs px-2 py-0.5 rounded-full ${assuranceConfig.bgColor} ${assuranceConfig.color}`}>
          Up to {assuranceConfig.shortLabel}
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {claims.map((claim) => (
          <span
            key={claim.credentialId}
            className="inline-flex items-center gap-1 px-2 py-1 bg-white dark:bg-gray-800 rounded text-xs text-gray-700 dark:text-gray-300 shadow-sm"
          >
            {claim.displayName}
            {claim.includeProof && (
              <span className="text-emerald-500" title="With proof">🔏</span>
            )}
            <button
              type="button"
              onClick={() => onRemove(claim.credentialId)}
              className="ml-1 text-gray-400 hover:text-red-500"
            >
              ×
            </button>
          </span>
        ))}
      </div>
    </div>
  );
}

// Main component
export default function ComposeWithClaims({
  recipientEmail,
  recipientDid,
  onClaimsChange,
  compact = false,
}: ComposeWithClaimsProps) {
  const [isExpanded, setIsExpanded] = useState(!compact);
  const [selectedCredentials, setSelectedCredentials] = useState<Map<string, { includeProof: boolean }>>(new Map());

  // Fetch user's credentials
  const { data: credentialsData, isLoading } = useQuery({
    queryKey: ['credentials'],
    queryFn: () => api.claims.listCredentials(),
  });

  const credentials = credentialsData?.credentials || [];

  // Convert selection to attached claims
  const attachedClaims: AttachedClaim[] = useMemo(() => {
    return credentials
      .filter((cred) => selectedCredentials.has(cred.id))
      .map((cred) => ({
        credentialId: cred.id,
        claimType: cred.type,
        displayName: cred.display_name,
        assuranceLevel: cred.assurance_level,
        includeProof: selectedCredentials.get(cred.id)?.includeProof || false,
      }));
  }, [credentials, selectedCredentials]);

  // Notify parent of changes
  const handleToggleCredential = (credentialId: string) => {
    setSelectedCredentials((prev) => {
      const next = new Map(prev);
      if (next.has(credentialId)) {
        next.delete(credentialId);
      } else {
        next.set(credentialId, { includeProof: false });
      }
      return next;
    });

    // Update attached claims for parent
    setTimeout(() => {
      const newClaims = credentials
        .filter((cred) => {
          const isSelected = selectedCredentials.has(cred.id);
          const willBeSelected = cred.id === credentialId ? !isSelected : isSelected;
          return willBeSelected;
        })
        .map((cred) => ({
          credentialId: cred.id,
          claimType: cred.type,
          displayName: cred.display_name,
          assuranceLevel: cred.assurance_level,
          includeProof: selectedCredentials.get(cred.id)?.includeProof || false,
        }));
      onClaimsChange?.(newClaims);
    }, 0);
  };

  const handleToggleProof = (credentialId: string) => {
    setSelectedCredentials((prev) => {
      const next = new Map(prev);
      const current = next.get(credentialId);
      if (current) {
        next.set(credentialId, { includeProof: !current.includeProof });
      }
      return next;
    });
  };

  const handleRemoveClaim = (credentialId: string) => {
    handleToggleCredential(credentialId);
  };

  // Compact view
  if (compact && !isExpanded) {
    return (
      <div className="border-t border-gray-200 dark:border-gray-700 pt-3">
        <button
          type="button"
          onClick={() => setIsExpanded(true)}
          className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          <span>Attach verifiable claims</span>
          {attachedClaims.length > 0 && (
            <span className="px-1.5 py-0.5 bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400 rounded text-xs">
              {attachedClaims.length}
            </span>
          )}
        </button>

        {attachedClaims.length > 0 && (
          <div className="mt-2">
            <AttachedClaimsSummary claims={attachedClaims} onRemove={handleRemoveClaim} />
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <svg className="w-5 h-5 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Attach Verifiable Claims
          </h3>
        </div>
        {compact && (
          <button
            type="button"
            onClick={() => setIsExpanded(false)}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
            </svg>
          </button>
        )}
      </div>

      {/* Description */}
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
        Attach claims to establish your identity and build trust with the recipient.
        {recipientEmail && (
          <span className="block mt-1">
            Sending to: <span className="font-medium">{recipientEmail}</span>
          </span>
        )}
      </p>

      {/* Attached claims summary */}
      {attachedClaims.length > 0 && (
        <div className="mb-4">
          <AttachedClaimsSummary claims={attachedClaims} onRemove={handleRemoveClaim} />
        </div>
      )}

      {/* Quick attach badges */}
      {credentials.length > 0 && (
        <div className="mb-4">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wider">
            Quick Attach
          </p>
          <QuickAttachBadges
            credentials={credentials}
            selectedIds={new Set(selectedCredentials.keys())}
            onToggle={handleToggleCredential}
          />
        </div>
      )}

      {/* Credentials list */}
      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500" />
        </div>
      ) : credentials.length === 0 ? (
        <div className="text-center py-6 px-4 rounded-lg bg-gray-50 dark:bg-gray-800/50">
          <svg className="w-10 h-10 text-gray-300 dark:text-gray-600 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
            No credentials available
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500">
            Verify your identity to attach claims to emails
          </p>
          <button
            type="button"
            className="mt-3 px-4 py-2 text-xs font-medium text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg transition-colors"
          >
            Get Verified
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
            Your Credentials ({credentials.length})
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            {credentials.map((credential) => (
              <CredentialCard
                key={credential.id}
                credential={credential}
                isSelected={selectedCredentials.has(credential.id)}
                onToggle={() => handleToggleCredential(credential.id)}
                onToggleProof={() => handleToggleProof(credential.id)}
                includeProof={selectedCredentials.get(credential.id)?.includeProof || false}
              />
            ))}
          </div>
        </div>
      )}

      {/* Info about claims */}
      <div className="mt-4 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
        <p className="text-xs text-blue-700 dark:text-blue-300">
          <strong>How it works:</strong> Attached claims are cryptographically signed and can be independently verified by the recipient.
          Including proofs allows verification without contacting the issuer.
        </p>
      </div>
    </div>
  );
}

// Export sub-components for flexible use
export { CredentialCard, QuickAttachBadges, AttachedClaimsSummary };
