// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Contact Profile Component
 *
 * A comprehensive view of a contact showing:
 * - Contact information and avatar
 * - Trust score history over time (sparkline)
 * - Assurance level and verification status
 * - Trust path from user to contact
 * - All attestations (given and received)
 * - Communication statistics
 * - Actions (create attestation, view trust graph)
 */

import { useState, useMemo } from 'react';
import {
  useSenderTrust,
  useAssuranceLevel,
  useTrustedBy,
  useTrusters,
  useCreateAttestation,
} from '@/hooks/useEpistemicServices';
import type {
  AssuranceLevel,
  TrustEdge,
  RelationType,
} from '@/services/api';

interface ContactProfileProps {
  contactDid: string;
  contactEmail: string;
  contactName?: string;
  userDid: string;
  onClose?: () => void;
  onViewInGraph?: (did: string) => void;
}

// Assurance level configuration
const assuranceConfig: Record<AssuranceLevel, {
  label: string;
  shortLabel: string;
  color: string;
  bgColor: string;
  description: string;
  attackCost: string;
}> = {
  e0_anonymous: {
    label: 'Anonymous',
    shortLabel: 'E0',
    color: 'text-gray-600',
    bgColor: 'bg-gray-100 dark:bg-gray-800',
    description: 'No verified identity',
    attackCost: '$0',
  },
  e1_verified_email: {
    label: 'Email Verified',
    shortLabel: 'E1',
    color: 'text-blue-600',
    bgColor: 'bg-blue-50 dark:bg-blue-900/30',
    description: 'Email address verified',
    attackCost: '$100',
  },
  e2_gitcoin_passport: {
    label: 'Gitcoin Passport',
    shortLabel: 'E2',
    color: 'text-indigo-600',
    bgColor: 'bg-indigo-50 dark:bg-indigo-900/30',
    description: 'Sybil-resistant identity',
    attackCost: '$1,000',
  },
  e3_multi_factor: {
    label: 'Multi-Factor Verified',
    shortLabel: 'E3',
    color: 'text-purple-600',
    bgColor: 'bg-purple-50 dark:bg-purple-900/30',
    description: 'Multiple identity proofs',
    attackCost: '$100,000',
  },
  e4_constitutional: {
    label: 'Constitutional Identity',
    shortLabel: 'E4',
    color: 'text-emerald-600',
    bgColor: 'bg-emerald-50 dark:bg-emerald-900/30',
    description: 'Highest assurance level',
    attackCost: '$10,000,000',
  },
};

// Relationship type labels
const relationshipLabels: Record<string, string> = {
  direct_trust: 'Direct Trust',
  introduction: 'Introduction',
  organization_member: 'Organization',
  credential_issuer: 'Credential Issuer',
  transitive_trust: 'Transitive',
  vouch: 'Vouch',
};

// Generate initials from name or email
function getInitials(name?: string, email?: string): string {
  if (name) {
    return name
      .split(' ')
      .map(n => n[0])
      .slice(0, 2)
      .join('')
      .toUpperCase();
  }
  if (email) {
    return email[0].toUpperCase();
  }
  return '?';
}

// Hash string to color
function stringToColor(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  const colors = [
    'bg-red-500', 'bg-orange-500', 'bg-amber-500', 'bg-yellow-500',
    'bg-lime-500', 'bg-green-500', 'bg-emerald-500', 'bg-teal-500',
    'bg-cyan-500', 'bg-sky-500', 'bg-blue-500', 'bg-indigo-500',
    'bg-violet-500', 'bg-purple-500', 'bg-fuchsia-500', 'bg-pink-500',
  ];
  return colors[Math.abs(hash) % colors.length];
}

// Simple sparkline component for trust history
function TrustSparkline({ data }: { data: number[] }) {
  if (data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((value, i) => {
    const x = (i / (data.length - 1)) * 100;
    const y = 100 - ((value - min) / range) * 100;
    return `${x},${y}`;
  }).join(' ');

  const trend = data[data.length - 1] - data[0];

  return (
    <div className="flex items-center gap-2">
      <svg viewBox="0 0 100 100" className="w-20 h-8" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke={trend >= 0 ? '#10b981' : '#ef4444'}
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span className={`text-xs font-medium ${trend >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
        {trend >= 0 ? '+' : ''}{(trend * 100).toFixed(0)}%
      </span>
    </div>
  );
}

// Trust path visualization
function TrustPathVisualization({
  path,
  userDid,
  contactDid,
}: {
  path?: { hops: Array<{ from_did: string; to_did: string; weight: number }>; total_weight: number };
  userDid: string;
  contactDid: string;
}) {
  if (!path || path.hops.length === 0) {
    return (
      <div className="p-4 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
        <p className="text-sm text-amber-700 dark:text-amber-300 font-medium">
          No Trust Path
        </p>
        <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
          This contact is not in your trust network. Consider requesting an introduction.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 overflow-x-auto py-2">
        {/* You */}
        <div className="flex-shrink-0 flex flex-col items-center">
          <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center text-white text-sm font-medium">
            You
          </div>
        </div>

        {path.hops.map((hop, i) => (
          <div key={i} className="flex items-center">
            {/* Arrow with weight */}
            <div className="flex flex-col items-center px-2">
              <div className="w-12 h-0.5 bg-gray-300 dark:bg-gray-600 relative">
                <div
                  className="absolute inset-y-0 left-0 bg-emerald-500"
                  style={{ width: `${hop.weight * 100}%` }}
                />
              </div>
              <span className="text-xs text-gray-500 mt-1">{Math.round(hop.weight * 100)}%</span>
            </div>

            {/* Hop node */}
            <div className="flex-shrink-0 flex flex-col items-center">
              <div className="w-10 h-10 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-gray-600 dark:text-gray-300 text-xs">
                {i + 1}
              </div>
            </div>
          </div>
        ))}

        {/* Final arrow to contact */}
        <div className="flex items-center">
          <div className="flex flex-col items-center px-2">
            <div className="w-12 h-0.5 bg-emerald-500" />
          </div>
          <div className="flex-shrink-0 flex flex-col items-center">
            <div className="w-10 h-10 rounded-full bg-emerald-500 flex items-center justify-center text-white text-xs font-medium">
              {getInitials(undefined, contactDid.slice(-8))}
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-500 dark:text-gray-400">
          {path.hops.length} hop{path.hops.length !== 1 ? 's' : ''} away
        </span>
        <span className={`font-medium ${
          path.total_weight >= 0.7 ? 'text-emerald-600' :
          path.total_weight >= 0.4 ? 'text-amber-600' :
          'text-red-600'
        }`}>
          {Math.round(path.total_weight * 100)}% transitive trust
        </span>
      </div>
    </div>
  );
}

// Attestation list
function AttestationList({
  title,
  edges,
  emptyMessage,
}: {
  title: string;
  edges: TrustEdge[];
  emptyMessage: string;
}) {
  if (edges.length === 0) {
    return (
      <div className="text-sm text-gray-500 dark:text-gray-400 italic">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
        {title}
      </h4>
      <div className="space-y-2">
        {edges.slice(0, 5).map((edge, i) => (
          <div
            key={i}
            className="flex items-center justify-between p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50"
          >
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-xs">
                {getInitials(undefined, edge.from_did.slice(-8))}
              </div>
              <div>
                <p className="text-sm text-gray-700 dark:text-gray-300 truncate max-w-[120px]">
                  {edge.from_did.slice(-12)}
                </p>
                <p className="text-xs text-gray-500">
                  {relationshipLabels[edge.relationship.toString().toLowerCase()] || edge.relationship}
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className={`text-sm font-medium ${
                edge.trust_score >= 0.7 ? 'text-emerald-600' :
                edge.trust_score >= 0.4 ? 'text-amber-600' :
                'text-red-600'
              }`}>
                {Math.round(edge.trust_score * 100)}%
              </p>
            </div>
          </div>
        ))}
        {edges.length > 5 && (
          <p className="text-xs text-gray-500 text-center">
            +{edges.length - 5} more
          </p>
        )}
      </div>
    </div>
  );
}

// Create attestation form
function CreateAttestationForm({
  contactDid,
  onSuccess,
}: {
  contactDid: string;
  onSuccess?: () => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [relationship, setRelationship] = useState<string>('direct_trust');
  const [trustScore, setTrustScore] = useState(0.7);
  const [message, setMessage] = useState('');

  const createAttestation = useCreateAttestation();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await createAttestation.mutateAsync({
        subject_did: contactDid,
        relationship: relationship as RelationType,
        trust_score: trustScore,
        message: message || 'Trust attestation',
        expires_days: 365,
      });
      setIsOpen(false);
      setMessage('');
      onSuccess?.();
    } catch (error) {
      console.error('Failed to create attestation:', error);
    }
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="w-full py-2.5 px-4 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
      >
        Create Trust Attestation
      </button>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4 p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Relationship Type
        </label>
        <select
          value={relationship}
          onChange={(e) => setRelationship(e.target.value)}
          className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-sm"
        >
          <option value="direct_trust">Direct Trust (Personal Contact)</option>
          <option value="vouch">Vouch (Professional Reference)</option>
          <option value="organization_member">Organization Member</option>
          <option value="introduction">Introduction</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Trust Level: {Math.round(trustScore * 100)}%
        </label>
        <input
          type="range"
          min="0.1"
          max="1"
          step="0.05"
          value={trustScore}
          onChange={(e) => setTrustScore(parseFloat(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Low</span>
          <span>Medium</span>
          <span>High</span>
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Message (optional)
        </label>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Why do you trust this person?"
          className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-sm resize-none"
          rows={2}
        />
      </div>

      <div className="flex gap-2">
        <button
          type="submit"
          disabled={createAttestation.isPending}
          className="flex-1 py-2 px-4 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50"
        >
          {createAttestation.isPending ? 'Creating...' : 'Create Attestation'}
        </button>
        <button
          type="button"
          onClick={() => setIsOpen(false)}
          className="py-2 px-4 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
        >
          Cancel
        </button>
      </div>
    </form>
  );
}

// Main component
export default function ContactProfile({
  contactDid,
  contactEmail,
  contactName,
  userDid,
  onClose,
  onViewInGraph,
}: ContactProfileProps) {
  const { path, hasPath, trustWeight, pathLength, isLoading: trustLoading } = useSenderTrust(contactDid, userDid);
  const { data: assurance, isLoading: assuranceLoading } = useAssuranceLevel(contactDid);
  const { data: trustedByData } = useTrustedBy(contactDid);
  const { data: trustersData } = useTrusters(contactDid);

  const assuranceInfo = assurance ? assuranceConfig[assurance.level] : assuranceConfig.e0_anonymous;

  // Mock trust history data (would come from persistence service)
  const trustHistory = useMemo(() => {
    if (!hasPath) return [];
    // Simulate historical trust scores
    const base = trustWeight;
    return Array.from({ length: 10 }, (_, i) => {
      const variance = (Math.random() - 0.5) * 0.2;
      return Math.max(0, Math.min(1, base + variance - (i * 0.02)));
    }).reverse();
  }, [hasPath, trustWeight]);

  const isLoading = trustLoading || assuranceLoading;

  return (
    <div className="h-full flex flex-col bg-white dark:bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
          Contact Profile
        </h2>
        {onClose && (
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {/* Contact Header */}
        <div className="p-4 border-b border-gray-100 dark:border-gray-800">
          <div className="flex items-center gap-4">
            {/* Avatar */}
            <div className={`w-16 h-16 rounded-full ${stringToColor(contactEmail)} flex items-center justify-center text-white text-xl font-medium`}>
              {getInitials(contactName, contactEmail)}
            </div>

            {/* Info */}
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 truncate">
                {contactName || contactEmail}
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400 truncate">
                {contactEmail}
              </p>
              <p className="text-xs text-gray-400 dark:text-gray-500 truncate mt-1 font-mono">
                {contactDid.slice(0, 30)}...
              </p>
            </div>
          </div>

          {/* Trust Score Summary */}
          {!isLoading && (
            <div className="mt-4 flex items-center gap-4">
              {/* Trust Score */}
              <div className="flex-1 p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Trust Score</p>
                <div className="flex items-center gap-2">
                  <span className={`text-2xl font-bold ${
                    trustWeight >= 0.7 ? 'text-emerald-600' :
                    trustWeight >= 0.4 ? 'text-amber-600' :
                    hasPath ? 'text-red-600' : 'text-gray-400'
                  }`}>
                    {hasPath ? Math.round(trustWeight * 100) : '—'}
                  </span>
                  {trustHistory.length > 0 && <TrustSparkline data={trustHistory} />}
                </div>
              </div>

              {/* Assurance Level */}
              <div className={`p-3 rounded-lg ${assuranceInfo.bgColor}`}>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Assurance</p>
                <p className={`text-lg font-bold ${assuranceInfo.color}`}>
                  {assuranceInfo.shortLabel}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Assurance Details */}
        <div className="p-4 border-b border-gray-100 dark:border-gray-800">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Identity Assurance
          </h4>
          <div className={`p-3 rounded-lg ${assuranceInfo.bgColor}`}>
            <div className="flex items-center justify-between mb-2">
              <span className={`text-sm font-medium ${assuranceInfo.color}`}>
                {assuranceInfo.label}
              </span>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                Attack cost: {assuranceInfo.attackCost}
              </span>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              {assuranceInfo.description}
            </p>
          </div>
        </div>

        {/* Trust Path */}
        <div className="p-4 border-b border-gray-100 dark:border-gray-800">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Trust Path
          </h4>
          <TrustPathVisualization
            path={path}
            userDid={userDid}
            contactDid={contactDid}
          />
        </div>

        {/* Attestations */}
        <div className="p-4 border-b border-gray-100 dark:border-gray-800 space-y-4">
          <AttestationList
            title="Trusts This Contact"
            edges={trustersData?.edges || []}
            emptyMessage="No one has attested trust for this contact yet"
          />
          <AttestationList
            title="Trusted By This Contact"
            edges={trustedByData?.edges || []}
            emptyMessage="This contact hasn't attested trust for anyone"
          />
        </div>

        {/* Actions */}
        <div className="p-4 space-y-3">
          <CreateAttestationForm contactDid={contactDid} />

          {onViewInGraph && (
            <button
              onClick={() => onViewInGraph(contactDid)}
              className="w-full py-2.5 px-4 text-sm font-medium text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-800 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg transition-colors"
            >
              View in Trust Graph
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
