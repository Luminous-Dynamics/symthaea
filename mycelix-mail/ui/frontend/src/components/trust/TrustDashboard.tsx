// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Trust Dashboard Component
 *
 * Comprehensive dashboard for managing your trust network:
 * - Trust network overview and statistics
 * - Attestation management (create, view, revoke)
 * - Trust graph visualization
 * - Pending introductions and recommendations
 * - Trust score analytics over time
 *
 * Central hub for all epistemic identity operations.
 */

import { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/services/api';
import {
  useTrustGraph,
  useTrustedBy,
  useTrusters,
  useCreateAttestation,
} from '@/hooks/useEpistemicServices';
import type { TrustEdge, RelationType, AssuranceLevel } from '@/services/api';

interface TrustDashboardProps {
  userDid: string;
  onViewContact?: (did: string) => void;
  onViewGraph?: () => void;
  className?: string;
}

// Relationship type labels and colors
const relationshipConfig: Record<string, {
  label: string;
  icon: string;
  color: string;
  bgColor: string;
}> = {
  direct_trust: {
    label: 'Direct Trust',
    icon: '🤝',
    color: 'text-blue-600',
    bgColor: 'bg-blue-50 dark:bg-blue-900/30',
  },
  introduction: {
    label: 'Introduction',
    icon: '👋',
    color: 'text-indigo-600',
    bgColor: 'bg-indigo-50 dark:bg-indigo-900/30',
  },
  organization_member: {
    label: 'Organization',
    icon: '🏢',
    color: 'text-purple-600',
    bgColor: 'bg-purple-50 dark:bg-purple-900/30',
  },
  credential_issuer: {
    label: 'Credential Issuer',
    icon: '🎓',
    color: 'text-teal-600',
    bgColor: 'bg-teal-50 dark:bg-teal-900/30',
  },
  transitive_trust: {
    label: 'Transitive',
    icon: '🔗',
    color: 'text-gray-600',
    bgColor: 'bg-gray-50 dark:bg-gray-800',
  },
  vouch: {
    label: 'Vouch',
    icon: '✅',
    color: 'text-emerald-600',
    bgColor: 'bg-emerald-50 dark:bg-emerald-900/30',
  },
};

// Statistics card
function StatCard({
  label,
  value,
  subValue,
  icon,
  trend,
  color = 'blue',
}: {
  label: string;
  value: number | string;
  subValue?: string;
  icon: string;
  trend?: { value: number; isPositive: boolean };
  color?: string;
}) {
  const colorClasses: Record<string, string> = {
    blue: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
    emerald: 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800',
    purple: 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800',
    amber: 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800',
  };

  return (
    <div className={`p-4 rounded-lg border ${colorClasses[color]}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">{label}</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-gray-100 mt-1">
            {value}
          </p>
          {subValue && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{subValue}</p>
          )}
        </div>
        <span className="text-2xl">{icon}</span>
      </div>
      {trend && (
        <div className={`mt-2 text-xs font-medium ${
          trend.isPositive ? 'text-emerald-600' : 'text-red-600'
        }`}>
          {trend.isPositive ? '↑' : '↓'} {Math.abs(trend.value)}% this month
        </div>
      )}
    </div>
  );
}

// Trust score gauge
function TrustScoreGauge({ score, label }: { score: number; label: string }) {
  const angle = (score * 180) - 90; // -90 to 90 degrees

  const getColor = () => {
    if (score >= 0.7) return '#10b981'; // emerald
    if (score >= 0.4) return '#f59e0b'; // amber
    return '#ef4444'; // red
  };

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 100 60" className="w-32 h-20">
        {/* Background arc */}
        <path
          d="M 10 50 A 40 40 0 0 1 90 50"
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="8"
          strokeLinecap="round"
        />
        {/* Colored arc */}
        <path
          d="M 10 50 A 40 40 0 0 1 90 50"
          fill="none"
          stroke={getColor()}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={`${score * 126} 126`}
        />
        {/* Needle */}
        <line
          x1="50"
          y1="50"
          x2={50 + Math.cos((angle * Math.PI) / 180) * 30}
          y2={50 + Math.sin((angle * Math.PI) / 180) * 30}
          stroke="#374151"
          strokeWidth="2"
          strokeLinecap="round"
        />
        {/* Center dot */}
        <circle cx="50" cy="50" r="4" fill="#374151" />
      </svg>
      <div className="text-center mt-1">
        <p className="text-lg font-bold text-gray-900 dark:text-gray-100">
          {Math.round(score * 100)}%
        </p>
        <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
      </div>
    </div>
  );
}

// Attestation list item
function AttestationItem({
  edge,
  isOutgoing,
  onView,
  onRevoke,
}: {
  edge: TrustEdge;
  isOutgoing: boolean;
  onView?: (did: string) => void;
  onRevoke?: (edge: TrustEdge) => void;
}) {
  const relationshipType = typeof edge.relationship === 'string'
    ? edge.relationship.toLowerCase()
    : 'direct_trust';
  const config = relationshipConfig[relationshipType] || relationshipConfig.direct_trust;
  const contactDid = isOutgoing ? edge.to_did : edge.from_did;

  return (
    <div className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors group">
      <div className="flex items-center gap-3">
        {/* Avatar */}
        <div className={`w-10 h-10 rounded-full ${config.bgColor} flex items-center justify-center text-lg`}>
          {config.icon}
        </div>

        {/* Info */}
        <div>
          <p
            className="text-sm font-medium text-gray-900 dark:text-gray-100 cursor-pointer hover:text-blue-600"
            onClick={() => onView?.(contactDid)}
          >
            {contactDid.slice(0, 20)}...{contactDid.slice(-8)}
          </p>
          <div className="flex items-center gap-2 mt-0.5">
            <span className={`text-xs ${config.color}`}>{config.label}</span>
            <span className="text-xs text-gray-400">•</span>
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {new Date(edge.established_at).toLocaleDateString()}
            </span>
          </div>
        </div>
      </div>

      {/* Trust score and actions */}
      <div className="flex items-center gap-3">
        <span className={`text-sm font-medium ${
          edge.trust_score >= 0.7 ? 'text-emerald-600' :
          edge.trust_score >= 0.4 ? 'text-amber-600' :
          'text-red-600'
        }`}>
          {Math.round(edge.trust_score * 100)}%
        </span>
        {isOutgoing && onRevoke && (
          <button
            onClick={() => onRevoke(edge)}
            className="opacity-0 group-hover:opacity-100 text-xs text-red-500 hover:text-red-700 transition-opacity"
          >
            Revoke
          </button>
        )}
      </div>
    </div>
  );
}

// Create attestation form
function CreateAttestationForm({
  onSuccess,
}: {
  onSuccess?: () => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [subjectDid, setSubjectDid] = useState('');
  const [relationship, setRelationship] = useState<string>('direct_trust');
  const [trustScore, setTrustScore] = useState(0.7);
  const [message, setMessage] = useState('');
  const [expiresInDays, setExpiresInDays] = useState(365);

  const createAttestation = useCreateAttestation();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await createAttestation.mutateAsync({
        subject_did: subjectDid,
        relationship: relationship as RelationType,
        trust_score: trustScore,
        message: message || 'Trust attestation',
        expires_days: expiresInDays,
      });
      setIsOpen(false);
      setSubjectDid('');
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
        className="w-full py-3 px-4 text-sm font-medium text-blue-600 dark:text-blue-400 border-2 border-dashed border-blue-200 dark:border-blue-800 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg transition-colors flex items-center justify-center gap-2"
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
        </svg>
        Create New Trust Attestation
      </button>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg space-y-4">
      <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
        Create Trust Attestation
      </h4>

      <div>
        <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
          Subject DID (who you're attesting to)
        </label>
        <input
          type="text"
          value={subjectDid}
          onChange={(e) => setSubjectDid(e.target.value)}
          placeholder="did:mycelix:..."
          className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-sm"
          required
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
            Relationship Type
          </label>
          <select
            value={relationship}
            onChange={(e) => setRelationship(e.target.value)}
            className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-sm"
          >
            {Object.entries(relationshipConfig).map(([key, config]) => (
              <option key={key} value={key}>
                {config.icon} {config.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
            Expires In
          </label>
          <select
            value={expiresInDays}
            onChange={(e) => setExpiresInDays(parseInt(e.target.value))}
            className="w-full px-3 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-sm"
          >
            <option value={30}>30 days</option>
            <option value={90}>90 days</option>
            <option value={180}>6 months</option>
            <option value={365}>1 year</option>
            <option value={730}>2 years</option>
          </select>
        </div>
      </div>

      <div>
        <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
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
        <div className="flex justify-between text-xs text-gray-400 mt-1">
          <span>Low (10%)</span>
          <span>High (100%)</span>
        </div>
      </div>

      <div>
        <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
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
          disabled={createAttestation.isPending || !subjectDid.trim()}
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

// Pending introductions list
function PendingIntroductions({
  userDid,
  onAccept,
  onReject,
}: {
  userDid: string;
  onAccept: (introId: string) => void;
  onReject: (introId: string) => void;
}) {
  // Mock pending introductions (would come from API)
  const pendingIntros = [
    {
      id: '1',
      introducerDid: 'did:mycelix:alice',
      introducerName: 'Alice',
      subjectDid: 'did:mycelix:charlie',
      subjectName: 'Charlie',
      message: 'Charlie is a trusted colleague from my previous company.',
      createdAt: new Date().toISOString(),
    },
  ];

  if (pendingIntros.length === 0) {
    return (
      <p className="text-sm text-gray-500 dark:text-gray-400 italic text-center py-4">
        No pending introductions
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {pendingIntros.map((intro) => (
        <div key={intro.id} className="p-4 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-full bg-amber-200 dark:bg-amber-800 flex items-center justify-center text-lg">
              👋
            </div>
            <div className="flex-1">
              <p className="text-sm text-gray-900 dark:text-gray-100">
                <span className="font-medium">{intro.introducerName}</span> wants to introduce you to{' '}
                <span className="font-medium">{intro.subjectName}</span>
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 italic">
                "{intro.message}"
              </p>
              <div className="flex gap-2 mt-3">
                <button
                  onClick={() => onAccept(intro.id)}
                  className="px-3 py-1.5 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-700 rounded transition-colors"
                >
                  Accept
                </button>
                <button
                  onClick={() => onReject(intro.id)}
                  className="px-3 py-1.5 text-xs font-medium text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded transition-colors"
                >
                  Decline
                </button>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// Main component
export default function TrustDashboard({
  userDid,
  onViewContact,
  onViewGraph,
  className = '',
}: TrustDashboardProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'outgoing' | 'incoming' | 'pending'>('overview');
  const queryClient = useQueryClient();

  // Fetch trust data
  const { data: outgoingData } = useTrustedBy(userDid);
  const { data: incomingData } = useTrusters(userDid);
  const { data: graphData } = useTrustGraph(userDid, 2);

  const outgoingEdges = outgoingData?.edges || [];
  const incomingEdges = incomingData?.edges || [];

  // Calculate statistics
  const stats = useMemo(() => {
    const totalConnections = new Set([
      ...outgoingEdges.map(e => e.to_did),
      ...incomingEdges.map(e => e.from_did),
    ]).size;

    const avgTrustGiven = outgoingEdges.length > 0
      ? outgoingEdges.reduce((sum, e) => sum + e.trust_score, 0) / outgoingEdges.length
      : 0;

    const avgTrustReceived = incomingEdges.length > 0
      ? incomingEdges.reduce((sum, e) => sum + e.trust_score, 0) / incomingEdges.length
      : 0;

    // Count by relationship type
    const relationshipCounts: Record<string, number> = {};
    [...outgoingEdges, ...incomingEdges].forEach((e) => {
      const rel = typeof e.relationship === 'string' ? e.relationship.toLowerCase() : 'direct_trust';
      relationshipCounts[rel] = (relationshipCounts[rel] || 0) + 1;
    });

    return {
      totalConnections,
      outgoingCount: outgoingEdges.length,
      incomingCount: incomingEdges.length,
      avgTrustGiven,
      avgTrustReceived,
      relationshipCounts,
    };
  }, [outgoingEdges, incomingEdges]);

  const handleAcceptIntroduction = (introId: string) => {
    console.log('Accept introduction:', introId);
  };

  const handleRejectIntroduction = (introId: string) => {
    console.log('Reject introduction:', introId);
  };

  const handleRevokeAttestation = (edge: TrustEdge) => {
    if (confirm('Are you sure you want to revoke this attestation?')) {
      console.log('Revoke attestation for:', edge.to_did);
      // Would call API to revoke
    }
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'outgoing', label: 'I Trust', icon: '→', count: stats.outgoingCount },
    { id: 'incoming', label: 'Trust Me', icon: '←', count: stats.incomingCount },
    { id: 'pending', label: 'Pending', icon: '⏳', count: 1 },
  ];

  return (
    <div className={`bg-white dark:bg-gray-900 rounded-xl shadow-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-blue-600 to-indigo-600">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white">Trust Dashboard</h2>
            <p className="text-blue-100 text-sm mt-0.5">
              Manage your epistemic network
            </p>
          </div>
          {onViewGraph && (
            <button
              onClick={onViewGraph}
              className="px-4 py-2 text-sm font-medium text-blue-600 bg-white rounded-lg hover:bg-blue-50 transition-colors"
            >
              View Graph
            </button>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors relative ${
              activeTab === tab.id
                ? 'text-blue-600 dark:text-blue-400 bg-white dark:bg-gray-900'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <span className="flex items-center justify-center gap-2">
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
              {tab.count !== undefined && tab.count > 0 && (
                <span className={`px-1.5 py-0.5 rounded-full text-xs ${
                  activeTab === tab.id
                    ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                }`}>
                  {tab.count}
                </span>
              )}
            </span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-6">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Stats Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard
                label="Total Connections"
                value={stats.totalConnections}
                icon="🌐"
                color="blue"
                trend={{ value: 12, isPositive: true }}
              />
              <StatCard
                label="Trust Given"
                value={stats.outgoingCount}
                subValue={`Avg: ${Math.round(stats.avgTrustGiven * 100)}%`}
                icon="→"
                color="emerald"
              />
              <StatCard
                label="Trust Received"
                value={stats.incomingCount}
                subValue={`Avg: ${Math.round(stats.avgTrustReceived * 100)}%`}
                icon="←"
                color="purple"
              />
              <StatCard
                label="Pending Actions"
                value={1}
                icon="⏳"
                color="amber"
              />
            </div>

            {/* Trust Gauges */}
            <div className="grid grid-cols-2 gap-6 py-4">
              <div className="flex flex-col items-center">
                <TrustScoreGauge score={stats.avgTrustGiven} label="Avg Trust Given" />
              </div>
              <div className="flex flex-col items-center">
                <TrustScoreGauge score={stats.avgTrustReceived} label="Avg Trust Received" />
              </div>
            </div>

            {/* Relationship Distribution */}
            <div>
              <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                Relationship Types
              </h3>
              <div className="flex flex-wrap gap-2">
                {Object.entries(stats.relationshipCounts).map(([type, count]) => {
                  const config = relationshipConfig[type] || relationshipConfig.direct_trust;
                  return (
                    <div
                      key={type}
                      className={`flex items-center gap-2 px-3 py-2 rounded-lg ${config.bgColor}`}
                    >
                      <span>{config.icon}</span>
                      <span className={`text-sm font-medium ${config.color}`}>
                        {config.label}
                      </span>
                      <span className="text-sm text-gray-500">({count})</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Quick Create */}
            <CreateAttestationForm
              onSuccess={() => queryClient.invalidateQueries({ queryKey: ['trust'] })}
            />
          </div>
        )}

        {/* Outgoing Tab (I Trust) */}
        {activeTab === 'outgoing' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                People You Trust ({outgoingEdges.length})
              </h3>
            </div>

            {outgoingEdges.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-gray-500 dark:text-gray-400 mb-4">
                  You haven't attested trust to anyone yet
                </p>
                <CreateAttestationForm
                  onSuccess={() => queryClient.invalidateQueries({ queryKey: ['trust'] })}
                />
              </div>
            ) : (
              <div className="space-y-2">
                {outgoingEdges.map((edge, i) => (
                  <AttestationItem
                    key={i}
                    edge={edge}
                    isOutgoing={true}
                    onView={onViewContact}
                    onRevoke={handleRevokeAttestation}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Incoming Tab (Trust Me) */}
        {activeTab === 'incoming' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                People Who Trust You ({incomingEdges.length})
              </h3>
            </div>

            {incomingEdges.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-gray-500 dark:text-gray-400">
                  No one has attested trust to you yet
                </p>
                <p className="text-xs text-gray-400 mt-2">
                  Build your reputation by being trustworthy and asking connections to vouch for you
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                {incomingEdges.map((edge, i) => (
                  <AttestationItem
                    key={i}
                    edge={edge}
                    isOutgoing={false}
                    onView={onViewContact}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Pending Tab */}
        {activeTab === 'pending' && (
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">
              Pending Introductions
            </h3>
            <PendingIntroductions
              userDid={userDid}
              onAccept={handleAcceptIntroduction}
              onReject={handleRejectIntroduction}
            />
          </div>
        )}
      </div>
    </div>
  );
}

// Export sub-components
export { StatCard, TrustScoreGauge, AttestationItem, CreateAttestationForm, PendingIntroductions };
