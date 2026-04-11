// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Insights Panel
 *
 * A comprehensive sidebar panel that displays all epistemic information
 * about an email and its sender:
 * - AI-powered insights (intent, sentiment, urgency)
 * - Trust graph visualization
 * - Verifiable claims and credentials
 * - Reply suggestions
 *
 * This integrates 0TML (claims) and Symthaea (AI) capabilities.
 */

import { useState } from 'react';
import type { Email } from '@/types';
import {
  useEmailEpistemicData,
  useSenderTrust,
  useReplySuggestions,
  useCreateAttestation,
  useAIStatus,
} from '@/hooks/useEpistemicServices';
import type {
  AIInsights,
  AssuranceLevelResponse,
  TrustPath,
  ReplySuggestion,
  EmailIntent,
  RelationType,
} from '@/services/api';

interface EpistemicInsightsPanelProps {
  email: Email;
  userDid: string;
  onClose: () => void;
  onUseReply?: (reply: string) => void;
}

// Intent configuration
const intentConfig: Record<EmailIntent, { icon: string; label: string; color: string }> = {
  meeting_request: { icon: '📅', label: 'Meeting Request', color: 'text-blue-600 bg-blue-50 dark:bg-blue-900/30' },
  action_item: { icon: '✅', label: 'Action Required', color: 'text-orange-600 bg-orange-50 dark:bg-orange-900/30' },
  fyi: { icon: '📋', label: 'FYI', color: 'text-gray-600 bg-gray-50 dark:bg-gray-800' },
  question: { icon: '❓', label: 'Question', color: 'text-purple-600 bg-purple-50 dark:bg-purple-900/30' },
  social_greeting: { icon: '👋', label: 'Social', color: 'text-green-600 bg-green-50 dark:bg-green-900/30' },
  commercial: { icon: '🏪', label: 'Commercial', color: 'text-amber-600 bg-amber-50 dark:bg-amber-900/30' },
  spam: { icon: '🚫', label: 'Likely Spam', color: 'text-red-600 bg-red-50 dark:bg-red-900/30' },
  unknown: { icon: '❔', label: 'Unknown', color: 'text-gray-500 bg-gray-50 dark:bg-gray-800' },
};

// Assurance level configuration
const assuranceConfig = {
  e0_anonymous: { label: 'Anonymous', color: 'text-gray-500', cost: '$0', tier: 0 },
  e1_verified_email: { label: 'Email Verified', color: 'text-blue-500', cost: '$100', tier: 1 },
  e2_gitcoin_passport: { label: 'Gitcoin Passport', color: 'text-indigo-500', cost: '$1K', tier: 2 },
  e3_multi_factor: { label: 'Multi-Factor', color: 'text-purple-500', cost: '$100K', tier: 3 },
  e4_constitutional: { label: 'Constitutional', color: 'text-emerald-500', cost: '$10M', tier: 4 },
};

// Section component for collapsible sections
function Section({
  title,
  icon,
  children,
  defaultOpen = true,
  badge,
}: {
  title: string;
  icon: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  badge?: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border-b border-gray-200 dark:border-gray-700">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-lg">{icon}</span>
          <span className="font-medium text-sm text-gray-900 dark:text-gray-100">{title}</span>
          {badge}
        </div>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {isOpen && <div className="px-4 pb-4">{children}</div>}
    </div>
  );
}

// AI Insights section
function AIInsightsSection({ insights, isLoading }: { insights?: AIInsights; isLoading: boolean }) {
  if (isLoading) {
    return (
      <div className="space-y-3 animate-pulse">
        <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded-lg w-24" />
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-full" />
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4" />
      </div>
    );
  }

  if (!insights) {
    return (
      <p className="text-sm text-gray-500 dark:text-gray-400">
        AI analysis unavailable
      </p>
    );
  }

  const intent = intentConfig[insights.intent];

  return (
    <div className="space-y-4">
      {/* Intent Badge */}
      <div className="flex items-center gap-2">
        <span className={`inline-flex items-center px-3 py-1.5 rounded-full text-sm font-medium ${intent.color}`}>
          <span className="mr-1.5">{intent.icon}</span>
          {intent.label}
        </span>
        {insights.reply_needed && (
          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300">
            Reply needed
          </span>
        )}
      </div>

      {/* Sentiment & Urgency */}
      <div className="grid grid-cols-2 gap-3">
        <div className="p-2 rounded-lg bg-gray-50 dark:bg-gray-800">
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Sentiment</p>
          <p className="text-sm font-medium capitalize text-gray-900 dark:text-gray-100">
            {insights.sentiment === 'positive' && '😊 '}
            {insights.sentiment === 'negative' && '😔 '}
            {insights.sentiment === 'mixed' && '😐 '}
            {insights.sentiment}
          </p>
        </div>
        <div className="p-2 rounded-lg bg-gray-50 dark:bg-gray-800">
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Urgency</p>
          <p className={`text-sm font-medium capitalize ${
            insights.urgency === 'high' ? 'text-red-600' :
            insights.urgency === 'medium' ? 'text-amber-600' :
            'text-gray-600 dark:text-gray-300'
          }`}>
            {insights.urgency === 'high' && '🔴 '}
            {insights.urgency === 'medium' && '🟡 '}
            {insights.urgency}
          </p>
        </div>
      </div>

      {/* Key Topics */}
      {insights.key_topics.length > 0 && (
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Key Topics</p>
          <div className="flex flex-wrap gap-1.5">
            {insights.key_topics.map((topic, i) => (
              <span
                key={i}
                className="px-2 py-0.5 text-xs rounded-full bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300"
              >
                {topic}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Suggested Labels */}
      {insights.suggested_labels.length > 0 && (
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Suggested Labels</p>
          <div className="flex flex-wrap gap-1.5">
            {insights.suggested_labels.map((label, i) => (
              <button
                key={i}
                className="px-2 py-0.5 text-xs rounded-full bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors"
              >
                + {label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Confidence */}
      <div className="flex items-center gap-2 pt-2 border-t border-gray-100 dark:border-gray-700">
        <span className="text-xs text-gray-400">Confidence</span>
        <div className="flex-1 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-emerald-500 rounded-full"
            style={{ width: `${insights.confidence * 100}%` }}
          />
        </div>
        <span className="text-xs text-gray-500">{Math.round(insights.confidence * 100)}%</span>
      </div>
    </div>
  );
}

// Trust Path Visualization
function TrustPathSection({
  path,
  assurance,
  isLoading,
  hasPath,
}: {
  path?: TrustPath;
  assurance?: AssuranceLevelResponse;
  isLoading: boolean;
  hasPath: boolean;
}) {
  if (isLoading) {
    return (
      <div className="space-y-3 animate-pulse">
        <div className="h-12 bg-gray-200 dark:bg-gray-700 rounded-lg" />
        <div className="h-20 bg-gray-200 dark:bg-gray-700 rounded-lg" />
      </div>
    );
  }

  const assuranceInfo = assurance ? assuranceConfig[assurance.level] : null;

  return (
    <div className="space-y-4">
      {/* Assurance Level */}
      {assuranceInfo && (
        <div className="p-3 rounded-lg bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-800/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-500 dark:text-gray-400">Identity Assurance</span>
            <span className={`text-xs font-medium ${assuranceInfo.color}`}>
              E{assuranceInfo.tier}
            </span>
          </div>
          <p className={`text-sm font-medium ${assuranceInfo.color}`}>
            {assuranceInfo.label}
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Attack cost: {assuranceInfo.cost}
          </p>
        </div>
      )}

      {/* Trust Path */}
      {hasPath && path ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500 dark:text-gray-400">Trust Path</span>
            <span className="text-xs font-medium text-emerald-600 dark:text-emerald-400">
              {path.path_length} hop{path.path_length !== 1 ? 's' : ''}
            </span>
          </div>

          {/* Visual path */}
          <div className="flex items-center gap-1 overflow-x-auto py-2">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
              <span className="text-xs">You</span>
            </div>
            {path.hops.map((hop, i) => (
              <div key={i} className="flex items-center">
                <div className="w-8 h-0.5 bg-gray-300 dark:bg-gray-600" />
                <div className="flex-shrink-0 px-2 py-1 rounded bg-gray-100 dark:bg-gray-700 text-xs">
                  {Math.round(hop.weight * 100)}%
                </div>
                <div className="w-8 h-0.5 bg-gray-300 dark:bg-gray-600" />
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center">
                  <span className="text-xs">{i + 1}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Total trust weight */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 dark:text-gray-400">Transitive trust:</span>
            <span className={`text-sm font-medium ${
              path.total_weight >= 0.7 ? 'text-emerald-600' :
              path.total_weight >= 0.4 ? 'text-amber-600' :
              'text-red-600'
            }`}>
              {Math.round(path.total_weight * 100)}%
            </span>
          </div>
        </div>
      ) : (
        <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
          <p className="text-sm text-amber-700 dark:text-amber-300">
            No trust path to sender
          </p>
          <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
            This sender is not in your trust network
          </p>
        </div>
      )}
    </div>
  );
}

// Reply Suggestions Section
function ReplySuggestionsSection({
  suggestions,
  isLoading,
  onUseReply,
}: {
  suggestions?: ReplySuggestion[];
  isLoading: boolean;
  onUseReply?: (reply: string) => void;
}) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  if (isLoading) {
    return (
      <div className="space-y-2 animate-pulse">
        <div className="h-16 bg-gray-200 dark:bg-gray-700 rounded-lg" />
        <div className="h-16 bg-gray-200 dark:bg-gray-700 rounded-lg" />
      </div>
    );
  }

  if (!suggestions || suggestions.length === 0) {
    return (
      <p className="text-sm text-gray-500 dark:text-gray-400">
        No reply suggestions available
      </p>
    );
  }

  return (
    <div className="space-y-2">
      {suggestions.map((suggestion, i) => (
        <div
          key={i}
          className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden"
        >
          <button
            onClick={() => setExpandedIndex(expandedIndex === i ? null : i)}
            className="w-full flex items-center justify-between px-3 py-2 hover:bg-gray-50 dark:hover:bg-gray-800/50"
          >
            <div className="flex items-center gap-2">
              <span className="text-xs px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700 capitalize">
                {suggestion.tone}
              </span>
              <span className="text-xs text-gray-500">
                {Math.round(suggestion.confidence * 100)}% match
              </span>
            </div>
            <svg
              className={`w-4 h-4 text-gray-400 transition-transform ${expandedIndex === i ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {expandedIndex === i && (
            <div className="px-3 pb-3 space-y-2">
              <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                {suggestion.body}
              </p>
              {onUseReply && (
                <button
                  onClick={() => onUseReply(suggestion.body)}
                  className="text-xs text-blue-600 hover:text-blue-700 dark:text-blue-400 font-medium"
                >
                  Use this reply
                </button>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// Attestation Creator
function CreateAttestationSection({
  senderDid,
  onSuccess,
}: {
  senderDid: string;
  onSuccess?: () => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [message, setMessage] = useState('');
  const [relationship, setRelationship] = useState<RelationType>('personal_contact');
  const [trustScore, setTrustScore] = useState(0.7);

  const createAttestation = useCreateAttestation();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await createAttestation.mutateAsync({
        subject_did: senderDid,
        message,
        relationship,
        trust_score: trustScore,
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
        className="w-full py-2 px-3 text-sm font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-800 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors"
      >
        + Create Trust Attestation
      </button>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-3 p-3 border border-gray-200 dark:border-gray-700 rounded-lg">
      <div>
        <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
          Relationship
        </label>
        <select
          value={relationship}
          onChange={(e) => setRelationship(e.target.value as RelationType)}
          className="w-full px-2 py-1.5 text-sm border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800"
        >
          <option value="personal_contact">Personal Contact</option>
          <option value="professional_colleague">Professional Colleague</option>
          <option value="organization_member">Organization Member</option>
          <option value="verified_service">Verified Service</option>
          <option value="community_member">Community Member</option>
        </select>
      </div>

      <div>
        <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
          Trust Level: {Math.round(trustScore * 100)}%
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={trustScore}
          onChange={(e) => setTrustScore(parseFloat(e.target.value))}
          className="w-full"
        />
      </div>

      <div>
        <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
          Message (optional)
        </label>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Why do you trust this person?"
          className="w-full px-2 py-1.5 text-sm border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800 resize-none"
          rows={2}
        />
      </div>

      <div className="flex gap-2">
        <button
          type="submit"
          disabled={createAttestation.isPending}
          className="flex-1 py-1.5 px-3 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded transition-colors disabled:opacity-50"
        >
          {createAttestation.isPending ? 'Creating...' : 'Create'}
        </button>
        <button
          type="button"
          onClick={() => setIsOpen(false)}
          className="py-1.5 px-3 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 rounded transition-colors"
        >
          Cancel
        </button>
      </div>
    </form>
  );
}

// AI Status indicator
function AIStatusIndicator() {
  const { data: status, isLoading } = useAIStatus();

  if (isLoading || !status) return null;

  return (
    <div className="flex items-center gap-1.5">
      <span className={`w-2 h-2 rounded-full ${status.active ? 'bg-emerald-500' : 'bg-gray-400'}`} />
      <span className="text-xs text-gray-500 dark:text-gray-400">
        {status.active ? 'AI Active' : 'AI Inactive'}
      </span>
    </div>
  );
}

// Main Panel Component
export default function EpistemicInsightsPanel({
  email,
  userDid,
  onClose,
  onUseReply,
}: EpistemicInsightsPanelProps) {
  const senderDid = `did:mycelix:${email.from.address.replace('@', '-at-')}`;

  const { analysis, intent, assurance, isLoading: epistemicLoading } = useEmailEpistemicData(email, senderDid);
  const { path, hasPath, isLoading: trustLoading } = useSenderTrust(senderDid, userDid);
  const { data: repliesData, isLoading: repliesLoading } = useReplySuggestions(email, true);

  return (
    <div className="h-full flex flex-col bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <div>
          <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Epistemic Insights
          </h2>
          <AIStatusIndicator />
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          aria-label="Close panel"
        >
          <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        <Section
          title="AI Analysis"
          icon="🧠"
          badge={
            analysis?.reply_needed ? (
              <span className="ml-2 w-2 h-2 rounded-full bg-red-500" />
            ) : undefined
          }
        >
          <AIInsightsSection insights={analysis} isLoading={epistemicLoading} />
        </Section>

        <Section title="Trust Path" icon="🔗" defaultOpen={true}>
          <TrustPathSection
            path={path}
            assurance={assurance}
            isLoading={trustLoading}
            hasPath={hasPath}
          />
        </Section>

        <Section title="Reply Suggestions" icon="💬" defaultOpen={false}>
          <ReplySuggestionsSection
            suggestions={repliesData?.suggestions}
            isLoading={repliesLoading}
            onUseReply={onUseReply}
          />
        </Section>

        <Section title="Trust Actions" icon="🤝" defaultOpen={false}>
          <CreateAttestationSection senderDid={senderDid} />
        </Section>
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 text-center">
        <p className="text-xs text-gray-400">
          Powered by Symthaea AI + 0TML
        </p>
      </div>
    </div>
  );
}
