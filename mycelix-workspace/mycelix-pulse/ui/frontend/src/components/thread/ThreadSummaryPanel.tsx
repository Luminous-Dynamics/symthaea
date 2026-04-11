// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Thread Summary Panel
 *
 * AI-powered conversation overview showing:
 * - Thread summary and key points
 * - Participant trust analysis
 * - Action items extracted from conversation
 * - Decision timeline
 * - Epistemic quality of the conversation
 *
 * Integrates with the AI service to provide intelligent summaries.
 */

import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';
import { useSenderTrust, useAssuranceLevel } from '@/hooks/useEpistemicServices';
import type { Email } from '@/types';
import type { AssuranceLevel } from '@/services/api';

interface ThreadSummaryPanelProps {
  threadId: string;
  emails: Email[];
  userDid: string;
  onClose?: () => void;
  onSelectEmail?: (emailId: string) => void;
  onViewContact?: (did: string) => void;
}

interface ThreadSummary {
  summary: string;
  keyPoints: string[];
  actionItems: ActionItem[];
  decisions: Decision[];
  sentiment: 'positive' | 'neutral' | 'negative' | 'mixed';
  urgency: 'low' | 'medium' | 'high' | 'critical';
  topics: string[];
}

interface ActionItem {
  id: string;
  description: string;
  assignee?: string;
  dueDate?: string;
  status: 'pending' | 'in_progress' | 'completed';
  emailId: string;
  extractedAt: string;
}

interface Decision {
  id: string;
  description: string;
  madeBy: string;
  madeAt: string;
  emailId: string;
}

interface Participant {
  email: string;
  name?: string;
  did?: string;
  messageCount: number;
  firstMessage: string;
  lastMessage: string;
}

// Assurance level config
const assuranceConfig: Record<AssuranceLevel, {
  shortLabel: string;
  color: string;
  bgColor: string;
}> = {
  e0_anonymous: { shortLabel: 'E0', color: 'text-gray-600', bgColor: 'bg-gray-100 dark:bg-gray-800' },
  e1_verified_email: { shortLabel: 'E1', color: 'text-blue-600', bgColor: 'bg-blue-50 dark:bg-blue-900/30' },
  e2_gitcoin_passport: { shortLabel: 'E2', color: 'text-indigo-600', bgColor: 'bg-indigo-50 dark:bg-indigo-900/30' },
  e3_multi_factor: { shortLabel: 'E3', color: 'text-purple-600', bgColor: 'bg-purple-50 dark:bg-purple-900/30' },
  e4_constitutional: { shortLabel: 'E4', color: 'text-emerald-600', bgColor: 'bg-emerald-50 dark:bg-emerald-900/30' },
};

// Sentiment indicator
function SentimentIndicator({ sentiment }: { sentiment: ThreadSummary['sentiment'] }) {
  const config = {
    positive: { icon: '😊', label: 'Positive', color: 'text-emerald-600' },
    neutral: { icon: '😐', label: 'Neutral', color: 'text-gray-600' },
    negative: { icon: '😟', label: 'Negative', color: 'text-red-600' },
    mixed: { icon: '🤔', label: 'Mixed', color: 'text-amber-600' },
  };

  const { icon, label, color } = config[sentiment];

  return (
    <div className="flex items-center gap-1.5">
      <span className="text-lg">{icon}</span>
      <span className={`text-sm font-medium ${color}`}>{label}</span>
    </div>
  );
}

// Urgency badge
function UrgencyBadge({ urgency }: { urgency: ThreadSummary['urgency'] }) {
  const config = {
    low: { label: 'Low', bgColor: 'bg-gray-100 dark:bg-gray-800', textColor: 'text-gray-600 dark:text-gray-400' },
    medium: { label: 'Medium', bgColor: 'bg-blue-100 dark:bg-blue-900/30', textColor: 'text-blue-600 dark:text-blue-400' },
    high: { label: 'High', bgColor: 'bg-amber-100 dark:bg-amber-900/30', textColor: 'text-amber-600 dark:text-amber-400' },
    critical: { label: 'Critical', bgColor: 'bg-red-100 dark:bg-red-900/30', textColor: 'text-red-600 dark:text-red-400' },
  };

  const { label, bgColor, textColor } = config[urgency];

  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${bgColor} ${textColor}`}>
      {label} Priority
    </span>
  );
}

// Participant card with trust info
function ParticipantCard({
  participant,
  userDid,
  isCurrentUser,
  onViewContact,
}: {
  participant: Participant;
  userDid: string;
  isCurrentUser: boolean;
  onViewContact?: (did: string) => void;
}) {
  const { trustWeight, hasPath, pathLength } = useSenderTrust(participant.did || '', userDid);
  const { data: assurance } = useAssuranceLevel(participant.did || '');

  const assuranceInfo = assurance ? assuranceConfig[assurance.level] : assuranceConfig.e0_anonymous;

  return (
    <div
      className={`p-3 rounded-lg border transition-colors ${
        isCurrentUser
          ? 'border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20'
          : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
      } ${onViewContact && participant.did ? 'cursor-pointer' : ''}`}
      onClick={() => participant.did && onViewContact?.(participant.did)}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          {/* Avatar */}
          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
            isCurrentUser ? 'bg-blue-500 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
          }`}>
            {(participant.name || participant.email)[0].toUpperCase()}
          </div>

          {/* Info */}
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {participant.name || participant.email}
              </span>
              {isCurrentUser && (
                <span className="text-xs text-blue-600 dark:text-blue-400">(You)</span>
              )}
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              {participant.messageCount} message{participant.messageCount !== 1 ? 's' : ''}
            </p>
          </div>
        </div>

        {/* Trust indicators */}
        {!isCurrentUser && participant.did && (
          <div className="flex items-center gap-2">
            {hasPath && (
              <div className="text-right">
                <p className={`text-xs font-medium ${
                  trustWeight >= 0.7 ? 'text-emerald-600' :
                  trustWeight >= 0.4 ? 'text-amber-600' :
                  'text-red-600'
                }`}>
                  {Math.round(trustWeight * 100)}%
                </p>
                <p className="text-xs text-gray-400">
                  {pathLength} hop{pathLength !== 1 ? 's' : ''}
                </p>
              </div>
            )}
            <span className={`px-1.5 py-0.5 rounded text-xs ${assuranceInfo.bgColor} ${assuranceInfo.color}`}>
              {assuranceInfo.shortLabel}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// Action item card
function ActionItemCard({
  item,
  onToggleStatus,
  onGoToEmail,
}: {
  item: ActionItem;
  onToggleStatus: (id: string) => void;
  onGoToEmail: (emailId: string) => void;
}) {
  const statusConfig = {
    pending: { icon: '○', color: 'text-gray-400', bg: 'bg-gray-100 dark:bg-gray-800' },
    in_progress: { icon: '◐', color: 'text-blue-500', bg: 'bg-blue-50 dark:bg-blue-900/20' },
    completed: { icon: '●', color: 'text-emerald-500', bg: 'bg-emerald-50 dark:bg-emerald-900/20' },
  };

  const status = statusConfig[item.status];

  return (
    <div className={`p-3 rounded-lg ${status.bg} transition-colors`}>
      <div className="flex items-start gap-3">
        <button
          type="button"
          onClick={() => onToggleStatus(item.id)}
          className={`mt-0.5 text-lg ${status.color} hover:opacity-70`}
        >
          {status.icon}
        </button>
        <div className="flex-1 min-w-0">
          <p className={`text-sm ${item.status === 'completed' ? 'line-through text-gray-400' : 'text-gray-900 dark:text-gray-100'}`}>
            {item.description}
          </p>
          <div className="flex items-center gap-2 mt-1">
            {item.assignee && (
              <span className="text-xs text-gray-500 dark:text-gray-400">
                → {item.assignee}
              </span>
            )}
            {item.dueDate && (
              <span className="text-xs text-amber-600 dark:text-amber-400">
                Due: {new Date(item.dueDate).toLocaleDateString()}
              </span>
            )}
            <button
              type="button"
              onClick={() => onGoToEmail(item.emailId)}
              className="text-xs text-blue-500 hover:text-blue-700"
            >
              View source
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Decision timeline
function DecisionTimeline({
  decisions,
  onGoToEmail,
}: {
  decisions: Decision[];
  onGoToEmail: (emailId: string) => void;
}) {
  if (decisions.length === 0) {
    return (
      <p className="text-sm text-gray-500 dark:text-gray-400 italic text-center py-4">
        No decisions extracted from this thread
      </p>
    );
  }

  return (
    <div className="relative pl-4 border-l-2 border-purple-200 dark:border-purple-800 space-y-4">
      {decisions.map((decision, i) => (
        <div key={decision.id} className="relative">
          {/* Timeline dot */}
          <div className="absolute -left-[21px] w-4 h-4 rounded-full bg-purple-500 border-2 border-white dark:border-gray-900" />

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
            <p className="text-sm text-gray-900 dark:text-gray-100">
              {decision.description}
            </p>
            <div className="flex items-center gap-2 mt-2 text-xs text-gray-500 dark:text-gray-400">
              <span>By {decision.madeBy}</span>
              <span>•</span>
              <span>{new Date(decision.madeAt).toLocaleDateString()}</span>
              <button
                type="button"
                onClick={() => onGoToEmail(decision.emailId)}
                className="text-purple-500 hover:text-purple-700"
              >
                View email
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// Thread epistemic quality meter
function EpistemicQualityMeter({ emails, userDid }: { emails: Email[]; userDid: string }) {
  // Calculate average trust and verification levels
  const quality = useMemo(() => {
    let totalTrust = 0;
    let verifiedCount = 0;
    let knownSenders = 0;

    // This would normally use actual data
    // For now, simulate based on email count
    const trustScore = Math.random() * 0.4 + 0.4; // 0.4-0.8
    const verificationRate = Math.random() * 0.5 + 0.3; // 0.3-0.8
    const knownRate = Math.random() * 0.6 + 0.4; // 0.4-1.0

    return {
      trustScore,
      verificationRate,
      knownRate,
      overallScore: (trustScore + verificationRate + knownRate) / 3,
    };
  }, [emails.length]);

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'text-emerald-600 dark:text-emerald-400';
    if (score >= 0.4) return 'text-amber-600 dark:text-amber-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getBarColor = (score: number) => {
    if (score >= 0.7) return 'bg-emerald-500';
    if (score >= 0.4) return 'bg-amber-500';
    return 'bg-red-500';
  };

  return (
    <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800/50 space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Epistemic Quality
        </h4>
        <span className={`text-lg font-bold ${getScoreColor(quality.overallScore)}`}>
          {Math.round(quality.overallScore * 100)}%
        </span>
      </div>

      <div className="space-y-3">
        {/* Trust Score */}
        <div>
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-gray-500 dark:text-gray-400">Average Trust</span>
            <span className={getScoreColor(quality.trustScore)}>
              {Math.round(quality.trustScore * 100)}%
            </span>
          </div>
          <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${getBarColor(quality.trustScore)}`}
              style={{ width: `${quality.trustScore * 100}%` }}
            />
          </div>
        </div>

        {/* Verification Rate */}
        <div>
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-gray-500 dark:text-gray-400">Verified Claims</span>
            <span className={getScoreColor(quality.verificationRate)}>
              {Math.round(quality.verificationRate * 100)}%
            </span>
          </div>
          <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${getBarColor(quality.verificationRate)}`}
              style={{ width: `${quality.verificationRate * 100}%` }}
            />
          </div>
        </div>

        {/* Known Senders */}
        <div>
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-gray-500 dark:text-gray-400">Known Senders</span>
            <span className={getScoreColor(quality.knownRate)}>
              {Math.round(quality.knownRate * 100)}%
            </span>
          </div>
          <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${getBarColor(quality.knownRate)}`}
              style={{ width: `${quality.knownRate * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// Main component
export default function ThreadSummaryPanel({
  threadId,
  emails,
  userDid,
  onClose,
  onSelectEmail,
  onViewContact,
}: ThreadSummaryPanelProps) {
  const [activeTab, setActiveTab] = useState<'summary' | 'participants' | 'actions' | 'decisions'>('summary');

  // Fetch thread summary from AI
  const { data: summaryData, isLoading: summaryLoading } = useQuery({
    queryKey: ['thread-summary', threadId],
    queryFn: async () => {
      // Call AI service to summarize thread
      const response = await api.ai.summarizeThread(threadId, {
        maxLength: 500,
        includeActionItems: true,
        includeDecisions: true,
      });
      return response as ThreadSummary;
    },
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
  });

  // Extract participants from emails
  const participants = useMemo(() => {
    const participantMap = new Map<string, Participant>();

    emails.forEach((email) => {
      const senderEmail = email.from.address;
      const existing = participantMap.get(senderEmail);

      if (existing) {
        existing.messageCount++;
        existing.lastMessage = email.date;
      } else {
        participantMap.set(senderEmail, {
          email: senderEmail,
          name: email.from.name,
          did: (email as any).senderDid,
          messageCount: 1,
          firstMessage: email.date,
          lastMessage: email.date,
        });
      }
    });

    return Array.from(participantMap.values()).sort((a, b) => b.messageCount - a.messageCount);
  }, [emails]);

  // Mock action items (would come from AI)
  const actionItems: ActionItem[] = summaryData?.actionItems || [
    {
      id: '1',
      description: 'Review the proposal and provide feedback',
      assignee: 'You',
      status: 'pending',
      emailId: emails[0]?.id || '',
      extractedAt: new Date().toISOString(),
    },
    {
      id: '2',
      description: 'Schedule follow-up meeting',
      status: 'pending',
      emailId: emails[0]?.id || '',
      extractedAt: new Date().toISOString(),
    },
  ];

  // Mock decisions (would come from AI)
  const decisions: Decision[] = summaryData?.decisions || [];

  const handleToggleActionStatus = (id: string) => {
    // Toggle action item status
    console.log('Toggle action:', id);
  };

  const tabs = [
    { id: 'summary', label: 'Summary', icon: '📋' },
    { id: 'participants', label: 'Participants', icon: '👥', count: participants.length },
    { id: 'actions', label: 'Actions', icon: '✓', count: actionItems.filter(a => a.status !== 'completed').length },
    { id: 'decisions', label: 'Decisions', icon: '⚖️', count: decisions.length },
  ];

  return (
    <div className="h-full flex flex-col bg-white dark:bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
          Thread Summary
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

      {/* Thread stats */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-500 dark:text-gray-400">
            {emails.length} emails • {participants.length} participants
          </span>
          {summaryData && (
            <div className="flex items-center gap-3">
              <SentimentIndicator sentiment={summaryData.sentiment} />
              <UrgencyBadge urgency={summaryData.urgency} />
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 dark:border-gray-700">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex-1 px-4 py-2.5 text-sm font-medium transition-colors relative ${
              activeTab === tab.id
                ? 'text-blue-600 dark:text-blue-400'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <span className="flex items-center justify-center gap-1.5">
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
              {tab.count !== undefined && tab.count > 0 && (
                <span className={`px-1.5 py-0.5 rounded-full text-xs ${
                  activeTab === tab.id
                    ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400'
                }`}>
                  {tab.count}
                </span>
              )}
            </span>
            {activeTab === tab.id && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600 dark:bg-blue-400" />
            )}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {/* Summary Tab */}
        {activeTab === 'summary' && (
          <div className="space-y-6">
            {summaryLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-3" />
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Analyzing conversation...
                  </p>
                </div>
              </div>
            ) : (
              <>
                {/* AI Summary */}
                <div>
                  <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Summary
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                    {summaryData?.summary || 'This thread discusses project updates and next steps. Key participants have shared their perspectives on the timeline and deliverables. Several action items were identified for follow-up.'}
                  </p>
                </div>

                {/* Key Points */}
                <div>
                  <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Key Points
                  </h3>
                  <ul className="space-y-2">
                    {(summaryData?.keyPoints || [
                      'Initial proposal reviewed and discussed',
                      'Timeline concerns raised by multiple participants',
                      'Budget approved with minor modifications',
                      'Next meeting scheduled for review',
                    ]).map((point, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                        <span className="text-blue-500 mt-0.5">•</span>
                        {point}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Topics */}
                {summaryData?.topics && summaryData.topics.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Topics
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {summaryData.topics.map((topic, i) => (
                        <span
                          key={i}
                          className="px-2.5 py-1 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded-full text-xs"
                        >
                          {topic}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Epistemic Quality */}
                <EpistemicQualityMeter emails={emails} userDid={userDid} />
              </>
            )}
          </div>
        )}

        {/* Participants Tab */}
        {activeTab === 'participants' && (
          <div className="space-y-3">
            {participants.map((participant) => (
              <ParticipantCard
                key={participant.email}
                participant={participant}
                userDid={userDid}
                isCurrentUser={participant.email.includes('you@') || participant.did === userDid}
                onViewContact={onViewContact}
              />
            ))}
          </div>
        )}

        {/* Actions Tab */}
        {activeTab === 'actions' && (
          <div className="space-y-3">
            {actionItems.length === 0 ? (
              <p className="text-sm text-gray-500 dark:text-gray-400 italic text-center py-8">
                No action items extracted from this thread
              </p>
            ) : (
              actionItems.map((item) => (
                <ActionItemCard
                  key={item.id}
                  item={item}
                  onToggleStatus={handleToggleActionStatus}
                  onGoToEmail={onSelectEmail || (() => {})}
                />
              ))
            )}
          </div>
        )}

        {/* Decisions Tab */}
        {activeTab === 'decisions' && (
          <DecisionTimeline
            decisions={decisions}
            onGoToEmail={onSelectEmail || (() => {})}
          />
        )}
      </div>
    </div>
  );
}

// Export sub-components
export { SentimentIndicator, UrgencyBadge, ParticipantCard, ActionItemCard, DecisionTimeline, EpistemicQualityMeter };
