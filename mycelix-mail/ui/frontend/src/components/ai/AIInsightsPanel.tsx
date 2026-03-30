// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useState } from 'react';

// Types matching backend
type EmailIntent =
  | 'meeting_request'
  | 'action_item'
  | 'fyi'
  | 'question'
  | 'social_greeting'
  | 'commercial'
  | 'spam'
  | 'unknown';

interface ReplySuggestion {
  content: string;
  tone: string;
  length: string;
  confidence: number;
}

interface AIInsights {
  intent: EmailIntent;
  priority: number;
  summary: string;
  suggested_actions: string[];
  safe: boolean;
  metadata: {
    processing_time_ms: number;
    consciousness_level: number;
    local_only: boolean;
  };
}

interface ThreadSummary {
  bullets: string[];
  topic: string;
  participants: string[];
  action_items: string[];
  confidence: number;
  consciousness_level: number;
}

interface AIInsightsPanelProps {
  insights?: AIInsights;
  threadSummary?: ThreadSummary;
  replySuggestions?: ReplySuggestion[];
  onSelectReply?: (content: string) => void;
  loading?: boolean;
  className?: string;
}

// Intent configuration
const intentConfig: Record<EmailIntent, { label: string; icon: string; color: string }> = {
  meeting_request: {
    label: 'Meeting Request',
    icon: '📅',
    color: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-200',
  },
  action_item: {
    label: 'Action Required',
    icon: '✅',
    color: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-200',
  },
  fyi: {
    label: 'FYI',
    icon: 'ℹ️',
    color: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200',
  },
  question: {
    label: 'Question',
    icon: '❓',
    color: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-200',
  },
  social_greeting: {
    label: 'Social',
    icon: '👋',
    color: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-200',
  },
  commercial: {
    label: 'Commercial',
    icon: '🏪',
    color: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-200',
  },
  spam: {
    label: 'Likely Spam',
    icon: '⚠️',
    color: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-200',
  },
  unknown: {
    label: 'Unknown',
    icon: '❔',
    color: 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400',
  },
};

// Intent badge component
function IntentBadge({ intent }: { intent: EmailIntent }) {
  const config = intentConfig[intent];
  return (
    <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${config.color}`}>
      <span className="mr-1.5">{config.icon}</span>
      {config.label}
    </span>
  );
}

// Priority indicator
function PriorityIndicator({ priority }: { priority: number }) {
  const level = priority >= 0.7 ? 'high' : priority >= 0.4 ? 'medium' : 'low';
  const bars = level === 'high' ? 3 : level === 'medium' ? 2 : 1;

  return (
    <div className="flex items-center space-x-0.5">
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          className={`w-1 rounded-full ${
            i <= bars
              ? level === 'high'
                ? 'bg-red-500'
                : level === 'medium'
                ? 'bg-yellow-500'
                : 'bg-green-500'
              : 'bg-gray-200 dark:bg-gray-700'
          }`}
          style={{ height: `${8 + i * 4}px` }}
        />
      ))}
      <span className="ml-1.5 text-xs text-gray-500 dark:text-gray-400 capitalize">
        {level}
      </span>
    </div>
  );
}

// Consciousness indicator (shows AI is local)
function ConsciousnessIndicator({ level, localOnly }: { level: number; localOnly: boolean }) {
  return (
    <div className="flex items-center text-xs text-gray-500 dark:text-gray-400">
      <span className="w-2 h-2 rounded-full bg-emerald-500 mr-1.5 animate-pulse" />
      <span>
        Local AI {localOnly ? '(private)' : ''} · Φ={level.toFixed(2)}
      </span>
    </div>
  );
}

// Reply suggestion card
function ReplySuggestionCard({
  suggestion,
  onSelect,
}: {
  suggestion: ReplySuggestion;
  onSelect: () => void;
}) {
  const toneEmoji: Record<string, string> = {
    formal: '👔',
    professional: '💼',
    friendly: '😊',
    brief: '⚡',
    detailed: '📝',
  };

  return (
    <button
      onClick={onSelect}
      className="w-full p-3 text-left rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
          {toneEmoji[suggestion.tone] || '💬'} {suggestion.tone} · {suggestion.length}
        </span>
        <span className="text-xs text-gray-400">
          {(suggestion.confidence * 100).toFixed(0)}% match
        </span>
      </div>
      <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-2">
        {suggestion.content}
      </p>
    </button>
  );
}

export default function AIInsightsPanel({
  insights,
  threadSummary,
  replySuggestions,
  onSelectReply,
  loading = false,
  className = '',
}: AIInsightsPanelProps) {
  const [expanded, setExpanded] = useState(true);

  if (loading) {
    return (
      <div className={`bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 p-4 ${className}`}>
        <div className="flex items-center space-x-3">
          <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-gray-600 dark:text-gray-400">
            Analyzing with local AI...
          </span>
        </div>
      </div>
    );
  }

  if (!insights) {
    return null;
  }

  return (
    <div className={`bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 ${className}`}>
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between border-b border-gray-200 dark:border-gray-700"
      >
        <div className="flex items-center space-x-3">
          <span className="text-lg">🧠</span>
          <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            AI Insights
          </span>
          <IntentBadge intent={insights.intent} />
        </div>
        <div className="flex items-center space-x-3">
          <PriorityIndicator priority={insights.priority} />
          <svg
            className={`w-5 h-5 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {expanded && (
        <div className="p-4 space-y-4">
          {/* Safety warning */}
          {!insights.safe && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
              <div className="flex items-center text-red-800 dark:text-red-200">
                <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
                <span className="text-sm font-medium">Safety concern detected</span>
              </div>
            </div>
          )}

          {/* Summary */}
          <div>
            <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
              Summary
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              {insights.summary}
            </p>
          </div>

          {/* Thread summary (if available) */}
          {threadSummary && threadSummary.bullets.length > 0 && (
            <div>
              <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                Thread Summary
              </h4>
              <ul className="space-y-1">
                {threadSummary.bullets.map((bullet, i) => (
                  <li key={i} className="flex items-start text-sm text-gray-700 dark:text-gray-300">
                    <span className="text-blue-500 mr-2">•</span>
                    {bullet}
                  </li>
                ))}
              </ul>
              {threadSummary.action_items.length > 0 && (
                <div className="mt-2 pt-2 border-t border-gray-100 dark:border-gray-800">
                  <span className="text-xs font-medium text-orange-600 dark:text-orange-400">
                    Action items:
                  </span>
                  <ul className="mt-1 space-y-1">
                    {threadSummary.action_items.map((item, i) => (
                      <li key={i} className="flex items-center text-sm text-gray-700 dark:text-gray-300">
                        <span className="text-orange-500 mr-2">☐</span>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Suggested actions */}
          {insights.suggested_actions.length > 0 && (
            <div>
              <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                Suggested Actions
              </h4>
              <div className="flex flex-wrap gap-2">
                {insights.suggested_actions.map((action, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center px-2.5 py-1 rounded-full text-xs bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300"
                  >
                    {action}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Reply suggestions */}
          {replySuggestions && replySuggestions.length > 0 && onSelectReply && (
            <div>
              <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                Quick Replies
              </h4>
              <div className="space-y-2">
                {replySuggestions.map((suggestion, i) => (
                  <ReplySuggestionCard
                    key={i}
                    suggestion={suggestion}
                    onSelect={() => onSelectReply(suggestion.content)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Footer with consciousness indicator */}
          <div className="pt-3 border-t border-gray-100 dark:border-gray-800 flex items-center justify-between">
            <ConsciousnessIndicator
              level={insights.metadata.consciousness_level}
              localOnly={insights.metadata.local_only}
            />
            <span className="text-xs text-gray-400">
              {insights.metadata.processing_time_ms}ms
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// Compact intent badge for email list
export function EmailIntentBadge({ intent }: { intent: EmailIntent }) {
  return <IntentBadge intent={intent} />;
}

// Priority dot for email list
export function EmailPriorityDot({ priority }: { priority: number }) {
  const color =
    priority >= 0.7
      ? 'bg-red-500'
      : priority >= 0.4
      ? 'bg-yellow-500'
      : 'bg-green-500';

  return (
    <span
      className={`w-2 h-2 rounded-full ${color}`}
      title={`Priority: ${(priority * 100).toFixed(0)}%`}
    />
  );
}
