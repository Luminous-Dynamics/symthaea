// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Email Row Component
 *
 * An enhanced email list row that displays epistemic indicators:
 * - Intent badge (meeting, action item, question, etc.)
 * - Trust level indicator
 * - Claim verification status
 * - Urgency indicator
 *
 * This component is designed to be used alongside or in place of
 * standard email list rows to provide at-a-glance epistemic context.
 */

import { memo } from 'react';
import type { Email } from '@/types';
import type { EmailIntent, AssuranceLevel } from '@/services/api';

interface EpistemicEmailRowProps {
  email: Email;
  isSelected?: boolean;
  onClick?: () => void;
  // Optional pre-fetched epistemic data (to avoid N+1 queries)
  epistemicData?: {
    intent?: EmailIntent;
    urgency?: 'high' | 'medium' | 'low' | 'none';
    trustScore?: number;
    assuranceLevel?: AssuranceLevel;
    hasVerifiedClaims?: boolean;
    replyNeeded?: boolean;
  };
}

// Compact intent indicators
const intentIndicators: Record<EmailIntent, { icon: string; color: string; label: string }> = {
  meeting_request: { icon: '📅', color: 'bg-blue-500', label: 'Meeting' },
  action_item: { icon: '✅', color: 'bg-orange-500', label: 'Action' },
  fyi: { icon: '📋', color: 'bg-gray-400', label: 'FYI' },
  question: { icon: '❓', color: 'bg-purple-500', label: 'Question' },
  social_greeting: { icon: '👋', color: 'bg-green-500', label: 'Social' },
  commercial: { icon: '🏪', color: 'bg-amber-500', label: 'Commercial' },
  spam: { icon: '🚫', color: 'bg-red-500', label: 'Spam' },
  unknown: { icon: '❔', color: 'bg-gray-300', label: 'Unknown' },
};

// Trust level colors
function getTrustColor(score?: number): string {
  if (score === undefined) return 'bg-gray-300';
  if (score >= 70) return 'bg-emerald-500';
  if (score >= 40) return 'bg-amber-500';
  return 'bg-red-500';
}

// Assurance level badge
function getAssuranceBadge(level?: AssuranceLevel): { color: string; label: string } | null {
  if (!level) return null;

  const badges: Record<AssuranceLevel, { color: string; label: string }> = {
    e0_anonymous: { color: 'bg-gray-200 text-gray-600', label: 'E0' },
    e1_verified_email: { color: 'bg-blue-100 text-blue-700', label: 'E1' },
    e2_gitcoin_passport: { color: 'bg-indigo-100 text-indigo-700', label: 'E2' },
    e3_multi_factor: { color: 'bg-purple-100 text-purple-700', label: 'E3' },
    e4_constitutional: { color: 'bg-emerald-100 text-emerald-700', label: 'E4' },
  };

  return badges[level];
}

// Intent Pill Component
function IntentPill({ intent }: { intent: EmailIntent }) {
  const config = intentIndicators[intent];
  return (
    <span
      className="inline-flex items-center justify-center w-5 h-5 rounded-full text-xs"
      title={config.label}
    >
      {config.icon}
    </span>
  );
}

// Trust Dot Component
function TrustDot({ score }: { score?: number }) {
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full ${getTrustColor(score)}`}
      title={score !== undefined ? `Trust: ${score}%` : 'Trust unknown'}
    />
  );
}

// Urgency Indicator
function UrgencyIndicator({ urgency }: { urgency: 'high' | 'medium' | 'low' | 'none' }) {
  if (urgency === 'none' || urgency === 'low') return null;

  return (
    <span
      className={`inline-flex items-center justify-center w-4 h-4 rounded-full text-xs font-bold text-white ${
        urgency === 'high' ? 'bg-red-500' : 'bg-amber-500'
      }`}
      title={`${urgency} urgency`}
    >
      !
    </span>
  );
}

// Verified Claims Badge
function VerifiedClaimsBadge({ hasVerifiedClaims }: { hasVerifiedClaims: boolean }) {
  if (!hasVerifiedClaims) return null;

  return (
    <span
      className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-emerald-100 text-emerald-600 text-xs"
      title="Has verified claims"
    >
      ✓
    </span>
  );
}

// Reply Needed Indicator
function ReplyNeededIndicator({ needed }: { needed: boolean }) {
  if (!needed) return null;

  return (
    <span
      className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
      title="Reply needed"
    >
      Reply
    </span>
  );
}

/**
 * Epistemic Indicators Bar
 *
 * A compact bar showing all epistemic indicators for an email.
 * Can be used as a supplement to existing email row components.
 */
export function EpistemicIndicatorsBar({
  epistemicData,
  compact = false,
}: {
  epistemicData?: EpistemicEmailRowProps['epistemicData'];
  compact?: boolean;
}) {
  if (!epistemicData) return null;

  const { intent, urgency, trustScore, assuranceLevel, hasVerifiedClaims, replyNeeded } = epistemicData;
  const assuranceBadge = getAssuranceBadge(assuranceLevel);

  if (compact) {
    // Ultra-compact version: just dots and icons
    return (
      <div className="flex items-center gap-1">
        <TrustDot score={trustScore} />
        {intent && intent !== 'unknown' && <IntentPill intent={intent} />}
        {urgency && <UrgencyIndicator urgency={urgency} />}
        {hasVerifiedClaims && <VerifiedClaimsBadge hasVerifiedClaims={true} />}
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      {/* Trust Score */}
      <div className="flex items-center gap-1">
        <TrustDot score={trustScore} />
        {trustScore !== undefined && (
          <span className="text-xs text-gray-500 dark:text-gray-400 tabular-nums">
            {trustScore}
          </span>
        )}
      </div>

      {/* Assurance Level */}
      {assuranceBadge && (
        <span className={`text-xs px-1 rounded ${assuranceBadge.color}`}>
          {assuranceBadge.label}
        </span>
      )}

      {/* Intent */}
      {intent && intent !== 'unknown' && <IntentPill intent={intent} />}

      {/* Urgency */}
      {urgency && <UrgencyIndicator urgency={urgency} />}

      {/* Verified Claims */}
      {hasVerifiedClaims && <VerifiedClaimsBadge hasVerifiedClaims={true} />}

      {/* Reply Needed */}
      {replyNeeded && <ReplyNeededIndicator needed={true} />}
    </div>
  );
}

/**
 * Main Epistemic Email Row Component
 *
 * A complete email row with epistemic enhancements.
 */
function EpistemicEmailRow({
  email,
  isSelected,
  onClick,
  epistemicData,
}: EpistemicEmailRowProps) {
  const fromName = email.from.name || email.from.address;
  const truncatedSubject = email.subject.length > 60
    ? email.subject.slice(0, 57) + '...'
    : email.subject;

  return (
    <div
      onClick={onClick}
      className={`
        flex items-center gap-3 px-4 py-3 cursor-pointer transition-colors
        ${isSelected
          ? 'bg-blue-50 dark:bg-blue-900/20 border-l-2 border-blue-500'
          : 'hover:bg-gray-50 dark:hover:bg-gray-800/50 border-l-2 border-transparent'
        }
        ${!email.isRead ? 'font-medium' : 'font-normal'}
      `}
    >
      {/* Trust indicator */}
      <div className="flex-shrink-0">
        <TrustDot score={epistemicData?.trustScore} />
      </div>

      {/* Sender & Subject */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`text-sm truncate ${!email.isRead ? 'text-gray-900 dark:text-gray-100' : 'text-gray-600 dark:text-gray-400'}`}>
            {fromName}
          </span>
          {epistemicData?.assuranceLevel && epistemicData.assuranceLevel !== 'e0_anonymous' && (
            <span className={`text-xs px-1 rounded ${getAssuranceBadge(epistemicData.assuranceLevel)?.color}`}>
              {getAssuranceBadge(epistemicData.assuranceLevel)?.label}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-sm truncate ${!email.isRead ? 'text-gray-800 dark:text-gray-200' : 'text-gray-500 dark:text-gray-500'}`}>
            {truncatedSubject}
          </span>
        </div>
      </div>

      {/* Epistemic Indicators */}
      <div className="flex-shrink-0 flex items-center gap-2">
        {epistemicData?.intent && epistemicData.intent !== 'unknown' && (
          <IntentPill intent={epistemicData.intent} />
        )}
        {epistemicData?.urgency && epistemicData.urgency !== 'none' && (
          <UrgencyIndicator urgency={epistemicData.urgency} />
        )}
        {epistemicData?.hasVerifiedClaims && (
          <VerifiedClaimsBadge hasVerifiedClaims={true} />
        )}
        {epistemicData?.replyNeeded && (
          <ReplyNeededIndicator needed={true} />
        )}
      </div>

      {/* Date */}
      <div className="flex-shrink-0 text-xs text-gray-500 dark:text-gray-400 tabular-nums">
        {new Date(email.date).toLocaleDateString(undefined, {
          month: 'short',
          day: 'numeric',
        })}
      </div>
    </div>
  );
}

export default memo(EpistemicEmailRow);
