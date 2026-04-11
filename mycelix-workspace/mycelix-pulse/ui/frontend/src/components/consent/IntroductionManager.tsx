// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useState } from 'react';

// Types
interface Introduction {
  id: string;
  introducer_did: string;
  introducer_name?: string;
  introducee_did: string;
  introducee_name?: string;
  target_did: string;
  message: string;
  trust_stake: number;
  status: 'pending' | 'accepted' | 'declined' | 'expired';
  expires_at: string;
  created_at: string;
}

interface AttentionBid {
  id: string;
  sender_did: string;
  sender_name?: string;
  recipient_did: string;
  stake_amount: number;
  message_preview: string;
  email_id: string;
  expires_at: string;
  created_at: string;
}

interface AttentionPolicy {
  min_stake: number;
  trusted_free: boolean;
  introduction_discount: number;
  auto_refund_trusted: boolean;
}

// Introduction request card
interface IntroductionCardProps {
  introduction: Introduction;
  onAccept: (id: string) => void;
  onDecline: (id: string) => void;
  loading?: boolean;
}

export function IntroductionCard({
  introduction,
  onAccept,
  onDecline,
  loading = false,
}: IntroductionCardProps) {
  const stakePercentage = (introduction.trust_stake * 100).toFixed(0);
  const expiresAt = new Date(introduction.expires_at);
  const isExpired = expiresAt < new Date();

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <div className="flex items-start space-x-3">
        {/* Introducer avatar */}
        <div className="w-10 h-10 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
          <span className="text-blue-600 dark:text-blue-300 font-semibold">
            {(introduction.introducer_name || introduction.introducer_did.slice(-2)).slice(0, 2).toUpperCase()}
          </span>
        </div>

        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-900 dark:text-gray-100">
              <span className="font-medium">
                {introduction.introducer_name || introduction.introducer_did.slice(-8)}
              </span>
              <span className="text-gray-500 dark:text-gray-400"> wants to introduce </span>
              <span className="font-medium">
                {introduction.introducee_name || introduction.introducee_did.slice(-8)}
              </span>
            </p>
            <span
              className={`text-xs px-2 py-0.5 rounded-full ${
                introduction.status === 'pending'
                  ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-200'
                  : introduction.status === 'accepted'
                  ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-200'
                  : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
              }`}
            >
              {introduction.status}
            </span>
          </div>

          {/* Message */}
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            "{introduction.message}"
          </p>

          {/* Trust stake info */}
          <div className="mt-3 flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center">
              <svg className="w-4 h-4 mr-1 text-amber-500" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
              <span>Staking {stakePercentage}% of their trust</span>
            </div>
            <div>
              {isExpired ? (
                <span className="text-red-500">Expired</span>
              ) : (
                <span>Expires {expiresAt.toLocaleDateString()}</span>
              )}
            </div>
          </div>

          {/* Actions */}
          {introduction.status === 'pending' && !isExpired && (
            <div className="mt-4 flex space-x-2">
              <button
                onClick={() => onAccept(introduction.id)}
                disabled={loading}
                className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
              >
                Accept Introduction
              </button>
              <button
                onClick={() => onDecline(introduction.id)}
                disabled={loading}
                className="flex-1 px-3 py-2 bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
              >
                Decline
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Attention bid card
interface AttentionBidCardProps {
  bid: AttentionBid;
  onView: (id: string) => void;
  onReject: (id: string) => void;
  onReportSpam: (id: string) => void;
  loading?: boolean;
}

export function AttentionBidCard({
  bid,
  onView,
  onReject,
  onReportSpam,
  loading = false,
}: AttentionBidCardProps) {
  const expiresAt = new Date(bid.expires_at);
  const isExpired = expiresAt < new Date();

  return (
    <div className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-lg border border-amber-200 dark:border-amber-800 p-4">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          {/* Header */}
          <div className="flex items-center space-x-2">
            <div className="flex items-center px-2 py-1 bg-amber-100 dark:bg-amber-900/50 rounded-full">
              <span className="text-amber-600 dark:text-amber-300 font-bold text-sm">
                {bid.stake_amount}
              </span>
              <span className="text-amber-500 dark:text-amber-400 text-xs ml-1">credits</span>
            </div>
            <span className="text-sm text-gray-600 dark:text-gray-400">
              from unknown sender
            </span>
          </div>

          {/* Message preview */}
          <p className="mt-2 text-sm text-gray-700 dark:text-gray-300 line-clamp-2">
            {bid.message_preview}
          </p>

          {/* Expiry */}
          <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
            {isExpired ? (
              <span className="text-red-500">Expired</span>
            ) : (
              <>Expires {expiresAt.toLocaleDateString()}</>
            )}
          </p>
        </div>
      </div>

      {/* Actions */}
      {!isExpired && (
        <div className="mt-4 flex space-x-2">
          <button
            onClick={() => onView(bid.id)}
            disabled={loading}
            className="flex-1 px-3 py-2 bg-amber-600 hover:bg-amber-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
          >
            View (Claim {bid.stake_amount} credits)
          </button>
          <button
            onClick={() => onReject(bid.id)}
            disabled={loading}
            className="px-3 py-2 bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
            title="Refund stake to sender"
          >
            Reject
          </button>
          <button
            onClick={() => onReportSpam(bid.id)}
            disabled={loading}
            className="px-3 py-2 bg-red-100 hover:bg-red-200 dark:bg-red-900/30 dark:hover:bg-red-900/50 text-red-700 dark:text-red-300 text-sm font-medium rounded-lg transition-colors disabled:opacity-50"
            title="Burn stake and report"
          >
            Spam
          </button>
        </div>
      )}
    </div>
  );
}

// Introduction form
interface CreateIntroductionProps {
  onSubmit: (data: {
    introducee_did: string;
    target_did: string;
    message: string;
  }) => void;
  loading?: boolean;
  className?: string;
}

export function CreateIntroductionForm({
  onSubmit,
  loading = false,
  className = '',
}: CreateIntroductionProps) {
  const [introduceeDid, setIntroduceeDid] = useState('');
  const [targetDid, setTargetDid] = useState('');
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      introducee_did: introduceeDid,
      target_did: targetDid,
      message,
    });
  };

  return (
    <form onSubmit={handleSubmit} className={`space-y-4 ${className}`}>
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Introduce (DID)
        </label>
        <input
          type="text"
          value={introduceeDid}
          onChange={(e) => setIntroduceeDid(e.target.value)}
          placeholder="did:mycelix:..."
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          To (DID)
        </label>
        <input
          type="text"
          value={targetDid}
          onChange={(e) => setTargetDid(e.target.value)}
          placeholder="did:mycelix:..."
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Introduction Message
        </label>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="I'd like to introduce you to..."
          rows={3}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
          required
        />
      </div>

      <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-3">
        <p className="text-xs text-amber-800 dark:text-amber-200">
          <strong>Note:</strong> You will stake 10% of your trust with the recipient on this introduction.
          If they accept and have a positive experience, your stake is returned with a bonus.
          If the introducee behaves badly, you lose your stake.
        </p>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50"
      >
        {loading ? 'Creating...' : 'Create Introduction'}
      </button>
    </form>
  );
}

// Attention policy settings
interface AttentionPolicySettingsProps {
  policy: AttentionPolicy;
  onUpdate: (policy: AttentionPolicy) => void;
  loading?: boolean;
  className?: string;
}

export function AttentionPolicySettings({
  policy,
  onUpdate,
  loading = false,
  className = '',
}: AttentionPolicySettingsProps) {
  const [localPolicy, setLocalPolicy] = useState(policy);

  const handleSave = () => {
    onUpdate(localPolicy);
  };

  return (
    <div className={`space-y-4 ${className}`}>
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
        Attention Settings
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400">
        Control how unknown senders can reach you.
      </p>

      <div className="space-y-4">
        {/* Minimum stake */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Minimum Stake (credits)
          </label>
          <input
            type="number"
            value={localPolicy.min_stake}
            onChange={(e) => setLocalPolicy({ ...localPolicy, min_stake: parseInt(e.target.value) || 0 })}
            min={0}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
          />
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Unknown senders must stake at least this many credits to reach your inbox.
          </p>
        </div>

        {/* Trusted free toggle */}
        <div className="flex items-center justify-between">
          <div>
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Trusted Senders Free
            </span>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              People you trust can email without staking.
            </p>
          </div>
          <button
            onClick={() => setLocalPolicy({ ...localPolicy, trusted_free: !localPolicy.trusted_free })}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
              localPolicy.trusted_free ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                localPolicy.trusted_free ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
        </div>

        {/* Introduction discount */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Introduction Discount (%)
          </label>
          <input
            type="number"
            value={(localPolicy.introduction_discount * 100).toFixed(0)}
            onChange={(e) =>
              setLocalPolicy({
                ...localPolicy,
                introduction_discount: (parseInt(e.target.value) || 0) / 100,
              })
            }
            min={0}
            max={100}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
          />
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Discount on stake for people introduced by someone you trust.
          </p>
        </div>

        {/* Auto refund toggle */}
        <div className="flex items-center justify-between">
          <div>
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Auto-Refund Trusted
            </span>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Refund stake if you reply positively (builds trust).
            </p>
          </div>
          <button
            onClick={() => setLocalPolicy({ ...localPolicy, auto_refund_trusted: !localPolicy.auto_refund_trusted })}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
              localPolicy.auto_refund_trusted ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                localPolicy.auto_refund_trusted ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
        </div>
      </div>

      <button
        onClick={handleSave}
        disabled={loading}
        className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50"
      >
        {loading ? 'Saving...' : 'Save Settings'}
      </button>
    </div>
  );
}

// Combined consent manager component
interface ConsentManagerProps {
  introductions: Introduction[];
  attentionBids: AttentionBid[];
  policy: AttentionPolicy;
  onAcceptIntro: (id: string) => void;
  onDeclineIntro: (id: string) => void;
  onViewBid: (id: string) => void;
  onRejectBid: (id: string) => void;
  onReportSpam: (id: string) => void;
  onUpdatePolicy: (policy: AttentionPolicy) => void;
  className?: string;
}

export default function ConsentManager({
  introductions,
  attentionBids,
  policy,
  onAcceptIntro,
  onDeclineIntro,
  onViewBid,
  onRejectBid,
  onReportSpam,
  onUpdatePolicy,
  className = '',
}: ConsentManagerProps) {
  const [activeTab, setActiveTab] = useState<'introductions' | 'bids' | 'settings'>('introductions');

  const pendingIntros = introductions.filter((i) => i.status === 'pending');
  const pendingBids = attentionBids.filter((b) => new Date(b.expires_at) > new Date());

  return (
    <div className={`bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 ${className}`}>
      {/* Header with tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="flex -mb-px">
          <button
            onClick={() => setActiveTab('introductions')}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'introductions'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            Introductions
            {pendingIntros.length > 0 && (
              <span className="ml-2 px-2 py-0.5 text-xs bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300 rounded-full">
                {pendingIntros.length}
              </span>
            )}
          </button>
          <button
            onClick={() => setActiveTab('bids')}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'bids'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            Attention Bids
            {pendingBids.length > 0 && (
              <span className="ml-2 px-2 py-0.5 text-xs bg-amber-100 dark:bg-amber-900 text-amber-600 dark:text-amber-300 rounded-full">
                {pendingBids.length}
              </span>
            )}
          </button>
          <button
            onClick={() => setActiveTab('settings')}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'settings'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            Settings
          </button>
        </nav>
      </div>

      {/* Content */}
      <div className="p-4">
        {activeTab === 'introductions' && (
          <div className="space-y-3">
            {pendingIntros.length === 0 ? (
              <p className="text-sm text-gray-500 dark:text-gray-400 text-center py-8">
                No pending introductions
              </p>
            ) : (
              pendingIntros.map((intro) => (
                <IntroductionCard
                  key={intro.id}
                  introduction={intro}
                  onAccept={onAcceptIntro}
                  onDecline={onDeclineIntro}
                />
              ))
            )}
          </div>
        )}

        {activeTab === 'bids' && (
          <div className="space-y-3">
            {pendingBids.length === 0 ? (
              <p className="text-sm text-gray-500 dark:text-gray-400 text-center py-8">
                No attention bids
              </p>
            ) : (
              pendingBids.map((bid) => (
                <AttentionBidCard
                  key={bid.id}
                  bid={bid}
                  onView={onViewBid}
                  onReject={onRejectBid}
                  onReportSpam={onReportSpam}
                />
              ))
            )}
          </div>
        )}

        {activeTab === 'settings' && (
          <AttentionPolicySettings policy={policy} onUpdate={onUpdatePolicy} />
        )}
      </div>
    </div>
  );
}
