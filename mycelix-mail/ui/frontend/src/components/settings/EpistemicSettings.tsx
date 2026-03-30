// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Settings Panel
 *
 * Comprehensive settings for epistemic mail features:
 * - Trust network configuration
 * - AI insights preferences
 * - Notification settings
 * - Privacy controls
 * - Quarantine policies
 *
 * Integrates with the main settings page.
 */

import { useState, useEffect } from 'react';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// Epistemic settings store
export interface EpistemicSettingsState {
  // Trust settings
  trust: {
    enabled: boolean;
    autoFetchSummaries: boolean;
    defaultTrustScore: number;
    trustDecayDays: number;
    maxPathLength: number;
    requireMinTrust: number;
  };

  // Quarantine settings
  quarantine: {
    enabled: boolean;
    policy: 'strict' | 'moderate' | 'relaxed';
    autoPromoteAfterDays: number;
    notifyOnQuarantine: boolean;
    allowAttachments: boolean;
  };

  // AI settings
  ai: {
    enabled: boolean;
    intentDetection: boolean;
    prioritization: boolean;
    summarization: boolean;
    replySuggestions: boolean;
    localProcessing: boolean;
    modelSize: 'small' | 'medium' | 'large';
  };

  // Notification settings
  notifications: {
    enabled: boolean;
    attestationReceived: boolean;
    introductionRequest: boolean;
    trustScoreChange: boolean;
    credentialExpiring: boolean;
    quarantineAlert: boolean;
    weeklyDigest: boolean;
  };

  // Privacy settings
  privacy: {
    shareReadReceipts: boolean;
    allowIntroductions: boolean;
    publicProfile: boolean;
    showInTrustGraph: boolean;
  };

  // Display settings
  display: {
    showTrustBadges: boolean;
    showEpistemicTier: boolean;
    showAIInsights: boolean;
    compactMode: boolean;
    defaultSidebarView: 'insights' | 'contact' | 'thread' | 'none';
  };

  // Actions
  updateTrust: (updates: Partial<EpistemicSettingsState['trust']>) => void;
  updateQuarantine: (updates: Partial<EpistemicSettingsState['quarantine']>) => void;
  updateAI: (updates: Partial<EpistemicSettingsState['ai']>) => void;
  updateNotifications: (updates: Partial<EpistemicSettingsState['notifications']>) => void;
  updatePrivacy: (updates: Partial<EpistemicSettingsState['privacy']>) => void;
  updateDisplay: (updates: Partial<EpistemicSettingsState['display']>) => void;
  resetToDefaults: () => void;
}

const defaultSettings: Omit<EpistemicSettingsState, 'updateTrust' | 'updateQuarantine' | 'updateAI' | 'updateNotifications' | 'updatePrivacy' | 'updateDisplay' | 'resetToDefaults'> = {
  trust: {
    enabled: true,
    autoFetchSummaries: true,
    defaultTrustScore: 0.5,
    trustDecayDays: 365,
    maxPathLength: 3,
    requireMinTrust: 0.3,
  },
  quarantine: {
    enabled: true,
    policy: 'moderate',
    autoPromoteAfterDays: 0,
    notifyOnQuarantine: true,
    allowAttachments: false,
  },
  ai: {
    enabled: true,
    intentDetection: true,
    prioritization: true,
    summarization: true,
    replySuggestions: false,
    localProcessing: true,
    modelSize: 'medium',
  },
  notifications: {
    enabled: true,
    attestationReceived: true,
    introductionRequest: true,
    trustScoreChange: true,
    credentialExpiring: true,
    quarantineAlert: true,
    weeklyDigest: false,
  },
  privacy: {
    shareReadReceipts: false,
    allowIntroductions: true,
    publicProfile: false,
    showInTrustGraph: true,
  },
  display: {
    showTrustBadges: true,
    showEpistemicTier: true,
    showAIInsights: true,
    compactMode: false,
    defaultSidebarView: 'insights',
  },
};

export const useEpistemicSettings = create<EpistemicSettingsState>()(
  persist(
    (set) => ({
      ...defaultSettings,

      updateTrust: (updates) =>
        set((state) => ({ trust: { ...state.trust, ...updates } })),

      updateQuarantine: (updates) =>
        set((state) => ({ quarantine: { ...state.quarantine, ...updates } })),

      updateAI: (updates) =>
        set((state) => ({ ai: { ...state.ai, ...updates } })),

      updateNotifications: (updates) =>
        set((state) => ({ notifications: { ...state.notifications, ...updates } })),

      updatePrivacy: (updates) =>
        set((state) => ({ privacy: { ...state.privacy, ...updates } })),

      updateDisplay: (updates) =>
        set((state) => ({ display: { ...state.display, ...updates } })),

      resetToDefaults: () => set(defaultSettings),
    }),
    {
      name: 'epistemic-settings',
    }
  )
);

// Toggle switch component
function Toggle({
  enabled,
  onChange,
  label,
  description,
  disabled = false,
}: {
  enabled: boolean;
  onChange: (value: boolean) => void;
  label: string;
  description?: string;
  disabled?: boolean;
}) {
  return (
    <label className={`flex items-start justify-between py-3 ${disabled ? 'opacity-50' : 'cursor-pointer'}`}>
      <div className="flex-1 min-w-0 pr-4">
        <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{label}</p>
        {description && (
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{description}</p>
        )}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={enabled}
        disabled={disabled}
        onClick={() => !disabled && onChange(!enabled)}
        className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${
          enabled ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'
        } ${disabled ? '' : 'focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'}`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transform transition-transform ${
            enabled ? 'translate-x-5' : 'translate-x-0'
          }`}
        />
      </button>
    </label>
  );
}

// Slider component
function Slider({
  value,
  onChange,
  label,
  description,
  min,
  max,
  step = 1,
  formatValue,
}: {
  value: number;
  onChange: (value: number) => void;
  label: string;
  description?: string;
  min: number;
  max: number;
  step?: number;
  formatValue?: (value: number) => string;
}) {
  return (
    <div className="py-3">
      <div className="flex items-center justify-between mb-2">
        <div>
          <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{label}</p>
          {description && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{description}</p>
          )}
        </div>
        <span className="text-sm font-medium text-blue-600 dark:text-blue-400">
          {formatValue ? formatValue(value) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
      />
    </div>
  );
}

// Select component
function Select({
  value,
  onChange,
  label,
  description,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  label: string;
  description?: string;
  options: Array<{ value: string; label: string }>;
}) {
  return (
    <div className="py-3">
      <div className="flex items-center justify-between">
        <div className="flex-1 min-w-0 pr-4">
          <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{label}</p>
          {description && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{description}</p>
          )}
        </div>
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="px-3 py-1.5 text-sm border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

// Section component
function SettingsSection({
  title,
  icon,
  description,
  children,
}: {
  title: string;
  icon: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
        <div className="flex items-center gap-2">
          <span className="text-lg">{icon}</span>
          <h3 className="font-semibold text-gray-900 dark:text-gray-100">{title}</h3>
        </div>
        {description && (
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{description}</p>
        )}
      </div>
      <div className="px-4 divide-y divide-gray-100 dark:divide-gray-800">
        {children}
      </div>
    </div>
  );
}

// Main settings component
interface EpistemicSettingsProps {
  onClose?: () => void;
}

export default function EpistemicSettings({ onClose }: EpistemicSettingsProps) {
  const settings = useEpistemicSettings();
  const [activeTab, setActiveTab] = useState<'trust' | 'ai' | 'notifications' | 'privacy' | 'display'>('trust');

  const tabs = [
    { id: 'trust', label: 'Trust & Security', icon: '🛡️' },
    { id: 'ai', label: 'AI Insights', icon: '🤖' },
    { id: 'notifications', label: 'Notifications', icon: '🔔' },
    { id: 'privacy', label: 'Privacy', icon: '🔐' },
    { id: 'display', label: 'Display', icon: '🎨' },
  ];

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">Epistemic Settings</h2>
          <p className="text-sm text-gray-500 dark:text-gray-400">Configure trust, AI, and privacy features</p>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
          >
            <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 dark:border-gray-700 px-6 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center gap-2 px-4 py-3 text-sm font-medium whitespace-nowrap border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Trust & Security Tab */}
        {activeTab === 'trust' && (
          <>
            <SettingsSection
              title="Trust Network"
              icon="🌐"
              description="Configure how trust is calculated and managed"
            >
              <Toggle
                enabled={settings.trust.enabled}
                onChange={(v) => settings.updateTrust({ enabled: v })}
                label="Enable Trust System"
                description="Show trust indicators and filter by trust level"
              />
              <Toggle
                enabled={settings.trust.autoFetchSummaries}
                onChange={(v) => settings.updateTrust({ autoFetchSummaries: v })}
                label="Auto-fetch Trust Summaries"
                description="Automatically fetch trust data for senders"
                disabled={!settings.trust.enabled}
              />
              <Slider
                value={settings.trust.defaultTrustScore}
                onChange={(v) => settings.updateTrust({ defaultTrustScore: v })}
                label="Default Trust Score"
                description="Trust score for new contacts"
                min={0}
                max={1}
                step={0.1}
                formatValue={(v) => `${Math.round(v * 100)}%`}
              />
              <Slider
                value={settings.trust.maxPathLength}
                onChange={(v) => settings.updateTrust({ maxPathLength: v })}
                label="Max Trust Path Length"
                description="Maximum hops for transitive trust"
                min={1}
                max={5}
                formatValue={(v) => `${v} hops`}
              />
              <Slider
                value={settings.trust.trustDecayDays}
                onChange={(v) => settings.updateTrust({ trustDecayDays: v })}
                label="Trust Decay Period"
                description="Days before trust attestations expire"
                min={30}
                max={730}
                step={30}
                formatValue={(v) => `${v} days`}
              />
            </SettingsSection>

            <SettingsSection
              title="Quarantine"
              icon="🛡️"
              description="Control how untrusted emails are handled"
            >
              <Toggle
                enabled={settings.quarantine.enabled}
                onChange={(v) => settings.updateQuarantine({ enabled: v })}
                label="Enable Quarantine"
                description="Hold emails from untrusted senders for review"
              />
              <Select
                value={settings.quarantine.policy}
                onChange={(v) => settings.updateQuarantine({ policy: v as any })}
                label="Quarantine Policy"
                description="How strict to be with unknown senders"
                options={[
                  { value: 'strict', label: 'Strict - Quarantine all unknown' },
                  { value: 'moderate', label: 'Moderate - Use trust heuristics' },
                  { value: 'relaxed', label: 'Relaxed - Only quarantine suspicious' },
                ]}
              />
              <Toggle
                enabled={settings.quarantine.notifyOnQuarantine}
                onChange={(v) => settings.updateQuarantine({ notifyOnQuarantine: v })}
                label="Quarantine Notifications"
                description="Notify when emails are quarantined"
                disabled={!settings.quarantine.enabled}
              />
              <Toggle
                enabled={settings.quarantine.allowAttachments}
                onChange={(v) => settings.updateQuarantine({ allowAttachments: v })}
                label="Allow Attachments from Quarantine"
                description="Allow downloading attachments from quarantined emails"
                disabled={!settings.quarantine.enabled}
              />
            </SettingsSection>
          </>
        )}

        {/* AI Insights Tab */}
        {activeTab === 'ai' && (
          <SettingsSection
            title="AI Features"
            icon="🤖"
            description="Configure AI-powered email analysis"
          >
            <Toggle
              enabled={settings.ai.enabled}
              onChange={(v) => settings.updateAI({ enabled: v })}
              label="Enable AI Insights"
              description="Use AI to analyze emails and provide insights"
            />
            <Toggle
              enabled={settings.ai.localProcessing}
              onChange={(v) => settings.updateAI({ localProcessing: v })}
              label="Local Processing Only"
              description="Process all data locally (never sent to cloud)"
              disabled={!settings.ai.enabled}
            />
            <Select
              value={settings.ai.modelSize}
              onChange={(v) => settings.updateAI({ modelSize: v as any })}
              label="AI Model Size"
              description="Larger models are more accurate but slower"
              options={[
                { value: 'small', label: 'Small (Fast, less accurate)' },
                { value: 'medium', label: 'Medium (Balanced)' },
                { value: 'large', label: 'Large (Slow, most accurate)' },
              ]}
            />
            <Toggle
              enabled={settings.ai.intentDetection}
              onChange={(v) => settings.updateAI({ intentDetection: v })}
              label="Intent Detection"
              description="Detect questions, requests, and action items"
              disabled={!settings.ai.enabled}
            />
            <Toggle
              enabled={settings.ai.prioritization}
              onChange={(v) => settings.updateAI({ prioritization: v })}
              label="Smart Prioritization"
              description="Highlight urgent and important emails"
              disabled={!settings.ai.enabled}
            />
            <Toggle
              enabled={settings.ai.summarization}
              onChange={(v) => settings.updateAI({ summarization: v })}
              label="Thread Summaries"
              description="Generate summaries for long conversations"
              disabled={!settings.ai.enabled}
            />
            <Toggle
              enabled={settings.ai.replySuggestions}
              onChange={(v) => settings.updateAI({ replySuggestions: v })}
              label="Reply Suggestions"
              description="Suggest responses based on email content"
              disabled={!settings.ai.enabled}
            />
          </SettingsSection>
        )}

        {/* Notifications Tab */}
        {activeTab === 'notifications' && (
          <SettingsSection
            title="Trust Notifications"
            icon="🔔"
            description="Choose which trust events to be notified about"
          >
            <Toggle
              enabled={settings.notifications.enabled}
              onChange={(v) => settings.updateNotifications({ enabled: v })}
              label="Enable Notifications"
              description="Show notifications for trust-related events"
            />
            <Toggle
              enabled={settings.notifications.attestationReceived}
              onChange={(v) => settings.updateNotifications({ attestationReceived: v })}
              label="New Attestations"
              description="When someone vouches for you"
              disabled={!settings.notifications.enabled}
            />
            <Toggle
              enabled={settings.notifications.introductionRequest}
              onChange={(v) => settings.updateNotifications({ introductionRequest: v })}
              label="Introduction Requests"
              description="When someone wants to introduce you"
              disabled={!settings.notifications.enabled}
            />
            <Toggle
              enabled={settings.notifications.trustScoreChange}
              onChange={(v) => settings.updateNotifications({ trustScoreChange: v })}
              label="Trust Score Changes"
              description="When a contact's trust score changes significantly"
              disabled={!settings.notifications.enabled}
            />
            <Toggle
              enabled={settings.notifications.credentialExpiring}
              onChange={(v) => settings.updateNotifications({ credentialExpiring: v })}
              label="Credential Expiration"
              description="When your credentials are about to expire"
              disabled={!settings.notifications.enabled}
            />
            <Toggle
              enabled={settings.notifications.quarantineAlert}
              onChange={(v) => settings.updateNotifications({ quarantineAlert: v })}
              label="Quarantine Alerts"
              description="When emails are held in quarantine"
              disabled={!settings.notifications.enabled}
            />
            <Toggle
              enabled={settings.notifications.weeklyDigest}
              onChange={(v) => settings.updateNotifications({ weeklyDigest: v })}
              label="Weekly Digest"
              description="Summary of trust network activity"
              disabled={!settings.notifications.enabled}
            />
          </SettingsSection>
        )}

        {/* Privacy Tab */}
        {activeTab === 'privacy' && (
          <SettingsSection
            title="Privacy Controls"
            icon="🔐"
            description="Manage how your information is shared"
          >
            <Toggle
              enabled={settings.privacy.publicProfile}
              onChange={(v) => settings.updatePrivacy({ publicProfile: v })}
              label="Public Profile"
              description="Allow others to see your profile and credentials"
            />
            <Toggle
              enabled={settings.privacy.showInTrustGraph}
              onChange={(v) => settings.updatePrivacy({ showInTrustGraph: v })}
              label="Appear in Trust Graphs"
              description="Allow others to see you in their trust network visualization"
            />
            <Toggle
              enabled={settings.privacy.allowIntroductions}
              onChange={(v) => settings.updatePrivacy({ allowIntroductions: v })}
              label="Allow Introductions"
              description="Let trusted contacts introduce you to others"
            />
            <Toggle
              enabled={settings.privacy.shareReadReceipts}
              onChange={(v) => settings.updatePrivacy({ shareReadReceipts: v })}
              label="Share Read Receipts"
              description="Let senders know when you've read their email"
            />
          </SettingsSection>
        )}

        {/* Display Tab */}
        {activeTab === 'display' && (
          <SettingsSection
            title="Display Options"
            icon="🎨"
            description="Customize how epistemic features appear"
          >
            <Toggle
              enabled={settings.display.showTrustBadges}
              onChange={(v) => settings.updateDisplay({ showTrustBadges: v })}
              label="Show Trust Badges"
              description="Display trust level badges on emails"
            />
            <Toggle
              enabled={settings.display.showEpistemicTier}
              onChange={(v) => settings.updateDisplay({ showEpistemicTier: v })}
              label="Show Epistemic Tiers"
              description="Display T0-T4 tier indicators"
            />
            <Toggle
              enabled={settings.display.showAIInsights}
              onChange={(v) => settings.updateDisplay({ showAIInsights: v })}
              label="Show AI Insights"
              description="Display AI-generated insights on emails"
            />
            <Toggle
              enabled={settings.display.compactMode}
              onChange={(v) => settings.updateDisplay({ compactMode: v })}
              label="Compact Mode"
              description="Show less detail for more emails on screen"
            />
            <Select
              value={settings.display.defaultSidebarView}
              onChange={(v) => settings.updateDisplay({ defaultSidebarView: v as any })}
              label="Default Sidebar View"
              description="Which panel to show when opening an email"
              options={[
                { value: 'insights', label: 'AI Insights' },
                { value: 'contact', label: 'Contact Profile' },
                { value: 'thread', label: 'Thread Summary' },
                { value: 'none', label: 'None (collapsed)' },
              ]}
            />
          </SettingsSection>
        )}

        {/* Reset button */}
        <div className="flex justify-end pt-4">
          <button
            onClick={() => {
              if (confirm('Reset all epistemic settings to defaults?')) {
                settings.resetToDefaults();
              }
            }}
            className="px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
          >
            Reset to Defaults
          </button>
        </div>
      </div>
    </div>
  );
}
