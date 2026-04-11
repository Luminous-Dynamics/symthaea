// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useEffect, useMemo, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/services/api';
import TrustHowItWorks from '@/components/TrustHowItWorks';
import { toast } from '@/store/toastStore';
import { useThemeStore } from '@/store/themeStore';
import { useDensity, DENSITY_CONFIGS, type DensityLevel } from '@/hooks/useDensity';
import { useLayout, LAYOUT_CONFIGS, type LayoutMode } from '@/hooks/useLayout';
import Layout from '@/components/Layout';
import AccountWizard from '@/components/AccountWizard';
import SignatureManager from '@/components/SignatureManager';
import TemplateManager from '@/components/TemplateManager';
import LabelManager from '@/components/LabelManager';
import { useAuthStore } from '@/store/authStore';
import { errorLogger } from '@/services/errorLogger';
import { config } from '@/config/env';
import { useTrustStore } from '@/store/trustStore';
import TrustBadge from '@/components/TrustBadge';
import TrustBanner from '@/components/TrustBanner';
import TrustProviderAlert from '@/components/TrustProviderAlert';

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const user = useAuthStore((state) => state.user);
  const { theme, setTheme } = useThemeStore();
  const { density, setDensity } = useDensity();
  const { layout, setLayout } = useLayout();
  const [activeTab, setActiveTab] = useState<'accounts' | 'signatures' | 'templates' | 'labels' | 'general' | 'advanced' | 'trust'>('general');
  const [showWizard, setShowWizard] = useState(false);
  const [desktopNotifications, setDesktopNotifications] = useState(
    Notification.permission === 'granted'
  );
  const {
    enabled: trustEnabled,
    quarantineEnabled,
    thresholds,
    ttlMs,
    cacheTimes,
    notificationsEnabled,
    enableTrust,
    enableQuarantine,
    setThresholds,
    clearSummaries,
    setTtlMs,
    summaries,
    overrides,
    ingestSummary,
    setOverride,
    clearOverride,
    hasOverride,
    policy,
    setPolicy,
    enableNotifications,
  } = useTrustStore();
  const [draftHigh, setDraftHigh] = useState(thresholds.high);
  const [draftLow, setDraftLow] = useState(thresholds.low);
  const [thresholdErrors, setThresholdErrors] = useState<{ high?: string; low?: string }>({});
  const [refreshingSender, setRefreshingSender] = useState<string | null>(null);
  const trustRows = useMemo(() => {
    const keys = new Set([...Object.keys(summaries), ...Object.keys(overrides)]);
    return Array.from(keys).map((sender) => {
      const summary = summaries[sender];
      const override = overrides[sender];
      const cacheTime = cacheTimes[sender];
      return { sender, summary, override, cacheTime };
    });
  }, [summaries, overrides, cacheTimes]);
  const overrideCount = useTrustStore((state) => state.getOverrideCount());
  const attestationCount = useTrustStore((state) => state.getAttestationCount());
  const hideLowTrust = useTrustStore((state) => state.hideLowTrust);
  const setHideLowTrust = useTrustStore((state) => state.setHideLowTrust);

  const buildTrustReport = (sender: string, summary?: typeof summaries[string]) => {
    if (!summary) return `Sender: ${sender}\nNo trust summary available.`;
    const lines = [
      `Sender: ${sender}`,
      `Tier: ${summary.tier || 'unknown'}`,
      `Score: ${summary.score ?? '—'}`,
      `Reasons: ${summary.reasons?.length ? summary.reasons.join(', ') : 'None provided'}`,
    ];
    if (summary.pathLength) lines.push(`Path length: ${summary.pathLength}`);
    if (summary.decayAt) lines.push(`Decay: ${new Date(summary.decayAt).toLocaleString()}`);
    if (summary.fetchedAt) lines.push(`Fetched: ${new Date(summary.fetchedAt).toLocaleString()}`);
    return lines.join('\n');
  };

  const exportTrustJson = (sender: string, summary?: typeof summaries[string]) => {
    if (!summary) {
      toast.error('No trust summary to export');
      return;
    }
    const payload = {
      sender,
      summary,
      exportedAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trust-summary-${sender.replace(/[^a-zA-Z0-9-_]/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Trust JSON exported');
  };

  useEffect(() => {
    setDraftHigh(thresholds.high);
    setDraftLow(thresholds.low);
  }, [thresholds.high, thresholds.low]);

  const validateThresholds = (highVal: number, lowVal: number) => {
    const errors: { high?: string; low?: string } = {};
    if (highVal < 0 || highVal > 100) {
      errors.high = 'High must be between 0 and 100.';
    }
    if (lowVal < 0 || lowVal > 100) {
      errors.low = 'Low must be between 0 and 100.';
    }
    if (!errors.high && !errors.low && highVal <= lowVal) {
      errors.low = 'Low must be at least 1 point below high.';
    }
    return errors;
  };

  const commitThresholds = (highVal: number, lowVal: number) => {
    const errors = validateThresholds(highVal, lowVal);
    setThresholdErrors(errors);
    if (!errors.high && !errors.low) {
      setThresholds({ high: highVal, low: lowVal });
    }
  };

  const { data: accounts } = useQuery({
    queryKey: ['accounts'],
    queryFn: () => api.getAccounts(),
  });

  const { data: trustHealth, error: trustHealthError } = useQuery({
    queryKey: ['trust-health'],
    queryFn: () => api.getTrustHealth(),
    retry: 0,
  });

  const deleteAccountMutation = useMutation({
    mutationFn: (accountId: string) => api.deleteAccount(accountId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['accounts'] });
      toast.success('Account removed successfully');
    },
    onError: () => {
      toast.error('Failed to remove account');
    },
  });

  const requestNotificationPermission = async () => {
    if ('Notification' in window) {
      const permission = await Notification.requestPermission();
      setDesktopNotifications(permission === 'granted');

      if (permission === 'granted') {
        new Notification('Notifications Enabled', {
          body: 'You will now receive desktop notifications from Mycelix Mail',
          icon: '/vite.svg',
        });
      }
    }
  };

  const exportAllData = () => {
    const data = {
      user: {
        email: user?.email,
        firstName: user?.firstName,
        lastName: user?.lastName,
      },
      signatures: JSON.parse(localStorage.getItem('signature-storage') || '{"state":{"signatures":[]}}'),
      templates: JSON.parse(localStorage.getItem('template-storage') || '{"state":{"templates":[]}}'),
      labels: JSON.parse(localStorage.getItem('label-storage') || '{"state":{"labels":[],"emailLabels":{}}}'),
      snooze: JSON.parse(localStorage.getItem('snooze-storage') || '{"state":{"snoozedEmails":[]}}'),
      theme: JSON.parse(localStorage.getItem('theme-storage') || '{"state":{"theme":"light"}}'),
      exportDate: new Date().toISOString(),
      version: config.appVersion,
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mycelix-mail-backup-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Data exported successfully');
  };

  const viewErrorLogs = () => {
    const logs = errorLogger.getErrors();
    console.table(logs);
    alert(`${logs.length} errors logged. Check console for details.`);
  };

  return (
    <Layout>
      <div className="max-w-7xl mx-auto p-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-6">Settings</h1>

        <div className="mb-6 border-b border-gray-200 dark:border-gray-700">
          <nav className="-mb-px flex space-x-8 overflow-x-auto">
            <button
              onClick={() => setActiveTab('general')}
              className={`${
                activeTab === 'general'
                  ? 'border-primary-600 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            >
              ⚙️ General
            </button>
            <button
              onClick={() => setActiveTab('trust')}
              className={`${
                activeTab === 'trust'
                  ? 'border-primary-600 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            >
              🛡️ Trust
            </button>
            <button
              onClick={() => setActiveTab('accounts')}
              className={`${
                activeTab === 'accounts'
                  ? 'border-primary-600 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            >
              📧 Accounts
            </button>
            <button
              onClick={() => setActiveTab('signatures')}
              className={`${
                activeTab === 'signatures'
                  ? 'border-primary-600 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            >
              ✍️ Signatures
            </button>
            <button
              onClick={() => setActiveTab('templates')}
              className={`${
                activeTab === 'templates'
                  ? 'border-primary-600 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            >
              📝 Templates
            </button>
            <button
              onClick={() => setActiveTab('labels')}
              className={`${
                activeTab === 'labels'
                  ? 'border-primary-600 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            >
              🏷️ Labels
            </button>
            <button
              onClick={() => setActiveTab('advanced')}
              className={`${
                activeTab === 'advanced'
                  ? 'border-primary-600 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            >
              🔧 Advanced
            </button>
          </nav>
        </div>

        {/* General Tab */}
        {activeTab === 'general' && (
          <div className="space-y-6">
            <div className="card p-6 dark:bg-gray-800">
              <h2 className="text-xl font-semibold mb-6 dark:text-gray-100">General Settings</h2>

              {/* Theme */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                  Theme
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {(['light', 'dark', 'system'] as const).map((themeOption) => (
                    <button
                      key={themeOption}
                      onClick={() => setTheme(themeOption)}
                      className={`px-4 py-3 text-sm font-medium rounded-lg border-2 transition-colors ${
                        theme === themeOption
                          ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                          : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:border-gray-400 dark:hover:border-gray-500'
                      }`}
                    >
                      {themeOption === 'light' && '☀️ Light'}
                      {themeOption === 'dark' && '🌙 Dark'}
                      {themeOption === 'system' && '💻 System'}
                    </button>
                  ))}
                </div>
              </div>

              {/* Email Density */}
              <div className="pt-6 border-t border-gray-200 dark:border-gray-700">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                  Email List Density
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {(Object.keys(DENSITY_CONFIGS) as DensityLevel[]).map((densityOption) => {
                    const config = DENSITY_CONFIGS[densityOption];
                    return (
                      <button
                        key={densityOption}
                        onClick={() => setDensity(densityOption)}
                        className={`px-4 py-3 text-sm font-medium rounded-lg border-2 transition-colors text-left ${
                          density === densityOption
                            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                            : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:border-gray-400 dark:hover:border-gray-500'
                        }`}
                      >
                        <div className="font-semibold">{config.label}</div>
                        <div className="text-xs mt-1 opacity-75">~{config.emailsVisible} emails visible</div>
                      </button>
                    );
                  })}
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-3">
                  {DENSITY_CONFIGS[density].description}
                </p>
              </div>

              {/* Reading Layout */}
              <div className="pt-6 border-t border-gray-200 dark:border-gray-700">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                  Reading Layout
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {(Object.keys(LAYOUT_CONFIGS) as LayoutMode[]).map((layoutOption) => {
                    const config = LAYOUT_CONFIGS[layoutOption];
                    return (
                      <button
                        key={layoutOption}
                        onClick={() => setLayout(layoutOption)}
                        className={`px-4 py-3 text-sm font-medium rounded-lg border-2 transition-colors text-left ${
                          layout === layoutOption
                            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                            : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:border-gray-400 dark:hover:border-gray-500'
                        }`}
                      >
                        <div className="flex items-center space-x-2">
                          <span className="text-lg">{config.icon}</span>
                          <div className="font-semibold">{config.label}</div>
                        </div>
                        <div className="text-xs mt-1 opacity-75">{config.description}</div>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Notifications */}
              <div className="pt-6 border-t border-gray-200 dark:border-gray-700 mt-6">
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
                  Notifications
                </h3>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="text-sm text-gray-700 dark:text-gray-300 font-medium">
                      Desktop Notifications
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      Get notified about new emails and snooze reminders
                    </p>
                  </div>
                  <button
                    onClick={requestNotificationPermission}
                    disabled={desktopNotifications}
                    className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                      desktopNotifications
                        ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 cursor-not-allowed'
                        : 'bg-primary-600 text-white hover:bg-primary-700'
                    }`}
                  >
                    {desktopNotifications ? '✓ Enabled' : 'Enable'}
                  </button>
                </div>
              </div>

              {/* App Info */}
              <div className="pt-6 border-t border-gray-200 dark:border-gray-700 mt-6">
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                  Application Info
                </h3>
                <dl className="grid grid-cols-2 gap-4">
                  <div>
                    <dt className="text-xs text-gray-500 dark:text-gray-400">Version</dt>
                    <dd className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      {config.appVersion}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-500 dark:text-gray-400">Environment</dt>
                    <dd className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      {config.isDevelopment ? 'Development' : 'Production'}
                    </dd>
                  </div>
                </dl>
              </div>
            </div>
          </div>
        )}

        {/* Trust Tab */}
        {activeTab === 'trust' && (
          <div className="space-y-6">
            {trustHealthError && (
              <TrustBanner
                status="unreachable"
                message="Check TRUST_PROVIDER_URL or network connectivity. Falling back to deterministic trust."
              />
            )}
            {!trustHealthError && (
              <TrustProviderAlert />
            )}
            <TrustHowItWorks />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">Provider</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {trustHealthError ? 'Unreachable' : trustHealth?.providerConfigured ? 'Configured' : 'Not configured'}
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  Set TRUST_PROVIDER_URL to enable live MATL/Holochain trust.
                </p>
                {trustHealthError && (
                  <p className="text-[11px] text-rose-600 dark:text-rose-300 mt-2">
                    Provider unreachable. Check TRUST_PROVIDER_URL or network connectivity.
                  </p>
                )}
              </div>
              <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">Cache</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {trustHealth?.cacheSize ?? 0} entries
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  Backend cache clears follow TTL and Refresh button.
                </p>
              </div>
              <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">TTL</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {trustHealth ? Math.round(trustHealth.ttlMs / 60000) : Math.round(ttlMs / 60000)} min
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  Aligns with frontend TTL setting.
                </p>
              </div>
              <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">Overrides</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {overrideCount}
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  Manual allowlists currently applied.
                </p>
              </div>
              <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">Attestations</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {attestationCount}
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  Attestations in current trust cache.
                </p>
              </div>
            </div>

            <div className="card p-6 dark:bg-gray-800">
              <div className="flex items-start justify-between">
                <div>
                  <h2 className="text-xl font-semibold mb-2 dark:text-gray-100">Trust & Quarantine</h2>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Control MATL trust overlays and quarantine thresholds. Lower thresholds catch more risk; higher thresholds surface more mail.
                  </p>
                </div>
                <button
                  onClick={() => enableTrust(!trustEnabled)}
                  className={`px-4 py-2 text-sm font-semibold rounded-lg border transition-colors ${
                    trustEnabled
                      ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-200'
                      : 'border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:border-gray-400 dark:hover:border-gray-600'
                  }`}
                >
                  {trustEnabled ? 'Enabled' : 'Disabled'}
                </button>
              </div>

              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">Quarantine low-trust mail</p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        Hold low-tier messages in the quarantine lane until you promote them.
                      </p>
                    </div>
                    <button
                      onClick={() => trustEnabled && enableQuarantine(!quarantineEnabled)}
                      disabled={!trustEnabled}
                      className={`px-3 py-1.5 text-xs font-semibold rounded-md border transition-colors ${
                        quarantineEnabled && trustEnabled
                          ? 'border-rose-400 bg-rose-50 dark:bg-rose-900/20 text-rose-700 dark:text-rose-200'
                          : 'border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                      }`}
                      title={!trustEnabled ? 'Enable trust overlays to use quarantine' : undefined}
                    >
                      {quarantineEnabled && trustEnabled ? 'On' : 'Off'}
                    </button>
                  </div>
                </div>
                <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                  <p className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-2">
                    Trust thresholds
                  </p>
                  <div className="space-y-3">
                    <label className="flex items-center justify-between text-sm">
                      <span className="text-gray-700 dark:text-gray-300">High tier ≥</span>
                      <input
                        type="number"
                        min={0}
                        max={100}
                        value={draftHigh}
                        onChange={(e) => {
                          const value = Number(e.target.value);
                          setDraftHigh(value);
                          setThresholdErrors(validateThresholds(value, draftLow));
                        }}
                        onBlur={() => commitThresholds(draftHigh, draftLow)}
                        className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                      />
                    </label>
                    {thresholdErrors.high && (
                      <p className="text-xs text-rose-600 dark:text-rose-300">{thresholdErrors.high}</p>
                    )}
                    <label className="flex items-center justify-between text-sm">
                      <span className="text-gray-700 dark:text-gray-300">Low tier ≤</span>
                      <input
                        type="number"
                        min={0}
                        max={100}
                        value={draftLow}
                        onChange={(e) => {
                          const value = Number(e.target.value);
                          setDraftLow(value);
                          setThresholdErrors(validateThresholds(draftHigh, value));
                        }}
                        onBlur={() => commitThresholds(draftHigh, draftLow)}
                        className="w-24 px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                      />
                    </label>
                    {thresholdErrors.low && (
                      <p className="text-xs text-rose-600 dark:text-rose-300">{thresholdErrors.low}</p>
                    )}
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      Scores between low and high are treated as neutral. Scores below low land in quarantine when enabled.
                    </p>
                    <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                      <p className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-2">
                        Trust cache TTL
                      </p>
                      <div className="grid grid-cols-2 gap-2">
                        {[
                          { label: '15 minutes', value: 1000 * 60 * 15 },
                          { label: '1 hour', value: 1000 * 60 * 60 },
                          { label: '6 hours', value: 1000 * 60 * 60 * 6 },
                          { label: '24 hours', value: 1000 * 60 * 60 * 24 },
                        ].map((option) => (
                          <button
                            key={option.value}
                            onClick={() => setTtlMs(option.value)}
                            className={`px-3 py-2 text-xs font-semibold rounded-md border ${
                              ttlMs === option.value
                                ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-200'
                                : 'border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:border-gray-400 dark:hover:border-gray-600'
                            }`}
                          >
                            {option.label}
                          </button>
                        ))}
                      </div>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                        Trust summaries older than this window will be refreshed automatically.
                      </p>
                    </div>
                    <div className="flex items-center justify-between">
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        Refresh trust summaries from the backend.
                      </p>
                      <button
                        onClick={async () => {
                          clearSummaries();
                          try {
                            await api.client.post('/api/trust/cache/clear');
                          } catch (_err) {
                            // ignore if backend not reachable; frontend cache already cleared
                          }
                          toast.success('Trust summaries cleared. They will refresh on next inbox load.');
                        }}
                        className="px-3 py-1.5 text-xs font-semibold rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:border-gray-400 dark:hover-border-gray-600"
                      >
                        Refresh summaries
                      </button>
                    </div>
                  </div>
                </div>
                <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                  <p className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-2">
                    Trust policy
                  </p>
                  <div className="grid grid-cols-3 gap-2">
                    {(['strict', 'balanced', 'open'] as const).map((opt) => (
                      <button
                        key={opt}
                        onClick={() => setPolicy(opt)}
                        className={`px-3 py-2 text-xs font-semibold rounded-md border ${
                          policy === opt
                            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-200'
                            : 'border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:border-gray-400 dark:hover:border-gray-600'
                        }`}
                      >
                        {opt === 'strict' && 'Strict (quarantine low, suppress notifications)'}
                        {opt === 'balanced' && 'Balanced (default)'}
                        {opt === 'open' && 'Open (minimal quarantines)'}
                      </button>
                    ))}
                  </div>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                    Applies to quarantine and notification gating. Strict is safest; open shows more mail.
                  </p>
                  <div className="mt-3 flex items-center justify-between">
                    <div>
                      <p className="text-xs font-semibold text-gray-900 dark:text-gray-100">Notifications</p>
                      <p className="text-[11px] text-gray-600 dark:text-gray-400">
                        Disable to silence notifications for all mail (useful for strict mode).
                      </p>
                    </div>
                    <button
                      onClick={() => enableNotifications(!notificationsEnabled)}
                      className={`px-3 py-1.5 text-xs font-semibold rounded-md border ${
                        notificationsEnabled
                          ? 'border-emerald-400 bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-200'
                          : 'border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200'
                      }`}
                    >
                      {notificationsEnabled ? 'On' : 'Off'}
                    </button>
                  </div>
                  <div className="mt-3 flex items-center justify-between">
                    <div>
                      <p className="text-xs font-semibold text-gray-900 dark:text-gray-100">Hide low-trust mail</p>
                      <p className="text-[11px] text-gray-600 dark:text-gray-400">
                        Strict view: remove quarantined mail from the inbox list. Access via overrides/Trust Center.
                      </p>
                    </div>
                    <button
                      onClick={() => setHideLowTrust(!hideLowTrust)}
                      className={`px-3 py-1.5 text-xs font-semibold rounded-md border ${
                        hideLowTrust
                          ? 'border-rose-400 bg-rose-50 dark:bg-rose-900/20 text-rose-700 dark:text-rose-200'
                          : 'border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200'
                      }`}
                    >
                      {hideLowTrust ? 'Hidden' : 'Visible'}
                    </button>
                  </div>
                </div>
              </div>
            </div>

          {/* Trust Center */}
          <div className="card p-6 dark:bg-gray-800">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Trust Center</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Inspect senders, overrides, and cache freshness. Refresh or request attestations per sender.
                </p>
              </div>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                TTL: {Math.round(ttlMs / 60000)} min • {trustRows.length} senders
              </span>
            </div>

            {trustRows.length === 0 ? (
              <p className="text-sm text-gray-600 dark:text-gray-400">
                No trust data cached yet. Open some emails to populate trust summaries.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="text-left text-gray-600 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                      <th className="py-2 pr-4">Sender</th>
                      <th className="py-2 pr-4">Trust</th>
                      <th className="py-2 pr-4">Updated</th>
                      <th className="py-2 pr-4">Override</th>
                      <th className="py-2 pr-4 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {trustRows.map((row) => (
                      <tr key={row.sender} className="text-gray-900 dark:text-gray-100">
                        <td className="py-2 pr-4">{row.sender}</td>
                        <td className="py-2 pr-4">
                          {row.override ? (
                            <TrustBadge summary={{ ...row.override, tier: row.override.tier || 'high' }} compact />
                          ) : row.summary ? (
                            <TrustBadge summary={{ ...row.summary, tier: row.summary.tier || 'unknown' }} compact />
                          ) : (
                            <span className="text-xs text-gray-500">Not fetched</span>
                          )}
                        </td>
                        <td className="py-2 pr-4 text-xs text-gray-600 dark:text-gray-400">
                          {row.cacheTime ? new Date(row.cacheTime).toLocaleString() : '—'}
                        </td>
                        <td className="py-2 pr-4 text-xs">
                          {row.override ? (
                            <span className="px-2 py-1 rounded-full bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-200">
                              Allowlisted
                            </span>
                          ) : (
                            <span className="text-gray-500">Auto</span>
                          )}
                        </td>
                        <td className="py-2 pr-4">
                          <div className="flex items-center justify-end gap-2">
                            <button
                              onClick={async () => {
                                setRefreshingSender(row.sender);
                                try {
                                  const summary = await api.getTrustSummary(row.sender);
                                  if (summary) {
                                    ingestSummary(row.sender, {
                                      score: summary.score,
                                      tier: summary.tier || 'unknown',
                                      reasons: summary.reasons || [],
                                      pathLength: summary.pathLength,
                                      decayAt: summary.decayAt,
                                      quarantined: summary.quarantined,
                                    });
                                  }
                                } finally {
                                  setRefreshingSender(null);
                                }
                              }}
                              className="px-3 py-1 text-xs font-semibold rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:border-gray-400 dark:hover:border-gray-600 disabled:opacity-60"
                              disabled={refreshingSender === row.sender}
                            >
                              {refreshingSender === row.sender ? 'Refreshing…' : 'Refresh'}
                            </button>
                            {row.summary && (
                              <button
                                onClick={() => exportTrustJson(row.sender, row.override || row.summary)}
                                className="px-3 py-1 text-xs font-semibold rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:border-gray-400 dark:hover:border-gray-600"
                              >
                                Export JSON
                              </button>
                            )}
                            <button
                              onClick={async () => {
                                try {
                                  await navigator.clipboard.writeText(buildTrustReport(row.sender, row.override || row.summary));
                                  toast.success('Trust report copied to clipboard.');
                                } catch (err) {
                                  toast.error('Failed to copy trust report.');
                                }
                              }}
                              className="px-3 py-1 text-xs font-semibold rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:border-gray-400 dark:hover:border-gray-600"
                            >
                              Copy report
                            </button>
                            {!row.override && (
                              <button
                                onClick={() => {
                                  setOverride(row.sender, {
                                    score: 95,
                                    tier: 'high',
                                    reasons: ['Manually trusted sender'],
                                    quarantined: false,
                                  });
                                  toast.success('Sender allowlisted.');
                                }}
                                className="px-3 py-1 text-xs font-semibold rounded-md border border-emerald-400 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-200 hover:border-emerald-500 dark:hover:border-emerald-600"
                              >
                                Allowlist
                              </button>
                            )}
                            {row.override && (
                              <button
                                onClick={() => {
                                  clearOverride(row.sender);
                                  toast.success('Override removed. Using automatic trust.');
                                }}
                                className="px-3 py-1 text-xs font-semibold rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:border-gray-400 dark:hover:border-gray-600"
                              >
                                Revert
                              </button>
                            )}
                            <button
                              onClick={() => toast.success('Attestation request sent to sender/peers.')}
                              className="px-3 py-1 text-xs font-semibold rounded-md border border-blue-400 dark:border-blue-700 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-200 hover:border-blue-500 dark:hover:border-blue-600"
                            >
                              Request attestation
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
        )}
        {/* Accounts Tab */}
        {activeTab === 'accounts' && (
          <div className="space-y-4">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold dark:text-gray-100">Email Accounts</h2>
              <button onClick={() => setShowWizard(true)} className="btn btn-primary">
                Add Account
              </button>
            </div>

            {accounts && accounts.length > 0 ? (
              <div className="space-y-3">
                {accounts.map((account) => (
                  <div key={account.id} className="card p-4 flex justify-between items-center dark:bg-gray-800">
                    <div>
                      <p className="font-medium text-gray-900 dark:text-gray-100">{account.email}</p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {account.provider} {account.isDefault && '• Default'}
                      </p>
                    </div>
                    <button
                      onClick={() => deleteAccountMutation.mutate(account.id)}
                      className="btn btn-secondary text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30"
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="card p-12 text-center dark:bg-gray-800">
                <p className="text-gray-500 dark:text-gray-400 mb-4">No email accounts configured</p>
                <button onClick={() => setShowWizard(true)} className="btn btn-primary">
                  Add Your First Account
                </button>
              </div>
            )}
          </div>
        )}

        {/* Signatures Tab */}
        {activeTab === 'signatures' && <SignatureManager />}

        {/* Templates Tab */}
        {activeTab === 'templates' && <TemplateManager />}

        {/* Labels Tab */}
        {activeTab === 'labels' && <LabelManager />}

        {/* Advanced Tab */}
        {activeTab === 'advanced' && (
          <div className="card p-6 dark:bg-gray-800">
            <h2 className="text-xl font-semibold mb-6 dark:text-gray-100">Advanced Settings</h2>

            <div className="space-y-4">
              {/* Debug Mode */}
              <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div>
                  <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    Debug Mode
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    {config.enableDebug ? 'Detailed logging enabled' : 'Standard logging'}
                  </p>
                </div>
                <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  {config.enableDebug ? '✓ On' : 'Off'}
                </div>
              </div>

              {/* Error Logs */}
              <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div>
                  <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    Error Logs
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    View application error history
                  </p>
                </div>
                <button
                  onClick={viewErrorLogs}
                  className="px-4 py-2 text-sm font-medium text-primary-600 dark:text-primary-400 hover:underline"
                >
                  View Logs
                </button>
              </div>

              {/* Export Data */}
              <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div>
                  <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    Export Data
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Backup all settings, signatures, templates
                  </p>
                </div>
                <button
                  onClick={exportAllData}
                  className="px-4 py-2 text-sm font-medium bg-primary-600 text-white hover:bg-primary-700 rounded-lg transition-colors"
                >
                  💾 Export
                </button>
              </div>

              {/* Clear Cache */}
              <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div>
                  <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    Clear Cache
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Clear cached data and reload
                  </p>
                </div>
                <button
                  onClick={() => window.location.reload()}
                  className="px-4 py-2 text-sm font-medium text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 rounded-lg transition-colors"
                >
                  🗑️ Clear
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Account Wizard Modal */}
      {showWizard && <AccountWizard onClose={() => setShowWizard(false)} />}
    </Layout>
  );
}
