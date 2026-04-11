// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Settings Panel Component for Mycelix Mail
 *
 * Comprehensive settings and preferences management UI
 */

import React, { useState, useEffect, useCallback } from 'react';

// Types
export interface UserPreferences {
  // Display
  theme: 'light' | 'dark' | 'system';
  language: string;
  dateFormat: 'relative' | 'absolute' | 'iso';
  timeFormat: '12h' | '24h';
  timezone: string;

  // Email
  emailsPerPage: number;
  defaultFolder: 'inbox' | 'all' | 'important';
  conversationView: boolean;
  showPreview: boolean;
  previewLines: number;

  // Compose
  defaultSignature: string;
  autoSave: boolean;
  autoSaveInterval: number;
  spellCheck: boolean;
  confirmSend: boolean;

  // Notifications
  desktopNotifications: boolean;
  soundEnabled: boolean;
  notifyOnlyTrusted: boolean;
  quietHoursEnabled: boolean;
  quietHoursStart: string;
  quietHoursEnd: string;

  // Trust
  defaultTrustThreshold: number;
  autoBlockUntrusted: boolean;
  showTrustBadges: boolean;
  warnOnLowTrust: boolean;

  // Privacy
  readReceipts: boolean;
  typingIndicators: boolean;
  onlineStatus: boolean;
  encryptByDefault: boolean;

  // Sync
  syncFrequency: number;
  offlineMode: boolean;
  backgroundSync: boolean;

  // Advanced
  developerMode: boolean;
  experimentalFeatures: boolean;
}

export interface SettingsPanelProps {
  preferences: UserPreferences;
  onSave: (preferences: UserPreferences) => Promise<void>;
  onReset?: () => void;
  onExportData?: () => Promise<void>;
  onDeleteAccount?: () => Promise<void>;
}

const DEFAULT_PREFERENCES: UserPreferences = {
  // Display
  theme: 'system',
  language: 'en',
  dateFormat: 'relative',
  timeFormat: '12h',
  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,

  // Email
  emailsPerPage: 50,
  defaultFolder: 'inbox',
  conversationView: true,
  showPreview: true,
  previewLines: 2,

  // Compose
  defaultSignature: '',
  autoSave: true,
  autoSaveInterval: 30,
  spellCheck: true,
  confirmSend: false,

  // Notifications
  desktopNotifications: true,
  soundEnabled: true,
  notifyOnlyTrusted: false,
  quietHoursEnabled: false,
  quietHoursStart: '22:00',
  quietHoursEnd: '07:00',

  // Trust
  defaultTrustThreshold: 0.3,
  autoBlockUntrusted: false,
  showTrustBadges: true,
  warnOnLowTrust: true,

  // Privacy
  readReceipts: false,
  typingIndicators: false,
  onlineStatus: false,
  encryptByDefault: true,

  // Sync
  syncFrequency: 60,
  offlineMode: false,
  backgroundSync: true,

  // Advanced
  developerMode: false,
  experimentalFeatures: false,
};

type SettingsSection =
  | 'general'
  | 'email'
  | 'compose'
  | 'notifications'
  | 'trust'
  | 'privacy'
  | 'sync'
  | 'advanced'
  | 'account';

interface SectionConfig {
  id: SettingsSection;
  label: string;
  icon: string;
}

const SECTIONS: SectionConfig[] = [
  { id: 'general', label: 'General', icon: '⚙️' },
  { id: 'email', label: 'Email', icon: '📧' },
  { id: 'compose', label: 'Compose', icon: '✍️' },
  { id: 'notifications', label: 'Notifications', icon: '🔔' },
  { id: 'trust', label: 'Trust & Security', icon: '🛡️' },
  { id: 'privacy', label: 'Privacy', icon: '🔒' },
  { id: 'sync', label: 'Sync & Offline', icon: '🔄' },
  { id: 'advanced', label: 'Advanced', icon: '🔧' },
  { id: 'account', label: 'Account', icon: '👤' },
];

export function SettingsPanel({
  preferences: initialPreferences,
  onSave,
  onReset,
  onExportData,
  onDeleteAccount,
}: SettingsPanelProps) {
  const [preferences, setPreferences] = useState<UserPreferences>({
    ...DEFAULT_PREFERENCES,
    ...initialPreferences,
  });
  const [activeSection, setActiveSection] = useState<SettingsSection>('general');
  const [isSaving, setIsSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track changes
  useEffect(() => {
    const hasChanged = JSON.stringify(preferences) !== JSON.stringify(initialPreferences);
    setHasChanges(hasChanged);
  }, [preferences, initialPreferences]);

  // Update a preference
  const updatePreference = useCallback(<K extends keyof UserPreferences>(
    key: K,
    value: UserPreferences[K]
  ) => {
    setPreferences(prev => ({ ...prev, [key]: value }));
  }, []);

  // Save preferences
  const handleSave = async () => {
    setIsSaving(true);
    setError(null);
    try {
      await onSave(preferences);
      setHasChanges(false);
    } catch (err) {
      setError('Failed to save preferences. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  // Reset to defaults
  const handleReset = () => {
    if (window.confirm('Reset all settings to default values?')) {
      setPreferences(DEFAULT_PREFERENCES);
      onReset?.();
    }
  };

  // Render section content
  const renderSection = () => {
    switch (activeSection) {
      case 'general':
        return (
          <GeneralSettings
            preferences={preferences}
            onChange={updatePreference}
          />
        );
      case 'email':
        return (
          <EmailSettings
            preferences={preferences}
            onChange={updatePreference}
          />
        );
      case 'compose':
        return (
          <ComposeSettings
            preferences={preferences}
            onChange={updatePreference}
          />
        );
      case 'notifications':
        return (
          <NotificationSettings
            preferences={preferences}
            onChange={updatePreference}
          />
        );
      case 'trust':
        return (
          <TrustSettings
            preferences={preferences}
            onChange={updatePreference}
          />
        );
      case 'privacy':
        return (
          <PrivacySettings
            preferences={preferences}
            onChange={updatePreference}
          />
        );
      case 'sync':
        return (
          <SyncSettings
            preferences={preferences}
            onChange={updatePreference}
          />
        );
      case 'advanced':
        return (
          <AdvancedSettings
            preferences={preferences}
            onChange={updatePreference}
          />
        );
      case 'account':
        return (
          <AccountSettings
            onExportData={onExportData}
            onDeleteAccount={onDeleteAccount}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="settings-panel">
      {/* Sidebar */}
      <nav className="settings-nav" aria-label="Settings sections">
        {SECTIONS.map(section => (
          <button
            key={section.id}
            className={`settings-nav-item ${activeSection === section.id ? 'active' : ''}`}
            onClick={() => setActiveSection(section.id)}
            aria-current={activeSection === section.id ? 'page' : undefined}
          >
            <span className="settings-nav-icon" aria-hidden="true">{section.icon}</span>
            <span className="settings-nav-label">{section.label}</span>
          </button>
        ))}
      </nav>

      {/* Content */}
      <main className="settings-content">
        <header className="settings-header">
          <h2>{SECTIONS.find(s => s.id === activeSection)?.label}</h2>
        </header>

        <div className="settings-body">
          {renderSection()}
        </div>

        {/* Footer */}
        {activeSection !== 'account' && (
          <footer className="settings-footer">
            {error && <div className="settings-error" role="alert">{error}</div>}

            <div className="settings-actions">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleReset}
                disabled={isSaving}
              >
                Reset to Defaults
              </button>

              <button
                type="button"
                className="btn btn-primary"
                onClick={handleSave}
                disabled={!hasChanges || isSaving}
              >
                {isSaving ? 'Saving...' : 'Save Changes'}
              </button>
            </div>
          </footer>
        )}
      </main>
    </div>
  );
}

// Section Components

interface SectionProps {
  preferences: UserPreferences;
  onChange: <K extends keyof UserPreferences>(key: K, value: UserPreferences[K]) => void;
}

function GeneralSettings({ preferences, onChange }: SectionProps) {
  return (
    <div className="settings-section">
      <SettingGroup title="Appearance">
        <SelectSetting
          label="Theme"
          value={preferences.theme}
          options={[
            { value: 'light', label: 'Light' },
            { value: 'dark', label: 'Dark' },
            { value: 'system', label: 'System Default' },
          ]}
          onChange={v => onChange('theme', v as 'light' | 'dark' | 'system')}
        />

        <SelectSetting
          label="Language"
          value={preferences.language}
          options={[
            { value: 'en', label: 'English' },
            { value: 'es', label: 'Español' },
            { value: 'fr', label: 'Français' },
            { value: 'de', label: 'Deutsch' },
            { value: 'ja', label: '日本語' },
          ]}
          onChange={v => onChange('language', v)}
        />
      </SettingGroup>

      <SettingGroup title="Date & Time">
        <SelectSetting
          label="Date Format"
          value={preferences.dateFormat}
          options={[
            { value: 'relative', label: 'Relative (2 hours ago)' },
            { value: 'absolute', label: 'Absolute (Jan 3, 2026)' },
            { value: 'iso', label: 'ISO (2026-01-03)' },
          ]}
          onChange={v => onChange('dateFormat', v as 'relative' | 'absolute' | 'iso')}
        />

        <SelectSetting
          label="Time Format"
          value={preferences.timeFormat}
          options={[
            { value: '12h', label: '12-hour (2:30 PM)' },
            { value: '24h', label: '24-hour (14:30)' },
          ]}
          onChange={v => onChange('timeFormat', v as '12h' | '24h')}
        />

        <SelectSetting
          label="Timezone"
          value={preferences.timezone}
          options={Intl.supportedValuesOf('timeZone').map(tz => ({
            value: tz,
            label: tz.replace(/_/g, ' '),
          }))}
          onChange={v => onChange('timezone', v)}
        />
      </SettingGroup>
    </div>
  );
}

function EmailSettings({ preferences, onChange }: SectionProps) {
  return (
    <div className="settings-section">
      <SettingGroup title="Inbox">
        <SelectSetting
          label="Default View"
          value={preferences.defaultFolder}
          options={[
            { value: 'inbox', label: 'Inbox' },
            { value: 'all', label: 'All Mail' },
            { value: 'important', label: 'Important' },
          ]}
          onChange={v => onChange('defaultFolder', v as 'inbox' | 'all' | 'important')}
        />

        <RangeSetting
          label="Emails per Page"
          value={preferences.emailsPerPage}
          min={10}
          max={100}
          step={10}
          onChange={v => onChange('emailsPerPage', v)}
        />
      </SettingGroup>

      <SettingGroup title="Display">
        <ToggleSetting
          label="Conversation View"
          description="Group emails by thread"
          value={preferences.conversationView}
          onChange={v => onChange('conversationView', v)}
        />

        <ToggleSetting
          label="Show Preview"
          description="Display email snippet in list"
          value={preferences.showPreview}
          onChange={v => onChange('showPreview', v)}
        />

        {preferences.showPreview && (
          <RangeSetting
            label="Preview Lines"
            value={preferences.previewLines}
            min={1}
            max={5}
            step={1}
            onChange={v => onChange('previewLines', v)}
          />
        )}
      </SettingGroup>
    </div>
  );
}

function ComposeSettings({ preferences, onChange }: SectionProps) {
  return (
    <div className="settings-section">
      <SettingGroup title="Drafts">
        <ToggleSetting
          label="Auto-save Drafts"
          description="Automatically save drafts while composing"
          value={preferences.autoSave}
          onChange={v => onChange('autoSave', v)}
        />

        {preferences.autoSave && (
          <RangeSetting
            label="Auto-save Interval (seconds)"
            value={preferences.autoSaveInterval}
            min={10}
            max={120}
            step={10}
            onChange={v => onChange('autoSaveInterval', v)}
          />
        )}
      </SettingGroup>

      <SettingGroup title="Writing">
        <ToggleSetting
          label="Spell Check"
          description="Check spelling as you type"
          value={preferences.spellCheck}
          onChange={v => onChange('spellCheck', v)}
        />

        <ToggleSetting
          label="Confirm Before Sending"
          description="Show confirmation dialog before sending"
          value={preferences.confirmSend}
          onChange={v => onChange('confirmSend', v)}
        />
      </SettingGroup>

      <SettingGroup title="Signature">
        <TextareaSetting
          label="Default Signature"
          value={preferences.defaultSignature}
          placeholder="Enter your email signature..."
          onChange={v => onChange('defaultSignature', v)}
        />
      </SettingGroup>
    </div>
  );
}

function NotificationSettings({ preferences, onChange }: SectionProps) {
  return (
    <div className="settings-section">
      <SettingGroup title="Notifications">
        <ToggleSetting
          label="Desktop Notifications"
          description="Show notifications for new emails"
          value={preferences.desktopNotifications}
          onChange={v => onChange('desktopNotifications', v)}
        />

        <ToggleSetting
          label="Sound"
          description="Play sound for new emails"
          value={preferences.soundEnabled}
          onChange={v => onChange('soundEnabled', v)}
        />

        <ToggleSetting
          label="Trusted Senders Only"
          description="Only notify for emails from trusted senders"
          value={preferences.notifyOnlyTrusted}
          onChange={v => onChange('notifyOnlyTrusted', v)}
        />
      </SettingGroup>

      <SettingGroup title="Quiet Hours">
        <ToggleSetting
          label="Enable Quiet Hours"
          description="Mute notifications during specified hours"
          value={preferences.quietHoursEnabled}
          onChange={v => onChange('quietHoursEnabled', v)}
        />

        {preferences.quietHoursEnabled && (
          <>
            <TimeSetting
              label="Start Time"
              value={preferences.quietHoursStart}
              onChange={v => onChange('quietHoursStart', v)}
            />
            <TimeSetting
              label="End Time"
              value={preferences.quietHoursEnd}
              onChange={v => onChange('quietHoursEnd', v)}
            />
          </>
        )}
      </SettingGroup>
    </div>
  );
}

function TrustSettings({ preferences, onChange }: SectionProps) {
  return (
    <div className="settings-section">
      <SettingGroup title="Trust Thresholds">
        <RangeSetting
          label="Default Trust Threshold"
          description="Minimum trust level for unrestricted email delivery"
          value={preferences.defaultTrustThreshold}
          min={0}
          max={1}
          step={0.1}
          formatValue={v => `${Math.round(v * 100)}%`}
          onChange={v => onChange('defaultTrustThreshold', v)}
        />
      </SettingGroup>

      <SettingGroup title="Untrusted Senders">
        <ToggleSetting
          label="Auto-block Untrusted"
          description="Automatically block emails from very low trust senders"
          value={preferences.autoBlockUntrusted}
          onChange={v => onChange('autoBlockUntrusted', v)}
        />

        <ToggleSetting
          label="Warn on Low Trust"
          description="Show warning banner for low-trust emails"
          value={preferences.warnOnLowTrust}
          onChange={v => onChange('warnOnLowTrust', v)}
        />
      </SettingGroup>

      <SettingGroup title="Display">
        <ToggleSetting
          label="Show Trust Badges"
          description="Display trust level indicators on emails"
          value={preferences.showTrustBadges}
          onChange={v => onChange('showTrustBadges', v)}
        />
      </SettingGroup>
    </div>
  );
}

function PrivacySettings({ preferences, onChange }: SectionProps) {
  return (
    <div className="settings-section">
      <SettingGroup title="Encryption">
        <ToggleSetting
          label="Encrypt by Default"
          description="End-to-end encrypt all emails"
          value={preferences.encryptByDefault}
          onChange={v => onChange('encryptByDefault', v)}
        />
      </SettingGroup>

      <SettingGroup title="Activity Status">
        <ToggleSetting
          label="Read Receipts"
          description="Let senders know when you read their emails"
          value={preferences.readReceipts}
          onChange={v => onChange('readReceipts', v)}
        />

        <ToggleSetting
          label="Typing Indicators"
          description="Show when you are typing a reply"
          value={preferences.typingIndicators}
          onChange={v => onChange('typingIndicators', v)}
        />

        <ToggleSetting
          label="Online Status"
          description="Show others when you are online"
          value={preferences.onlineStatus}
          onChange={v => onChange('onlineStatus', v)}
        />
      </SettingGroup>
    </div>
  );
}

function SyncSettings({ preferences, onChange }: SectionProps) {
  return (
    <div className="settings-section">
      <SettingGroup title="Synchronization">
        <RangeSetting
          label="Sync Frequency (seconds)"
          description="How often to check for new emails"
          value={preferences.syncFrequency}
          min={15}
          max={300}
          step={15}
          onChange={v => onChange('syncFrequency', v)}
        />

        <ToggleSetting
          label="Background Sync"
          description="Sync emails in the background"
          value={preferences.backgroundSync}
          onChange={v => onChange('backgroundSync', v)}
        />
      </SettingGroup>

      <SettingGroup title="Offline Mode">
        <ToggleSetting
          label="Offline Mode"
          description="Cache emails for offline access"
          value={preferences.offlineMode}
          onChange={v => onChange('offlineMode', v)}
        />
      </SettingGroup>
    </div>
  );
}

function AdvancedSettings({ preferences, onChange }: SectionProps) {
  return (
    <div className="settings-section">
      <SettingGroup title="Developer Options">
        <ToggleSetting
          label="Developer Mode"
          description="Enable developer tools and logging"
          value={preferences.developerMode}
          onChange={v => onChange('developerMode', v)}
        />

        <ToggleSetting
          label="Experimental Features"
          description="Enable experimental features (may be unstable)"
          value={preferences.experimentalFeatures}
          onChange={v => onChange('experimentalFeatures', v)}
        />
      </SettingGroup>
    </div>
  );
}

interface AccountSettingsProps {
  onExportData?: () => Promise<void>;
  onDeleteAccount?: () => Promise<void>;
}

function AccountSettings({ onExportData, onDeleteAccount }: AccountSettingsProps) {
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = async () => {
    if (!onExportData) return;
    setIsExporting(true);
    try {
      await onExportData();
    } finally {
      setIsExporting(false);
    }
  };

  const handleDelete = async () => {
    if (!onDeleteAccount) return;
    const confirmed = window.confirm(
      'Are you sure you want to delete your account? This action cannot be undone.'
    );
    if (confirmed) {
      await onDeleteAccount();
    }
  };

  return (
    <div className="settings-section">
      <SettingGroup title="Data">
        <div className="setting-action">
          <div>
            <h4>Export Your Data</h4>
            <p>Download all your emails, contacts, and settings</p>
          </div>
          <button
            className="btn btn-secondary"
            onClick={handleExport}
            disabled={!onExportData || isExporting}
          >
            {isExporting ? 'Exporting...' : 'Export Data'}
          </button>
        </div>
      </SettingGroup>

      <SettingGroup title="Danger Zone">
        <div className="setting-action danger">
          <div>
            <h4>Delete Account</h4>
            <p>Permanently delete your account and all data</p>
          </div>
          <button
            className="btn btn-danger"
            onClick={handleDelete}
            disabled={!onDeleteAccount}
          >
            Delete Account
          </button>
        </div>
      </SettingGroup>
    </div>
  );
}

// Setting Input Components

interface SettingGroupProps {
  title: string;
  children: React.ReactNode;
}

function SettingGroup({ title, children }: SettingGroupProps) {
  return (
    <div className="setting-group">
      <h3 className="setting-group-title">{title}</h3>
      <div className="setting-group-content">{children}</div>
    </div>
  );
}

interface ToggleSettingProps {
  label: string;
  description?: string;
  value: boolean;
  onChange: (value: boolean) => void;
}

function ToggleSetting({ label, description, value, onChange }: ToggleSettingProps) {
  const id = `toggle-${label.replace(/\s+/g, '-').toLowerCase()}`;
  return (
    <div className="setting-item">
      <div className="setting-info">
        <label htmlFor={id} className="setting-label">{label}</label>
        {description && <p className="setting-description">{description}</p>}
      </div>
      <input
        id={id}
        type="checkbox"
        className="setting-toggle"
        checked={value}
        onChange={e => onChange(e.target.checked)}
      />
    </div>
  );
}

interface SelectSettingProps {
  label: string;
  description?: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}

function SelectSetting({ label, description, value, options, onChange }: SelectSettingProps) {
  const id = `select-${label.replace(/\s+/g, '-').toLowerCase()}`;
  return (
    <div className="setting-item">
      <div className="setting-info">
        <label htmlFor={id} className="setting-label">{label}</label>
        {description && <p className="setting-description">{description}</p>}
      </div>
      <select
        id={id}
        className="setting-select"
        value={value}
        onChange={e => onChange(e.target.value)}
      >
        {options.map(opt => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

interface RangeSettingProps {
  label: string;
  description?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  formatValue?: (value: number) => string;
  onChange: (value: number) => void;
}

function RangeSetting({
  label,
  description,
  value,
  min,
  max,
  step,
  formatValue = v => String(v),
  onChange,
}: RangeSettingProps) {
  const id = `range-${label.replace(/\s+/g, '-').toLowerCase()}`;
  return (
    <div className="setting-item">
      <div className="setting-info">
        <label htmlFor={id} className="setting-label">{label}</label>
        {description && <p className="setting-description">{description}</p>}
      </div>
      <div className="setting-range">
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={e => onChange(Number(e.target.value))}
        />
        <span className="setting-range-value">{formatValue(value)}</span>
      </div>
    </div>
  );
}

interface TextareaSettingProps {
  label: string;
  value: string;
  placeholder?: string;
  onChange: (value: string) => void;
}

function TextareaSetting({ label, value, placeholder, onChange }: TextareaSettingProps) {
  const id = `textarea-${label.replace(/\s+/g, '-').toLowerCase()}`;
  return (
    <div className="setting-item vertical">
      <label htmlFor={id} className="setting-label">{label}</label>
      <textarea
        id={id}
        className="setting-textarea"
        value={value}
        placeholder={placeholder}
        rows={4}
        onChange={e => onChange(e.target.value)}
      />
    </div>
  );
}

interface TimeSettingProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
}

function TimeSetting({ label, value, onChange }: TimeSettingProps) {
  const id = `time-${label.replace(/\s+/g, '-').toLowerCase()}`;
  return (
    <div className="setting-item">
      <label htmlFor={id} className="setting-label">{label}</label>
      <input
        id={id}
        type="time"
        className="setting-time"
        value={value}
        onChange={e => onChange(e.target.value)}
      />
    </div>
  );
}

export default SettingsPanel;
