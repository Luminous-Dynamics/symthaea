// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Privacy Center Components
 *
 * Provides email expiration, message recall, tracking detection,
 * read receipt control, anonymous mode, and link proxying settings.
 */

import React, { useState, useEffect } from 'react';

// ============================================================================
// Types
// ============================================================================

interface ExpirationSettings {
  type: 'time' | 'views' | 'on_read';
  hours?: number;
  maxViews?: number;
  notifyRecipient: boolean;
  deleteAttachments: boolean;
  leavePlaceholder: boolean;
}

interface RecallStatus {
  canRecall: boolean;
  reason?: string;
  supportedRecipients: string[];
  unsupportedRecipients: string[];
}

interface RecallRequest {
  id: string;
  emailId: string;
  status: 'pending' | 'partial_success' | 'success' | 'failed';
  recipientStatuses: RecipientRecallStatus[];
}

interface RecipientRecallStatus {
  email: string;
  status: 'pending' | 'recalled' | 'already_read' | 'not_supported' | 'failed';
  reason?: string;
}

interface TrackingDetection {
  emailId: string;
  trackingPixels: DetectedTracker[];
  trackingLinks: DetectedTracker[];
  totalTrackers: number;
  riskLevel: 'none' | 'low' | 'medium' | 'high';
}

interface DetectedTracker {
  type: 'pixel' | 'link' | 'beacon';
  url: string;
  domain: string;
  company?: string;
  purpose: string;
  blocked: boolean;
}

interface ReadReceiptSettings {
  blockOutgoing: boolean;
  blockIncomingRequests: boolean;
  autoDenyMdn: boolean;
  whitelist: string[];
  blacklist: string[];
}

interface AnonymousIdentity {
  id: string;
  aliasEmail: string;
  displayName?: string;
  createdAt: string;
  expiresAt?: string;
  emailCount: number;
  isActive: boolean;
}

interface LinkProxySettings {
  enabled: boolean;
  stripUtmParams: boolean;
  blockKnownTrackers: boolean;
  warnOnRedirect: boolean;
  whitelistDomains: string[];
}

// ============================================================================
// Email Expiration Composer
// ============================================================================

interface ExpirationComposerProps {
  onApply: (settings: ExpirationSettings) => void;
  onCancel: () => void;
}

export function ExpirationComposer({ onApply, onCancel }: ExpirationComposerProps) {
  const [settings, setSettings] = useState<ExpirationSettings>({
    type: 'time',
    hours: 24,
    notifyRecipient: true,
    deleteAttachments: true,
    leavePlaceholder: true,
  });

  return (
    <div className="expiration-composer">
      <h3><TimerIcon /> Set Email Expiration</h3>

      <div className="expiration-types">
        <label className={settings.type === 'time' ? 'selected' : ''}>
          <input
            type="radio"
            checked={settings.type === 'time'}
            onChange={() => setSettings({ ...settings, type: 'time' })}
          />
          <ClockIcon />
          <span>Time-based</span>
        </label>
        <label className={settings.type === 'views' ? 'selected' : ''}>
          <input
            type="radio"
            checked={settings.type === 'views'}
            onChange={() => setSettings({ ...settings, type: 'views' })}
          />
          <EyeIcon />
          <span>View-based</span>
        </label>
        <label className={settings.type === 'on_read' ? 'selected' : ''}>
          <input
            type="radio"
            checked={settings.type === 'on_read'}
            onChange={() => setSettings({ ...settings, type: 'on_read' })}
          />
          <CheckCircleIcon />
          <span>After read</span>
        </label>
      </div>

      {settings.type === 'time' && (
        <div className="time-setting">
          <label>Expires after:</label>
          <select
            value={settings.hours}
            onChange={e => setSettings({ ...settings, hours: parseInt(e.target.value) })}
          >
            <option value={1}>1 hour</option>
            <option value={6}>6 hours</option>
            <option value={12}>12 hours</option>
            <option value={24}>1 day</option>
            <option value={72}>3 days</option>
            <option value={168}>1 week</option>
          </select>
        </div>
      )}

      {settings.type === 'views' && (
        <div className="views-setting">
          <label>Maximum views:</label>
          <input
            type="number"
            min={1}
            max={100}
            value={settings.maxViews || 1}
            onChange={e => setSettings({ ...settings, maxViews: parseInt(e.target.value) })}
          />
        </div>
      )}

      <div className="expiration-options">
        <label>
          <input
            type="checkbox"
            checked={settings.notifyRecipient}
            onChange={e => setSettings({ ...settings, notifyRecipient: e.target.checked })}
          />
          Notify recipient about expiration
        </label>
        <label>
          <input
            type="checkbox"
            checked={settings.deleteAttachments}
            onChange={e => setSettings({ ...settings, deleteAttachments: e.target.checked })}
          />
          Delete attachments on expiration
        </label>
        <label>
          <input
            type="checkbox"
            checked={settings.leavePlaceholder}
            onChange={e => setSettings({ ...settings, leavePlaceholder: e.target.checked })}
          />
          Leave placeholder message
        </label>
      </div>

      <div className="actions">
        <button className="secondary" onClick={onCancel}>Cancel</button>
        <button className="primary" onClick={() => onApply(settings)}>
          Apply Expiration
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Message Recall Panel
// ============================================================================

interface RecallPanelProps {
  emailId: string;
  subject: string;
  sentAt: string;
  recipients: string[];
  onRecall: () => void;
}

export function RecallPanel({ emailId, subject, sentAt, recipients, onRecall }: RecallPanelProps) {
  const [recallStatus, setRecallStatus] = useState<RecallStatus | null>(null);
  const [recallRequest, setRecallRequest] = useState<RecallRequest | null>(null);
  const [loading, setLoading] = useState(true);
  const [recalling, setRecalling] = useState(false);

  useEffect(() => {
    checkRecallEligibility();
  }, [emailId]);

  const checkRecallEligibility = async () => {
    try {
      const response = await fetch(`/api/emails/${emailId}/recall/check`);
      const data = await response.json();
      setRecallStatus(data);
    } catch (error) {
      console.error('Failed to check recall status:', error);
    }
    setLoading(false);
  };

  const handleRecall = async () => {
    setRecalling(true);
    try {
      const response = await fetch(`/api/emails/${emailId}/recall`, { method: 'POST' });
      const data = await response.json();
      setRecallRequest(data);
      onRecall();
    } catch (error) {
      console.error('Failed to recall email:', error);
    }
    setRecalling(false);
  };

  if (loading) {
    return <div className="recall-panel loading">Checking recall eligibility...</div>;
  }

  if (recallRequest) {
    return (
      <div className="recall-panel recall-status">
        <h3><UndoIcon /> Recall Status</h3>
        <div className={`overall-status ${recallRequest.status}`}>
          {recallRequest.status === 'success' && 'Email successfully recalled'}
          {recallRequest.status === 'partial_success' && 'Email partially recalled'}
          {recallRequest.status === 'failed' && 'Recall failed'}
          {recallRequest.status === 'pending' && 'Recall in progress...'}
        </div>
        <ul className="recipient-statuses">
          {recallRequest.recipientStatuses.map(rs => (
            <li key={rs.email} className={`status-${rs.status}`}>
              <span className="email">{rs.email}</span>
              <span className="status">
                {rs.status === 'recalled' && <CheckIcon />}
                {rs.status === 'already_read' && <EyeIcon />}
                {rs.status === 'not_supported' && <AlertIcon />}
                {rs.status === 'failed' && <XIcon />}
                {rs.status === 'pending' && <LoadingIcon />}
              </span>
              {rs.reason && <span className="reason">{rs.reason}</span>}
            </li>
          ))}
        </ul>
      </div>
    );
  }

  return (
    <div className="recall-panel">
      <h3><UndoIcon /> Recall Email</h3>
      <div className="email-info">
        <p className="subject">{subject}</p>
        <p className="sent">Sent {formatTime(sentAt)}</p>
      </div>

      {recallStatus?.canRecall ? (
        <>
          {recallStatus.unsupportedRecipients.length > 0 && (
            <div className="warning">
              <AlertIcon />
              <p>
                Some recipients don't support recall:
                {recallStatus.unsupportedRecipients.join(', ')}
              </p>
            </div>
          )}
          <button
            className="recall-button primary"
            onClick={handleRecall}
            disabled={recalling}
          >
            {recalling ? 'Recalling...' : 'Recall Email'}
          </button>
        </>
      ) : (
        <div className="cannot-recall">
          <XCircleIcon />
          <p>{recallStatus?.reason || 'This email cannot be recalled'}</p>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Tracking Detection Display
// ============================================================================

interface TrackingDetectionDisplayProps {
  detection: TrackingDetection;
  onBlockAll: () => void;
  onViewDetails: () => void;
}

export function TrackingDetectionDisplay({
  detection,
  onBlockAll,
  onViewDetails,
}: TrackingDetectionDisplayProps) {
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'high': return 'var(--color-danger)';
      case 'medium': return 'var(--color-warning)';
      case 'low': return 'var(--color-info)';
      default: return 'var(--color-success)';
    }
  };

  if (detection.totalTrackers === 0) {
    return (
      <div className="tracking-detection clean">
        <ShieldCheckIcon />
        <span>No tracking detected</span>
      </div>
    );
  }

  return (
    <div className="tracking-detection detected" style={{ borderColor: getRiskColor(detection.riskLevel) }}>
      <div className="detection-header">
        <ShieldAlertIcon />
        <span className="count">{detection.totalTrackers} tracker{detection.totalTrackers > 1 ? 's' : ''} detected</span>
        <span className={`risk-badge ${detection.riskLevel}`}>{detection.riskLevel} risk</span>
      </div>

      <div className="tracker-summary">
        {detection.trackingPixels.length > 0 && (
          <span className="pixel-count">
            <ImageIcon /> {detection.trackingPixels.length} tracking pixel{detection.trackingPixels.length > 1 ? 's' : ''}
          </span>
        )}
        {detection.trackingLinks.length > 0 && (
          <span className="link-count">
            <LinkIcon /> {detection.trackingLinks.length} tracking link{detection.trackingLinks.length > 1 ? 's' : ''}
          </span>
        )}
      </div>

      <div className="detection-actions">
        <button onClick={onViewDetails}>View Details</button>
        <button className="primary" onClick={onBlockAll}>Block All</button>
      </div>
    </div>
  );
}

interface TrackingDetailsModalProps {
  detection: TrackingDetection;
  onClose: () => void;
  onBlockTracker: (url: string) => void;
}

export function TrackingDetailsModal({ detection, onClose, onBlockTracker }: TrackingDetailsModalProps) {
  const allTrackers = [...detection.trackingPixels, ...detection.trackingLinks];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal tracking-details-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Tracking Details</h2>
          <button className="close-button" onClick={onClose}><CloseIcon /></button>
        </div>

        <div className="modal-body">
          {allTrackers.map((tracker, index) => (
            <div key={index} className={`tracker-item ${tracker.blocked ? 'blocked' : ''}`}>
              <div className="tracker-type">
                {tracker.type === 'pixel' && <ImageIcon />}
                {tracker.type === 'link' && <LinkIcon />}
                {tracker.type === 'beacon' && <RadioIcon />}
              </div>
              <div className="tracker-info">
                <span className="domain">{tracker.domain}</span>
                {tracker.company && <span className="company">by {tracker.company}</span>}
                <span className="purpose">{tracker.purpose}</span>
                <span className="url">{tracker.url}</span>
              </div>
              <div className="tracker-action">
                {tracker.blocked ? (
                  <span className="blocked-badge"><CheckIcon /> Blocked</span>
                ) : (
                  <button onClick={() => onBlockTracker(tracker.url)}>Block</button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Read Receipt Settings
// ============================================================================

interface ReadReceiptSettingsProps {
  settings: ReadReceiptSettings;
  onChange: (settings: ReadReceiptSettings) => void;
}

export function ReadReceiptSettingsPanel({ settings, onChange }: ReadReceiptSettingsProps) {
  const [newWhitelist, setNewWhitelist] = useState('');
  const [newBlacklist, setNewBlacklist] = useState('');

  const addToWhitelist = () => {
    if (newWhitelist) {
      onChange({
        ...settings,
        whitelist: [...settings.whitelist, newWhitelist],
      });
      setNewWhitelist('');
    }
  };

  const addToBlacklist = () => {
    if (newBlacklist) {
      onChange({
        ...settings,
        blacklist: [...settings.blacklist, newBlacklist],
      });
      setNewBlacklist('');
    }
  };

  const removeFromWhitelist = (item: string) => {
    onChange({
      ...settings,
      whitelist: settings.whitelist.filter(w => w !== item),
    });
  };

  const removeFromBlacklist = (item: string) => {
    onChange({
      ...settings,
      blacklist: settings.blacklist.filter(b => b !== item),
    });
  };

  return (
    <div className="read-receipt-settings">
      <h3>Read Receipt Settings</h3>

      <div className="setting-group">
        <label>
          <input
            type="checkbox"
            checked={settings.autoDenyMdn}
            onChange={e => onChange({ ...settings, autoDenyMdn: e.target.checked })}
          />
          Automatically deny read receipt requests
        </label>
        <p className="help-text">
          Senders will not be notified when you open their emails
        </p>
      </div>

      <div className="setting-group">
        <label>
          <input
            type="checkbox"
            checked={settings.blockOutgoing}
            onChange={e => onChange({ ...settings, blockOutgoing: e.target.checked })}
          />
          Don't request read receipts on my sent emails
        </label>
      </div>

      <div className="setting-group">
        <label>
          <input
            type="checkbox"
            checked={settings.blockIncomingRequests}
            onChange={e => onChange({ ...settings, blockIncomingRequests: e.target.checked })}
          />
          Hide read receipt requests in incoming emails
        </label>
      </div>

      <div className="list-section">
        <h4>Whitelist (always send receipts to)</h4>
        <div className="add-form">
          <input
            type="text"
            value={newWhitelist}
            onChange={e => setNewWhitelist(e.target.value)}
            placeholder="email@example.com or @domain.com"
          />
          <button onClick={addToWhitelist}>Add</button>
        </div>
        <ul className="item-list">
          {settings.whitelist.map(item => (
            <li key={item}>
              <span>{item}</span>
              <button onClick={() => removeFromWhitelist(item)}><XIcon /></button>
            </li>
          ))}
        </ul>
      </div>

      <div className="list-section">
        <h4>Blacklist (never send receipts to)</h4>
        <div className="add-form">
          <input
            type="text"
            value={newBlacklist}
            onChange={e => setNewBlacklist(e.target.value)}
            placeholder="email@example.com or @domain.com"
          />
          <button onClick={addToBlacklist}>Add</button>
        </div>
        <ul className="item-list">
          {settings.blacklist.map(item => (
            <li key={item}>
              <span>{item}</span>
              <button onClick={() => removeFromBlacklist(item)}><XIcon /></button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

// ============================================================================
// Anonymous Identity Manager
// ============================================================================

interface AnonymousIdentityManagerProps {
  identities: AnonymousIdentity[];
  onCreate: (displayName?: string, expiresDays?: number) => void;
  onDeactivate: (identityId: string) => void;
}

export function AnonymousIdentityManager({
  identities,
  onCreate,
  onDeactivate,
}: AnonymousIdentityManagerProps) {
  const [showCreate, setShowCreate] = useState(false);
  const [displayName, setDisplayName] = useState('');
  const [expiresDays, setExpiresDays] = useState<number | undefined>(undefined);

  const handleCreate = () => {
    onCreate(displayName || undefined, expiresDays);
    setDisplayName('');
    setExpiresDays(undefined);
    setShowCreate(false);
  };

  return (
    <div className="anonymous-identity-manager">
      <div className="section-header">
        <h3><MaskIcon /> Anonymous Identities</h3>
        <button className="primary" onClick={() => setShowCreate(true)}>
          <PlusIcon /> Create Identity
        </button>
      </div>

      <p className="description">
        Send emails anonymously through relay addresses. Recipients won't see your real email.
      </p>

      {showCreate && (
        <div className="create-identity-form">
          <div className="form-group">
            <label>Display Name (optional)</label>
            <input
              type="text"
              value={displayName}
              onChange={e => setDisplayName(e.target.value)}
              placeholder="Anonymous"
            />
          </div>
          <div className="form-group">
            <label>Expires after (optional)</label>
            <select
              value={expiresDays || ''}
              onChange={e => setExpiresDays(e.target.value ? parseInt(e.target.value) : undefined)}
            >
              <option value="">Never</option>
              <option value="1">1 day</option>
              <option value="7">1 week</option>
              <option value="30">1 month</option>
              <option value="90">3 months</option>
            </select>
          </div>
          <div className="form-actions">
            <button className="secondary" onClick={() => setShowCreate(false)}>Cancel</button>
            <button className="primary" onClick={handleCreate}>Create</button>
          </div>
        </div>
      )}

      <div className="identities-list">
        {identities.map(identity => (
          <div key={identity.id} className={`identity-item ${identity.isActive ? '' : 'inactive'}`}>
            <div className="identity-info">
              <span className="alias-email">{identity.aliasEmail}</span>
              {identity.displayName && (
                <span className="display-name">"{identity.displayName}"</span>
              )}
              <div className="identity-meta">
                <span>Created {formatDate(identity.createdAt)}</span>
                {identity.expiresAt && (
                  <span>Expires {formatDate(identity.expiresAt)}</span>
                )}
                <span>{identity.emailCount} emails sent</span>
              </div>
            </div>
            <div className="identity-status">
              {identity.isActive ? (
                <span className="status active">Active</span>
              ) : (
                <span className="status inactive">Inactive</span>
              )}
            </div>
            <div className="identity-actions">
              <button
                className="copy"
                onClick={() => navigator.clipboard.writeText(identity.aliasEmail)}
                title="Copy email"
              >
                <CopyIcon />
              </button>
              {identity.isActive && (
                <button
                  className="deactivate"
                  onClick={() => onDeactivate(identity.id)}
                >
                  Deactivate
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Link Proxy Settings
// ============================================================================

interface LinkProxySettingsProps {
  settings: LinkProxySettings;
  onChange: (settings: LinkProxySettings) => void;
}

export function LinkProxySettingsPanel({ settings, onChange }: LinkProxySettingsProps) {
  const [newDomain, setNewDomain] = useState('');

  const addWhitelistDomain = () => {
    if (newDomain) {
      onChange({
        ...settings,
        whitelistDomains: [...settings.whitelistDomains, newDomain],
      });
      setNewDomain('');
    }
  };

  const removeDomain = (domain: string) => {
    onChange({
      ...settings,
      whitelistDomains: settings.whitelistDomains.filter(d => d !== domain),
    });
  };

  return (
    <div className="link-proxy-settings">
      <h3><ShieldIcon /> Link Privacy</h3>

      <div className="setting-group">
        <label>
          <input
            type="checkbox"
            checked={settings.enabled}
            onChange={e => onChange({ ...settings, enabled: e.target.checked })}
          />
          Enable link privacy protection
        </label>
        <p className="help-text">
          Masks your clicks from senders by routing links through our privacy proxy
        </p>
      </div>

      {settings.enabled && (
        <>
          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={settings.stripUtmParams}
                onChange={e => onChange({ ...settings, stripUtmParams: e.target.checked })}
              />
              Strip tracking parameters (utm_*, fbclid, etc.)
            </label>
          </div>

          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={settings.blockKnownTrackers}
                onChange={e => onChange({ ...settings, blockKnownTrackers: e.target.checked })}
              />
              Block known tracking domains
            </label>
          </div>

          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={settings.warnOnRedirect}
                onChange={e => onChange({ ...settings, warnOnRedirect: e.target.checked })}
              />
              Warn before following redirects
            </label>
          </div>

          <div className="whitelist-section">
            <h4>Trusted Domains (bypass proxy)</h4>
            <div className="add-form">
              <input
                type="text"
                value={newDomain}
                onChange={e => setNewDomain(e.target.value)}
                placeholder="example.com"
              />
              <button onClick={addWhitelistDomain}>Add</button>
            </div>
            <ul className="domain-list">
              {settings.whitelistDomains.map(domain => (
                <li key={domain}>
                  <span>{domain}</span>
                  <button onClick={() => removeDomain(domain)}><XIcon /></button>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  );
}

// ============================================================================
// Privacy Center Main Component
// ============================================================================

export default function PrivacyCenter() {
  const [activeSection, setActiveSection] = useState<'tracking' | 'receipts' | 'anonymous' | 'links'>('tracking');
  const [readReceiptSettings, setReadReceiptSettings] = useState<ReadReceiptSettings>({
    blockOutgoing: false,
    blockIncomingRequests: true,
    autoDenyMdn: true,
    whitelist: [],
    blacklist: [],
  });
  const [linkProxySettings, setLinkProxySettings] = useState<LinkProxySettings>({
    enabled: true,
    stripUtmParams: true,
    blockKnownTrackers: true,
    warnOnRedirect: true,
    whitelistDomains: [],
  });
  const [identities, setIdentities] = useState<AnonymousIdentity[]>([]);

  useEffect(() => {
    fetchSettings();
    fetchIdentities();
  }, []);

  const fetchSettings = async () => {
    // Fetch privacy settings
  };

  const fetchIdentities = async () => {
    // Fetch anonymous identities
  };

  const handleSaveReadReceipts = async (settings: ReadReceiptSettings) => {
    setReadReceiptSettings(settings);
    await fetch('/api/privacy/read-receipts', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
  };

  const handleSaveLinkProxy = async (settings: LinkProxySettings) => {
    setLinkProxySettings(settings);
    await fetch('/api/privacy/link-proxy', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
  };

  const handleCreateIdentity = async (displayName?: string, expiresDays?: number) => {
    const response = await fetch('/api/privacy/anonymous-identities', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ displayName, expiresDays }),
    });
    const newIdentity = await response.json();
    setIdentities([newIdentity, ...identities]);
  };

  const handleDeactivateIdentity = async (identityId: string) => {
    await fetch(`/api/privacy/anonymous-identities/${identityId}/deactivate`, {
      method: 'POST',
    });
    setIdentities(identities.map(i =>
      i.id === identityId ? { ...i, isActive: false } : i
    ));
  };

  return (
    <div className="privacy-center">
      <h1><ShieldIcon /> Privacy Center</h1>

      <nav className="privacy-nav">
        <button
          className={activeSection === 'tracking' ? 'active' : ''}
          onClick={() => setActiveSection('tracking')}
        >
          <ShieldAlertIcon /> Tracking Protection
        </button>
        <button
          className={activeSection === 'receipts' ? 'active' : ''}
          onClick={() => setActiveSection('receipts')}
        >
          <MailOpenIcon /> Read Receipts
        </button>
        <button
          className={activeSection === 'anonymous' ? 'active' : ''}
          onClick={() => setActiveSection('anonymous')}
        >
          <MaskIcon /> Anonymous Mode
        </button>
        <button
          className={activeSection === 'links' ? 'active' : ''}
          onClick={() => setActiveSection('links')}
        >
          <LinkIcon /> Link Privacy
        </button>
      </nav>

      <div className="privacy-content">
        {activeSection === 'tracking' && (
          <div className="tracking-section">
            <h2>Tracking Protection</h2>
            <p>
              Mycelix Mail automatically detects and blocks tracking pixels and links
              in your incoming emails.
            </p>
            <div className="stats-cards">
              <div className="stat-card">
                <span className="value">1,247</span>
                <span className="label">Trackers blocked this month</span>
              </div>
              <div className="stat-card">
                <span className="value">89%</span>
                <span className="label">Emails with trackers</span>
              </div>
            </div>
          </div>
        )}

        {activeSection === 'receipts' && (
          <ReadReceiptSettingsPanel
            settings={readReceiptSettings}
            onChange={handleSaveReadReceipts}
          />
        )}

        {activeSection === 'anonymous' && (
          <AnonymousIdentityManager
            identities={identities}
            onCreate={handleCreateIdentity}
            onDeactivate={handleDeactivateIdentity}
          />
        )}

        {activeSection === 'links' && (
          <LinkProxySettingsPanel
            settings={linkProxySettings}
            onChange={handleSaveLinkProxy}
          />
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Helper Functions
// ============================================================================

function formatTime(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
  return date.toLocaleDateString();
}

function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleDateString();
}

// ============================================================================
// Icon Components
// ============================================================================

function TimerIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </svg>;
}

function ClockIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </svg>;
}

function EyeIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
    <circle cx="12" cy="12" r="3" />
  </svg>;
}

function CheckCircleIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
    <polyline points="22 4 12 14.01 9 11.01" />
  </svg>;
}

function UndoIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M3 7v6h6" />
    <path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13" />
  </svg>;
}

function CheckIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="20 6 9 17 4 12" />
  </svg>;
}

function XIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>;
}

function AlertIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="8" x2="12" y2="12" />
    <line x1="12" y1="16" x2="12.01" y2="16" />
  </svg>;
}

function LoadingIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="12" y1="2" x2="12" y2="6" />
    <line x1="12" y1="18" x2="12" y2="22" />
    <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
    <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
  </svg>;
}

function XCircleIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <line x1="15" y1="9" x2="9" y2="15" />
    <line x1="9" y1="9" x2="15" y2="15" />
  </svg>;
}

function ShieldCheckIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    <polyline points="9 12 12 15 16 10" />
  </svg>;
}

function ShieldAlertIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    <line x1="12" y1="8" x2="12" y2="12" />
    <line x1="12" y1="16" x2="12.01" y2="16" />
  </svg>;
}

function ShieldIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
  </svg>;
}

function ImageIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
    <circle cx="8.5" cy="8.5" r="1.5" />
    <polyline points="21 15 16 10 5 21" />
  </svg>;
}

function LinkIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
    <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
  </svg>;
}

function RadioIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="2" />
    <path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14" />
  </svg>;
}

function CloseIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>;
}

function MaskIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
  </svg>;
}

function PlusIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>;
}

function CopyIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>;
}

function MailOpenIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21.2 8.4c.5.38.8.97.8 1.6v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V10a2 2 0 0 1 .8-1.6l8-6a2 2 0 0 1 2.4 0l8 6Z" />
    <path d="m22 10-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 10" />
  </svg>;
}
