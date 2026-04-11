// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Track AD: Data Sovereignty
 *
 * Privacy dashboard, data export, deletion requests, consent management,
 * audit logs, and full user control over personal data.
 */

import React, { useState, useEffect, useCallback } from 'react';

// ============================================================================
// Types
// ============================================================================

interface PrivacyDashboard {
  dataSummary: DataSummary;
  consentStatus: ConsentStatus[];
  pendingRequests: DataSubjectRequest[];
  exports: DataExport[];
  retentionInfo: RetentionInfo;
  thirdPartyAccess: ThirdPartyAccess[];
}

interface DataSummary {
  totalEmails: number;
  totalContacts: number;
  totalEvents: number;
  totalAttachments: number;
  storageUsedBytes: number;
  accountCreatedAt: string;
  lastActivity: string;
}

interface ConsentStatus {
  consentType: ConsentType;
  granted: boolean;
  grantedAt?: string;
  required: boolean;
}

type ConsentType =
  | 'TermsOfService'
  | 'PrivacyPolicy'
  | 'DataProcessing'
  | 'Marketing'
  | 'Analytics'
  | 'ThirdPartySharing'
  | 'CrossBorderTransfer'
  | 'AiProcessing';

interface DataSubjectRequest {
  id: string;
  requestType: 'Access' | 'Rectification' | 'Erasure' | 'Restriction' | 'Portability' | 'Objection';
  status: 'Pending' | 'Processing' | 'Completed' | 'Rejected';
  submittedAt: string;
  completedAt?: string;
}

interface DataExport {
  id: string;
  format: 'Json' | 'Mbox' | 'Eml' | 'Csv' | 'Pdf';
  status: 'Queued' | 'Processing' | 'Ready' | 'Expired';
  createdAt: string;
  expiresAt?: string;
  downloadUrl?: string;
  fileSizeBytes?: number;
}

interface RetentionInfo {
  policies: RetentionPolicy[];
  nextCleanup?: string;
  itemsPendingDeletion: number;
}

interface RetentionPolicy {
  dataType: string;
  retentionDays: number;
  itemsAffected: number;
}

interface ThirdPartyAccess {
  serviceName: string;
  permissions: string[];
  authorizedAt: string;
  lastAccessed?: string;
  canRevoke: boolean;
}

interface AuditLogEntry {
  id: string;
  timestamp: string;
  action: string;
  resourceType: string;
  resourceId?: string;
  ipAddress?: string;
  details: string;
  success: boolean;
}

interface ExportScope {
  emails: boolean;
  contacts: boolean;
  calendar: boolean;
  settings: boolean;
  trustData: boolean;
  activityLogs: boolean;
  attachments: boolean;
}

// ============================================================================
// Hooks
// ============================================================================

function usePrivacyDashboard() {
  const [dashboard, setDashboard] = useState<PrivacyDashboard | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/sovereignty/dashboard')
      .then(res => res.json())
      .then(setDashboard)
      .finally(() => setLoading(false));
  }, []);

  const refresh = useCallback(() => {
    setLoading(true);
    fetch('/api/sovereignty/dashboard')
      .then(res => res.json())
      .then(setDashboard)
      .finally(() => setLoading(false));
  }, []);

  return { dashboard, loading, refresh };
}

function useConsent() {
  const [consents, setConsents] = useState<ConsentStatus[]>([]);

  useEffect(() => {
    fetch('/api/sovereignty/consent')
      .then(res => res.json())
      .then(setConsents);
  }, []);

  const updateConsent = useCallback(async (consentType: ConsentType, granted: boolean) => {
    const endpoint = granted
      ? '/api/sovereignty/consent/grant'
      : '/api/sovereignty/consent/revoke';

    await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ consentType }),
    });

    setConsents(prev => prev.map(c =>
      c.consentType === consentType
        ? { ...c, granted, grantedAt: granted ? new Date().toISOString() : undefined }
        : c
    ));
  }, []);

  return { consents, updateConsent };
}

function useDataExport() {
  const [exports, setExports] = useState<DataExport[]>([]);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    fetch('/api/sovereignty/exports')
      .then(res => res.json())
      .then(setExports);
  }, []);

  const createExport = useCallback(async (format: DataExport['format'], scope: ExportScope) => {
    setCreating(true);
    try {
      const response = await fetch('/api/sovereignty/exports', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format, scope }),
      });
      const newExport = await response.json();
      setExports(prev => [newExport, ...prev]);
      return newExport;
    } finally {
      setCreating(false);
    }
  }, []);

  return { exports, creating, createExport };
}

function useDataDeletion() {
  const [requests, setRequests] = useState<DataSubjectRequest[]>([]);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    fetch('/api/sovereignty/deletion-requests')
      .then(res => res.json())
      .then(setRequests);
  }, []);

  const requestDeletion = useCallback(async (scope: 'account' | 'emails' | 'contacts') => {
    setCreating(true);
    try {
      const response = await fetch('/api/sovereignty/deletion-requests', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scope }),
      });
      const request = await response.json();
      setRequests(prev => [request, ...prev]);
      return request;
    } finally {
      setCreating(false);
    }
  }, []);

  const cancelDeletion = useCallback(async (requestId: string) => {
    await fetch(`/api/sovereignty/deletion-requests/${requestId}/cancel`, { method: 'POST' });
    setRequests(prev => prev.filter(r => r.id !== requestId));
  }, []);

  return { requests, creating, requestDeletion, cancelDeletion };
}

function useAuditLog() {
  const [entries, setEntries] = useState<AuditLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [hasMore, setHasMore] = useState(true);

  const loadMore = useCallback(async (offset: number = 0) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/sovereignty/audit-log?offset=${offset}&limit=50`);
      const data = await response.json();
      if (offset === 0) {
        setEntries(data.entries);
      } else {
        setEntries(prev => [...prev, ...data.entries]);
      }
      setHasMore(data.hasMore);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadMore(0);
  }, [loadMore]);

  return { entries, loading, hasMore, loadMore };
}

function useThirdPartyAccess() {
  const [access, setAccess] = useState<ThirdPartyAccess[]>([]);

  useEffect(() => {
    fetch('/api/sovereignty/third-party-access')
      .then(res => res.json())
      .then(setAccess);
  }, []);

  const revoke = useCallback(async (serviceName: string) => {
    await fetch(`/api/sovereignty/third-party-access/${encodeURIComponent(serviceName)}/revoke`, {
      method: 'POST',
    });
    setAccess(prev => prev.filter(a => a.serviceName !== serviceName));
  }, []);

  return { access, revoke };
}

// ============================================================================
// Components
// ============================================================================

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function DataSummaryPanel({ summary }: { summary: DataSummary }) {
  return (
    <div className="data-summary-panel">
      <h3>Your Data</h3>

      <div className="stats-grid">
        <div className="stat">
          <span className="value">{summary.totalEmails.toLocaleString()}</span>
          <span className="label">Emails</span>
        </div>
        <div className="stat">
          <span className="value">{summary.totalContacts.toLocaleString()}</span>
          <span className="label">Contacts</span>
        </div>
        <div className="stat">
          <span className="value">{summary.totalEvents.toLocaleString()}</span>
          <span className="label">Calendar Events</span>
        </div>
        <div className="stat">
          <span className="value">{summary.totalAttachments.toLocaleString()}</span>
          <span className="label">Attachments</span>
        </div>
      </div>

      <div className="storage-usage">
        <span className="label">Storage Used:</span>
        <span className="value">{formatBytes(summary.storageUsedBytes)}</span>
      </div>

      <div className="account-info">
        <div className="info-item">
          <span className="label">Account Created:</span>
          <span className="value">{new Date(summary.accountCreatedAt).toLocaleDateString()}</span>
        </div>
        <div className="info-item">
          <span className="label">Last Activity:</span>
          <span className="value">{new Date(summary.lastActivity).toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
}

function ConsentPanel() {
  const { consents, updateConsent } = useConsent();

  const consentLabels: Record<ConsentType, { label: string; description: string }> = {
    TermsOfService: {
      label: 'Terms of Service',
      description: 'Required to use the service',
    },
    PrivacyPolicy: {
      label: 'Privacy Policy',
      description: 'Required to use the service',
    },
    DataProcessing: {
      label: 'Data Processing',
      description: 'Allow processing of your email data for core functionality',
    },
    Marketing: {
      label: 'Marketing Communications',
      description: 'Receive product updates and promotional emails',
    },
    Analytics: {
      label: 'Analytics',
      description: 'Help improve the product with anonymized usage data',
    },
    ThirdPartySharing: {
      label: 'Third-Party Sharing',
      description: 'Allow sharing data with integrated services',
    },
    CrossBorderTransfer: {
      label: 'Cross-Border Transfer',
      description: 'Allow data transfer outside your region',
    },
    AiProcessing: {
      label: 'AI Processing',
      description: 'Use AI features for email summarization and suggestions',
    },
  };

  return (
    <div className="consent-panel">
      <h3>Consent Preferences</h3>
      <p className="description">
        Control how your data is used. Required consents cannot be revoked while using the service.
      </p>

      <div className="consent-list">
        {consents.map(consent => {
          const info = consentLabels[consent.consentType];
          return (
            <div key={consent.consentType} className={`consent-item ${consent.required ? 'required' : ''}`}>
              <div className="consent-info">
                <span className="label">{info?.label || consent.consentType}</span>
                <span className="description">{info?.description}</span>
                {consent.grantedAt && (
                  <span className="granted-at">
                    Granted: {new Date(consent.grantedAt).toLocaleDateString()}
                  </span>
                )}
              </div>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={consent.granted}
                  disabled={consent.required}
                  onChange={e => updateConsent(consent.consentType, e.target.checked)}
                />
                <span className="slider" />
              </label>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DataExportPanel() {
  const { exports, creating, createExport } = useDataExport();
  const [showCreate, setShowCreate] = useState(false);
  const [format, setFormat] = useState<DataExport['format']>('Json');
  const [scope, setScope] = useState<ExportScope>({
    emails: true,
    contacts: true,
    calendar: true,
    settings: true,
    trustData: true,
    activityLogs: false,
    attachments: true,
  });

  const handleCreate = async () => {
    await createExport(format, scope);
    setShowCreate(false);
  };

  const statusColors = {
    Queued: '#6b7280',
    Processing: '#eab308',
    Ready: '#22c55e',
    Expired: '#ef4444',
  };

  return (
    <div className="data-export-panel">
      <header>
        <h3>Data Export</h3>
        <button onClick={() => setShowCreate(true)}>Request Export</button>
      </header>

      <p className="description">
        Download a copy of your data. Exports are available for 7 days after generation.
      </p>

      {exports.length === 0 ? (
        <div className="empty-state">No exports yet</div>
      ) : (
        <div className="export-list">
          {exports.map(exp => (
            <div key={exp.id} className="export-item">
              <div className="export-info">
                <span className="format">{exp.format}</span>
                <span className="date">{new Date(exp.createdAt).toLocaleDateString()}</span>
                {exp.fileSizeBytes && (
                  <span className="size">{formatBytes(exp.fileSizeBytes)}</span>
                )}
              </div>
              <span className="status" style={{ color: statusColors[exp.status] }}>
                {exp.status}
              </span>
              {exp.status === 'Ready' && exp.downloadUrl && (
                <a href={exp.downloadUrl} className="download-btn">Download</a>
              )}
            </div>
          ))}
        </div>
      )}

      {showCreate && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Request Data Export</h3>

            <div className="form-group">
              <label>Format</label>
              <select value={format} onChange={e => setFormat(e.target.value as DataExport['format'])}>
                <option value="Json">JSON (machine-readable)</option>
                <option value="Mbox">MBOX (email standard)</option>
                <option value="Eml">EML files (zipped)</option>
                <option value="Csv">CSV (spreadsheet)</option>
                <option value="Pdf">PDF (human-readable report)</option>
              </select>
            </div>

            <div className="form-group">
              <label>Include</label>
              <div className="scope-options">
                {Object.entries(scope).map(([key, value]) => (
                  <label key={key} className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={value}
                      onChange={e => setScope(prev => ({ ...prev, [key]: e.target.checked }))}
                    />
                    {key.replace(/([A-Z])/g, ' $1').trim()}
                  </label>
                ))}
              </div>
            </div>

            <div className="modal-actions">
              <button onClick={() => setShowCreate(false)}>Cancel</button>
              <button onClick={handleCreate} disabled={creating} className="primary">
                {creating ? 'Creating...' : 'Create Export'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function DataDeletionPanel() {
  const { requests, creating, requestDeletion, cancelDeletion } = useDataDeletion();
  const [showConfirm, setShowConfirm] = useState<'account' | 'emails' | 'contacts' | null>(null);

  const handleDelete = async () => {
    if (showConfirm) {
      await requestDeletion(showConfirm);
      setShowConfirm(null);
    }
  };

  const statusColors = {
    Pending: '#6b7280',
    Processing: '#eab308',
    Completed: '#22c55e',
    Rejected: '#ef4444',
  };

  return (
    <div className="data-deletion-panel">
      <h3>Right to Erasure</h3>

      <p className="description">
        Request deletion of your data. This action may be irreversible.
      </p>

      <div className="deletion-options">
        <div className="option">
          <div className="option-info">
            <span className="label">Delete All Emails</span>
            <span className="description">Remove all emails but keep your account</span>
          </div>
          <button onClick={() => setShowConfirm('emails')} className="danger">
            Request Deletion
          </button>
        </div>

        <div className="option">
          <div className="option-info">
            <span className="label">Delete All Contacts</span>
            <span className="description">Remove all contacts and trust data</span>
          </div>
          <button onClick={() => setShowConfirm('contacts')} className="danger">
            Request Deletion
          </button>
        </div>

        <div className="option warning">
          <div className="option-info">
            <span className="label">Delete Entire Account</span>
            <span className="description">Permanently delete your account and all data</span>
          </div>
          <button onClick={() => setShowConfirm('account')} className="danger">
            Delete Account
          </button>
        </div>
      </div>

      {requests.length > 0 && (
        <div className="deletion-requests">
          <h4>Pending Requests</h4>
          {requests.map(request => (
            <div key={request.id} className="request-item">
              <span className="type">{request.requestType}</span>
              <span className="date">{new Date(request.submittedAt).toLocaleDateString()}</span>
              <span className="status" style={{ color: statusColors[request.status] }}>
                {request.status}
              </span>
              {request.status === 'Pending' && (
                <button onClick={() => cancelDeletion(request.id)}>Cancel</button>
              )}
            </div>
          ))}
        </div>
      )}

      {showConfirm && (
        <div className="modal-overlay">
          <div className="modal danger-modal">
            <h3>Confirm Deletion</h3>
            <p>
              {showConfirm === 'account' && 'This will permanently delete your account and all associated data. This action cannot be undone.'}
              {showConfirm === 'emails' && 'This will permanently delete all your emails. This action cannot be undone.'}
              {showConfirm === 'contacts' && 'This will permanently delete all your contacts and trust data. This action cannot be undone.'}
            </p>
            <div className="modal-actions">
              <button onClick={() => setShowConfirm(null)}>Cancel</button>
              <button onClick={handleDelete} disabled={creating} className="danger">
                {creating ? 'Processing...' : 'Confirm Deletion'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ThirdPartyAccessPanel() {
  const { access, revoke } = useThirdPartyAccess();

  return (
    <div className="third-party-panel">
      <h3>Third-Party Access</h3>

      <p className="description">
        Services and applications that have access to your data.
      </p>

      {access.length === 0 ? (
        <div className="empty-state">No third-party access granted</div>
      ) : (
        <div className="access-list">
          {access.map(app => (
            <div key={app.serviceName} className="access-item">
              <div className="access-info">
                <span className="service-name">{app.serviceName}</span>
                <div className="permissions">
                  {app.permissions.map(perm => (
                    <span key={perm} className="permission">{perm}</span>
                  ))}
                </div>
                <span className="dates">
                  Authorized: {new Date(app.authorizedAt).toLocaleDateString()}
                  {app.lastAccessed && ` • Last used: ${new Date(app.lastAccessed).toLocaleDateString()}`}
                </span>
              </div>
              {app.canRevoke && (
                <button onClick={() => revoke(app.serviceName)} className="danger">
                  Revoke
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function AuditLogPanel() {
  const { entries, loading, hasMore, loadMore } = useAuditLog();

  const actionIcons: Record<string, string> = {
    Login: '🔓',
    Logout: '🔒',
    EmailRead: '📖',
    EmailSent: '📤',
    EmailDeleted: '🗑️',
    SettingsChanged: '⚙️',
    DataExported: '📦',
  };

  return (
    <div className="audit-log-panel">
      <h3>Activity Log</h3>

      <p className="description">
        A record of actions taken on your account.
      </p>

      <div className="log-list">
        {entries.map(entry => (
          <div key={entry.id} className={`log-entry ${entry.success ? 'success' : 'failure'}`}>
            <span className="icon">{actionIcons[entry.action] || '📋'}</span>
            <div className="entry-content">
              <span className="action">{entry.action}</span>
              <span className="details">{entry.details}</span>
              <span className="meta">
                {new Date(entry.timestamp).toLocaleString()}
                {entry.ipAddress && ` • ${entry.ipAddress}`}
              </span>
            </div>
            <span className={`status ${entry.success ? 'success' : 'failure'}`}>
              {entry.success ? '✓' : '✗'}
            </span>
          </div>
        ))}
      </div>

      {hasMore && (
        <button
          onClick={() => loadMore(entries.length)}
          disabled={loading}
          className="load-more"
        >
          {loading ? 'Loading...' : 'Load More'}
        </button>
      )}
    </div>
  );
}

function RetentionPanel({ info }: { info: RetentionInfo }) {
  return (
    <div className="retention-panel">
      <h3>Data Retention</h3>

      <p className="description">
        How long your data is kept before automatic cleanup.
      </p>

      <div className="policies">
        {info.policies.map((policy, i) => (
          <div key={i} className="policy-item">
            <span className="data-type">{policy.dataType}</span>
            <span className="retention">{policy.retentionDays} days</span>
            <span className="affected">{policy.itemsAffected} items</span>
          </div>
        ))}
      </div>

      {info.nextCleanup && (
        <div className="next-cleanup">
          <span className="label">Next cleanup:</span>
          <span className="value">{new Date(info.nextCleanup).toLocaleDateString()}</span>
        </div>
      )}

      {info.itemsPendingDeletion > 0 && (
        <div className="pending">
          {info.itemsPendingDeletion} items scheduled for deletion
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

interface DataSovereigntyProps {
  initialTab?: string;
}

export function DataSovereignty({ initialTab = 'overview' }: DataSovereigntyProps) {
  const { dashboard, loading, refresh } = usePrivacyDashboard();
  const [activeTab, setActiveTab] = useState(initialTab);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'consent', label: 'Consent', icon: '✓' },
    { id: 'export', label: 'Export', icon: '📥' },
    { id: 'deletion', label: 'Deletion', icon: '🗑️' },
    { id: 'access', label: 'Access', icon: '🔗' },
    { id: 'audit', label: 'Audit Log', icon: '📋' },
  ];

  if (loading) {
    return <div className="data-sovereignty loading">Loading privacy dashboard...</div>;
  }

  if (!dashboard) {
    return <div className="data-sovereignty error">Failed to load privacy dashboard</div>;
  }

  return (
    <div className="data-sovereignty">
      <header className="page-header">
        <h1>Privacy & Data</h1>
        <p>You control your data. Export, delete, or manage access at any time.</p>
      </header>

      <nav className="sovereignty-tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={activeTab === tab.id ? 'active' : ''}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="icon">{tab.icon}</span>
            <span className="label">{tab.label}</span>
          </button>
        ))}
      </nav>

      <div className="sovereignty-content">
        {activeTab === 'overview' && (
          <div className="overview-grid">
            <DataSummaryPanel summary={dashboard.dataSummary} />
            <RetentionPanel info={dashboard.retentionInfo} />
          </div>
        )}

        {activeTab === 'consent' && <ConsentPanel />}
        {activeTab === 'export' && <DataExportPanel />}
        {activeTab === 'deletion' && <DataDeletionPanel />}
        {activeTab === 'access' && <ThirdPartyAccessPanel />}
        {activeTab === 'audit' && <AuditLogPanel />}
      </div>
    </div>
  );
}

export default DataSovereignty;
