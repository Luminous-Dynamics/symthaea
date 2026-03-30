// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Track X: Security Fortress
 *
 * Security dashboard, WebAuthn/passkey management, phishing protection,
 * encryption settings, audit logs, and breach monitoring.
 */

import React, { useState, useEffect, useCallback } from 'react';

// ============================================================================
// Types
// ============================================================================

interface SecurityScore {
  overall: number;
  authentication: number;
  encryption: number;
  phishing: number;
  privacy: number;
  recommendations: SecurityRecommendation[];
}

interface SecurityRecommendation {
  id: string;
  category: string;
  title: string;
  description: string;
  impact: 'Critical' | 'High' | 'Medium' | 'Low';
  action: string;
}

interface Passkey {
  id: string;
  name: string;
  createdAt: string;
  lastUsed?: string;
  deviceInfo: string;
  credentialId: string;
}

interface PhishingReport {
  id: string;
  emailId: string;
  subject: string;
  from: string;
  detectedAt: string;
  threatType: 'Phishing' | 'Malware' | 'Spam' | 'Impersonation' | 'BEC';
  confidence: number;
  indicators: string[];
  action: 'Quarantined' | 'Warned' | 'Blocked' | 'Reported';
}

interface PhishingStats {
  totalDetected: number;
  blockedThisMonth: number;
  falsePositives: number;
  protectionLevel: 'Maximum' | 'Balanced' | 'Minimal';
}

interface EncryptionSettings {
  atRestEnabled: boolean;
  inTransitEnabled: boolean;
  endToEndEnabled: boolean;
  pgpEnabled: boolean;
  smimeEnabled: boolean;
  defaultEncryption: 'None' | 'TLS' | 'PGP' | 'SMIME';
  autoEncryptContacts: boolean;
}

interface PGPKey {
  id: string;
  fingerprint: string;
  email: string;
  createdAt: string;
  expiresAt?: string;
  algorithm: string;
  keySize: number;
  isPrivate: boolean;
}

interface AuditLogEntry {
  id: string;
  timestamp: string;
  eventType: string;
  actor: string;
  target?: string;
  details: string;
  ipAddress: string;
  userAgent: string;
  success: boolean;
}

interface BreachAlert {
  id: string;
  source: string;
  detectedAt: string;
  affectedEmail: string;
  breachDate?: string;
  dataTypes: string[];
  severity: 'Critical' | 'High' | 'Medium' | 'Low';
  acknowledged: boolean;
  actionsTaken: string[];
}

interface ActiveSession {
  id: string;
  deviceName: string;
  browser: string;
  ipAddress: string;
  location?: string;
  createdAt: string;
  lastActive: string;
  isCurrent: boolean;
}

// ============================================================================
// Hooks
// ============================================================================

function useSecurityScore() {
  const [score, setScore] = useState<SecurityScore | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/security/score')
      .then(res => res.json())
      .then(setScore)
      .finally(() => setLoading(false));
  }, []);

  const refresh = useCallback(() => {
    setLoading(true);
    fetch('/api/security/score')
      .then(res => res.json())
      .then(setScore)
      .finally(() => setLoading(false));
  }, []);

  return { score, loading, refresh };
}

function usePasskeys() {
  const [passkeys, setPasskeys] = useState<Passkey[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/security/passkeys')
      .then(res => res.json())
      .then(setPasskeys)
      .finally(() => setLoading(false));
  }, []);

  const registerPasskey = useCallback(async (name: string) => {
    // Start WebAuthn registration
    const optionsResponse = await fetch('/api/security/passkeys/register/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    const options = await optionsResponse.json();

    // Create credential using browser API
    const credential = await navigator.credentials.create({
      publicKey: {
        ...options,
        challenge: Uint8Array.from(atob(options.challenge), c => c.charCodeAt(0)),
        user: {
          ...options.user,
          id: Uint8Array.from(atob(options.user.id), c => c.charCodeAt(0)),
        },
      },
    }) as PublicKeyCredential;

    // Send credential to server
    const attestation = credential.response as AuthenticatorAttestationResponse;
    const finishResponse = await fetch('/api/security/passkeys/register/finish', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: credential.id,
        rawId: btoa(String.fromCharCode(...new Uint8Array(credential.rawId))),
        response: {
          clientDataJSON: btoa(String.fromCharCode(...new Uint8Array(attestation.clientDataJSON))),
          attestationObject: btoa(String.fromCharCode(...new Uint8Array(attestation.attestationObject))),
        },
        type: credential.type,
      }),
    });

    const newPasskey = await finishResponse.json();
    setPasskeys(prev => [...prev, newPasskey]);
    return newPasskey;
  }, []);

  const deletePasskey = useCallback(async (id: string) => {
    await fetch(`/api/security/passkeys/${id}`, { method: 'DELETE' });
    setPasskeys(prev => prev.filter(p => p.id !== id));
  }, []);

  const renamePasskey = useCallback(async (id: string, name: string) => {
    await fetch(`/api/security/passkeys/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    setPasskeys(prev => prev.map(p => p.id === id ? { ...p, name } : p));
  }, []);

  return { passkeys, loading, registerPasskey, deletePasskey, renamePasskey };
}

function usePhishingProtection() {
  const [reports, setReports] = useState<PhishingReport[]>([]);
  const [stats, setStats] = useState<PhishingStats | null>(null);

  useEffect(() => {
    fetch('/api/security/phishing/reports')
      .then(res => res.json())
      .then(setReports);

    fetch('/api/security/phishing/stats')
      .then(res => res.json())
      .then(setStats);
  }, []);

  const reportPhishing = useCallback(async (emailId: string) => {
    const response = await fetch('/api/security/phishing/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ emailId }),
    });
    const report = await response.json();
    setReports(prev => [report, ...prev]);
    return report;
  }, []);

  const markSafe = useCallback(async (reportId: string) => {
    await fetch(`/api/security/phishing/reports/${reportId}/safe`, { method: 'POST' });
    setReports(prev => prev.filter(r => r.id !== reportId));
  }, []);

  const setProtectionLevel = useCallback(async (level: PhishingStats['protectionLevel']) => {
    await fetch('/api/security/phishing/protection-level', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ level }),
    });
    setStats(prev => prev ? { ...prev, protectionLevel: level } : null);
  }, []);

  return { reports, stats, reportPhishing, markSafe, setProtectionLevel };
}

function useEncryption() {
  const [settings, setSettings] = useState<EncryptionSettings | null>(null);
  const [pgpKeys, setPgpKeys] = useState<PGPKey[]>([]);

  useEffect(() => {
    fetch('/api/security/encryption/settings')
      .then(res => res.json())
      .then(setSettings);

    fetch('/api/security/encryption/pgp-keys')
      .then(res => res.json())
      .then(setPgpKeys);
  }, []);

  const updateSettings = useCallback(async (updates: Partial<EncryptionSettings>) => {
    await fetch('/api/security/encryption/settings', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    setSettings(prev => prev ? { ...prev, ...updates } : null);
  }, []);

  const generatePGPKey = useCallback(async (email: string, passphrase: string) => {
    const response = await fetch('/api/security/encryption/pgp-keys/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, passphrase }),
    });
    const key = await response.json();
    setPgpKeys(prev => [...prev, key]);
    return key;
  }, []);

  const importPGPKey = useCallback(async (armoredKey: string) => {
    const response = await fetch('/api/security/encryption/pgp-keys/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ armoredKey }),
    });
    const key = await response.json();
    setPgpKeys(prev => [...prev, key]);
    return key;
  }, []);

  const deletePGPKey = useCallback(async (keyId: string) => {
    await fetch(`/api/security/encryption/pgp-keys/${keyId}`, { method: 'DELETE' });
    setPgpKeys(prev => prev.filter(k => k.id !== keyId));
  }, []);

  return { settings, pgpKeys, updateSettings, generatePGPKey, importPGPKey, deletePGPKey };
}

function useAuditLog() {
  const [entries, setEntries] = useState<AuditLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [hasMore, setHasMore] = useState(true);

  const loadMore = useCallback(async (offset: number = 0) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/security/audit-log?offset=${offset}&limit=50`);
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

  const exportLog = useCallback(async (format: 'csv' | 'json') => {
    const response = await fetch(`/api/security/audit-log/export?format=${format}`);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `audit-log.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  return { entries, loading, hasMore, loadMore, exportLog };
}

function useBreachMonitoring() {
  const [alerts, setAlerts] = useState<BreachAlert[]>([]);
  const [monitoredEmails, setMonitoredEmails] = useState<string[]>([]);

  useEffect(() => {
    fetch('/api/security/breaches/alerts')
      .then(res => res.json())
      .then(setAlerts);

    fetch('/api/security/breaches/monitored-emails')
      .then(res => res.json())
      .then(setMonitoredEmails);
  }, []);

  const acknowledgeAlert = useCallback(async (alertId: string) => {
    await fetch(`/api/security/breaches/alerts/${alertId}/acknowledge`, { method: 'POST' });
    setAlerts(prev => prev.map(a => a.id === alertId ? { ...a, acknowledged: true } : a));
  }, []);

  const addMonitoredEmail = useCallback(async (email: string) => {
    await fetch('/api/security/breaches/monitored-emails', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    });
    setMonitoredEmails(prev => [...prev, email]);
  }, []);

  const removeMonitoredEmail = useCallback(async (email: string) => {
    await fetch(`/api/security/breaches/monitored-emails/${encodeURIComponent(email)}`, {
      method: 'DELETE',
    });
    setMonitoredEmails(prev => prev.filter(e => e !== email));
  }, []);

  const checkNow = useCallback(async () => {
    const response = await fetch('/api/security/breaches/check', { method: 'POST' });
    const newAlerts = await response.json();
    setAlerts(prev => [...newAlerts, ...prev]);
    return newAlerts;
  }, []);

  return { alerts, monitoredEmails, acknowledgeAlert, addMonitoredEmail, removeMonitoredEmail, checkNow };
}

function useSessions() {
  const [sessions, setSessions] = useState<ActiveSession[]>([]);

  useEffect(() => {
    fetch('/api/security/sessions')
      .then(res => res.json())
      .then(setSessions);
  }, []);

  const revokeSession = useCallback(async (sessionId: string) => {
    await fetch(`/api/security/sessions/${sessionId}`, { method: 'DELETE' });
    setSessions(prev => prev.filter(s => s.id !== sessionId));
  }, []);

  const revokeAllOther = useCallback(async () => {
    await fetch('/api/security/sessions/revoke-others', { method: 'POST' });
    setSessions(prev => prev.filter(s => s.isCurrent));
  }, []);

  return { sessions, revokeSession, revokeAllOther };
}

// ============================================================================
// Components
// ============================================================================

function SecurityDashboard() {
  const { score, loading, refresh } = useSecurityScore();

  if (loading) {
    return <div className="security-dashboard loading">Loading security score...</div>;
  }

  if (!score) {
    return <div className="security-dashboard error">Failed to load security score</div>;
  }

  const getScoreColor = (value: number) =>
    value >= 80 ? '#22c55e' : value >= 50 ? '#eab308' : '#ef4444';

  const categories = [
    { key: 'authentication', label: 'Authentication', icon: '🔐' },
    { key: 'encryption', label: 'Encryption', icon: '🔒' },
    { key: 'phishing', label: 'Phishing Protection', icon: '🎣' },
    { key: 'privacy', label: 'Privacy', icon: '👁️' },
  ] as const;

  return (
    <div className="security-dashboard">
      <header>
        <h2>Security Overview</h2>
        <button onClick={refresh}>Refresh</button>
      </header>

      <div className="overall-score">
        <div
          className="score-circle"
          style={{ '--score-color': getScoreColor(score.overall) } as React.CSSProperties}
        >
          <span className="score">{Math.round(score.overall)}</span>
          <span className="label">Security Score</span>
        </div>
      </div>

      <div className="category-scores">
        {categories.map(cat => (
          <div key={cat.key} className="category">
            <span className="icon">{cat.icon}</span>
            <span className="label">{cat.label}</span>
            <div className="score-bar">
              <div
                className="fill"
                style={{
                  width: `${score[cat.key]}%`,
                  backgroundColor: getScoreColor(score[cat.key]),
                }}
              />
            </div>
            <span className="value">{Math.round(score[cat.key])}</span>
          </div>
        ))}
      </div>

      {score.recommendations.length > 0 && (
        <div className="recommendations">
          <h3>Security Recommendations</h3>
          {score.recommendations.map(rec => (
            <div key={rec.id} className={`recommendation ${rec.impact.toLowerCase()}`}>
              <div className="header">
                <span className="title">{rec.title}</span>
                <span className="impact">{rec.impact}</span>
              </div>
              <p className="description">{rec.description}</p>
              <button className="action-btn">{rec.action}</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function PasskeyManager() {
  const { passkeys, loading, registerPasskey, deletePasskey, renamePasskey } = usePasskeys();
  const [showRegister, setShowRegister] = useState(false);
  const [newName, setNewName] = useState('');
  const [registering, setRegistering] = useState(false);

  const handleRegister = async () => {
    if (!newName) return;
    setRegistering(true);
    try {
      await registerPasskey(newName);
      setShowRegister(false);
      setNewName('');
    } catch (error) {
      console.error('Passkey registration failed:', error);
    } finally {
      setRegistering(false);
    }
  };

  return (
    <div className="passkey-manager">
      <header>
        <h2>Passkeys & WebAuthn</h2>
        <button onClick={() => setShowRegister(true)}>Add Passkey</button>
      </header>

      <p className="description">
        Passkeys provide phishing-resistant authentication using biometrics or hardware security keys.
      </p>

      {loading ? (
        <div className="loading">Loading passkeys...</div>
      ) : passkeys.length === 0 ? (
        <div className="empty-state">
          <p>No passkeys registered</p>
          <button onClick={() => setShowRegister(true)}>Register your first passkey</button>
        </div>
      ) : (
        <div className="passkey-list">
          {passkeys.map(passkey => (
            <div key={passkey.id} className="passkey-item">
              <div className="icon">🔑</div>
              <div className="info">
                <span className="name">{passkey.name}</span>
                <span className="device">{passkey.deviceInfo}</span>
                <span className="dates">
                  Created: {new Date(passkey.createdAt).toLocaleDateString()}
                  {passkey.lastUsed && ` • Last used: ${new Date(passkey.lastUsed).toLocaleDateString()}`}
                </span>
              </div>
              <div className="actions">
                <button onClick={() => {
                  const name = prompt('New name:', passkey.name);
                  if (name) renamePasskey(passkey.id, name);
                }}>
                  Rename
                </button>
                <button onClick={() => deletePasskey(passkey.id)} className="danger">
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {showRegister && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Register New Passkey</h3>
            <p>Give this passkey a name to identify it later.</p>
            <input
              type="text"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              placeholder="e.g., MacBook Pro Touch ID"
            />
            <div className="modal-actions">
              <button onClick={() => setShowRegister(false)}>Cancel</button>
              <button onClick={handleRegister} disabled={!newName || registering}>
                {registering ? 'Registering...' : 'Register'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function PhishingProtectionPanel() {
  const { reports, stats, markSafe, setProtectionLevel } = usePhishingProtection();

  const threatIcons: Record<string, string> = {
    Phishing: '🎣',
    Malware: '🦠',
    Spam: '📧',
    Impersonation: '🎭',
    BEC: '💼',
  };

  return (
    <div className="phishing-protection-panel">
      <header>
        <h2>Phishing Protection</h2>
      </header>

      {stats && (
        <div className="stats">
          <div className="stat">
            <span className="value">{stats.totalDetected}</span>
            <span className="label">Total Detected</span>
          </div>
          <div className="stat">
            <span className="value">{stats.blockedThisMonth}</span>
            <span className="label">Blocked This Month</span>
          </div>
          <div className="stat">
            <span className="value">{stats.falsePositives}</span>
            <span className="label">False Positives</span>
          </div>
        </div>
      )}

      {stats && (
        <div className="protection-level">
          <h3>Protection Level</h3>
          <div className="level-selector">
            {(['Maximum', 'Balanced', 'Minimal'] as const).map(level => (
              <button
                key={level}
                className={stats.protectionLevel === level ? 'active' : ''}
                onClick={() => setProtectionLevel(level)}
              >
                {level}
              </button>
            ))}
          </div>
          <p className="hint">
            {stats.protectionLevel === 'Maximum' && 'Aggressive scanning, may have more false positives'}
            {stats.protectionLevel === 'Balanced' && 'Good balance of protection and accuracy'}
            {stats.protectionLevel === 'Minimal' && 'Only obvious threats, fewer interruptions'}
          </p>
        </div>
      )}

      <div className="recent-threats">
        <h3>Recent Detections</h3>
        {reports.length === 0 ? (
          <div className="empty">No threats detected recently</div>
        ) : (
          <div className="threat-list">
            {reports.slice(0, 10).map(report => (
              <div key={report.id} className={`threat-item ${report.threatType.toLowerCase()}`}>
                <span className="icon">{threatIcons[report.threatType] || '⚠️'}</span>
                <div className="info">
                  <span className="type">{report.threatType}</span>
                  <span className="from">{report.from}</span>
                  <span className="subject">{report.subject}</span>
                  <div className="indicators">
                    {report.indicators.slice(0, 3).map((ind, i) => (
                      <span key={i} className="indicator">{ind}</span>
                    ))}
                  </div>
                </div>
                <div className="meta">
                  <span className="confidence">{Math.round(report.confidence * 100)}% confident</span>
                  <span className="action">{report.action}</span>
                </div>
                <button onClick={() => markSafe(report.id)}>Mark Safe</button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function EncryptionPanel() {
  const { settings, pgpKeys, updateSettings, generatePGPKey, importPGPKey, deletePGPKey } = useEncryption();
  const [showGenerate, setShowGenerate] = useState(false);
  const [newKeyEmail, setNewKeyEmail] = useState('');
  const [newKeyPassphrase, setNewKeyPassphrase] = useState('');

  if (!settings) {
    return <div className="loading">Loading encryption settings...</div>;
  }

  const handleGenerate = async () => {
    if (newKeyEmail && newKeyPassphrase) {
      await generatePGPKey(newKeyEmail, newKeyPassphrase);
      setShowGenerate(false);
      setNewKeyEmail('');
      setNewKeyPassphrase('');
    }
  };

  return (
    <div className="encryption-panel">
      <header>
        <h2>Encryption Settings</h2>
      </header>

      <div className="encryption-options">
        <div className="option">
          <label>
            <input
              type="checkbox"
              checked={settings.atRestEnabled}
              onChange={e => updateSettings({ atRestEnabled: e.target.checked })}
            />
            Encryption at rest
          </label>
          <span className="hint">Encrypt stored emails on disk</span>
        </div>

        <div className="option">
          <label>
            <input
              type="checkbox"
              checked={settings.inTransitEnabled}
              onChange={e => updateSettings({ inTransitEnabled: e.target.checked })}
            />
            Encryption in transit (TLS)
          </label>
          <span className="hint">Require TLS for sending/receiving</span>
        </div>

        <div className="option">
          <label>
            <input
              type="checkbox"
              checked={settings.endToEndEnabled}
              onChange={e => updateSettings({ endToEndEnabled: e.target.checked })}
            />
            End-to-end encryption
          </label>
          <span className="hint">Encrypt message content, not just transport</span>
        </div>

        <div className="option">
          <label>
            <input
              type="checkbox"
              checked={settings.pgpEnabled}
              onChange={e => updateSettings({ pgpEnabled: e.target.checked })}
            />
            Enable PGP/GPG
          </label>
        </div>

        <div className="option">
          <label>
            <input
              type="checkbox"
              checked={settings.smimeEnabled}
              onChange={e => updateSettings({ smimeEnabled: e.target.checked })}
            />
            Enable S/MIME
          </label>
        </div>

        <div className="option">
          <label>
            <input
              type="checkbox"
              checked={settings.autoEncryptContacts}
              onChange={e => updateSettings({ autoEncryptContacts: e.target.checked })}
            />
            Auto-encrypt when recipient has public key
          </label>
        </div>

        <div className="default-encryption">
          <label>Default encryption method:</label>
          <select
            value={settings.defaultEncryption}
            onChange={e => updateSettings({ defaultEncryption: e.target.value as EncryptionSettings['defaultEncryption'] })}
          >
            <option value="None">None</option>
            <option value="TLS">TLS (transport only)</option>
            <option value="PGP">PGP</option>
            <option value="SMIME">S/MIME</option>
          </select>
        </div>
      </div>

      <div className="pgp-keys-section">
        <div className="section-header">
          <h3>PGP Keys</h3>
          <button onClick={() => setShowGenerate(true)}>Generate Key</button>
        </div>

        {pgpKeys.length === 0 ? (
          <div className="empty">No PGP keys configured</div>
        ) : (
          <div className="key-list">
            {pgpKeys.map(key => (
              <div key={key.id} className={`key-item ${key.isPrivate ? 'private' : 'public'}`}>
                <span className="icon">{key.isPrivate ? '🔐' : '🔓'}</span>
                <div className="info">
                  <span className="email">{key.email}</span>
                  <span className="fingerprint">{key.fingerprint}</span>
                  <span className="meta">
                    {key.algorithm} {key.keySize}-bit •
                    Created: {new Date(key.createdAt).toLocaleDateString()}
                    {key.expiresAt && ` • Expires: ${new Date(key.expiresAt).toLocaleDateString()}`}
                  </span>
                </div>
                <button onClick={() => deletePGPKey(key.id)} className="danger">Delete</button>
              </div>
            ))}
          </div>
        )}
      </div>

      {showGenerate && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Generate PGP Key</h3>
            <div className="form-group">
              <label>Email</label>
              <input
                type="email"
                value={newKeyEmail}
                onChange={e => setNewKeyEmail(e.target.value)}
                placeholder="your@email.com"
              />
            </div>
            <div className="form-group">
              <label>Passphrase</label>
              <input
                type="password"
                value={newKeyPassphrase}
                onChange={e => setNewKeyPassphrase(e.target.value)}
                placeholder="Strong passphrase"
              />
            </div>
            <div className="modal-actions">
              <button onClick={() => setShowGenerate(false)}>Cancel</button>
              <button onClick={handleGenerate} disabled={!newKeyEmail || !newKeyPassphrase}>
                Generate
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function AuditLogPanel() {
  const { entries, loading, hasMore, loadMore, exportLog } = useAuditLog();

  const eventIcons: Record<string, string> = {
    Login: '🔓',
    Logout: '🔒',
    PasswordChange: '🔑',
    EmailSent: '📤',
    EmailDeleted: '🗑️',
    SettingsChanged: '⚙️',
    PasskeyAdded: '🔐',
    PasskeyRemoved: '🔐',
    ApiAccess: '🔌',
    ExportData: '📦',
  };

  return (
    <div className="audit-log-panel">
      <header>
        <h2>Security Audit Log</h2>
        <div className="export-buttons">
          <button onClick={() => exportLog('csv')}>Export CSV</button>
          <button onClick={() => exportLog('json')}>Export JSON</button>
        </div>
      </header>

      <div className="log-entries">
        {entries.map(entry => (
          <div key={entry.id} className={`log-entry ${entry.success ? 'success' : 'failure'}`}>
            <span className="icon">{eventIcons[entry.eventType] || '📋'}</span>
            <div className="content">
              <div className="header">
                <span className="event-type">{entry.eventType}</span>
                <span className="timestamp">{new Date(entry.timestamp).toLocaleString()}</span>
              </div>
              <span className="details">{entry.details}</span>
              <div className="meta">
                <span className="actor">{entry.actor}</span>
                {entry.target && <span className="target">→ {entry.target}</span>}
                <span className="ip">{entry.ipAddress}</span>
              </div>
            </div>
            <span className={`status ${entry.success ? 'success' : 'failure'}`}>
              {entry.success ? '✓' : '✗'}
            </span>
          </div>
        ))}
      </div>

      {hasMore && (
        <button
          className="load-more"
          onClick={() => loadMore(entries.length)}
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Load More'}
        </button>
      )}
    </div>
  );
}

function BreachMonitoringPanel() {
  const { alerts, monitoredEmails, acknowledgeAlert, addMonitoredEmail, removeMonitoredEmail, checkNow } = useBreachMonitoring();
  const [newEmail, setNewEmail] = useState('');

  const handleAddEmail = () => {
    if (newEmail && !monitoredEmails.includes(newEmail)) {
      addMonitoredEmail(newEmail);
      setNewEmail('');
    }
  };

  const unacknowledgedAlerts = alerts.filter(a => !a.acknowledged);

  return (
    <div className="breach-monitoring-panel">
      <header>
        <h2>Breach Monitoring</h2>
        <button onClick={checkNow}>Check Now</button>
      </header>

      <div className="monitored-emails">
        <h3>Monitored Emails</h3>
        <div className="email-input">
          <input
            type="email"
            value={newEmail}
            onChange={e => setNewEmail(e.target.value)}
            placeholder="Add email to monitor"
          />
          <button onClick={handleAddEmail}>Add</button>
        </div>
        <div className="email-list">
          {monitoredEmails.map(email => (
            <div key={email} className="email-item">
              <span>{email}</span>
              <button onClick={() => removeMonitoredEmail(email)}>Remove</button>
            </div>
          ))}
        </div>
      </div>

      <div className="breach-alerts">
        <h3>
          Breach Alerts
          {unacknowledgedAlerts.length > 0 && (
            <span className="badge">{unacknowledgedAlerts.length}</span>
          )}
        </h3>

        {alerts.length === 0 ? (
          <div className="empty">No breaches detected for your monitored emails</div>
        ) : (
          <div className="alert-list">
            {alerts.map(alert => (
              <div
                key={alert.id}
                className={`alert-item ${alert.severity.toLowerCase()} ${alert.acknowledged ? 'acknowledged' : ''}`}
              >
                <div className="header">
                  <span className="source">{alert.source}</span>
                  <span className="severity">{alert.severity}</span>
                </div>
                <span className="email">{alert.affectedEmail}</span>
                <div className="data-types">
                  {alert.dataTypes.map(type => (
                    <span key={type} className="data-type">{type}</span>
                  ))}
                </div>
                <span className="date">
                  Detected: {new Date(alert.detectedAt).toLocaleDateString()}
                  {alert.breachDate && ` • Breach: ${new Date(alert.breachDate).toLocaleDateString()}`}
                </span>
                {!alert.acknowledged && (
                  <button onClick={() => acknowledgeAlert(alert.id)}>Acknowledge</button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function SessionsPanel() {
  const { sessions, revokeSession, revokeAllOther } = useSessions();

  return (
    <div className="sessions-panel">
      <header>
        <h2>Active Sessions</h2>
        <button onClick={revokeAllOther} className="danger">
          Revoke All Other Sessions
        </button>
      </header>

      <div className="session-list">
        {sessions.map(session => (
          <div key={session.id} className={`session-item ${session.isCurrent ? 'current' : ''}`}>
            <span className="icon">
              {session.browser.includes('Chrome') ? '🌐' :
               session.browser.includes('Firefox') ? '🦊' :
               session.browser.includes('Safari') ? '🧭' : '💻'}
            </span>
            <div className="info">
              <span className="device">{session.deviceName}</span>
              <span className="browser">{session.browser}</span>
              <span className="location">
                {session.ipAddress}
                {session.location && ` • ${session.location}`}
              </span>
              <span className="activity">
                {session.isCurrent ? 'Current session' : `Last active: ${new Date(session.lastActive).toLocaleString()}`}
              </span>
            </div>
            {!session.isCurrent && (
              <button onClick={() => revokeSession(session.id)} className="danger">
                Revoke
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

interface SecurityCenterProps {
  initialTab?: string;
}

export function SecurityCenter({ initialTab = 'overview' }: SecurityCenterProps) {
  const [activeTab, setActiveTab] = useState(initialTab);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '🛡️' },
    { id: 'passkeys', label: 'Passkeys', icon: '🔑' },
    { id: 'phishing', label: 'Phishing', icon: '🎣' },
    { id: 'encryption', label: 'Encryption', icon: '🔒' },
    { id: 'audit', label: 'Audit Log', icon: '📋' },
    { id: 'breaches', label: 'Breaches', icon: '⚠️' },
    { id: 'sessions', label: 'Sessions', icon: '💻' },
  ];

  return (
    <div className="security-center">
      <nav className="security-tabs">
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

      <div className="security-content">
        {activeTab === 'overview' && <SecurityDashboard />}
        {activeTab === 'passkeys' && <PasskeyManager />}
        {activeTab === 'phishing' && <PhishingProtectionPanel />}
        {activeTab === 'encryption' && <EncryptionPanel />}
        {activeTab === 'audit' && <AuditLogPanel />}
        {activeTab === 'breaches' && <BreachMonitoringPanel />}
        {activeTab === 'sessions' && <SessionsPanel />}
      </div>
    </div>
  );
}

export default SecurityCenter;
