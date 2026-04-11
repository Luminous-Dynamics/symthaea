// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Multi-Account Manager Component
 *
 * Manage multiple email accounts, unified inbox settings, and send-as identities
 */

import React, { useState, useEffect } from 'react';

interface EmailAccount {
  id: string;
  emailAddress: string;
  displayName?: string;
  accountType: 'imap' | 'exchange' | 'gmail' | 'outlook' | 'custom';
  provider?: string;
  isPrimary: boolean;
  isEnabled: boolean;
  color?: string;
  lastSyncAt?: string;
  syncStatus: 'active' | 'syncing' | 'error' | 'paused' | 'disconnected';
}

interface AccountStats {
  accountId: string;
  emailAddress: string;
  totalEmails: number;
  unreadCount: number;
  storageUsedBytes: number;
  lastSyncAt?: string;
  syncStatus: string;
}

interface SendAsIdentity {
  id: string;
  accountId: string;
  emailAddress: string;
  displayName?: string;
  signatureId?: string;
  isDefault: boolean;
  replyTo?: string;
}

type Tab = 'accounts' | 'unified' | 'identities';

export default function AccountsManager() {
  const [activeTab, setActiveTab] = useState<Tab>('accounts');

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Email Accounts</h1>

      <div className="flex gap-4 border-b border-border mb-6">
        {(['accounts', 'unified', 'identities'] as Tab[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`pb-3 px-2 border-b-2 capitalize ${
              activeTab === tab ? 'border-primary font-medium' : 'border-transparent text-muted'
            }`}
          >
            {tab === 'unified' ? 'Unified Inbox' : tab === 'identities' ? 'Send As' : tab}
          </button>
        ))}
      </div>

      {activeTab === 'accounts' && <AccountsTab />}
      {activeTab === 'unified' && <UnifiedInboxTab />}
      {activeTab === 'identities' && <IdentitiesTab />}
    </div>
  );
}

function AccountsTab() {
  const [accounts, setAccounts] = useState<EmailAccount[]>([]);
  const [stats, setStats] = useState<AccountStats[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAddModal, setShowAddModal] = useState(false);

  useEffect(() => {
    fetchAccounts();
    fetchStats();
  }, []);

  const fetchAccounts = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/accounts');
      if (response.ok) setAccounts(await response.json());
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    const response = await fetch('/api/accounts/stats');
    if (response.ok) setStats(await response.json());
  };

  const handleSetPrimary = async (id: string) => {
    await fetch(`/api/accounts/${id}/primary`, { method: 'POST' });
    fetchAccounts();
  };

  const handleToggle = async (id: string, enabled: boolean) => {
    await fetch(`/api/accounts/${id}/toggle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ is_enabled: enabled }),
    });
    fetchAccounts();
  };

  const handleSync = async (id: string) => {
    await fetch(`/api/accounts/${id}/sync`, { method: 'POST' });
    fetchStats();
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this account? All emails from this account will be removed.')) return;
    await fetch(`/api/accounts/${id}`, { method: 'DELETE' });
    fetchAccounts();
  };

  const getAccountStat = (id: string) => stats.find((s) => s.accountId === id);

  const getSyncStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800';
      case 'syncing': return 'bg-blue-100 text-blue-800';
      case 'error': return 'bg-red-100 text-red-800';
      case 'paused': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  const getProviderIcon = (type: string) => {
    switch (type) {
      case 'gmail': return 'G';
      case 'outlook': return 'O';
      case 'exchange': return 'E';
      default: return '@';
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-end">
        <button onClick={() => setShowAddModal(true)} className="px-4 py-2 bg-primary text-white rounded">
          Add Account
        </button>
      </div>

      {accounts.length === 0 ? (
        <div className="text-center py-12 border border-dashed border-border rounded-lg">
          <p className="text-muted mb-4">No email accounts connected</p>
          <button onClick={() => setShowAddModal(true)} className="px-4 py-2 bg-primary text-white rounded">
            Add Your First Account
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {accounts.map((account) => {
            const stat = getAccountStat(account.id);
            return (
              <div
                key={account.id}
                className={`border rounded-lg p-4 ${!account.isEnabled ? 'opacity-50' : ''}`}
                style={{ borderLeftWidth: 4, borderLeftColor: account.color || '#6366f1' }}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div
                      className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold"
                      style={{ backgroundColor: account.color || '#6366f1' }}
                    >
                      {getProviderIcon(account.accountType)}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-semibold">{account.displayName || account.emailAddress}</span>
                        {account.isPrimary && (
                          <span className="px-2 py-0.5 bg-primary/10 text-primary rounded text-xs">Primary</span>
                        )}
                      </div>
                      <div className="text-sm text-muted">{account.emailAddress}</div>
                      <div className="flex items-center gap-3 mt-1 text-sm">
                        <span className={`px-2 py-0.5 rounded text-xs ${getSyncStatusColor(account.syncStatus)}`}>
                          {account.syncStatus}
                        </span>
                        <span className="text-muted capitalize">{account.accountType}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button onClick={() => handleSync(account.id)} className="px-3 py-1 text-sm border border-border rounded hover:bg-muted/30">
                      Sync
                    </button>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={account.isEnabled}
                        onChange={(e) => handleToggle(account.id, e.target.checked)}
                      />
                      <span className="text-sm">Enabled</span>
                    </label>
                  </div>
                </div>

                {stat && (
                  <div className="mt-4 pt-4 border-t border-border grid grid-cols-4 gap-4 text-sm">
                    <div>
                      <div className="text-muted">Total Emails</div>
                      <div className="font-semibold">{stat.totalEmails.toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="text-muted">Unread</div>
                      <div className="font-semibold">{stat.unreadCount.toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="text-muted">Storage</div>
                      <div className="font-semibold">{formatBytes(stat.storageUsedBytes)}</div>
                    </div>
                    <div>
                      <div className="text-muted">Last Sync</div>
                      <div className="font-semibold">
                        {stat.lastSyncAt ? new Date(stat.lastSyncAt).toLocaleString() : 'Never'}
                      </div>
                    </div>
                  </div>
                )}

                <div className="mt-4 flex items-center gap-4 text-sm">
                  {!account.isPrimary && (
                    <button onClick={() => handleSetPrimary(account.id)} className="text-primary hover:underline">
                      Set as Primary
                    </button>
                  )}
                  <button className="text-primary hover:underline">Settings</button>
                  <button onClick={() => handleDelete(account.id)} className="text-red-500 hover:underline">
                    Remove
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {showAddModal && <AddAccountModal onClose={() => setShowAddModal(false)} onSuccess={() => { setShowAddModal(false); fetchAccounts(); }} />}
    </div>
  );
}

function AddAccountModal({
  onClose,
  onSuccess,
}: {
  onClose: () => void;
  onSuccess: () => void;
}) {
  const [step, setStep] = useState<'type' | 'credentials' | 'testing'>('type');
  const [accountType, setAccountType] = useState<string>('');
  const [email, setEmail] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [imapHost, setImapHost] = useState('');
  const [imapPort, setImapPort] = useState('993');
  const [smtpHost, setSmtpHost] = useState('');
  const [smtpPort, setSmtpPort] = useState('587');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const providers = [
    { id: 'gmail', name: 'Gmail', icon: 'G', color: '#EA4335' },
    { id: 'outlook', name: 'Outlook', icon: 'O', color: '#0078D4' },
    { id: 'imap', name: 'IMAP/SMTP', icon: '@', color: '#6366f1' },
  ];

  const handleOAuth = async (provider: string) => {
    window.location.href = `/api/auth/${provider}/connect`;
  };

  const handleImapSubmit = async () => {
    setStep('testing');
    setError('');

    try {
      const response = await fetch('/api/accounts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email_address: email,
          display_name: displayName,
          account_type: 'imap',
          credentials: {
            imap_host: imapHost,
            imap_port: parseInt(imapPort),
            smtp_host: smtpHost,
            smtp_port: parseInt(smtpPort),
            imap_username: username,
            imap_password: password,
            smtp_username: username,
            smtp_password: password,
            use_ssl: true,
          },
        }),
      });

      if (response.ok) {
        onSuccess();
      } else {
        const data = await response.json();
        setError(data.message || 'Failed to add account');
        setStep('credentials');
      }
    } catch (e) {
      setError('Connection failed. Please check your settings.');
      setStep('credentials');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-md">
        <div className="p-4 border-b border-border flex items-center justify-between">
          <h2 className="text-lg font-semibold">Add Email Account</h2>
          <button onClick={onClose} className="text-muted hover:text-foreground">X</button>
        </div>

        <div className="p-4">
          {step === 'type' && (
            <div className="space-y-3">
              <p className="text-muted mb-4">Choose your email provider</p>
              {providers.map((provider) => (
                <button
                  key={provider.id}
                  onClick={() => {
                    if (provider.id === 'imap') {
                      setAccountType(provider.id);
                      setStep('credentials');
                    } else {
                      handleOAuth(provider.id);
                    }
                  }}
                  className="w-full flex items-center gap-3 p-3 border border-border rounded-lg hover:bg-muted/30"
                >
                  <div
                    className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold"
                    style={{ backgroundColor: provider.color }}
                  >
                    {provider.icon}
                  </div>
                  <span className="font-medium">{provider.name}</span>
                </button>
              ))}
            </div>
          )}

          {step === 'credentials' && (
            <div className="space-y-4">
              <button onClick={() => setStep('type')} className="text-sm text-muted hover:text-foreground">
                &larr; Back
              </button>

              {error && (
                <div className="p-3 bg-red-50 border border-red-200 rounded text-red-800 text-sm">
                  {error}
                </div>
              )}

              <div>
                <label className="block text-sm font-medium mb-1">Email Address *</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-3 py-2 border border-border rounded"
                  placeholder="you@example.com"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Display Name</label>
                <input
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  className="w-full px-3 py-2 border border-border rounded"
                  placeholder="John Doe"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">IMAP Host *</label>
                  <input
                    type="text"
                    value={imapHost}
                    onChange={(e) => setImapHost(e.target.value)}
                    className="w-full px-3 py-2 border border-border rounded"
                    placeholder="imap.example.com"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">IMAP Port</label>
                  <input
                    type="text"
                    value={imapPort}
                    onChange={(e) => setImapPort(e.target.value)}
                    className="w-full px-3 py-2 border border-border rounded"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">SMTP Host *</label>
                  <input
                    type="text"
                    value={smtpHost}
                    onChange={(e) => setSmtpHost(e.target.value)}
                    className="w-full px-3 py-2 border border-border rounded"
                    placeholder="smtp.example.com"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">SMTP Port</label>
                  <input
                    type="text"
                    value={smtpPort}
                    onChange={(e) => setSmtpPort(e.target.value)}
                    className="w-full px-3 py-2 border border-border rounded"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Username *</label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full px-3 py-2 border border-border rounded"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Password *</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-3 py-2 border border-border rounded"
                />
              </div>

              <button
                onClick={handleImapSubmit}
                disabled={!email || !imapHost || !smtpHost || !username || !password}
                className="w-full py-2 bg-primary text-white rounded disabled:opacity-50"
              >
                Add Account
              </button>
            </div>
          )}

          {step === 'testing' && (
            <div className="text-center py-8">
              <div className="animate-spin h-12 w-12 border-2 border-primary border-t-transparent rounded-full mx-auto mb-4" />
              <p>Testing connection...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function UnifiedInboxTab() {
  const [accounts, setAccounts] = useState<EmailAccount[]>([]);
  const [settings, setSettings] = useState({
    unifiedInboxEnabled: true,
    showAccountColors: true,
    groupByAccount: false,
    defaultSortOrder: 'date',
  });

  useEffect(() => {
    fetch('/api/accounts').then((r) => r.json()).then(setAccounts);
  }, []);

  const handleSettingChange = (key: string, value: any) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
    fetch('/api/settings/unified-inbox', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: value }),
    });
  };

  const handleColorChange = async (accountId: string, color: string) => {
    await fetch(`/api/accounts/${accountId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ color }),
    });
  };

  const colors = ['#EF4444', '#F59E0B', '#10B981', '#3B82F6', '#6366F1', '#8B5CF6', '#EC4899'];

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <label className="flex items-center justify-between p-4 border border-border rounded-lg">
          <div>
            <p className="font-medium">Unified Inbox</p>
            <p className="text-sm text-muted">Show emails from all accounts in one view</p>
          </div>
          <input
            type="checkbox"
            checked={settings.unifiedInboxEnabled}
            onChange={(e) => handleSettingChange('unifiedInboxEnabled', e.target.checked)}
            className="h-5 w-5"
          />
        </label>

        <label className="flex items-center justify-between p-4 border border-border rounded-lg">
          <div>
            <p className="font-medium">Show Account Colors</p>
            <p className="text-sm text-muted">Display colored indicators for each account</p>
          </div>
          <input
            type="checkbox"
            checked={settings.showAccountColors}
            onChange={(e) => handleSettingChange('showAccountColors', e.target.checked)}
            className="h-5 w-5"
          />
        </label>

        <label className="flex items-center justify-between p-4 border border-border rounded-lg">
          <div>
            <p className="font-medium">Group by Account</p>
            <p className="text-sm text-muted">Separate emails by account in unified view</p>
          </div>
          <input
            type="checkbox"
            checked={settings.groupByAccount}
            onChange={(e) => handleSettingChange('groupByAccount', e.target.checked)}
            className="h-5 w-5"
          />
        </label>
      </div>

      <div>
        <h3 className="font-semibold mb-4">Account Colors</h3>
        <div className="space-y-3">
          {accounts.map((account) => (
            <div key={account.id} className="flex items-center justify-between p-3 border border-border rounded-lg">
              <span>{account.emailAddress}</span>
              <div className="flex gap-2">
                {colors.map((color) => (
                  <button
                    key={color}
                    onClick={() => handleColorChange(account.id, color)}
                    className={`w-6 h-6 rounded-full border-2 ${account.color === color ? 'border-foreground' : 'border-transparent'}`}
                    style={{ backgroundColor: color }}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function IdentitiesTab() {
  const [identities, setIdentities] = useState<SendAsIdentity[]>([]);
  const [accounts, setAccounts] = useState<EmailAccount[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [newIdentity, setNewIdentity] = useState({ accountId: '', emailAddress: '', displayName: '', replyTo: '' });

  useEffect(() => {
    Promise.all([
      fetch('/api/accounts').then((r) => r.json()),
      fetch('/api/identities').then((r) => r.json()),
    ]).then(([acc, ids]) => {
      setAccounts(acc);
      setIdentities(ids);
      setLoading(false);
    });
  }, []);

  const handleAdd = async () => {
    await fetch('/api/identities', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        account_id: newIdentity.accountId,
        email_address: newIdentity.emailAddress,
        display_name: newIdentity.displayName,
        reply_to: newIdentity.replyTo || null,
      }),
    });
    const response = await fetch('/api/identities');
    setIdentities(await response.json());
    setShowForm(false);
    setNewIdentity({ accountId: '', emailAddress: '', displayName: '', replyTo: '' });
  };

  const handleSetDefault = async (accountId: string, identityId: string) => {
    await fetch(`/api/identities/${identityId}/default`, { method: 'POST' });
    const response = await fetch('/api/identities');
    setIdentities(await response.json());
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this identity?')) return;
    await fetch(`/api/identities/${id}`, { method: 'DELETE' });
    const response = await fetch('/api/identities');
    setIdentities(await response.json());
  };

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  const groupedIdentities = identities.reduce((acc, id) => {
    if (!acc[id.accountId]) acc[id.accountId] = [];
    acc[id.accountId].push(id);
    return acc;
  }, {} as Record<string, SendAsIdentity[]>);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <p className="text-muted">Configure different identities for sending emails</p>
        <button onClick={() => setShowForm(!showForm)} className="px-4 py-2 bg-primary text-white rounded">
          {showForm ? 'Cancel' : 'Add Identity'}
        </button>
      </div>

      {showForm && (
        <div className="border border-border rounded-lg p-4 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Account *</label>
            <select
              value={newIdentity.accountId}
              onChange={(e) => setNewIdentity({ ...newIdentity, accountId: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded"
            >
              <option value="">Select account...</option>
              {accounts.map((acc) => (
                <option key={acc.id} value={acc.id}>{acc.emailAddress}</option>
              ))}
            </select>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Email Address *</label>
              <input
                type="email"
                value={newIdentity.emailAddress}
                onChange={(e) => setNewIdentity({ ...newIdentity, emailAddress: e.target.value })}
                className="w-full px-3 py-2 border border-border rounded"
                placeholder="alias@example.com"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Display Name</label>
              <input
                type="text"
                value={newIdentity.displayName}
                onChange={(e) => setNewIdentity({ ...newIdentity, displayName: e.target.value })}
                className="w-full px-3 py-2 border border-border rounded"
                placeholder="John Doe"
              />
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Reply-To Address</label>
            <input
              type="email"
              value={newIdentity.replyTo}
              onChange={(e) => setNewIdentity({ ...newIdentity, replyTo: e.target.value })}
              className="w-full px-3 py-2 border border-border rounded"
              placeholder="replies@example.com"
            />
          </div>
          <div className="flex justify-end">
            <button
              onClick={handleAdd}
              disabled={!newIdentity.accountId || !newIdentity.emailAddress}
              className="px-4 py-2 bg-primary text-white rounded disabled:opacity-50"
            >
              Add Identity
            </button>
          </div>
        </div>
      )}

      {accounts.map((account) => {
        const accountIdentities = groupedIdentities[account.id] || [];
        return (
          <div key={account.id} className="border border-border rounded-lg">
            <div className="p-3 bg-muted/20 border-b border-border">
              <span className="font-medium">{account.emailAddress}</span>
            </div>
            {accountIdentities.length === 0 ? (
              <div className="p-4 text-muted text-sm">No additional identities</div>
            ) : (
              <div className="divide-y divide-border">
                {accountIdentities.map((identity) => (
                  <div key={identity.id} className="p-4 flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{identity.displayName || identity.emailAddress}</span>
                        {identity.isDefault && (
                          <span className="px-2 py-0.5 bg-primary/10 text-primary rounded text-xs">Default</span>
                        )}
                      </div>
                      <div className="text-sm text-muted">{identity.emailAddress}</div>
                      {identity.replyTo && (
                        <div className="text-sm text-muted">Reply-To: {identity.replyTo}</div>
                      )}
                    </div>
                    <div className="flex gap-2">
                      {!identity.isDefault && (
                        <button
                          onClick={() => handleSetDefault(account.id, identity.id)}
                          className="text-sm text-primary hover:underline"
                        >
                          Set Default
                        </button>
                      )}
                      <button
                        onClick={() => handleDelete(identity.id)}
                        className="text-sm text-red-500 hover:underline"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
