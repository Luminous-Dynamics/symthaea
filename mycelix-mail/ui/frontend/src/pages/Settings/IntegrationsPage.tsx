// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Integrations Page
 *
 * Webhooks, API keys, and external service connectors
 */

import React, { useEffect, useState } from 'react';

interface Webhook {
  id: string;
  name: string;
  url: string;
  secret: string;
  events: string[];
  isActive: boolean;
  createdAt: string;
  lastTriggeredAt?: string;
  failureCount: number;
}

interface ApiKey {
  id: string;
  name: string;
  keyPrefix: string;
  scopes: string[];
  expiresAt?: string;
  createdAt: string;
  lastUsedAt?: string;
}

interface Connector {
  id: string;
  connectorType: string;
  name: string;
  triggers: string[];
  isActive: boolean;
  createdAt: string;
}

type Tab = 'webhooks' | 'api-keys' | 'connectors';

export default function IntegrationsPage() {
  const [activeTab, setActiveTab] = useState<Tab>('webhooks');
  const [webhooks, setWebhooks] = useState<Webhook[]>([]);
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const [loading, setLoading] = useState(true);

  // Modals
  const [showWebhookModal, setShowWebhookModal] = useState(false);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [showConnectorModal, setShowConnectorModal] = useState(false);
  const [newApiKey, setNewApiKey] = useState<string | null>(null);

  useEffect(() => {
    fetchAll();
  }, []);

  async function fetchAll() {
    setLoading(true);
    await Promise.all([fetchWebhooks(), fetchApiKeys(), fetchConnectors()]);
    setLoading(false);
  }

  async function fetchWebhooks() {
    const response = await fetch('/api/integrations/webhooks');
    if (response.ok) setWebhooks(await response.json());
  }

  async function fetchApiKeys() {
    const response = await fetch('/api/integrations/api-keys');
    if (response.ok) setApiKeys(await response.json());
  }

  async function fetchConnectors() {
    const response = await fetch('/api/integrations/connectors');
    if (response.ok) setConnectors(await response.json());
  }

  async function deleteWebhook(id: string) {
    if (!confirm('Delete this webhook?')) return;
    await fetch(`/api/integrations/webhooks/${id}`, { method: 'DELETE' });
    fetchWebhooks();
  }

  async function revokeApiKey(id: string) {
    if (!confirm('Revoke this API key? This cannot be undone.')) return;
    await fetch(`/api/integrations/api-keys/${id}`, { method: 'DELETE' });
    fetchApiKeys();
  }

  async function deleteConnector(id: string) {
    if (!confirm('Delete this connector?')) return;
    await fetch(`/api/integrations/connectors/${id}`, { method: 'DELETE' });
    fetchConnectors();
  }

  const eventLabels: Record<string, string> = {
    'email.received': 'Email Received',
    'email.sent': 'Email Sent',
    'email.replied': 'Email Replied',
    'email.archived': 'Email Archived',
    'email.deleted': 'Email Deleted',
    'contact.created': 'Contact Created',
    'contact.updated': 'Contact Updated',
    'assignment.created': 'Assignment Created',
    'sla.breached': 'SLA Breached',
  };

  const scopeLabels: Record<string, string> = {
    'email:read': 'Read Emails',
    'email:write': 'Write Emails',
    'email:send': 'Send Emails',
    'contacts:read': 'Read Contacts',
    'contacts:write': 'Write Contacts',
    'calendar:read': 'Read Calendar',
    'calendar:write': 'Write Calendar',
    'settings:read': 'Read Settings',
    'settings:write': 'Write Settings',
    'webhooks:manage': 'Manage Webhooks',
    full: 'Full Access',
  };

  const connectorTypes: Record<string, { name: string; icon: string }> = {
    slack: { name: 'Slack', icon: 'Slack' },
    discord: { name: 'Discord', icon: 'Discord' },
    teams: { name: 'Microsoft Teams', icon: 'Teams' },
    email: { name: 'Email', icon: 'Mail' },
    custom: { name: 'Custom Webhook', icon: 'Link' },
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold">Integrations</h1>
        <p className="text-muted">Connect external services and automate workflows</p>
      </div>

      {/* Tabs */}
      <div className="border-b border-border mb-6">
        <div className="flex gap-6">
          {([
            { id: 'webhooks', label: 'Webhooks', count: webhooks.length },
            { id: 'api-keys', label: 'API Keys', count: apiKeys.length },
            { id: 'connectors', label: 'Connectors', count: connectors.length },
          ] as const).map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`pb-3 px-1 border-b-2 font-medium ${
                activeTab === tab.id
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted hover:text-foreground'
              }`}
            >
              {tab.label}
              <span className="ml-2 px-2 py-0.5 bg-muted/30 rounded-full text-xs">
                {tab.count}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Webhooks Tab */}
      {activeTab === 'webhooks' && (
        <div>
          <div className="flex justify-between items-center mb-4">
            <p className="text-sm text-muted">
              Webhooks send HTTP requests when events occur in your mailbox.
            </p>
            <button
              onClick={() => setShowWebhookModal(true)}
              className="px-4 py-2 bg-primary text-white rounded-lg"
            >
              Add Webhook
            </button>
          </div>

          {webhooks.length === 0 ? (
            <div className="text-center py-12 bg-muted/10 rounded-lg">
              <p className="text-muted">No webhooks configured</p>
            </div>
          ) : (
            <div className="space-y-4">
              {webhooks.map((webhook) => (
                <div key={webhook.id} className="border border-border rounded-lg p-4">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="font-semibold">{webhook.name}</h3>
                        <span
                          className={`px-2 py-0.5 rounded text-xs ${
                            webhook.isActive
                              ? 'bg-green-100 text-green-700'
                              : 'bg-gray-100 text-gray-600'
                          }`}
                        >
                          {webhook.isActive ? 'Active' : 'Inactive'}
                        </span>
                        {webhook.failureCount > 0 && (
                          <span className="px-2 py-0.5 bg-red-100 text-red-700 rounded text-xs">
                            {webhook.failureCount} failures
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-muted mt-1 font-mono">{webhook.url}</p>
                    </div>
                    <button
                      onClick={() => deleteWebhook(webhook.id)}
                      className="text-red-500 hover:underline text-sm"
                    >
                      Delete
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-3">
                    {webhook.events.map((event) => (
                      <span key={event} className="px-2 py-1 bg-muted/30 rounded text-xs">
                        {eventLabels[event] || event}
                      </span>
                    ))}
                  </div>
                  <div className="flex gap-4 mt-3 text-xs text-muted">
                    <span>Created {new Date(webhook.createdAt).toLocaleDateString()}</span>
                    {webhook.lastTriggeredAt && (
                      <span>
                        Last triggered {new Date(webhook.lastTriggeredAt).toLocaleString()}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* API Keys Tab */}
      {activeTab === 'api-keys' && (
        <div>
          <div className="flex justify-between items-center mb-4">
            <p className="text-sm text-muted">
              API keys allow programmatic access to your mailbox.
            </p>
            <button
              onClick={() => setShowApiKeyModal(true)}
              className="px-4 py-2 bg-primary text-white rounded-lg"
            >
              Create API Key
            </button>
          </div>

          {newApiKey && (
            <div className="mb-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="font-semibold text-yellow-800">Save your API key now!</p>
              <p className="text-sm text-yellow-700 mt-1">
                This is the only time you'll see this key. Copy it and store it securely.
              </p>
              <div className="mt-2 flex items-center gap-2">
                <code className="flex-1 p-2 bg-white border rounded font-mono text-sm">
                  {newApiKey}
                </code>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(newApiKey);
                  }}
                  className="px-3 py-2 bg-yellow-600 text-white rounded"
                >
                  Copy
                </button>
              </div>
              <button
                onClick={() => setNewApiKey(null)}
                className="mt-2 text-sm text-yellow-700 hover:underline"
              >
                I've saved it, dismiss this
              </button>
            </div>
          )}

          {apiKeys.length === 0 ? (
            <div className="text-center py-12 bg-muted/10 rounded-lg">
              <p className="text-muted">No API keys created</p>
            </div>
          ) : (
            <div className="space-y-4">
              {apiKeys.map((key) => (
                <div key={key.id} className="border border-border rounded-lg p-4">
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="font-semibold">{key.name}</h3>
                      <p className="text-sm text-muted mt-1 font-mono">{key.keyPrefix}...</p>
                    </div>
                    <button
                      onClick={() => revokeApiKey(key.id)}
                      className="text-red-500 hover:underline text-sm"
                    >
                      Revoke
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-3">
                    {key.scopes.map((scope) => (
                      <span key={scope} className="px-2 py-1 bg-muted/30 rounded text-xs">
                        {scopeLabels[scope] || scope}
                      </span>
                    ))}
                  </div>
                  <div className="flex gap-4 mt-3 text-xs text-muted">
                    <span>Created {new Date(key.createdAt).toLocaleDateString()}</span>
                    {key.lastUsedAt && (
                      <span>Last used {new Date(key.lastUsedAt).toLocaleString()}</span>
                    )}
                    {key.expiresAt && (
                      <span className="text-orange-600">
                        Expires {new Date(key.expiresAt).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Connectors Tab */}
      {activeTab === 'connectors' && (
        <div>
          <div className="flex justify-between items-center mb-4">
            <p className="text-sm text-muted">
              Connectors send notifications to external services.
            </p>
            <button
              onClick={() => setShowConnectorModal(true)}
              className="px-4 py-2 bg-primary text-white rounded-lg"
            >
              Add Connector
            </button>
          </div>

          {connectors.length === 0 ? (
            <div className="text-center py-12 bg-muted/10 rounded-lg">
              <p className="text-muted">No connectors configured</p>
              <p className="text-sm text-muted mt-2">
                Connect Slack, Discord, or Teams to get notifications
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {connectors.map((connector) => (
                <div key={connector.id} className="border border-border rounded-lg p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-muted/30 rounded flex items-center justify-center">
                        {connectorTypes[connector.connectorType]?.icon || 'Link'}
                      </div>
                      <div>
                        <h3 className="font-semibold">{connector.name}</h3>
                        <p className="text-sm text-muted">
                          {connectorTypes[connector.connectorType]?.name || connector.connectorType}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => deleteConnector(connector.id)}
                      className="text-red-500 hover:underline text-sm"
                    >
                      Delete
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-3">
                    {connector.triggers.map((trigger) => (
                      <span key={trigger} className="px-2 py-1 bg-muted/30 rounded text-xs capitalize">
                        {trigger.replace('_', ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Webhook Modal */}
      {showWebhookModal && (
        <WebhookModal
          onClose={() => setShowWebhookModal(false)}
          onCreated={() => {
            setShowWebhookModal(false);
            fetchWebhooks();
          }}
          eventLabels={eventLabels}
        />
      )}

      {/* API Key Modal */}
      {showApiKeyModal && (
        <ApiKeyModal
          onClose={() => setShowApiKeyModal(false)}
          onCreated={(key) => {
            setShowApiKeyModal(false);
            setNewApiKey(key);
            fetchApiKeys();
          }}
          scopeLabels={scopeLabels}
        />
      )}

      {/* Connector Modal */}
      {showConnectorModal && (
        <ConnectorModal
          onClose={() => setShowConnectorModal(false)}
          onCreated={() => {
            setShowConnectorModal(false);
            fetchConnectors();
          }}
          connectorTypes={connectorTypes}
        />
      )}
    </div>
  );
}

function WebhookModal({
  onClose,
  onCreated,
  eventLabels,
}: {
  onClose: () => void;
  onCreated: () => void;
  eventLabels: Record<string, string>;
}) {
  const [name, setName] = useState('');
  const [url, setUrl] = useState('');
  const [events, setEvents] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);

  async function handleCreate() {
    setSaving(true);
    try {
      const response = await fetch('/api/integrations/webhooks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, url, events }),
      });
      if (response.ok) onCreated();
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-lg">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold">Add Webhook</h2>
        </div>
        <div className="p-4 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Webhook"
              className="w-full px-3 py-2 border border-border rounded"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">URL</label>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/webhook"
              className="w-full px-3 py-2 border border-border rounded font-mono text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Events</label>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {Object.entries(eventLabels).map(([event, label]) => (
                <label key={event} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={events.includes(event)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setEvents([...events, event]);
                      } else {
                        setEvents(events.filter((ev) => ev !== event));
                      }
                    }}
                  />
                  {label}
                </label>
              ))}
            </div>
          </div>
        </div>
        <div className="p-4 border-t border-border flex justify-end gap-2">
          <button onClick={onClose} className="px-4 py-2 border border-border rounded">
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={!name || !url || events.length === 0 || saving}
            className="px-4 py-2 bg-primary text-white rounded disabled:opacity-50"
          >
            {saving ? 'Creating...' : 'Create Webhook'}
          </button>
        </div>
      </div>
    </div>
  );
}

function ApiKeyModal({
  onClose,
  onCreated,
  scopeLabels,
}: {
  onClose: () => void;
  onCreated: (key: string) => void;
  scopeLabels: Record<string, string>;
}) {
  const [name, setName] = useState('');
  const [scopes, setScopes] = useState<string[]>([]);
  const [expiresIn, setExpiresIn] = useState<string>('never');
  const [saving, setSaving] = useState(false);

  async function handleCreate() {
    setSaving(true);
    try {
      const response = await fetch('/api/integrations/api-keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          scopes,
          expiresIn: expiresIn === 'never' ? null : expiresIn,
        }),
      });
      if (response.ok) {
        const data = await response.json();
        onCreated(data.key);
      }
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-lg">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold">Create API Key</h2>
        </div>
        <div className="p-4 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Integration"
              className="w-full px-3 py-2 border border-border rounded"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Scopes</label>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {Object.entries(scopeLabels).map(([scope, label]) => (
                <label key={scope} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={scopes.includes(scope)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setScopes([...scopes, scope]);
                      } else {
                        setScopes(scopes.filter((s) => s !== scope));
                      }
                    }}
                  />
                  {label}
                </label>
              ))}
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Expiration</label>
            <select
              value={expiresIn}
              onChange={(e) => setExpiresIn(e.target.value)}
              className="w-full px-3 py-2 border border-border rounded"
            >
              <option value="never">Never expires</option>
              <option value="30d">30 days</option>
              <option value="90d">90 days</option>
              <option value="1y">1 year</option>
            </select>
          </div>
        </div>
        <div className="p-4 border-t border-border flex justify-end gap-2">
          <button onClick={onClose} className="px-4 py-2 border border-border rounded">
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={!name || scopes.length === 0 || saving}
            className="px-4 py-2 bg-primary text-white rounded disabled:opacity-50"
          >
            {saving ? 'Creating...' : 'Create Key'}
          </button>
        </div>
      </div>
    </div>
  );
}

function ConnectorModal({
  onClose,
  onCreated,
  connectorTypes,
}: {
  onClose: () => void;
  onCreated: () => void;
  connectorTypes: Record<string, { name: string; icon: string }>;
}) {
  const [type, setType] = useState<string>('slack');
  const [name, setName] = useState('');
  const [webhookUrl, setWebhookUrl] = useState('');
  const [triggers, setTriggers] = useState<string[]>(['new_email']);
  const [saving, setSaving] = useState(false);

  const triggerOptions = [
    { value: 'new_email', label: 'New Email' },
    { value: 'urgent_email', label: 'Urgent Email' },
    { value: 'vip_email', label: 'VIP Email' },
    { value: 'assigned_email', label: 'Assigned Email' },
    { value: 'sla_warning', label: 'SLA Warning' },
    { value: 'daily_digest', label: 'Daily Digest' },
  ];

  async function handleCreate() {
    setSaving(true);
    try {
      const response = await fetch('/api/integrations/connectors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          connectorType: type,
          name,
          config: { webhook_url: webhookUrl },
          triggers,
        }),
      });
      if (response.ok) onCreated();
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-lg">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold">Add Connector</h2>
        </div>
        <div className="p-4 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Service</label>
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(connectorTypes).map(([key, { name: label }]) => (
                <button
                  key={key}
                  onClick={() => setType(key)}
                  className={`p-3 border rounded text-center ${
                    type === key ? 'border-primary bg-primary/10' : 'border-border'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Slack Notifications"
              className="w-full px-3 py-2 border border-border rounded"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Webhook URL</label>
            <input
              type="url"
              value={webhookUrl}
              onChange={(e) => setWebhookUrl(e.target.value)}
              placeholder="https://hooks.slack.com/services/..."
              className="w-full px-3 py-2 border border-border rounded font-mono text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Triggers</label>
            <div className="space-y-2">
              {triggerOptions.map((trigger) => (
                <label key={trigger.value} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={triggers.includes(trigger.value)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setTriggers([...triggers, trigger.value]);
                      } else {
                        setTriggers(triggers.filter((t) => t !== trigger.value));
                      }
                    }}
                  />
                  {trigger.label}
                </label>
              ))}
            </div>
          </div>
        </div>
        <div className="p-4 border-t border-border flex justify-end gap-2">
          <button onClick={onClose} className="px-4 py-2 border border-border rounded">
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={!name || !webhookUrl || triggers.length === 0 || saving}
            className="px-4 py-2 bg-primary text-white rounded disabled:opacity-50"
          >
            {saving ? 'Creating...' : 'Add Connector'}
          </button>
        </div>
      </div>
    </div>
  );
}
