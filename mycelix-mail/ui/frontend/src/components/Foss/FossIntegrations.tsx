// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Track Y: FOSS Integration Hub
 *
 * Integration dashboard for Nextcloud, Matrix, Jitsi, Vikunja, Keycloak,
 * n8n, Meilisearch, Gitea, Joplin, Bitwarden, CalDAV/CardDAV, and more.
 */

import React, { useState, useEffect, useCallback } from 'react';

// ============================================================================
// Types
// ============================================================================

interface Integration {
  id: string;
  name: string;
  category: 'Storage' | 'Chat' | 'Tasks' | 'Auth' | 'Automation' | 'Search' | 'Dev' | 'Notes' | 'Security' | 'Calendar';
  status: 'Connected' | 'Disconnected' | 'Error' | 'Configuring';
  icon: string;
  description: string;
  features: string[];
  configUrl?: string;
  lastSync?: string;
  error?: string;
}

interface NextcloudConfig {
  serverUrl: string;
  username: string;
  connected: boolean;
  features: {
    files: boolean;
    calendar: boolean;
    contacts: boolean;
    deck: boolean;
    notes: boolean;
  };
  quotaUsed: number;
  quotaTotal: number;
}

interface MatrixConfig {
  homeserver: string;
  userId: string;
  connected: boolean;
  rooms: { roomId: string; name: string; unreadCount: number }[];
  emailToMatrix: boolean;
  matrixToEmail: boolean;
}

interface JitsiConfig {
  serverUrl: string;
  connected: boolean;
  autoCreateMeetings: boolean;
  defaultRoomPrefix: string;
  enableLobby: boolean;
  enableRecording: boolean;
}

interface VikunjaConfig {
  serverUrl: string;
  connected: boolean;
  defaultProject?: string;
  projects: { id: string; title: string; taskCount: number }[];
  emailToTask: boolean;
}

interface OIDCProvider {
  id: string;
  name: string;
  issuer: string;
  clientId: string;
  connected: boolean;
  userCount: number;
}

interface N8nConfig {
  serverUrl: string;
  connected: boolean;
  workflows: { id: string; name: string; active: boolean }[];
  webhookSecret: string;
}

interface MeilisearchConfig {
  serverUrl: string;
  connected: boolean;
  indexedEmails: number;
  indexSize: number;
  lastIndexed?: string;
  searchEnabled: boolean;
}

interface GiteaConfig {
  serverUrl: string;
  connected: boolean;
  organizations: string[];
  issueToEmail: boolean;
  emailToIssue: boolean;
}

interface JoplinConfig {
  serverUrl: string;
  connected: boolean;
  notebooks: { id: string; title: string }[];
  emailToNote: boolean;
  syncAttachments: boolean;
}

interface BitwardenConfig {
  serverUrl: string;
  connected: boolean;
  vaultUnlocked: boolean;
  autoFillEnabled: boolean;
}

interface DAVConfig {
  caldavUrl: string;
  carddavUrl: string;
  connected: boolean;
  calendars: { id: string; name: string; color: string }[];
  addressBooks: { id: string; name: string; contactCount: number }[];
  syncInterval: number;
}

// ============================================================================
// Hooks
// ============================================================================

function useIntegrations() {
  const [integrations, setIntegrations] = useState<Integration[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/foss/integrations')
      .then(res => res.json())
      .then(setIntegrations)
      .finally(() => setLoading(false));
  }, []);

  const testConnection = useCallback(async (integrationId: string) => {
    const response = await fetch(`/api/foss/integrations/${integrationId}/test`, {
      method: 'POST',
    });
    const result = await response.json();
    setIntegrations(prev => prev.map(i =>
      i.id === integrationId ? { ...i, status: result.success ? 'Connected' : 'Error', error: result.error } : i
    ));
    return result;
  }, []);

  const disconnect = useCallback(async (integrationId: string) => {
    await fetch(`/api/foss/integrations/${integrationId}/disconnect`, { method: 'POST' });
    setIntegrations(prev => prev.map(i =>
      i.id === integrationId ? { ...i, status: 'Disconnected' } : i
    ));
  }, []);

  return { integrations, loading, testConnection, disconnect };
}

function useNextcloud() {
  const [config, setConfig] = useState<NextcloudConfig | null>(null);

  useEffect(() => {
    fetch('/api/foss/nextcloud/config')
      .then(res => res.json())
      .then(setConfig);
  }, []);

  const connect = useCallback(async (serverUrl: string, username: string, password: string) => {
    const response = await fetch('/api/foss/nextcloud/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ serverUrl, username, password }),
    });
    const result = await response.json();
    if (result.success) {
      setConfig(prev => prev ? { ...prev, serverUrl, username, connected: true } : null);
    }
    return result;
  }, []);

  const updateFeatures = useCallback(async (features: Partial<NextcloudConfig['features']>) => {
    await fetch('/api/foss/nextcloud/features', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(features),
    });
    setConfig(prev => prev ? { ...prev, features: { ...prev.features, ...features } } : null);
  }, []);

  const syncNow = useCallback(async () => {
    await fetch('/api/foss/nextcloud/sync', { method: 'POST' });
  }, []);

  return { config, connect, updateFeatures, syncNow };
}

function useMatrix() {
  const [config, setConfig] = useState<MatrixConfig | null>(null);

  useEffect(() => {
    fetch('/api/foss/matrix/config')
      .then(res => res.json())
      .then(setConfig);
  }, []);

  const connect = useCallback(async (homeserver: string, userId: string, accessToken: string) => {
    const response = await fetch('/api/foss/matrix/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ homeserver, userId, accessToken }),
    });
    const result = await response.json();
    if (result.success) {
      setConfig(prev => prev ? { ...prev, homeserver, userId, connected: true } : null);
    }
    return result;
  }, []);

  const updateBridgeSettings = useCallback(async (settings: { emailToMatrix?: boolean; matrixToEmail?: boolean }) => {
    await fetch('/api/foss/matrix/bridge', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    setConfig(prev => prev ? { ...prev, ...settings } : null);
  }, []);

  return { config, connect, updateBridgeSettings };
}

function useJitsi() {
  const [config, setConfig] = useState<JitsiConfig | null>(null);

  useEffect(() => {
    fetch('/api/foss/jitsi/config')
      .then(res => res.json())
      .then(setConfig);
  }, []);

  const updateConfig = useCallback(async (updates: Partial<JitsiConfig>) => {
    await fetch('/api/foss/jitsi/config', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    setConfig(prev => prev ? { ...prev, ...updates } : null);
  }, []);

  const createMeeting = useCallback(async (subject: string, participants: string[]) => {
    const response = await fetch('/api/foss/jitsi/meetings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subject, participants }),
    });
    return await response.json();
  }, []);

  return { config, updateConfig, createMeeting };
}

function useVikunja() {
  const [config, setConfig] = useState<VikunjaConfig | null>(null);

  useEffect(() => {
    fetch('/api/foss/vikunja/config')
      .then(res => res.json())
      .then(setConfig);
  }, []);

  const connect = useCallback(async (serverUrl: string, apiToken: string) => {
    const response = await fetch('/api/foss/vikunja/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ serverUrl, apiToken }),
    });
    return await response.json();
  }, []);

  const createTaskFromEmail = useCallback(async (emailId: string, projectId: string) => {
    const response = await fetch('/api/foss/vikunja/tasks/from-email', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ emailId, projectId }),
    });
    return await response.json();
  }, []);

  return { config, connect, createTaskFromEmail };
}

function useOIDC() {
  const [providers, setProviders] = useState<OIDCProvider[]>([]);

  useEffect(() => {
    fetch('/api/foss/oidc/providers')
      .then(res => res.json())
      .then(setProviders);
  }, []);

  const addProvider = useCallback(async (config: Omit<OIDCProvider, 'id' | 'connected' | 'userCount'>) => {
    const response = await fetch('/api/foss/oidc/providers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    const provider = await response.json();
    setProviders(prev => [...prev, provider]);
    return provider;
  }, []);

  const removeProvider = useCallback(async (providerId: string) => {
    await fetch(`/api/foss/oidc/providers/${providerId}`, { method: 'DELETE' });
    setProviders(prev => prev.filter(p => p.id !== providerId));
  }, []);

  return { providers, addProvider, removeProvider };
}

function useN8n() {
  const [config, setConfig] = useState<N8nConfig | null>(null);

  useEffect(() => {
    fetch('/api/foss/n8n/config')
      .then(res => res.json())
      .then(setConfig);
  }, []);

  const connect = useCallback(async (serverUrl: string, apiKey: string) => {
    const response = await fetch('/api/foss/n8n/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ serverUrl, apiKey }),
    });
    return await response.json();
  }, []);

  const triggerWorkflow = useCallback(async (workflowId: string, data: object) => {
    const response = await fetch(`/api/foss/n8n/workflows/${workflowId}/trigger`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return await response.json();
  }, []);

  return { config, connect, triggerWorkflow };
}

function useMeilisearch() {
  const [config, setConfig] = useState<MeilisearchConfig | null>(null);

  useEffect(() => {
    fetch('/api/foss/meilisearch/config')
      .then(res => res.json())
      .then(setConfig);
  }, []);

  const connect = useCallback(async (serverUrl: string, apiKey: string) => {
    const response = await fetch('/api/foss/meilisearch/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ serverUrl, apiKey }),
    });
    return await response.json();
  }, []);

  const reindex = useCallback(async () => {
    await fetch('/api/foss/meilisearch/reindex', { method: 'POST' });
  }, []);

  return { config, connect, reindex };
}

function useDAV() {
  const [config, setConfig] = useState<DAVConfig | null>(null);

  useEffect(() => {
    fetch('/api/foss/dav/config')
      .then(res => res.json())
      .then(setConfig);
  }, []);

  const connect = useCallback(async (caldavUrl: string, carddavUrl: string, username: string, password: string) => {
    const response = await fetch('/api/foss/dav/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ caldavUrl, carddavUrl, username, password }),
    });
    return await response.json();
  }, []);

  const syncNow = useCallback(async () => {
    await fetch('/api/foss/dav/sync', { method: 'POST' });
  }, []);

  const updateSyncInterval = useCallback(async (interval: number) => {
    await fetch('/api/foss/dav/sync-interval', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ interval }),
    });
    setConfig(prev => prev ? { ...prev, syncInterval: interval } : null);
  }, []);

  return { config, connect, syncNow, updateSyncInterval };
}

// ============================================================================
// Components
// ============================================================================

function IntegrationCard({ integration, onTest, onDisconnect }: {
  integration: Integration;
  onTest: () => void;
  onDisconnect: () => void;
}) {
  const statusColors = {
    Connected: '#22c55e',
    Disconnected: '#6b7280',
    Error: '#ef4444',
    Configuring: '#eab308',
  };

  return (
    <div className={`integration-card ${integration.status.toLowerCase()}`}>
      <div className="header">
        <span className="icon">{integration.icon}</span>
        <div className="info">
          <span className="name">{integration.name}</span>
          <span className="category">{integration.category}</span>
        </div>
        <span
          className="status"
          style={{ color: statusColors[integration.status] }}
        >
          {integration.status}
        </span>
      </div>
      <p className="description">{integration.description}</p>
      <div className="features">
        {integration.features.slice(0, 3).map((f, i) => (
          <span key={i} className="feature">{f}</span>
        ))}
        {integration.features.length > 3 && (
          <span className="more">+{integration.features.length - 3}</span>
        )}
      </div>
      {integration.error && (
        <div className="error">{integration.error}</div>
      )}
      {integration.lastSync && (
        <span className="last-sync">
          Last sync: {new Date(integration.lastSync).toLocaleString()}
        </span>
      )}
      <div className="actions">
        <button onClick={onTest}>Test</button>
        {integration.status === 'Connected' && (
          <button onClick={onDisconnect} className="danger">Disconnect</button>
        )}
        {integration.configUrl && (
          <a href={integration.configUrl} className="config-link">Configure</a>
        )}
      </div>
    </div>
  );
}

function IntegrationsList() {
  const { integrations, loading, testConnection, disconnect } = useIntegrations();
  const [filter, setFilter] = useState<string>('all');

  const categories = ['all', 'Storage', 'Chat', 'Tasks', 'Auth', 'Automation', 'Search', 'Dev', 'Notes', 'Security', 'Calendar'];

  const filtered = filter === 'all'
    ? integrations
    : integrations.filter(i => i.category === filter);

  const connectedCount = integrations.filter(i => i.status === 'Connected').length;

  return (
    <div className="integrations-list">
      <header>
        <h2>FOSS Integrations</h2>
        <span className="count">{connectedCount}/{integrations.length} connected</span>
      </header>

      <div className="filters">
        {categories.map(cat => (
          <button
            key={cat}
            className={filter === cat ? 'active' : ''}
            onClick={() => setFilter(cat)}
          >
            {cat === 'all' ? 'All' : cat}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="loading">Loading integrations...</div>
      ) : (
        <div className="grid">
          {filtered.map(integration => (
            <IntegrationCard
              key={integration.id}
              integration={integration}
              onTest={() => testConnection(integration.id)}
              onDisconnect={() => disconnect(integration.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function NextcloudPanel() {
  const { config, connect, updateFeatures, syncNow } = useNextcloud();
  const [showConnect, setShowConnect] = useState(false);
  const [serverUrl, setServerUrl] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleConnect = async () => {
    const result = await connect(serverUrl, username, password);
    if (result.success) {
      setShowConnect(false);
    }
  };

  const formatBytes = (bytes: number) => `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;

  if (!config) {
    return <div className="loading">Loading Nextcloud config...</div>;
  }

  return (
    <div className="nextcloud-panel">
      <header>
        <h2>Nextcloud</h2>
        {config.connected ? (
          <span className="status connected">Connected</span>
        ) : (
          <button onClick={() => setShowConnect(true)}>Connect</button>
        )}
      </header>

      {config.connected && (
        <>
          <div className="connection-info">
            <span className="server">{config.serverUrl}</span>
            <span className="user">@{config.username}</span>
          </div>

          <div className="quota">
            <span className="label">Storage</span>
            <div className="progress-bar">
              <div
                className="fill"
                style={{ width: `${(config.quotaUsed / config.quotaTotal) * 100}%` }}
              />
            </div>
            <span className="usage">
              {formatBytes(config.quotaUsed)} / {formatBytes(config.quotaTotal)}
            </span>
          </div>

          <div className="features">
            <h3>Enabled Features</h3>
            {Object.entries(config.features).map(([key, enabled]) => (
              <label key={key} className="feature-toggle">
                <input
                  type="checkbox"
                  checked={enabled}
                  onChange={e => updateFeatures({ [key]: e.target.checked })}
                />
                {key.charAt(0).toUpperCase() + key.slice(1)}
              </label>
            ))}
          </div>

          <button onClick={syncNow} className="sync-btn">Sync Now</button>
        </>
      )}

      {showConnect && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Connect to Nextcloud</h3>
            <div className="form-group">
              <label>Server URL</label>
              <input
                type="url"
                value={serverUrl}
                onChange={e => setServerUrl(e.target.value)}
                placeholder="https://nextcloud.example.com"
              />
            </div>
            <div className="form-group">
              <label>Username</label>
              <input
                type="text"
                value={username}
                onChange={e => setUsername(e.target.value)}
              />
            </div>
            <div className="form-group">
              <label>Password / App Token</label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
              />
            </div>
            <div className="modal-actions">
              <button onClick={() => setShowConnect(false)}>Cancel</button>
              <button onClick={handleConnect}>Connect</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MatrixPanel() {
  const { config, connect, updateBridgeSettings } = useMatrix();
  const [showConnect, setShowConnect] = useState(false);
  const [homeserver, setHomeserver] = useState('');
  const [userId, setUserId] = useState('');
  const [accessToken, setAccessToken] = useState('');

  const handleConnect = async () => {
    const result = await connect(homeserver, userId, accessToken);
    if (result.success) {
      setShowConnect(false);
    }
  };

  if (!config) {
    return <div className="loading">Loading Matrix config...</div>;
  }

  return (
    <div className="matrix-panel">
      <header>
        <h2>Matrix</h2>
        {config.connected ? (
          <span className="status connected">Connected</span>
        ) : (
          <button onClick={() => setShowConnect(true)}>Connect</button>
        )}
      </header>

      {config.connected && (
        <>
          <div className="connection-info">
            <span className="homeserver">{config.homeserver}</span>
            <span className="user">{config.userId}</span>
          </div>

          <div className="rooms">
            <h3>Rooms ({config.rooms.length})</h3>
            <div className="room-list">
              {config.rooms.slice(0, 5).map(room => (
                <div key={room.roomId} className="room-item">
                  <span className="name">{room.name}</span>
                  {room.unreadCount > 0 && (
                    <span className="unread">{room.unreadCount}</span>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="bridge-settings">
            <h3>Email Bridge</h3>
            <label>
              <input
                type="checkbox"
                checked={config.emailToMatrix}
                onChange={e => updateBridgeSettings({ emailToMatrix: e.target.checked })}
              />
              Forward emails to Matrix
            </label>
            <label>
              <input
                type="checkbox"
                checked={config.matrixToEmail}
                onChange={e => updateBridgeSettings({ matrixToEmail: e.target.checked })}
              />
              Send Matrix messages as email
            </label>
          </div>
        </>
      )}

      {showConnect && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Connect to Matrix</h3>
            <div className="form-group">
              <label>Homeserver</label>
              <input
                type="url"
                value={homeserver}
                onChange={e => setHomeserver(e.target.value)}
                placeholder="https://matrix.org"
              />
            </div>
            <div className="form-group">
              <label>User ID</label>
              <input
                type="text"
                value={userId}
                onChange={e => setUserId(e.target.value)}
                placeholder="@user:matrix.org"
              />
            </div>
            <div className="form-group">
              <label>Access Token</label>
              <input
                type="password"
                value={accessToken}
                onChange={e => setAccessToken(e.target.value)}
              />
            </div>
            <div className="modal-actions">
              <button onClick={() => setShowConnect(false)}>Cancel</button>
              <button onClick={handleConnect}>Connect</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function JitsiPanel() {
  const { config, updateConfig, createMeeting } = useJitsi();
  const [meetingSubject, setMeetingSubject] = useState('');

  if (!config) {
    return <div className="loading">Loading Jitsi config...</div>;
  }

  const handleCreateMeeting = async () => {
    if (meetingSubject) {
      const meeting = await createMeeting(meetingSubject, []);
      window.open(meeting.url, '_blank');
      setMeetingSubject('');
    }
  };

  return (
    <div className="jitsi-panel">
      <header>
        <h2>Jitsi Meet</h2>
        <span className={`status ${config.connected ? 'connected' : 'disconnected'}`}>
          {config.connected ? 'Connected' : 'Not configured'}
        </span>
      </header>

      <div className="form-group">
        <label>Server URL</label>
        <input
          type="url"
          value={config.serverUrl}
          onChange={e => updateConfig({ serverUrl: e.target.value })}
          placeholder="https://meet.jit.si"
        />
      </div>

      <div className="settings">
        <label>
          <input
            type="checkbox"
            checked={config.autoCreateMeetings}
            onChange={e => updateConfig({ autoCreateMeetings: e.target.checked })}
          />
          Auto-create meetings from calendar events
        </label>
        <label>
          <input
            type="checkbox"
            checked={config.enableLobby}
            onChange={e => updateConfig({ enableLobby: e.target.checked })}
          />
          Enable lobby by default
        </label>
        <label>
          <input
            type="checkbox"
            checked={config.enableRecording}
            onChange={e => updateConfig({ enableRecording: e.target.checked })}
          />
          Enable recording
        </label>
      </div>

      <div className="quick-meeting">
        <h3>Quick Meeting</h3>
        <input
          type="text"
          value={meetingSubject}
          onChange={e => setMeetingSubject(e.target.value)}
          placeholder="Meeting subject"
        />
        <button onClick={handleCreateMeeting}>Create & Join</button>
      </div>
    </div>
  );
}

function VikunjaPanel() {
  const { config, connect, createTaskFromEmail } = useVikunja();

  if (!config) {
    return <div className="loading">Loading Vikunja config...</div>;
  }

  return (
    <div className="vikunja-panel">
      <header>
        <h2>Vikunja</h2>
        <span className={`status ${config.connected ? 'connected' : 'disconnected'}`}>
          {config.connected ? 'Connected' : 'Not connected'}
        </span>
      </header>

      {config.connected && (
        <>
          <div className="connection-info">
            <span className="server">{config.serverUrl}</span>
          </div>

          <div className="projects">
            <h3>Projects</h3>
            <div className="project-list">
              {config.projects.map(project => (
                <div key={project.id} className="project-item">
                  <span className="title">{project.title}</span>
                  <span className="count">{project.taskCount} tasks</span>
                </div>
              ))}
            </div>
          </div>

          <div className="email-integration">
            <label>
              <input
                type="checkbox"
                checked={config.emailToTask}
                readOnly
              />
              Convert emails to tasks
            </label>
            <p className="hint">
              Forward emails to tasks@vikunja or use the "Create Task" action
            </p>
          </div>
        </>
      )}
    </div>
  );
}

function DAVPanel() {
  const { config, connect, syncNow, updateSyncInterval } = useDAV();
  const [showConnect, setShowConnect] = useState(false);
  const [caldavUrl, setCaldavUrl] = useState('');
  const [carddavUrl, setCarddavUrl] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleConnect = async () => {
    const result = await connect(caldavUrl, carddavUrl, username, password);
    if (result.success) {
      setShowConnect(false);
    }
  };

  if (!config) {
    return <div className="loading">Loading CalDAV/CardDAV config...</div>;
  }

  return (
    <div className="dav-panel">
      <header>
        <h2>CalDAV / CardDAV</h2>
        {config.connected ? (
          <span className="status connected">Connected</span>
        ) : (
          <button onClick={() => setShowConnect(true)}>Connect</button>
        )}
      </header>

      {config.connected && (
        <>
          <div className="calendars">
            <h3>Calendars ({config.calendars.length})</h3>
            <div className="calendar-list">
              {config.calendars.map(cal => (
                <div key={cal.id} className="calendar-item">
                  <span
                    className="color-dot"
                    style={{ backgroundColor: cal.color }}
                  />
                  <span className="name">{cal.name}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="address-books">
            <h3>Address Books ({config.addressBooks.length})</h3>
            <div className="addressbook-list">
              {config.addressBooks.map(ab => (
                <div key={ab.id} className="addressbook-item">
                  <span className="name">{ab.name}</span>
                  <span className="count">{ab.contactCount} contacts</span>
                </div>
              ))}
            </div>
          </div>

          <div className="sync-settings">
            <label>Sync interval (minutes)</label>
            <select
              value={config.syncInterval}
              onChange={e => updateSyncInterval(parseInt(e.target.value))}
            >
              <option value={5}>5 minutes</option>
              <option value={15}>15 minutes</option>
              <option value={30}>30 minutes</option>
              <option value={60}>1 hour</option>
            </select>
            <button onClick={syncNow}>Sync Now</button>
          </div>
        </>
      )}

      {showConnect && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Connect CalDAV/CardDAV</h3>
            <div className="form-group">
              <label>CalDAV URL</label>
              <input
                type="url"
                value={caldavUrl}
                onChange={e => setCaldavUrl(e.target.value)}
                placeholder="https://example.com/dav/calendars/"
              />
            </div>
            <div className="form-group">
              <label>CardDAV URL</label>
              <input
                type="url"
                value={carddavUrl}
                onChange={e => setCarddavUrl(e.target.value)}
                placeholder="https://example.com/dav/contacts/"
              />
            </div>
            <div className="form-group">
              <label>Username</label>
              <input
                type="text"
                value={username}
                onChange={e => setUsername(e.target.value)}
              />
            </div>
            <div className="form-group">
              <label>Password</label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
              />
            </div>
            <div className="modal-actions">
              <button onClick={() => setShowConnect(false)}>Cancel</button>
              <button onClick={handleConnect}>Connect</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function OIDCPanel() {
  const { providers, addProvider, removeProvider } = useOIDC();
  const [showAdd, setShowAdd] = useState(false);
  const [name, setName] = useState('');
  const [issuer, setIssuer] = useState('');
  const [clientId, setClientId] = useState('');

  const handleAdd = async () => {
    if (name && issuer && clientId) {
      await addProvider({ name, issuer, clientId });
      setShowAdd(false);
      setName('');
      setIssuer('');
      setClientId('');
    }
  };

  return (
    <div className="oidc-panel">
      <header>
        <h2>SSO / OIDC Providers</h2>
        <button onClick={() => setShowAdd(true)}>Add Provider</button>
      </header>

      <p className="description">
        Connect identity providers like Keycloak, Authentik, or any OIDC-compliant service.
      </p>

      {providers.length === 0 ? (
        <div className="empty">No SSO providers configured</div>
      ) : (
        <div className="provider-list">
          {providers.map(provider => (
            <div key={provider.id} className="provider-item">
              <div className="info">
                <span className="name">{provider.name}</span>
                <span className="issuer">{provider.issuer}</span>
                <span className="users">{provider.userCount} users</span>
              </div>
              <span className={`status ${provider.connected ? 'connected' : 'error'}`}>
                {provider.connected ? 'Active' : 'Error'}
              </span>
              <button onClick={() => removeProvider(provider.id)} className="danger">
                Remove
              </button>
            </div>
          ))}
        </div>
      )}

      {showAdd && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Add OIDC Provider</h3>
            <div className="form-group">
              <label>Name</label>
              <input
                type="text"
                value={name}
                onChange={e => setName(e.target.value)}
                placeholder="e.g., Company SSO"
              />
            </div>
            <div className="form-group">
              <label>Issuer URL</label>
              <input
                type="url"
                value={issuer}
                onChange={e => setIssuer(e.target.value)}
                placeholder="https://auth.example.com/realms/main"
              />
            </div>
            <div className="form-group">
              <label>Client ID</label>
              <input
                type="text"
                value={clientId}
                onChange={e => setClientId(e.target.value)}
              />
            </div>
            <div className="modal-actions">
              <button onClick={() => setShowAdd(false)}>Cancel</button>
              <button onClick={handleAdd}>Add Provider</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MeilisearchPanel() {
  const { config, connect, reindex } = useMeilisearch();

  if (!config) {
    return <div className="loading">Loading Meilisearch config...</div>;
  }

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  return (
    <div className="meilisearch-panel">
      <header>
        <h2>Meilisearch</h2>
        <span className={`status ${config.connected ? 'connected' : 'disconnected'}`}>
          {config.connected ? 'Connected' : 'Not connected'}
        </span>
      </header>

      {config.connected && (
        <>
          <div className="stats">
            <div className="stat">
              <span className="value">{config.indexedEmails.toLocaleString()}</span>
              <span className="label">Indexed Emails</span>
            </div>
            <div className="stat">
              <span className="value">{formatBytes(config.indexSize)}</span>
              <span className="label">Index Size</span>
            </div>
          </div>

          {config.lastIndexed && (
            <span className="last-indexed">
              Last indexed: {new Date(config.lastIndexed).toLocaleString()}
            </span>
          )}

          <div className="actions">
            <button onClick={reindex}>Reindex All</button>
          </div>
        </>
      )}
    </div>
  );
}

function N8nPanel() {
  const { config, connect, triggerWorkflow } = useN8n();

  if (!config) {
    return <div className="loading">Loading n8n config...</div>;
  }

  return (
    <div className="n8n-panel">
      <header>
        <h2>n8n Automation</h2>
        <span className={`status ${config.connected ? 'connected' : 'disconnected'}`}>
          {config.connected ? 'Connected' : 'Not connected'}
        </span>
      </header>

      {config.connected && (
        <>
          <div className="connection-info">
            <span className="server">{config.serverUrl}</span>
          </div>

          <div className="workflows">
            <h3>Workflows ({config.workflows.length})</h3>
            <div className="workflow-list">
              {config.workflows.map(workflow => (
                <div key={workflow.id} className="workflow-item">
                  <span className={`status-dot ${workflow.active ? 'active' : 'inactive'}`} />
                  <span className="name">{workflow.name}</span>
                  <button onClick={() => triggerWorkflow(workflow.id, {})}>
                    Trigger
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="webhook-info">
            <h3>Webhook Secret</h3>
            <code>{config.webhookSecret}</code>
            <p className="hint">Use this secret to verify incoming webhooks from n8n</p>
          </div>
        </>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

interface FossIntegrationsProps {
  initialTab?: string;
}

export function FossIntegrations({ initialTab = 'all' }: FossIntegrationsProps) {
  const [activeTab, setActiveTab] = useState(initialTab);

  const tabs = [
    { id: 'all', label: 'All', icon: '🔗' },
    { id: 'nextcloud', label: 'Nextcloud', icon: '☁️' },
    { id: 'matrix', label: 'Matrix', icon: '💬' },
    { id: 'jitsi', label: 'Jitsi', icon: '📹' },
    { id: 'vikunja', label: 'Vikunja', icon: '✅' },
    { id: 'dav', label: 'CalDAV', icon: '📅' },
    { id: 'oidc', label: 'SSO', icon: '🔐' },
    { id: 'search', label: 'Search', icon: '🔍' },
    { id: 'n8n', label: 'n8n', icon: '⚡' },
  ];

  return (
    <div className="foss-integrations">
      <nav className="foss-tabs">
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

      <div className="foss-content">
        {activeTab === 'all' && <IntegrationsList />}
        {activeTab === 'nextcloud' && <NextcloudPanel />}
        {activeTab === 'matrix' && <MatrixPanel />}
        {activeTab === 'jitsi' && <JitsiPanel />}
        {activeTab === 'vikunja' && <VikunjaPanel />}
        {activeTab === 'dav' && <DAVPanel />}
        {activeTab === 'oidc' && <OIDCPanel />}
        {activeTab === 'search' && <MeilisearchPanel />}
        {activeTab === 'n8n' && <N8nPanel />}
      </div>
    </div>
  );
}

export default FossIntegrations;
