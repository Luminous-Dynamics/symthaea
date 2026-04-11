// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Track Z: Plugin/Extension System
 *
 * Plugin marketplace, installation, configuration, and management.
 * Supports email plugins, integrations, UI extensions, and trust plugins.
 */

import React, { useState, useEffect, useCallback } from 'react';

// ============================================================================
// Types
// ============================================================================

interface Plugin {
  id: string;
  name: string;
  version: string;
  description: string;
  author: string;
  license: string;
  homepage?: string;
  repository?: string;
  icon?: string;
  category: PluginCategory;
  capabilities: PluginCapability[];
  permissions: PluginPermission[];
  status: PluginStatus;
  installed: boolean;
  enabled: boolean;
  hasUpdate: boolean;
  latestVersion?: string;
  rating: number;
  downloads: number;
}

type PluginCategory = 'Email' | 'Integration' | 'Trust' | 'UI' | 'Security' | 'Productivity' | 'AI';
type PluginCapability = 'EmailProcessing' | 'Integration' | 'Trust' | 'UI' | 'Storage' | 'Notifications' | 'Scheduling';
type PluginPermission = 'ReadEmails' | 'WriteEmails' | 'ReadContacts' | 'WriteContacts' | 'ExternalNetwork' | 'Storage';
type PluginStatus = 'Active' | 'Inactive' | 'Error' | 'Updating';

interface PluginConfig {
  pluginId: string;
  settings: Record<string, PluginSetting>;
  enabled: boolean;
}

interface PluginSetting {
  key: string;
  label: string;
  description: string;
  type: 'string' | 'number' | 'boolean' | 'select' | 'secret';
  value: unknown;
  default: unknown;
  options?: { label: string; value: string }[];
  required: boolean;
}

interface PluginHealth {
  pluginId: string;
  healthy: boolean;
  message?: string;
  lastCheck: string;
  metrics: Record<string, number>;
}

interface MarketplacePlugin extends Plugin {
  screenshots: string[];
  longDescription: string;
  changelog: string;
  reviews: PluginReview[];
}

interface PluginReview {
  id: string;
  author: string;
  rating: number;
  comment: string;
  date: string;
}

// ============================================================================
// Hooks
// ============================================================================

function usePlugins() {
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/plugins')
      .then(res => res.json())
      .then(setPlugins)
      .finally(() => setLoading(false));
  }, []);

  const install = useCallback(async (pluginId: string) => {
    const response = await fetch(`/api/plugins/${pluginId}/install`, { method: 'POST' });
    const result = await response.json();
    if (result.success) {
      setPlugins(prev => prev.map(p =>
        p.id === pluginId ? { ...p, installed: true, status: 'Active' } : p
      ));
    }
    return result;
  }, []);

  const uninstall = useCallback(async (pluginId: string) => {
    await fetch(`/api/plugins/${pluginId}`, { method: 'DELETE' });
    setPlugins(prev => prev.map(p =>
      p.id === pluginId ? { ...p, installed: false, enabled: false } : p
    ));
  }, []);

  const enable = useCallback(async (pluginId: string) => {
    await fetch(`/api/plugins/${pluginId}/enable`, { method: 'POST' });
    setPlugins(prev => prev.map(p =>
      p.id === pluginId ? { ...p, enabled: true, status: 'Active' } : p
    ));
  }, []);

  const disable = useCallback(async (pluginId: string) => {
    await fetch(`/api/plugins/${pluginId}/disable`, { method: 'POST' });
    setPlugins(prev => prev.map(p =>
      p.id === pluginId ? { ...p, enabled: false, status: 'Inactive' } : p
    ));
  }, []);

  const update = useCallback(async (pluginId: string) => {
    setPlugins(prev => prev.map(p =>
      p.id === pluginId ? { ...p, status: 'Updating' } : p
    ));
    const response = await fetch(`/api/plugins/${pluginId}/update`, { method: 'POST' });
    const result = await response.json();
    if (result.success) {
      setPlugins(prev => prev.map(p =>
        p.id === pluginId ? { ...p, version: result.version, hasUpdate: false, status: 'Active' } : p
      ));
    }
    return result;
  }, []);

  return { plugins, loading, install, uninstall, enable, disable, update };
}

function usePluginConfig(pluginId: string) {
  const [config, setConfig] = useState<PluginConfig | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (pluginId) {
      fetch(`/api/plugins/${pluginId}/config`)
        .then(res => res.json())
        .then(setConfig)
        .finally(() => setLoading(false));
    }
  }, [pluginId]);

  const updateSetting = useCallback(async (key: string, value: unknown) => {
    await fetch(`/api/plugins/${pluginId}/config`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: value }),
    });
    setConfig(prev => prev ? {
      ...prev,
      settings: {
        ...prev.settings,
        [key]: { ...prev.settings[key], value },
      },
    } : null);
  }, [pluginId]);

  return { config, loading, updateSetting };
}

function useMarketplace() {
  const [plugins, setPlugins] = useState<MarketplacePlugin[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [category, setCategory] = useState<PluginCategory | 'All'>('All');

  useEffect(() => {
    const params = new URLSearchParams();
    if (searchQuery) params.set('q', searchQuery);
    if (category !== 'All') params.set('category', category);

    fetch(`/api/plugins/marketplace?${params}`)
      .then(res => res.json())
      .then(setPlugins)
      .finally(() => setLoading(false));
  }, [searchQuery, category]);

  return { plugins, loading, searchQuery, setSearchQuery, category, setCategory };
}

function usePluginHealth() {
  const [health, setHealth] = useState<Record<string, PluginHealth>>({});

  useEffect(() => {
    fetch('/api/plugins/health')
      .then(res => res.json())
      .then((data: PluginHealth[]) => {
        const healthMap: Record<string, PluginHealth> = {};
        data.forEach(h => { healthMap[h.pluginId] = h; });
        setHealth(healthMap);
      });
  }, []);

  return health;
}

// ============================================================================
// Components
// ============================================================================

function PluginCard({ plugin, onInstall, onUninstall, onEnable, onDisable, onUpdate, onConfigure }: {
  plugin: Plugin;
  onInstall: () => void;
  onUninstall: () => void;
  onEnable: () => void;
  onDisable: () => void;
  onUpdate: () => void;
  onConfigure: () => void;
}) {
  const categoryIcons: Record<PluginCategory, string> = {
    Email: '📧',
    Integration: '🔗',
    Trust: '🤝',
    UI: '🎨',
    Security: '🔒',
    Productivity: '⚡',
    AI: '🤖',
  };

  const statusColors: Record<PluginStatus, string> = {
    Active: '#22c55e',
    Inactive: '#6b7280',
    Error: '#ef4444',
    Updating: '#eab308',
  };

  return (
    <div className={`plugin-card ${plugin.installed ? 'installed' : ''}`}>
      <div className="plugin-header">
        <div className="plugin-icon">
          {plugin.icon ? <img src={plugin.icon} alt="" /> : categoryIcons[plugin.category]}
        </div>
        <div className="plugin-info">
          <h3>{plugin.name}</h3>
          <span className="version">v{plugin.version}</span>
          {plugin.hasUpdate && (
            <span className="update-badge">Update available</span>
          )}
        </div>
        {plugin.installed && (
          <span
            className="status-indicator"
            style={{ backgroundColor: statusColors[plugin.status] }}
            title={plugin.status}
          />
        )}
      </div>

      <p className="description">{plugin.description}</p>

      <div className="plugin-meta">
        <span className="author">by {plugin.author}</span>
        <span className="category">{plugin.category}</span>
        <div className="rating">
          {'★'.repeat(Math.round(plugin.rating))}{'☆'.repeat(5 - Math.round(plugin.rating))}
          <span className="count">({plugin.downloads.toLocaleString()})</span>
        </div>
      </div>

      <div className="capabilities">
        {plugin.capabilities.slice(0, 3).map(cap => (
          <span key={cap} className="capability">{cap}</span>
        ))}
      </div>

      <div className="plugin-actions">
        {!plugin.installed ? (
          <button onClick={onInstall} className="primary">Install</button>
        ) : (
          <>
            {plugin.hasUpdate && (
              <button onClick={onUpdate} className="update">Update</button>
            )}
            {plugin.enabled ? (
              <button onClick={onDisable}>Disable</button>
            ) : (
              <button onClick={onEnable} className="primary">Enable</button>
            )}
            <button onClick={onConfigure}>Configure</button>
            <button onClick={onUninstall} className="danger">Uninstall</button>
          </>
        )}
      </div>
    </div>
  );
}

function PluginConfigModal({ plugin, onClose }: {
  plugin: Plugin;
  onClose: () => void;
}) {
  const { config, loading, updateSetting } = usePluginConfig(plugin.id);

  if (loading) {
    return (
      <div className="modal-overlay">
        <div className="modal">
          <div className="loading">Loading configuration...</div>
        </div>
      </div>
    );
  }

  if (!config) {
    return null;
  }

  return (
    <div className="modal-overlay">
      <div className="modal plugin-config-modal">
        <header>
          <h2>Configure {plugin.name}</h2>
          <button onClick={onClose} className="close-btn">×</button>
        </header>

        <div className="config-form">
          {Object.values(config.settings).map(setting => (
            <div key={setting.key} className="setting-item">
              <label>
                {setting.label}
                {setting.required && <span className="required">*</span>}
              </label>
              <p className="hint">{setting.description}</p>

              {setting.type === 'string' && (
                <input
                  type="text"
                  value={setting.value as string}
                  onChange={e => updateSetting(setting.key, e.target.value)}
                />
              )}

              {setting.type === 'secret' && (
                <input
                  type="password"
                  value={setting.value as string}
                  onChange={e => updateSetting(setting.key, e.target.value)}
                />
              )}

              {setting.type === 'number' && (
                <input
                  type="number"
                  value={setting.value as number}
                  onChange={e => updateSetting(setting.key, parseFloat(e.target.value))}
                />
              )}

              {setting.type === 'boolean' && (
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={setting.value as boolean}
                    onChange={e => updateSetting(setting.key, e.target.checked)}
                  />
                  <span className="slider" />
                </label>
              )}

              {setting.type === 'select' && setting.options && (
                <select
                  value={setting.value as string}
                  onChange={e => updateSetting(setting.key, e.target.value)}
                >
                  {setting.options.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              )}
            </div>
          ))}
        </div>

        <div className="modal-actions">
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}

function InstalledPlugins() {
  const { plugins, loading, uninstall, enable, disable, update } = usePlugins();
  const health = usePluginHealth();
  const [configPlugin, setConfigPlugin] = useState<Plugin | null>(null);

  const installedPlugins = plugins.filter(p => p.installed);

  if (loading) {
    return <div className="loading">Loading plugins...</div>;
  }

  return (
    <div className="installed-plugins">
      <header>
        <h2>Installed Plugins</h2>
        <span className="count">{installedPlugins.length} installed</span>
      </header>

      {installedPlugins.length === 0 ? (
        <div className="empty-state">
          <p>No plugins installed yet.</p>
          <p>Visit the Marketplace to discover plugins.</p>
        </div>
      ) : (
        <div className="plugins-grid">
          {installedPlugins.map(plugin => (
            <PluginCard
              key={plugin.id}
              plugin={plugin}
              onInstall={() => {}}
              onUninstall={() => uninstall(plugin.id)}
              onEnable={() => enable(plugin.id)}
              onDisable={() => disable(plugin.id)}
              onUpdate={() => update(plugin.id)}
              onConfigure={() => setConfigPlugin(plugin)}
            />
          ))}
        </div>
      )}

      {configPlugin && (
        <PluginConfigModal
          plugin={configPlugin}
          onClose={() => setConfigPlugin(null)}
        />
      )}
    </div>
  );
}

function Marketplace() {
  const { plugins, loading, searchQuery, setSearchQuery, category, setCategory } = useMarketplace();
  const { install } = usePlugins();
  const [selectedPlugin, setSelectedPlugin] = useState<MarketplacePlugin | null>(null);

  const categories: (PluginCategory | 'All')[] = [
    'All', 'Email', 'Integration', 'Trust', 'UI', 'Security', 'Productivity', 'AI'
  ];

  return (
    <div className="marketplace">
      <header>
        <h2>Plugin Marketplace</h2>
        <div className="search-bar">
          <input
            type="text"
            placeholder="Search plugins..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
          />
        </div>
      </header>

      <div className="category-tabs">
        {categories.map(cat => (
          <button
            key={cat}
            className={category === cat ? 'active' : ''}
            onClick={() => setCategory(cat)}
          >
            {cat}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="loading">Loading marketplace...</div>
      ) : (
        <div className="plugins-grid">
          {plugins.map(plugin => (
            <PluginCard
              key={plugin.id}
              plugin={plugin}
              onInstall={() => install(plugin.id)}
              onUninstall={() => {}}
              onEnable={() => {}}
              onDisable={() => {}}
              onUpdate={() => {}}
              onConfigure={() => setSelectedPlugin(plugin)}
            />
          ))}
        </div>
      )}

      {selectedPlugin && (
        <PluginDetailModal
          plugin={selectedPlugin}
          onClose={() => setSelectedPlugin(null)}
          onInstall={() => install(selectedPlugin.id)}
        />
      )}
    </div>
  );
}

function PluginDetailModal({ plugin, onClose, onInstall }: {
  plugin: MarketplacePlugin;
  onClose: () => void;
  onInstall: () => void;
}) {
  const [activeTab, setActiveTab] = useState<'overview' | 'reviews' | 'changelog'>('overview');

  return (
    <div className="modal-overlay">
      <div className="modal plugin-detail-modal">
        <header>
          <h2>{plugin.name}</h2>
          <button onClick={onClose} className="close-btn">×</button>
        </header>

        <div className="tabs">
          <button
            className={activeTab === 'overview' ? 'active' : ''}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button
            className={activeTab === 'reviews' ? 'active' : ''}
            onClick={() => setActiveTab('reviews')}
          >
            Reviews ({plugin.reviews.length})
          </button>
          <button
            className={activeTab === 'changelog' ? 'active' : ''}
            onClick={() => setActiveTab('changelog')}
          >
            Changelog
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'overview' && (
            <div className="overview">
              <div className="screenshots">
                {plugin.screenshots.map((url, i) => (
                  <img key={i} src={url} alt={`Screenshot ${i + 1}`} />
                ))}
              </div>
              <div className="long-description">{plugin.longDescription}</div>
              <div className="permissions">
                <h4>Required Permissions</h4>
                <ul>
                  {plugin.permissions.map(perm => (
                    <li key={perm}>{perm}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {activeTab === 'reviews' && (
            <div className="reviews">
              {plugin.reviews.map(review => (
                <div key={review.id} className="review">
                  <div className="review-header">
                    <span className="author">{review.author}</span>
                    <span className="rating">{'★'.repeat(review.rating)}</span>
                    <span className="date">{new Date(review.date).toLocaleDateString()}</span>
                  </div>
                  <p>{review.comment}</p>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'changelog' && (
            <div className="changelog">
              <pre>{plugin.changelog}</pre>
            </div>
          )}
        </div>

        <div className="modal-actions">
          <button onClick={onClose}>Close</button>
          {!plugin.installed && (
            <button onClick={onInstall} className="primary">Install</button>
          )}
        </div>
      </div>
    </div>
  );
}

function DeveloperTools() {
  const [manifestUrl, setManifestUrl] = useState('');
  const [loading, setLoading] = useState(false);

  const installFromUrl = async () => {
    setLoading(true);
    try {
      await fetch('/api/plugins/install-from-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ manifestUrl }),
      });
      setManifestUrl('');
      alert('Plugin installed successfully');
    } catch (error) {
      alert('Failed to install plugin');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="developer-tools">
      <header>
        <h2>Developer Tools</h2>
      </header>

      <div className="install-from-url">
        <h3>Install from URL</h3>
        <p>Install a plugin directly from a manifest URL (for development)</p>
        <div className="form-row">
          <input
            type="url"
            placeholder="https://example.com/plugin/manifest.json"
            value={manifestUrl}
            onChange={e => setManifestUrl(e.target.value)}
          />
          <button onClick={installFromUrl} disabled={loading || !manifestUrl}>
            {loading ? 'Installing...' : 'Install'}
          </button>
        </div>
      </div>

      <div className="create-plugin">
        <h3>Create a Plugin</h3>
        <p>Learn how to create plugins for Mycelix Mail</p>
        <a href="/docs/developer/plugins" className="docs-link">
          View Plugin Development Guide
        </a>
      </div>

      <div className="plugin-template">
        <h3>Plugin Template</h3>
        <p>Download a starter template for plugin development</p>
        <button onClick={() => window.open('/api/plugins/template', '_blank')}>
          Download Template
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

interface PluginManagerProps {
  initialTab?: string;
}

export function PluginManager({ initialTab = 'installed' }: PluginManagerProps) {
  const [activeTab, setActiveTab] = useState(initialTab);

  const tabs = [
    { id: 'installed', label: 'Installed', icon: '📦' },
    { id: 'marketplace', label: 'Marketplace', icon: '🛒' },
    { id: 'developer', label: 'Developer', icon: '🛠️' },
  ];

  return (
    <div className="plugin-manager">
      <nav className="plugin-tabs">
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

      <div className="plugin-content">
        {activeTab === 'installed' && <InstalledPlugins />}
        {activeTab === 'marketplace' && <Marketplace />}
        {activeTab === 'developer' && <DeveloperTools />}
      </div>
    </div>
  );
}

export default PluginManager;
