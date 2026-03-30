// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Developer Platform Components
 *
 * Provides plugin management, webhook configuration, custom actions,
 * OAuth app management, and marketplace browsing.
 */

import React, { useState, useEffect } from 'react';

// ============================================================================
// Types
// ============================================================================

interface Plugin {
  id: string;
  name: string;
  slug: string;
  version: string;
  description: string;
  author: PluginAuthor;
  iconUrl?: string;
  permissions: string[];
  status: 'active' | 'disabled' | 'pending_review';
  installed: boolean;
  installCount: number;
  rating: number;
}

interface PluginAuthor {
  name: string;
  email?: string;
  verified: boolean;
}

interface Webhook {
  id: string;
  name: string;
  url: string;
  events: string[];
  enabled: boolean;
  lastTriggered?: string;
  failureCount: number;
}

interface WebhookDelivery {
  id: string;
  event: string;
  responseStatus?: number;
  success: boolean;
  deliveredAt: string;
}

interface CustomAction {
  id: string;
  name: string;
  description?: string;
  icon?: string;
  script: string;
  location: 'toolbar' | 'context_menu' | 'compose_toolbar';
  shortcut?: string;
  enabled: boolean;
}

interface OAuthApp {
  id: string;
  name: string;
  description?: string;
  clientId: string;
  scopes: string[];
  authorizedAt: string;
  lastUsed?: string;
}

// ============================================================================
// Plugin Marketplace
// ============================================================================

interface PluginMarketplaceProps {
  onInstall: (pluginId: string) => void;
  onUninstall: (pluginId: string) => void;
}

export function PluginMarketplace({ onInstall, onUninstall }: PluginMarketplaceProps) {
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [category, setCategory] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'popular' | 'rating' | 'recent'>('popular');
  const [selectedPlugin, setSelectedPlugin] = useState<Plugin | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPlugins();
  }, [category, sortBy]);

  const fetchPlugins = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/marketplace/plugins?category=${category}&sort=${sortBy}`);
      const data = await response.json();
      setPlugins(data.plugins);
    } catch (error) {
      console.error('Failed to fetch plugins:', error);
    }
    setLoading(false);
  };

  const filteredPlugins = plugins.filter(plugin =>
    plugin.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    plugin.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const categories = [
    { id: 'all', name: 'All' },
    { id: 'productivity', name: 'Productivity' },
    { id: 'integrations', name: 'Integrations' },
    { id: 'security', name: 'Security' },
    { id: 'ai', name: 'AI & Automation' },
    { id: 'customization', name: 'Customization' },
  ];

  return (
    <div className="plugin-marketplace">
      <div className="marketplace-header">
        <h2>Plugin Marketplace</h2>
        <div className="search-bar">
          <SearchIcon />
          <input
            type="text"
            placeholder="Search plugins..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      <div className="marketplace-filters">
        <div className="categories">
          {categories.map(cat => (
            <button
              key={cat.id}
              className={category === cat.id ? 'active' : ''}
              onClick={() => setCategory(cat.id)}
            >
              {cat.name}
            </button>
          ))}
        </div>
        <div className="sort-options">
          <label>Sort by:</label>
          <select value={sortBy} onChange={e => setSortBy(e.target.value as any)}>
            <option value="popular">Most Popular</option>
            <option value="rating">Highest Rated</option>
            <option value="recent">Recently Added</option>
          </select>
        </div>
      </div>

      {loading ? (
        <div className="loading">Loading plugins...</div>
      ) : (
        <div className="plugin-grid">
          {filteredPlugins.map(plugin => (
            <PluginCard
              key={plugin.id}
              plugin={plugin}
              onSelect={() => setSelectedPlugin(plugin)}
              onInstall={() => onInstall(plugin.id)}
              onUninstall={() => onUninstall(plugin.id)}
            />
          ))}
        </div>
      )}

      {selectedPlugin && (
        <PluginDetailModal
          plugin={selectedPlugin}
          onClose={() => setSelectedPlugin(null)}
          onInstall={() => {
            onInstall(selectedPlugin.id);
            setSelectedPlugin(null);
          }}
          onUninstall={() => {
            onUninstall(selectedPlugin.id);
            setSelectedPlugin(null);
          }}
        />
      )}
    </div>
  );
}

interface PluginCardProps {
  plugin: Plugin;
  onSelect: () => void;
  onInstall: () => void;
  onUninstall: () => void;
}

function PluginCard({ plugin, onSelect, onInstall, onUninstall }: PluginCardProps) {
  return (
    <div className="plugin-card" onClick={onSelect}>
      <div className="plugin-icon">
        {plugin.iconUrl ? (
          <img src={plugin.iconUrl} alt={plugin.name} />
        ) : (
          <PluginIcon />
        )}
      </div>
      <div className="plugin-info">
        <h3>{plugin.name}</h3>
        <p className="author">
          by {plugin.author.name}
          {plugin.author.verified && <VerifiedIcon />}
        </p>
        <p className="description">{plugin.description}</p>
        <div className="plugin-stats">
          <span className="installs">{formatNumber(plugin.installCount)} installs</span>
          <span className="rating">
            <StarIcon /> {plugin.rating.toFixed(1)}
          </span>
        </div>
      </div>
      <div className="plugin-actions" onClick={e => e.stopPropagation()}>
        {plugin.installed ? (
          <button className="uninstall" onClick={onUninstall}>
            Uninstall
          </button>
        ) : (
          <button className="install" onClick={onInstall}>
            Install
          </button>
        )}
      </div>
    </div>
  );
}

interface PluginDetailModalProps {
  plugin: Plugin;
  onClose: () => void;
  onInstall: () => void;
  onUninstall: () => void;
}

function PluginDetailModal({ plugin, onClose, onInstall, onUninstall }: PluginDetailModalProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'permissions' | 'reviews'>('overview');

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal plugin-detail-modal" onClick={e => e.stopPropagation()}>
        <button className="close-button" onClick={onClose}>
          <CloseIcon />
        </button>

        <div className="plugin-header">
          <div className="plugin-icon-large">
            {plugin.iconUrl ? (
              <img src={plugin.iconUrl} alt={plugin.name} />
            ) : (
              <PluginIcon />
            )}
          </div>
          <div className="plugin-title">
            <h2>{plugin.name}</h2>
            <p className="author">
              by {plugin.author.name}
              {plugin.author.verified && <VerifiedIcon />}
            </p>
            <p className="version">v{plugin.version}</p>
          </div>
          <div className="plugin-action">
            {plugin.installed ? (
              <button className="uninstall" onClick={onUninstall}>
                Uninstall
              </button>
            ) : (
              <button className="install primary" onClick={onInstall}>
                Install
              </button>
            )}
          </div>
        </div>

        <div className="tabs">
          <button
            className={activeTab === 'overview' ? 'active' : ''}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button
            className={activeTab === 'permissions' ? 'active' : ''}
            onClick={() => setActiveTab('permissions')}
          >
            Permissions
          </button>
          <button
            className={activeTab === 'reviews' ? 'active' : ''}
            onClick={() => setActiveTab('reviews')}
          >
            Reviews
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'overview' && (
            <div className="overview-tab">
              <p className="description">{plugin.description}</p>
              <div className="stats">
                <div className="stat">
                  <span className="value">{formatNumber(plugin.installCount)}</span>
                  <span className="label">Installs</span>
                </div>
                <div className="stat">
                  <span className="value">{plugin.rating.toFixed(1)}</span>
                  <span className="label">Rating</span>
                </div>
              </div>
            </div>
          )}
          {activeTab === 'permissions' && (
            <div className="permissions-tab">
              <p className="permissions-intro">
                This plugin requires the following permissions:
              </p>
              <ul className="permissions-list">
                {plugin.permissions.map(perm => (
                  <li key={perm}>
                    <PermissionIcon permission={perm} />
                    <span>{formatPermission(perm)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {activeTab === 'reviews' && (
            <div className="reviews-tab">
              <p>Reviews coming soon...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Webhook Manager
// ============================================================================

interface WebhookManagerProps {
  webhooks: Webhook[];
  onCreate: (name: string, url: string, events: string[]) => void;
  onDelete: (webhookId: string) => void;
  onToggle: (webhookId: string, enabled: boolean) => void;
}

export function WebhookManager({ webhooks, onCreate, onDelete, onToggle }: WebhookManagerProps) {
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newName, setNewName] = useState('');
  const [newUrl, setNewUrl] = useState('');
  const [selectedEvents, setSelectedEvents] = useState<string[]>([]);
  const [expandedWebhook, setExpandedWebhook] = useState<string | null>(null);

  const availableEvents = [
    'email.received',
    'email.sent',
    'email.opened',
    'email.archived',
    'email.deleted',
    'contact.created',
    'contact.updated',
    'label.created',
    'attachment.uploaded',
  ];

  const handleCreate = () => {
    if (newName && newUrl && selectedEvents.length > 0) {
      onCreate(newName, newUrl, selectedEvents);
      setNewName('');
      setNewUrl('');
      setSelectedEvents([]);
      setShowCreateForm(false);
    }
  };

  const toggleEvent = (event: string) => {
    setSelectedEvents(prev =>
      prev.includes(event)
        ? prev.filter(e => e !== event)
        : [...prev, event]
    );
  };

  return (
    <div className="webhook-manager">
      <div className="section-header">
        <h2>Webhooks</h2>
        <button className="primary" onClick={() => setShowCreateForm(true)}>
          <PlusIcon /> New Webhook
        </button>
      </div>

      {showCreateForm && (
        <div className="create-webhook-form">
          <div className="form-group">
            <label>Name</label>
            <input
              type="text"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              placeholder="My Webhook"
            />
          </div>
          <div className="form-group">
            <label>URL</label>
            <input
              type="url"
              value={newUrl}
              onChange={e => setNewUrl(e.target.value)}
              placeholder="https://example.com/webhook"
            />
          </div>
          <div className="form-group">
            <label>Events</label>
            <div className="event-checkboxes">
              {availableEvents.map(event => (
                <label key={event}>
                  <input
                    type="checkbox"
                    checked={selectedEvents.includes(event)}
                    onChange={() => toggleEvent(event)}
                  />
                  {event}
                </label>
              ))}
            </div>
          </div>
          <div className="form-actions">
            <button className="secondary" onClick={() => setShowCreateForm(false)}>
              Cancel
            </button>
            <button
              className="primary"
              onClick={handleCreate}
              disabled={!newName || !newUrl || selectedEvents.length === 0}
            >
              Create Webhook
            </button>
          </div>
        </div>
      )}

      <div className="webhooks-list">
        {webhooks.map(webhook => (
          <div key={webhook.id} className="webhook-item">
            <div className="webhook-header" onClick={() => setExpandedWebhook(
              expandedWebhook === webhook.id ? null : webhook.id
            )}>
              <div className="webhook-status">
                <span className={`status-dot ${webhook.enabled ? 'active' : 'disabled'}`} />
              </div>
              <div className="webhook-info">
                <h3>{webhook.name}</h3>
                <span className="webhook-url">{webhook.url}</span>
              </div>
              <div className="webhook-meta">
                {webhook.failureCount > 0 && (
                  <span className="failure-badge">{webhook.failureCount} failures</span>
                )}
                {webhook.lastTriggered && (
                  <span className="last-triggered">
                    Last: {formatTime(webhook.lastTriggered)}
                  </span>
                )}
              </div>
              <div className="webhook-actions">
                <button
                  className={`toggle ${webhook.enabled ? 'on' : 'off'}`}
                  onClick={e => {
                    e.stopPropagation();
                    onToggle(webhook.id, !webhook.enabled);
                  }}
                >
                  {webhook.enabled ? 'Enabled' : 'Disabled'}
                </button>
                <button
                  className="delete"
                  onClick={e => {
                    e.stopPropagation();
                    onDelete(webhook.id);
                  }}
                >
                  <TrashIcon />
                </button>
              </div>
            </div>
            {expandedWebhook === webhook.id && (
              <div className="webhook-details">
                <h4>Subscribed Events</h4>
                <ul className="events-list">
                  {webhook.events.map(event => (
                    <li key={event}>{event}</li>
                  ))}
                </ul>
                <WebhookDeliveryHistory webhookId={webhook.id} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function WebhookDeliveryHistory({ webhookId }: { webhookId: string }) {
  const [deliveries, setDeliveries] = useState<WebhookDelivery[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDeliveries();
  }, [webhookId]);

  const fetchDeliveries = async () => {
    try {
      const response = await fetch(`/api/webhooks/${webhookId}/deliveries?limit=10`);
      const data = await response.json();
      setDeliveries(data.deliveries);
    } catch (error) {
      console.error('Failed to fetch deliveries:', error);
    }
    setLoading(false);
  };

  if (loading) return <div className="loading">Loading history...</div>;

  return (
    <div className="delivery-history">
      <h4>Recent Deliveries</h4>
      {deliveries.length === 0 ? (
        <p className="no-deliveries">No deliveries yet</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Status</th>
              <th>Event</th>
              <th>Response</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {deliveries.map(delivery => (
              <tr key={delivery.id}>
                <td>
                  <span className={`status ${delivery.success ? 'success' : 'failed'}`}>
                    {delivery.success ? <CheckIcon /> : <XIcon />}
                  </span>
                </td>
                <td>{delivery.event}</td>
                <td>{delivery.responseStatus || 'N/A'}</td>
                <td>{formatTime(delivery.deliveredAt)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ============================================================================
// Custom Actions Editor
// ============================================================================

interface CustomActionsEditorProps {
  actions: CustomAction[];
  onSave: (action: CustomAction) => void;
  onDelete: (actionId: string) => void;
}

export function CustomActionsEditor({ actions, onSave, onDelete }: CustomActionsEditorProps) {
  const [editingAction, setEditingAction] = useState<CustomAction | null>(null);
  const [isNew, setIsNew] = useState(false);

  const createNewAction = () => {
    setEditingAction({
      id: '',
      name: '',
      script: '// Your action script here\n// Available: email, context\n\nconsole.log("Action executed!");',
      location: 'toolbar',
      enabled: true,
    });
    setIsNew(true);
  };

  const handleSave = () => {
    if (editingAction) {
      onSave(editingAction);
      setEditingAction(null);
      setIsNew(false);
    }
  };

  return (
    <div className="custom-actions-editor">
      <div className="section-header">
        <h2>Custom Actions</h2>
        <button className="primary" onClick={createNewAction}>
          <PlusIcon /> New Action
        </button>
      </div>

      <div className="actions-list">
        {actions.map(action => (
          <div key={action.id} className="action-item">
            <div className="action-info">
              <h3>{action.name}</h3>
              {action.description && <p>{action.description}</p>}
              <div className="action-meta">
                <span className="location">{action.location.replace('_', ' ')}</span>
                {action.shortcut && <kbd>{action.shortcut}</kbd>}
              </div>
            </div>
            <div className="action-controls">
              <button onClick={() => {
                setEditingAction(action);
                setIsNew(false);
              }}>
                Edit
              </button>
              <button className="delete" onClick={() => onDelete(action.id)}>
                <TrashIcon />
              </button>
            </div>
          </div>
        ))}
      </div>

      {editingAction && (
        <div className="modal-overlay" onClick={() => setEditingAction(null)}>
          <div className="modal action-editor-modal" onClick={e => e.stopPropagation()}>
            <h2>{isNew ? 'Create Action' : 'Edit Action'}</h2>

            <div className="form-group">
              <label>Name</label>
              <input
                type="text"
                value={editingAction.name}
                onChange={e => setEditingAction({ ...editingAction, name: e.target.value })}
                placeholder="My Action"
              />
            </div>

            <div className="form-group">
              <label>Description (optional)</label>
              <input
                type="text"
                value={editingAction.description || ''}
                onChange={e => setEditingAction({ ...editingAction, description: e.target.value })}
                placeholder="What this action does"
              />
            </div>

            <div className="form-group">
              <label>Location</label>
              <select
                value={editingAction.location}
                onChange={e => setEditingAction({
                  ...editingAction,
                  location: e.target.value as any
                })}
              >
                <option value="toolbar">Email Toolbar</option>
                <option value="context_menu">Context Menu</option>
                <option value="compose_toolbar">Compose Toolbar</option>
              </select>
            </div>

            <div className="form-group">
              <label>Keyboard Shortcut (optional)</label>
              <input
                type="text"
                value={editingAction.shortcut || ''}
                onChange={e => setEditingAction({ ...editingAction, shortcut: e.target.value })}
                placeholder="e.g., Ctrl+Shift+A"
              />
            </div>

            <div className="form-group">
              <label>Script</label>
              <textarea
                className="script-editor"
                value={editingAction.script}
                onChange={e => setEditingAction({ ...editingAction, script: e.target.value })}
                rows={15}
              />
              <p className="script-help">
                Available variables: <code>email</code> (current email object),
                <code>context</code> (action context)
              </p>
            </div>

            <div className="modal-footer">
              <button className="secondary" onClick={() => setEditingAction(null)}>
                Cancel
              </button>
              <button
                className="primary"
                onClick={handleSave}
                disabled={!editingAction.name || !editingAction.script}
              >
                Save Action
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// OAuth Apps Manager
// ============================================================================

interface OAuthAppsManagerProps {
  apps: OAuthApp[];
  onRevoke: (appId: string) => void;
}

export function OAuthAppsManager({ apps, onRevoke }: OAuthAppsManagerProps) {
  return (
    <div className="oauth-apps-manager">
      <div className="section-header">
        <h2>Connected Apps</h2>
        <p className="description">
          These apps have access to your Mycelix Mail account
        </p>
      </div>

      {apps.length === 0 ? (
        <div className="no-apps">
          <p>No apps connected</p>
        </div>
      ) : (
        <div className="apps-list">
          {apps.map(app => (
            <div key={app.id} className="app-item">
              <div className="app-info">
                <h3>{app.name}</h3>
                {app.description && <p>{app.description}</p>}
                <div className="app-meta">
                  <span className="authorized">
                    Authorized {formatDate(app.authorizedAt)}
                  </span>
                  {app.lastUsed && (
                    <span className="last-used">
                      Last used {formatTime(app.lastUsed)}
                    </span>
                  )}
                </div>
                <div className="app-scopes">
                  <h4>Permissions:</h4>
                  <ul>
                    {app.scopes.map(scope => (
                      <li key={scope}>{formatScope(scope)}</li>
                    ))}
                  </ul>
                </div>
              </div>
              <div className="app-actions">
                <button className="revoke" onClick={() => onRevoke(app.id)}>
                  Revoke Access
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Developer Portal Main Component
// ============================================================================

export default function DeveloperPortal() {
  const [activeSection, setActiveSection] = useState<'plugins' | 'webhooks' | 'actions' | 'apps'>('plugins');
  const [webhooks, setWebhooks] = useState<Webhook[]>([]);
  const [actions, setActions] = useState<CustomAction[]>([]);
  const [apps, setApps] = useState<OAuthApp[]>([]);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    // Fetch webhooks, actions, and apps
    try {
      const [webhooksRes, actionsRes, appsRes] = await Promise.all([
        fetch('/api/webhooks'),
        fetch('/api/actions'),
        fetch('/api/oauth/apps'),
      ]);
      setWebhooks(await webhooksRes.json());
      setActions(await actionsRes.json());
      setApps(await appsRes.json());
    } catch (error) {
      console.error('Failed to fetch data:', error);
    }
  };

  const handleInstallPlugin = async (pluginId: string) => {
    await fetch(`/api/plugins/${pluginId}/install`, { method: 'POST' });
    // Refresh
  };

  const handleUninstallPlugin = async (pluginId: string) => {
    await fetch(`/api/plugins/${pluginId}/uninstall`, { method: 'DELETE' });
    // Refresh
  };

  const handleCreateWebhook = async (name: string, url: string, events: string[]) => {
    await fetch('/api/webhooks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, url, events }),
    });
    fetchData();
  };

  const handleDeleteWebhook = async (webhookId: string) => {
    await fetch(`/api/webhooks/${webhookId}`, { method: 'DELETE' });
    fetchData();
  };

  const handleToggleWebhook = async (webhookId: string, enabled: boolean) => {
    await fetch(`/api/webhooks/${webhookId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    fetchData();
  };

  const handleSaveAction = async (action: CustomAction) => {
    const method = action.id ? 'PUT' : 'POST';
    const url = action.id ? `/api/actions/${action.id}` : '/api/actions';
    await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(action),
    });
    fetchData();
  };

  const handleDeleteAction = async (actionId: string) => {
    await fetch(`/api/actions/${actionId}`, { method: 'DELETE' });
    fetchData();
  };

  const handleRevokeApp = async (appId: string) => {
    await fetch(`/api/oauth/apps/${appId}`, { method: 'DELETE' });
    fetchData();
  };

  return (
    <div className="developer-portal">
      <nav className="portal-nav">
        <button
          className={activeSection === 'plugins' ? 'active' : ''}
          onClick={() => setActiveSection('plugins')}
        >
          <PluginIcon /> Plugins
        </button>
        <button
          className={activeSection === 'webhooks' ? 'active' : ''}
          onClick={() => setActiveSection('webhooks')}
        >
          <WebhookIcon /> Webhooks
        </button>
        <button
          className={activeSection === 'actions' ? 'active' : ''}
          onClick={() => setActiveSection('actions')}
        >
          <CodeIcon /> Custom Actions
        </button>
        <button
          className={activeSection === 'apps' ? 'active' : ''}
          onClick={() => setActiveSection('apps')}
        >
          <KeyIcon /> Connected Apps
        </button>
      </nav>

      <div className="portal-content">
        {activeSection === 'plugins' && (
          <PluginMarketplace
            onInstall={handleInstallPlugin}
            onUninstall={handleUninstallPlugin}
          />
        )}
        {activeSection === 'webhooks' && (
          <WebhookManager
            webhooks={webhooks}
            onCreate={handleCreateWebhook}
            onDelete={handleDeleteWebhook}
            onToggle={handleToggleWebhook}
          />
        )}
        {activeSection === 'actions' && (
          <CustomActionsEditor
            actions={actions}
            onSave={handleSaveAction}
            onDelete={handleDeleteAction}
          />
        )}
        {activeSection === 'apps' && (
          <OAuthAppsManager
            apps={apps}
            onRevoke={handleRevokeApp}
          />
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Helper Functions
// ============================================================================

function formatNumber(num: number): string {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
}

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

function formatPermission(perm: string): string {
  return perm
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, str => str.toUpperCase());
}

function formatScope(scope: string): string {
  return scope
    .replace(/[._]/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase());
}

// ============================================================================
// Icon Components
// ============================================================================

function SearchIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>;
}

function PluginIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 2L2 7l10 5 10-5-10-5z" />
    <path d="M2 17l10 5 10-5M2 12l10 5 10-5" />
  </svg>;
}

function VerifiedIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
  </svg>;
}

function StarIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
  </svg>;
}

function CloseIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>;
}

function PlusIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>;
}

function TrashIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
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

function WebhookIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M18 16.98h-5.99c-1.1 0-1.95.94-2.48 1.9A4 4 0 0 1 2 17c.01-.7.2-1.4.57-2" />
    <path d="m6 17 3.13-5.78c.53-.97.43-2.17-.26-3.02A4 4 0 0 1 12 2a4 4 0 0 1 3.49 6" />
    <path d="m12 6 3.13 5.73c.53.98 1.45 1.69 2.6 1.95A4 4 0 0 1 18 22a4 4 0 0 1-3.49-6" />
  </svg>;
}

function CodeIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="16 18 22 12 16 6" />
    <polyline points="8 6 2 12 8 18" />
  </svg>;
}

function KeyIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4" />
  </svg>;
}

function PermissionIcon({ permission }: { permission: string }) {
  // Return appropriate icon based on permission type
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
  </svg>;
}
