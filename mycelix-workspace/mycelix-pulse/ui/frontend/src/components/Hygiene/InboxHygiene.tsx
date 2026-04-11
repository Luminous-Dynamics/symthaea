// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Track U: Email Hygiene & Inbox Zero
 *
 * Smart unsubscribe, newsletter digest, email debt tracking,
 * bulk actions, quiet hours, and sender reputation.
 */

import React, { useState, useEffect, useCallback } from 'react';

// ============================================================================
// Types
// ============================================================================

interface InboxHealth {
  totalEmails: number;
  unreadCount: number;
  inboxZeroStreak: number;
  lastInboxZero?: string;
  healthScore: number;
  emailDebt: EmailDebt;
  categories: CategoryBreakdown[];
  recommendations: HygieneRecommendation[];
}

interface EmailDebt {
  overdueReplies: OverdueEmail[];
  agingUnread: AgingEmail[];
  totalDebtScore: number;
  oldestUnreadDays: number;
  averageResponseTimeHours: number;
}

interface OverdueEmail {
  emailId: string;
  subject: string;
  from: string;
  receivedAt: string;
  daysOverdue: number;
  priority: 'Critical' | 'High' | 'Medium' | 'Low';
  suggestedAction: string;
}

interface AgingEmail {
  emailId: string;
  subject: string;
  from: string;
  receivedAt: string;
  ageDays: number;
  category: string;
}

interface CategoryBreakdown {
  category: string;
  count: number;
  unread: number;
  percentage: number;
}

interface HygieneRecommendation {
  id: string;
  recommendationType: string;
  title: string;
  description: string;
  impact: 'High' | 'Medium' | 'Low';
  affectedCount: number;
}

interface Subscription {
  id: string;
  senderEmail: string;
  senderName?: string;
  emailCount: number;
  firstSeen: string;
  lastSeen: string;
  status: 'Active' | 'Unsubscribed' | 'Pending' | 'Failed';
}

interface DigestConfig {
  id: string;
  name: string;
  senders: string[];
  frequency: 'Daily' | 'Weekly' | 'BiWeekly' | 'Monthly';
  deliveryTime: string;
  deliveryDay?: string;
  enabled: boolean;
}

interface QuietHoursConfig {
  enabled: boolean;
  startTime: string;
  endTime: string;
  days: string[];
  exceptions: { exceptionType: string; value: string }[];
  autoDeferLowPriority: boolean;
}

interface SenderReputation {
  sender: string;
  domain: string;
  reputationScore: number;
  emailCount: number;
  openRate: number;
  replyRate: number;
  archiveRate: number;
  deleteRate: number;
  isBlocked: boolean;
  isVip: boolean;
}

// ============================================================================
// Hooks
// ============================================================================

function useInboxHealth() {
  const [health, setHealth] = useState<InboxHealth | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/hygiene/health')
      .then(res => res.json())
      .then(setHealth)
      .finally(() => setLoading(false));
  }, []);

  const refresh = useCallback(() => {
    setLoading(true);
    fetch('/api/hygiene/health')
      .then(res => res.json())
      .then(setHealth)
      .finally(() => setLoading(false));
  }, []);

  return { health, loading, refresh };
}

function useSubscriptions() {
  const [subscriptions, setSubscriptions] = useState<Subscription[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/hygiene/subscriptions')
      .then(res => res.json())
      .then(setSubscriptions)
      .finally(() => setLoading(false));
  }, []);

  const unsubscribe = useCallback(async (id: string) => {
    const response = await fetch(`/api/hygiene/subscriptions/${id}/unsubscribe`, {
      method: 'POST',
    });
    const result = await response.json();
    setSubscriptions(prev => prev.map(s =>
      s.id === id ? { ...s, status: result.success ? 'Unsubscribed' : 'Failed' } : s
    ));
    return result;
  }, []);

  const bulkUnsubscribe = useCallback(async (ids: string[]) => {
    const response = await fetch('/api/hygiene/subscriptions/bulk-unsubscribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids }),
    });
    return await response.json();
  }, []);

  return { subscriptions, loading, unsubscribe, bulkUnsubscribe };
}

function useDigests() {
  const [digests, setDigests] = useState<DigestConfig[]>([]);

  useEffect(() => {
    fetch('/api/hygiene/digests')
      .then(res => res.json())
      .then(setDigests);
  }, []);

  const createDigest = useCallback(async (config: Omit<DigestConfig, 'id'>) => {
    const response = await fetch('/api/hygiene/digests', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    const newDigest = await response.json();
    setDigests(prev => [...prev, newDigest]);
    return newDigest;
  }, []);

  const updateDigest = useCallback(async (id: string, updates: Partial<DigestConfig>) => {
    await fetch(`/api/hygiene/digests/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    setDigests(prev => prev.map(d => d.id === id ? { ...d, ...updates } : d));
  }, []);

  const deleteDigest = useCallback(async (id: string) => {
    await fetch(`/api/hygiene/digests/${id}`, { method: 'DELETE' });
    setDigests(prev => prev.filter(d => d.id !== id));
  }, []);

  return { digests, createDigest, updateDigest, deleteDigest };
}

function useQuietHours() {
  const [config, setConfig] = useState<QuietHoursConfig | null>(null);
  const [isQuietTime, setIsQuietTime] = useState(false);

  useEffect(() => {
    fetch('/api/hygiene/quiet-hours')
      .then(res => res.json())
      .then(setConfig);

    fetch('/api/hygiene/quiet-hours/status')
      .then(res => res.json())
      .then(data => setIsQuietTime(data.isQuietTime));
  }, []);

  const updateConfig = useCallback(async (updates: Partial<QuietHoursConfig>) => {
    await fetch('/api/hygiene/quiet-hours', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    setConfig(prev => prev ? { ...prev, ...updates } : null);
  }, []);

  return { config, isQuietTime, updateConfig };
}

function useBulkActions() {
  const [preview, setPreview] = useState<{ id: string; subject: string; from: string }[] | null>(null);
  const [loading, setLoading] = useState(false);

  const previewAction = useCallback(async (query: string, action: string) => {
    setLoading(true);
    try {
      const response = await fetch('/api/hygiene/bulk-action/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, actionType: action, dryRun: true }),
      });
      const result = await response.json();
      setPreview(result.preview);
      return result;
    } finally {
      setLoading(false);
    }
  }, []);

  const executeAction = useCallback(async (query: string, action: string) => {
    setLoading(true);
    try {
      const response = await fetch('/api/hygiene/bulk-action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, actionType: action, dryRun: false }),
      });
      return await response.json();
    } finally {
      setLoading(false);
      setPreview(null);
    }
  }, []);

  return { preview, loading, previewAction, executeAction };
}

// ============================================================================
// Components
// ============================================================================

export function InboxHealthDashboard() {
  const { health, loading, refresh } = useInboxHealth();

  if (loading) {
    return <div className="inbox-health loading">Loading inbox health...</div>;
  }

  if (!health) {
    return <div className="inbox-health error">Failed to load inbox health</div>;
  }

  const healthColor = health.healthScore >= 80 ? '#22c55e' :
                      health.healthScore >= 50 ? '#eab308' : '#ef4444';

  return (
    <div className="inbox-health-dashboard">
      <header>
        <h2>Inbox Health</h2>
        <button onClick={refresh}>Refresh</button>
      </header>

      <div className="health-score-card">
        <div
          className="score-circle"
          style={{ '--score-color': healthColor } as React.CSSProperties}
        >
          <span className="score">{Math.round(health.healthScore)}</span>
          <span className="label">Health Score</span>
        </div>

        <div className="stats">
          <div className="stat">
            <span className="value">{health.totalEmails}</span>
            <span className="label">Total Emails</span>
          </div>
          <div className="stat">
            <span className="value">{health.unreadCount}</span>
            <span className="label">Unread</span>
          </div>
          <div className="stat">
            <span className="value">{health.inboxZeroStreak}</span>
            <span className="label">Inbox Zero Streak</span>
          </div>
        </div>
      </div>

      <div className="email-debt-section">
        <h3>Email Debt</h3>
        <div className="debt-overview">
          <div className="debt-score">
            <span className="value">{Math.round(health.emailDebt.totalDebtScore)}</span>
            <span className="label">Debt Score</span>
          </div>
          <div className="debt-stats">
            <span>{health.emailDebt.overdueReplies.length} overdue replies</span>
            <span>{health.emailDebt.agingUnread.length} aging unread</span>
            <span>Oldest: {health.emailDebt.oldestUnreadDays} days</span>
            <span>Avg response: {health.emailDebt.averageResponseTimeHours.toFixed(1)}h</span>
          </div>
        </div>

        {health.emailDebt.overdueReplies.length > 0 && (
          <div className="overdue-list">
            <h4>Overdue Replies</h4>
            {health.emailDebt.overdueReplies.slice(0, 5).map(email => (
              <div key={email.emailId} className={`overdue-item ${email.priority.toLowerCase()}`}>
                <div className="header">
                  <span className="from">{email.from}</span>
                  <span className="days">{email.daysOverdue} days overdue</span>
                </div>
                <div className="subject">{email.subject}</div>
                <span className="action">{email.suggestedAction}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {health.recommendations.length > 0 && (
        <div className="recommendations-section">
          <h3>Recommendations</h3>
          {health.recommendations.map(rec => (
            <RecommendationCard key={rec.id} recommendation={rec} />
          ))}
        </div>
      )}

      <div className="categories-section">
        <h3>Email Categories</h3>
        <div className="category-bars">
          {health.categories.map(cat => (
            <div key={cat.category} className="category-bar">
              <div className="category-header">
                <span className="name">{cat.category}</span>
                <span className="count">{cat.count} ({cat.unread} unread)</span>
              </div>
              <div className="bar">
                <div
                  className="fill"
                  style={{ width: `${cat.percentage}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

interface RecommendationCardProps {
  recommendation: HygieneRecommendation;
}

function RecommendationCard({ recommendation }: RecommendationCardProps) {
  const impactColors = { High: '#ef4444', Medium: '#eab308', Low: '#22c55e' };
  const typeIcons: Record<string, string> = {
    Unsubscribe: '🔕',
    BulkArchive: '📦',
    BulkDelete: '🗑️',
    CreateFilter: '🏷️',
    BlockSender: '🚫',
    DigestNewsletter: '📰',
  };

  return (
    <div className="recommendation-card">
      <div className="icon">{typeIcons[recommendation.recommendationType] || '💡'}</div>
      <div className="content">
        <h4>{recommendation.title}</h4>
        <p>{recommendation.description}</p>
        <div className="meta">
          <span
            className="impact"
            style={{ color: impactColors[recommendation.impact] }}
          >
            {recommendation.impact} impact
          </span>
          <span className="affected">{recommendation.affectedCount} emails affected</span>
        </div>
      </div>
      <button className="apply-btn">Apply</button>
    </div>
  );
}

export function SubscriptionManager() {
  const { subscriptions, loading, unsubscribe, bulkUnsubscribe } = useSubscriptions();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState<'all' | 'active' | 'unsubscribed'>('active');

  const toggleSelect = (id: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectAll = () => {
    const filtered = subscriptions.filter(s =>
      filter === 'all' ? true :
      filter === 'active' ? s.status === 'Active' :
      s.status === 'Unsubscribed'
    );
    setSelected(new Set(filtered.map(s => s.id)));
  };

  const handleBulkUnsubscribe = async () => {
    if (selected.size === 0) return;
    await bulkUnsubscribe(Array.from(selected));
    setSelected(new Set());
  };

  const filteredSubscriptions = subscriptions.filter(s =>
    filter === 'all' ? true :
    filter === 'active' ? s.status === 'Active' :
    s.status === 'Unsubscribed'
  );

  return (
    <div className="subscription-manager">
      <header>
        <h2>Subscriptions & Newsletters</h2>
        <div className="filters">
          {(['all', 'active', 'unsubscribed'] as const).map(f => (
            <button
              key={f}
              className={filter === f ? 'active' : ''}
              onClick={() => setFilter(f)}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </header>

      <div className="actions-bar">
        <button onClick={selectAll}>Select All</button>
        <button
          onClick={handleBulkUnsubscribe}
          disabled={selected.size === 0}
          className="danger"
        >
          Unsubscribe Selected ({selected.size})
        </button>
      </div>

      {loading ? (
        <div className="loading">Loading subscriptions...</div>
      ) : (
        <div className="subscription-list">
          {filteredSubscriptions.map(sub => (
            <div key={sub.id} className={`subscription-item ${sub.status.toLowerCase()}`}>
              <input
                type="checkbox"
                checked={selected.has(sub.id)}
                onChange={() => toggleSelect(sub.id)}
              />
              <div className="info">
                <span className="sender">{sub.senderName || sub.senderEmail}</span>
                <span className="email">{sub.senderEmail}</span>
                <span className="count">{sub.emailCount} emails</span>
                <span className="dates">
                  {new Date(sub.firstSeen).toLocaleDateString()} -
                  {new Date(sub.lastSeen).toLocaleDateString()}
                </span>
              </div>
              <span className={`status ${sub.status.toLowerCase()}`}>{sub.status}</span>
              {sub.status === 'Active' && (
                <button onClick={() => unsubscribe(sub.id)}>Unsubscribe</button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function DigestManager() {
  const { digests, createDigest, updateDigest, deleteDigest } = useDigests();
  const { subscriptions } = useSubscriptions();
  const [showCreate, setShowCreate] = useState(false);
  const [newDigest, setNewDigest] = useState({
    name: '',
    senders: [] as string[],
    frequency: 'Daily' as DigestConfig['frequency'],
    deliveryTime: '09:00',
    deliveryDay: undefined as string | undefined,
    enabled: true,
  });

  const handleCreate = async () => {
    await createDigest(newDigest);
    setShowCreate(false);
    setNewDigest({
      name: '',
      senders: [],
      frequency: 'Daily',
      deliveryTime: '09:00',
      deliveryDay: undefined,
      enabled: true,
    });
  };

  const activeSenders = subscriptions
    .filter(s => s.status === 'Active')
    .map(s => s.senderEmail);

  return (
    <div className="digest-manager">
      <header>
        <h2>Newsletter Digests</h2>
        <button onClick={() => setShowCreate(true)}>Create Digest</button>
      </header>

      <p className="description">
        Bundle newsletters into periodic digests to reduce inbox clutter.
      </p>

      {digests.length === 0 ? (
        <div className="empty-state">
          <p>No digests configured yet.</p>
          <button onClick={() => setShowCreate(true)}>Create your first digest</button>
        </div>
      ) : (
        <div className="digest-list">
          {digests.map(digest => (
            <div key={digest.id} className="digest-card">
              <div className="header">
                <h3>{digest.name}</h3>
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={digest.enabled}
                    onChange={e => updateDigest(digest.id, { enabled: e.target.checked })}
                  />
                  <span className="slider" />
                </label>
              </div>
              <div className="details">
                <span className="frequency">{digest.frequency}</span>
                <span className="time">at {digest.deliveryTime}</span>
                {digest.deliveryDay && <span className="day">on {digest.deliveryDay}</span>}
              </div>
              <div className="senders">
                {digest.senders.slice(0, 3).map(s => (
                  <span key={s} className="sender">{s}</span>
                ))}
                {digest.senders.length > 3 && (
                  <span className="more">+{digest.senders.length - 3} more</span>
                )}
              </div>
              <div className="actions">
                <button onClick={() => deleteDigest(digest.id)}>Delete</button>
              </div>
            </div>
          ))}
        </div>
      )}

      {showCreate && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Create Newsletter Digest</h3>

            <div className="form-group">
              <label>Name</label>
              <input
                type="text"
                value={newDigest.name}
                onChange={e => setNewDigest(prev => ({ ...prev, name: e.target.value }))}
                placeholder="e.g., Weekly Newsletter Roundup"
              />
            </div>

            <div className="form-group">
              <label>Frequency</label>
              <select
                value={newDigest.frequency}
                onChange={e => setNewDigest(prev => ({
                  ...prev,
                  frequency: e.target.value as DigestConfig['frequency']
                }))}
              >
                <option value="Daily">Daily</option>
                <option value="Weekly">Weekly</option>
                <option value="BiWeekly">Bi-Weekly</option>
                <option value="Monthly">Monthly</option>
              </select>
            </div>

            <div className="form-group">
              <label>Delivery Time</label>
              <input
                type="time"
                value={newDigest.deliveryTime}
                onChange={e => setNewDigest(prev => ({ ...prev, deliveryTime: e.target.value }))}
              />
            </div>

            {newDigest.frequency !== 'Daily' && (
              <div className="form-group">
                <label>Delivery Day</label>
                <select
                  value={newDigest.deliveryDay || ''}
                  onChange={e => setNewDigest(prev => ({ ...prev, deliveryDay: e.target.value || undefined }))}
                >
                  <option value="">Select day</option>
                  {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].map(day => (
                    <option key={day} value={day.toLowerCase()}>{day}</option>
                  ))}
                </select>
              </div>
            )}

            <div className="form-group">
              <label>Include senders</label>
              <div className="sender-selector">
                {activeSenders.map(sender => (
                  <label key={sender} className="sender-option">
                    <input
                      type="checkbox"
                      checked={newDigest.senders.includes(sender)}
                      onChange={e => {
                        if (e.target.checked) {
                          setNewDigest(prev => ({ ...prev, senders: [...prev.senders, sender] }));
                        } else {
                          setNewDigest(prev => ({ ...prev, senders: prev.senders.filter(s => s !== sender) }));
                        }
                      }}
                    />
                    {sender}
                  </label>
                ))}
              </div>
            </div>

            <div className="modal-actions">
              <button onClick={() => setShowCreate(false)}>Cancel</button>
              <button onClick={handleCreate} disabled={!newDigest.name || newDigest.senders.length === 0}>
                Create Digest
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function QuietHoursSettings() {
  const { config, isQuietTime, updateConfig } = useQuietHours();

  if (!config) {
    return <div className="loading">Loading quiet hours settings...</div>;
  }

  const days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'];

  return (
    <div className="quiet-hours-settings">
      <header>
        <h2>Quiet Hours</h2>
        {isQuietTime && <span className="quiet-badge">Currently in quiet hours</span>}
      </header>

      <div className="main-toggle">
        <label>
          <input
            type="checkbox"
            checked={config.enabled}
            onChange={e => updateConfig({ enabled: e.target.checked })}
          />
          Enable Quiet Hours
        </label>
      </div>

      {config.enabled && (
        <>
          <div className="time-range">
            <div className="form-group">
              <label>Start Time</label>
              <input
                type="time"
                value={config.startTime}
                onChange={e => updateConfig({ startTime: e.target.value })}
              />
            </div>
            <span className="separator">to</span>
            <div className="form-group">
              <label>End Time</label>
              <input
                type="time"
                value={config.endTime}
                onChange={e => updateConfig({ endTime: e.target.value })}
              />
            </div>
          </div>

          <div className="days-selector">
            <label>Active Days</label>
            <div className="day-buttons">
              {days.map(day => (
                <button
                  key={day}
                  className={config.days.includes(day) ? 'active' : ''}
                  onClick={() => {
                    const newDays = config.days.includes(day)
                      ? config.days.filter(d => d !== day)
                      : [...config.days, day];
                    updateConfig({ days: newDays });
                  }}
                >
                  {day.slice(0, 3)}
                </button>
              ))}
            </div>
          </div>

          <div className="option">
            <label>
              <input
                type="checkbox"
                checked={config.autoDeferLowPriority}
                onChange={e => updateConfig({ autoDeferLowPriority: e.target.checked })}
              />
              Auto-defer low priority emails during quiet hours
            </label>
          </div>

          <div className="exceptions">
            <h3>Exceptions (always notify)</h3>
            <p className="hint">VIP contacts and emails matching these rules will still notify you.</p>
            {config.exceptions.map((exc, i) => (
              <div key={i} className="exception-item">
                <span>{exc.exceptionType}: {exc.value}</span>
                <button onClick={() => {
                  const newExceptions = config.exceptions.filter((_, idx) => idx !== i);
                  updateConfig({ exceptions: newExceptions });
                }}>Remove</button>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export function BulkActionsPanel() {
  const { preview, loading, previewAction, executeAction } = useBulkActions();
  const [query, setQuery] = useState('');
  const [action, setAction] = useState<string>('Archive');

  const presets = [
    { label: 'Promotional from last month', query: 'promotional from last month' },
    { label: 'Read newsletters older than 30 days', query: 'newsletter read older than 30 days' },
    { label: 'All unread from 90 days ago', query: 'unread older than 90 days' },
    { label: 'Large attachments', query: 'has:attachment larger:10MB' },
  ];

  const actions = ['Archive', 'Delete', 'MarkRead', 'MarkUnread', 'MoveTo', 'AddLabel', 'RemoveLabel'];

  const handlePreview = () => {
    if (query) {
      previewAction(query, action);
    }
  };

  const handleExecute = async () => {
    if (query && preview) {
      const result = await executeAction(query, action);
      alert(`${result.affectedCount} emails processed`);
    }
  };

  return (
    <div className="bulk-actions-panel">
      <header>
        <h2>Bulk Actions</h2>
        <p>Use natural language to describe which emails to act on</p>
      </header>

      <div className="presets">
        <span>Quick actions:</span>
        {presets.map((preset, i) => (
          <button key={i} onClick={() => setQuery(preset.query)}>
            {preset.label}
          </button>
        ))}
      </div>

      <div className="query-input">
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="e.g., Archive all promotional emails from last month"
        />
      </div>

      <div className="action-selector">
        <label>Action:</label>
        <select value={action} onChange={e => setAction(e.target.value)}>
          {actions.map(a => (
            <option key={a} value={a}>{a.replace(/([A-Z])/g, ' $1').trim()}</option>
          ))}
        </select>
      </div>

      <div className="action-buttons">
        <button onClick={handlePreview} disabled={loading || !query}>
          {loading ? 'Loading...' : 'Preview'}
        </button>
        <button
          onClick={handleExecute}
          disabled={loading || !preview}
          className="execute"
        >
          Execute
        </button>
      </div>

      {preview && (
        <div className="preview-section">
          <h3>Preview ({preview.length} emails)</h3>
          <div className="preview-list">
            {preview.slice(0, 10).map(email => (
              <div key={email.id} className="preview-item">
                <span className="from">{email.from}</span>
                <span className="subject">{email.subject}</span>
              </div>
            ))}
            {preview.length > 10 && (
              <div className="more">...and {preview.length - 10} more</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

interface InboxHygieneProps {
  onEmailSelect?: (emailId: string) => void;
}

export function InboxHygiene({ onEmailSelect }: InboxHygieneProps) {
  const [activeTab, setActiveTab] = useState<string>('health');

  const tabs = [
    { id: 'health', label: 'Health', icon: '❤️' },
    { id: 'subscriptions', label: 'Subscriptions', icon: '🔔' },
    { id: 'digests', label: 'Digests', icon: '📰' },
    { id: 'bulk', label: 'Bulk Actions', icon: '⚡' },
    { id: 'quiet', label: 'Quiet Hours', icon: '🌙' },
  ];

  return (
    <div className="inbox-hygiene">
      <nav className="hygiene-tabs">
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

      <div className="hygiene-content">
        {activeTab === 'health' && <InboxHealthDashboard />}
        {activeTab === 'subscriptions' && <SubscriptionManager />}
        {activeTab === 'digests' && <DigestManager />}
        {activeTab === 'bulk' && <BulkActionsPanel />}
        {activeTab === 'quiet' && <QuietHoursSettings />}
      </div>
    </div>
  );
}

export default InboxHygiene;
