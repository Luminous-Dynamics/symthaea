// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Track AC: Observability & Operations Dashboard
 *
 * Real-time monitoring, metrics visualization, health checks,
 * alerting management, and system overview.
 */

import React, { useState, useEffect, useCallback } from 'react';

// ============================================================================
// Types
// ============================================================================

interface DashboardData {
  overview: OverviewStats;
  emailStats: EmailStats;
  syncStats: SyncStats;
  apiStats: ApiStats;
  systemStats: SystemStats;
  recentAlerts: Alert[];
}

interface OverviewStats {
  healthStatus: 'Healthy' | 'Degraded' | 'Unhealthy';
  uptimeSeconds: number;
  version: string;
  totalUsers: number;
  activeSessions: number;
}

interface EmailStats {
  receivedToday: number;
  sentToday: number;
  receivedHourly: number[];
  sentHourly: number[];
  averageSizeBytes: number;
  spamBlocked: number;
}

interface SyncStats {
  totalSyncs: number;
  failedSyncs: number;
  averageDurationMs: number;
  pendingItems: number;
  lastSync?: string;
}

interface ApiStats {
  requestsToday: number;
  averageLatencyMs: number;
  errorRatePercent: number;
  requestsPerMinute: number[];
  topEndpoints: EndpointStats[];
}

interface EndpointStats {
  path: string;
  method: string;
  count: number;
  averageLatencyMs: number;
  errorRate: number;
}

interface SystemStats {
  memoryUsedBytes: number;
  memoryTotalBytes: number;
  cpuPercent: number;
  diskUsedBytes: number;
  diskTotalBytes: number;
  loadAverage: [number, number, number];
}

interface Alert {
  id: string;
  ruleId: string;
  ruleName: string;
  severity: 'Critical' | 'Warning' | 'Info';
  message: string;
  firedAt: string;
  resolvedAt?: string;
  labels: Record<string, string>;
}

interface AlertRule {
  id: string;
  name: string;
  description: string;
  condition: string;
  severity: 'Critical' | 'Warning' | 'Info';
  channels: string[];
  cooldownMinutes: number;
  enabled: boolean;
}

interface HealthCheck {
  name: string;
  status: 'Healthy' | 'Degraded' | 'Unhealthy';
  message?: string;
  durationMs: number;
  metadata: Record<string, unknown>;
}

interface HealthStatus {
  status: 'Healthy' | 'Degraded' | 'Unhealthy';
  checks: HealthCheck[];
  timestamp: string;
  version: string;
  uptimeSeconds: number;
}

// ============================================================================
// Hooks
// ============================================================================

function useDashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    fetch('/api/observability/dashboard')
      .then(res => res.json())
      .then(setData)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [refresh]);

  return { data, loading, refresh };
}

function useHealth() {
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    const fetchHealth = () => {
      fetch('/api/health')
        .then(res => res.json())
        .then(setHealth);
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 10000); // Every 10s
    return () => clearInterval(interval);
  }, []);

  return health;
}

function useAlerts() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [rules, setRules] = useState<AlertRule[]>([]);

  useEffect(() => {
    fetch('/api/observability/alerts')
      .then(res => res.json())
      .then(setAlerts);

    fetch('/api/observability/alert-rules')
      .then(res => res.json())
      .then(setRules);
  }, []);

  const acknowledgeAlert = useCallback(async (alertId: string) => {
    await fetch(`/api/observability/alerts/${alertId}/acknowledge`, { method: 'POST' });
    setAlerts(prev => prev.filter(a => a.id !== alertId));
  }, []);

  const toggleRule = useCallback(async (ruleId: string, enabled: boolean) => {
    await fetch(`/api/observability/alert-rules/${ruleId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    setRules(prev => prev.map(r => r.id === ruleId ? { ...r, enabled } : r));
  }, []);

  return { alerts, rules, acknowledgeAlert, toggleRule };
}

function useMetrics(query: string, timeRange: string) {
  const [data, setData] = useState<{ timestamp: number; value: number }[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`/api/observability/metrics?query=${encodeURIComponent(query)}&range=${timeRange}`)
      .then(res => res.json())
      .then(setData)
      .finally(() => setLoading(false));
  }, [query, timeRange]);

  return { data, loading };
}

// ============================================================================
// Components
// ============================================================================

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

function formatDuration(seconds: number): string {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);

  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

function OverviewPanel({ data }: { data: DashboardData }) {
  const health = useHealth();

  const statusColors = {
    Healthy: '#22c55e',
    Degraded: '#eab308',
    Unhealthy: '#ef4444',
  };

  return (
    <div className="overview-panel">
      <div className="status-card">
        <div
          className="status-indicator"
          style={{ backgroundColor: statusColors[data.overview.healthStatus] }}
        />
        <div className="status-info">
          <span className="status-label">{data.overview.healthStatus}</span>
          <span className="version">v{data.overview.version}</span>
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <span className="value">{formatDuration(data.overview.uptimeSeconds)}</span>
          <span className="label">Uptime</span>
        </div>
        <div className="stat-card">
          <span className="value">{data.overview.totalUsers}</span>
          <span className="label">Total Users</span>
        </div>
        <div className="stat-card">
          <span className="value">{data.overview.activeSessions}</span>
          <span className="label">Active Sessions</span>
        </div>
      </div>

      {health && (
        <div className="health-checks">
          <h4>Health Checks</h4>
          {health.checks.map(check => (
            <div key={check.name} className={`health-check ${check.status.toLowerCase()}`}>
              <span className="check-status" style={{ color: statusColors[check.status] }}>
                {check.status === 'Healthy' ? '✓' : check.status === 'Degraded' ? '!' : '✗'}
              </span>
              <span className="check-name">{check.name}</span>
              <span className="check-duration">{check.durationMs}ms</span>
              {check.message && <span className="check-message">{check.message}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function EmailMetricsPanel({ stats }: { stats: EmailStats }) {
  const maxReceived = Math.max(...stats.receivedHourly, 1);
  const maxSent = Math.max(...stats.sentHourly, 1);

  return (
    <div className="email-metrics-panel">
      <h3>Email Activity</h3>

      <div className="metrics-row">
        <div className="metric">
          <span className="value">{stats.receivedToday.toLocaleString()}</span>
          <span className="label">Received Today</span>
        </div>
        <div className="metric">
          <span className="value">{stats.sentToday.toLocaleString()}</span>
          <span className="label">Sent Today</span>
        </div>
        <div className="metric">
          <span className="value">{formatBytes(stats.averageSizeBytes)}</span>
          <span className="label">Avg Size</span>
        </div>
        <div className="metric">
          <span className="value">{stats.spamBlocked}</span>
          <span className="label">Spam Blocked</span>
        </div>
      </div>

      <div className="hourly-chart">
        <h4>Last 24 Hours</h4>
        <div className="chart-container">
          <div className="bars received">
            {stats.receivedHourly.map((count, i) => (
              <div
                key={i}
                className="bar"
                style={{ height: `${(count / maxReceived) * 100}%` }}
                title={`${count} received`}
              />
            ))}
          </div>
          <div className="bars sent">
            {stats.sentHourly.map((count, i) => (
              <div
                key={i}
                className="bar"
                style={{ height: `${(count / maxSent) * 100}%` }}
                title={`${count} sent`}
              />
            ))}
          </div>
        </div>
        <div className="legend">
          <span className="received">Received</span>
          <span className="sent">Sent</span>
        </div>
      </div>
    </div>
  );
}

function SyncMetricsPanel({ stats }: { stats: SyncStats }) {
  const successRate = stats.totalSyncs > 0
    ? ((stats.totalSyncs - stats.failedSyncs) / stats.totalSyncs * 100).toFixed(1)
    : '100';

  return (
    <div className="sync-metrics-panel">
      <h3>Sync Status</h3>

      <div className="metrics-row">
        <div className="metric">
          <span className="value">{stats.totalSyncs.toLocaleString()}</span>
          <span className="label">Total Syncs</span>
        </div>
        <div className="metric">
          <span className="value">{successRate}%</span>
          <span className="label">Success Rate</span>
        </div>
        <div className="metric">
          <span className="value">{stats.averageDurationMs}ms</span>
          <span className="label">Avg Duration</span>
        </div>
        <div className="metric">
          <span className="value">{stats.pendingItems}</span>
          <span className="label">Pending Items</span>
        </div>
      </div>

      {stats.lastSync && (
        <div className="last-sync">
          Last sync: {new Date(stats.lastSync).toLocaleString()}
        </div>
      )}
    </div>
  );
}

function ApiMetricsPanel({ stats }: { stats: ApiStats }) {
  const maxRpm = Math.max(...stats.requestsPerMinute, 1);

  return (
    <div className="api-metrics-panel">
      <h3>API Performance</h3>

      <div className="metrics-row">
        <div className="metric">
          <span className="value">{stats.requestsToday.toLocaleString()}</span>
          <span className="label">Requests Today</span>
        </div>
        <div className="metric">
          <span className="value">{stats.averageLatencyMs}ms</span>
          <span className="label">Avg Latency</span>
        </div>
        <div className="metric">
          <span className={`value ${stats.errorRatePercent > 1 ? 'warning' : ''}`}>
            {stats.errorRatePercent.toFixed(2)}%
          </span>
          <span className="label">Error Rate</span>
        </div>
      </div>

      <div className="rpm-chart">
        <h4>Requests/Minute (Last Hour)</h4>
        <div className="chart-container">
          {stats.requestsPerMinute.map((count, i) => (
            <div
              key={i}
              className="bar"
              style={{ height: `${(count / maxRpm) * 100}%` }}
              title={`${count} req/min`}
            />
          ))}
        </div>
      </div>

      <div className="top-endpoints">
        <h4>Top Endpoints</h4>
        <table>
          <thead>
            <tr>
              <th>Endpoint</th>
              <th>Count</th>
              <th>Latency</th>
              <th>Errors</th>
            </tr>
          </thead>
          <tbody>
            {stats.topEndpoints.map((ep, i) => (
              <tr key={i}>
                <td><code>{ep.method} {ep.path}</code></td>
                <td>{ep.count.toLocaleString()}</td>
                <td>{ep.averageLatencyMs}ms</td>
                <td className={ep.errorRate > 1 ? 'warning' : ''}>{ep.errorRate.toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SystemMetricsPanel({ stats }: { stats: SystemStats }) {
  const memoryPercent = (stats.memoryUsedBytes / stats.memoryTotalBytes) * 100;
  const diskPercent = (stats.diskUsedBytes / stats.diskTotalBytes) * 100;

  return (
    <div className="system-metrics-panel">
      <h3>System Resources</h3>

      <div className="resource-meters">
        <div className="resource">
          <div className="resource-header">
            <span className="label">CPU</span>
            <span className="value">{stats.cpuPercent.toFixed(1)}%</span>
          </div>
          <div className="meter">
            <div
              className="fill"
              style={{
                width: `${stats.cpuPercent}%`,
                backgroundColor: stats.cpuPercent > 80 ? '#ef4444' : stats.cpuPercent > 60 ? '#eab308' : '#22c55e',
              }}
            />
          </div>
        </div>

        <div className="resource">
          <div className="resource-header">
            <span className="label">Memory</span>
            <span className="value">{formatBytes(stats.memoryUsedBytes)} / {formatBytes(stats.memoryTotalBytes)}</span>
          </div>
          <div className="meter">
            <div
              className="fill"
              style={{
                width: `${memoryPercent}%`,
                backgroundColor: memoryPercent > 80 ? '#ef4444' : memoryPercent > 60 ? '#eab308' : '#22c55e',
              }}
            />
          </div>
        </div>

        <div className="resource">
          <div className="resource-header">
            <span className="label">Disk</span>
            <span className="value">{formatBytes(stats.diskUsedBytes)} / {formatBytes(stats.diskTotalBytes)}</span>
          </div>
          <div className="meter">
            <div
              className="fill"
              style={{
                width: `${diskPercent}%`,
                backgroundColor: diskPercent > 80 ? '#ef4444' : diskPercent > 60 ? '#eab308' : '#22c55e',
              }}
            />
          </div>
        </div>
      </div>

      <div className="load-average">
        <span className="label">Load Average:</span>
        <span className="values">
          {stats.loadAverage[0].toFixed(2)} /
          {stats.loadAverage[1].toFixed(2)} /
          {stats.loadAverage[2].toFixed(2)}
        </span>
      </div>
    </div>
  );
}

function AlertsPanel() {
  const { alerts, rules, acknowledgeAlert, toggleRule } = useAlerts();
  const [activeTab, setActiveTab] = useState<'active' | 'rules'>('active');

  const severityIcons = {
    Critical: '🔴',
    Warning: '🟡',
    Info: '🔵',
  };

  return (
    <div className="alerts-panel">
      <header>
        <h3>Alerts</h3>
        <div className="tabs">
          <button
            className={activeTab === 'active' ? 'active' : ''}
            onClick={() => setActiveTab('active')}
          >
            Active ({alerts.length})
          </button>
          <button
            className={activeTab === 'rules' ? 'active' : ''}
            onClick={() => setActiveTab('rules')}
          >
            Rules
          </button>
        </div>
      </header>

      {activeTab === 'active' && (
        <div className="active-alerts">
          {alerts.length === 0 ? (
            <div className="no-alerts">No active alerts</div>
          ) : (
            alerts.map(alert => (
              <div key={alert.id} className={`alert ${alert.severity.toLowerCase()}`}>
                <span className="severity">{severityIcons[alert.severity]}</span>
                <div className="alert-content">
                  <span className="rule-name">{alert.ruleName}</span>
                  <span className="message">{alert.message}</span>
                  <span className="time">{new Date(alert.firedAt).toLocaleString()}</span>
                </div>
                <button onClick={() => acknowledgeAlert(alert.id)}>Acknowledge</button>
              </div>
            ))
          )}
        </div>
      )}

      {activeTab === 'rules' && (
        <div className="alert-rules">
          {rules.map(rule => (
            <div key={rule.id} className={`rule ${rule.enabled ? 'enabled' : 'disabled'}`}>
              <span className="severity">{severityIcons[rule.severity]}</span>
              <div className="rule-content">
                <span className="rule-name">{rule.name}</span>
                <span className="description">{rule.description}</span>
                <span className="condition">{rule.condition}</span>
              </div>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={rule.enabled}
                  onChange={e => toggleRule(rule.id, e.target.checked)}
                />
                <span className="slider" />
              </label>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function LogsPanel() {
  const [logs, setLogs] = useState<string[]>([]);
  const [filter, setFilter] = useState('');
  const [level, setLevel] = useState<'all' | 'error' | 'warn' | 'info'>('all');

  useEffect(() => {
    // Would use WebSocket for real-time logs
    fetch('/api/observability/logs?limit=100')
      .then(res => res.json())
      .then(setLogs);
  }, []);

  const filteredLogs = logs.filter(log => {
    if (filter && !log.toLowerCase().includes(filter.toLowerCase())) return false;
    if (level !== 'all') {
      if (level === 'error' && !log.includes('[ERROR]')) return false;
      if (level === 'warn' && !log.includes('[WARN]')) return false;
      if (level === 'info' && !log.includes('[INFO]')) return false;
    }
    return true;
  });

  return (
    <div className="logs-panel">
      <header>
        <h3>Logs</h3>
        <div className="filters">
          <input
            type="text"
            placeholder="Filter logs..."
            value={filter}
            onChange={e => setFilter(e.target.value)}
          />
          <select value={level} onChange={e => setLevel(e.target.value as typeof level)}>
            <option value="all">All Levels</option>
            <option value="error">Errors</option>
            <option value="warn">Warnings</option>
            <option value="info">Info</option>
          </select>
        </div>
      </header>

      <div className="log-viewer">
        {filteredLogs.map((log, i) => (
          <div
            key={i}
            className={`log-line ${
              log.includes('[ERROR]') ? 'error' :
              log.includes('[WARN]') ? 'warn' : 'info'
            }`}
          >
            {log}
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

interface ObservabilityDashboardProps {
  initialTab?: string;
}

export function ObservabilityDashboard({ initialTab = 'overview' }: ObservabilityDashboardProps) {
  const { data, loading, refresh } = useDashboard();
  const [activeTab, setActiveTab] = useState(initialTab);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'metrics', label: 'Metrics', icon: '📈' },
    { id: 'alerts', label: 'Alerts', icon: '🔔' },
    { id: 'logs', label: 'Logs', icon: '📋' },
  ];

  if (loading) {
    return <div className="observability-dashboard loading">Loading dashboard...</div>;
  }

  if (!data) {
    return <div className="observability-dashboard error">Failed to load dashboard</div>;
  }

  return (
    <div className="observability-dashboard">
      <nav className="dashboard-tabs">
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
        <button onClick={refresh} className="refresh-btn">Refresh</button>
      </nav>

      <div className="dashboard-content">
        {activeTab === 'overview' && (
          <div className="overview-grid">
            <OverviewPanel data={data} />
            <SystemMetricsPanel stats={data.systemStats} />
            <EmailMetricsPanel stats={data.emailStats} />
            <SyncMetricsPanel stats={data.syncStats} />
          </div>
        )}

        {activeTab === 'metrics' && (
          <div className="metrics-grid">
            <ApiMetricsPanel stats={data.apiStats} />
            <EmailMetricsPanel stats={data.emailStats} />
            <SyncMetricsPanel stats={data.syncStats} />
            <SystemMetricsPanel stats={data.systemStats} />
          </div>
        )}

        {activeTab === 'alerts' && <AlertsPanel />}
        {activeTab === 'logs' && <LogsPanel />}
      </div>
    </div>
  );
}

export default ObservabilityDashboard;
