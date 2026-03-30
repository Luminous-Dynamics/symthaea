// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Admin Dashboard
 *
 * Main admin console with system overview and quick actions
 */

import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime_seconds: number;
  components: ComponentHealth[];
}

interface ComponentHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  latency_ms?: number;
  message?: string;
}

interface SystemMetrics {
  requests_total: number;
  requests_per_second: number;
  active_users: number;
  active_sessions: number;
  emails_stored: number;
  storage_used_bytes: number;
  cpu_usage_percent: number;
  memory_usage_percent: number;
}

interface RecentActivity {
  id: string;
  type: string;
  description: string;
  timestamp: string;
  user?: string;
}

export default function AdminDashboard() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [activities, setActivities] = useState<RecentActivity[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  async function fetchDashboardData() {
    try {
      const [healthRes, metricsRes] = await Promise.all([
        fetch('/api/admin/health'),
        fetch('/api/admin/metrics'),
      ]);

      if (healthRes.ok) setHealth(await healthRes.json());
      if (metricsRes.ok) setMetrics(await metricsRes.json());
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setLoading(false);
    }
  }

  function formatBytes(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = bytes;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex++;
    }
    return `${value.toFixed(1)} ${units[unitIndex]}`;
  }

  function formatUptime(seconds: number): string {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${days}d ${hours}h ${minutes}m`;
  }

  const statusColors = {
    healthy: 'bg-green-500',
    degraded: 'bg-yellow-500',
    unhealthy: 'bg-red-500',
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Admin Dashboard</h1>
        <div className="flex items-center gap-2">
          <span
            className={`w-3 h-3 rounded-full ${
              statusColors[health?.status || 'unhealthy']
            }`}
          />
          <span className="text-sm capitalize">{health?.status || 'Unknown'}</span>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Active Users"
          value={metrics?.active_users || 0}
          icon="👥"
        />
        <StatCard
          title="Active Sessions"
          value={metrics?.active_sessions || 0}
          icon="🔐"
        />
        <StatCard
          title="Emails Stored"
          value={metrics?.emails_stored || 0}
          icon="📧"
        />
        <StatCard
          title="Storage Used"
          value={formatBytes(metrics?.storage_used_bytes || 0)}
          icon="💾"
        />
      </div>

      {/* System Health */}
      <div className="bg-surface rounded-lg border border-border p-4">
        <h2 className="text-lg font-semibold mb-4">System Health</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {health?.components.map((component) => (
            <div
              key={component.name}
              className="flex items-center justify-between p-3 bg-background rounded border border-border"
            >
              <div className="flex items-center gap-2">
                <span
                  className={`w-2 h-2 rounded-full ${
                    statusColors[component.status]
                  }`}
                />
                <span className="capitalize">{component.name}</span>
              </div>
              {component.latency_ms !== undefined && (
                <span className="text-sm text-muted">{component.latency_ms}ms</span>
              )}
            </div>
          ))}
        </div>
        <div className="mt-4 text-sm text-muted">
          Version: {health?.version} | Uptime: {formatUptime(health?.uptime_seconds || 0)}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <QuickAction
          title="Manage Tenants"
          description="View and manage organizations"
          href="/admin/tenants"
          icon="🏢"
        />
        <QuickAction
          title="Manage Users"
          description="User accounts and permissions"
          href="/admin/users"
          icon="👤"
        />
        <QuickAction
          title="System Config"
          description="Configure system settings"
          href="/admin/config"
          icon="⚙️"
        />
        <QuickAction
          title="View Logs"
          description="System logs and events"
          href="/admin/logs"
          icon="📋"
        />
      </div>

      {/* Resource Usage */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-surface rounded-lg border border-border p-4">
          <h2 className="text-lg font-semibold mb-4">CPU Usage</h2>
          <div className="relative pt-1">
            <div className="flex mb-2 items-center justify-between">
              <span className="text-sm">{metrics?.cpu_usage_percent.toFixed(1)}%</span>
            </div>
            <div className="overflow-hidden h-2 text-xs flex rounded bg-gray-200">
              <div
                style={{ width: `${metrics?.cpu_usage_percent || 0}%` }}
                className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500"
              />
            </div>
          </div>
        </div>

        <div className="bg-surface rounded-lg border border-border p-4">
          <h2 className="text-lg font-semibold mb-4">Memory Usage</h2>
          <div className="relative pt-1">
            <div className="flex mb-2 items-center justify-between">
              <span className="text-sm">{metrics?.memory_usage_percent.toFixed(1)}%</span>
            </div>
            <div className="overflow-hidden h-2 text-xs flex rounded bg-gray-200">
              <div
                style={{ width: `${metrics?.memory_usage_percent || 0}%` }}
                className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-purple-500"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-surface rounded-lg border border-border p-4">
        <h2 className="text-lg font-semibold mb-4">Recent Activity</h2>
        <div className="space-y-2">
          {activities.length === 0 ? (
            <p className="text-muted text-center py-4">No recent activity</p>
          ) : (
            activities.map((activity) => (
              <div
                key={activity.id}
                className="flex items-center justify-between p-2 hover:bg-background rounded"
              >
                <div>
                  <span className="font-medium">{activity.type}</span>
                  <span className="text-muted ml-2">{activity.description}</span>
                </div>
                <span className="text-sm text-muted">{activity.timestamp}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  icon,
}: {
  title: string;
  value: string | number;
  icon: string;
}) {
  return (
    <div className="bg-surface rounded-lg border border-border p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-muted">{title}</p>
          <p className="text-2xl font-bold">{value}</p>
        </div>
        <span className="text-3xl">{icon}</span>
      </div>
    </div>
  );
}

function QuickAction({
  title,
  description,
  href,
  icon,
}: {
  title: string;
  description: string;
  href: string;
  icon: string;
}) {
  return (
    <Link
      to={href}
      className="bg-surface rounded-lg border border-border p-4 hover:border-primary transition-colors"
    >
      <div className="flex items-start gap-3">
        <span className="text-2xl">{icon}</span>
        <div>
          <h3 className="font-semibold">{title}</h3>
          <p className="text-sm text-muted">{description}</p>
        </div>
      </div>
    </Link>
  );
}
