// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

interface DashboardStats {
  users: { total: number; active: number; newToday: number };
  songs: { total: number; pending: number; flagged: number };
  reports: { total: number; pending: number; resolved: number };
  revenue: { total: string; today: string };
}

export default function AdminDashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/admin/stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-purple-500" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-8">Admin Dashboard</h1>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard
            title="Total Users"
            value={stats?.users.total || 0}
            subtitle={`${stats?.users.newToday || 0} new today`}
            color="blue"
          />
          <StatCard
            title="Total Songs"
            value={stats?.songs.total || 0}
            subtitle={`${stats?.songs.pending || 0} pending review`}
            color="green"
          />
          <StatCard
            title="Reports"
            value={stats?.reports.pending || 0}
            subtitle={`${stats?.reports.resolved || 0} resolved`}
            color="yellow"
          />
          <StatCard
            title="Revenue"
            value={stats?.revenue.total || '0 ETH'}
            subtitle={`${stats?.revenue.today || '0 ETH'} today`}
            color="purple"
          />
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <QuickActionCard
            title="Content Moderation"
            description="Review flagged content and reports"
            href="/admin/moderation"
            icon="🛡️"
          />
          <QuickActionCard
            title="User Management"
            description="Manage users, roles, and permissions"
            href="/admin/users"
            icon="👥"
          />
          <QuickActionCard
            title="Analytics"
            description="View platform analytics and metrics"
            href="/admin/analytics"
            icon="📊"
          />
        </div>

        {/* Recent Activity */}
        <div className="bg-gray-800 rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
          <RecentActivityList />
        </div>
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  subtitle,
  color,
}: {
  title: string;
  value: number | string;
  subtitle: string;
  color: 'blue' | 'green' | 'yellow' | 'purple';
}) {
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-green-600',
    yellow: 'from-yellow-500 to-yellow-600',
    purple: 'from-purple-500 to-purple-600',
  };

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color]} rounded-xl p-6`}>
      <p className="text-sm text-white/80">{title}</p>
      <p className="text-3xl font-bold mt-1">{value.toLocaleString()}</p>
      <p className="text-sm text-white/60 mt-2">{subtitle}</p>
    </div>
  );
}

function QuickActionCard({
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
      href={href}
      className="bg-gray-800 rounded-xl p-6 hover:bg-gray-700 transition-colors group"
    >
      <span className="text-4xl">{icon}</span>
      <h3 className="text-lg font-semibold mt-3 group-hover:text-purple-400 transition-colors">
        {title}
      </h3>
      <p className="text-gray-400 text-sm mt-1">{description}</p>
    </Link>
  );
}

function RecentActivityList() {
  const [activities, setActivities] = useState<Array<{
    id: string;
    type: string;
    message: string;
    timestamp: string;
  }>>([]);

  useEffect(() => {
    fetchActivities();
  }, []);

  const fetchActivities = async () => {
    try {
      const response = await fetch('/api/admin/activity');
      if (response.ok) {
        const data = await response.json();
        setActivities(data.activities);
      }
    } catch (error) {
      console.error('Failed to fetch activities:', error);
    }
  };

  if (activities.length === 0) {
    return <p className="text-gray-500">No recent activity</p>;
  }

  return (
    <div className="space-y-3">
      {activities.map((activity) => (
        <div
          key={activity.id}
          className="flex items-center justify-between py-2 border-b border-gray-700 last:border-0"
        >
          <div className="flex items-center gap-3">
            <span className="text-lg">
              {activity.type === 'report' && '🚨'}
              {activity.type === 'user' && '👤'}
              {activity.type === 'song' && '🎵'}
              {activity.type === 'moderation' && '🛡️'}
            </span>
            <span className="text-gray-300">{activity.message}</span>
          </div>
          <span className="text-gray-500 text-sm">
            {new Date(activity.timestamp).toLocaleTimeString()}
          </span>
        </div>
      ))}
    </div>
  );
}
