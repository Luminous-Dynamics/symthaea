// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useEffect } from 'react';

interface Report {
  id: string;
  type: 'song' | 'user' | 'comment' | 'playlist';
  reason: string;
  status: 'pending' | 'reviewing' | 'resolved' | 'dismissed';
  reportedBy: {
    address: string;
    name?: string;
  };
  target: {
    id: string;
    type: string;
    title?: string;
    name?: string;
  };
  createdAt: string;
  description?: string;
}

interface FlaggedContent {
  id: string;
  type: 'song' | 'comment';
  reason: string;
  confidence: number;
  content: {
    id: string;
    title?: string;
    text?: string;
    artist?: string;
  };
  flaggedAt: string;
}

type TabType = 'reports' | 'flagged' | 'queue';

export default function ModerationPage() {
  const [activeTab, setActiveTab] = useState<TabType>('reports');
  const [reports, setReports] = useState<Report[]>([]);
  const [flaggedContent, setFlaggedContent] = useState<FlaggedContent[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'pending' | 'reviewing'>('pending');

  useEffect(() => {
    fetchData();
  }, [activeTab, filter]);

  const fetchData = async () => {
    setLoading(true);
    try {
      if (activeTab === 'reports') {
        const response = await fetch(`/api/admin/reports?status=${filter}`);
        if (response.ok) {
          const data = await response.json();
          setReports(data.reports);
        }
      } else if (activeTab === 'flagged') {
        const response = await fetch('/api/admin/flagged');
        if (response.ok) {
          const data = await response.json();
          setFlaggedContent(data.content);
        }
      }
    } catch (error) {
      console.error('Failed to fetch data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleReportAction = async (reportId: string, action: 'resolve' | 'dismiss' | 'escalate') => {
    try {
      const response = await fetch(`/api/admin/reports/${reportId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
      if (response.ok) {
        fetchData();
      }
    } catch (error) {
      console.error('Failed to update report:', error);
    }
  };

  const handleContentAction = async (contentId: string, action: 'approve' | 'remove' | 'warn') => {
    try {
      const response = await fetch(`/api/admin/content/${contentId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
      if (response.ok) {
        fetchData();
      }
    } catch (error) {
      console.error('Failed to moderate content:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Content Moderation</h1>
          <div className="flex items-center gap-4">
            <span className="text-gray-400">Filter:</span>
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value as typeof filter)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2"
            >
              <option value="all">All</option>
              <option value="pending">Pending</option>
              <option value="reviewing">Reviewing</option>
            </select>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-800 mb-6">
          <TabButton
            active={activeTab === 'reports'}
            onClick={() => setActiveTab('reports')}
            label="User Reports"
            count={reports.filter(r => r.status === 'pending').length}
          />
          <TabButton
            active={activeTab === 'flagged'}
            onClick={() => setActiveTab('flagged')}
            label="Auto-Flagged"
            count={flaggedContent.length}
          />
          <TabButton
            active={activeTab === 'queue'}
            onClick={() => setActiveTab('queue')}
            label="Review Queue"
          />
        </div>

        {/* Content */}
        {loading ? (
          <div className="flex justify-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-purple-500" />
          </div>
        ) : (
          <>
            {activeTab === 'reports' && (
              <ReportsList reports={reports} onAction={handleReportAction} />
            )}
            {activeTab === 'flagged' && (
              <FlaggedContentList content={flaggedContent} onAction={handleContentAction} />
            )}
            {activeTab === 'queue' && (
              <ReviewQueue />
            )}
          </>
        )}
      </div>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  label,
  count,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  count?: number;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-3 font-medium transition-colors relative ${
        active
          ? 'text-purple-400 border-b-2 border-purple-400'
          : 'text-gray-400 hover:text-white'
      }`}
    >
      {label}
      {count !== undefined && count > 0 && (
        <span className="ml-2 bg-red-500 text-white text-xs rounded-full px-2 py-0.5">
          {count}
        </span>
      )}
    </button>
  );
}

function ReportsList({
  reports,
  onAction,
}: {
  reports: Report[];
  onAction: (id: string, action: 'resolve' | 'dismiss' | 'escalate') => void;
}) {
  if (reports.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        No reports to review
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {reports.map((report) => (
        <div
          key={report.id}
          className="bg-gray-800 rounded-xl p-6 border border-gray-700"
        >
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  report.type === 'song' ? 'bg-blue-500/20 text-blue-400' :
                  report.type === 'user' ? 'bg-green-500/20 text-green-400' :
                  report.type === 'comment' ? 'bg-yellow-500/20 text-yellow-400' :
                  'bg-purple-500/20 text-purple-400'
                }`}>
                  {report.type.toUpperCase()}
                </span>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  report.status === 'pending' ? 'bg-red-500/20 text-red-400' :
                  report.status === 'reviewing' ? 'bg-yellow-500/20 text-yellow-400' :
                  'bg-green-500/20 text-green-400'
                }`}>
                  {report.status}
                </span>
              </div>
              <h3 className="text-lg font-semibold">
                {report.target.title || report.target.name || report.target.id}
              </h3>
              <p className="text-gray-400 mt-1">
                <span className="font-medium">Reason:</span> {report.reason}
              </p>
              {report.description && (
                <p className="text-gray-400 mt-2">{report.description}</p>
              )}
              <p className="text-gray-500 text-sm mt-2">
                Reported by {report.reportedBy.name || report.reportedBy.address.slice(0, 10)}... • {new Date(report.createdAt).toLocaleString()}
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => onAction(report.id, 'resolve')}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm transition-colors"
              >
                Resolve
              </button>
              <button
                onClick={() => onAction(report.id, 'dismiss')}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg text-sm transition-colors"
              >
                Dismiss
              </button>
              <button
                onClick={() => onAction(report.id, 'escalate')}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-sm transition-colors"
              >
                Escalate
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function FlaggedContentList({
  content,
  onAction,
}: {
  content: FlaggedContent[];
  onAction: (id: string, action: 'approve' | 'remove' | 'warn') => void;
}) {
  if (content.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        No flagged content
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {content.map((item) => (
        <div
          key={item.id}
          className="bg-gray-800 rounded-xl p-6 border border-gray-700"
        >
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <span className="px-2 py-1 rounded text-xs font-medium bg-orange-500/20 text-orange-400">
                  AUTO-FLAGGED
                </span>
                <span className="text-gray-400 text-sm">
                  Confidence: {Math.round(item.confidence * 100)}%
                </span>
              </div>
              <h3 className="text-lg font-semibold">
                {item.content.title || item.content.text?.slice(0, 50) || 'Unknown'}
              </h3>
              <p className="text-gray-400 mt-1">
                <span className="font-medium">Reason:</span> {item.reason}
              </p>
              <p className="text-gray-500 text-sm mt-2">
                Flagged {new Date(item.flaggedAt).toLocaleString()}
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => onAction(item.id, 'approve')}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm transition-colors"
              >
                Approve
              </button>
              <button
                onClick={() => onAction(item.id, 'warn')}
                className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg text-sm transition-colors"
              >
                Warn User
              </button>
              <button
                onClick={() => onAction(item.id, 'remove')}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-sm transition-colors"
              >
                Remove
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ReviewQueue() {
  return (
    <div className="text-center py-12 text-gray-500">
      Review queue coming soon
    </div>
  );
}
