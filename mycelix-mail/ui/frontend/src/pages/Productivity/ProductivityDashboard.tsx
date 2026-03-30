// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Productivity Intelligence Dashboard
 *
 * Provides response time analytics, relationship scores, productivity metrics,
 * email load forecasting, writing analytics, and goals/streaks.
 */

import React, { useState, useEffect } from 'react';

// ============================================================================
// Types
// ============================================================================

interface DashboardData {
  period: { start: string; end: string; label: string };
  emailVolume: EmailVolume;
  productivityScore: number;
  peakHours: PeakHour[];
  categoryBreakdown: CategoryStat[];
  comparison: PeriodComparison;
  inboxHealth: InboxHealth;
}

interface EmailVolume {
  sent: number;
  received: number;
  archived: number;
  deleted: number;
  dailyAverageSent: number;
  dailyAverageReceived: number;
}

interface PeakHour {
  hour: number;
  sentCount: number;
  receivedCount: number;
  isProductive: boolean;
}

interface CategoryStat {
  category: string;
  count: number;
  percentage: number;
}

interface PeriodComparison {
  sentChangePercent: number;
  receivedChangePercent: number;
  responseTimeChangePercent: number;
  direction: 'better' | 'same' | 'worse';
}

interface InboxHealth {
  unreadCount: number;
  oldestUnreadDays: number;
  inboxZeroDays: number;
  averageProcessingTimeHours: number;
}

interface ResponseTimeStats {
  avgResponseTimeHours: number;
  medianResponseTimeHours: number;
  fastestResponseMinutes: number;
  slowestResponseHours: number;
  responseRate: number;
  bySender: SenderStats[];
  trend: 'improving' | 'stable' | 'declining';
}

interface SenderStats {
  email: string;
  name?: string;
  avgResponseTimeHours: number;
  totalEmails: number;
  totalResponses: number;
}

interface RelationshipScore {
  contactEmail: string;
  contactName?: string;
  overallScore: number;
  components: {
    frequencyScore: number;
    recencyScore: number;
    reciprocityScore: number;
    responseScore: number;
    engagementScore: number;
  };
  relationshipType: string;
  trend: 'strengthening' | 'stable' | 'weakening';
}

interface EmailForecast {
  predictions: DailyPrediction[];
  weeklyPattern: { busiestDay: string; quietestDay: string };
  confidence: number;
}

interface DailyPrediction {
  date: string;
  predictedReceived: number;
  predictedSent: number;
  confidenceLow: number;
  confidenceHigh: number;
  isBusyDay: boolean;
}

interface Goal {
  id: string;
  goalType: string;
  target: number;
  current: number;
  period: string;
  progressPercent: number;
  status: 'on_track' | 'at_risk' | 'achieved' | 'failed';
}

interface Streak {
  streakType: string;
  currentCount: number;
  bestCount: number;
  isActive: boolean;
}

// ============================================================================
// Main Dashboard Component
// ============================================================================

export default function ProductivityDashboard() {
  const [period, setPeriod] = useState<7 | 30 | 90>(30);
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [responseStats, setResponseStats] = useState<ResponseTimeStats | null>(null);
  const [relationships, setRelationships] = useState<RelationshipScore[]>([]);
  const [forecast, setForecast] = useState<EmailForecast | null>(null);
  const [goals, setGoals] = useState<Goal[]>([]);
  const [streaks, setStreaks] = useState<Streak[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'response' | 'relationships' | 'forecast' | 'goals'>('overview');

  useEffect(() => {
    fetchAllData();
  }, [period]);

  const fetchAllData = async () => {
    setLoading(true);
    try {
      const [dashboardRes, responseRes, relationshipsRes, forecastRes, goalsRes] = await Promise.all([
        fetch(`/api/productivity/dashboard?days=${period}`),
        fetch(`/api/productivity/response-times?days=${period}`),
        fetch(`/api/productivity/relationships?limit=20`),
        fetch(`/api/productivity/forecast?days=14`),
        fetch(`/api/productivity/goals`),
      ]);

      setDashboard(await dashboardRes.json());
      setResponseStats(await responseRes.json());
      setRelationships(await relationshipsRes.json());
      setForecast(await forecastRes.json());
      const goalsData = await goalsRes.json();
      setGoals(goalsData.goals);
      setStreaks(goalsData.streaks);
    } catch (error) {
      console.error('Failed to fetch productivity data:', error);
    }
    setLoading(false);
  };

  if (loading) {
    return <div className="productivity-dashboard loading">Loading productivity insights...</div>;
  }

  return (
    <div className="productivity-dashboard">
      <header className="dashboard-header">
        <div className="title-section">
          <h1><ChartIcon /> Productivity Intelligence</h1>
          <p>Insights into your email habits and communication patterns</p>
        </div>
        <div className="period-selector">
          <button className={period === 7 ? 'active' : ''} onClick={() => setPeriod(7)}>
            7 days
          </button>
          <button className={period === 30 ? 'active' : ''} onClick={() => setPeriod(30)}>
            30 days
          </button>
          <button className={period === 90 ? 'active' : ''} onClick={() => setPeriod(90)}>
            90 days
          </button>
        </div>
      </header>

      <nav className="dashboard-tabs">
        <button
          className={activeTab === 'overview' ? 'active' : ''}
          onClick={() => setActiveTab('overview')}
        >
          <OverviewIcon /> Overview
        </button>
        <button
          className={activeTab === 'response' ? 'active' : ''}
          onClick={() => setActiveTab('response')}
        >
          <ClockIcon /> Response Time
        </button>
        <button
          className={activeTab === 'relationships' ? 'active' : ''}
          onClick={() => setActiveTab('relationships')}
        >
          <UsersIcon /> Relationships
        </button>
        <button
          className={activeTab === 'forecast' ? 'active' : ''}
          onClick={() => setActiveTab('forecast')}
        >
          <TrendIcon /> Forecast
        </button>
        <button
          className={activeTab === 'goals' ? 'active' : ''}
          onClick={() => setActiveTab('goals')}
        >
          <TargetIcon /> Goals
        </button>
      </nav>

      <main className="dashboard-content">
        {activeTab === 'overview' && dashboard && (
          <OverviewTab dashboard={dashboard} />
        )}
        {activeTab === 'response' && responseStats && (
          <ResponseTimeTab stats={responseStats} />
        )}
        {activeTab === 'relationships' && (
          <RelationshipsTab relationships={relationships} />
        )}
        {activeTab === 'forecast' && forecast && (
          <ForecastTab forecast={forecast} />
        )}
        {activeTab === 'goals' && (
          <GoalsTab goals={goals} streaks={streaks} />
        )}
      </main>
    </div>
  );
}

// ============================================================================
// Overview Tab
// ============================================================================

function OverviewTab({ dashboard }: { dashboard: DashboardData }) {
  const scoreColor = dashboard.productivityScore >= 70 ? 'good' :
    dashboard.productivityScore >= 40 ? 'moderate' : 'needs-attention';

  return (
    <div className="overview-tab">
      <div className="stats-row">
        <div className={`productivity-score ${scoreColor}`}>
          <div className="score-circle">
            <svg viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="45"
                fill="none"
                stroke="currentColor"
                strokeWidth="10"
                strokeDasharray={`${dashboard.productivityScore * 2.83} 283`}
                transform="rotate(-90 50 50)"
              />
            </svg>
            <span className="score-value">{Math.round(dashboard.productivityScore)}</span>
          </div>
          <span className="score-label">Productivity Score</span>
        </div>

        <div className="volume-stats">
          <div className="stat-card">
            <span className="stat-value">{dashboard.emailVolume.received}</span>
            <span className="stat-label">Received</span>
            <ChangeIndicator value={dashboard.comparison.receivedChangePercent} />
          </div>
          <div className="stat-card">
            <span className="stat-value">{dashboard.emailVolume.sent}</span>
            <span className="stat-label">Sent</span>
            <ChangeIndicator value={dashboard.comparison.sentChangePercent} />
          </div>
          <div className="stat-card">
            <span className="stat-value">{dashboard.emailVolume.archived}</span>
            <span className="stat-label">Archived</span>
          </div>
        </div>
      </div>

      <div className="charts-row">
        <div className="chart-card">
          <h3>Activity by Hour</h3>
          <HourlyChart data={dashboard.peakHours} />
        </div>

        <div className="chart-card">
          <h3>Categories</h3>
          <CategoryChart data={dashboard.categoryBreakdown} />
        </div>
      </div>

      <div className="health-section">
        <h3>Inbox Health</h3>
        <div className="health-metrics">
          <div className={`health-metric ${dashboard.inboxHealth.unreadCount < 10 ? 'good' : dashboard.inboxHealth.unreadCount < 50 ? 'moderate' : 'attention'}`}>
            <span className="metric-value">{dashboard.inboxHealth.unreadCount}</span>
            <span className="metric-label">Unread emails</span>
          </div>
          <div className={`health-metric ${dashboard.inboxHealth.oldestUnreadDays < 3 ? 'good' : dashboard.inboxHealth.oldestUnreadDays < 7 ? 'moderate' : 'attention'}`}>
            <span className="metric-value">{dashboard.inboxHealth.oldestUnreadDays}</span>
            <span className="metric-label">Oldest unread (days)</span>
          </div>
          <div className="health-metric">
            <span className="metric-value">{dashboard.inboxHealth.inboxZeroDays}</span>
            <span className="metric-label">Inbox zero days</span>
          </div>
          <div className="health-metric">
            <span className="metric-value">{dashboard.inboxHealth.averageProcessingTimeHours.toFixed(1)}h</span>
            <span className="metric-label">Avg processing time</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function ChangeIndicator({ value }: { value: number }) {
  const isPositive = value > 0;
  const isSignificant = Math.abs(value) > 5;

  return (
    <span className={`change-indicator ${isPositive ? 'up' : 'down'} ${isSignificant ? 'significant' : ''}`}>
      {isPositive ? '↑' : '↓'} {Math.abs(value).toFixed(0)}%
    </span>
  );
}

function HourlyChart({ data }: { data: PeakHour[] }) {
  const maxCount = Math.max(...data.map(d => d.sentCount + d.receivedCount));

  return (
    <div className="hourly-chart">
      {Array.from({ length: 24 }, (_, hour) => {
        const hourData = data.find(d => d.hour === hour) || { sentCount: 0, receivedCount: 0 };
        const total = hourData.sentCount + hourData.receivedCount;
        const height = maxCount > 0 ? (total / maxCount) * 100 : 0;

        return (
          <div
            key={hour}
            className="hour-bar"
            style={{ height: `${height}%` }}
            title={`${hour}:00 - Sent: ${hourData.sentCount}, Received: ${hourData.receivedCount}`}
          >
            <div className="sent-portion" style={{ height: `${hourData.sentCount / Math.max(total, 1) * 100}%` }} />
          </div>
        );
      })}
      <div className="hour-labels">
        <span>12am</span>
        <span>6am</span>
        <span>12pm</span>
        <span>6pm</span>
      </div>
    </div>
  );
}

function CategoryChart({ data }: { data: CategoryStat[] }) {
  const colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#00BCD4'];

  return (
    <div className="category-chart">
      <div className="category-bars">
        {data.slice(0, 6).map((cat, index) => (
          <div key={cat.category} className="category-item">
            <div className="category-label">{cat.category}</div>
            <div className="category-bar-wrapper">
              <div
                className="category-bar"
                style={{
                  width: `${cat.percentage}%`,
                  backgroundColor: colors[index % colors.length],
                }}
              />
            </div>
            <div className="category-value">{cat.count}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Response Time Tab
// ============================================================================

function ResponseTimeTab({ stats }: { stats: ResponseTimeStats }) {
  const trendIcon = stats.trend === 'improving' ? '↓' : stats.trend === 'declining' ? '↑' : '→';
  const trendColor = stats.trend === 'improving' ? 'good' : stats.trend === 'declining' ? 'attention' : 'neutral';

  return (
    <div className="response-time-tab">
      <div className="response-summary">
        <div className="summary-stat main">
          <span className="stat-value">{stats.avgResponseTimeHours.toFixed(1)}h</span>
          <span className="stat-label">Average Response Time</span>
          <span className={`trend ${trendColor}`}>
            {trendIcon} {stats.trend}
          </span>
        </div>

        <div className="summary-stats">
          <div className="summary-stat">
            <span className="stat-value">{stats.medianResponseTimeHours.toFixed(1)}h</span>
            <span className="stat-label">Median</span>
          </div>
          <div className="summary-stat">
            <span className="stat-value">{stats.fastestResponseMinutes}m</span>
            <span className="stat-label">Fastest</span>
          </div>
          <div className="summary-stat">
            <span className="stat-value">{stats.slowestResponseHours}h</span>
            <span className="stat-label">Slowest</span>
          </div>
          <div className="summary-stat">
            <span className="stat-value">{stats.responseRate.toFixed(0)}%</span>
            <span className="stat-label">Response Rate</span>
          </div>
        </div>
      </div>

      <div className="sender-breakdown">
        <h3>Response Time by Contact</h3>
        <table>
          <thead>
            <tr>
              <th>Contact</th>
              <th>Avg Response</th>
              <th>Emails</th>
              <th>Response Rate</th>
            </tr>
          </thead>
          <tbody>
            {stats.bySender.slice(0, 10).map(sender => (
              <tr key={sender.email}>
                <td>
                  <span className="sender-name">{sender.name || sender.email}</span>
                  {sender.name && <span className="sender-email">{sender.email}</span>}
                </td>
                <td>{sender.avgResponseTimeHours.toFixed(1)}h</td>
                <td>{sender.totalEmails}</td>
                <td>
                  {sender.totalEmails > 0
                    ? ((sender.totalResponses / sender.totalEmails) * 100).toFixed(0)
                    : 0}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ============================================================================
// Relationships Tab
// ============================================================================

function RelationshipsTab({ relationships }: { relationships: RelationshipScore[] }) {
  const [selectedContact, setSelectedContact] = useState<RelationshipScore | null>(null);

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'VeryActive': return '#4CAF50';
      case 'Active': return '#8BC34A';
      case 'Moderate': return '#FF9800';
      case 'Dormant': return '#9E9E9E';
      case 'New': return '#2196F3';
      default: return '#ccc';
    }
  };

  return (
    <div className="relationships-tab">
      <div className="relationships-grid">
        {relationships.map(rel => (
          <div
            key={rel.contactEmail}
            className={`relationship-card ${selectedContact?.contactEmail === rel.contactEmail ? 'selected' : ''}`}
            onClick={() => setSelectedContact(rel)}
          >
            <div className="contact-info">
              <div className="contact-avatar" style={{ borderColor: getTypeColor(rel.relationshipType) }}>
                {(rel.contactName || rel.contactEmail).charAt(0).toUpperCase()}
              </div>
              <div className="contact-details">
                <span className="contact-name">{rel.contactName || rel.contactEmail}</span>
                {rel.contactName && <span className="contact-email">{rel.contactEmail}</span>}
              </div>
            </div>
            <div className="relationship-score">
              <div className="score-bar">
                <div className="score-fill" style={{ width: `${rel.overallScore}%` }} />
              </div>
              <span className="score-value">{Math.round(rel.overallScore)}</span>
            </div>
            <div className="relationship-meta">
              <span className="relationship-type" style={{ color: getTypeColor(rel.relationshipType) }}>
                {rel.relationshipType}
              </span>
              <span className={`relationship-trend ${rel.trend}`}>
                {rel.trend === 'strengthening' ? '↗' : rel.trend === 'weakening' ? '↘' : '→'}
              </span>
            </div>
          </div>
        ))}
      </div>

      {selectedContact && (
        <div className="relationship-detail">
          <h3>Relationship Details: {selectedContact.contactName || selectedContact.contactEmail}</h3>
          <div className="component-scores">
            <ComponentScore label="Frequency" value={selectedContact.components.frequencyScore} />
            <ComponentScore label="Recency" value={selectedContact.components.recencyScore} />
            <ComponentScore label="Reciprocity" value={selectedContact.components.reciprocityScore} />
            <ComponentScore label="Response Time" value={selectedContact.components.responseScore} />
            <ComponentScore label="Engagement" value={selectedContact.components.engagementScore} />
          </div>
        </div>
      )}
    </div>
  );
}

function ComponentScore({ label, value }: { label: string; value: number }) {
  return (
    <div className="component-score">
      <span className="component-label">{label}</span>
      <div className="component-bar">
        <div className="component-fill" style={{ width: `${value}%` }} />
      </div>
      <span className="component-value">{Math.round(value)}</span>
    </div>
  );
}

// ============================================================================
// Forecast Tab
// ============================================================================

function ForecastTab({ forecast }: { forecast: EmailForecast }) {
  return (
    <div className="forecast-tab">
      <div className="forecast-summary">
        <div className="summary-item">
          <CalendarIcon />
          <span>Busiest day: <strong>{forecast.weeklyPattern.busiestDay}</strong></span>
        </div>
        <div className="summary-item">
          <CalendarIcon />
          <span>Quietest day: <strong>{forecast.weeklyPattern.quietestDay}</strong></span>
        </div>
        <div className="summary-item">
          <span>Forecast confidence: <strong>{(forecast.confidence * 100).toFixed(0)}%</strong></span>
        </div>
      </div>

      <div className="forecast-chart">
        <h3>14-Day Email Forecast</h3>
        <div className="forecast-bars">
          {forecast.predictions.map(pred => (
            <div
              key={pred.date}
              className={`forecast-day ${pred.isBusyDay ? 'busy' : ''}`}
            >
              <div className="day-bars">
                <div
                  className="received-bar"
                  style={{ height: `${pred.predictedReceived * 2}px` }}
                  title={`Expected: ${pred.predictedReceived} emails`}
                />
                <div className="confidence-range" style={{
                  top: `${(pred.confidenceHigh - pred.predictedReceived) * 2}px`,
                  height: `${(pred.confidenceHigh - pred.confidenceLow) * 2}px`,
                }} />
              </div>
              <span className="day-label">{formatDay(pred.date)}</span>
              <span className="day-value">{pred.predictedReceived}</span>
            </div>
          ))}
        </div>
        <div className="forecast-legend">
          <span><span className="legend-box received" /> Expected emails</span>
          <span><span className="legend-box confidence" /> Confidence range</span>
        </div>
      </div>
    </div>
  );
}

function formatDay(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', { weekday: 'short', day: 'numeric' });
}

// ============================================================================
// Goals Tab
// ============================================================================

function GoalsTab({ goals, streaks }: { goals: Goal[]; streaks: Streak[] }) {
  return (
    <div className="goals-tab">
      <div className="streaks-section">
        <h3><FlameIcon /> Active Streaks</h3>
        <div className="streaks-grid">
          {streaks.map(streak => (
            <div key={streak.streakType} className={`streak-card ${streak.isActive ? 'active' : 'inactive'}`}>
              <div className="streak-icon">
                {getStreakIcon(streak.streakType)}
              </div>
              <div className="streak-info">
                <span className="streak-name">{formatStreakType(streak.streakType)}</span>
                <div className="streak-counts">
                  <span className="current-streak">
                    <FlameIcon /> {streak.currentCount} day{streak.currentCount !== 1 ? 's' : ''}
                  </span>
                  <span className="best-streak">
                    Best: {streak.bestCount}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="goals-section">
        <h3><TargetIcon /> Current Goals</h3>
        <div className="goals-list">
          {goals.map(goal => (
            <div key={goal.id} className={`goal-card ${goal.status}`}>
              <div className="goal-info">
                <span className="goal-name">{formatGoalType(goal.goalType)}</span>
                <span className="goal-period">{goal.period}</span>
              </div>
              <div className="goal-progress">
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${Math.min(goal.progressPercent, 100)}%` }}
                  />
                </div>
                <span className="progress-text">
                  {goal.current} / {goal.target}
                </span>
              </div>
              <div className={`goal-status ${goal.status}`}>
                {goal.status === 'achieved' && <CheckIcon />}
                {goal.status === 'on_track' && <TrendUpIcon />}
                {goal.status === 'at_risk' && <AlertIcon />}
                {goal.status === 'failed' && <XIcon />}
                {goal.status.replace('_', ' ')}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function getStreakIcon(type: string) {
  switch (type) {
    case 'InboxZero': return <InboxIcon />;
    case 'QuickResponse': return <ClockIcon />;
    case 'DailyProcessing': return <CheckSquareIcon />;
    case 'NoUnread': return <MailIcon />;
    default: return <StarIcon />;
  }
}

function formatStreakType(type: string): string {
  switch (type) {
    case 'InboxZero': return 'Inbox Zero';
    case 'QuickResponse': return 'Quick Responder';
    case 'DailyProcessing': return 'Daily Processing';
    case 'NoUnread': return 'No Unread';
    default: return type;
  }
}

function formatGoalType(type: string): string {
  switch (type) {
    case 'InboxZero': return 'Reach Inbox Zero';
    case 'ResponseTime': return 'Response Time';
    case 'ProcessedEmails': return 'Process Emails';
    case 'SentEmails': return 'Send Emails';
    case 'UnsubscribeCount': return 'Unsubscribes';
    default: return type;
  }
}

// ============================================================================
// Icon Components
// ============================================================================

function ChartIcon() {
  return <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="20" x2="18" y2="10" />
    <line x1="12" y1="20" x2="12" y2="4" />
    <line x1="6" y1="20" x2="6" y2="14" />
  </svg>;
}

function OverviewIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="3" width="7" height="7" />
    <rect x="14" y="3" width="7" height="7" />
    <rect x="14" y="14" width="7" height="7" />
    <rect x="3" y="14" width="7" height="7" />
  </svg>;
}

function ClockIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </svg>;
}

function UsersIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </svg>;
}

function TrendIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
    <polyline points="17 6 23 6 23 12" />
  </svg>;
}

function TargetIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <circle cx="12" cy="12" r="6" />
    <circle cx="12" cy="12" r="2" />
  </svg>;
}

function CalendarIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
    <line x1="16" y1="2" x2="16" y2="6" />
    <line x1="8" y1="2" x2="8" y2="6" />
    <line x1="3" y1="10" x2="21" y2="10" />
  </svg>;
}

function FlameIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z" />
  </svg>;
}

function InboxIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="22 12 16 12 14 15 10 15 8 12 2 12" />
    <path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" />
  </svg>;
}

function CheckSquareIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="9 11 12 14 22 4" />
    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
  </svg>;
}

function MailIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
    <polyline points="22,6 12,13 2,6" />
  </svg>;
}

function StarIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
  </svg>;
}

function CheckIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="20 6 9 17 4 12" />
  </svg>;
}

function TrendUpIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
    <polyline points="17 6 23 6 23 12" />
  </svg>;
}

function AlertIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="8" x2="12" y2="12" />
    <line x1="12" y1="16" x2="12.01" y2="16" />
  </svg>;
}

function XIcon() {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>;
}
