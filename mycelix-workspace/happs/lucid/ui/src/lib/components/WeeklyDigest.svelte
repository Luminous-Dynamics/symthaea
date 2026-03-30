<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import {
    generateWeeklyDigest,
    formatDigestPeriod,
    getDigestAge,
    getActiveNotifications,
    dismissNotification,
    subscribeToNotifications,
    type WeeklyDigest,
    type InsightNotification,
    type BlindSpot,
    type CrossDomainInsight,
    type SignificantShift,
  } from '../services/insight-surfacing';

  /**
   * WeeklyDigest - Comprehensive weekly summary of collective insights
   *
   * Shows:
   * - Summary and highlights
   * - Topic trends
   * - Blind spots
   * - Cross-domain connections
   * - Significant shifts
   * - Actionable recommendations
   */

  // ============================================================================
  // STATE
  // ============================================================================

  let digest: WeeklyDigest | null = null;
  let notifications: InsightNotification[] = [];
  let isLoading = true;
  let error: string | null = null;
  let activeTab: 'overview' | 'topics' | 'insights' | 'notifications' = 'overview';

  // ============================================================================
  // LIFECYCLE
  // ============================================================================

  let unsubscribe: (() => void) | null = null;

  onMount(() => {
    // Load digest
    generateWeeklyDigest()
      .then((result) => {
        digest = result;
      })
      .catch((e) => {
        error = e instanceof Error ? e.message : 'Failed to generate digest';
      })
      .finally(() => {
        isLoading = false;
      });

    // Subscribe to notifications
    unsubscribe = subscribeToNotifications((notifs) => {
      notifications = notifs.filter((n) => !n.dismissed);
    });

    return () => {
      if (unsubscribe) {
        unsubscribe();
      }
    };
  });

  // ============================================================================
  // HANDLERS
  // ============================================================================

  async function handleRefresh() {
    isLoading = true;
    try {
      digest = await generateWeeklyDigest();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to refresh';
    }
    isLoading = false;
  }

  function handleDismissNotification(id: string) {
    dismissNotification(id);
  }

  // ============================================================================
  // HELPERS
  // ============================================================================

  function getTrendIcon(trend: 'rising' | 'stable' | 'declining'): string {
    switch (trend) {
      case 'rising':
        return '📈';
      case 'declining':
        return '📉';
      default:
        return '➡️';
    }
  }

  function getBlindSpotIcon(reason: BlindSpot['reason']): string {
    switch (reason) {
      case 'high_uncertainty':
        return '❓';
      case 'low_coverage':
        return '🔍';
      case 'expert_disagreement':
        return '⚔️';
      case 'stale_data':
        return '⏰';
    }
  }

  function getShiftIcon(type: SignificantShift['shiftType']): string {
    switch (type) {
      case 'consensus_formed':
        return '🎯';
      case 'consensus_broken':
        return '💔';
      case 'confidence_change':
        return '📊';
      case 'new_topic_emerged':
        return '🌱';
      case 'topic_abandoned':
        return '🍂';
    }
  }

  function getNotificationPriorityColor(priority: InsightNotification['priority']): string {
    switch (priority) {
      case 'high':
        return 'var(--color-error, #ef4444)';
      case 'medium':
        return 'var(--color-warning, #f59e0b)';
      default:
        return 'var(--color-primary, #3b82f6)';
    }
  }
</script>

<div class="weekly-digest">
  <!-- Header -->
  <div class="header">
    <div class="title-section">
      <h2>Weekly Digest</h2>
      {#if digest}
        <span class="period">{formatDigestPeriod(digest)}</span>
        <span class="age">({getDigestAge(digest)})</span>
      {/if}
    </div>
    <button class="refresh-btn" on:click={handleRefresh} disabled={isLoading}>
      {isLoading ? 'Loading...' : 'Refresh'}
    </button>
  </div>

  {#if isLoading && !digest}
    <div class="loading">
      <div class="spinner"></div>
      <span>Generating weekly digest...</span>
    </div>
  {:else if error}
    <div class="error">
      <span class="icon">⚠️</span>
      <span>{error}</span>
    </div>
  {:else if !digest}
    <div class="empty">
      <span class="icon">📊</span>
      <p>No data available for digest.</p>
      <p class="hint">Share beliefs and participate in the collective to generate insights.</p>
    </div>
  {:else}
    <!-- Tabs -->
    <div class="tabs">
      <button
        class="tab"
        class:active={activeTab === 'overview'}
        on:click={() => (activeTab = 'overview')}
      >
        Overview
      </button>
      <button
        class="tab"
        class:active={activeTab === 'topics'}
        on:click={() => (activeTab = 'topics')}
      >
        Topics ({digest.topTopics.length})
      </button>
      <button
        class="tab"
        class:active={activeTab === 'insights'}
        on:click={() => (activeTab = 'insights')}
      >
        Insights ({digest.blindSpots.length + digest.crossDomainInsights.length})
      </button>
      <button
        class="tab"
        class:active={activeTab === 'notifications'}
        on:click={() => (activeTab = 'notifications')}
      >
        Alerts {#if notifications.length > 0}<span class="badge">{notifications.length}</span>{/if}
      </button>
    </div>

    <!-- Content -->
    <div class="content">
      {#if activeTab === 'overview'}
        <!-- Summary -->
        <div class="section">
          <p class="summary">{digest.summary}</p>
        </div>

        <!-- Metrics -->
        <div class="metrics-grid">
          <div class="metric">
            <span class="value">{digest.metrics.newBeliefs}</span>
            <span class="label">New Beliefs</span>
          </div>
          <div class="metric">
            <span class="value">{digest.metrics.consensusReached}</span>
            <span class="label">Consensus Formed</span>
          </div>
          <div class="metric">
            <span class="value">{Math.round(digest.metrics.averageConfidence * 100)}%</span>
            <span class="label">Avg Confidence</span>
          </div>
        </div>

        <!-- Highlights -->
        {#if digest.highlights.length > 0}
          <div class="section">
            <h3>✨ Highlights</h3>
            <ul class="list highlights">
              {#each digest.highlights as highlight}
                <li>{highlight}</li>
              {/each}
            </ul>
          </div>
        {/if}

        <!-- Concerns -->
        {#if digest.concerns.length > 0}
          <div class="section concerns">
            <h3>⚠️ Concerns</h3>
            <ul class="list">
              {#each digest.concerns as concern}
                <li>{concern}</li>
              {/each}
            </ul>
          </div>
        {/if}

        <!-- Recommendations -->
        {#if digest.recommendations.length > 0}
          <div class="section recommendations">
            <h3>💡 Recommendations</h3>
            <ul class="list">
              {#each digest.recommendations as rec}
                <li>{rec}</li>
              {/each}
            </ul>
          </div>
        {/if}
      {:else if activeTab === 'topics'}
        <!-- Topic Trends -->
        <div class="topic-list">
          {#each digest.topTopics as topic}
            <div class="topic-card">
              <div class="topic-header">
                <span class="trend-icon">{getTrendIcon(topic.trend)}</span>
                <span class="topic-name">{topic.tag}</span>
                <span class="belief-count">{topic.beliefCount} beliefs</span>
              </div>
              <div class="topic-stats">
                <div class="stat">
                  <span class="label">Trend</span>
                  <span class="value {topic.trend}">{topic.trend}</span>
                </div>
                {#if topic.consensusLevel !== null}
                  <div class="stat">
                    <span class="label">Consensus</span>
                    <span class="value">{Math.round(topic.consensusLevel * 100)}%</span>
                  </div>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      {:else if activeTab === 'insights'}
        <!-- Blind Spots -->
        {#if digest.blindSpots.length > 0}
          <div class="section">
            <h3>🔍 Blind Spots</h3>
            {#each digest.blindSpots as blindSpot}
              <div class="insight-card blind-spot">
                <div class="insight-header">
                  <span class="icon">{getBlindSpotIcon(blindSpot.reason)}</span>
                  <span class="topic">{blindSpot.topic}</span>
                  <span class="severity" style="opacity: {0.5 + blindSpot.severity * 0.5}">
                    Severity: {Math.round(blindSpot.severity * 100)}%
                  </span>
                </div>
                <p class="description">{blindSpot.description}</p>
                <p class="suggestion">💡 {blindSpot.suggestion}</p>
              </div>
            {/each}
          </div>
        {/if}

        <!-- Cross-Domain Insights -->
        {#if digest.crossDomainInsights.length > 0}
          <div class="section">
            <h3>🔗 Cross-Domain Connections</h3>
            {#each digest.crossDomainInsights as insight}
              <div class="insight-card cross-domain">
                <div class="insight-header">
                  <span class="domains">{insight.domains.join(' ↔ ')}</span>
                  <span class="strength">Strength: {Math.round(insight.strength * 100)}%</span>
                </div>
                <p class="description">{insight.description}</p>
                <span class="connection-type {insight.connectionType}">{insight.connectionType}</span>
              </div>
            {/each}
          </div>
        {/if}

        <!-- Significant Shifts -->
        {#if digest.significantShifts.length > 0}
          <div class="section">
            <h3>🔄 Significant Shifts</h3>
            {#each digest.significantShifts as shift}
              <div class="insight-card shift">
                <div class="insight-header">
                  <span class="icon">{getShiftIcon(shift.shiftType)}</span>
                  <span class="topic">{shift.topic}</span>
                </div>
                <div class="shift-change">
                  <span class="previous">{shift.previousState}</span>
                  <span class="arrow">→</span>
                  <span class="current">{shift.currentState}</span>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      {:else if activeTab === 'notifications'}
        {#if notifications.length === 0}
          <div class="empty-notifications">
            <span class="icon">🔔</span>
            <p>No new notifications</p>
          </div>
        {:else}
          <div class="notification-list">
            {#each notifications as notification}
              <div class="notification-card" style="border-left-color: {getNotificationPriorityColor(notification.priority)}">
                <div class="notification-header">
                  <span class="title">{notification.title}</span>
                  <button class="dismiss" on:click={() => handleDismissNotification(notification.id)}>
                    ✕
                  </button>
                </div>
                <p class="message">{notification.message}</p>
                <span class="time">{new Date(notification.timestamp).toLocaleString()}</span>
              </div>
            {/each}
          </div>
        {/if}
      {/if}
    </div>
  {/if}
</div>

<style>
  .weekly-digest {
    background: var(--bg-secondary, #1e293b);
    border-radius: 12px;
    padding: 20px;
    color: var(--text-primary, #f8fafc);
    max-height: 80vh;
    overflow-y: auto;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border, #334155);
  }

  .title-section h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
  }

  .period {
    font-size: 12px;
    color: var(--text-muted, #94a3b8);
    display: block;
    margin-top: 4px;
  }

  .age {
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
  }

  .refresh-btn {
    padding: 8px 16px;
    background: var(--color-primary, #3b82f6);
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 12px;
    cursor: pointer;
  }

  .refresh-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .loading,
  .empty,
  .error {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: var(--text-muted, #94a3b8);
    gap: 12px;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid var(--border, #334155);
    border-top-color: var(--color-primary, #3b82f6);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .error {
    color: var(--color-error, #ef4444);
  }

  .tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 16px;
    background: var(--bg-tertiary, #0f172a);
    padding: 4px;
    border-radius: 8px;
  }

  .tab {
    flex: 1;
    padding: 8px 12px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-muted, #94a3b8);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .tab.active {
    background: var(--color-primary, #3b82f6);
    color: white;
  }

  .tab .badge {
    background: var(--color-error, #ef4444);
    color: white;
    font-size: 10px;
    padding: 1px 5px;
    border-radius: 8px;
    margin-left: 4px;
  }

  .content {
    min-height: 200px;
  }

  .section {
    margin-bottom: 20px;
  }

  .section h3 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
  }

  .summary {
    font-size: 14px;
    line-height: 1.6;
    margin: 0;
    padding: 12px;
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 20px;
  }

  .metric {
    text-align: center;
    padding: 16px;
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
  }

  .metric .value {
    display: block;
    font-size: 24px;
    font-weight: 600;
    color: var(--color-primary, #3b82f6);
  }

  .metric .label {
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
  }

  .list {
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .list li {
    padding: 8px 12px;
    background: var(--bg-tertiary, #0f172a);
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 13px;
  }

  .section.concerns h3 {
    color: var(--color-warning, #f59e0b);
  }

  .section.recommendations h3 {
    color: var(--color-success, #22c55e);
  }

  .topic-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .topic-card {
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 12px;
  }

  .topic-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }

  .topic-name {
    font-weight: 500;
    flex: 1;
  }

  .belief-count {
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
  }

  .topic-stats {
    display: flex;
    gap: 16px;
  }

  .topic-stats .stat {
    font-size: 11px;
  }

  .topic-stats .label {
    color: var(--text-muted, #94a3b8);
    margin-right: 4px;
  }

  .topic-stats .rising {
    color: var(--color-success, #22c55e);
  }

  .topic-stats .declining {
    color: var(--color-error, #ef4444);
  }

  .insight-card {
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
  }

  .insight-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }

  .insight-header .topic,
  .insight-header .domains {
    font-weight: 500;
    flex: 1;
  }

  .insight-header .severity,
  .insight-header .strength {
    font-size: 10px;
    color: var(--text-muted, #94a3b8);
  }

  .description {
    margin: 0 0 8px 0;
    font-size: 12px;
    color: var(--text-secondary, #cbd5e1);
  }

  .suggestion {
    margin: 0;
    font-size: 11px;
    color: var(--color-success, #22c55e);
    font-style: italic;
  }

  .connection-type {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    text-transform: uppercase;
  }

  .connection-type.similarity {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
  }

  .connection-type.contradiction {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .connection-type.complementary {
    background: rgba(59, 130, 246, 0.2);
    color: #3b82f6;
  }

  .shift-change {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
  }

  .shift-change .previous {
    color: var(--text-muted, #94a3b8);
  }

  .shift-change .arrow {
    color: var(--color-primary, #3b82f6);
  }

  .shift-change .current {
    color: var(--text-primary, #f8fafc);
    font-weight: 500;
  }

  .empty-notifications {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px;
    color: var(--text-muted, #94a3b8);
  }

  .empty-notifications .icon {
    font-size: 32px;
    margin-bottom: 12px;
  }

  .notification-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .notification-card {
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 12px;
    border-left: 3px solid;
  }

  .notification-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 6px;
  }

  .notification-header .title {
    font-weight: 500;
    font-size: 13px;
  }

  .notification-header .dismiss {
    background: none;
    border: none;
    color: var(--text-muted, #94a3b8);
    cursor: pointer;
    padding: 2px;
  }

  .notification-card .message {
    margin: 0 0 8px 0;
    font-size: 12px;
    color: var(--text-secondary, #cbd5e1);
  }

  .notification-card .time {
    font-size: 10px;
    color: var(--text-muted, #94a3b8);
  }
</style>
