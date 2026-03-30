<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import type { Thought } from '@mycelix/lucid-client';
  import {
    type CollectiveView,
    type CollectiveStats,
    type BeliefShare,
    type EmergentPattern,
    getCollectiveStats,
    getCollectiveBeliefs,
    getPatterns,
    getSimulatedCollectiveView,
  } from '$services/collective-sensemaking';
  import { lucidClient, isConnected } from '$stores/holochain';
  import {
    signalConnectionStatus,
    unreadSignalCount,
    allSignals,
    connectSignals,
    disconnectSignals,
    markSignalsRead,
    subscribeToSignals,
    type AnySignal,
  } from '$services/realtime-signals';
  import ConsensusRealityMap from './ConsensusRealityMap.svelte';
  import VotingDashboard from './VotingDashboard.svelte';
  import AgentRelationshipNetwork from './AgentRelationshipNetwork.svelte';
  import CollectiveSensemakingDemo from './CollectiveSensemakingDemo.svelte';

  /**
   * CollectiveSensemakingPanel - Main container for collective sensemaking features
   *
   * Features:
   * - Stats header (beliefs count, patterns, peers)
   * - Tab navigation (Patterns, Consensus, Network, Vote)
   * - Auto-refresh toggle
   * - Connection status indicator
   * - Content area for sub-components
   */

  // ============================================================================
  // PROPS
  // ============================================================================

  /** Refresh interval in milliseconds */
  export let refreshInterval: number = 30000;

  /** Initial tab to display */
  export let initialTab: 'patterns' | 'consensus' | 'network' | 'vote' | 'demo' = 'consensus';

  /** Selected thought for voting (optional) */
  export let selectedThought: Thought | null = null;

  // ============================================================================
  // STATE
  // ============================================================================

  const dispatch = createEventDispatcher();

  let activeTab: 'patterns' | 'consensus' | 'network' | 'vote' | 'demo' = initialTab;
  let autoRefresh = true;
  let refreshTimer: number | null = null;
  let isLoading = true;
  let error: string | null = null;

  // Collective data
  let stats: CollectiveStats | null = null;
  let beliefs: BeliefShare[] = [];
  let patterns: EmergentPattern[] = [];
  let collectiveView: CollectiveView | null = null;

  // Signal state
  let showSignalPanel = false;
  let signalUnsubscribe: (() => void) | null = null;

  // Selected belief for detail view
  let selectedBelief: BeliefShare | null = null;

  // ============================================================================
  // DATA FETCHING
  // ============================================================================

  async function loadCollectiveData() {
    isLoading = true;
    error = null;

    try {
      if ($isConnected) {
        // Load from Holochain
        const [fetchedStats, fetchedBeliefs, fetchedPatterns] = await Promise.all([
          getCollectiveStats(),
          getCollectiveBeliefs(),
          getPatterns(),
        ]);

        stats = fetchedStats;
        beliefs = fetchedBeliefs;
        patterns = fetchedPatterns;

        collectiveView = {
          myShares: beliefs.slice(0, 10),
          publicShares: beliefs,
          consensusRecords: [],
          patterns,
          stats: stats || { total_belief_shares: 0, total_patterns: 0, active_validators: 0 },
        };
      } else {
        // Use simulated data
        collectiveView = getSimulatedCollectiveView();
        stats = collectiveView.stats;
        beliefs = collectiveView.publicShares;
        patterns = collectiveView.patterns;
      }
    } catch (e) {
      console.error('Failed to load collective data:', e);
      error = e instanceof Error ? e.message : 'Failed to load collective data';
      // Fall back to simulation
      collectiveView = getSimulatedCollectiveView();
      stats = collectiveView.stats;
      beliefs = collectiveView.publicShares;
      patterns = collectiveView.patterns;
    } finally {
      isLoading = false;
    }
  }

  function startAutoRefresh() {
    if (refreshTimer) clearInterval(refreshTimer);
    if (autoRefresh) {
      refreshTimer = setInterval(loadCollectiveData, refreshInterval) as unknown as number;
    }
  }

  function stopAutoRefresh() {
    if (refreshTimer) {
      clearInterval(refreshTimer);
      refreshTimer = null;
    }
  }

  // ============================================================================
  // EVENT HANDLERS
  // ============================================================================

  function handleTabChange(tab: typeof activeTab) {
    activeTab = tab;
    dispatch('tabchange', { tab });
  }

  function handleBeliefSelect(event: CustomEvent<BeliefShare>) {
    selectedBelief = event.detail;
    dispatch('beliefselect', event.detail);
  }

  function handleVoteSubmit(event: CustomEvent) {
    dispatch('votesubmit', event.detail);
    // Refresh data after vote
    loadCollectiveData();
  }

  function toggleAutoRefresh() {
    autoRefresh = !autoRefresh;
    if (autoRefresh) {
      startAutoRefresh();
    } else {
      stopAutoRefresh();
    }
  }

  // ============================================================================
  // SIGNAL HANDLERS
  // ============================================================================

  function handleSignal(signal: AnySignal) {
    // Refresh data when relevant signals arrive
    if (signal.type === 'belief_shared' || signal.type === 'consensus_changed') {
      loadCollectiveData();
    }
  }

  function toggleSignalPanel() {
    showSignalPanel = !showSignalPanel;
    if (showSignalPanel) {
      markSignalsRead();
    }
  }

  // ============================================================================
  // LIFECYCLE
  // ============================================================================

  onMount(async () => {
    await loadCollectiveData();
    startAutoRefresh();

    // Connect to real-time signals
    await connectSignals();
    signalUnsubscribe = subscribeToSignals('*', handleSignal);
  });

  onDestroy(() => {
    stopAutoRefresh();
    if (signalUnsubscribe) {
      signalUnsubscribe();
    }
    disconnectSignals();
  });

  // ============================================================================
  // REACTIVE
  // ============================================================================

  $: if (autoRefresh && refreshInterval) {
    startAutoRefresh();
  }
</script>

<div class="collective-panel">
  <!-- Header with stats -->
  <div class="panel-header">
    <div class="title-section">
      <h2>Collective Sensemaking</h2>
      <div class="status-indicators">
        <div class="connection-status" class:connected={$isConnected}>
          <span class="status-dot"></span>
          <span class="status-text">{$isConnected ? 'Holochain' : 'Simulated'}</span>
        </div>
        <button
          class="signal-indicator"
          class:active={$signalConnectionStatus === 'connected' || $signalConnectionStatus === 'simulated'}
          class:has-unread={$unreadSignalCount > 0}
          on:click={toggleSignalPanel}
          title="Real-time signals ({$signalConnectionStatus})"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M5 12.55a11 11 0 0 1 14.08 0" />
            <path d="M1.42 9a16 16 0 0 1 21.16 0" />
            <path d="M8.53 16.11a6 6 0 0 1 6.95 0" />
            <circle cx="12" cy="20" r="1" fill="currentColor" />
          </svg>
          {#if $unreadSignalCount > 0}
            <span class="unread-badge">{$unreadSignalCount > 99 ? '99+' : $unreadSignalCount}</span>
          {/if}
        </button>
      </div>
    </div>

    <div class="stats-row">
      <div class="stat">
        <span class="stat-value">{stats?.total_belief_shares ?? 0}</span>
        <span class="stat-label">beliefs</span>
      </div>
      <div class="stat">
        <span class="stat-value">{stats?.total_patterns ?? 0}</span>
        <span class="stat-label">patterns</span>
      </div>
      <div class="stat">
        <span class="stat-value">{stats?.active_validators ?? 0}</span>
        <span class="stat-label">peers</span>
      </div>
      <div class="stat-divider"></div>
      <button
        class="refresh-toggle"
        class:active={autoRefresh}
        on:click={toggleAutoRefresh}
        title={autoRefresh ? 'Disable auto-refresh' : 'Enable auto-refresh'}
      >
        <svg class="refresh-icon" class:spinning={autoRefresh && isLoading} viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      </button>
    </div>
  </div>

  <!-- Tab navigation -->
  <div class="tab-nav">
    <button
      class="tab-btn"
      class:active={activeTab === 'patterns'}
      on:click={() => handleTabChange('patterns')}
    >
      <span class="tab-icon">*</span>
      Patterns
    </button>
    <button
      class="tab-btn"
      class:active={activeTab === 'consensus'}
      on:click={() => handleTabChange('consensus')}
    >
      <span class="tab-icon">O</span>
      Consensus
    </button>
    <button
      class="tab-btn"
      class:active={activeTab === 'network'}
      on:click={() => handleTabChange('network')}
    >
      <span class="tab-icon">@</span>
      Network
    </button>
    <button
      class="tab-btn"
      class:active={activeTab === 'vote'}
      on:click={() => handleTabChange('vote')}
    >
      <span class="tab-icon">!</span>
      Vote
    </button>
    <button
      class="tab-btn demo-tab"
      class:active={activeTab === 'demo'}
      on:click={() => handleTabChange('demo')}
      title="Interactive walkthrough"
    >
      <span class="tab-icon">?</span>
      Demo
    </button>
  </div>

  <!-- Content area -->
  <div class="tab-content">
    {#if isLoading && !collectiveView}
      <div class="loading-state">
        <div class="spinner"></div>
        <span>Loading collective data...</span>
      </div>
    {:else if error && !collectiveView}
      <div class="error-state">
        <span class="error-icon">!</span>
        <span>{error}</span>
        <button on:click={loadCollectiveData}>Retry</button>
      </div>
    {:else}
      {#if activeTab === 'patterns'}
        <div class="patterns-tab">
          {#if patterns.length === 0}
            <div class="empty-state">
              <span class="empty-icon">*</span>
              <p>No emergent patterns detected yet.</p>
              <p class="empty-hint">Share more beliefs to discover collective patterns.</p>
            </div>
          {:else}
            <div class="patterns-list">
              {#each patterns as pattern (pattern.pattern_id)}
                <div class="pattern-card">
                  <div class="pattern-header">
                    <span class="pattern-type">{pattern.pattern_type}</span>
                    <span class="pattern-confidence">{Math.round(pattern.confidence * 100)}%</span>
                  </div>
                  <p class="pattern-description">{pattern.description}</p>
                  <div class="pattern-meta">
                    <span>{pattern.belief_hashes.length} beliefs</span>
                    <span class="dot"></span>
                    <span>{new Date(pattern.detected_at / 1000).toLocaleDateString()}</span>
                  </div>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      {:else if activeTab === 'consensus'}
        <ConsensusRealityMap
          {collectiveView}
          on:nodeselect={handleBeliefSelect}
        />
      {:else if activeTab === 'network'}
        <AgentRelationshipNetwork />
      {:else if activeTab === 'vote'}
        <VotingDashboard
          belief={selectedBelief}
          thought={selectedThought}
          on:votesubmit={handleVoteSubmit}
        />
      {:else if activeTab === 'demo'}
        <CollectiveSensemakingDemo
          on:complete={() => handleTabChange('consensus')}
          on:skipdemo={() => handleTabChange('consensus')}
        />
      {/if}
    {/if}
  </div>

  <!-- Signal Panel (slide-out) -->
  {#if showSignalPanel}
    <div class="signal-panel">
      <div class="signal-panel-header">
        <h3>Recent Activity</h3>
        <button class="close-btn" on:click={() => showSignalPanel = false}>×</button>
      </div>
      <div class="signal-list">
        {#each $allSignals.slice(0, 20) as signal (signal.timestamp)}
          <div class="signal-item" class:belief={signal.type === 'belief_shared'} class:vote={signal.type === 'vote_cast'} class:consensus={signal.type === 'consensus_changed'}>
            <span class="signal-type">{signal.type.replace('_', ' ')}</span>
            <span class="signal-time">{new Date(signal.timestamp / 1000).toLocaleTimeString()}</span>
          </div>
        {:else}
          <div class="no-signals">No recent activity</div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .collective-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-secondary, #1e293b);
    border-radius: 12px;
    overflow: hidden;
  }

  /* Header */
  .panel-header {
    padding: 16px;
    border-bottom: 1px solid var(--border, #334155);
  }

  .title-section {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .title-section h2 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary, #f8fafc);
  }

  .connection-status {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: var(--bg-tertiary, #0f172a);
    border-radius: 12px;
    font-size: 11px;
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--color-warning, #f59e0b);
  }

  .connection-status.connected .status-dot {
    background: var(--color-success, #22c55e);
  }

  .status-text {
    color: var(--text-muted, #94a3b8);
  }

  .status-indicators {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .signal-indicator {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    padding: 0;
    background: var(--bg-tertiary, #0f172a);
    border: 1px solid var(--border, #334155);
    border-radius: 6px;
    color: var(--text-muted, #94a3b8);
    cursor: pointer;
    transition: all 0.2s;
  }

  .signal-indicator:hover {
    background: var(--border, #334155);
    color: var(--text-primary, #f8fafc);
  }

  .signal-indicator.active {
    color: var(--color-success, #22c55e);
  }

  .signal-indicator.has-unread {
    border-color: var(--color-primary, #3b82f6);
  }

  .signal-indicator svg {
    width: 16px;
    height: 16px;
  }

  .unread-badge {
    position: absolute;
    top: -4px;
    right: -4px;
    min-width: 16px;
    height: 16px;
    padding: 0 4px;
    font-size: 10px;
    font-weight: 600;
    line-height: 16px;
    text-align: center;
    background: var(--color-primary, #3b82f6);
    border-radius: 8px;
    color: white;
  }

  /* Signal Panel */
  .signal-panel {
    position: absolute;
    top: 60px;
    right: 16px;
    width: 280px;
    max-height: 400px;
    background: var(--bg-secondary, #1e293b);
    border: 1px solid var(--border, #334155);
    border-radius: 8px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    z-index: 100;
    overflow: hidden;
  }

  .signal-panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border, #334155);
  }

  .signal-panel-header h3 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary, #f8fafc);
  }

  .signal-panel-header .close-btn {
    padding: 4px 8px;
    font-size: 16px;
    background: none;
    border: none;
    color: var(--text-muted, #94a3b8);
    cursor: pointer;
  }

  .signal-list {
    max-height: 340px;
    overflow-y: auto;
  }

  .signal-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border, #334155);
    font-size: 12px;
  }

  .signal-item:last-child {
    border-bottom: none;
  }

  .signal-item.belief .signal-type {
    color: var(--color-primary, #3b82f6);
  }

  .signal-item.vote .signal-type {
    color: var(--color-success, #22c55e);
  }

  .signal-item.consensus .signal-type {
    color: var(--color-accent, #a855f7);
  }

  .signal-type {
    font-weight: 500;
    text-transform: capitalize;
  }

  .signal-time {
    color: var(--text-muted, #94a3b8);
  }

  .no-signals {
    padding: 24px;
    text-align: center;
    color: var(--text-muted, #94a3b8);
  }

  .stats-row {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .stat-value {
    font-size: 18px;
    font-weight: 700;
    color: var(--color-primary, #3b82f6);
  }

  .stat-label {
    font-size: 10px;
    color: var(--text-muted, #94a3b8);
    text-transform: uppercase;
  }

  .stat-divider {
    width: 1px;
    height: 24px;
    background: var(--border, #334155);
    margin-left: auto;
  }

  .refresh-toggle {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border: none;
    border-radius: 6px;
    background: var(--bg-tertiary, #0f172a);
    color: var(--text-muted, #94a3b8);
    cursor: pointer;
    transition: all 0.2s;
  }

  .refresh-toggle:hover {
    background: var(--border, #334155);
    color: var(--text-primary, #f8fafc);
  }

  .refresh-toggle.active {
    color: var(--color-primary, #3b82f6);
  }

  .refresh-icon {
    width: 16px;
    height: 16px;
  }

  .refresh-icon.spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* Tab navigation */
  .tab-nav {
    display: flex;
    padding: 0 16px;
    gap: 4px;
    background: var(--bg-tertiary, #0f172a);
  }

  .tab-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 10px 16px;
    border: none;
    background: transparent;
    color: var(--text-muted, #94a3b8);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
  }

  .tab-btn:hover {
    color: var(--text-primary, #f8fafc);
  }

  .tab-btn.active {
    color: var(--color-primary, #3b82f6);
    border-bottom-color: var(--color-primary, #3b82f6);
  }

  .tab-icon {
    font-size: 14px;
    opacity: 0.7;
  }

  .tab-btn.demo-tab {
    margin-left: auto;
    color: var(--color-accent, #a855f7);
  }

  .tab-btn.demo-tab:hover {
    color: var(--color-accent-light, #c084fc);
  }

  .tab-btn.demo-tab.active {
    color: var(--color-accent, #a855f7);
    border-bottom-color: var(--color-accent, #a855f7);
  }

  /* Content area */
  .tab-content {
    flex: 1;
    overflow: auto;
    padding: 16px;
  }

  /* Loading & Error states */
  .loading-state,
  .error-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    gap: 12px;
    color: var(--text-muted, #94a3b8);
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid var(--border, #334155);
    border-top-color: var(--color-primary, #3b82f6);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .error-icon,
  .empty-icon {
    font-size: 32px;
    opacity: 0.5;
  }

  .error-state button {
    padding: 8px 16px;
    background: var(--color-primary, #3b82f6);
    border: none;
    border-radius: 6px;
    color: white;
    cursor: pointer;
  }

  .empty-hint {
    font-size: 12px;
    opacity: 0.7;
  }

  /* Patterns tab */
  .patterns-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .pattern-card {
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 12px;
    border: 1px solid var(--border, #334155);
  }

  .pattern-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .pattern-type {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--color-secondary, #8b5cf6);
    background: rgba(139, 92, 246, 0.15);
    padding: 2px 8px;
    border-radius: 4px;
  }

  .pattern-confidence {
    font-size: 12px;
    font-weight: 600;
    color: var(--color-success, #22c55e);
  }

  .pattern-description {
    margin: 0 0 8px;
    font-size: 13px;
    color: var(--text-primary, #f8fafc);
    line-height: 1.4;
  }

  .pattern-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
  }

  .dot {
    width: 3px;
    height: 3px;
    border-radius: 50%;
    background: currentColor;
  }
</style>
