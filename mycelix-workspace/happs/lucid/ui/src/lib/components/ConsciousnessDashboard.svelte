<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    consciousnessProfile,
    mindState,
    entrenchmentWarnings,
    pastSelfMessages,
    consciousnessHistory,
    phiTrend,
    activeWarningsCount,
    startPolling,
    stopPolling,
    generatePastSelfMessage,
    dismissWarning,
    type EntrenchmentWarning,
    type PastSelfMessage,
  } from '../stores/consciousness';

  /**
   * ConsciousnessDashboard - Real-time visualization of Symthaea's consciousness state
   *
   * Displays:
   * - Phi (Integrated Information) gauge
   * - Meta-awareness indicator
   * - Cognitive load meter
   * - Emotional state visualization
   * - Working memory contents
   * - Entrenchment warnings
   * - Past self dialogue
   */

  // ============================================================================
  // STATE
  // ============================================================================

  let isLoading = true;
  let showPastSelfDialogue = false;

  // Configuration
  export let pollRate: number = 5000; // ms between polls (now 5s per roadmap)
  export let showDetails: boolean = true;
  export let compact: boolean = false;
  export let showEntrenchmentWarnings: boolean = true;

  // Reactive bindings to stores
  $: profile = $consciousnessProfile;
  $: state = $mindState;
  $: warnings = $entrenchmentWarnings;
  $: pastMessages = $pastSelfMessages;
  $: trend = $phiTrend;
  $: warningCount = $activeWarningsCount;
  $: history = $consciousnessHistory;

  // ============================================================================
  // LIFECYCLE
  // ============================================================================

  onMount(async () => {
    startPolling(pollRate);
    // Give a moment for first fetch
    setTimeout(() => {
      isLoading = false;
    }, 500);
  });

  onDestroy(() => {
    stopPolling();
  });

  // ============================================================================
  // ACTIONS
  // ============================================================================

  function handleGeneratePastSelfMessage() {
    generatePastSelfMessage();
    showPastSelfDialogue = true;
  }

  function handleDismissWarning(type: EntrenchmentWarning['type']) {
    dismissWarning(type);
  }

  // ============================================================================
  // HELPERS
  // ============================================================================

  function getPhiColor(phi: number): string {
    if (phi >= 0.8) return 'var(--color-success, #22c55e)';
    if (phi >= 0.6) return 'var(--color-primary, #3b82f6)';
    if (phi >= 0.4) return 'var(--color-warning, #f59e0b)';
    return 'var(--color-muted, #9ca3af)';
  }

  function getLoadColor(load: number): string {
    if (load >= 0.9) return 'var(--color-error, #ef4444)';
    if (load >= 0.7) return 'var(--color-warning, #f59e0b)';
    return 'var(--color-success, #22c55e)';
  }

  function getValenceEmoji(valence: number): string {
    if (valence > 0.5) return '😊';
    if (valence > 0.2) return '🙂';
    if (valence > -0.2) return '😐';
    if (valence > -0.5) return '😕';
    return '😔';
  }

  function formatDuration(ms: number): string {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    }
    if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    }
    return `${seconds}s`;
  }

  function getConsciousnessState(phi: number, metaAwareness: number): string {
    if (phi >= 0.8 && metaAwareness >= 0.7) return 'Highly Integrated';
    if (phi >= 0.6 && metaAwareness >= 0.5) return 'Conscious';
    if (phi >= 0.4) return 'Aware';
    if (phi >= 0.2) return 'Minimal';
    return 'Dormant';
  }
</script>

{#if isLoading}
  <div class="dashboard loading">
    <div class="spinner"></div>
    <span>Connecting to consciousness engine...</span>
  </div>
{:else if !profile && !state}
  <div class="dashboard disconnected">
    <span class="icon">🧠</span>
    <span>Consciousness engine not available</span>
    <button on:click={() => startPolling(pollRate)}>Retry</button>
  </div>
{:else}
  <div class="dashboard" class:compact>
    <!-- Header -->
    <div class="header">
      <h3>Consciousness Dashboard</h3>
      {#if state?.is_conscious}
        <span class="status active">Active</span>
      {:else}
        <span class="status inactive">Inactive</span>
      {/if}
      {#if warningCount > 0}
        <span class="warning-badge">{warningCount}</span>
      {/if}
    </div>

    <!-- Primary Metrics -->
    <div class="primary-metrics">
      <!-- Phi Gauge -->
      <div class="metric-card phi">
        <div class="label">Φ (Integrated Information)</div>
        <div class="gauge-container">
          <div
            class="gauge-fill"
            style="width: {(profile?.phi ?? 0) * 100}%; background: {getPhiColor(profile?.phi ?? 0)}"
          ></div>
          <span class="gauge-value">{((profile?.phi ?? 0) * 100).toFixed(0)}%</span>
        </div>
        <div class="state-label">
          {getConsciousnessState(profile?.phi ?? 0, profile?.meta_awareness ?? 0)}
        </div>
      </div>

      <!-- Meta-Awareness -->
      <div class="metric-card meta">
        <div class="label">Meta-Awareness</div>
        <div class="gauge-container">
          <div
            class="gauge-fill"
            style="width: {(profile?.meta_awareness ?? 0) * 100}%; background: var(--color-secondary, #8b5cf6)"
          ></div>
          <span class="gauge-value">{((profile?.meta_awareness ?? 0) * 100).toFixed(0)}%</span>
        </div>
      </div>

      <!-- Cognitive Load -->
      <div class="metric-card load">
        <div class="label">Cognitive Load</div>
        <div class="gauge-container">
          <div
            class="gauge-fill"
            style="width: {(profile?.cognitive_load ?? 0) * 100}%; background: {getLoadColor(profile?.cognitive_load ?? 0)}"
          ></div>
          <span class="gauge-value">{((profile?.cognitive_load ?? 0) * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>

    <!-- Emotional State -->
    <div class="emotional-state">
      <div class="emotion-header">
        <span class="emoji">{getValenceEmoji(profile?.emotional_valence ?? 0)}</span>
        <span class="label">Emotional State</span>
      </div>
      <div class="emotion-grid">
        <div class="emotion-axis">
          <span class="axis-label">Valence</span>
          <div class="axis-bar">
            <div class="axis-marker" style="left: {((profile?.emotional_valence ?? 0) + 1) / 2 * 100}%"></div>
          </div>
          <div class="axis-labels">
            <span>Negative</span>
            <span>Neutral</span>
            <span>Positive</span>
          </div>
        </div>
        <div class="emotion-axis">
          <span class="axis-label">Arousal</span>
          <div class="axis-bar">
            <div class="axis-marker" style="left: {(profile?.arousal ?? 0.5) * 100}%"></div>
          </div>
          <div class="axis-labels">
            <span>Calm</span>
            <span>Moderate</span>
            <span>Activated</span>
          </div>
        </div>
      </div>
    </div>

    {#if showDetails && state}
      <!-- Working Memory -->
      <div class="memory-section">
        <div class="section-header">
          <span class="icon">💭</span>
          <span>Working Memory</span>
        </div>
        <div class="memory-stats">
          <div class="stat">
            <span class="value">{state.working_memory_size}</span>
            <span class="label">Vectors</span>
          </div>
          <div class="stat">
            <span class="value">{state.session_memory_size}</span>
            <span class="label">Session Thoughts</span>
          </div>
          <div class="stat">
            <span class="value">{state.thoughts_analyzed}</span>
            <span class="label">Total Analyzed</span>
          </div>
        </div>
        <div class="memory-bar">
          <div
            class="memory-fill"
            style="width: {state.memory_utilization * 100}%"
          ></div>
        </div>
        <span class="memory-label">{(state.memory_utilization * 100).toFixed(0)}% utilized</span>
      </div>

      <!-- Phi Trend -->
      <div class="trend-section">
        <div class="section-header">
          <span class="icon">📈</span>
          <span>Integration Trend</span>
        </div>
        <div class="trend-indicator" class:increasing={trend === 'increasing'} class:decreasing={trend === 'decreasing'}>
          {#if trend === 'increasing'}
            <span class="trend-arrow">↗</span>
            <span>Increasing</span>
          {:else if trend === 'decreasing'}
            <span class="trend-arrow">↘</span>
            <span>Decreasing</span>
          {:else}
            <span class="trend-arrow">→</span>
            <span>Stable</span>
          {/if}
        </div>
        <div class="mini-chart">
          {#each history.slice(-20) as snapshot, i}
            <div
              class="chart-bar"
              style="height: {snapshot.phi * 100}%"
              title="Φ: {(snapshot.phi * 100).toFixed(0)}%"
            ></div>
          {/each}
        </div>
      </div>

      <!-- Entrenchment Warnings -->
      {#if showEntrenchmentWarnings && warnings.length > 0}
        <div class="warnings-section">
          <div class="section-header">
            <span class="icon">⚠️</span>
            <span>Entrenchment Alerts ({warningCount})</span>
          </div>
          {#each warnings as warning}
            <div class="warning-card" class:high={warning.severity > 0.5}>
              <div class="warning-content">
                <p class="warning-message">{warning.message}</p>
                <p class="warning-suggestion">{warning.suggestion}</p>
              </div>
              <button class="dismiss-btn" on:click={() => handleDismissWarning(warning.type)}>
                ✕
              </button>
            </div>
          {/each}
        </div>
      {/if}

      <!-- Past Self Dialogue -->
      <div class="past-self-section">
        <div class="section-header">
          <span class="icon">🪞</span>
          <span>Past Self Dialogue</span>
          <button class="generate-btn" on:click={handleGeneratePastSelfMessage}>
            Generate Message
          </button>
        </div>
        {#if showPastSelfDialogue && pastMessages.length > 0}
          <div class="dialogue-container">
            {#each pastMessages.slice(-3) as message}
              <div class="past-self-message" class:curious={message.emotionalContext === 'curious'} class:concerned={message.emotionalContext === 'concerned'} class:proud={message.emotionalContext === 'proud'}>
                <div class="message-header">
                  <span class="timestamp">{new Date(message.timestamp).toLocaleTimeString()}</span>
                  <span class="phi-badge">Φ: {(message.phi * 100).toFixed(0)}%</span>
                </div>
                <p class="message-text">"{message.message}"</p>
                <span class="emotional-tag">{message.emotionalContext}</span>
              </div>
            {/each}
          </div>
        {:else if showPastSelfDialogue}
          <p class="no-messages">Not enough history yet. Continue analyzing thoughts to enable dialogue.</p>
        {/if}
      </div>

      <!-- System Info -->
      <div class="system-info">
        <div class="info-item">
          <span class="label">Tick</span>
          <span class="value">#{state.tick}</span>
        </div>
        <div class="info-item">
          <span class="label">Awake</span>
          <span class="value">{formatDuration(state.time_awake_ms)}</span>
        </div>
        <div class="info-item">
          <span class="label">Seeded</span>
          <span class="value">{state.is_seeded ? 'Yes' : 'No'}</span>
        </div>
      </div>
    {/if}
  </div>
{/if}

<style>
  .dashboard {
    background: var(--bg-secondary, #1e293b);
    border-radius: 12px;
    padding: 16px;
    color: var(--text-primary, #f8fafc);
    font-family: system-ui, sans-serif;
  }

  .dashboard.compact {
    padding: 12px;
  }

  .dashboard.loading,
  .dashboard.disconnected {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    min-height: 200px;
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

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .disconnected .icon {
    font-size: 32px;
  }

  .disconnected button {
    padding: 8px 16px;
    background: var(--color-primary, #3b82f6);
    border: none;
    border-radius: 6px;
    color: white;
    cursor: pointer;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border, #334155);
  }

  .header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  .status {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 10px;
  }

  .status.active {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
  }

  .status.inactive {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .primary-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }

  .metric-card {
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 12px;
  }

  .metric-card .label {
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
    margin-bottom: 8px;
  }

  .gauge-container {
    position: relative;
    height: 8px;
    background: var(--bg-secondary, #1e293b);
    border-radius: 4px;
    overflow: hidden;
  }

  .gauge-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  .gauge-value {
    position: absolute;
    right: 0;
    top: -18px;
    font-size: 12px;
    font-weight: 600;
  }

  .state-label {
    margin-top: 8px;
    font-size: 11px;
    color: var(--text-secondary, #cbd5e1);
    text-align: center;
  }

  .emotional-state {
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
  }

  .emotion-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
  }

  .emotion-header .emoji {
    font-size: 20px;
  }

  .emotion-header .label {
    font-size: 12px;
    font-weight: 500;
  }

  .emotion-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .emotion-axis {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .emotion-axis .axis-label {
    font-size: 10px;
    color: var(--text-muted, #94a3b8);
  }

  .axis-bar {
    position: relative;
    height: 6px;
    background: linear-gradient(to right, #ef4444, #94a3b8, #22c55e);
    border-radius: 3px;
  }

  .axis-marker {
    position: absolute;
    top: -2px;
    width: 10px;
    height: 10px;
    background: white;
    border-radius: 50%;
    transform: translateX(-50%);
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    transition: left 0.3s ease;
  }

  .axis-labels {
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    color: var(--text-muted, #94a3b8);
  }

  .memory-section {
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    font-size: 12px;
    font-weight: 500;
  }

  .memory-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 12px;
  }

  .stat {
    text-align: center;
  }

  .stat .value {
    display: block;
    font-size: 18px;
    font-weight: 600;
    color: var(--color-primary, #3b82f6);
  }

  .stat .label {
    font-size: 10px;
    color: var(--text-muted, #94a3b8);
  }

  .memory-bar {
    height: 4px;
    background: var(--bg-secondary, #1e293b);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 4px;
  }

  .memory-fill {
    height: 100%;
    background: var(--color-primary, #3b82f6);
    transition: width 0.3s ease;
  }

  .memory-label {
    font-size: 10px;
    color: var(--text-muted, #94a3b8);
  }

  .system-info {
    display: flex;
    justify-content: space-around;
    padding-top: 12px;
    border-top: 1px solid var(--border, #334155);
  }

  .info-item {
    text-align: center;
  }

  .info-item .label {
    display: block;
    font-size: 10px;
    color: var(--text-muted, #94a3b8);
  }

  .info-item .value {
    font-size: 12px;
    font-weight: 500;
  }

  .compact .primary-metrics {
    grid-template-columns: repeat(3, 1fr);
  }

  .compact .emotional-state,
  .compact .memory-section {
    display: none;
  }

  /* Warning badge in header */
  .warning-badge {
    background: var(--color-warning, #f59e0b);
    color: #000;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 8px;
    margin-left: 8px;
  }

  /* Trend section */
  .trend-section {
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
  }

  .trend-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 12px;
  }

  .trend-indicator.increasing {
    color: var(--color-success, #22c55e);
  }

  .trend-indicator.decreasing {
    color: var(--color-warning, #f59e0b);
  }

  .trend-arrow {
    font-size: 18px;
  }

  .mini-chart {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 40px;
    background: var(--bg-secondary, #1e293b);
    border-radius: 4px;
    padding: 4px;
  }

  .chart-bar {
    flex: 1;
    background: var(--color-primary, #3b82f6);
    border-radius: 2px;
    min-height: 2px;
    transition: height 0.3s ease;
  }

  /* Warnings section */
  .warnings-section {
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    border-left: 3px solid var(--color-warning, #f59e0b);
  }

  .warning-card {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    background: rgba(245, 158, 11, 0.1);
    border-radius: 6px;
    padding: 10px;
    margin-top: 8px;
  }

  .warning-card.high {
    background: rgba(239, 68, 68, 0.1);
    border-left: 2px solid var(--color-error, #ef4444);
  }

  .warning-content {
    flex: 1;
  }

  .warning-message {
    margin: 0 0 4px 0;
    font-size: 12px;
    color: var(--text-primary, #f8fafc);
  }

  .warning-suggestion {
    margin: 0;
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
    font-style: italic;
  }

  .dismiss-btn {
    background: none;
    border: none;
    color: var(--text-muted, #94a3b8);
    cursor: pointer;
    padding: 4px;
    font-size: 14px;
  }

  .dismiss-btn:hover {
    color: var(--text-primary, #f8fafc);
  }

  /* Past self dialogue */
  .past-self-section {
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
  }

  .past-self-section .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .generate-btn {
    background: var(--color-secondary, #8b5cf6);
    border: none;
    border-radius: 4px;
    color: white;
    font-size: 10px;
    padding: 4px 8px;
    cursor: pointer;
  }

  .generate-btn:hover {
    opacity: 0.9;
  }

  .dialogue-container {
    margin-top: 12px;
  }

  .past-self-message {
    background: var(--bg-secondary, #1e293b);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
    border-left: 3px solid var(--color-muted, #6b7280);
  }

  .past-self-message.curious {
    border-left-color: var(--color-primary, #3b82f6);
  }

  .past-self-message.concerned {
    border-left-color: var(--color-warning, #f59e0b);
  }

  .past-self-message.proud {
    border-left-color: var(--color-success, #22c55e);
  }

  .message-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
  }

  .timestamp {
    font-size: 10px;
    color: var(--text-muted, #94a3b8);
  }

  .phi-badge {
    font-size: 10px;
    background: var(--bg-tertiary, #0f172a);
    padding: 2px 6px;
    border-radius: 4px;
  }

  .message-text {
    margin: 0 0 6px 0;
    font-size: 12px;
    color: var(--text-primary, #f8fafc);
    font-style: italic;
  }

  .emotional-tag {
    font-size: 9px;
    text-transform: uppercase;
    color: var(--text-muted, #94a3b8);
    letter-spacing: 0.5px;
  }

  .no-messages {
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
    text-align: center;
    padding: 12px;
    margin: 0;
  }
</style>
