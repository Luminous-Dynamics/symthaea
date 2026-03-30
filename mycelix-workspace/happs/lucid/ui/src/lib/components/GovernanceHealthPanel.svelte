<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { invoke } from '@tauri-apps/api/core';
  import { listen, type UnlistenFn } from '@tauri-apps/api/event';

  interface GovernanceStats {
    proposalCount: number;
    avgPhiCoverage: number;
    totalPhiEnhanced: number;
    totalReputationOnly: number;
    overallPhiCoverage: number;
    totalAttested: number;
    totalSnapshot: number;
    totalUnavailable: number;
  }

  interface AgentAttestationStatus {
    phi: number;
    cycleId: number;
    capturedAtMs: number;
    ageSecs: number;
    decayedPhi: number;
    remainingSecs: number;
    isStale: boolean;
  }

  let stats: GovernanceStats | null = null;
  let myAttestation: AgentAttestationStatus | null = null;
  let loading = true;
  let error: string | null = null;
  let unlistenTally: UnlistenFn | null = null;
  let unlistenAttestation: UnlistenFn | null = null;
  let pollInterval: ReturnType<typeof setInterval> | null = null;
  let tickInterval: ReturnType<typeof setInterval> | null = null;
  let lastRefresh: number = 0;
  let now: number = Date.now();
  let isRefreshing = false;

  async function loadStats() {
    try {
      stats = await invoke<GovernanceStats>('get_governance_stats');
      error = null;
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
    }
  }

  async function loadMyAttestation() {
    try {
      myAttestation = await invoke<AgentAttestationStatus | null>('get_my_attestation_status');
    } catch {
      // Not critical — may not have an attestation yet
    }
  }

  async function refresh() {
    isRefreshing = true;
    await Promise.all([loadStats(), loadMyAttestation()]);
    lastRefresh = Date.now();
    isRefreshing = false;
  }

  function totalVotes(s: GovernanceStats): number {
    return s.totalPhiEnhanced + s.totalReputationOnly;
  }

  function coveragePercent(s: GovernanceStats): string {
    return (s.overallPhiCoverage * 100).toFixed(1);
  }

  function coverageColor(coverage: number): string {
    if (coverage >= 0.8) return '#22c55e';
    if (coverage >= 0.5) return '#eab308';
    return '#ef4444';
  }

  function coverageLabel(coverage: number): string {
    if (coverage >= 0.8) return 'Strong';
    if (coverage >= 0.5) return 'Moderate';
    if (coverage > 0) return 'Weak';
    return 'None';
  }

  function formatAge(secs: number): string {
    if (secs < 60) return `${Math.round(secs)}s ago`;
    if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
    return `${(secs / 3600).toFixed(1)}h ago`;
  }

  function formatRemaining(secs: number): string {
    if (secs <= 0) return 'expired';
    if (secs < 60) return `${Math.round(secs)}s left`;
    if (secs < 3600) return `${Math.round(secs / 60)}m left`;
    return `${(secs / 3600).toFixed(1)}h left`;
  }

  onMount(async () => {
    await Promise.all([loadStats(), loadMyAttestation()]);
    lastRefresh = Date.now();

    // Listen for real-time events from Tauri backend
    try {
      unlistenTally = await listen('tally-cached', () => {
        loadStats();
      });
      unlistenAttestation = await listen('attestation-cached', () => {
        loadMyAttestation();
      });
    } catch {
      // Not in Tauri environment — fall back to polling
    }

    // Poll every 30s as fallback for missed events
    pollInterval = setInterval(() => {
      loadStats();
      loadMyAttestation();
    }, 30_000);

    // Tick every 5s to keep "last updated" display reactive
    tickInterval = setInterval(() => {
      now = Date.now();
    }, 5_000);
  });

  onDestroy(() => {
    if (unlistenTally) unlistenTally();
    if (unlistenAttestation) unlistenAttestation();
    if (pollInterval) clearInterval(pollInterval);
    if (tickInterval) clearInterval(tickInterval);
  });
</script>

<div class="governance-health" role="region" aria-label="Governance health dashboard">
  <div class="health-header">
    <h3>Governance Health</h3>
    <div class="header-controls">
      {#if lastRefresh > 0}
        <span class="last-updated" title="Last refreshed at {new Date(lastRefresh).toLocaleTimeString()}">
          {formatAge((now - lastRefresh) / 1000)}
        </span>
      {/if}
      <button
        class="refresh-btn"
        class:spinning={isRefreshing}
        on:click={refresh}
        disabled={isRefreshing}
        title="Refresh governance data"
        aria-label="Refresh governance data"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <polyline points="23 4 23 10 17 10"></polyline>
          <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
        </svg>
      </button>
    </div>
  </div>

  <!-- Your Attestation Status -->
  <div class="my-attestation">
    <h4>Your Attestation</h4>
    {#if myAttestation}
      <div class="attestation-row">
        <div class="phi-display" class:stale={myAttestation.isStale}>
          <span class="phi-value">{myAttestation.decayedPhi.toFixed(2)}</span>
          <span class="phi-label">
            {#if myAttestation.isStale}
              Stale
            {:else}
              Active
            {/if}
          </span>
        </div>
        <div class="attestation-meta">
          <span>Raw: {myAttestation.phi.toFixed(3)}</span>
          <span>{formatAge(myAttestation.ageSecs)}</span>
          <span>{formatRemaining(myAttestation.remainingSecs)}</span>
          <span>Cycle #{myAttestation.cycleId}</span>
        </div>
      </div>
    {:else}
      <p class="no-attestation">No attestation recorded. Run Symthaea to generate Phi data.</p>
    {/if}
  </div>

  {#if loading}
    <p class="loading">Loading governance stats...</p>
  {:else if error}
    <p class="error">{error}</p>
  {:else if stats && stats.proposalCount > 0}
    <h4>Network Stats</h4>
    <div class="stats-grid">
      <div class="stat-card">
        <span class="stat-value">{stats.proposalCount}</span>
        <span class="stat-label">Proposals</span>
      </div>

      <div class="stat-card">
        <span class="stat-value">{totalVotes(stats)}</span>
        <span class="stat-label">Total Votes</span>
      </div>

      <div class="stat-card coverage" style="--coverage-color: {coverageColor(stats.overallPhiCoverage)}">
        <span class="stat-value">{coveragePercent(stats)}%</span>
        <span class="stat-label">Phi Coverage ({coverageLabel(stats.overallPhiCoverage)})</span>
      </div>

      <div class="stat-card">
        <span class="stat-value">{(stats.avgPhiCoverage * 100).toFixed(1)}%</span>
        <span class="stat-label">Avg Coverage/Proposal</span>
      </div>
    </div>

    <h4>Provenance Breakdown</h4>
    <div class="provenance-bar" role="img" aria-label="Provenance breakdown: {stats.totalAttested} attested, {stats.totalSnapshot} snapshot, {stats.totalUnavailable} unavailable">
      {#if stats.totalAttested > 0}
        <div
          class="bar-segment attested"
          style="flex: {stats.totalAttested}"
          title="{stats.totalAttested} attested votes (signed Symthaea attestation)"
          aria-label="{stats.totalAttested} attested"
        >
          {stats.totalAttested}
        </div>
      {/if}
      {#if stats.totalSnapshot > 0}
        <div
          class="bar-segment snapshot"
          style="flex: {stats.totalSnapshot}"
          title="{stats.totalSnapshot} snapshot votes (legacy unsigned)"
          aria-label="{stats.totalSnapshot} snapshot"
        >
          {stats.totalSnapshot}
        </div>
      {/if}
      {#if stats.totalUnavailable > 0}
        <div
          class="bar-segment unavailable"
          style="flex: {stats.totalUnavailable}"
          title="{stats.totalUnavailable} reputation-only votes (no Phi data)"
          aria-label="{stats.totalUnavailable} unavailable"
        >
          {stats.totalUnavailable}
        </div>
      {/if}
    </div>
    <div class="provenance-legend">
      <span class="legend-item" title="Signed Symthaea attestation"><span class="dot attested" aria-hidden="true"></span><span class="dot-icon" aria-hidden="true">{'\u2713'}</span> Attested ({stats.totalAttested})</span>
      <span class="legend-item" title="Legacy unsigned snapshot"><span class="dot snapshot" aria-hidden="true"></span><span class="dot-icon" aria-hidden="true">{'\u25CB'}</span> Snapshot ({stats.totalSnapshot})</span>
      <span class="legend-item" title="Reputation-only (no Phi data)"><span class="dot unavailable" aria-hidden="true"></span><span class="dot-icon" aria-hidden="true">{'\u2014'}</span> Unavailable ({stats.totalUnavailable})</span>
    </div>
  {:else}
    <p class="empty">No governance data cached yet.</p>
  {/if}
</div>

<style>
  .governance-health {
    padding: 1rem;
  }

  .health-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
  }

  h3 {
    margin: 0;
    font-size: 1.1rem;
    color: var(--text-primary, #e0e0e0);
  }

  .header-controls {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .last-updated {
    font-size: 0.7rem;
    color: var(--text-secondary, #888);
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    padding: 0;
    background: var(--surface-secondary, #16213e);
    border: 1px solid var(--border, #334155);
    border-radius: 6px;
    color: var(--text-secondary, #888);
    cursor: pointer;
    transition: all 0.2s;
  }

  .refresh-btn:hover:not(:disabled) {
    color: var(--text-primary, #e0e0e0);
    border-color: var(--text-secondary, #888);
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .refresh-btn svg {
    width: 14px;
    height: 14px;
  }

  .refresh-btn.spinning svg {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  h4 {
    margin: 1rem 0 0.5rem;
    font-size: 0.9rem;
    color: var(--text-secondary, #aaa);
  }

  .loading, .empty {
    color: var(--text-secondary, #888);
    font-size: 0.85rem;
  }

  .error {
    color: #ef4444;
    font-size: 0.85rem;
  }

  /* My Attestation */
  .my-attestation {
    padding: 0.75rem;
    border-radius: 8px;
    background: var(--surface-secondary, #16213e);
    margin-bottom: 0.5rem;
  }

  .my-attestation h4 {
    margin: 0 0 0.5rem;
    font-size: 0.85rem;
  }

  .attestation-row {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .phi-display {
    text-align: center;
    padding: 0.4rem 0.75rem;
    border-radius: 8px;
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.3);
    min-width: 70px;
  }

  .phi-display.stale {
    background: rgba(239, 68, 68, 0.15);
    border-color: rgba(239, 68, 68, 0.3);
  }

  .phi-value {
    display: block;
    font-size: 1.4rem;
    font-weight: 700;
    color: #22c55e;
  }

  .phi-display.stale .phi-value {
    color: #ef4444;
  }

  .phi-label {
    display: block;
    font-size: 0.65rem;
    color: var(--text-secondary, #888);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .attestation-meta {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    font-size: 0.75rem;
    color: var(--text-secondary, #888);
  }

  .no-attestation {
    font-size: 0.8rem;
    color: var(--text-secondary, #666);
    margin: 0;
  }

  /* Stats Grid */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 0.5rem;
  }

  .stat-card {
    padding: 0.5rem;
    border-radius: 6px;
    background: var(--surface-secondary, #16213e);
    text-align: center;
  }

  .stat-card.coverage {
    border-left: 3px solid var(--coverage-color, #888);
  }

  .stat-value {
    display: block;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text-primary, #e0e0e0);
  }

  .stat-label {
    display: block;
    font-size: 0.7rem;
    color: var(--text-secondary, #888);
    margin-top: 0.15rem;
  }

  /* Provenance Bar */
  .provenance-bar {
    display: flex;
    height: 24px;
    border-radius: 4px;
    overflow: hidden;
    gap: 1px;
  }

  .bar-segment {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 600;
    color: white;
    min-width: 20px;
  }

  .bar-segment.attested { background: #22c55e; }
  .bar-segment.snapshot { background: #3b82f6; }
  .bar-segment.unavailable {
    background: #6b7280;
    background-image: repeating-linear-gradient(
      45deg,
      transparent,
      transparent 3px,
      rgba(255, 255, 255, 0.1) 3px,
      rgba(255, 255, 255, 0.1) 6px
    );
  }

  .provenance-legend {
    display: flex;
    gap: 1rem;
    margin-top: 0.4rem;
    font-size: 0.75rem;
    color: var(--text-secondary, #aaa);
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
  }

  .dot.attested { background: #22c55e; }
  .dot.snapshot { background: #3b82f6; }
  .dot.unavailable {
    background: #6b7280;
    background-image: repeating-linear-gradient(
      45deg,
      transparent,
      transparent 2px,
      rgba(255, 255, 255, 0.15) 2px,
      rgba(255, 255, 255, 0.15) 4px
    );
  }

  .dot-icon {
    font-size: 0.65rem;
    font-weight: 700;
    margin-left: -2px;
    margin-right: 1px;
  }
</style>
