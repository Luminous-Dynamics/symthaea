<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { thoughts } from '../stores/thoughts';
  import {
    analyzeCoherenceLocally,
    generateWorldviewProfile,
    checkCoherenceWithSymthaea,
    type CoherenceResult,
    type WorldviewProfile,
  } from '../services/symthaea';

  let coherenceResult: CoherenceResult | null = null;
  let worldviewProfile: WorldviewProfile | null = null;
  let isAnalyzing = false;
  let symthaeaConnected = false;

  async function runAnalysis() {
    if ($thoughts.length === 0) return;

    isAnalyzing = true;

    // Try Symthaea first
    const symthaeaResult = await checkCoherenceWithSymthaea($thoughts.map(t => t.id));

    if (symthaeaResult) {
      coherenceResult = symthaeaResult;
      symthaeaConnected = true;
    } else {
      // Fall back to local analysis
      coherenceResult = analyzeCoherenceLocally($thoughts);
      symthaeaConnected = false;
    }

    worldviewProfile = generateWorldviewProfile($thoughts);
    isAnalyzing = false;
  }

  function getScoreColor(score: number): string {
    if (score >= 0.8) return '#10b981';
    if (score >= 0.6) return '#3b82f6';
    if (score >= 0.4) return '#f59e0b';
    return '#ef4444';
  }

  function getSeverityColor(severity: string): string {
    switch (severity) {
      case 'high': return '#ef4444';
      case 'medium': return '#f59e0b';
      default: return '#6b7280';
    }
  }

  // Auto-analyze when thoughts change significantly
  $: if ($thoughts.length > 0 && !coherenceResult) {
    runAnalysis();
  }
</script>

<div class="coherence-panel">
  <header>
    <h3>Worldview Coherence</h3>
    {#if symthaeaConnected}
      <span class="badge connected">Symthaea Connected</span>
    {:else}
      <span class="badge local">Local Analysis</span>
    {/if}
    <button class="refresh-btn" on:click={runAnalysis} disabled={isAnalyzing}>
      {isAnalyzing ? '...' : '↻'}
    </button>
  </header>

  {#if isAnalyzing}
    <div class="loading">
      <div class="spinner"></div>
      <span>Analyzing coherence...</span>
    </div>
  {:else if coherenceResult}
    <!-- Overall Score -->
    <div class="overall-score">
      <div
        class="score-ring"
        style="--score: {coherenceResult.overallScore}; --color: {getScoreColor(coherenceResult.overallScore)}"
      >
        <span class="score-value">{Math.round(coherenceResult.overallScore * 100)}</span>
        <span class="score-label">Coherence</span>
      </div>
    </div>

    <!-- Dimension Scores -->
    <div class="dimensions">
      <div class="dimension">
        <span class="dim-label">Logical</span>
        <div class="dim-bar">
          <div
            class="dim-fill"
            style="width: {coherenceResult.dimensions.logical * 100}%; background: {getScoreColor(coherenceResult.dimensions.logical)}"
          ></div>
        </div>
        <span class="dim-value">{Math.round(coherenceResult.dimensions.logical * 100)}%</span>
      </div>

      <div class="dimension">
        <span class="dim-label">Temporal</span>
        <div class="dim-bar">
          <div
            class="dim-fill"
            style="width: {coherenceResult.dimensions.temporal * 100}%; background: {getScoreColor(coherenceResult.dimensions.temporal)}"
          ></div>
        </div>
        <span class="dim-value">{Math.round(coherenceResult.dimensions.temporal * 100)}%</span>
      </div>

      <div class="dimension">
        <span class="dim-label">Epistemic</span>
        <div class="dim-bar">
          <div
            class="dim-fill"
            style="width: {coherenceResult.dimensions.epistemic * 100}%; background: {getScoreColor(coherenceResult.dimensions.epistemic)}"
          ></div>
        </div>
        <span class="dim-value">{Math.round(coherenceResult.dimensions.epistemic * 100)}%</span>
      </div>

      <div class="dimension">
        <span class="dim-label">Harmonic</span>
        <div class="dim-bar">
          <div
            class="dim-fill"
            style="width: {coherenceResult.dimensions.harmonic * 100}%; background: {getScoreColor(coherenceResult.dimensions.harmonic)}"
          ></div>
        </div>
        <span class="dim-value">{Math.round(coherenceResult.dimensions.harmonic * 100)}%</span>
      </div>
    </div>

    <!-- Contradictions -->
    {#if coherenceResult.contradictions.length > 0}
      <div class="contradictions">
        <h4>Potential Contradictions ({coherenceResult.contradictions.length})</h4>
        <ul>
          {#each coherenceResult.contradictions.slice(0, 5) as contradiction}
            <li>
              <span class="severity" style="background: {getSeverityColor(contradiction.severity)}">
                {contradiction.severity}
              </span>
              <span class="desc">{contradiction.description}</span>
            </li>
          {/each}
        </ul>
      </div>
    {/if}

    <!-- Suggestions -->
    {#if coherenceResult.suggestions.length > 0}
      <div class="suggestions">
        <h4>Suggestions</h4>
        <ul>
          {#each coherenceResult.suggestions as suggestion}
            <li>{suggestion}</li>
          {/each}
        </ul>
      </div>
    {/if}

    <!-- Worldview Profile -->
    {#if worldviewProfile}
      <div class="worldview">
        <h4>Worldview Profile</h4>

        {#if worldviewProfile.coreBeliefs.length > 0}
          <div class="section">
            <span class="section-label">Core Belief Clusters</span>
            <div class="clusters">
              {#each worldviewProfile.coreBeliefs.slice(0, 5) as cluster}
                <span class="cluster" style="opacity: {0.5 + cluster.coherenceScore * 0.5}">
                  {cluster.name}
                  <span class="cluster-count">{cluster.thoughts.length}</span>
                </span>
              {/each}
            </div>
          </div>
        {/if}

        {#if worldviewProfile.valueHierarchy.length > 0}
          <div class="section">
            <span class="section-label">Value Hierarchy</span>
            <ol class="values">
              {#each worldviewProfile.valueHierarchy.slice(0, 5) as value, i}
                <li style="opacity: {1 - i * 0.15}">{value}</li>
              {/each}
            </ol>
          </div>
        {/if}

        {#if worldviewProfile.epistemicProfile.domainExpertise.length > 0}
          <div class="section">
            <span class="section-label">Domain Focus</span>
            <div class="domains">
              {#each worldviewProfile.epistemicProfile.domainExpertise as domain}
                <span class="domain">{domain}</span>
              {/each}
            </div>
          </div>
        {/if}
      </div>
    {/if}
  {:else}
    <div class="empty">
      <p>Add thoughts to see coherence analysis</p>
    </div>
  {/if}
</div>

<style>
  .coherence-panel {
    background: #1a1a2e;
    border: 1px solid #2a2a4e;
    border-radius: 12px;
    padding: 20px;
  }

  header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
  }

  header h3 {
    margin: 0;
    font-size: 1rem;
    color: #fff;
  }

  .badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
  }

  .badge.connected {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .badge.local {
    background: rgba(107, 114, 128, 0.2);
    color: #9ca3af;
  }

  .refresh-btn {
    margin-left: auto;
    width: 32px;
    height: 32px;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 6px;
    color: #888;
    cursor: pointer;
    font-size: 1rem;
  }

  .refresh-btn:hover:not(:disabled) {
    background: #3a3a5e;
    color: #fff;
  }

  .loading {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 40px;
    color: #888;
  }

  .spinner {
    width: 20px;
    height: 20px;
    border: 2px solid #333;
    border-top-color: #7c3aed;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* Overall Score */
  .overall-score {
    display: flex;
    justify-content: center;
    margin-bottom: 24px;
  }

  .score-ring {
    position: relative;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    background: conic-gradient(
      var(--color) calc(var(--score) * 360deg),
      #252540 calc(var(--score) * 360deg)
    );
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  .score-ring::before {
    content: '';
    position: absolute;
    inset: 10px;
    background: #1a1a2e;
    border-radius: 50%;
  }

  .score-value {
    position: relative;
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
  }

  .score-label {
    position: relative;
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
  }

  /* Dimensions */
  .dimensions {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 24px;
  }

  .dimension {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .dim-label {
    width: 80px;
    font-size: 0.8rem;
    color: #888;
  }

  .dim-bar {
    flex: 1;
    height: 8px;
    background: #252540;
    border-radius: 4px;
    overflow: hidden;
  }

  .dim-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
  }

  .dim-value {
    width: 40px;
    text-align: right;
    font-size: 0.8rem;
    color: #e5e5e5;
    font-family: monospace;
  }

  /* Contradictions */
  .contradictions {
    margin-bottom: 20px;
  }

  .contradictions h4,
  .suggestions h4,
  .worldview h4 {
    margin: 0 0 12px;
    font-size: 0.85rem;
    color: #888;
  }

  .contradictions ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  .contradictions li {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px;
    background: #252540;
    border-radius: 6px;
    margin-bottom: 6px;
  }

  .severity {
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    color: white;
  }

  .desc {
    font-size: 0.85rem;
    color: #a5a5c5;
  }

  /* Suggestions */
  .suggestions ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  .suggestions li {
    padding: 8px 12px;
    background: rgba(124, 58, 237, 0.1);
    border-left: 3px solid #7c3aed;
    border-radius: 0 6px 6px 0;
    margin-bottom: 6px;
    font-size: 0.85rem;
    color: #a5a5c5;
  }

  /* Worldview */
  .worldview {
    border-top: 1px solid #2a2a4e;
    padding-top: 20px;
    margin-top: 20px;
  }

  .section {
    margin-bottom: 16px;
  }

  .section-label {
    display: block;
    font-size: 0.7rem;
    color: #666;
    text-transform: uppercase;
    margin-bottom: 8px;
  }

  .clusters {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .cluster {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: #252540;
    border-radius: 4px;
    font-size: 0.85rem;
    color: #e5e5e5;
  }

  .cluster-count {
    padding: 1px 5px;
    background: #3a3a5e;
    border-radius: 3px;
    font-size: 0.7rem;
    color: #888;
  }

  .values {
    margin: 0;
    padding-left: 20px;
    color: #e5e5e5;
    font-size: 0.85rem;
  }

  .values li {
    margin-bottom: 4px;
  }

  .domains {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .domain {
    padding: 4px 10px;
    background: #252540;
    border-radius: 4px;
    font-size: 0.8rem;
    color: #888;
  }

  .empty {
    text-align: center;
    padding: 40px;
    color: #666;
  }
</style>
