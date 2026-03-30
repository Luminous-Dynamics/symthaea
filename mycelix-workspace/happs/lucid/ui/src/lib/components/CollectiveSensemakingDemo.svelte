<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import {
    shareToCollective,
    voteOnBelief,
    getCollectiveBeliefs,
    getConsensus,
    detectPatterns,
    getCollectiveStats,
    simulateShare,
    simulateVote,
    calculateLocalConsensus,
    ValidationVoteType,
    ConsensusType,
    type BeliefShare,
  } from '$services/collective-sensemaking';

  const dispatch = createEventDispatcher<{
    complete: void;
    skipdemo: void;
  }>();
  import { ThoughtType, EmpiricalLevel, NormativeLevel, MaterialityLevel, HarmonicLevel } from '@mycelix/lucid-client';
  import type { Thought } from '@mycelix/lucid-client';
  import EpistemicBadge from './EpistemicBadge.svelte';

  /**
   * CollectiveSensemakingDemo - Interactive walkthrough of collective features
   *
   * Demonstrates:
   * 1. Sharing beliefs to the collective
   * 2. Casting votes on beliefs
   * 3. Viewing consensus formation
   * 4. Discovering emergent patterns
   */

  // Demo state
  let currentStep = $state(0);
  let isRunning = $state(false);
  let demoResults = $state<Record<string, unknown>>({});

  // Sample beliefs for demo
  const demoBelief: Thought = {
    id: 'demo-belief-1',
    content: 'Decentralized systems offer more resilience than centralized ones',
    thought_type: ThoughtType.Claim,
    epistemic: {
      empirical: EmpiricalLevel.E2,
      normative: NormativeLevel.N2,
      materiality: MaterialityLevel.M3,
      harmonic: HarmonicLevel.H2,
    },
    confidence: 0.8,
    tags: ['technology', 'decentralization', 'systems'],
    domain: 'technology',
    related_thoughts: [],
    source_hashes: [],
    parent_thought: null,
    created_at: Date.now() * 1000,
    updated_at: Date.now() * 1000,
    version: 1,
  };

  const demoSteps = [
    {
      title: 'Share a Belief',
      description: 'Share your thought with the collective for validation',
      action: 'share',
    },
    {
      title: 'Cast Your Vote',
      description: 'Vote on the shared belief to contribute to consensus',
      action: 'vote',
    },
    {
      title: 'Simulate Peer Votes',
      description: 'See how other validators might respond',
      action: 'simulate_votes',
    },
    {
      title: 'View Consensus',
      description: 'Observe how consensus emerges from the votes',
      action: 'consensus',
    },
    {
      title: 'Discover Patterns',
      description: 'Detect emergent patterns across the collective',
      action: 'patterns',
    },
  ];

  async function runDemoStep(step: number) {
    isRunning = true;

    try {
      switch (demoSteps[step].action) {
        case 'share':
          // Simulate sharing a belief
          const shared = simulateShare(demoBelief);
          demoResults = {
            ...demoResults,
            sharedBelief: shared,
          };
          break;

        case 'vote':
          // Cast our vote (simulateVote stores internally)
          const sharedBelief = demoResults.sharedBelief as BeliefShare | undefined;
          if (sharedBelief) {
            simulateVote(sharedBelief.content_hash, ValidationVoteType.Corroborate);
            demoResults = {
              ...demoResults,
              voteCount: 1,
            };
          }
          break;

        case 'simulate_votes':
          // Simulate other validators voting
          const beliefForVotes = demoResults.sharedBelief as BeliefShare | undefined;
          if (beliefForVotes) {
            simulateVote(beliefForVotes.content_hash, ValidationVoteType.Corroborate);
            simulateVote(beliefForVotes.content_hash, ValidationVoteType.Plausible);
            simulateVote(beliefForVotes.content_hash, ValidationVoteType.Corroborate);
            simulateVote(beliefForVotes.content_hash, ValidationVoteType.Plausible);
            simulateVote(beliefForVotes.content_hash, ValidationVoteType.Abstain);
            demoResults = {
              ...demoResults,
              voteCount: 6,
            };
          }
          break;

        case 'consensus':
          // Calculate local consensus using content hash
          const beliefForConsensus = demoResults.sharedBelief as BeliefShare | undefined;
          if (beliefForConsensus) {
            const consensus = calculateLocalConsensus(beliefForConsensus.content_hash);
            demoResults = {
              ...demoResults,
              consensus,
            };
          }
          break;

        case 'patterns':
          // Show pattern detection results (simulated)
          const patterns = [
            {
              pattern_id: 'demo-pattern-1',
              representative_content: 'Technology and decentralization themes',
              member_count: 5,
              pattern_type: 'Convergence',
              coherence: 0.82,
              tags: ['technology', 'decentralization'],
            },
          ];
          demoResults = {
            ...demoResults,
            patterns,
          };
          break;
      }
    } catch (error) {
      console.error('Demo step failed:', error);
    }

    isRunning = false;
  }

  function nextStep() {
    if (currentStep < demoSteps.length - 1) {
      currentStep++;
      runDemoStep(currentStep);
    }
  }

  function prevStep() {
    if (currentStep > 0) {
      currentStep--;
    }
  }

  function resetDemo() {
    currentStep = 0;
    demoResults = {};
  }

  function startDemo() {
    resetDemo();
    runDemoStep(0);
  }

  function getConsensusLabel(type: string): string {
    switch (type) {
      case 'StrongConsensus': return 'Strong Consensus';
      case 'ModerateConsensus': return 'Moderate Consensus';
      case 'WeakConsensus': return 'Weak Consensus';
      case 'Contested': return 'Contested';
      case 'Insufficient': return 'Insufficient Votes';
      default: return type;
    }
  }

  function getConsensusColor(type: string): string {
    switch (type) {
      case 'StrongConsensus': return '#22c55e';
      case 'ModerateConsensus': return '#84cc16';
      case 'WeakConsensus': return '#eab308';
      case 'Contested': return '#ef4444';
      default: return '#6b7280';
    }
  }

  function getVoteLabel(type: string): string {
    switch (type) {
      case 'Corroborate': return 'Corroborate';
      case 'Plausible': return 'Plausible';
      case 'Abstain': return 'Abstain';
      case 'Implausible': return 'Implausible';
      case 'Contradict': return 'Contradict';
      default: return type;
    }
  }

  function getVoteColor(type: string): string {
    switch (type) {
      case 'Corroborate': return '#22c55e';
      case 'Plausible': return '#84cc16';
      case 'Abstain': return '#6b7280';
      case 'Implausible': return '#f97316';
      case 'Contradict': return '#ef4444';
      default: return '#6b7280';
    }
  }
</script>

<div class="demo-container">
  <header class="demo-header">
    <h2>Collective Sensemaking Demo</h2>
    <p>Experience the distributed belief validation system</p>
  </header>

  <!-- Progress Indicator -->
  <div class="progress-bar">
    {#each demoSteps as step, i}
      <div
        class="progress-step"
        class:active={i === currentStep}
        class:completed={i < currentStep}
      >
        <div class="step-number">{i + 1}</div>
        <div class="step-title">{step.title}</div>
      </div>
      {#if i < demoSteps.length - 1}
        <div class="step-connector" class:active={i < currentStep}></div>
      {/if}
    {/each}
  </div>

  <!-- Current Step Content -->
  <div class="step-content">
    <h3>{demoSteps[currentStep].title}</h3>
    <p class="step-description">{demoSteps[currentStep].description}</p>

    <!-- Step-specific content -->
    {#if currentStep === 0}
      <!-- Share a Belief -->
      <div class="belief-preview">
        <div class="belief-content">"{demoBelief.content}"</div>
        <div class="belief-meta">
          <span class="belief-type">{demoBelief.thought_type}</span>
          <span class="belief-confidence">Confidence: {Math.round(demoBelief.confidence * 100)}%</span>
        </div>
        <div class="belief-tags">
          {#each demoBelief.tags as tag}
            <span class="tag">{tag}</span>
          {/each}
        </div>
      </div>

      {#if demoResults.sharedBelief}
        <div class="result-box success">
          <span class="icon">&#10003;</span>
          <div>
            <strong>Belief Shared!</strong>
            <p>Content hash: <code>{(demoResults.sharedBelief as any).content_hash.slice(0, 16)}...</code></p>
          </div>
        </div>
      {/if}

    {:else if currentStep === 1}
      <!-- Cast Your Vote -->
      <div class="vote-options">
        <p>How would you validate this belief?</p>
        <div class="vote-buttons">
          <button
            class="vote-btn corroborate"
            class:selected={(demoResults.myVote as any)?.vote_type === 'Corroborate'}
            disabled={!!demoResults.myVote}
          >
            Corroborate
          </button>
          <button class="vote-btn plausible" disabled>Plausible</button>
          <button class="vote-btn abstain" disabled>Abstain</button>
          <button class="vote-btn implausible" disabled>Implausible</button>
          <button class="vote-btn contradict" disabled>Contradict</button>
        </div>
      </div>

      {#if demoResults.myVote}
        <div class="result-box success">
          <span class="icon">&#10003;</span>
          <div>
            <strong>Vote Cast!</strong>
            <p>You voted: <span style="color: {getVoteColor((demoResults.myVote as any).vote_type)}">{getVoteLabel((demoResults.myVote as any).vote_type)}</span></p>
          </div>
        </div>
      {/if}

    {:else if currentStep === 2}
      <!-- Simulate Peer Votes -->
      <div class="votes-list">
        <h4>Validation Votes ({(demoResults.votes as any[])?.length || 0})</h4>
        {#if demoResults.votes}
          <div class="votes-grid">
            {#each (demoResults.votes as any[]) as vote, i}
              <div class="vote-card">
                <span class="voter-icon">{i === 0 ? 'You' : `Peer ${i}`}</span>
                <span
                  class="vote-type"
                  style="color: {getVoteColor(vote.vote_type)}"
                >
                  {getVoteLabel(vote.vote_type)}
                </span>
                <span class="vote-weight">Weight: {vote.voter_weight.toFixed(2)}</span>
              </div>
            {/each}
          </div>
        {/if}
      </div>

    {:else if currentStep === 3}
      <!-- View Consensus -->
      {#if demoResults.consensus}
        {@const consensus = demoResults.consensus as any}
        <div class="consensus-display">
          <div
            class="consensus-badge"
            style="background: {getConsensusColor(consensus.consensusType)}"
          >
            {getConsensusLabel(consensus.consensusType)}
          </div>

          <div class="consensus-stats">
            <div class="stat">
              <span class="stat-value">{Math.round(consensus.agreementScore * 100)}%</span>
              <span class="stat-label">Agreement</span>
            </div>
            <div class="stat">
              <span class="stat-value">{consensus.voterCount}</span>
              <span class="stat-label">Validators</span>
            </div>
          </div>

          <div class="breakdown-chart">
            <h4>Vote Breakdown</h4>
            <div class="breakdown-bars">
              {#each Object.entries(consensus.breakdown) as [type, count]}
                <div class="breakdown-row">
                  <span class="breakdown-label">{getVoteLabel(type.charAt(0).toUpperCase() + type.slice(1))}</span>
                  <div class="breakdown-bar">
                    <div
                      class="bar-fill"
                      style="width: {((count as number) / consensus.totalWeight) * 100}%; background: {getVoteColor(type.charAt(0).toUpperCase() + type.slice(1))}"
                    ></div>
                  </div>
                  <span class="breakdown-count">{(count as number).toFixed(1)}</span>
                </div>
              {/each}
            </div>
          </div>
        </div>
      {/if}

    {:else if currentStep === 4}
      <!-- Discover Patterns -->
      {#if demoResults.patterns}
        <div class="patterns-list">
          <h4>Emergent Patterns Detected</h4>
          {#each (demoResults.patterns as any[]) as pattern}
            <div class="pattern-card">
              <div class="pattern-header">
                <span class="pattern-type">{pattern.pattern_type}</span>
                <span class="pattern-coherence">Coherence: {Math.round(pattern.coherence * 100)}%</span>
              </div>
              <p class="pattern-content">{pattern.representative_content}</p>
              <div class="pattern-meta">
                <span>{pattern.member_count} beliefs in cluster</span>
                <div class="pattern-tags">
                  {#each pattern.tags as tag}
                    <span class="tag">{tag}</span>
                  {/each}
                </div>
              </div>
            </div>
          {/each}
        </div>

        <div class="result-box success">
          <span class="icon">&#10003;</span>
          <div>
            <strong>Demo Complete!</strong>
            <p>You've experienced the full collective sensemaking flow.</p>
          </div>
        </div>
      {/if}
    {/if}
  </div>

  <!-- Navigation -->
  <div class="demo-nav">
    <button class="btn ghost skip-btn" onclick={() => dispatch('skipdemo')}>
      Skip Demo
    </button>

    <div class="nav-main">
      {#if currentStep === 0 && !demoResults.sharedBelief}
        <button class="btn primary" onclick={startDemo} disabled={isRunning}>
          {isRunning ? 'Running...' : 'Start Demo'}
        </button>
      {:else}
        <button
          class="btn secondary"
          onclick={prevStep}
          disabled={currentStep === 0 || isRunning}
        >
          Previous
        </button>

        {#if currentStep < demoSteps.length - 1}
          <button
            class="btn primary"
            onclick={nextStep}
            disabled={isRunning}
          >
            {isRunning ? 'Running...' : 'Next Step'}
          </button>
        {:else}
          <button class="btn secondary" onclick={resetDemo}>
            Restart
          </button>
          <button class="btn primary" onclick={() => dispatch('complete')}>
            Done
          </button>
        {/if}
      {/if}
    </div>
  </div>
</div>

<style>
  .demo-container {
    padding: 24px;
    max-width: 800px;
    margin: 0 auto;
  }

  .demo-header {
    text-align: center;
    margin-bottom: 32px;
  }

  .demo-header h2 {
    margin: 0 0 8px;
    color: #e2e8f0;
  }

  .demo-header p {
    margin: 0;
    color: #94a3b8;
  }

  /* Progress Bar */
  .progress-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 32px;
    flex-wrap: wrap;
    gap: 8px;
  }

  .progress-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .step-number {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #334155;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    color: #94a3b8;
    transition: all 0.2s;
  }

  .progress-step.active .step-number {
    background: #7c3aed;
    color: white;
  }

  .progress-step.completed .step-number {
    background: #22c55e;
    color: white;
  }

  .step-title {
    font-size: 11px;
    color: #64748b;
    max-width: 80px;
    text-align: center;
  }

  .progress-step.active .step-title {
    color: #e2e8f0;
  }

  .step-connector {
    width: 40px;
    height: 2px;
    background: #334155;
    margin-top: -20px;
  }

  .step-connector.active {
    background: #22c55e;
  }

  /* Step Content */
  .step-content {
    background: #1e293b;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    min-height: 300px;
  }

  .step-content h3 {
    margin: 0 0 8px;
    color: #e2e8f0;
  }

  .step-description {
    color: #94a3b8;
    margin: 0 0 24px;
  }

  /* Belief Preview */
  .belief-preview {
    background: #0f172a;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
  }

  .belief-content {
    font-size: 16px;
    color: #e2e8f0;
    font-style: italic;
    margin-bottom: 12px;
  }

  .belief-meta {
    display: flex;
    gap: 16px;
    margin-bottom: 8px;
  }

  .belief-type {
    background: #7c3aed;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
  }

  .belief-confidence {
    color: #94a3b8;
    font-size: 12px;
  }

  .belief-tags, .pattern-tags {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .tag {
    background: #334155;
    color: #94a3b8;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
  }

  /* Vote Options */
  .vote-options p {
    color: #94a3b8;
    margin-bottom: 12px;
  }

  .vote-buttons {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .vote-btn {
    padding: 8px 16px;
    border-radius: 6px;
    border: 2px solid transparent;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.2s;
  }

  .vote-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .vote-btn.corroborate { background: #22c55e20; color: #22c55e; border-color: #22c55e; }
  .vote-btn.plausible { background: #84cc1620; color: #84cc16; border-color: #84cc16; }
  .vote-btn.abstain { background: #6b728020; color: #6b7280; border-color: #6b7280; }
  .vote-btn.implausible { background: #f9731620; color: #f97316; border-color: #f97316; }
  .vote-btn.contradict { background: #ef444420; color: #ef4444; border-color: #ef4444; }

  .vote-btn.selected {
    background: #22c55e;
    color: white;
  }

  /* Votes List */
  .votes-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 12px;
    margin-top: 16px;
  }

  .vote-card {
    background: #0f172a;
    border-radius: 8px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .voter-icon {
    font-weight: 600;
    color: #e2e8f0;
  }

  .vote-type {
    font-weight: 500;
  }

  .vote-weight {
    font-size: 12px;
    color: #64748b;
  }

  /* Consensus Display */
  .consensus-display {
    text-align: center;
  }

  .consensus-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 20px;
    font-weight: 600;
    color: white;
    margin-bottom: 24px;
  }

  .consensus-stats {
    display: flex;
    justify-content: center;
    gap: 48px;
    margin-bottom: 24px;
  }

  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .stat-value {
    font-size: 32px;
    font-weight: 700;
    color: #e2e8f0;
  }

  .stat-label {
    font-size: 12px;
    color: #64748b;
  }

  /* Breakdown Chart */
  .breakdown-chart {
    background: #0f172a;
    border-radius: 8px;
    padding: 16px;
    text-align: left;
  }

  .breakdown-chart h4 {
    margin: 0 0 12px;
    color: #94a3b8;
    font-size: 14px;
  }

  .breakdown-bars {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .breakdown-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .breakdown-label {
    width: 100px;
    font-size: 12px;
    color: #94a3b8;
  }

  .breakdown-bar {
    flex: 1;
    height: 8px;
    background: #1e293b;
    border-radius: 4px;
    overflow: hidden;
  }

  .bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
  }

  .breakdown-count {
    width: 40px;
    text-align: right;
    font-size: 12px;
    color: #64748b;
  }

  /* Patterns */
  .patterns-list h4 {
    margin: 0 0 16px;
    color: #94a3b8;
  }

  .pattern-card {
    background: #0f172a;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
  }

  .pattern-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .pattern-type {
    background: #3b82f6;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
  }

  .pattern-coherence {
    color: #22c55e;
    font-size: 12px;
  }

  .pattern-content {
    color: #e2e8f0;
    margin: 0 0 12px;
  }

  .pattern-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #64748b;
    font-size: 12px;
  }

  /* Result Box */
  .result-box {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 16px;
    border-radius: 8px;
    margin-top: 16px;
  }

  .result-box.success {
    background: #22c55e20;
    border: 1px solid #22c55e40;
  }

  .result-box .icon {
    font-size: 20px;
    color: #22c55e;
  }

  .result-box strong {
    color: #e2e8f0;
  }

  .result-box p {
    margin: 4px 0 0;
    color: #94a3b8;
    font-size: 14px;
  }

  .result-box code {
    background: #0f172a;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: monospace;
  }

  /* Navigation */
  .demo-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid #334155;
  }

  .nav-main {
    display: flex;
    gap: 12px;
  }

  .skip-btn {
    font-size: 13px;
  }

  .btn {
    padding: 10px 24px;
    border-radius: 8px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    border: none;
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn.primary {
    background: #7c3aed;
    color: white;
  }

  .btn.primary:hover:not(:disabled) {
    background: #6d28d9;
  }

  .btn.secondary {
    background: #334155;
    color: #e2e8f0;
  }

  .btn.secondary:hover:not(:disabled) {
    background: #475569;
  }

  .btn.ghost {
    background: transparent;
    color: #94a3b8;
    padding: 8px 16px;
  }

  .btn.ghost:hover:not(:disabled) {
    color: #e2e8f0;
    background: rgba(255, 255, 255, 0.05);
  }
</style>
