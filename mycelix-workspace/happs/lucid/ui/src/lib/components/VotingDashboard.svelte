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
  import { createEventDispatcher, onMount, onDestroy } from 'svelte';
  import type { Thought, EpistemicClassification } from '@mycelix/lucid-client';
  import { EmpiricalLevel, NormativeLevel, MaterialityLevel, HarmonicLevel } from '@mycelix/lucid-client';
  import type { ActionHash } from '@holochain/client';
  import {
    type BeliefShare,
    type ValidationVote,
    type ConsensusRecord,
    ValidationVoteType,
    ConsensusType,
    voteOnBelief,
    getConsensus,
    getBeliefVotes,
    simulateVote,
    calculateLocalConsensus,
  } from '$services/collective-sensemaking';
  import {
    castAnonymousVote,
    isZkpReady,
    type AnonymousVoteResult,
  } from '$services/anonymous-voting';
  import { isConnected } from '$stores/holochain';
  import { requestReputationProof, getProofBatchStats } from '$services/proof-batching';
  import { getReputation, recordActivity } from '$services/reputation-decay';
  import EpistemicBadge from './EpistemicBadge.svelte';
  import ConsensusHistoryTimeline from './ConsensusHistoryTimeline.svelte';

  /**
   * VotingDashboard - Cast votes on beliefs, view consensus evolution
   *
   * Features:
   * - Belief preview with EpistemicBadge
   * - Vote buttons (Corroborate, Plausible, Abstain, Implausible, Contradict)
   * - Evidence addition option
   * - Consensus display with percentage and breakdown
   */

  // ============================================================================
  // PROPS
  // ============================================================================

  /** The belief to vote on (from collective) */
  export let belief: BeliefShare | null = null;

  /** Alternative: a personal thought to contextualize */
  export let thought: Thought | null = null;

  // ============================================================================
  // STATE
  // ============================================================================

  const dispatch = createEventDispatcher();

  let selectedVote: ValidationVoteType | null = null;
  let evidence: string = '';
  let showEvidenceInput = false;
  let isSubmitting = false;
  let submitError: string | null = null;

  // Consensus data
  let consensus: ConsensusRecord | null = null;
  let votes: ValidationVote[] = [];
  let consensusHistory: ConsensusRecord[] = [];
  let isLoadingConsensus = false;
  let showTimeline = false;

  // Eligibility proof state
  let eligibilityProof: string | null = null;
  let isGeneratingProof = false;
  const VOTING_THRESHOLD = 50; // Minimum reputation to vote

  // Anonymous voting state
  let useAnonymousVoting = true;
  let zkpAvailable = false;
  let anonymousVoteResult: AnonymousVoteResult | null = null;

  // Vote breakdown for visualization
  let voteBreakdown: { type: ValidationVoteType; count: number; percentage: number }[] = [];

  // Phi coverage — fraction of votes backed by verified consciousness data
  // Will be sourced from governance PhiWeightedTally when bridge is wired
  let phiCoverage = 0;
  let phiEnhancedCount = 0;
  let reputationOnlyCount = 0;

  // ============================================================================
  // COMPUTED
  // ============================================================================

  $: displayBelief = belief || (thought ? convertThoughtToBelief(thought) : null);
  $: epistemic = parseEpistemicCode(displayBelief?.epistemic_code);
  $: hasVoted = selectedVote !== null;

  // Load consensus when belief changes
  $: if (displayBelief) {
    loadConsensusData();
  }

  // Calculate vote breakdown
  $: {
    if (votes.length > 0) {
      const counts = new Map<ValidationVoteType, number>();
      for (const vote of votes) {
        counts.set(vote.vote_type, (counts.get(vote.vote_type) || 0) + 1);
      }

      voteBreakdown = [
        ValidationVoteType.Corroborate,
        ValidationVoteType.Plausible,
        ValidationVoteType.Abstain,
        ValidationVoteType.Implausible,
        ValidationVoteType.Contradict,
      ].map((type) => ({
        type,
        count: counts.get(type) || 0,
        percentage: ((counts.get(type) || 0) / votes.length) * 100,
      }));

      // Simulate phi coverage from voter weights:
      // Voters with weight >= 0.7 are treated as having attested Phi
      // TODO: Replace with real PhiWeightedTally.phiCoverage from governance bridge
      phiEnhancedCount = votes.filter(v => v.voter_weight >= 0.7).length;
      reputationOnlyCount = votes.length - phiEnhancedCount;
      phiCoverage = phiEnhancedCount / votes.length;
    } else {
      voteBreakdown = [];
      phiCoverage = 0;
      phiEnhancedCount = 0;
      reputationOnlyCount = 0;
    }
  }

  // ============================================================================
  // DATA LOADING
  // ============================================================================

  async function loadConsensusData() {
    if (!displayBelief) return;

    isLoadingConsensus = true;
    try {
      if ($isConnected && belief) {
        // Would need actual ActionHash - for now use simulated
        consensus = calculateLocalConsensus(displayBelief.content_hash);
      } else {
        consensus = calculateLocalConsensus(displayBelief.content_hash);
      }

      // Simulate some votes for demonstration
      if (!consensus && displayBelief) {
        // Generate simulated votes
        votes = generateSimulatedVotes();
        consensus = {
          belief_share_hash: new Uint8Array(39) as unknown as ActionHash,
          consensus_type: ConsensusType.ModerateConsensus,
          validator_count: votes.length,
          agreement_score: 0.72,
          summary: `${votes.length} validators, 72% agreement`,
          reached_at: Date.now() * 1000,
        };
      }
    } catch (e) {
      console.error('Failed to load consensus:', e);
    } finally {
      isLoadingConsensus = false;
    }
  }

  function generateSimulatedVotes(): ValidationVote[] {
    const voteTypes = [
      ValidationVoteType.Corroborate,
      ValidationVoteType.Corroborate,
      ValidationVoteType.Corroborate,
      ValidationVoteType.Plausible,
      ValidationVoteType.Plausible,
      ValidationVoteType.Abstain,
      ValidationVoteType.Implausible,
    ];

    return voteTypes.map((type, i) => ({
      belief_share_hash: new Uint8Array(39) as unknown as ActionHash,
      vote_type: type,
      evidence: null,
      voter_weight: 0.5 + Math.random() * 0.5,
      voted_at: Date.now() * 1000 - i * 60000000,
    }));
  }

  // ============================================================================
  // HELPERS
  // ============================================================================

  function convertThoughtToBelief(t: Thought): BeliefShare {
    const epistemicCode = `${t.epistemic.empirical}${t.epistemic.normative}${t.epistemic.materiality}${t.epistemic.harmonic}`;
    return {
      content_hash: t.id,
      content: t.content,
      belief_type: t.thought_type,
      epistemic_code: epistemicCode,
      confidence: t.confidence,
      tags: t.tags || [],
      shared_at: Date.now() * 1000,
      evidence_hashes: [],
      embedding: [],
      stance: null,
    };
  }

  function parseEpistemicCode(code: string | undefined): EpistemicClassification | null {
    if (!code) return null;
    // Parse E1N2M1H3 format
    const match = code.match(/E(\d)N(\d)M(\d)H(\d)/);
    if (!match) return null;
    const empLevels = [EmpiricalLevel.E0, EmpiricalLevel.E1, EmpiricalLevel.E2, EmpiricalLevel.E3, EmpiricalLevel.E4];
    const normLevels = [NormativeLevel.N0, NormativeLevel.N1, NormativeLevel.N2, NormativeLevel.N3];
    const matLevels = [MaterialityLevel.M0, MaterialityLevel.M1, MaterialityLevel.M2, MaterialityLevel.M3];
    const harmLevels = [HarmonicLevel.H0, HarmonicLevel.H1, HarmonicLevel.H2, HarmonicLevel.H3, HarmonicLevel.H4];
    return {
      empirical: empLevels[parseInt(match[1])] ?? EmpiricalLevel.E0,
      normative: normLevels[parseInt(match[2])] ?? NormativeLevel.N0,
      materiality: matLevels[parseInt(match[3])] ?? MaterialityLevel.M0,
      harmonic: harmLevels[parseInt(match[4])] ?? HarmonicLevel.H0,
    };
  }

  function getVoteLabel(type: ValidationVoteType): string {
    switch (type) {
      case ValidationVoteType.Corroborate: return 'Corroborate';
      case ValidationVoteType.Plausible: return 'Plausible';
      case ValidationVoteType.Abstain: return 'Abstain';
      case ValidationVoteType.Implausible: return 'Implausible';
      case ValidationVoteType.Contradict: return 'Contradict';
      default: return 'Unknown';
    }
  }

  function getVoteColor(type: ValidationVoteType): string {
    switch (type) {
      case ValidationVoteType.Corroborate: return '#22c55e';
      case ValidationVoteType.Plausible: return '#3b82f6';
      case ValidationVoteType.Abstain: return '#94a3b8';
      case ValidationVoteType.Implausible: return '#f59e0b';
      case ValidationVoteType.Contradict: return '#ef4444';
      default: return '#6b7280';
    }
  }

  function getConsensusLabel(type: ConsensusType): string {
    switch (type) {
      case ConsensusType.StrongConsensus: return 'Strong';
      case ConsensusType.ModerateConsensus: return 'Moderate';
      case ConsensusType.WeakConsensus: return 'Weak';
      case ConsensusType.Contested: return 'Contested';
      default: return 'Pending';
    }
  }

  function getConsensusColor(type: ConsensusType): string {
    switch (type) {
      case ConsensusType.StrongConsensus: return '#22c55e';
      case ConsensusType.ModerateConsensus: return '#3b82f6';
      case ConsensusType.WeakConsensus: return '#f59e0b';
      case ConsensusType.Contested: return '#ef4444';
      default: return '#6b7280';
    }
  }

  // ============================================================================
  // EVENT HANDLERS
  // ============================================================================

  function selectVote(type: ValidationVoteType) {
    selectedVote = type;
    submitError = null;
  }

  async function submitVote() {
    if (!selectedVote || !displayBelief) return;

    isSubmitting = true;
    submitError = null;
    anonymousVoteResult = null;

    try {
      const agentId = 'current-user'; // Would be actual agent ID
      const reputationState = getReputation(agentId);
      const currentReputation = reputationState?.currentReputation ?? 100;

      // Use anonymous voting with ZK proofs if enabled and available
      if (useAnonymousVoting && zkpAvailable) {
        isGeneratingProof = true;
        anonymousVoteResult = await castAnonymousVote({
          beliefHash: displayBelief.content_hash,
          voteType: selectedVote,
          evidence: evidence || undefined,
          reputationThreshold: VOTING_THRESHOLD,
        });
        isGeneratingProof = false;

        if (!anonymousVoteResult.success) {
          submitError = anonymousVoteResult.error || 'Anonymous vote failed';
          return;
        }

        eligibilityProof = anonymousVoteResult.proofInfo?.commitment || null;
      } else {
        // Generate eligibility proof using batch service (non-anonymous)
        isGeneratingProof = true;
        try {
          const proof = await requestReputationProof(
            currentReputation,
            VOTING_THRESHOLD,
            'high' // High priority for user-initiated vote
          );
          eligibilityProof = proof.proof_json;
        } catch (proofError) {
          console.warn('Could not generate eligibility proof, proceeding without:', proofError);
        }
        isGeneratingProof = false;

        if ($isConnected && belief) {
          // Would need actual ActionHash
          // await voteOnBelief(beliefHash, selectedVote, evidence || undefined, eligibilityProof);
        }

        // Simulate vote
        simulateVote(displayBelief.content_hash, selectedVote);
      }

      // Record activity for reputation bonus
      recordActivity(agentId, 'vote');

      dispatch('votesubmit', {
        beliefHash: displayBelief.content_hash,
        voteType: selectedVote,
        evidence: evidence || null,
        eligibilityProof,
        anonymous: useAnonymousVoting && zkpAvailable,
        attestationId: anonymousVoteResult?.attestationId,
      });

      // Reset and reload
      selectedVote = null;
      evidence = '';
      showEvidenceInput = false;
      eligibilityProof = null;
      await loadConsensusData();
    } catch (e) {
      console.error('Failed to submit vote:', e);
      submitError = e instanceof Error ? e.message : 'Failed to submit vote';
    } finally {
      isSubmitting = false;
      isGeneratingProof = false;
    }
  }

  function toggleEvidenceInput() {
    showEvidenceInput = !showEvidenceInput;
  }

  // ============================================================================
  // KEYBOARD SHORTCUTS
  // ============================================================================

  const voteKeyMap: Record<string, ValidationVoteType> = {
    '1': ValidationVoteType.Corroborate,
    '2': ValidationVoteType.Plausible,
    '3': ValidationVoteType.Abstain,
    '4': ValidationVoteType.Implausible,
    '5': ValidationVoteType.Contradict,
  };

  function handleKeydown(event: KeyboardEvent) {
    // Don't handle if typing in input
    const target = event.target as HTMLElement;
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;

    // Don't handle if no belief selected
    if (!displayBelief) return;

    // Vote shortcuts (1-5)
    if (event.key in voteKeyMap) {
      event.preventDefault();
      selectVote(voteKeyMap[event.key]);
    }

    // Enter to submit vote
    if (event.key === 'Enter' && selectedVote && !isSubmitting) {
      event.preventDefault();
      submitVote();
    }

    // E to toggle evidence input
    if (event.key === 'e' || event.key === 'E') {
      event.preventDefault();
      toggleEvidenceInput();
    }

    // Escape to clear selection
    if (event.key === 'Escape') {
      selectedVote = null;
      showEvidenceInput = false;
    }
  }

  onMount(async () => {
    window.addEventListener('keydown', handleKeydown);
    // Check if ZKP is available for anonymous voting
    zkpAvailable = await isZkpReady();
  });

  onDestroy(() => {
    window.removeEventListener('keydown', handleKeydown);
  });
</script>

<div class="voting-dashboard">
  {#if !displayBelief}
    <div class="empty-state">
      <span class="empty-icon">?</span>
      <p>Select a belief to vote on</p>
      <p class="empty-hint">Click on a node in the Consensus map or share a thought to the collective.</p>
    </div>
  {:else}
    <!-- Belief Preview -->
    <div class="belief-preview">
      <div class="preview-header">
        <span class="belief-type">{displayBelief.belief_type}</span>
        {#if epistemic}
          <EpistemicBadge
            {epistemic}
            compact={true}
            confidence={displayBelief.confidence}
          />
        {/if}
      </div>
      <p class="belief-content">{displayBelief.content}</p>
      {#if displayBelief.tags.length > 0}
        <div class="belief-tags">
          {#each displayBelief.tags as tag}
            <span class="tag">{tag}</span>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Vote Section -->
    <div class="vote-section">
      <div class="section-header">
        <h3 class="section-title">Your Vote</h3>
        <span class="shortcuts-hint" title="Keyboard shortcuts: 1-5 to vote, E for evidence, Enter to submit, Esc to cancel">
          Keyboard shortcuts enabled
        </span>
      </div>
      <div class="vote-buttons">
        <button
          class="vote-btn corroborate"
          class:selected={selectedVote === ValidationVoteType.Corroborate}
          on:click={() => selectVote(ValidationVoteType.Corroborate)}
          disabled={isSubmitting}
          title="Press 1 to select"
        >
          <span class="kbd">1</span>
          Corroborate
        </button>
        <button
          class="vote-btn plausible"
          class:selected={selectedVote === ValidationVoteType.Plausible}
          on:click={() => selectVote(ValidationVoteType.Plausible)}
          disabled={isSubmitting}
          title="Press 2 to select"
        >
          <span class="kbd">2</span>
          Plausible
        </button>
        <button
          class="vote-btn abstain"
          class:selected={selectedVote === ValidationVoteType.Abstain}
          on:click={() => selectVote(ValidationVoteType.Abstain)}
          disabled={isSubmitting}
          title="Press 3 to select"
        >
          <span class="kbd">3</span>
          Abstain
        </button>
        <button
          class="vote-btn implausible"
          class:selected={selectedVote === ValidationVoteType.Implausible}
          on:click={() => selectVote(ValidationVoteType.Implausible)}
          disabled={isSubmitting}
          title="Press 4 to select"
        >
          <span class="kbd">4</span>
          Implausible
        </button>
        <button
          class="vote-btn contradict"
          class:selected={selectedVote === ValidationVoteType.Contradict}
          on:click={() => selectVote(ValidationVoteType.Contradict)}
          disabled={isSubmitting}
          title="Press 5 to select"
        >
          <span class="kbd">5</span>
          Contradict
        </button>
      </div>

      <!-- Evidence input -->
      <div class="evidence-section">
        <button class="evidence-toggle" on:click={toggleEvidenceInput} title="Press E to toggle">
          <span class="kbd kbd-small">E</span>
          {showEvidenceInput ? 'Hide evidence' : 'Add evidence'}
        </button>
        {#if showEvidenceInput}
          <textarea
            class="evidence-input"
            bind:value={evidence}
            placeholder="Provide supporting evidence or reasoning for your vote..."
            rows="3"
          ></textarea>
        {/if}
      </div>

      <!-- Anonymous voting toggle -->
      <div class="anonymous-toggle">
        <label class="toggle-label">
          <input
            type="checkbox"
            bind:checked={useAnonymousVoting}
            disabled={!zkpAvailable || isSubmitting}
          />
          <span class="toggle-text">
            Anonymous voting
            {#if !zkpAvailable}
              <span class="unavailable">(ZKP unavailable)</span>
            {:else if isGeneratingProof}
              <span class="generating">Generating proof...</span>
            {/if}
          </span>
        </label>
        {#if useAnonymousVoting && zkpAvailable}
          <p class="toggle-description">
            Your vote will be verified by zero-knowledge proof without revealing your identity.
          </p>
        {/if}
      </div>

      <!-- Submit button -->
      {#if hasVoted}
        <button
          class="submit-btn"
          on:click={submitVote}
          disabled={isSubmitting}
          title="Press Enter to submit"
        >
          {#if isSubmitting}
            {isGeneratingProof ? 'Generating ZK proof...' : 'Submitting...'}
          {:else}
            Submit Vote {useAnonymousVoting && zkpAvailable ? '(Anonymous)' : ''}
          {/if}
          {#if !isSubmitting}<span class="kbd kbd-inline">Enter</span>{/if}
        </button>
        {#if submitError}
          <p class="error-message">{submitError}</p>
        {/if}
        {#if anonymousVoteResult?.success}
          <div class="success-message">
            <span class="success-icon">Vote verified</span>
            <span class="proof-info">ZK proof commitment: {anonymousVoteResult.proofInfo?.commitment?.slice(0, 12)}...</span>
          </div>
        {/if}
      {/if}
    </div>

    <!-- Consensus Section -->
    <div class="consensus-section">
      <div class="consensus-header">
        <h3 class="section-title">Consensus</h3>
        <button
          class="timeline-toggle"
          class:active={showTimeline}
          on:click={() => showTimeline = !showTimeline}
          title="Show vote history"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12,6 12,12 16,14"></polyline>
          </svg>
          History
        </button>
      </div>
      {#if isLoadingConsensus}
        <div class="loading-consensus">
          <span class="spinner"></span>
          Loading consensus data...
        </div>
      {:else if consensus}
        <div class="consensus-overview">
          <div class="consensus-score">
            <span class="score-value" style="color: {getConsensusColor(consensus.consensus_type)}">
              {Math.round(consensus.agreement_score * 100)}%
            </span>
            <span class="score-label">Agreement</span>
          </div>
          <div class="consensus-meta">
            <span class="consensus-type" style="background: {getConsensusColor(consensus.consensus_type)}20; color: {getConsensusColor(consensus.consensus_type)}">
              {getConsensusLabel(consensus.consensus_type)}
            </span>
            <span class="validator-count">{consensus.validator_count} validators</span>
          </div>
        </div>

        <!-- Phi Coverage indicator -->
        {#if votes.length > 0 && !showTimeline}
          <div class="phi-coverage">
            <div class="phi-coverage-header">
              <span class="phi-coverage-label">Consciousness Coverage</span>
              <span class="phi-coverage-value" class:high={phiCoverage >= 0.7} class:medium={phiCoverage >= 0.4 && phiCoverage < 0.7} class:low={phiCoverage < 0.4}>
                {Math.round(phiCoverage * 100)}%
              </span>
            </div>
            <div class="phi-coverage-bar">
              <div
                class="phi-coverage-fill"
                class:high={phiCoverage >= 0.7}
                class:medium={phiCoverage >= 0.4 && phiCoverage < 0.7}
                class:low={phiCoverage < 0.4}
                style="width: {phiCoverage * 100}%"
              ></div>
            </div>
            <div class="phi-coverage-detail">
              <span>{phiEnhancedCount} attested</span>
              <span>{reputationOnlyCount} reputation-only</span>
            </div>
          </div>
        {/if}

        <!-- Vote breakdown chart -->
        {#if voteBreakdown.length > 0 && !showTimeline}
          <div class="vote-breakdown">
            <h4 class="breakdown-title">Vote Breakdown</h4>
            <div class="breakdown-bars">
              {#each voteBreakdown as item}
                <div class="breakdown-row">
                  <span class="breakdown-label">{getVoteLabel(item.type)}</span>
                  <div class="breakdown-bar">
                    <div
                      class="breakdown-fill"
                      style="width: {item.percentage}%; background: {getVoteColor(item.type)}"
                    ></div>
                  </div>
                  <span class="breakdown-count">{item.count}</span>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        <!-- Consensus History Timeline -->
        {#if showTimeline}
          <div class="timeline-wrapper">
            <ConsensusHistoryTimeline
              belief={displayBelief}
              history={consensusHistory}
              {votes}
            />
          </div>
        {/if}
      {:else}
        <div class="no-consensus">
          <p>No consensus data yet.</p>
          <p class="no-consensus-hint">Be the first to vote on this belief!</p>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .voting-dashboard {
    display: flex;
    flex-direction: column;
    gap: 16px;
    height: 100%;
  }

  /* Empty state */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    color: var(--text-muted, #94a3b8);
    text-align: center;
  }

  .empty-icon {
    font-size: 48px;
    opacity: 0.3;
    margin-bottom: 8px;
  }

  .empty-state p {
    margin: 4px 0;
  }

  .empty-hint {
    font-size: 12px;
    opacity: 0.7;
  }

  /* Belief preview */
  .belief-preview {
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 16px;
    border: 1px solid var(--border, #334155);
  }

  .preview-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .belief-type {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--color-primary, #3b82f6);
    background: rgba(59, 130, 246, 0.15);
    padding: 2px 8px;
    border-radius: 4px;
  }

  .belief-content {
    margin: 0 0 12px;
    font-size: 14px;
    line-height: 1.5;
    color: var(--text-primary, #f8fafc);
  }

  .belief-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .tag {
    font-size: 11px;
    padding: 2px 8px;
    background: var(--bg-secondary, #1e293b);
    border-radius: 4px;
    color: var(--text-muted, #94a3b8);
  }

  /* Vote section */
  .vote-section {
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 16px;
    border: 1px solid var(--border, #334155);
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .section-title {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary, #f8fafc);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .shortcuts-hint {
    font-size: 10px;
    color: var(--text-muted, #94a3b8);
    opacity: 0.7;
    cursor: help;
  }

  .shortcuts-hint:hover {
    opacity: 1;
  }

  .vote-buttons {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 12px;
  }

  .vote-btn {
    padding: 10px 8px;
    border: 2px solid transparent;
    border-radius: 6px;
    background: var(--bg-secondary, #1e293b);
    color: var(--text-muted, #94a3b8);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }

  .vote-btn:hover:not(:disabled) {
    color: var(--text-primary, #f8fafc);
  }

  .vote-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .vote-btn.corroborate { border-color: #22c55e33; }
  .vote-btn.corroborate:hover, .vote-btn.corroborate.selected {
    background: #22c55e20;
    border-color: #22c55e;
    color: #22c55e;
  }

  .vote-btn.plausible { border-color: #3b82f633; }
  .vote-btn.plausible:hover, .vote-btn.plausible.selected {
    background: #3b82f620;
    border-color: #3b82f6;
    color: #3b82f6;
  }

  .vote-btn.abstain { border-color: #94a3b833; }
  .vote-btn.abstain:hover, .vote-btn.abstain.selected {
    background: #94a3b820;
    border-color: #94a3b8;
    color: #94a3b8;
  }

  .vote-btn.implausible { border-color: #f59e0b33; }
  .vote-btn.implausible:hover, .vote-btn.implausible.selected {
    background: #f59e0b20;
    border-color: #f59e0b;
    color: #f59e0b;
  }

  .vote-btn.contradict { border-color: #ef444433; }
  .vote-btn.contradict:hover, .vote-btn.contradict.selected {
    background: #ef444420;
    border-color: #ef4444;
    color: #ef4444;
  }

  /* Keyboard hint badges */
  .kbd {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 18px;
    height: 18px;
    padding: 0 5px;
    margin-right: 6px;
    font-size: 10px;
    font-weight: 600;
    font-family: ui-monospace, monospace;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 3px;
    color: var(--text-muted, #94a3b8);
    vertical-align: middle;
  }

  .vote-btn:hover .kbd,
  .vote-btn.selected .kbd {
    background: rgba(0, 0, 0, 0.4);
    border-color: rgba(255, 255, 255, 0.25);
    color: inherit;
  }

  .kbd-small {
    min-width: 16px;
    height: 16px;
    font-size: 9px;
    margin-right: 4px;
  }

  .kbd-inline {
    margin-left: 8px;
    margin-right: 0;
  }

  /* Evidence section */
  .evidence-section {
    margin-bottom: 12px;
  }

  .evidence-toggle {
    display: inline-flex;
    align-items: center;
    background: none;
    border: none;
    color: var(--color-primary, #3b82f6);
    font-size: 12px;
    cursor: pointer;
    padding: 4px 0;
  }

  .evidence-toggle:hover {
    text-decoration: underline;
  }

  .evidence-toggle .kbd {
    text-decoration: none;
  }

  .evidence-input {
    width: 100%;
    margin-top: 8px;
    padding: 10px;
    background: var(--bg-secondary, #1e293b);
    border: 1px solid var(--border, #334155);
    border-radius: 6px;
    color: var(--text-primary, #f8fafc);
    font-size: 13px;
    resize: vertical;
  }

  .evidence-input:focus {
    outline: none;
    border-color: var(--color-primary, #3b82f6);
  }

  /* Submit button */
  .submit-btn {
    width: 100%;
    padding: 12px;
    background: var(--color-primary, #3b82f6);
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
  }

  .submit-btn:hover:not(:disabled) {
    background: #2563eb;
  }

  .submit-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .error-message {
    margin-top: 8px;
    font-size: 12px;
    color: var(--color-error, #ef4444);
    text-align: center;
  }

  /* Anonymous voting toggle */
  .anonymous-toggle {
    margin-bottom: 12px;
    padding: 10px;
    background: var(--bg-secondary, #1e293b);
    border-radius: 6px;
    border: 1px solid var(--border, #334155);
  }

  .toggle-label {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    font-size: 12px;
    color: var(--text-primary, #f8fafc);
  }

  .toggle-label input {
    width: 16px;
    height: 16px;
    accent-color: var(--color-primary, #3b82f6);
  }

  .toggle-label input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .unavailable {
    color: var(--text-muted, #94a3b8);
    font-size: 10px;
  }

  .generating {
    color: var(--color-primary, #3b82f6);
    font-size: 10px;
  }

  .toggle-description {
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
    margin: 8px 0 0 24px;
    line-height: 1.4;
  }

  .success-message {
    margin-top: 8px;
    padding: 10px;
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .success-icon {
    color: #22c55e;
    font-size: 12px;
    font-weight: 600;
  }

  .proof-info {
    font-size: 10px;
    color: var(--text-muted, #94a3b8);
    font-family: ui-monospace, monospace;
  }

  /* Consensus section */
  .consensus-section {
    background: var(--bg-tertiary, #0f172a);
    border-radius: 8px;
    padding: 16px;
    border: 1px solid var(--border, #334155);
  }

  .consensus-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .timeline-toggle {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    font-size: 11px;
    background: var(--bg-secondary, #1e293b);
    border: 1px solid var(--border, #334155);
    border-radius: 4px;
    color: var(--text-muted, #94a3b8);
    cursor: pointer;
    transition: all 0.2s;
  }

  .timeline-toggle:hover {
    color: var(--text-primary, #f8fafc);
    border-color: var(--border-hover, #475569);
  }

  .timeline-toggle.active {
    background: var(--color-primary, #3b82f6);
    border-color: var(--color-primary, #3b82f6);
    color: white;
  }

  .timeline-toggle svg {
    width: 14px;
    height: 14px;
  }

  .timeline-wrapper {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border, #334155);
    max-height: 300px;
    overflow-y: auto;
  }

  .loading-consensus {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-muted, #94a3b8);
    font-size: 13px;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border, #334155);
    border-top-color: var(--color-primary, #3b82f6);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .consensus-overview {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .consensus-score {
    display: flex;
    flex-direction: column;
  }

  .score-value {
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
  }

  .score-label {
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
    text-transform: uppercase;
  }

  .consensus-meta {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
  }

  .consensus-type {
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
  }

  .validator-count {
    font-size: 12px;
    color: var(--text-muted, #94a3b8);
  }

  /* Phi coverage */
  .phi-coverage {
    padding: 10px 0;
    border-top: 1px solid var(--border, #334155);
    margin-bottom: 4px;
  }

  .phi-coverage-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }

  .phi-coverage-label {
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
    font-weight: 500;
  }

  .phi-coverage-value {
    font-size: 13px;
    font-weight: 700;
  }

  .phi-coverage-value.high { color: #22c55e; }
  .phi-coverage-value.medium { color: #f59e0b; }
  .phi-coverage-value.low { color: #94a3b8; }

  .phi-coverage-bar {
    height: 4px;
    background: var(--bg-secondary, #1e293b);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 6px;
  }

  .phi-coverage-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
    background: #94a3b8;
  }

  .phi-coverage-fill.high { background: #22c55e; }
  .phi-coverage-fill.medium { background: #f59e0b; }

  .phi-coverage-detail {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-muted, #94a3b8);
    opacity: 0.8;
  }

  /* Vote breakdown */
  .vote-breakdown {
    padding-top: 12px;
    border-top: 1px solid var(--border, #334155);
  }

  .breakdown-title {
    margin: 0 0 12px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted, #94a3b8);
  }

  .breakdown-bars {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .breakdown-row {
    display: grid;
    grid-template-columns: 80px 1fr 30px;
    align-items: center;
    gap: 8px;
  }

  .breakdown-label {
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
  }

  .breakdown-bar {
    height: 6px;
    background: var(--bg-secondary, #1e293b);
    border-radius: 3px;
    overflow: hidden;
  }

  .breakdown-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .breakdown-count {
    font-size: 11px;
    color: var(--text-muted, #94a3b8);
    text-align: right;
  }

  /* No consensus */
  .no-consensus {
    text-align: center;
    color: var(--text-muted, #94a3b8);
  }

  .no-consensus p {
    margin: 4px 0;
  }

  .no-consensus-hint {
    font-size: 12px;
    opacity: 0.7;
  }
</style>
