<script lang="ts">
  import type { Thought } from '@mycelix/lucid-client';
  import EpistemicBadge from './EpistemicBadge.svelte';
  import MarkdownRenderer from './MarkdownRenderer.svelte';
  import { selectedThought, deleteThought, coherenceStatus, type ThoughtCoherenceStatus } from '../stores/thoughts';
  import { createEventDispatcher } from 'svelte';

  export let thought: Thought;
  export let showCoherence = true;

  const dispatch = createEventDispatcher();

  $: isSelected = $selectedThought?.id === thought.id;
  $: confidencePercent = Math.round(thought.confidence * 100);
  $: createdDate = new Date(thought.created_at / 1000).toLocaleDateString();

  // Coherence status
  $: coherence = $coherenceStatus.get(thought.id) as ThoughtCoherenceStatus | undefined;
  $: hasContradictions = coherence && coherence.contradictions.length > 0;
  $: coherenceColor = getCoherenceColor(coherence?.coherenceScore);

  function getCoherenceColor(score: number | undefined): string {
    if (score === undefined) return '#6b7280';
    if (score >= 0.8) return '#10b981';
    if (score >= 0.6) return '#3b82f6';
    if (score >= 0.4) return '#f59e0b';
    return '#ef4444';
  }

  function handleSelect() {
    selectedThought.set(thought);
    dispatch('select', thought);
  }

  async function handleDelete(e: Event) {
    e.stopPropagation();
    if (confirm('Delete this thought?')) {
      await deleteThought(thought.id);
    }
  }

  // Thought type styling
  const typeColors: Record<string, string> = {
    Claim: '#3b82f6',
    Question: '#8b5cf6',
    Observation: '#10b981',
    Belief: '#f59e0b',
    Hypothesis: '#ec4899',
    Definition: '#6366f1',
    Argument: '#ef4444',
    Evidence: '#14b8a6',
    Intuition: '#a855f7',
    Memory: '#f97316',
    Goal: '#22c55e',
    Plan: '#0ea5e9',
    Reflection: '#64748b',
    Quote: '#84cc16',
    Note: '#78716c',
  };

  $: typeColor = typeColors[thought.thought_type] || '#6b7280';
</script>

<div
  class="thought-card"
  class:selected={isSelected}
  on:click={handleSelect}
  on:keydown={(e) => e.key === 'Enter' && handleSelect()}
  tabindex="0"
  role="button"
  aria-label="Thought: {thought.thought_type} - {thought.content.slice(0, 50)}{thought.content.length > 50 ? '...' : ''}"
  aria-pressed={isSelected}
>
  <header>
    <span class="type-badge" style="background: {typeColor}">
      {thought.thought_type}
    </span>
    <EpistemicBadge epistemic={thought.epistemic} compact />

    {#if showCoherence && coherence}
      <span
        class="coherence-indicator"
        class:warning={hasContradictions}
        style="--color: {coherenceColor}"
        title={hasContradictions
          ? `Coherence: ${Math.round(coherence.coherenceScore * 100)}% - ${coherence.contradictions.length} potential contradiction(s)`
          : `Coherence: ${Math.round(coherence.coherenceScore * 100)}% | Phi: ${coherence.phiScore.toFixed(2)}`}
        role="status"
        aria-label={hasContradictions
          ? `Coherence score ${Math.round(coherence.coherenceScore * 100)} percent with ${coherence.contradictions.length} potential contradictions`
          : `Coherence score ${Math.round(coherence.coherenceScore * 100)} percent, Phi score ${coherence.phiScore.toFixed(2)}`}
      >
        {#if hasContradictions}
          <span class="warning-icon" aria-hidden="true">!</span>
        {/if}
        <span class="phi-value" aria-hidden="true">{coherence.phiScore.toFixed(1)}</span>
      </span>
    {/if}

    <span class="confidence" title="Confidence: {confidencePercent}%">
      {confidencePercent}%
    </span>
  </header>

  <div class="content">
    <MarkdownRenderer content={thought.content} />
  </div>

  {#if thought.tags.length > 0}
    <div class="tags">
      {#each thought.tags as tag}
        <span class="tag">#{tag}</span>
      {/each}
    </div>
  {/if}

  {#if showCoherence && hasContradictions && isSelected && coherence}
    <div class="contradictions-section">
      <span class="contradictions-label">Potential Contradictions:</span>
      {#each coherence.contradictions.slice(0, 3) as contradiction}
        <div class="contradiction-item">
          <span class="severity-dot" style="background: {contradiction.severity > 0.7 ? '#ef4444' : contradiction.severity > 0.4 ? '#f59e0b' : '#6b7280'}"></span>
          <span class="contradiction-desc">{contradiction.description}</span>
        </div>
      {/each}
      {#if coherence.suggestions.length > 0}
        <div class="suggestion">
          {coherence.suggestions[0]}
        </div>
      {/if}
    </div>
  {/if}

  <footer>
    <span class="date">{createdDate}</span>
    {#if thought.domain}
      <span class="domain">{thought.domain}</span>
    {/if}
    <button
      class="delete-btn"
      on:click={handleDelete}
      title="Delete thought"
      aria-label="Delete this thought"
    >
      <span aria-hidden="true">×</span>
    </button>
  </footer>
</div>

<style>
  .thought-card {
    background: #1e1e2e;
    border: 1px solid #2a2a4e;
    border-radius: 12px;
    padding: 16px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .thought-card:hover {
    border-color: #4a4a7e;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }

  .thought-card.selected {
    border-color: #7c3aed;
    box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.3);
  }

  header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
  }

  .type-badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    color: white;
  }

  .coherence-indicator {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 6px;
    background: color-mix(in srgb, var(--color) 20%, transparent);
    border: 1px solid color-mix(in srgb, var(--color) 40%, transparent);
    border-radius: 4px;
    font-size: 0.7rem;
    font-family: monospace;
    color: var(--color);
    transition: all 0.2s;
  }

  .coherence-indicator.warning {
    animation: pulse-warning 2s ease-in-out infinite;
  }

  @keyframes pulse-warning {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  .warning-icon {
    font-weight: bold;
    font-size: 0.8rem;
  }

  .phi-value {
    font-weight: 600;
  }

  .confidence {
    margin-left: auto;
    font-size: 0.8rem;
    color: #888;
    font-family: monospace;
  }

  .content {
    margin: 0 0 12px;
    color: #e5e5e5;
    line-height: 1.5;
    font-size: 0.95rem;
    overflow: hidden;
  }

  .content :global(p:first-child) {
    margin-top: 0;
  }

  .content :global(p:last-child) {
    margin-bottom: 0;
  }

  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 12px;
  }

  .tag {
    padding: 2px 8px;
    background: #2a2a4e;
    border-radius: 4px;
    font-size: 0.75rem;
    color: #a5a5c5;
  }

  footer {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.75rem;
    color: #666;
  }

  .date {
    color: #555;
  }

  .domain {
    padding: 2px 6px;
    background: #252540;
    border-radius: 3px;
    color: #888;
  }

  .delete-btn {
    margin-left: auto;
    background: none;
    border: none;
    color: #666;
    font-size: 1.2rem;
    cursor: pointer;
    padding: 0 4px;
    opacity: 0;
    transition: opacity 0.2s, color 0.2s;
  }

  .thought-card:hover .delete-btn {
    opacity: 1;
  }

  .delete-btn:hover {
    color: #ef4444;
  }

  /* Contradictions Section */
  .contradictions-section {
    margin: 12px 0;
    padding: 10px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 8px;
  }

  .contradictions-label {
    display: block;
    font-size: 0.7rem;
    color: #ef4444;
    text-transform: uppercase;
    margin-bottom: 8px;
    font-weight: 600;
  }

  .contradiction-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  }

  .contradiction-item:last-of-type {
    border-bottom: none;
  }

  .severity-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 4px;
  }

  .contradiction-desc {
    font-size: 0.8rem;
    color: #a5a5c5;
    line-height: 1.4;
  }

  .suggestion {
    margin-top: 8px;
    padding: 6px 10px;
    background: rgba(124, 58, 237, 0.1);
    border-left: 2px solid #7c3aed;
    border-radius: 0 4px 4px 0;
    font-size: 0.75rem;
    color: #a5a5c5;
    font-style: italic;
  }
</style>
