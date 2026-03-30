<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import type { Thought } from '@mycelix/lucid-client';
  import { thoughts } from '../stores/thoughts';
  import {
    reviewSession,
    reviewStats,
    dueForReview,
    startReviewSession,
    recordReview,
    endReviewSession,
    getDailyPrompt,
    loadReviewItems,
    addToReview,
    type ReviewRating,
  } from '../stores/review';
  import MarkdownRenderer from './MarkdownRenderer.svelte';
  import EpistemicBadge from './EpistemicBadge.svelte';

  let dailyPrompt = '';
  let showAddToReview = false;

  onMount(() => {
    loadReviewItems();
    dailyPrompt = getDailyPrompt();
  });

  function getCurrentThought(): Thought | undefined {
    if (!$reviewSession) return undefined;
    const item = $reviewSession.items[$reviewSession.currentIndex];
    return $thoughts.find((t) => t.id === item?.thoughtId);
  }

  $: currentThought = $reviewSession ? getCurrentThought() : undefined;
  $: progress = $reviewSession
    ? ($reviewSession.currentIndex / $reviewSession.items.length) * 100
    : 0;

  function handleRating(rating: ReviewRating) {
    recordReview(rating);
  }

  function handleAddAllToReview() {
    $thoughts.forEach((t) => addToReview(t));
    showAddToReview = false;
  }
</script>

<div class="review-panel">
  {#if $reviewSession}
    <!-- Active Review Session -->
    <div class="session">
      <header>
        <h3>Review Session</h3>
        <span class="progress-text">
          {$reviewSession.currentIndex + 1} / {$reviewSession.items.length}
        </span>
        <button class="end-btn" on:click={endReviewSession}>End Session</button>
      </header>

      <div class="progress-bar">
        <div class="progress-fill" style="width: {progress}%"></div>
      </div>

      {#if currentThought}
        <div class="review-card">
          <div class="thought-type">{currentThought.thought_type}</div>

          <div class="content">
            <MarkdownRenderer content={currentThought.content} />
          </div>

          <div class="meta">
            <EpistemicBadge epistemic={currentThought.epistemic} compact />
            {#if currentThought.tags?.length}
              <div class="tags">
                {#each currentThought.tags as tag}
                  <span class="tag">#{tag}</span>
                {/each}
              </div>
            {/if}
          </div>

          <div class="question">
            Do you still believe this? How confident are you?
          </div>

          <div class="rating-buttons">
            <button class="rating forgot" on:click={() => handleRating('forgot')}>
              <span class="emoji">😕</span>
              <span class="label">Forgot</span>
              <span class="hint">1 day</span>
            </button>
            <button class="rating hard" on:click={() => handleRating('hard')}>
              <span class="emoji">🤔</span>
              <span class="label">Hard</span>
              <span class="hint">~3 days</span>
            </button>
            <button class="rating good" on:click={() => handleRating('good')}>
              <span class="emoji">😊</span>
              <span class="label">Good</span>
              <span class="hint">~1 week</span>
            </button>
            <button class="rating easy" on:click={() => handleRating('easy')}>
              <span class="emoji">😄</span>
              <span class="label">Easy</span>
              <span class="hint">~2 weeks</span>
            </button>
          </div>
        </div>
      {:else}
        <div class="complete">
          <span class="emoji">🎉</span>
          <h3>Session Complete!</h3>
          <p>You reviewed {$reviewSession.completed.length} thoughts.</p>
          <button on:click={endReviewSession}>Done</button>
        </div>
      {/if}
    </div>
  {:else}
    <!-- Review Dashboard -->
    <div class="dashboard">
      <div class="daily-prompt">
        <h4>Daily Reflection</h4>
        <p>{dailyPrompt}</p>
      </div>

      <div class="stats">
        <div class="stat">
          <span class="value">{$reviewStats.dueToday}</span>
          <span class="label">Due Today</span>
        </div>
        <div class="stat">
          <span class="value">{$reviewStats.total}</span>
          <span class="label">Total Tracked</span>
        </div>
        <div class="stat">
          <span class="value">{Math.round($reviewStats.avgConfidence * 100)}%</span>
          <span class="label">Avg Confidence</span>
        </div>
        <div class="stat">
          <span class="value">{Math.round($reviewStats.avgInterval)}d</span>
          <span class="label">Avg Interval</span>
        </div>
      </div>

      <div class="actions">
        {#if $dueForReview.length > 0}
          <button class="primary" on:click={() => startReviewSession()}>
            Start Review ({$dueForReview.length} due)
          </button>
        {:else}
          <button class="primary" disabled>No Reviews Due</button>
        {/if}

        {#if $reviewStats.total === 0}
          <button class="secondary" on:click={() => (showAddToReview = true)}>
            Add Thoughts to Review
          </button>
        {/if}
      </div>

      {#if showAddToReview}
        <div class="add-prompt">
          <p>Add all your thoughts to the spaced repetition system?</p>
          <div class="buttons">
            <button on:click={handleAddAllToReview}>Yes, Add All</button>
            <button class="cancel" on:click={() => (showAddToReview = false)}>Cancel</button>
          </div>
        </div>
      {/if}

      {#if $dueForReview.length > 0}
        <div class="due-list">
          <h4>Due for Review</h4>
          <ul>
            {#each $dueForReview.slice(0, 5) as thought}
              <li>
                <span class="type">{thought.thought_type}</span>
                <span class="text">{thought.content.slice(0, 50)}...</span>
              </li>
            {/each}
            {#if $dueForReview.length > 5}
              <li class="more">+{$dueForReview.length - 5} more</li>
            {/if}
          </ul>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .review-panel {
    background: #1a1a2e;
    border: 1px solid #2a2a4e;
    border-radius: 12px;
    padding: 20px;
  }

  /* Session */
  .session header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }

  .session h3 {
    margin: 0;
    font-size: 1rem;
    color: #fff;
  }

  .progress-text {
    font-size: 0.85rem;
    color: #888;
  }

  .end-btn {
    margin-left: auto;
    padding: 4px 12px;
    background: #252540;
    border: 1px solid #3a3a5e;
    border-radius: 4px;
    color: #888;
    font-size: 0.8rem;
    cursor: pointer;
  }

  .progress-bar {
    height: 4px;
    background: #252540;
    border-radius: 2px;
    margin-bottom: 20px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #7c3aed, #3b82f6);
    transition: width 0.3s ease;
  }

  .review-card {
    background: #1e1e2e;
    border: 1px solid #2a2a4e;
    border-radius: 8px;
    padding: 20px;
  }

  .thought-type {
    display: inline-block;
    padding: 2px 8px;
    background: #7c3aed;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    color: white;
    margin-bottom: 12px;
  }

  .content {
    margin-bottom: 16px;
  }

  .meta {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }

  .tags {
    display: flex;
    gap: 6px;
  }

  .tag {
    padding: 2px 8px;
    background: #2a2a4e;
    border-radius: 4px;
    font-size: 0.75rem;
    color: #a5a5c5;
  }

  .question {
    text-align: center;
    color: #888;
    font-size: 0.9rem;
    margin-bottom: 16px;
  }

  .rating-buttons {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }

  .rating {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 12px;
    background: #252540;
    border: 2px solid #3a3a5e;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .rating:hover {
    transform: translateY(-2px);
  }

  .rating.forgot:hover { border-color: #ef4444; background: rgba(239, 68, 68, 0.1); }
  .rating.hard:hover { border-color: #f59e0b; background: rgba(245, 158, 11, 0.1); }
  .rating.good:hover { border-color: #10b981; background: rgba(16, 185, 129, 0.1); }
  .rating.easy:hover { border-color: #3b82f6; background: rgba(59, 130, 246, 0.1); }

  .rating .emoji {
    font-size: 1.5rem;
  }

  .rating .label {
    font-size: 0.85rem;
    color: #e5e5e5;
  }

  .rating .hint {
    font-size: 0.7rem;
    color: #666;
  }

  .complete {
    text-align: center;
    padding: 40px;
  }

  .complete .emoji {
    font-size: 3rem;
    display: block;
    margin-bottom: 16px;
  }

  .complete h3 {
    margin: 0 0 8px;
    color: #fff;
  }

  .complete p {
    color: #888;
    margin-bottom: 20px;
  }

  .complete button {
    padding: 10px 24px;
    background: #7c3aed;
    border: none;
    border-radius: 6px;
    color: white;
    cursor: pointer;
  }

  /* Dashboard */
  .daily-prompt {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(59, 130, 246, 0.1));
    border: 1px solid #3a3a5e;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 20px;
  }

  .daily-prompt h4 {
    margin: 0 0 8px;
    font-size: 0.8rem;
    color: #7c3aed;
    text-transform: uppercase;
  }

  .daily-prompt p {
    margin: 0;
    color: #e5e5e5;
    font-style: italic;
  }

  .stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
  }

  .stat {
    text-align: center;
    padding: 12px;
    background: #252540;
    border-radius: 8px;
  }

  .stat .value {
    display: block;
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff;
  }

  .stat .label {
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
  }

  .actions {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
  }

  .actions button {
    flex: 1;
    padding: 12px;
    border-radius: 8px;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .actions .primary {
    background: #7c3aed;
    border: none;
    color: white;
  }

  .actions .primary:hover:not(:disabled) {
    background: #6d28d9;
  }

  .actions .primary:disabled {
    background: #4a4a6e;
    cursor: not-allowed;
  }

  .actions .secondary {
    background: none;
    border: 1px solid #3a3a5e;
    color: #888;
  }

  .actions .secondary:hover {
    border-color: #5a5a8e;
    color: #aaa;
  }

  .add-prompt {
    background: #252540;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    margin-bottom: 20px;
  }

  .add-prompt p {
    margin: 0 0 12px;
    color: #e5e5e5;
  }

  .add-prompt .buttons {
    display: flex;
    gap: 12px;
    justify-content: center;
  }

  .add-prompt button {
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
  }

  .add-prompt button:first-child {
    background: #7c3aed;
    border: none;
    color: white;
  }

  .add-prompt .cancel {
    background: none;
    border: 1px solid #3a3a5e;
    color: #888;
  }

  .due-list h4 {
    margin: 0 0 12px;
    font-size: 0.85rem;
    color: #888;
  }

  .due-list ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }

  .due-list li {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 0;
    border-bottom: 1px solid #2a2a4e;
  }

  .due-list li:last-child {
    border-bottom: none;
  }

  .due-list .type {
    padding: 2px 6px;
    background: #252540;
    border-radius: 3px;
    font-size: 0.7rem;
    color: #888;
  }

  .due-list .text {
    font-size: 0.85rem;
    color: #a5a5c5;
  }

  .due-list .more {
    color: #666;
    font-size: 0.8rem;
    justify-content: center;
  }

  @media (max-width: 600px) {
    .stats {
      grid-template-columns: repeat(2, 1fr);
    }

    .rating-buttons {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>
